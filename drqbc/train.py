# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import json

from pyvirtualdisplay import Display  

from pathlib import Path
import hydra, omegaconf
import numpy as np
import torch
from dm_env import specs
import wandb

# import dmc
import drqbc.utils as utils
from drqbc.utils import ExtendedTimeStep
from dm_env import StepType
from drqbc.logger import Logger
from drqbc.numpy_replay_buffer import EfficientReplayBuffer
# from video import TrainVideoRecorder, VideoRecorder
from cdmc.video import VideoRecorder
from drqbc.utils import load_offline_dataset_into_buffer

from cdmc.env.wrappers import make_env

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_space,
                                self.train_env.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb, offline=self.cfg.offline,
                             distracting_eval=self.cfg.eval_on_distracting, multitask_eval=self.cfg.eval_on_multitask)
        # create envs
        with open(f'{self.cfg.train_context_name}.json', 'r') as file:
            train_contexts = json.load(file)
        self.train_env = make_env(
            domain_name=self.cfg.domain_name,
            task_name=self.cfg.task_name,
            seed=self.cfg.seed,
            episode_length=1000,
            action_repeat=self.cfg.action_repeat,
            frame_stack=self.cfg.frame_stack,
            image_size=84,
            states=train_contexts['states'],
            video_paths=train_contexts['video_paths'],
            colors=[dict([(k, np.array(v)) for k,v in color_dict.items()]) for color_dict in train_contexts['colors']],
            from_pixels=True,
        )
        with open(f'{self.cfg.test_context_name}.json', 'r') as file:
            test_contexts = json.load(file)
        self.eval_env = make_env(
            domain_name=self.cfg.domain_name,
            task_name=self.cfg.task_name,
            seed=self.cfg.seed,
            episode_length=1000,
            action_repeat=self.cfg.action_repeat,
            frame_stack=self.cfg.frame_stack,
            image_size=84,
            states=test_contexts['states'],
            video_paths=test_contexts['video_paths'],
            colors=[dict([(k, np.array(v)) for k,v in color_dict.items()]) for color_dict in test_contexts['colors']],
            from_pixels=True,
        )


        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_buffer = EfficientReplayBuffer(self.cfg.replay_buffer_size,
                                                   self.cfg.batch_size,
                                                   self.cfg.nstep,
                                                   self.cfg.discount,
                                                   self.cfg.frame_stack,
                                                   data_specs)

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    def eval(self, train=True):
        if train:
            _env = self.train_env
            ty = 'train'
        else:
            _env = self.eval_env
            ty = 'eval'
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            obs = np.array(_env.reset())
            action_spec = _env.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
            time_step = ExtendedTimeStep(StepType.FIRST, 0.0, 1.0, obs, action)
            enable_videorecorder = (episode == 0) and (self.global_step % (self.cfg.eval_save_vid_every_step // self.cfg.action_repeat) == 0)
            self.video_recorder.init(enabled=enable_videorecorder)
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                    
                obs, reward, done, _ = _env.step(action)
                obs = np.array(obs)
                step_type = StepType.LAST if done else StepType.MID
                time_step = ExtendedTimeStep(step_type, reward, 1.0, obs, action)
                self.video_recorder.record(_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{ty}_{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty=ty) as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train_offline(self, offline_dir):
        # Open dataset, load as memory buffer
        load_offline_dataset_into_buffer(Path(offline_dir), self.replay_buffer, self.cfg.frame_stack,
                                         self.cfg.replay_buffer_size)

        if self.replay_buffer.index == -1:
            raise ValueError('No offline data loaded, check directory.')

        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames, 1)
        eval_every_step = utils.Every(self.cfg.eval_every_frames, 1)
        show_train_stats_every_step = utils.Every(self.cfg.show_train_stats_every_frames, 1)
        # only in distracting evaluation mode
        eval_save_vid_every_step = utils.Every(self.cfg.eval_save_vid_every_step,
                                               self.cfg.action_repeat)

        metrics = None
        step = 0
        while train_until_step(self.global_step):
            if show_train_stats_every_step(self.global_step):
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', step / elapsed_time)
                        log('total_time', total_time)
                        log('buffer_size', len(self.replay_buffer))
                        log('step', self.global_step)
                    step = 0
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
            step += 1
            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                # if self.eval_on_distracting:
                #     self.eval_distracting(eval_save_vid_every_step(self.global_step))
                # if self.eval_on_multitask:
                #     self.eval_multitask(eval_save_vid_every_step(self.global_step))
                self.eval(train=True)
                self.eval(train=False)

            # try to update the agent
            metrics = self.agent.update(self.replay_buffer, self.global_step)
            if show_train_stats_every_step(self.global_step):
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    with open("/home/max/Documents/phd/offline/fictional-octo-winner/wandb_info.txt") as file:
        lines = [line.rstrip() for line in file]
        os.environ["WANDB_API_KEY"] = lines[0]
        os.environ["WANDB_START_METHOD"] = "thread"
        wandb_project = "OfflineRLBenchmark"

    config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=False
    )
    train_context_name_short = os.path.split(cfg.train_context_name)[-1]
    config['train_context_name_short'] = '_'.join(train_context_name_short.split('_')[:-1])
    with wandb.init(project=wandb_project, entity=lines[1], config=config, tags=[cfg.experiment, f"{cfg.domain_name}_{cfg.task_name}", ]):
        from train import Workspace as W
        root_dir = Path.cwd()
        workspace = W(cfg)
        print(cfg)
        snapshot = root_dir / 'snapshot.pt'
        if snapshot.exists():
            print(f'resuming: {snapshot}')
            workspace.load_snapshot()
        if cfg.offline:
            workspace.train_offline(cfg.offline_dir)


if __name__ == '__main__':
    with Display(visible=False) as disp:
        main()
