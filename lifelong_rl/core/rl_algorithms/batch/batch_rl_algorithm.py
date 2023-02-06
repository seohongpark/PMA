import abc
import functools
from collections import defaultdict

import gtimer as gt
import gym
import numpy as np
import torch

from lifelong_rl.core import logger
from lifelong_rl.core.logging.utils import FigManager, get_option_colors, record_video, draw_2d_gaussians
from lifelong_rl.core.rl_algorithms.rl_algorithm import BaseRLAlgorithm
import lifelong_rl.torch.pytorch_util as ptu


class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):

    def __init__(
            self,
            trainer,
            exploration_policy,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop=1,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            save_snapshot_freq=100,
            post_epoch_funcs=None,
            num_epochs_per_eval=1,
            num_epochs_per_log=1,
            plot_axis=None,
            eval_record_video=False,
            video_skip_frames=1,
            train_model_determ='off',

            mppi_eval=False,
            mppi_num_evals=5,
            mppi_kwargs=None,
            penalty_type='none',
            penalty_lambdas=None,
            tasks=None,

            mbpo=False,
            mbpo_max_path_length=None,
    ):
        super().__init__(
            trainer,
            exploration_policy,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            save_snapshot_freq=save_snapshot_freq,
            post_epoch_funcs=post_epoch_funcs,
        )

        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        assert num_train_loops_per_epoch == 1
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

        self.did_eval = False
        self.num_epochs_per_eval = num_epochs_per_eval
        self.num_epochs_per_log = num_epochs_per_log
        self.plot_axis = plot_axis
        self.eval_record_video = eval_record_video
        self.video_skip_frames = video_skip_frames

        self.train_model_determ = train_model_determ

        self.mppi_eval = mppi_eval
        self.mppi_num_evals = mppi_num_evals
        self.mppi_kwargs = mppi_kwargs
        self.penalty_type = penalty_type
        self.penalty_lambdas = penalty_lambdas if penalty_lambdas is not None else [0.]
        self.tasks = tasks

        self.mbpo = mbpo
        self.mbpo_max_path_length = mbpo_max_path_length

        self.target_eval_paths = None

    def _train(self):
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

            self._fit_input_stats()

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            if self.num_epochs_per_eval > 0 and epoch % self.num_epochs_per_eval == 0:
                self._eval(epoch)
                self.did_eval = True
                gt.stamp('evaluation sampling')
            else:
                self.did_eval = False
                gt.stamp('evaluation sampling')

            exp_max_path_length = self.max_path_length if not self.mbpo else self.mbpo_max_path_length
            new_expl_paths = self.expl_data_collector.collect_new_paths(
                exp_max_path_length,
                self.num_expl_steps_per_train_loop,
                discard_incomplete_paths=False,
            )
            if 'reward' in new_expl_paths[0]['env_infos'][0]:
                # Log cp stats
                for key in ['reward', 'penalty', 'pure_reward']:
                    path_keys = [sum([path[key] for path in new_expl_path['env_infos']]) for new_expl_path in new_expl_paths]
                    logger.record_tabular(f'exploration/sum_{key}', np.mean(path_keys[:-1]))  # Discard incomplete path

            if self.train_model_determ == 'sepmod':
                new_det_paths = self.eval_data_collector.collect_new_paths(
                    exp_max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                # If want eval statistics, should carefully deal with clearing
                self.eval_data_collector.end_epoch(epoch)  # Clear paths
            else:
                new_det_paths = None
            gt.stamp('exploration sampling', unique=False)

            self.replay_buffer.add_paths(new_expl_paths)
            gt.stamp('data storing', unique=False)

            self.training_mode(True)
            for _ in range(self.num_trains_per_train_loop):
                self.trainer.train_from_paths(new_expl_paths, det_paths=new_det_paths)
            gt.stamp('training', unique=False)
            self.training_mode(False)

            self._fit_input_stats()

            if self.did_eval:
                self.eval_data_collector._epoch_paths = self.target_eval_paths
            self._end_epoch(epoch)

    def _eval(self, epoch):
        num_eval_paths = 50
        num_video_paths = 9

        # Persistent
        # Use the same skill for the entire trajectory
        sample_latent_every = self.eval_data_collector.sample_latent_every
        self.eval_data_collector.sample_latent_every = None
        # Video
        if self.eval_record_video:
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                num_video_paths * self.max_path_length,
                discard_incomplete_paths=True,
                render=True,
                )
            record_video('Video_Persistent', epoch, list(self.eval_data_collector._epoch_paths)[:2 * num_video_paths], skip_frames=self.video_skip_frames)  # At most 2 * num_video_paths videos (for memory)
            self.eval_data_collector.end_epoch(epoch)  # Clear paths
        # Figure
        self.eval_data_collector.collect_new_paths(
            self.max_path_length,
            num_eval_paths * self.max_path_length,
            discard_incomplete_paths=True,
        )
        with FigManager('Traj_Persistent', epoch) as fm:
            env = self.eval_data_collector._env
            trajs = self.eval_data_collector._epoch_paths
            latents = np.array([traj['latent'] for traj in trajs])
            colors = get_option_colors(latents)
            env.render_trajectories(
                list(trajs)[:num_eval_paths], colors[:num_eval_paths], self.plot_axis, fm.ax
            )
        self.target_eval_paths = self.eval_data_collector._epoch_paths
        self.eval_data_collector.end_epoch(epoch)  # Clear paths
        self.eval_data_collector.sample_latent_every = sample_latent_every

        # Resample
        # Respect the original sample_latent_every
        # Video
        if self.eval_record_video:
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                num_video_paths * self.max_path_length,
                discard_incomplete_paths=True,
                render=True,
                )
            record_video('Video_Resample', epoch, list(self.eval_data_collector._epoch_paths)[:2 * num_video_paths], skip_frames=self.video_skip_frames)
            self.eval_data_collector.end_epoch(epoch)  # Clear paths
        # Figure
        self.eval_data_collector.collect_new_paths(
            self.max_path_length,
            num_eval_paths * self.max_path_length,
            discard_incomplete_paths=True,
            )
        with FigManager('Traj_Resample', epoch) as fm:
            env = self.eval_data_collector._env
            trajs = self.eval_data_collector._epoch_paths
            colors = np.zeros((num_eval_paths, 4))
            colors[:, 3] = 1.
            env.render_trajectories(
                list(trajs)[:num_eval_paths], colors[:num_eval_paths], self.plot_axis, fm.ax
            )
        self.eval_data_collector.end_epoch(epoch)  # Clear paths

        if self.mppi_num_evals > 0:
            for task in self.tasks:
                for penalty_lambda in self.penalty_lambdas:
                    label = 'state_true_rew'
                    label = f'{label}_lam{penalty_lambda}'

                    renders = []
                    log_keys = ['mse', 'penalty']
                    rewards = []
                    logs = {log_key: [] for log_key in log_keys}
                    for i in range(self.mppi_num_evals):
                        result = self._mppi(
                            render=True, penalty_type=self.penalty_type, penalty_lambda=penalty_lambda, task=task,
                            **self.mppi_kwargs
                        )
                        rewards.append(result['actual_reward'])
                        for log_key in log_keys:
                            logs[log_key].extend(result[log_key])
                        if self.eval_record_video:
                            render = np.array([info['render'] for info in result['infos']])
                            renders.append(render)

                    logger.record_tabular(f'mppi_{label}_eval/{task}_reward', np.mean(rewards))
                    for log_key in log_keys:
                        logger.record_tabular(f'mppi_{label}_eval/{task}_{log_key}', np.mean(logs[log_key]))
                    if self.eval_record_video:
                        record_video(f'Video_Plan_{label}_{task}', epoch, trajectories=None, renders=renders, skip_frames=self.video_skip_frames)

    def _mppi(
            self,
            planning_horizon=15,
            primitive_horizon=1,
            num_candidate_sequences=256,
            refine_steps=10,
            gamma=1.0,
            action_std=1.0,
            smoothing_beta=0,
            penalty_type='none',
            penalty_lambda=0.,
            task='default',

            render=False,
    ):
        env = self.eval_data_collector._env
        episode_horizon = self.max_path_length
        latent_action_space_size = self.trainer.latent_dim
        policy = self.expl_policy
        if self.train_model_determ == 'sepmod':
            dynamics = self.trainer.det_discriminator
        else:
            dynamics = self.trainer.discriminator
        policy.eval()
        dynamics.eval()
        process_state = self.trainer.process_state  # For restrict obs

        step_idx = 0

        def _smooth_primitive_sequences(primitive_sequences):
            for planning_idx in range(1, primitive_sequences.shape[1]):
                primitive_sequences[:, planning_idx, :] = smoothing_beta * primitive_sequences[:, planning_idx - 1, :] + (1. - smoothing_beta) * primitive_sequences[:, planning_idx, :]
            return primitive_sequences

        def _get_init_primitive_parameters():
            prior_mean = functools.partial(
                np.random.multivariate_normal,
                mean=np.zeros(latent_action_space_size),
                cov=np.diag(np.ones(latent_action_space_size)))
            prior_cov = lambda: action_std * np.diag(np.ones(latent_action_space_size))
            return [prior_mean(), prior_cov()]

        # update new primitive means for horizon sequence
        def _update_parameters(candidates, reward, primitive_parameters):
            reward = np.exp(gamma * (reward - np.max(reward)))
            reward = reward / (reward.sum() + 1e-10)
            new_means = (candidates.T * reward).T.sum(axis=0)

            for planning_idx in range(candidates.shape[1]):
                primitive_parameters[planning_idx][0] = new_means[planning_idx]

        def _get_expected_primitive(params):
            return params[0]

        obs = env.reset()
        if isinstance(env.observation_space, gym.spaces.Dict):
            # For manipulation environments, hacky
            task = obs
        actual_coords = [np.expand_dims(obs[:2], 0)]
        actual_reward = 0.

        primitive_parameters = []
        chosen_primitives = []

        for _ in range(planning_horizon):
            primitive_parameters.append(_get_init_primitive_parameters())

        penalty_indices1 = []
        penalty_indices2 = []
        for i in range(dynamics.ensemble_size):
            for j in range(i + 1, dynamics.ensemble_size):
                penalty_indices1.append(i)
                penalty_indices2.append(j)

        infos = []
        logs = defaultdict(list)
        for i in range(episode_horizon // primitive_horizon):
            for j in range(refine_steps):
                concat_mean = np.concatenate([p[0] for p in primitive_parameters])
                concat_std = np.full_like(concat_mean, action_std)
                candidate_primitive_sequences = np.random.normal(concat_mean, concat_std, (num_candidate_sequences, concat_mean.shape[0]))
                candidate_primitive_sequences = np.clip(candidate_primitive_sequences, -1., 1.)
                candidate_primitive_sequences = candidate_primitive_sequences.reshape((num_candidate_sequences, planning_horizon, latent_action_space_size))
                candidate_primitive_sequences = _smooth_primitive_sequences(candidate_primitive_sequences)

                running_cur_state = np.array([process_state(obs)] * num_candidate_sequences)
                running_reward = np.zeros(num_candidate_sequences)
                running_done = np.zeros(num_candidate_sequences)
                for planning_idx in range(planning_horizon):
                    cur_primitives = candidate_primitive_sequences[:, planning_idx, :]
                    for _ in range(primitive_horizon):
                        predicted_next_states = dynamics.predict_next_obs(running_cur_state, cur_primitives, full_ensemble=True)
                        predicted_next_state = predicted_next_states.mean(axis=0)

                        if penalty_type == 'disagreement':
                            predicted_next_states1 = predicted_next_states[penalty_indices1]
                            predicted_next_states2 = predicted_next_states[penalty_indices2]
                            penalty = np.max(np.mean((predicted_next_states1 - predicted_next_states2) ** 2, axis=-1), axis=0)
                        else:
                            penalty = 0.

                        reward, done = env.compute_reward(running_cur_state, predicted_next_state, task=task)
                        running_reward += (reward - penalty_lambda * penalty) * (1 - running_done)
                        running_done = np.minimum(running_done + done, 1)
                        running_cur_state = predicted_next_state

                _update_parameters(candidate_primitive_sequences, running_reward, primitive_parameters)

            chosen_primitive = _get_expected_primitive(primitive_parameters[0])
            chosen_primitives.append(chosen_primitive)

            for _ in range(primitive_horizon):
                action = policy.get_only_action(obs, chosen_primitive, deterministic=True, z_eq_a=self.expl_data_collector.z_eq_a)[0]
                next_obs, reward, done, info = env.step(action, task=task, render=render)
                actual_reward += reward

                predicted_next_ensemble_states = dynamics.predict_next_obs(process_state(obs[np.newaxis]), chosen_primitive[np.newaxis], full_ensemble=True)
                predicted_next_state = predicted_next_ensemble_states.mean(axis=0)[0]
                mse = np.square(predicted_next_state - process_state(next_obs)).mean()

                if penalty_type == 'disagreement':
                    predicted_next_states1 = predicted_next_ensemble_states[penalty_indices1]
                    predicted_next_states2 = predicted_next_ensemble_states[penalty_indices2]
                    penalty = np.max(np.mean((predicted_next_states1 - predicted_next_states2) ** 2, axis=-1), axis=0)
                    logs['penalty'].append(penalty[0])
                else:
                    penalty = np.zeros(1)
                    logs['penalty'].append(0.)

                logs['mse'].append(mse)
                infos.append(info)

                # prepare for next iteration
                obs = next_obs
                actual_coords.append(np.expand_dims(obs[:2], 0))
                step_idx += 1

                if done:
                    break
            if done:
                break

            primitive_parameters.pop(0)
            primitive_parameters.append(_get_init_primitive_parameters())

        actual_coords = np.concatenate(actual_coords)
        return dict(
            logs,
            actual_reward=actual_reward,
            actual_coords=actual_coords,
            primitives=np.array(chosen_primitives),
            infos=infos,
        )
