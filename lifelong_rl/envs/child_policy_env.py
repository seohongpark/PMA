import random
from collections import defaultdict

import akro
import glob
import gym.spaces.utils
import numpy as np
import torch
import lifelong_rl.torch.pytorch_util as ptu


class ChildPolicyEnv(gym.Wrapper):
    def __init__(
            self,
            env,
            cp_path,
            cp_epoch,
            cp_z_eq_a,
            cp_action_range,
            cp_multi_step,
            cp_num_truncate_obs,
            penalty_type='none',
            penalty_lambda=0.,
            use_true_env=False,

            mbpo=False,
            mbpo_reset_ratio=0.,
            mbpo_max_path_length=None,

            cp_min_zero=False,
    ):
        super().__init__(env)

        candidates = glob.glob(cp_path)
        if len(candidates) == 0:
            raise Exception(f'Path does not exist: {cp_path}')
        if len(candidates) > 1:
            raise Exception(f'Multiple matching paths exist for: {cp_path}')
        cp_path = candidates[0]
        if cp_epoch is None:
            cp_files = glob.glob(f'{cp_path}/itr*.pt')
            cp_files.sort(key=lambda f: int(f.split('/itr_')[-1].split('.pt')[0]))
            cp_file = cp_files[-1]
        else:
            cp_file = glob.glob(f'{cp_path}/itr_{cp_epoch}.pt')[0]
        snapshot = torch.load(cp_file, map_location='cpu')

        self.z_eq_a = cp_z_eq_a
        self.mbpo = mbpo
        self.mbpo_reset_ratio = mbpo_reset_ratio
        self.mbpo_max_path_length = mbpo_max_path_length

        self.policy = snapshot['exploration/policy']
        if snapshot.get('trainer/det_discriminator') is not None:
            self.dynamics = snapshot['trainer/det_discriminator']
        else:
            self.dynamics = snapshot['trainer/discriminator']

        self.mbpo_replay_buffer = snapshot['replay_buffer/observations']

        self.policy.eval()
        self.dynamics.eval()

        self.dim_action = self.dynamics._latent_size
        self.action_range = cp_action_range

        self.multi_step = cp_multi_step
        self.num_truncate_obs = cp_num_truncate_obs

        self.observation_space = self.env.observation_space
        self.action_space = akro.Box(low=-self.action_range, high=self.action_range, shape=(self.dim_action,))

        self.use_true_env = use_true_env

        self.last_obs = None
        self.first_obs = None

        self.penalty_type = penalty_type
        self.penalty_lambda = penalty_lambda
        self.penalty_indices1 = []
        self.penalty_indices2 = []
        for i in range(self.dynamics.ensemble_size):
            for j in range(i + 1, self.dynamics.ensemble_size):
                self.penalty_indices1.append(i)
                self.penalty_indices2.append(j)

        self.cp_min_zero = cp_min_zero

    def reset(self, **kwargs):
        if not self.use_true_env and self.mbpo:
            if random.random() < self.mbpo_reset_ratio:
                ret = self.env.reset(**kwargs)
            else:
                ret = self.mbpo_replay_buffer[random.randint(0, self.mbpo_replay_buffer.shape[0] - 1)]
        else:
            ret = self.env.reset(**kwargs)

        self.last_obs = ret
        self.first_obs = ret

        return ret

    def step(self, action, **kwargs):
        cp_action = action.copy()
        sum_rewards = 0.

        done_final = False
        start_obs = self.last_obs
        for i in range(self.multi_step):
            cp_obs = self.last_obs
            if self.num_truncate_obs > 0:
                cp_obs = cp_obs[:-self.num_truncate_obs]

            if not self.use_true_env:
                predicted_next_states = self.dynamics.predict_next_obs(cp_obs.reshape(1, -1), cp_action.reshape(1, -1), full_ensemble=True)[:, 0, :]
                next_obs = predicted_next_states.mean(axis=0)

                if self.penalty_type == 'disagreement':
                    predicted_next_states1 = predicted_next_states[self.penalty_indices1]
                    predicted_next_states2 = predicted_next_states[self.penalty_indices2]
                    penalty = np.max(np.mean((predicted_next_states1 - predicted_next_states2) ** 2, axis=-1), axis=0)
                else:
                    penalty = 0.

                reward, done = self.env.compute_reward(cp_obs.reshape(1, -1), next_obs.reshape(1, -1))
                reward = reward[0]
                done = done[0]

                pure_reward = reward
                reward = reward - self.penalty_lambda * penalty
                if self.cp_min_zero:
                    reward = max(0., reward)

                info = {
                    'coordinates': start_obs[:2],
                    'next_coordinates': next_obs[:2],
                    'pure_reward': pure_reward,
                    'penalty': penalty,
                    'reward': reward,
                }
            else:
                if self.z_eq_a:
                    action = cp_action
                else:
                    action = self.policy.get_only_action(cp_obs, cp_action, deterministic=True)[0]

                next_obs, reward, done, info = self.env.step(action, **kwargs)
                info.update({
                    'pure_reward': reward,
                    'penalty': 0.,
                    'reward': reward,
                })

            self.last_obs = next_obs

            sum_rewards += reward

            if done:
                done_final = True
                break

        return next_obs, sum_rewards, done_final, info
