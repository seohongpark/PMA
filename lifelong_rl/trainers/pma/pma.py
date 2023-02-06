import gtimer as gt
import numpy as np
import torch

from collections import OrderedDict

from lifelong_rl.core.rl_algorithms.torch_rl_algorithm import TorchTrainer
import lifelong_rl.torch.pytorch_util as ptu
from lifelong_rl.trainers.pma.empowerment_functions import calculate_contrastive_empowerment
from lifelong_rl.util.eval_util import create_stats_ordered_dict
import lifelong_rl.util.pythonplusplus as ppp


class PMATrainer(TorchTrainer):
    def __init__(
            self,
            control_policy,
            discriminator,
            det_discriminator,
            replay_buffer,
            policy_trainer,
            replay_size,
            num_prior_samples=512,
            num_discrim_updates=32,
            num_policy_updates=64,
            discrim_learning_rate=1e-3,
            policy_batch_size=128,
            reward_bounds=(-10, 10),
            empowerment_horizon=1,
            reward_scale=5.,
            restrict_input_size=0,
            relabel_rewards=True,
            train_every=1,
            reward_mode=None,
            latent_dim=None,
            aux_reward_type='none',
            aux_reward_coef=0.,
    ):
        super().__init__()

        self.control_policy = control_policy
        self.discriminator = discriminator
        self.det_discriminator = det_discriminator
        self.replay_buffer = replay_buffer
        self.policy_trainer = policy_trainer

        self.obs_dim = replay_buffer.obs_dim()
        self.action_dim = replay_buffer.action_dim()
        self.latent_dim = latent_dim

        self.num_prior_samples = num_prior_samples
        self.num_discrim_updates = num_discrim_updates
        self.num_policy_updates = num_policy_updates
        self.policy_batch_size = policy_batch_size
        self.reward_bounds = reward_bounds
        self.empowerment_horizon = empowerment_horizon
        self.reward_scale = reward_scale
        self.restrict_input_size = restrict_input_size
        self.relabel_rewards = relabel_rewards
        self.reward_mode = reward_mode
        self.aux_reward_type = aux_reward_type
        self.aux_reward_coef = aux_reward_coef

        self.discrim_optim = torch.optim.Adam(discriminator.parameters(), lr=discrim_learning_rate)
        if det_discriminator is not None:
            self.det_discrim_optim = torch.optim.Adam(det_discriminator.parameters(), lr=discrim_learning_rate)
        else:
            self.det_discrim_optim = None

        self._obs = np.zeros((replay_size, self.obs_dim))
        self._next_obs = np.zeros((replay_size, self.obs_dim))       # obs + empowerment_horizon
        self._true_next_obs = np.zeros((replay_size, self.obs_dim))  # obs + 1 (normal next_obs)
        self._latents = np.zeros((replay_size, self.latent_dim))
        self._next_latents = np.zeros((replay_size, self.latent_dim))
        self._actions = np.zeros((replay_size, self.action_dim))
        self._rewards = np.zeros((replay_size, 1))
        self._env_rewards = np.zeros((replay_size, 1))
        self._next_env_rewards = np.zeros((replay_size, 1))
        self._logprobs = np.zeros((replay_size, 1))
        self._terminals = np.zeros((replay_size, 1))
        self._ptr = 0
        self.replay_size = replay_size
        self._cur_replay_size = 0

        self.obs_mean, self.obs_std = None, None

        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self._epoch_size = None
        self.eval_statistics = OrderedDict()

        self.train_every = train_every
        self._train_calls = 0

    def add_sample(self, obs, next_obs, true_next_obs, action,
                   latent, next_latent, logprob=None,
                   env_reward=None, next_env_reward=None, terminal=None, **kwargs):
        self._obs[self._ptr] = obs
        self._next_obs[self._ptr] = next_obs
        self._true_next_obs[self._ptr] = true_next_obs
        self._actions[self._ptr] = action
        self._latents[self._ptr] = latent
        self._next_latents[self._ptr] = next_latent
        self._env_rewards[self._ptr] = env_reward
        self._next_env_rewards[self._ptr] = next_env_reward
        self._terminals[self._ptr] = terminal

        if logprob is not None:
            self._logprobs[self._ptr] = logprob

        self._ptr = (self._ptr + 1) % self.replay_size
        self._cur_replay_size = min(self._cur_replay_size+1, self.replay_size)

    def process_state(self, state):
        if self.restrict_input_size > 0:
            if state.ndim == 1:
                return state[:self.restrict_input_size]
            elif state.ndim == 2:
                return state[:,:self.restrict_input_size]
        else:
            return state

    def split_execute(self, func, args, is_torch_dist=False):
        split_group = 25000
        batch_size = args[0].shape[0]
        results = []
        for split_idx in range(batch_size // split_group):
            start_split = split_idx * split_group
            end_split = (split_idx + 1) * split_group
            results.append(
                func(*[arg[start_split:end_split] for arg in args])
            )
        if batch_size % split_group:
            start_split = batch_size % split_group
            results.append(
                func(*[arg[-start_split:] for arg in args])
            )
        if not is_torch_dist:
            results = np.concatenate(results)
        else:
            mean = torch.concat([result.mean for result in results], dim=0)
            stddev = torch.concat([result.stddev for result in results], dim=0)
            dist = torch.distributions.independent.Independent(
                torch.distributions.Normal(mean, stddev), 1
            )
            results = dist

        return results

    def calculate_intrinsic_rewards(self, states, next_states, latents, actions=None, env_rewards=None, *args, **kwargs):
        self.discriminator.eval()
        if self.det_discriminator is not None:
            self.det_discriminator.eval()

        if self.restrict_input_size > 0:
            states = states[:,:self.restrict_input_size]
            next_states = next_states[:,:self.restrict_input_size]

        rewards, (logp, logp_altz, denom), reward_diagnostics = calculate_contrastive_empowerment(
            self.discriminator,
            states,
            next_states,
            latents,
            num_prior_samples=self.num_prior_samples,
            distribution_type='uniform',
            obs_mean=self.obs_mean,
            obs_std=self.obs_std,
            return_diagnostics=True,
        )
        reward_diagnostics['Intrinsic State Rewards'] = rewards.mean()
        assert len(rewards.shape) == 1

        if self.aux_reward_type != 'none':
            if self.aux_reward_type == 'disagreement':
                # Assert using det discriminator with ensembles
                predicted_next_states = self.det_discriminator.predict_next_obs(states, latents, full_ensemble=True)
                aux_rewards = self.aux_reward_coef * np.mean(np.var(predicted_next_states, axis=0), axis=-1)
            else:
                raise NotImplementedError
            orig_rewards = rewards
            rewards = rewards + aux_rewards
            reward_diagnostics.update({
                'Aux Orignal Reward': orig_rewards.mean(),
                'Aux Aux Reward': aux_rewards.mean(),
                'Aux Total Reward': rewards.mean(),
            })

        reward_diagnostics['Pure Intrinsic Reward'] = np.mean(rewards)

        return rewards, (logp, logp_altz, denom), reward_diagnostics

    def reward_postprocessing(self, rewards, *args, **kwargs):
        # Some scaling of the rewards can help; it is very finicky though
        rewards *= self.reward_scale
        rewards = np.clip(rewards, *self.reward_bounds)  # stabilizes training
        return rewards, dict()

    def train_from_paths(self, paths, train_discrim=True, train_policy=True, det_paths=None):

        """
        Reading new paths: append latent to state
        Note that is equivalent to on-policy when latent buffer size = sum of paths length
        """

        def get_train_discriminator_input(cur_paths):
            epoch_obs, epoch_next_obs, epoch_latents, epoch_actions = [], [], [], []
            for path in cur_paths:
                obs = path['observations']
                next_obs = path['next_observations']
                latents = path.get('latents', None)
                actions = path['actions']
                path_len = len(obs) - self.empowerment_horizon + 1

                for t in range(path_len):
                    epoch_obs.append(obs[t:t+1])
                    epoch_next_obs.append(next_obs[t+self.empowerment_horizon-1:t+self.empowerment_horizon])
                    epoch_latents.append(np.expand_dims(latents[t], axis=0))
                    epoch_actions.append(actions[t:t+1])

            epoch_obs = np.concatenate(epoch_obs, axis=0)
            epoch_next_obs = np.concatenate(epoch_next_obs, axis=0)
            epoch_latents = np.concatenate(epoch_latents, axis=0)
            epoch_actions = np.concatenate(epoch_actions, axis=0)

            return epoch_obs, epoch_next_obs, epoch_latents, epoch_actions

        for path in paths:
            obs = path['observations']
            next_obs = path['next_observations']
            actions = path['actions']
            latents = path.get('latents', None)
            path_len = len(obs) - self.empowerment_horizon + 1

            for t in range(path_len):
                self.add_sample(
                    obs[t],
                    next_obs[t+self.empowerment_horizon-1],
                    next_obs[t],
                    actions[t],
                    latents[t],
                    next_latent=latents[min(t + 1, path_len - 1)],
                    env_reward=path['rewards'][t],
                    next_env_reward=path['next_rewards'][t],
                    terminal=path['terminals'][t],
                )

        epoch_obs, epoch_next_obs, epoch_latents, epoch_actions = get_train_discriminator_input(paths)

        self._epoch_size = len(epoch_obs)

        gt.stamp('policy training', unique=False)

        """
        The rest is shared, train from buffer
        """

        if train_discrim:
            self.train_discriminator(epoch_obs, epoch_next_obs, epoch_latents)
            if self.det_discriminator is not None:
                epoch_obs, epoch_next_obs, epoch_latents, epoch_actions = get_train_discriminator_input(det_paths)
                self.train_discriminator(epoch_obs, epoch_next_obs, epoch_latents, det=True)
        if train_policy:
            self.train_from_buffer()

    def train_discriminator(self, obs, next_obs, latents, det=False):
        if det:
            discriminator = self.det_discriminator
            discrim_optim = self.det_discrim_optim
            label = 'Det '
        else:
            discriminator = self.discriminator
            discrim_optim = self.discrim_optim
            label = ''

        discriminator.train()
        start_discrim_loss = None

        if self.restrict_input_size > 0:
            obs = obs[:, :self.restrict_input_size]
            next_obs = next_obs[:, :self.restrict_input_size]

        for i in range(self.num_discrim_updates):
            batch = ppp.sample_batch(
                self.policy_batch_size,
                obs=obs,
                latents=latents,
                next_obs=next_obs,
            )
            batch = ptu.np_to_pytorch_batch(batch)

            discrim_loss = discriminator.get_loss(
                batch['obs'],
                batch['latents'],
                batch['next_obs'],
            )

            if i == 0:
                start_discrim_loss = discrim_loss

            discrim_optim.zero_grad()
            discrim_loss.backward()
            discrim_optim.step()

        if self._need_to_update_eval_statistics and self.num_discrim_updates > 0:
            self.eval_statistics[f'{label}Discriminator Loss'] = ptu.get_numpy(discrim_loss).mean()
            self.eval_statistics[f'{label}Discriminator Start Loss'] = ptu.get_numpy(start_discrim_loss).mean()

        gt.stamp(f'{label.lower()}discriminator training', unique=False)

    def train_from_buffer(self, reward_kwargs=None):

        """
        Compute intrinsic reward: approximate lower bound to I(s'; z | s)
        """

        # Precompute batch indices for efficient reward relabeling (only for low-level policy)
        full_inds_list = []
        ind_set = set()
        for i in range(self.num_policy_updates):
            inds = np.random.randint(0, self._cur_replay_size, self.policy_batch_size)
            full_inds_list.append(inds)
            ind_set |= set(inds)
        full_inds = list(ind_set)

        if self.relabel_rewards:
            rewards, (logp, logp_altz, denom), reward_diagnostics = self.calculate_intrinsic_rewards(
                self._obs[full_inds],
                self._next_obs[full_inds],
                self._latents[full_inds],
                actions=self._actions[full_inds],
                env_rewards=self._env_rewards[full_inds],
                reward_kwargs=reward_kwargs
            )
            orig_rewards = rewards.copy()
            rewards, postproc_dict = self.reward_postprocessing(rewards, reward_kwargs=reward_kwargs)
            reward_diagnostics.update(postproc_dict)
            self._rewards[full_inds] = np.expand_dims(rewards, axis=-1)

            gt.stamp('intrinsic reward calculation', unique=False)

        """
        Train policy
        """

        state_latents = np.concatenate([self._obs, self._latents], axis=-1)[:self._cur_replay_size]
        next_state_latents = np.concatenate([self._true_next_obs, self._next_latents], axis=-1)[:self._cur_replay_size]

        for i in range(self.num_policy_updates):
            batch = ppp.sample_batch(
                self.policy_batch_size,
                inds=full_inds_list[i],
                observations=state_latents,
                next_observations=next_state_latents,
                actions=self._actions[:self._cur_replay_size],
                rewards=self._rewards[:self._cur_replay_size],
                env_rewards=self._env_rewards[:self._cur_replay_size],
                next_env_rewards=self._next_env_rewards[:self._cur_replay_size],
                terminals=self._terminals[:self._cur_replay_size],
            )
            batch = ptu.np_to_pytorch_batch(batch)
            self.policy_trainer.train_from_torch(batch)

        gt.stamp('policy training', unique=False)

        """
        Diagnostics
        """

        if self._need_to_update_eval_statistics:
            # self._need_to_update_eval_statistics = False
            self.eval_statistics.update(self.policy_trainer.eval_statistics)

            if self.relabel_rewards:
                self.eval_statistics.update(reward_diagnostics)

                self.eval_statistics.update(create_stats_ordered_dict(
                    'Discriminator Log Pis',
                    logp,
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Discriminator Alt Log Pis',
                    logp_altz,
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Intrinsic Reward Denominator',
                    denom,
                ))

                self.eval_statistics.update(create_stats_ordered_dict(
                    'Intrinsic Rewards (Original)',
                    orig_rewards,
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Intrinsic Rewards (Processed)',
                    rewards,
                ))

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return dict(
            policy_trainer=self.eval_statistics,
        )

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True
        self.policy_trainer.end_epoch(epoch)

    @property
    def networks(self):
        networks = self.policy_trainer.networks + [self.discriminator]
        if self.det_discriminator is not None:
            networks = networks + [self.det_discriminator]
        return networks

    def get_snapshot(self):
        snapshot = dict(
            control_policy=self.control_policy,
            discriminator=self.discriminator,
            det_discriminator=self.det_discriminator,
        )

        for k, v in self.policy_trainer.get_snapshot().items():
            snapshot['policy_trainer/' + k] = v

        return snapshot
