import itertools

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn as nn

from collections import OrderedDict

import lifelong_rl.torch.pytorch_util as ptu
from lifelong_rl.torch.distributions import TanhNormal
from lifelong_rl.util.eval_util import create_stats_ordered_dict
from lifelong_rl.core.rl_algorithms.torch_rl_algorithm import TorchTrainer


class SACLatentTrainer(TorchTrainer):
    def __init__(
            self,
            env,                                # Associated environment for learning
            policy,                             # Associated policy (should be TanhGaussian)
            qf1,                                # Q function #1
            qf2,                                # Q function #2
            target_qf1,                         # Slow updater to Q function #1
            target_qf2,                         # Slow updater to Q function #2
            latent_dim,
            done_ground,

            initial_alpha=1.0,

            discount=0.99,                      # Discount factor
            reward_scale=1.0,                   # Scaling of rewards to modulate entropy bonus
            use_automatic_entropy_tuning=True,  # Whether to use the entropy-constrained variant
            target_entropy=None,                # Target entropy for entropy-constraint variant

            policy_lr=3e-4,                     # Learning rate of policy and entropy weight
            qf_lr=3e-4,                         # Learning rate of Q functions
            optimizer_class=optim.Adam,         # Class of optimizer for all networks

            soft_target_tau=5e-3,               # Rate of update of target networks
            target_update_period=1,             # How often to update target networks
    ):
        super().__init__()

        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.latent_dim = latent_dim
        self.done_ground = done_ground

        self.discount = discount
        self.reward_scale = reward_scale
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item() / 2
            self.log_alpha = ptu.full(1, fill_value=np.log(initial_alpha), requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )
        else:
            self.log_alpha = ptu.full(1, fill_value=np.log(initial_alpha), requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=0,
            )

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        policy_optim_parameters = itertools.chain(self.policy.parameters())

        self.policy_optimizer = optimizer_class(
            policy_optim_parameters,
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        combined_obs = batch['observations']
        obs_dim = combined_obs.size(1) - self.latent_dim
        next_combined_obs = batch['next_observations']
        next_obs = next_combined_obs[:, :obs_dim]
        next_latents = next_combined_obs[:, obs_dim:]
        actions = batch['actions']
        rewards = batch['rewards']
        terminals = batch.get('terminals', ptu.zeros(rewards.shape[0], 1))

        alpha = self.log_alpha.exp()

        """
        QF Loss
        """
        q1_pred = self.qf1(combined_obs, actions)
        q2_pred = self.qf2(combined_obs, actions)
        _, next_policy_mean, next_policy_logstd, *_ = self.policy(torch.concat([next_obs, next_latents], dim=1))
        next_dist = TanhNormal(next_policy_mean, next_policy_logstd.exp())
        new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
        new_log_pi = new_log_pi.sum(dim=-1, keepdims=True)
        target_q_values = torch.min(
            self.target_qf1(next_combined_obs, new_next_actions),
            self.target_qf2(next_combined_obs, new_next_actions),
        ) - alpha * new_log_pi

        if self.done_ground:
            future_values = (1. - terminals) * self.discount * target_q_values
        else:
            future_values = self.discount * target_q_values
        q_target = self.reward_scale * rewards + future_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach()) * 0.5
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach()) * 0.5

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        """
        Policy and Alpha Loss
        """
        _, policy_mean, policy_logstd, *_ = self.policy(torch.concat([next_obs, next_latents], dim=1))
        dist = TanhNormal(policy_mean, policy_logstd.exp())
        new_next_actions, log_pi = dist.rsample_and_logprob()
        log_pi = log_pi.sum(dim=-1, keepdims=True)
        q_new_actions = torch.min(
            self.qf1(next_combined_obs, new_next_actions),
            self.qf2(next_combined_obs, new_next_actions),
        )

        policy_loss = alpha * log_pi - q_new_actions

        policy_loss = policy_loss.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.use_automatic_entropy_tuning:
            _, policy_mean, policy_logstd, *_ = self.policy(torch.concat([next_obs, next_latents], dim=1))
            dist = TanhNormal(policy_mean, policy_logstd.exp())
            new_obs_actions, log_pi = dist.rsample_and_logprob()
            log_pi = log_pi.sum(dim=-1, keepdims=True)

            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        else:
            alpha_loss = 0
        alpha = self.log_alpha.exp()

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self._n_train_steps_total += 1

        self.try_update_target_networks()

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False

            policy_loss = (alpha * log_pi - q_new_actions).mean()
            policy_avg_std = torch.exp(policy_logstd).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics.update(create_stats_ordered_dict('Q1 Predictions', ptu.get_numpy(q1_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('Q2 Predictions', ptu.get_numpy(q2_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('Q Targets', ptu.get_numpy(q_target)))
            self.eval_statistics.update(create_stats_ordered_dict('Log Pis', ptu.get_numpy(log_pi)))
            self.eval_statistics.update(create_stats_ordered_dict('Policy mu', ptu.get_numpy(policy_mean)))
            self.eval_statistics.update(create_stats_ordered_dict('Policy log std', ptu.get_numpy(policy_logstd)))
            self.eval_statistics['Policy Std'] = np.mean(ptu.get_numpy(policy_avg_std))

            self.eval_statistics['Alpha'] = alpha.item()
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()

        self._n_train_steps_total += 1

    def try_update_target_networks(self):
        if self._n_train_steps_total % self.target_update_period == 0:
            self.update_target_networks()

    def update_target_networks(self):
        ptu.soft_update_from_to(
            self.qf1, self.target_qf1, self.soft_target_tau
        )
        ptu.soft_update_from_to(
            self.qf2, self.target_qf2, self.soft_target_tau
        )

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        networks = [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]
        return networks

    def get_snapshot(self):
        snapshot = dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
            log_alpha=self.log_alpha,
        )
        return snapshot
