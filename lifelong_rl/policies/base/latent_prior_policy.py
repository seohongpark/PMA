import random

import torch

from lifelong_rl.policies.base.base import ExplorationPolicy
import lifelong_rl.torch.pytorch_util as ptu


class PriorLatentPolicy(ExplorationPolicy):
    def __init__(
            self,
            policy,
            prior,
            unconditional=False,
            steps_between_sampling=100,
            zero_latent=False,
            random_ratio=0.,
    ):
        self.policy = policy
        self.prior = prior
        self.unconditional = unconditional
        self.steps_between_sampling = steps_between_sampling

        self.zero_latent = zero_latent
        self.zero_latent_tensor = ptu.zeros_like(self.prior.sample())
        self.random_ratio = random_ratio

        self._steps_since_last_sample = 0
        self._last_latent = self.zero_latent_tensor

    def set_latent(self, latent):
        self._last_latent = latent

    def get_current_latent(self):
        return ptu.get_numpy(self._last_latent)

    def sample_latent(self, state=None):
        if self.unconditional or state is None:  # this will probably be changed
            latent = self.prior.sample()  # n=1).squeeze(0)
        else:
            latent = self.prior.forward(ptu.from_numpy(state))
        latent = latent.to(ptu.device)
        self.set_latent(latent)
        return latent

    def get_action(self, state, deterministic=False, z_eq_a=False, random_ratio_override=None):
        if self.zero_latent:
            latent = ptu.zeros_like(self.zero_latent_tensor)
        else:
            latent = self._last_latent
        self._steps_since_last_sample += 1

        state = ptu.from_numpy(state)
        sz = torch.cat((state, latent))
        if z_eq_a:
            action = latent
        else:
            if deterministic:
                action, *_ = self.policy.forward(sz, deterministic=True)
            else:
                random_ratio = self.random_ratio if random_ratio_override is None else random_ratio_override
                if random.random() < random_ratio:
                    action = ptu.rand(self.policy.output_size) * 2 - 1
                else:
                    action, *_ = self.policy.forward(sz, deterministic=False)
        return ptu.get_numpy(action), dict()

    def get_only_action(self, state, latent, deterministic=False, z_eq_a=False):
        state = ptu.from_numpy(state)
        sz = torch.cat((state, ptu.from_numpy(latent)))
        if z_eq_a:
            action = ptu.from_numpy(latent)
        else:
            action, *_ = self.policy.forward(sz, deterministic=deterministic)
        return ptu.get_numpy(action), dict()

    def eval(self):
        self.policy.eval()

    def train(self):
        self.policy.train()
