import numpy as np
import torch

import lifelong_rl.torch.pytorch_util as ptu


class ParallelizedLayer(torch.nn.Module):
    def __init__(
            self,
            ensemble_size,
            input_dim,
            output_dim,
    ):
        super().__init__()

        w_init = ptu.randn((ensemble_size, input_dim, output_dim))
        ptu.glorot_uniform_unit(w_init)
        self.W = torch.nn.Parameter(w_init, requires_grad=True)

        b_init = ptu.zeros((ensemble_size, 1, output_dim)).float()
        self.b = torch.nn.Parameter(b_init, requires_grad=True)

    def forward(self, x):
        # assumes x is 3D: (ensemble_size, batch_size, dimension)
        # Batch matrix multiplication: (a, b, c) @ (a, c, d) -> (a, b, d)
        return x @ self.W + self.b


class SkillDynamics(torch.nn.Module):

    def __init__(
            self,
            observation_size,
            action_size,
            latent_size,
            concat_action,
            normalize_observations=True,
            squash_mean=False,
            fc_layer_params=(256, 256),
            fix_variance=True,
            activation_func=torch.nn.ReLU,
            omit_obs_dim=0,
            ensemble_size=1,
    ):
        super().__init__()

        self._observation_size = observation_size
        self._action_size = action_size
        self._latent_size = latent_size
        self._concat_action = concat_action
        self._normalize_observations = normalize_observations
        self._squash_mean = squash_mean

        self._fc_layer_params = fc_layer_params
        self._fix_variance = fix_variance
        self._omit_obs_dim = omit_obs_dim

        self.ensemble_size = ensemble_size

        in_dim = observation_size + latent_size - omit_obs_dim
        if concat_action:
            in_dim += action_size

        in_layers = []
        if self._normalize_observations:
            self.in_preproc = torch.nn.BatchNorm1d(in_dim, momentum=0.01, eps=1e-3)
            self.out_preproc = torch.nn.BatchNorm1d(observation_size, momentum=0.01, eps=1e-3, affine=False)
        in_layers.append(ParallelizedLayer(ensemble_size, in_dim, fc_layer_params[0]))
        self.in_func = torch.nn.Sequential(*in_layers)

        layers = []
        for i in range(len(fc_layer_params)-1):
            if i == 0:
                layers.append(activation_func())
            layers.append(ParallelizedLayer(ensemble_size, fc_layer_params[i], fc_layer_params[i+1]))
            layers.append(activation_func())
        self.model = torch.nn.Sequential(*layers)

        self.out_mean = ParallelizedLayer(ensemble_size, fc_layer_params[-1], observation_size)
        if not self._fix_variance:
            self.out_std = ParallelizedLayer(ensemble_size, fc_layer_params[-1], observation_size)

    def _get_distribution(self, obs, latents, actions=None, unnormalize=False, only_first=True):
        orig_obs = obs
        obs = obs[:, self._omit_obs_dim:]
        if self._concat_action:
            x = torch.cat([obs, latents, actions], dim=-1)
        else:
            x = torch.cat([obs, latents], dim=-1)
        if self._normalize_observations:
            x = self.in_preproc(x)
        x = x.unsqueeze(0)  # dim becomes 3
        x = x.repeat(self.ensemble_size, 1, 1)

        x = self.in_func(x)
        x = self.model(x)

        mean = self.out_mean(x)
        if self._squash_mean:
            mean = ptu.squash_to_range(mean, -30, 30)
        if self._fix_variance:
            std = ptu.ones(*mean.shape)
        else:
            log_std = self.out_std(x)
            low = np.log(np.exp(0.1) - 1)
            high = np.log(np.exp(10.0) - 1)
            log_std = ptu.squash_to_range(log_std, low, high)
            std = log_std.exp().add(1.).log()

        if only_first:
            mean = mean[0]
            std = std[0]

        if unnormalize:
            if self._normalize_observations:
                # Now normalized (s'-s)
                mean = mean * torch.sqrt(self.out_preproc.running_var + self.out_preproc.eps) + self.out_preproc.running_mean
                std = std * torch.sqrt(self.out_preproc.running_var + self.out_preproc.eps)
            # Now unnormalized (s'-s)
            mean = mean + orig_obs

        dist = torch.distributions.independent.Independent(
            torch.distributions.Normal(mean, std), 1
        )

        return dist

    def get_log_prob(self, obs, latents, next_obs, return_dist=False, only_first=True):
        assert not self._concat_action

        deltas = next_obs - obs
        if self._normalize_observations:
            deltas = self.out_preproc(deltas)
        dist = self._get_distribution(obs, latents, only_first=only_first)
        if return_dist:
            return dist.log_prob(deltas), dist
        else:
            return dist.log_prob(deltas)

    def get_loss(self, obs, latents, next_obs, weights=None):
        assert not self._concat_action

        total_loss = 0

        # Loss
        log_probs, dist = self.get_log_prob(obs, latents, next_obs, return_dist=True, only_first=False)
        if weights is not None:
            log_probs = log_probs * weights
        total_loss = total_loss + (-log_probs.mean())

        return total_loss

    def predict_next_obs(self, obs, latents, full_ensemble=False):
        assert not self._concat_action

        obs = ptu.from_numpy(obs)
        latents = ptu.from_numpy(latents)
        dist = self._get_distribution(obs, latents, unnormalize=True, only_first=False)
        if full_ensemble:
            next_obs = dist.mean
        else:
            next_obs = dist.mean.mean(dim=0)
        return ptu.get_numpy(next_obs)
