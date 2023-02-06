import functools
import glob
import torch

from lifelong_rl.policies.base.latent_prior_policy import PriorLatentPolicy
from lifelong_rl.policies.models.gaussian_policy import TanhGaussianPolicy
from lifelong_rl.models.networks import OmitMlp
from lifelong_rl.trainers.pma.pma import PMATrainer
from lifelong_rl.trainers.pma.skill_dynamics import SkillDynamics
import lifelong_rl.torch.pytorch_util as ptu
import lifelong_rl.util.pythonplusplus as ppp
from lifelong_rl.trainers.q_learning.sac_latent import SACLatentTrainer


def get_config(
        variant,
        expl_env,
        eval_env,
        obs_dim,
        action_dim,
        replay_buffer,
):

    """
    Policy construction
    """

    M = variant['policy_kwargs']['layer_size']
    latent_dim = variant['policy_kwargs']['latent_dim']
    omit_dim = variant['discriminator_kwargs']['omit_input_size']
    restrict_dim = variant['discriminator_kwargs']['restrict_input_size']
    discrim_kwargs = variant['discriminator_kwargs']

    restore_path = variant['restore_path']
    restore_epoch = variant['restore_epoch']
    if restore_path is not None:
        candidates = glob.glob(restore_path)
        if len(candidates) == 0:
            raise Exception(f'Path does not exist: {restore_path}')
        if len(candidates) > 1:
            raise Exception(f'Multiple matching paths exist for: {restore_path}')
        restore_path = candidates[0]
        if restore_epoch is None:
            restore_files = glob.glob(f'{restore_path}/itr*.pt')
            restore_files.sort(key=lambda f: int(f.split('/itr_')[-1].split('.pt')[0]))
            restore_file = restore_files[-1]
        else:
            restore_file = glob.glob(f'{restore_path}/itr_{restore_epoch}.pt')[0]
        snapshot = torch.load(restore_file, map_location=ptu.device)
    else:
        snapshot = None

    control_policy = TanhGaussianPolicy(
        obs_dim=obs_dim + latent_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
        omit_obs_dim=omit_dim,
    )
    if snapshot is not None:
        control_policy = snapshot['trainer/control_policy']

    prior = torch.distributions.uniform.Uniform(
        -ptu.ones(latent_dim), ptu.ones(latent_dim),
    )

    policy = PriorLatentPolicy(
        policy=control_policy,
        prior=prior,
        unconditional=True,
    )

    def get_qfs(latent_dim, action_dim):
        return ppp.group_init(
            4,
            OmitMlp,
            input_size=obs_dim + latent_dim + action_dim,
            omit_dim=omit_dim,
            output_size=1,
            hidden_sizes=[M, M],
            hidden_init=ptu.variance_scaling_init,
            w_scale=1 / 3,
        )

    qf1, qf2, target_qf1, target_qf2 = get_qfs(latent_dim, action_dim)
    if snapshot is not None:
        qf1 = snapshot['trainer/policy_trainer/qf1']
        qf2 = snapshot['trainer/policy_trainer/qf2']
        target_qf1 = snapshot['trainer/policy_trainer/target_qf1']
        target_qf2 = snapshot['trainer/policy_trainer/target_qf2']
        replay_buffer.load_snapshot(snapshot)

    """
    Discriminator
    """

    train_model_determ = variant['algorithm_kwargs']['train_model_determ']
    def get_discriminator(ensemble_size, fix_variance):
        return SkillDynamics(
            observation_size=obs_dim if restrict_dim == 0 else restrict_dim,
            action_size=action_dim,
            latent_size=latent_dim,
            concat_action=False,
            normalize_observations=discrim_kwargs.get('normalize_observations', True),
            squash_mean=False,
            fix_variance=fix_variance,
            fc_layer_params=[discrim_kwargs['layer_size']] * discrim_kwargs['num_layers'],
            omit_obs_dim=omit_dim,
            ensemble_size=ensemble_size,
        )

    discriminator = get_discriminator(ensemble_size=1, fix_variance=discrim_kwargs['fix_variance'])
    if train_model_determ == 'sepmod':
        det_discriminator = get_discriminator(ensemble_size=variant['ensemble_size'], fix_variance=discrim_kwargs['det_fix_variance'])
    else:
        det_discriminator = None
    if snapshot is not None:
        discriminator = snapshot['trainer/discriminator']
        if snapshot.get('trainer/det_discriminator') is not None:
            discriminator = snapshot['trainer/det_discriminator']  # Use deterministic one if possible
            det_discriminator = snapshot['trainer/det_discriminator']

    """
    Policy trainer
    """

    policy_trainer = SACLatentTrainer(
        env=expl_env,
        policy=control_policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        latent_dim=latent_dim,
        done_ground=variant['done_ground'],
        **variant['policy_trainer_kwargs'],
    )
    """
    Setup of intrinsic control
    """

    trainer_class = PMATrainer

    trainer = trainer_class(
        control_policy=control_policy,
        discriminator=discriminator,
        det_discriminator=det_discriminator,
        replay_buffer=replay_buffer,
        replay_size=variant['generated_replay_buffer_size'],
        policy_trainer=policy_trainer,
        restrict_input_size=restrict_dim,
        latent_dim=latent_dim,
        aux_reward_type=variant['aux_reward_type'],
        aux_reward_coef=variant['aux_reward_coef'],
        **variant['trainer_kwargs'],
    )

    """
    Create config dict
    """

    config = dict()
    config.update(dict(
        trainer=trainer,
        exploration_policy=policy,
        evaluation_policy=policy,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        replay_buffer=replay_buffer,
        prior=prior,
        control_policy=control_policy,
        latent_dim=latent_dim,
        policy_trainer=policy_trainer,
    ))
    config['algorithm_kwargs'] = variant.get('algorithm_kwargs', dict())
    config['offline_kwargs'] = variant.get('offline_kwargs', dict())

    if snapshot is not None:
        policy_trainer.log_alpha = snapshot['trainer/policy_trainer/log_alpha']
    return config
