import argparse
import os
import sys
from datetime import datetime

from experiment_configs.configs.pma.pma_config import get_config
from experiment_configs.algorithms.batch import get_algorithm
from experiment_configs.base_experiment import experiment as run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_group', type=str, default='Debug')
    parser.add_argument('--memo', type=str, default=None)
    parser.add_argument('--algo_name', type=str, default=None)  # Only for logging
    parser.add_argument('--env', type=str, default='maze', choices=[
        'maze', 'half_cheetah',
        'ant-v3', 'hopper-v3', 'walker2d-v3',
        'ip', 'idp', 'reacher',
    ])
    parser.add_argument('--tasks', type=str, default=['default'], nargs='*')
    parser.add_argument('--max_path_length', type=int, default=200)
    parser.add_argument('--use_gpu', type=int, default=0, choices=[0, 1])
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=1000000)
    parser.add_argument('--num_epochs_per_save', type=int, default=5000)
    parser.add_argument('--num_epochs_per_eval', type=int, default=500)
    parser.add_argument('--num_epochs_per_log', type=int, default=1)
    parser.add_argument('--plot_axis', type=float, default=None, nargs='*')
    parser.add_argument('--video_skip_frames', type=int, default=1)
    parser.add_argument('--model_master_dim', type=int, default=512)
    parser.add_argument('--dyn_num_layers', type=int, default=2)
    parser.add_argument('--dim_option', type=int, default=2)
    parser.add_argument('--collect_steps', type=int, default=2000)
    parser.add_argument('--num_policy_updates', type=int, default=64)
    parser.add_argument('--num_discrim_updates', type=int, default=None)
    parser.add_argument('--reward_scale', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--replay_buffer_size', type=int, default=None)

    parser.add_argument('--normalize_observations', type=int, default=1)
    parser.add_argument('--fix_variance', type=int, default=1)
    parser.add_argument('--det_fix_variance', type=int, default=1)

    parser.add_argument('--z_eq_a', type=int, default=0, choices=[0, 1])
    parser.add_argument('--sample_latent_every', type=int, default=1)
    parser.add_argument('--aux_reward_type', type=str, default='none', choices=['none', 'disagreement'])
    parser.add_argument('--aux_reward_coef', type=float, default=0.)

    parser.add_argument('--restore_path', type=str, default=None)  # Currently only for planning.
    parser.add_argument('--restore_idx', type=int, default=None)
    parser.add_argument('--restore_epoch', type=int, default=None)

    parser.add_argument('--cp_path', type=str, default=None)
    parser.add_argument('--cp_idx', type=int, default=None)
    parser.add_argument('--cp_epoch', type=int, default=None)
    parser.add_argument('--cp_z_eq_a', type=int, default=0, choices=[0, 1])  # Should be 1 when using z_eq_a child policy
    parser.add_argument('--cp_always_use_true_env', type=int, default=0, choices=[0, 1])
    parser.add_argument('--cp_min_zero', type=int, default=0, choices=[0, 1])  # Set to 1 when train MBPO on Ant

    parser.add_argument('--mbpo', type=int, default=0)
    parser.add_argument('--mbpo_reset_ratio', type=float, default=0)
    parser.add_argument('--mbpo_max_path_length', type=int, default=0)

    parser.add_argument('--train_model_determ', type=str, default='sepmod', choices=['off', 'sepmod'])
    parser.add_argument('--ensemble_size', type=int, default=1)

    parser.add_argument('--mppi_num_evals', type=int, default=2)
    parser.add_argument('--mppi_planning_horizon', type=int, default=5)
    parser.add_argument('--mppi_num_candidate_sequences', type=int, default=50)
    parser.add_argument('--mppi_refine_steps', type=int, default=10)
    parser.add_argument('--mppi_gamma', type=float, default=1.0)
    parser.add_argument('--mppi_action_std', type=float, default=1.0)
    parser.add_argument('--penalty_type', type=str, default='none', choices=['none', 'disagreement'])
    parser.add_argument('--penalty_lambdas', type=float, default=[0.], nargs='*')

    args = parser.parse_args(sys.argv[1:])

    if args.env == 'maze':
        env_kwargs = dict(
            n=args.max_path_length,
        )
    else:
        env_kwargs = dict()

    # For the environments with early termination, set done_ground=1.
    # We additionally find it helpful to use last_ground=1 for InvertedPendulum and InvertedDoublePendlum.
    if args.env == 'half_cheetah':
        done_ground = 0
        last_ground = 0
        omit_input_size = 1
    elif args.env == 'ant-v3':
        done_ground = 1
        last_ground = 0
        omit_input_size = 2
    elif args.env == 'hopper-v3':
        done_ground = 1
        last_ground = 0
        omit_input_size = 1
        env_kwargs.update(
            action_repetition=5,
        )
    elif args.env == 'walker2d-v3':
        done_ground = 1
        last_ground = 0
        omit_input_size = 1
        env_kwargs.update(
            action_repetition=5,
        )
    elif args.env == 'ip':
        done_ground = 1
        last_ground = 1
        omit_input_size = 0
    elif args.env == 'idp':
        done_ground = 1
        last_ground = 1
        omit_input_size = 0
    elif args.env == 'reacher':
        done_ground = 0
        last_ground = 0
        omit_input_size = 0
    elif args.env == 'maze':
        done_ground = 0
        last_ground = 0
        omit_input_size = 0
    else:
        raise NotImplementedError()

    if args.cp_path is not None:
        env_kwargs.update(
            cp_info=dict(
                cp_path=args.cp_path,
                cp_epoch=args.cp_epoch,
                cp_z_eq_a=args.cp_z_eq_a,
                cp_action_range=1.0,
                cp_multi_step=1,
                cp_num_truncate_obs=0,
                use_true_env=True if args.cp_always_use_true_env else False,

                mbpo=args.mbpo,
                mbpo_reset_ratio=args.mbpo_reset_ratio,
                mbpo_max_path_length=args.mbpo_max_path_length,

                penalty_type=args.penalty_type,
                penalty_lambda=args.penalty_lambdas[0],
                cp_min_zero=args.cp_min_zero,
            ),
        )

    use_gpu = args.use_gpu
    eval_record_video = (args.env != 'maze')
    replay_buffer_size = args.collect_steps if args.replay_buffer_size is None else args.replay_buffer_size
    variant = dict(
        seed=args.seed,
        memo=args.memo,
        algo_name=args.algo_name,
        algorithm='PMA',
        collector_type='batch_latent',
        replay_buffer_size=replay_buffer_size,
        generated_replay_buffer_size=replay_buffer_size,
        sample_latent_every=args.sample_latent_every,
        z_eq_a=args.z_eq_a,
        latent_dim=args.dim_option,
        aux_reward_type=args.aux_reward_type,
        aux_reward_coef=args.aux_reward_coef,
        ensemble_size=args.ensemble_size,
        done_ground=done_ground,
        last_ground=last_ground,

        restore_path=args.restore_path,
        restore_epoch=args.restore_epoch,
        env_name=args.env,
        env_kwargs=env_kwargs,
        policy_kwargs=dict(
            layer_size=args.model_master_dim,
            latent_dim=args.dim_option,
            omit_input_size=omit_input_size,
        ),
        discriminator_kwargs=dict(
            layer_size=args.model_master_dim,
            num_layers=args.dyn_num_layers,
            restrict_input_size=0,
            omit_input_size=omit_input_size,
            normalize_observations=args.normalize_observations,
            fix_variance=args.fix_variance,
            det_fix_variance=args.det_fix_variance,
        ),
        trainer_kwargs=dict(
            num_prior_samples=100,
            num_discrim_updates=args.num_policy_updates // 2 if args.num_discrim_updates is None else args.num_discrim_updates,
            num_policy_updates=args.num_policy_updates,
            discrim_learning_rate=args.learning_rate,
            policy_batch_size=args.batch_size,
            reward_bounds=(-1e12, 1e12),
            reward_scale=args.reward_scale,
        ),
        policy_trainer_kwargs=dict(
            discount=0.995,
            policy_lr=args.learning_rate,
            qf_lr=args.learning_rate,
            soft_target_tau=5e-3,
        ),
        algorithm_kwargs=dict(
            num_epochs=args.num_epochs,
            num_eval_steps_per_epoch=args.collect_steps,
            num_trains_per_train_loop=1,
            num_expl_steps_per_train_loop=args.collect_steps,
            min_num_steps_before_training=0,
            max_path_length=args.max_path_length,
            save_snapshot_freq=args.num_epochs_per_save,

            num_epochs_per_eval=args.num_epochs_per_eval,
            num_epochs_per_log=args.num_epochs_per_log,
            plot_axis=args.plot_axis,
            eval_record_video=eval_record_video,
            video_skip_frames=args.video_skip_frames,

            train_model_determ=args.train_model_determ,

            mppi_num_evals=args.mppi_num_evals,
            penalty_type=args.penalty_type,
            penalty_lambdas=args.penalty_lambdas,
            tasks=args.tasks,
            mppi_kwargs=dict(
                planning_horizon=args.mppi_planning_horizon,
                primitive_horizon=1,
                num_candidate_sequences=args.mppi_num_candidate_sequences,
                refine_steps=args.mppi_refine_steps,
                gamma=args.mppi_gamma,
                action_std=args.mppi_action_std,
                smoothing_beta=0.,
            ),

            mbpo=args.mbpo,
            mbpo_max_path_length=args.mbpo_max_path_length,
        ),
    )

    experiment_config = dict()

    if get_config is not None:
        experiment_config['get_config'] = get_config
    if get_algorithm is not None:
        experiment_config['get_algorithm'] = get_algorithm

    g_start_time = int(datetime.now().timestamp())

    exp_name = ''
    exp_name += f'sd{args.seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    if 'SLURM_RESTART_COUNT' in os.environ:
        exp_name += f'rs_{os.environ["SLURM_RESTART_COUNT"]}.'
    exp_name += f'{g_start_time}'

    def list_to_str(arg_list):
        return str(arg_list).replace(",", "|").replace(" ", "").replace("'", "")

    def add_name(abbr, argument, value_dict=None, max_length=None, log_only_if_changed=False):
        nonlocal exp_name

        value = getattr(args, argument)
        if log_only_if_changed and parser.get_default(argument) == value:
            return
        if isinstance(value, list):
            if value_dict is not None:
                value = [value_dict.get(v) for v in value]
            value = list_to_str(value)
        elif value_dict is not None:
            value = value_dict.get(value)

        if value is None:
            value = 'X'

        if max_length is not None:
            value = str(value)[:max_length]

        if isinstance(value, str):
            value = value.replace('/', '-')

        exp_name += f'_{abbr}{value}'

    add_name('', 'env', {
        'maze': 'MZ',
        'half_cheetah': 'CH',
        'ant-v3': 'ANT3',
        'hopper-v3': 'HP3',
        'walker2d-v3': 'WK3',
        'ip': 'IP',
        'idp': 'IDP',
        'reacher': 'RC',
    }, log_only_if_changed=False)

    add_name('mm', 'memo')
    add_name('do', 'dim_option')
    add_name('sl', 'sample_latent_every')
    add_name('za', 'z_eq_a')

    run_experiment(
        experiment_config=experiment_config,
        run_group=args.run_group,
        exp_prefix=exp_name,
        variant=variant,
        gpu_kwargs={'mode': use_gpu},
        log_to_wandb=('WANDB_API_KEY' in os.environ),
    )


if __name__ == "__main__":
    main()
