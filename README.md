# Predictable MDP Abstraction for Unsupervised Model-Based RL

## [Project Page](https://seohong.me/projects/pma/)

## Overview
This is the official implementation of **Predictable MDP Abstraction** (**PMA**).
The implementation is based on the codebase of [LiSP](https://github.com/kzl/lifelong_rl).

## Installation

```
conda create --name pma python=3.8
conda activate pma
pip install -r requirements.txt
```

## Examples

HalfCheeth (PMA)
```
python main_pma.py --run_group Exp --memo X --max_path_length 200 --model_master_dim 512 --num_epochs_per_save 5000 --num_epochs_per_eval 1000 --num_epochs_per_log 50 --use_gpu 1 --seed 0 --fix_variance 1 --normalize_observations 1 --train_model_determ sepmod --ensemble_size 5 --replay_buffer_size 100000 --num_epochs 10002 --mppi_num_evals 2 --mppi_planning_horizon 15 --mppi_num_candidate_sequences 256 --mppi_refine_steps 10 --mppi_gamma 1.0 --mppi_action_std 1.0 --penalty_type disagreement --env half_cheetah --dim_option 6 --sample_latent_every 1 --aux_reward_type disagreement --aux_reward_coef 0.03 --tasks forward backward --penalty_lambdas 1
```
HalfCheetah (Classic model trained with random actions)
```
python main_pma.py --run_group Exp --memo X --max_path_length 200 --model_master_dim 512 --num_epochs_per_save 5000 --num_epochs_per_eval 1000 --num_epochs_per_log 50 --use_gpu 1 --seed 0 --fix_variance 1 --normalize_observations 1 --train_model_determ sepmod --ensemble_size 5 --replay_buffer_size 100000 --num_epochs 10002 --mppi_num_evals 2 --mppi_planning_horizon 15 --mppi_num_candidate_sequences 256 --mppi_refine_steps 10 --mppi_gamma 1.0 --mppi_action_std 1.0 --penalty_type disagreement --env half_cheetah --dim_option 6 --sample_latent_every 1 --z_eq_a 1 --collect_steps 4000 --tasks forward backward --penalty_lambdas 1
```
Ant (PMA)
```
python main_pma.py --run_group Exp --memo X --max_path_length 200 --model_master_dim 512 --num_epochs_per_save 5000 --num_epochs_per_eval 1000 --num_epochs_per_log 50 --use_gpu 1 --seed 0 --fix_variance 1 --normalize_observations 1 --train_model_determ sepmod --ensemble_size 5 --replay_buffer_size 100000 --num_epochs 10002 --mppi_num_evals 2 --mppi_planning_horizon 15 --mppi_num_candidate_sequences 256 --mppi_refine_steps 10 --mppi_gamma 1.0 --mppi_action_std 1.0 --penalty_type disagreement --env ant-v3 --plot_axis -40 40 -40 40 --dim_option 8 --sample_latent_every 1 --aux_reward_type disagreement --aux_reward_coef 0.03 --tasks forward north --penalty_lambdas 20
```
Hopper (PMA)
```
python main_pma.py --run_group Exp --memo X --max_path_length 200 --model_master_dim 512 --num_epochs_per_save 5000 --num_epochs_per_eval 1000 --num_epochs_per_log 50 --use_gpu 1 --seed 0 --fix_variance 1 --normalize_observations 1 --train_model_determ sepmod --ensemble_size 5 --replay_buffer_size 100000 --num_epochs 10002 --mppi_num_evals 2 --mppi_planning_horizon 15 --mppi_num_candidate_sequences 256 --mppi_refine_steps 10 --mppi_gamma 1.0 --mppi_action_std 1.0 --penalty_type disagreement --env hopper-v3 --dim_option 3 --sample_latent_every 1 --aux_reward_type disagreement --aux_reward_coef 50 --tasks forward hop --penalty_lambdas 1 5
```
Walker2d (PMA)
```
python main_pma.py --run_group Exp --memo X --max_path_length 200 --model_master_dim 512 --num_epochs_per_save 5000 --num_epochs_per_eval 1000 --num_epochs_per_log 50 --use_gpu 1 --seed 0 --fix_variance 1 --normalize_observations 1 --train_model_determ sepmod --ensemble_size 5 --replay_buffer_size 100000 --num_epochs 10002 --mppi_num_evals 2 --mppi_planning_horizon 15 --mppi_num_candidate_sequences 256 --mppi_refine_steps 10 --mppi_gamma 1.0 --mppi_action_std 1.0 --penalty_type disagreement --env walker2d-v3 --dim_option 6 --sample_latent_every 1 --aux_reward_type disagreement --aux_reward_coef 5 --tasks forward backward --penalty_lambdas 1
```
InvertedPendulum (PMA)
```
python main_pma.py --run_group Exp --memo X --max_path_length 200 --model_master_dim 512 --num_epochs_per_save 5000 --num_epochs_per_eval 1000 --num_epochs_per_log 50 --use_gpu 1 --seed 0 --fix_variance 1 --normalize_observations 1 --train_model_determ sepmod --ensemble_size 5 --replay_buffer_size 100000 --num_epochs 10002 --mppi_num_evals 2 --mppi_planning_horizon 15 --mppi_num_candidate_sequences 256 --mppi_refine_steps 10 --mppi_gamma 1.0 --mppi_action_std 1.0 --penalty_type disagreement --env ip --dim_option 1 --sample_latent_every 1 --aux_reward_type disagreement --aux_reward_coef 0.03 --tasks forward stay --penalty_lambdas 0 1 5
```
InvertedDoublePendulum (PMA)
```
python main_pma.py --run_group Exp --memo X --max_path_length 200 --model_master_dim 512 --num_epochs_per_save 5000 --num_epochs_per_eval 1000 --num_epochs_per_log 50 --use_gpu 1 --seed 0 --fix_variance 1 --normalize_observations 1 --train_model_determ sepmod --ensemble_size 5 --replay_buffer_size 100000 --num_epochs 10002 --mppi_num_evals 2 --mppi_planning_horizon 15 --mppi_num_candidate_sequences 256 --mppi_refine_steps 10 --mppi_gamma 1.0 --mppi_action_std 1.0 --penalty_type disagreement --env idp --dim_option 1 --sample_latent_every 1 --aux_reward_type disagreement --aux_reward_coef 0.03 --tasks forward stay --penalty_lambdas 0 1 5
```
Reacher (PMA)
```
python main_pma.py --run_group Exp --memo X --max_path_length 200 --model_master_dim 512 --num_epochs_per_save 5000 --num_epochs_per_eval 1000 --num_epochs_per_log 50 --use_gpu 1 --seed 0 --fix_variance 1 --normalize_observations 1 --train_model_determ sepmod --ensemble_size 5 --replay_buffer_size 100000 --num_epochs 10002 --mppi_num_evals 2 --mppi_planning_horizon 15 --mppi_num_candidate_sequences 256 --mppi_refine_steps 10 --mppi_gamma 1.0 --mppi_action_std 1.0 --penalty_type disagreement --env reacher --dim_option 2 --sample_latent_every 1 --aux_reward_type disagreement --aux_reward_coef 0.03 --tasks default --penalty_lambdas 0 1 5
```

## License

MIT