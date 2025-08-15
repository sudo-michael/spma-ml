#!/bin/bash
#SBATCH -J spma_sb3
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=3
#SBATCH --output=solar/%N-%j.out
#SBATCH --partition=mars-lab-short
#SBATCH --nodelist=cs-venus-06

use_wandb=('true')
experiment_name=('spma_sb3_n_epochs_10_3')

# Common parameters
env_id_list=('Hopper-v4')
seed_list=(1 2 3 4 5)


# SPMA parameters
spma_eta_list=(0.7 0.9)
spma_n_steps_list=(2048)
spma_batch_size_list=(2048)
spma_use_armijo_actor_list=('true')
spma_use_armijo_critic_list=('true')
spma_total_timesteps_list=(2048000)
spma_n_epochs_list=(10)

parallel -j 3 \
    OMP_NUM_THREADS=1 \
    uv run run_experiment.py ::: \
    --use_wandb ::: ${use_wandb[@]} ::: \
    --exp_name ::: ${experiment_name[@]} ::: \
    --env_id ::: ${env_id_list[@]} ::: \
    --seed ::: ${seed_list[@]} ::: \
    'SPMA' ::: \
    --eta ::: ${spma_eta_list[@]} ::: \
    --n_steps ::: ${spma_n_steps_list[@]} ::: \
    --batch_size ::: ${spma_batch_size_list[@]} ::: \
    --total_timesteps ::: ${spma_total_timesteps_list[@]} ::: \
    --use_armijo_critic ::: ${spma_use_armijo_critic_list[@]} ::: \
    --use_armijo_actor ::: ${spma_use_armijo_actor_list[@]} ::: \
    --n_epochs ::: ${spma_n_epochs_list[@]}

