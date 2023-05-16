#!/bin/bash
#SBATCH --account=project_2006199
#SBATCH --job-name=train_ms_bcnn_beta
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=BEGIN,END,FAIL

module load pytorch

srun python train_model.py config/multiscale_bcnn_1 puhti --model multiscale_loss_bcnn --data fmi --callback ensemble &> train_ms_bcnn_beta_2.out

seff $SLURM_JOBID
