#!/bin/bash
#SBATCH --account=project_2006199
#SBATCH --job-name=predict_ms_bcnn_noiter
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=BEGIN,END,FAIL

module load pytorch

srun python predict_model.py results/multiscale_bcnn_noiter/multiscale_bcnn_noiter_ms_bcnn_noiter.ckpt config/ms_bcnn puhti --model multiscale_loss_bcnn --data fmi &> predict_ms_bcnn_noiter.out

seff $SLURM_JOBID
