#!/bin/bash
#SBATCH --account=
#SBATCH --job-name=predict_model
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=BEGIN,END,FAIL

module load pytorch

srun python predict_model.py data/deuce_model_checkpoints/deuce_model.ckpt config/deuce/deuce_continue_2 default --model bcnn --data fmi &> predict_model.out

seff $SLURM_JOBID
