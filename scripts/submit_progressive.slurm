#!/bin/bash
#SBATCH --job-name=curve_prog5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -e
cd /home/shahid-ahamed.hasib/Downloads/LiteEnhanceNet-main/experiments/phys_curve_prog5

source /home/shahid-ahamed.hasib/venvs/uie_gpu/bin/activate

python -u src/train_progressive.py --config configs/progressive.yaml
