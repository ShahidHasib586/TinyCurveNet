#!/usr/bin/env bash
#SBATCH --job-name=curve_s3c
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/shahid-ahamed.hasib/Downloads/LiteEnhanceNet-main/experiments/phys_curve_prog5/logs/slurm_stage3_continue_%j.out

set -euo pipefail

source /home/shahid-ahamed.hasib/venvs/uie_gpu/bin/activate

bash /home/shahid-ahamed.hasib/Downloads/LiteEnhanceNet-main/experiments/phys_curve_prog5/scripts/continue_stage3_from_global_best.sh