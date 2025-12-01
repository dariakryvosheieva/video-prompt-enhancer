#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=100G
python hf_inference.py
