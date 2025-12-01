# Prompt Enhancer LLM for Video Generation

Final project for MIT [6.7920](https://web.mit.edu/6.7920/www/): Reinforcement Learning

![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97HuggingFace-dariakryvosheieva%2Fvideo--prompt--enhancer-yellow?link=https%3A%2F%2Fhuggingface.co%2Fdariakryvosheieva%2Fvideo-prompt-enhancer)

## Overview

This repo contains training data, training scripts, and inference scripts for the project, which trained an LLM to translate simple video generation prompts into detailed, professional-grade prompts, eliminating the need for prompt engineering and bringing high-quality AI-generated videos to ordinary users.

The model was post-trained from [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) with LoRA via a two-stage procedure:
1. **Next-token prediction** on synthetic pairs of simple and corresponding detailed prompts;
2. **Online RL**:

    2.1. The model accepts a simple prompt and generates a detailed prompt;
   
    2.2. A Sora video is generated based on the detailed prompt;
   
    2.3. The video and its alignment with the simple prompt are scored using [VisionReward](https://github.com/zai-org/VisionReward/tree/main);
   
    2.4. The model is updated via PPO.

## Setup

1. Clone the repo: `git clone https://github.com/dariakryvosheieva/video-prompt-enhancer.git`
2. Create a virtual environment: `python -m venv .venv`
3. Activate the environment: `. .venv/bin/activate`
4. Install packages: `pip install -r requirements.txt`

## Repo Contents

* `data_stage1/`: (simple, detailed) prompt pairs for Stage 1
* `data_stage2/`: simple prompts for the Stage 2
* `out/`:
  * `qwen2.5-14b-prompt-enhancer-lora/`: checkpoint from Stage 1 (not included)
  * `qwen2.5-14b-prompt-enhancer-lora-stage2/`: checkpoint from Stage 2 (not included)
  * `video_cache/`: Sora videos generated in Stage 2
  * `stage2_traces.json`: a log of simple prompts sampled at each step, detailed prompts generated, and rewards earned
* `VisionReward_video/`: files required for VisionReward scoring
* `train_stage1.py`: training script for Stage 1
* `train_stage2.py`: training script for Stage 2
* `qualitative_eval.py`: inference from local checkpoints
* `hf_inference.py`: inference from the HF model
* `job.sh`: shell script for running training/inference on GPU

## Tech Stack

* PEFT
* PyTorch
* Sora API
* transformers
* TRL
* VisionReward
