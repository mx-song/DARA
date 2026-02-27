# DARA

Dual-LLM reinforcement learning experiments for budget allocation.

## What This Repo Contains

- `src/common`: shared environment and utility functions
- `src/llma/train.py`: standalone LLMA training
- `src/llmb/train.py`: LLMB training with LLMA-assisted rollout
- `scripts/generate_dataset.py`: synthetic dataset generation
- `configs/config.py`: centralized defaults for model path and logging config

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables (optional)

```bash
cp .env.example .env
```

Common variables:

- `DARA_MODEL_PATH` (default: `Qwen/Qwen2.5-3B-Instruct`)
- `WANDB_MODE` (default: `offline`)
- `WANDB_PROJECT` (default: `dara`)
- `WANDB_API_KEY` (only needed for online W&B)

### 3. Generate training dataset

```bash
python scripts/generate_dataset.py
```

### 4. Run training

```bash
python src/llma/train.py
python src/llmb/train.py
```

## Notes

- Current defaults target research experimentation, not production training pipelines.
- Multi-GPU and `flash_attention_2` settings depend on your local CUDA/PyTorch stack.
- If you do not want online tracking, keep `WANDB_MODE=offline`.

## License

MIT. See [LICENSE](LICENSE).
