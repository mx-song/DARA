"""Configuration file for DARA project."""
import os

# Model path or Hugging Face model id.
# Override by exporting DARA_MODEL_PATH.
MODEL_PATH = os.getenv("DARA_MODEL_PATH", "Qwen/Qwen2.5-3B-Instruct")

# Dataset parameters
P_NUM = 6  # Action dimension
N_TOTAL = 6  # Total budget
N_STEP = 3  # Steps per environment
N_ENVS = 1000  # Number of environments
N_TRY = 10  # Number of tries

# Training parameters
TRAINING_CONFIG = {
    'num_iterations': 50,
    'num_steps': 50,
    'batch_size': 6,
    'num_generations': 3,
    'max_completion_length': 500,
    'beta': 0.04,
    'learning_rate': 5e-6,
    'mu': 1,
    'epsilon': 0.1,
    'P_NUM': P_NUM,
    'N_TOTAL': N_TOTAL,
    'N_TRY': N_TRY
}

# Weights & Biases configuration.
# Never hardcode API keys in source code.
WANDB_CONFIG = {
    'mode': os.getenv('WANDB_MODE', 'offline'),
    'project': os.getenv('WANDB_PROJECT', 'dara')
}

# Paths
DATASET_DIR = "datasets"
OUTPUT_DIR = "outputs"
LOG_DIR = "logs"

# Create directories
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
