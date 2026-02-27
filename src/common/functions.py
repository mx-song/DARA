"""
Common utility functions shared across LLMA and LLMB
"""
import re
import numpy as np
import torch
import torch.nn as nn
import random


def extract_answer_from_model_output(response: str) -> np.ndarray:
    """
    Extract the answer list from the model's response in the specified format:

    <answer>
    [y1, y2, ..., yN]
    </answer>

    Steps:
    1. Locate the <answer> and </answer> tags.
    2. Use regular expressions to find the array within square brackets.
    3. Parse the list into a numpy array of floats.
    """
    # Step 1: Locate the answer section
    answer_start = response.lower().find('<answer>')
    answer_end = response.lower().find('</answer>')

    if answer_start == -1 or answer_end == -1:
        return np.array([], dtype=np.float32)

    # Extract the content between the tags
    answer_content = response[answer_start + len('<answer>'):answer_end].strip()

    # Step 2: Use regex to extract the list inside square brackets
    match = re.search(r'\[(.*?)\]', answer_content)
    if not match:
        return np.array([], dtype=np.float32)

    # Extract the content of the list
    number_str = match.group(1)

    # Handle potential separators and whitespace
    number_str = number_str.replace(';', ',')

    # Step 3: Convert to a list of floats
    try:
        nums = [float(x.strip()) for x in number_str.split(',') if x.strip()]
        return np.array(nums, dtype=np.float32)
    except ValueError:
        return np.array([], dtype=np.float32)


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def selective_log_softmax(logits, input_ids):
    """Compute log softmax for selected tokens"""
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)


def compute_log_probs(model, input_ids, attention_mask, logits_to_keep):
    """Compute log probabilities for generated tokens"""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]
    return selective_log_softmax(logits, input_ids)


def create_completion_mask(completion_ids, eos_token_id):
    """Create mask for completion tokens"""
    is_eos = completion_ids == eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
    mask_exists = is_eos.any(dim=1)
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
    sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
    return (sequence_indices <= eos_idx.unsqueeze(1)).int()


def optimize_model_memory(model):
    """Optimize model for memory efficiency during training"""
    model.train()
    model.config.use_cache = False

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    model.gradient_checkpointing_enable()
    return model
