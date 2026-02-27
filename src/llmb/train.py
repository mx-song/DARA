# -*- coding: utf-8 -*-
import re
import torch.nn as nn
import random
import copy
import os
import sys
import glob
import numpy as np
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.common.env import RewardFunctionEnv
from src.common.utils import load_data
from src.common.functions import (
    extract_answer_from_model_output,
    set_random_seed,
    selective_log_softmax,
    compute_log_probs,
    create_completion_mask,
    optimize_model_memory
)


def log(*args, **kwargs):
    message = " ".join(map(str, args))
    print(message)
    with open("local.txt", "a", encoding="utf-8") as f:
        f.write(message + "\n")


# set_random_seed and extract_answer_from_model_output are now imported from src.common.functions

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
    # 🔍 Step 1: Locate the answer section
    answer_start = response.lower().find('<answer>')
    answer_end = response.lower().find('</answer>')

    if answer_start == -1 or answer_end == -1:
        # log("[⚠️] Unable to find <answer> tags in the response.")
        return np.array([], dtype=np.float32)

    # Extract the content between the tags
    answer_content = response[answer_start + len('<answer>'):answer_end].strip()

    # 🔍 Step 2: Use regex to extract the list inside square brackets
    match = re.search(r'\[(.*?)\]', answer_content)
    if not match:
        # log("[⚠️] No valid list found inside <answer> tags.")
        return np.array([], dtype=np.float32)

    # Extract the content of the list
    number_str = match.group(1)

    # Handle potential separators and whitespace
    number_str = number_str.replace(';', ',')

    # 🔍 Step 3: Convert to a list of floats
    try:
        nums = [float(x.strip()) for x in number_str.split(',') if x.strip()]
        # log(f"[✅] Successfully parsed answer: {nums}")
        return np.array(nums, dtype=np.float32)
    except ValueError as e:
        # log(f"[⚠️] Parsing failed: {e}")
        return np.array([], dtype=np.float32)


def environment_reward(completions, samples, envs, NUM, TOTAL):
    rewards = []
    num_generations = len(completions) // len(envs)

    for env_index, env in enumerate(envs):
        start_idx = env_index * num_generations
        end_idx = (env_index + 1) * num_generations
        env_completions = completions[start_idx:end_idx]

        IsFIRST = True

        for gen_index, completion in enumerate(env_completions):
            response = completion[0]['content']
            # if env_index == 0 and gen_index == 0:
                # log(response)
            action = extract_answer_from_model_output(response)

            reward_nega = 0
            if action.shape != (NUM,):

                # log(f"This Wrong Response is: {response}")

                if action.shape[0] > NUM:

                    action = action[:NUM]
                    reward_nega = -1.5
                elif action.shape[0] < NUM:

                    action = np.pad(action, (0, NUM - action.shape[0]), 'constant', constant_values=0)
                    reward_nega = -3


            # env = RewardFunctionEnv(n=NUM, current_func_type=samples[i]['type'], current_func_params=samples[i]['params'])
            _, reward, _, _ = env.step(action, False)
            IsFIRST = False

            total_reward = reward.sum()
            total_reward = total_reward + reward_nega
            # total_reward = reward.sum()


            action_sum = np.sum(action)
            if action_sum != NUM:
                penalty = abs(NUM - action_sum)
                total_reward -= penalty

            # if env_index == 0 and gen_index == 0:

                # log(f"\n[✅] Sample Env {env_index + 1}, Generation {gen_index + 1}")
                # log("Parsed Action:", action.tolist())
                # log("Reward:", reward.tolist())
                # log("Total Reward:", total_reward)

            rewards.append(float(total_reward))

    # log(f"TAG1:rewards shape: {len(rewards)}")

    return rewards


# selective_log_softmax, compute_log_probs, create_completion_mask are now imported from src.common.functions


def generate_completions(model, tokenizer, prompts, num_generations=4, max_completion_length=32):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    prompt_ids = inputs["input_ids"].to(device)
    prompt_mask = inputs["attention_mask"].to(device)
    log(f"Input batch size: {prompt_ids.size(0)}, Device before model: {prompt_ids.device}")
    prompt_length = prompt_ids.size(1)
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
    outputs = model.generate(
        prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_completion_length,
        do_sample=True,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=False
    )
    log(f"Output batch size: {outputs.size(0)}, Device after model: {outputs.device}")
    completion_ids = outputs[:, prompt_length:]
    completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id)
    return prompt_ids, prompt_mask, completion_ids, completion_mask


def generate_rollout_data(model, ref_model, tokenizer, batch_samples, num_generations, max_completion_length, NUM=6,
                          TOTAL=5, envs=None, SHOW=False, use_model_a=False, model_a=None, ref_model_a=None):
    # target:
    # prompts = [build_prompt(sample, NUM=NUM, TOTAL=TOTAL) for sample in batch_samples]
    prompts = [env.generate_prompt(sample, NUM=NUM, TOTAL=TOTAL) for env, sample in zip(envs, batch_samples)]
    

    current_model = model_a if use_model_a and model_a is not None else model
    current_ref_model = ref_model_a if use_model_a and ref_model_a is not None else ref_model
    
    with torch.no_grad():

        prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(current_model, tokenizer, prompts,
                                                                                        num_generations,
                                                                                        max_completion_length)

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)



        old_log_probs = compute_log_probs(current_model, input_ids, attention_mask, logits_to_keep)
        ref_log_probs = compute_log_probs(current_ref_model, input_ids, attention_mask, logits_to_keep)
    formatted_completions = [[{'content': tokenizer.decode(ids, skip_special_tokens=True)}] for ids in completion_ids]

    repeated_samples = [s for s in batch_samples for _ in range(num_generations)]

    for i in range(0, len(formatted_completions), num_generations):
        # log("i:", i)

        action = extract_answer_from_model_output(formatted_completions[i][0]['content'])
        if action.shape != (NUM,):
            if action.shape[0] > NUM:
                action = action[:NUM]
            elif action.shape[0] < NUM:
                action = np.pad(action, (0, NUM - action.shape[0]), 'constant', constant_values=0)

        env_idx = int(i / num_generations)



        if use_model_a:

            _, reward, _, _ = envs[env_idx].step(action, False)
        else:

            if i % num_generations == 0:
                _, reward, _, _ = envs[env_idx].step(action, True)
            else:
                _, reward, _, _ = envs[env_idx].step(action, False)

        # if i == 0 and SHOW:
            # log("Prompt:", prompts[0]) 
            # log("Response:", formatted_completions[i][0]['content'])
        log("Reward:", reward, reward.sum(), sep=" ")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
        "logits_to_keep": logits_to_keep,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "formatted_completions": formatted_completions,
        "repeated_samples": repeated_samples,
        "batch_size": len(prompts),
        "num_generations": num_generations
    }


def grpo_loss(model, batch_samples, rollout_data, reward_function, envs, P_NUM, TOTAL, beta=0.01, epsilon=0.2, step=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    logits_to_keep = rollout_data["logits_to_keep"]
    old_log_probs = rollout_data["old_log_probs"]
    ref_log_probs = rollout_data["ref_log_probs"]
    token_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
    ratio = torch.exp(token_log_probs - old_log_probs)
    rewards = torch.tensor(
        reward_function(completions=rollout_data["formatted_completions"], samples=rollout_data["repeated_samples"],
                        envs=envs,
                        NUM=P_NUM, TOTAL=TOTAL),
        dtype=torch.float32, device=device)
    # log(f"Rewards: {rewards}")  # Debug rewards
    batch_size = rollout_data["batch_size"]
    num_generations = rollout_data["num_generations"]

    # log(f"TAG2:rewards shape: {rewards.shape}")

    rewards = rewards.view(batch_size, num_generations)
    # log(f"TAG3:rewards shape: {rewards.shape}")

    avg_reward = rewards.mean().item()
    # log("Average Reward:", avg_reward)


    with open("reward_log.txt", "a") as f:
        f.write(f"{avg_reward:.6f}\n")
    # print("Average Reward:", avg_reward)

    # rewards: Tensor of shape [batch_size, num_generations]
    batch_size = rollout_data["batch_size"]
    num_generations = rollout_data["num_generations"]
    rewards = rewards.view(batch_size, num_generations)


    flat = rewards.view(-1)

    best_idx = torch.argmax(flat).item()
    best_value = flat[best_idx].item()

    best_response = rollout_data["formatted_completions"][best_idx][0]["content"].strip()

    mean_rewards = rewards.mean(dim=1).repeat_interleave(num_generations)
    std_rewards = rewards.std(dim=1).repeat_interleave(num_generations)
    advantages = ((rewards.view(-1) - mean_rewards) / (std_rewards + 1e-4)).unsqueeze(1)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    surrogate_loss = torch.min(surr1, surr2)
    kl = torch.exp(ref_log_probs - token_log_probs) - (
            ref_log_probs - token_log_probs) - 1  # Δ = ref_log_probs - token_log_probs，kl ≈ e^Δ - Δ - 1
    per_token_loss = surrogate_loss - beta * kl




    loss = -((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(
        dim=1)).mean()
    return loss, avg_reward


# optimize_model_memory is now imported from src.common.functions


def create_envs(samples, NUM, initial_actions=None):
    """
    Pre-create environment instances to avoid repeated construction.
    initial_actions: Optional initial action list from LLMA outputs.
    """
    envs = []
    for idx, sample in enumerate(samples):
        initial_action = initial_actions[idx] if initial_actions is not None and idx < len(initial_actions) else None
        env = RewardFunctionEnv(n=NUM, current_func_type=sample['type'], current_func_params=sample['params'], 
                               initial_action=initial_action)
        envs.append(env)
    return envs


def train_with_grpo(model, tokenizer, train_data, num_iterations=1, num_steps=500, batch_size=4,
                    num_generations=4, max_completion_length=800, beta=0.1,
                    learning_rate=5e-6, mu=3, epsilon=0.2, reward_function=None, P_NUM=6, N_TOTAL=6, N_TRY=20,
                    device_ids=None, model_a=None, ref_model_a=None):
    assert device_ids is not None and len(device_ids) > 1, "This code needs at least 2 GPU cores to run!"

    # Wrap model with DataParallel if multiple GPUs are available.
    model = nn.DataParallel(model, device_ids=device_ids)
    log(f"Model wrapped with DataParallel across GPUs: {device_ids}")

    # Outer loop: iterative GRPO updates.

    for iteration in range(num_iterations):
        log(f"\nIteration {iteration + 1}/{num_iterations}")

        # Create a reference model (deep copy) and set it to eval mode.

        ref_model = copy.deepcopy(model.module)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        log("Reference model created.")

        # Reinitialize the optimizer for this iteration.



        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        model.train()

        if iteration % 10 == 0:
            Show = True
        else:
            Show = False

        # Inner loop: your original training steps.

        for step in range(num_steps):
            batch_samples = random.sample(train_data, batch_size)


            initial_actions = None
            if model_a is not None:
                log("Generating initial actions using model A...")

                temp_envs = []
                for sample in batch_samples:
                    temp_env = RewardFunctionEnv(n=P_NUM, current_func_type=sample['type'], 
                                                 current_func_params=sample['params'], initial_action=None)
                    temp_envs.append(temp_env)
                
                prompts = [env.generate_prompt(sample, NUM=P_NUM, TOTAL=N_TOTAL) 
                          for env, sample in zip(temp_envs, batch_samples)]
                

                with torch.no_grad():
                    prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(
                        model_a, tokenizer, prompts, num_generations=1, max_completion_length=max_completion_length)
                
                formatted_completions = [[{'content': tokenizer.decode(ids, skip_special_tokens=True)}] 
                                        for ids in completion_ids]
                initial_actions = []
                for completion in formatted_completions:
                    action = extract_answer_from_model_output(completion[0]['content'])
                    if action.shape != (P_NUM,):
                        if action.shape[0] > P_NUM:
                            action = action[:P_NUM]
                        elif action.shape[0] < P_NUM:
                            action = np.pad(action, (0, P_NUM - action.shape[0]), 'constant', constant_values=0)
                    initial_actions.append(action)
                log(f"Generated {len(initial_actions)} initial actions from model A")

            # import ipdb;ipdb.set_trace()
            envs = create_envs(batch_samples, P_NUM, initial_actions=initial_actions)

            rollout_list = []
            with torch.no_grad():
                for n_try in range(N_TRY):

                    use_model_a = (n_try == 0)
                    rd = generate_rollout_data(model.module, ref_model, tokenizer, batch_samples, num_generations,
                                               max_completion_length, NUM=P_NUM, TOTAL=N_TOTAL, envs=envs, SHOW=Show,
                                               use_model_a=use_model_a, model_a=model_a, ref_model_a=ref_model_a)
                    rollout_list.append(rd)

            for mb_idx, rollout_data in enumerate(rollout_list, start=1):
                for grpo_iter in range(mu):
                    loss, avg_reward = grpo_loss(model.module, batch_samples, rollout_data, reward_function, envs=envs,
                                                 P_NUM=P_NUM, TOTAL=N_TOTAL, beta=beta, epsilon=epsilon, step=step)
                    optimizer.zero_grad()
                    loss.backward()

                    # Check if gradients exist
                    num_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
                    total_params = sum(1 for _ in model.parameters())
                    percent_with_grad = 100.0 * num_with_grad / total_params

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                    optimizer.step()


                    wandb.log({
                        "loss": loss.item(),
                        "average_reward": avg_reward,
                        "iteration": iteration + 1,
                        "step": step + 1,
                        "grpo_iter": grpo_iter + 1,
                        "gradient_percentage": percent_with_grad,
                    })

                    log(f"Iteration {iteration + 1}/{num_iterations}, Step {step + 1}/{num_steps}, "
                          f"GRPO iter {grpo_iter + 1}/{mu}, loss: {loss.item():.4f}")

    return model.module


if __name__ == "__main__":
    P_NUM = 6
    N_TOTAL = 6
    N_STEP = 3
    N_ENVS = 1000
    N_TRY = 10

    set_random_seed(42)
    os.environ["WANDB_MODE"] = os.getenv("WANDB_MODE", "offline")
    os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", "dara")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"Using primary device: {device}")
    num_gpus = torch.cuda.device_count()
    log(f"Detected {num_gpus} GPUs")
    device_ids = list(range(1, num_gpus)) if num_gpus > 1 else None  # [0, 1, 2, ..., n-1]


    model_name = os.getenv("DARA_MODEL_PATH", "Qwen/Qwen2.5-3B-Instruct")
    

    log("Loading model B...")



    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2", device_map="balanced_low_0")
    # model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2",device_map="auto")
    log("Loaded model B")
    

    log("Loading model A...")
    model_a = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                   attn_implementation="flash_attention_2", device_map="balanced_low_0")
    model_a.eval()
    for param in model_a.parameters():
        param.requires_grad = False
    log("Loaded model A")


    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model_a.config.pad_token_id = tokenizer.eos_token_id
    model_a.config.eos_token_id = tokenizer.eos_token_id
    

    ref_model_a = copy.deepcopy(model_a)
    ref_model_a.eval()
    for param in ref_model_a.parameters():
        param.requires_grad = False

    all_data = load_data(f"datasets/reward_dataset_dim{P_NUM}_sum{N_TOTAL}_steps{N_STEP}_numEnvs{N_ENVS}.csv",
                         group_size=N_STEP, n_try_max=N_TRY, P_NUM=P_NUM)
    # random.shuffle(all_data)
    size_of_eval_data = 3
    eval_data = all_data[:size_of_eval_data]
    train_data = all_data[size_of_eval_data:]

    env = RewardFunctionEnv(n=P_NUM)

    model = optimize_model_memory(model)

    log("\nStarting RL fine-tuning using GRPO...")
    # This config was tested on a 8xA100 node, where each A100 is has 80GB of VRAM
    training_config = {'num_iterations': 50, 'num_steps': 50, 'batch_size': 8,
                       # reduce if you have fewer GPUs
                       'num_generations': 12,  # reduce if you have GPUs with less VRAM
                       'max_completion_length': 500,  # reduce if you have GPUs with less VRAM
                       'beta': 0.04, 'learning_rate': 5e-6, 'mu': 1, 'epsilon': 0.1, 'P_NUM': P_NUM, 'N_TOTAL': N_TOTAL,
                       'N_TRY': N_TRY}

    wandb.login()
    wandb.init(project=os.environ["WANDB_PROJECT"], reinit=True, save_code=True)
    log("Weights & Biases initialized.")

    model = train_with_grpo(model=model, tokenizer=tokenizer, train_data=train_data, reward_function=environment_reward,
                            device_ids=device_ids, model_a=model_a, ref_model_a=ref_model_a, **training_config)

    wandb.finish()
    log("Training completed and wandb run finished.")

    log("\nSaving GRPO fine-tuned model...")
    model.save_pretrained("grpo_finetuned_model")
    tokenizer.save_pretrained("grpo_finetuned_model")
