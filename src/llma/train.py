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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.common.env import RewardFunctionEnv
from src.common.utils import load_data


def log(*args, **kwargs):
    message = " ".join(map(str, args))
    print(message)
    with open("local.txt", "a", encoding="utf-8") as f:
        f.write(message + "\n")



def set_random_seed(seed: int = 42):
    """Set random seeds for reproducible training behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        # print("Unable to find <answer> tags in the response.")
        return np.array([], dtype=np.float32)

    # Extract the content between the tags
    answer_content = response[answer_start + len('<answer>'):answer_end].strip()

    # Step 2: Use regex to extract the list inside square brackets
    match = re.search(r'\[(.*?)\]', answer_content)
    if not match:
        # print("No valid list found inside <answer> tags.")
        return np.array([], dtype=np.float32)

    # Extract the content of the list
    number_str = match.group(1)

    # Handle potential separators and whitespace
    number_str = number_str.replace(';', ',')

    # Step 3: Convert to a list of floats
    try:
        nums = [float(x.strip()) for x in number_str.split(',') if x.strip()]
        # print(f"✅ Successfully parsed answer: {nums}")
        return np.array(nums, dtype=np.float32)
    except ValueError as e:
        # print(f"Parsing failed: {e}")
        return np.array([], dtype=np.float32)


def evaluate_model(model, tokenizer, eval_data, device, NUM=10, TOTAL=15, num_try=20, max_new_tokens=700):
    model.eval()
    all_rewards = []
    print("\n" + "=" * 50)
    print("EVALUATION IN ENVIRONMENT")
    print("=" * 50)

    for trial, sample in enumerate(eval_data):
        print(f"\nTrial {trial + 1}/{len(eval_data)}")

        temp_sample = sample
        env = RewardFunctionEnv(n=NUM, current_func_type=temp_sample['type'], current_func_params=temp_sample['params'])

        for n_try in range(1, num_try + 1):
            print(f"\nTry {n_try}/{num_try}")

            if n_try == 1:
                prompt = env.generate_prompt(temp_sample, NUM=NUM, TOTAL=TOTAL)
            # else:
                # prompt = env.generate_prompt2(temp_sample, NUM=NUM, TOTAL=TOTAL)
            print(prompt)

            # prompt = build_prompt(temp_sample, NUM=NUM, TOTAL=TOTAL)
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    forced_eos_token_id=tokenizer.eos_token_id,
                    early_stopping=True
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            action = extract_answer_from_model_output(response)

            reward_nega = 0

            if action.shape[0] > NUM:
                print(f"Action dimension exceeds expected size ({action.shape[0]} > {NUM}); truncating automatically.")
                action = action[:NUM]
                reward_nega = -1.5
            elif action.shape[0] < NUM:
                print(f"Action dimension is smaller than expected ({action.shape[0]} < {NUM}); padding with 1.")
                action = np.pad(action, (0, NUM - action.shape[0]), 'constant', constant_values=1)
                reward_nega = -1.5


            _, reward, _, _ = env.step(action, True)

            total_reward = reward.sum()
            total_reward = total_reward + reward_nega

            all_rewards.append(total_reward)

    avg_reward = np.mean(all_rewards)
    print(f"\n✅ Average Reward over {len(eval_data)} trials: {avg_reward:.4f}")
    print("=" * 50)

    model.train()
    return avg_reward


def environment_reward(completions, samples, envs, NUM, TOTAL):
    rewards = []
    num_generations = len(completions) // len(envs)

    for env_index, env in enumerate(envs):
        start_idx = env_index * num_generations
        end_idx = (env_index + 1) * num_generations
        env_completions = completions[start_idx:end_idx]

        IsFIRST = True


        data_sample = samples[env_index]
        hist_actions = data_sample["action"]  # shape: [T, NUM]
        hist_rewards = data_sample["reward"]  # shape: [T, NUM]

        for gen_index, completion in enumerate(env_completions):
            response = completion[0]['content']
            action = extract_answer_from_model_output(response)

            reward_nega = 0
            if action.shape != (NUM,):
                if action.shape[0] > NUM:
                    print(f"Action dimension exceeds expected size ({action.shape[0]} > {NUM}); truncating automatically.")
                    action = action[:NUM]
                    reward_nega = -1.5
                elif action.shape[0] < NUM:
                    print(f"Action dimension is smaller than expected ({action.shape[0]} < {NUM}); padding with 0.")
                    action = np.pad(action, (0, NUM - action.shape[0]), 'constant', constant_values=0)
                    reward_nega = -5

            _, reward, _, _ = env.step(action, IsFIRST)
            IsFIRST = False

            # import ipdb;ipdb.set_trace() 

            avg_rewards = np.mean(hist_rewards, axis=0)     # shape: [NUM]
            avg_actions = np.mean(hist_actions, axis=0)     # shape: [NUM]
            top2_indices = np.argsort(avg_rewards)[-2:]
            bottom2_indices = np.argsort(avg_rewards)[:2]

            delta = 0.1
            bonus_count = 0


            for i in top2_indices:
                if action[i] > avg_actions[i] + delta:
                    if action[i] - avg_actions[i] <= 0.3:
                        reward[i] += 1.0 + abs(action[i] - avg_actions[i])
                        bonus_count += 1
                    else:
                        reward[i] += 0.5
                        # bonus_count += 1


            for i in bottom2_indices:
                if action[i] < avg_actions[i] - delta and action[i] != 0:
                    if avg_actions[i] - action[i] <= 0.3:
                        reward[i] += 1.0 + abs(action[i] - avg_actions[i])
                        bonus_count += 1
                    else:
                        reward[i] += 0.5
                        # bonus_count += 1

            if bonus_count > 0:
                log(f"✓ {reward} Bonus applied on {bonus_count} dimensions.")

            total_reward = reward.sum() + reward_nega


            action_sum = np.sum(action)
            if action_sum != NUM:
                penalty = 2 * abs(NUM - action_sum)
                total_reward -= penalty


            if np.allclose(action, action[0]) and action[0] != 0:
                # log(action, action[0])
                total_reward -= 1
                # log("PENALTY!")

            rewards.append(float(total_reward))

    return rewards


def selective_log_softmax(logits, input_ids):
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)


def compute_log_probs(model, input_ids, attention_mask, logits_to_keep):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]
    return selective_log_softmax(logits, input_ids)
    # Batch, T(prompt+generation), vocab


def create_completion_mask(completion_ids, eos_token_id):
    is_eos = completion_ids == eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
    mask_exists = is_eos.any(dim=1)
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
    sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
    return (sequence_indices <= eos_idx.unsqueeze(1)).int()


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
                          TOTAL=5, envs=None):
    # target:
    # prompts = [build_prompt(sample, NUM=NUM, TOTAL=TOTAL) for sample in batch_samples]
    prompts = [env.generate_prompt(sample, NUM=NUM, TOTAL=TOTAL) for env, sample in zip(envs, batch_samples)]
    with torch.no_grad():

        prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(model, tokenizer, prompts,
                                                                                        num_generations,
                                                                                        max_completion_length)

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)



        old_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
        ref_log_probs = compute_log_probs(ref_model, input_ids, attention_mask, logits_to_keep)
    formatted_completions = [[{'content': tokenizer.decode(ids, skip_special_tokens=True)}] for ids in completion_ids]

    repeated_samples = [s for s in batch_samples for _ in range(num_generations)]

    for idx, (prompt, generations) in enumerate(zip(prompts, formatted_completions)):
        log(f"\n========== Sample {idx + 1} ==========")
        log("Prompt:\n", prompt.strip())
        for g_idx, gen in enumerate(generations):
            log(f"\n-- Generation {g_idx + 1} --")
            log("Response:\n", gen['content'].strip())

            break

        break

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
    # print(f"Rewards: {rewards}")  # Debug rewards
    batch_size = rollout_data["batch_size"]
    num_generations = rollout_data["num_generations"]

    # print(f"TAG2:rewards shape: {rewards.shape}")

    rewards = rewards.view(batch_size, num_generations)
    # print(f"TAG3:rewards shape: {rewards.shape}")

    avg_reward = rewards.mean().item()


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


def optimize_model_memory(model):
    model.train()


    model.config.use_cache = False

    # First ensure inputs will require gradients

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    else:


        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Then enable gradient checkpointing

    model.gradient_checkpointing_enable()

    return model


def log_to_file_and_console(data, filename="local.txt"):

    message = ", ".join([f"{k}={v}" for k, v in data.items()])
    

    print(message)
    

    with open(filename, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def create_envs(samples, NUM):
    """
    Pre-create environment instances to avoid repeated construction.
    """
    envs = []
    for sample in samples:
        env = RewardFunctionEnv(n=NUM, current_func_type=sample['type'], current_func_params=sample['params'])
        envs.append(env)
    return envs


def train_with_grpo(model, tokenizer, train_data, num_iterations=1, num_steps=500, batch_size=4,
                    num_generations=4, max_completion_length=800, beta=0.1,
                    learning_rate=5e-6, mu=3, epsilon=0.2, reward_function=None, P_NUM=6, N_TOTAL=6, N_TRY=20,
                    device_ids=None):
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

        # Inner loop: your original training steps.

        for step in range(num_steps):
            batch_samples = random.sample(train_data, batch_size)
            envs = create_envs(batch_samples, P_NUM)

            rollout_list = []
            with torch.no_grad():
                for n_try in range(N_TRY):
                    rd = generate_rollout_data(model.module, ref_model, tokenizer, batch_samples, num_generations,
                                               max_completion_length, NUM=P_NUM, TOTAL=N_TOTAL, envs=envs)
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

                    log_to_file_and_console({
                        "loss": loss.item(),
                        "average_reward": avg_reward,
                        "iteration": iteration + 1,
                        "step": step + 1,
                        "grpo_iter": grpo_iter + 1,
                        "gradient_percentage": percent_with_grad,
                    })


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
    N_TRY = 1

    all_data = load_data(f"datasets/reward_dataset_dim{P_NUM}_sum{N_TOTAL}_steps{N_STEP}_numEnvs{N_ENVS}.csv",
                         group_size=N_STEP, n_try_max=N_TRY, P_NUM=P_NUM)

    set_random_seed(42)
    os.environ["WANDB_MODE"] = os.getenv("WANDB_MODE", "offline")
    os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", "dara")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"Using primary device: {device}")
    num_gpus = torch.cuda.device_count()
    log(f"Detected {num_gpus} GPUs")
    device_ids = list(range(1, num_gpus)) if num_gpus > 1 else None  # [0, 1, 2, ..., n-1]


    model_name = os.getenv("DARA_MODEL_PATH", "Qwen/Qwen2.5-3B-Instruct")
    log("Loading model...")



    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2", device_map="balanced_low_0")
    # model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2",device_map="auto")
    log("Loaded model")


    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    all_data = load_data(f"datasets/reward_dataset_dim{P_NUM}_sum{N_TOTAL}_steps{N_STEP}_numEnvs{N_ENVS}.csv",
                         group_size=N_STEP, n_try_max=N_TRY, P_NUM=P_NUM)
    # random.shuffle(all_data)
    size_of_eval_data = 3
    eval_data = all_data[:size_of_eval_data]
    train_data = all_data[size_of_eval_data:]

    env = RewardFunctionEnv(n=P_NUM)
    log("\nInitial model evaluation before finetuning:")
    #### pre_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device, NUM=P_NUM, TOTAL=N_TOTAL, num_try=N_TRY, max_new_tokens=1000)
    # print(f"Pre-GRPO Accuracy: {pre_grpo_accuracy:.2f}%")

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
                            device_ids=device_ids, **training_config)

    wandb.finish()
    log("Training completed and wandb run finished.")

    log("\nFinal model evaluation after GRPO RL fine-tuning:")
    post_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device, NUM=P_NUM, TOTAL=P_NUM, num_try=N_TRY,
                                        max_new_tokens=1000)
    log(f"Post-GRPO Accuracy: {post_grpo_accuracy:.2f}%")

    log("\nSaving GRPO fine-tuned model...")
    model.save_pretrained("grpo_finetuned_model")
    tokenizer.save_pretrained("grpo_finetuned_model")
