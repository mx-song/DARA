import pandas as pd
import numpy as np
import csv
import os
from tqdm import tqdm
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.common.env import RewardFunctionEnv


n = 6
action_total_sum = 6
max_steps = 3
num_envs = 1000

output_dir = "datasets"
os.makedirs(output_dir, exist_ok=True)
filename = f"reward_dataset_dim{n}_sum{action_total_sum}_steps{max_steps}_numEnvs{num_envs}.csv"
filepath = os.path.join(output_dir, filename)
print(f"File path: {filepath}")


header = []
for j in range(n):
    header.extend([
        f"action_{j + 1}",
        f"reward_{j + 1}",
        f"func_type_{j + 1}",
        f"param1_{j + 1}",
        f"param2_{j + 1}",
        f"param3_{j + 1}"
    ])

with open(filepath, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)



def generate_random_action_vector(dim, total_sum, low=0.7, high=1.3, max_trials=1000):
    """
    Generate an action vector that satisfies:
    - Every element is in [low, high]
    - The element sum equals total_sum
    Uses rejection sampling with normalization-based scaling.
    """
    for _ in range(max_trials):

        base = np.random.rand(dim)
        base = base / base.sum()
        scaled = base * total_sum


        if np.all((scaled >= low) & (scaled <= high)):
            return scaled.astype(np.float32)

    raise ValueError(f"Failed to generate a valid action vector within {max_trials} attempts.")



print("Generated data...")
for _ in tqdm(range(num_envs)):
    env = RewardFunctionEnv(n=n)
    env.reset()

    rows = []


    for _ in range(max_steps):
        action = generate_random_action_vector(n, action_total_sum, low=0.7, high=1.3)
        obs, rewards, _, _ = env.step(action, IsFIRST=True, gen=True)

        row = []

        for i in range(n):
            row.extend([
                action[i],                 # action_i
                rewards[i],                # reward_i
                env.current_func_type[i],  # func_type_i
                env.current_func_params[i][0],  # param1_i
                env.current_func_params[i][1],  # param2_i
                env.current_func_params[i][2]   # param3_i
            ])


        rows.append(row)


    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

print(f"Data has been saved to {filepath}")
