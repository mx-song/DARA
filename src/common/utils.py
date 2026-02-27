import pandas as pd
import numpy as np
import os


def load_data(csv_path, group_size=20, n_try_max=20, P_NUM=6):
    """
    Load data from a CSV file and split rows into grouped samples.

    Args:
    - csv_path: CSV file path.
    - group_size: Number of rows per group (default: 20).
    - n_try_max: Max number of try-history rows (default: 20).

    Returns:
    - data: List of grouped sample dicts, each containing:
      - "action": Action matrix, shape (group_size, P_NUM)
      - "reward": Reward matrix, shape (group_size, P_NUM)
      - "type": Reward function type list
      - "params": Reward function parameters, shape (P_NUM, 3)
      - "try_action": Zero-initialized try-action buffer
      - "try_reward": Zero-initialized try-reward buffer
      - "n_try": Current try count initialized to 0
    """
    df = pd.read_csv(csv_path)
    data = []


    assert len(df) % group_size == 0, "CSV row count must be a multiple of group_size."

    for i in range(0, len(df), group_size):
        group = df.iloc[i:i + group_size]

        # import ipdb;ipdb.set_trace()


        actions = group[[f"action_{j + 1}" for j in range(P_NUM)]].to_numpy(dtype=np.float32)  # (20, 6)
        rewards = group[[f"reward_{j + 1}" for j in range(P_NUM)]].to_numpy(dtype=np.float32)  # (20, 6)


        func_types = group.loc[group.index[0], [f"func_type_{j + 1}" for j in range(P_NUM)]].to_list()
        params = []
        for j in range(P_NUM):
            params.append([
                group.iloc[0][f"param1_{j + 1}"],
                group.iloc[0][f"param2_{j + 1}"],
                group.iloc[0][f"param3_{j + 1}"]
            ])
        params = np.array(params, dtype=np.float32)


        data.append({
            "action": actions,  # shape (20, 6)
            "reward": rewards,  # shape (20, 6)
            "type": func_types,  # list of int
            "params": params,  # shape (6, 3)
            "try_action": np.zeros((n_try_max, 6), dtype=np.float32),
            "try_reward": np.zeros((n_try_max, 6), dtype=np.float32),
            "n_try": 0
        })

    return data
