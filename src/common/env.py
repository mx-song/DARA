import gym
from gym import spaces
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import os


class RewardFunctionEnv(gym.Env):
    def __init__(self, n: int, max_tries=20, current_func_type=None, current_func_params=None, initial_action=None):
        super(RewardFunctionEnv, self).__init__()
        self.n = n
        self.max_tries = max_tries
        self.action_space = spaces.Box(low=0.0, high=np.inf, shape=(n,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(n,), dtype=np.float32)


        self.n_try = 0
        self.try_action = np.zeros((max_tries, n), dtype=np.float32)
        self.try_reward = np.zeros((max_tries, n), dtype=np.float32)


        if current_func_type is not None and current_func_params is not None:
            # assert len(current_func_type) == n, "Length of current_func_type must match n"
            # assert len(current_func_params) == n, "Length of current_func_params must match n"
            self.current_func_type = current_func_type
            self.current_func_params = current_func_params
            self.reward_functions = [
                self.generateRewardFunction(t, p) for t, p in zip(current_func_type, current_func_params)
            ]
        else:

            self.current_func_type = []
            self.current_func_params = []
            self.reward_functions = [self.generateRewardFunction(i) for i in range(self.n)]



        if initial_action is not None:
            action = np.array(initial_action, dtype=np.float32)
            if action.shape != (n,):
                if action.shape[0] > n:
                    action = action[:n]
                elif action.shape[0] < n:
                    action = np.pad(action, (0, n - action.shape[0]), 'constant', constant_values=0)

            obs, rewards, _, _ = self.step(action, IsFIRST=True)


    def generateRewardFunction(self, index, func_type=None, params=None):
        """
        Generate a randomized quadratic reward function with
        a left-side axis of symmetry and monotonic decay on the right.
        """
        if func_type is None:
            randomGenerate = True
        else:
            randomGenerate = False

        if params is None:

            zero_point = random.uniform(-2, -0.1)

            right_zero_point = random.uniform(0.3, 4.0)
            vertex_y = random.uniform(0.5, 1.0)

            a = vertex_y / ((right_zero_point - zero_point) ** 2)
            params = [zero_point, vertex_y, a]
        else:
            zero_point, vertex_y, a = params

        if randomGenerate:
            self.current_func_type.append(2)
            self.current_func_params.append(params)

        def func(x):
            y = -a * (x - zero_point) ** 2 + vertex_y
            return float(np.clip(y, 0.0, 1.0))

        return func

    def plot_reward_functions(self):
        os.makedirs("reward_png", exist_ok=True)
        matplotlib.use('Agg')
        x_vals = np.linspace(-2, 5, 300)
        plt.figure(figsize=(10, 6))
        for i, f in enumerate(self.reward_functions):
            y_vals = [f(x) for x in x_vals]
            plt.plot(x_vals, y_vals, label=f"Reward Function {i + 1}")
        plt.title("Reward Functions (Symmetric on Left, Monotonic on Right)")
        plt.xlabel("x (Action)")
        plt.ylabel("Reward")
        plt.ylim(0, 1.1)
        plt.xlim(-2, 5)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"reward_png/reward_functions_symmetric_left.png")
        plt.close()

    def step(self, action, IsFIRST, gen=False):
        """
        Execute one step in the environment.
        
        Args:
            action: Action vector of shape (n,)
            IsFIRST: Whether this is the first step (for recording)
            gen: If True, return individual_rewards instead of new_rewards (for data generation)
        """
        action = np.array(action)
        assert action.shape == (self.n,), "Action must be a vector of length n"

        individual_rewards = np.array(
            [f(x) for f, x in zip(self.reward_functions, action)],
            dtype=np.float32
        )
        mean_reward = individual_rewards.mean()
        diffs = np.abs(individual_rewards - mean_reward)

        eps = 1e-8
        # LLMA uses -0.1 * (diffs + eps), LLMB uses -(diffs + eps)
        new_rewards = -0.1 * (diffs + eps) if gen else -(diffs + eps)

        observation = action
        done = True

        info = {
            "individual_rewards": individual_rewards,
            "mean_reward": mean_reward,
            "diffs": diffs,
            "clipped_rewards": new_rewards
        }

        if IsFIRST:
            self.append_try_sample(action, individual_rewards)
        
        if gen:
            return observation, individual_rewards, done, info
        
        return observation, new_rewards, done, info

    def append_try_sample(self, new_action, new_reward):
        assert isinstance(new_action, (list, np.ndarray)) and len(new_action) == self.n
        assert isinstance(new_reward, (list, np.ndarray)) and len(new_reward) == self.n

        new_action = np.array(new_action, dtype=np.float32)
        new_reward = np.array(new_reward, dtype=np.float32)

        if self.n_try >= self.try_action.shape[0]:
            raise ValueError(f"Exceeded maximum try count: n_try_max = {self.try_action.shape[0]}")

        self.try_action[self.n_try, :] = new_action
        self.try_reward[self.n_try, :] = new_reward
        self.n_try += 1

    def generate_prompt(self, data_sample, NUM=10, TOTAL=200, MAX_CHANGE=5, max_steps=20):
        """
        Built-in prompt generator replacing external build_prompt.
        Behavior:
        - Skip insertion when all history is zero.
        - Insert one item when only one non-zero history exists.
        - Insert the latest entries when multiple non-zero histories exist.
        """

        non_zero_history = []
        for idx in reversed(range(len(self.try_action))):
            action_row = np.array(self.try_action[idx])
            if action_row.sum() > 1e-6:
                reward_row = self.try_reward[idx]
                non_zero_history.append((action_row.tolist(), reward_row))
            if len(non_zero_history) >= 3:
                break


        if len(non_zero_history) == 0:
            try_block = "(No previous attempts)"
        elif len(non_zero_history) == 1:
            action, reward = non_zero_history[0]
            bids = ", ".join(f"{x:.2f}" for x in action)
            rois = ", ".join(f"{r:.2f}" for r in reward)
            try_block = f'"your latest try", "allocation": [{bids}], "rewards": [{rois}]'
        else:
            try_lines = []
            for i, (action, reward) in enumerate(reversed(non_zero_history)):
                bids = ", ".join(f"{x:.2f}" for x in action)
                rois = ", ".join(f"{r:.2f}" for r in reward)
                try_lines.append(f'"attempt {i + 1}", "allocation": [{bids}], "rewards": [{rois}]')
            try_block = "\n".join(try_lines)


        prompt = f"""You are given a total budget of {TOTAL} to allocate across {NUM} time periods.

    Based on your last attempt, identify the time period with the **lowest reward**, 
    and reallocate 0.1-0.3 of its budget to the time period with the **highest reward**. 
    Keep the allocations for other periods unchanged.

    Last attempt:
    {try_block}

    Please output the new allocation:
    <reason>
    ...
    </reason>
    <answer>
    [y1, y2, ..., y{NUM}]
    </answer>

    Your Response:

    <reason>
    """

        return prompt

    def reset(self):
        self.current_func_type = []
        self.current_func_params = []
        self.reward_functions = [self.generateRewardFunction(i) for i in range(self.n)]
        self.n_try = 0
        self.try_action.fill(0)
        self.try_reward.fill(0)

        if __name__ == "__main__":
            self.plot_reward_functions()
        return np.zeros(self.n, dtype=np.float32)

    def render(self, mode="human"):
        print("Rendering current reward functions at x = 1.0:")
        for i, f in enumerate(self.reward_functions):
            print(f"  Function {i + 1}: f(1.0) = {f(1.0):.4f}")

    def get_try_history(self):
        return self.try_action[:self.n_try], self.try_reward[:self.n_try]



def generate_random_action_vector(dim, total_sum):
    """
    Generate a random integer vector:
    - Dimension is `dim`
    - Sum of elements equals `total_sum`
    - All elements are integers
    """

    vec = np.random.dirichlet(np.ones(dim)) * total_sum

    int_vec = np.floor(vec).astype(int)
    remainder = total_sum - int_vec.sum()


    for _ in range(remainder):
        idx = np.random.randint(0, dim)
        int_vec[idx] += 1

    return int_vec.astype(np.float32)

if __name__ == "__main__":
    env = RewardFunctionEnv(n=6, max_tries=20)
    NUM = TOTAL = 6
    max_steps = 3

    def print_test(title):
        print("\n" + "=" * 10 + f" {title} " + "=" * 10 + "\n")


    print_test("Test Case 1: All-Zero Action History")
    data_sample = {
        "action": env.try_action[:max_steps],
        "reward": env.try_reward[:max_steps]
    }
    prompt = env.generate_prompt(data_sample, NUM=NUM, TOTAL=TOTAL, max_steps=max_steps)
    print(prompt)


    print_test("Test Case 2: Only One Non-Zero Action")
    single_action = np.ones(NUM, dtype=np.float32)
    single_reward = np.array([f(x) for f, x in zip(env.reward_functions, single_action)], dtype=np.float32)
    env.append_try_sample(single_action, single_reward)

    data_sample = {
        "action": env.try_action[:max_steps],
        "reward": env.try_reward[:max_steps]
    }
    prompt = env.generate_prompt(data_sample, NUM=NUM, TOTAL=TOTAL, max_steps=max_steps)
    print(prompt)


    print_test("Test Case 3: Two Non-Zero Actions")
    second_action = np.array([0.2, 0.8, 1.0, 1.2, 1.1, 1.7], dtype=np.float32)
    second_reward = np.array([f(x) for f, x in zip(env.reward_functions, second_action)], dtype=np.float32)
    env.append_try_sample(second_action, second_reward)

    data_sample = {
        "action": env.try_action[:max_steps],
        "reward": env.try_reward[:max_steps]
    }
    prompt = env.generate_prompt(data_sample, NUM=NUM, TOTAL=TOTAL, max_steps=max_steps)
    print(prompt)
