import numpy as np
import pytest

from src.common.env import RewardFunctionEnv, generate_random_action_vector


def make_fixed_env(n=3, max_tries=5):
    func_types = [2 for _ in range(n)]
    params = [
        [-1.0, 1.0, 0.25],
        [-1.2, 0.9, 0.2],
        [-0.8, 0.8, 0.3],
    ][:n]
    return RewardFunctionEnv(
        n=n,
        max_tries=max_tries,
        current_func_type=func_types,
        current_func_params=params,
    )


def test_env_init_shapes():
    env = RewardFunctionEnv(n=6, max_tries=10)
    assert env.n == 6
    assert env.max_tries == 10
    assert env.try_action.shape == (10, 6)
    assert env.try_reward.shape == (10, 6)
    assert env.n_try == 0
    assert len(env.reward_functions) == 6
    assert len(env.current_func_type) == 6
    assert len(env.current_func_params) == 6


@pytest.mark.parametrize(
    "action",
    [
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
        np.array([0.5, 0.6, 0.7], dtype=np.float32),
        np.array([2.0, 0.0, 1.0], dtype=np.float32),
    ],
)
def test_step_returns_expected_shapes(action):
    env = make_fixed_env(n=3)
    obs, rewards, done, info = env.step(action, IsFIRST=True, gen=False)
    assert done is True
    assert obs.shape == (3,)
    assert rewards.shape == (3,)
    assert isinstance(info, dict)
    assert info["individual_rewards"].shape == (3,)
    assert isinstance(info["mean_reward"], np.float32) or np.isscalar(info["mean_reward"])
    assert info["diffs"].shape == (3,)
    assert info["clipped_rewards"].shape == (3,)
    assert env.n_try == 1


def test_step_gen_mode_returns_individual_rewards():
    env = make_fixed_env(n=3)
    action = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    _, rewards_gen_false, _, info_false = env.step(action, IsFIRST=True, gen=False)
    _, rewards_gen_true, _, info_true = env.step(action, IsFIRST=False, gen=True)

    np.testing.assert_allclose(rewards_gen_true, info_true["individual_rewards"], atol=1e-6, rtol=1e-6)
    assert not np.allclose(rewards_gen_false, rewards_gen_true)


def test_step_invalid_action_shape_raises():
    env = make_fixed_env(n=3)
    with pytest.raises(AssertionError):
        env.step(np.array([1.0, 1.0], dtype=np.float32), IsFIRST=True)
    with pytest.raises(AssertionError):
        env.step(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32), IsFIRST=True)


def test_append_try_sample_increments_counter():
    env = make_fixed_env(n=3, max_tries=3)
    action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    reward = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    env.append_try_sample(action, reward)
    assert env.n_try == 1
    np.testing.assert_allclose(env.try_action[0], action)
    np.testing.assert_allclose(env.try_reward[0], reward)

    env.append_try_sample(action * 2, reward * 2)
    assert env.n_try == 2


def test_append_try_sample_overflow_raises():
    env = make_fixed_env(n=3, max_tries=1)
    action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    reward = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    env.append_try_sample(action, reward)
    with pytest.raises(ValueError):
        env.append_try_sample(action, reward)


@pytest.mark.parametrize(
    "initial_action,expected_n_try",
    [
        ([1.0, 1.0, 1.0], 1),
        ([1.0, 1.0], 1),
        ([1.0, 1.0, 1.0, 1.0], 1),
    ],
)
def test_initial_action_is_normalized_and_recorded(initial_action, expected_n_try):
    env = RewardFunctionEnv(n=3, max_tries=5, initial_action=initial_action)
    assert env.n_try == expected_n_try
    assert env.try_action[0].shape == (3,)


def test_generate_prompt_no_history():
    env = make_fixed_env(n=3, max_tries=5)
    prompt = env.generate_prompt(data_sample={}, NUM=3, TOTAL=6)
    assert "(No previous attempts)" in prompt
    assert "<answer>" in prompt
    assert "y3" in prompt


def test_generate_prompt_with_one_history():
    env = make_fixed_env(n=3, max_tries=5)
    action = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    reward = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    env.append_try_sample(action, reward)

    prompt = env.generate_prompt(data_sample={}, NUM=3, TOTAL=6)
    assert "your latest try" in prompt
    assert "1.00, 2.00, 3.00" in prompt


def test_generate_prompt_with_multiple_history():
    env = make_fixed_env(n=3, max_tries=5)
    env.append_try_sample(np.array([1.0, 1.0, 1.0], dtype=np.float32), np.array([0.1, 0.2, 0.3], dtype=np.float32))
    env.append_try_sample(np.array([2.0, 1.0, 0.5], dtype=np.float32), np.array([0.3, 0.4, 0.5], dtype=np.float32))
    env.append_try_sample(np.array([1.5, 2.0, 0.5], dtype=np.float32), np.array([0.2, 0.8, 0.6], dtype=np.float32))

    prompt = env.generate_prompt(data_sample={}, NUM=3, TOTAL=6)
    assert "attempt 1" in prompt
    assert "attempt 2" in prompt
    assert "attempt 3" in prompt


def test_reset_clears_history():
    env = make_fixed_env(n=3, max_tries=5)
    env.append_try_sample(np.array([1.0, 1.0, 1.0], dtype=np.float32), np.array([0.1, 0.2, 0.3], dtype=np.float32))
    env.append_try_sample(np.array([2.0, 1.0, 0.0], dtype=np.float32), np.array([0.3, 0.4, 0.1], dtype=np.float32))
    assert env.n_try == 2

    obs = env.reset()
    assert obs.shape == (3,)
    assert env.n_try == 0
    assert np.allclose(env.try_action, 0)
    assert np.allclose(env.try_reward, 0)


def test_get_try_history_returns_prefix():
    env = make_fixed_env(n=3, max_tries=5)
    env.append_try_sample(np.array([1.0, 1.0, 1.0], dtype=np.float32), np.array([0.1, 0.2, 0.3], dtype=np.float32))
    env.append_try_sample(np.array([2.0, 2.0, 2.0], dtype=np.float32), np.array([0.3, 0.3, 0.3], dtype=np.float32))

    try_a, try_r = env.get_try_history()
    assert try_a.shape == (2, 3)
    assert try_r.shape == (2, 3)
    np.testing.assert_allclose(try_a[0], np.array([1.0, 1.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(try_a[1], np.array([2.0, 2.0, 2.0], dtype=np.float32))


@pytest.mark.parametrize("dim,total_sum", [(3, 6), (6, 6), (10, 20), (12, 30)])
def test_generate_random_action_vector_properties(dim, total_sum):
    for _ in range(50):
        vec = generate_random_action_vector(dim, total_sum)
        assert vec.shape == (dim,)
        assert vec.dtype == np.float32
        assert np.all(vec >= 0)
        assert int(vec.sum()) == total_sum
        assert np.all(np.mod(vec, 1) == 0)


def test_generate_reward_function_value_range():
    env = make_fixed_env(n=3)
    f = env.reward_functions[0]
    xs = np.linspace(-10, 10, 200)
    ys = np.array([f(float(x)) for x in xs], dtype=np.float32)
    assert np.all(ys >= 0.0)
    assert np.all(ys <= 1.0)


def test_step_reward_sign_and_scale():
    env = make_fixed_env(n=3)
    action = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    _, reward_normal, _, _ = env.step(action, IsFIRST=False, gen=False)
    _, reward_gen, _, _ = env.step(action, IsFIRST=False, gen=True)

    assert np.all(reward_normal <= 1e-6)
    assert np.all(reward_gen >= 0.0)


@pytest.mark.parametrize("n", [2, 3, 6])
def test_env_supports_different_dimensions(n):
    env = RewardFunctionEnv(n=n, max_tries=4)
    action = np.ones(n, dtype=np.float32)
    obs, rewards, done, info = env.step(action, IsFIRST=True)

    assert obs.shape == (n,)
    assert rewards.shape == (n,)
    assert done is True
    assert info["individual_rewards"].shape == (n,)
    assert env.n_try == 1


def test_prompt_budget_and_dimension_reflect_inputs():
    env = make_fixed_env(n=3)
    prompt = env.generate_prompt(data_sample={}, NUM=7, TOTAL=42)
    assert "42" in prompt
    assert "7 time periods" in prompt
    assert "y7" in prompt


def test_render_does_not_raise():
    env = make_fixed_env(n=3)
    env.render(mode="human")


def test_plot_reward_functions_creates_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env = make_fixed_env(n=3)
    env.plot_reward_functions()
    out_path = tmp_path / "reward_png" / "reward_functions_symmetric_left.png"
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_step_records_when_isfirst_true_only():
    env = make_fixed_env(n=3, max_tries=5)
    action = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    env.step(action, IsFIRST=False)
    assert env.n_try == 0

    env.step(action, IsFIRST=True)
    assert env.n_try == 1

    env.step(action, IsFIRST=False)
    assert env.n_try == 1


@pytest.mark.parametrize("seed", [1, 2, 3, 42, 2026])
def test_generate_random_action_vector_seed_stability(seed):
    np.random.seed(seed)
    first = generate_random_action_vector(6, 6)
    np.random.seed(seed)
    second = generate_random_action_vector(6, 6)
    np.testing.assert_allclose(first, second, rtol=0, atol=0)
