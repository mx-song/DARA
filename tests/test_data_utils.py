import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common.utils import load_data


def _columns_for_dim(dim: int):
    cols = []
    for j in range(dim):
        cols.extend(
            [
                f"action_{j + 1}",
                f"reward_{j + 1}",
                f"func_type_{j + 1}",
                f"param1_{j + 1}",
                f"param2_{j + 1}",
                f"param3_{j + 1}",
            ]
        )
    return cols


def _write_dataset(path: Path, dim: int, group_size: int, groups: int):
    columns = _columns_for_dim(dim)
    rows = []
    for g in range(groups):
        for step in range(group_size):
            row = []
            for d in range(dim):
                action = float(g + step + d + 1)
                reward = float((g + 1) * (d + 1) * 0.1)
                func_type = 2
                p1 = -1.0 - d * 0.1
                p2 = 0.5 + d * 0.05
                p3 = 0.2 + d * 0.02
                row.extend([action, reward, func_type, p1, p2, p3])
            rows.append(row)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def test_load_data_basic_shape(tmp_path):
    csv_path = tmp_path / "toy.csv"
    _write_dataset(csv_path, dim=6, group_size=3, groups=2)

    data = load_data(str(csv_path), group_size=3, n_try_max=7, P_NUM=6)
    assert isinstance(data, list)
    assert len(data) == 2

    sample = data[0]
    assert sample["action"].shape == (3, 6)
    assert sample["reward"].shape == (3, 6)
    assert len(sample["type"]) == 6
    assert sample["params"].shape == (6, 3)
    assert sample["try_action"].shape == (7, 6)
    assert sample["try_reward"].shape == (7, 6)
    assert sample["n_try"] == 0


@pytest.mark.parametrize("group_size", [1, 2, 5])
def test_load_data_group_partition(tmp_path, group_size):
    csv_path = tmp_path / f"group_{group_size}.csv"
    groups = 4
    _write_dataset(csv_path, dim=6, group_size=group_size, groups=groups)
    data = load_data(str(csv_path), group_size=group_size, n_try_max=4, P_NUM=6)
    assert len(data) == groups
    for sample in data:
        assert sample["action"].shape == (group_size, 6)
        assert sample["reward"].shape == (group_size, 6)


def test_load_data_asserts_when_rows_not_divisible(tmp_path):
    csv_path = tmp_path / "bad.csv"
    columns = _columns_for_dim(6)
    rows = [[0.0 for _ in range(len(columns))] for _ in range(5)]
    pd.DataFrame(rows, columns=columns).to_csv(csv_path, index=False)
    with pytest.raises(AssertionError):
        load_data(str(csv_path), group_size=4, n_try_max=3, P_NUM=6)


def test_load_data_params_extracted_from_first_row_per_group(tmp_path):
    csv_path = tmp_path / "params.csv"
    columns = _columns_for_dim(6)
    rows = []
    for step in range(3):
        row = []
        for d in range(6):
            row.extend(
                [
                    float(step + d),
                    float(step + d) * 0.1,
                    2,
                    -10.0 + d,  # fixed per dimension
                    0.1 + d,
                    0.2 + d,
                ]
            )
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    for step in range(1, 3):
        for d in range(6):
            df.loc[step, f"param1_{d+1}"] = 999.0 + step
            df.loc[step, f"param2_{d+1}"] = 888.0 + step
            df.loc[step, f"param3_{d+1}"] = 777.0 + step
    df.to_csv(csv_path, index=False)

    data = load_data(str(csv_path), group_size=3, n_try_max=2, P_NUM=6)
    params = data[0]["params"]
    for d in range(6):
        assert params[d, 0] == pytest.approx(-10.0 + d)
        assert params[d, 1] == pytest.approx(0.1 + d)
        assert params[d, 2] == pytest.approx(0.2 + d)


@pytest.mark.parametrize("n_try_max", [1, 3, 10, 20])
def test_load_data_try_buffers_shape(tmp_path, n_try_max):
    csv_path = tmp_path / f"try_{n_try_max}.csv"
    _write_dataset(csv_path, dim=6, group_size=2, groups=1)
    data = load_data(str(csv_path), group_size=2, n_try_max=n_try_max, P_NUM=6)
    sample = data[0]
    assert sample["try_action"].shape == (n_try_max, 6)
    assert sample["try_reward"].shape == (n_try_max, 6)
    assert sample["try_action"].dtype == np.float32
    assert sample["try_reward"].dtype == np.float32


def test_load_data_dtype_guarantees(tmp_path):
    csv_path = tmp_path / "dtype.csv"
    _write_dataset(csv_path, dim=6, group_size=2, groups=1)
    data = load_data(str(csv_path), group_size=2, n_try_max=3, P_NUM=6)
    sample = data[0]
    assert sample["action"].dtype == np.float32
    assert sample["reward"].dtype == np.float32
    assert sample["params"].dtype == np.float32


def test_load_data_values_match_csv(tmp_path):
    csv_path = tmp_path / "values.csv"
    _write_dataset(csv_path, dim=6, group_size=2, groups=1)
    data = load_data(str(csv_path), group_size=2, n_try_max=3, P_NUM=6)
    sample = data[0]

    df = pd.read_csv(csv_path)
    expected_action = df[[f"action_{j + 1}" for j in range(6)]].to_numpy(dtype=np.float32)
    expected_reward = df[[f"reward_{j + 1}" for j in range(6)]].to_numpy(dtype=np.float32)

    np.testing.assert_allclose(sample["action"], expected_action, rtol=0, atol=0)
    np.testing.assert_allclose(sample["reward"], expected_reward, rtol=0, atol=0)


def test_load_data_multiple_groups_have_independent_buffers(tmp_path):
    csv_path = tmp_path / "independent.csv"
    _write_dataset(csv_path, dim=6, group_size=2, groups=2)
    data = load_data(str(csv_path), group_size=2, n_try_max=4, P_NUM=6)
    assert len(data) == 2

    data[0]["try_action"][0, 0] = 99.0
    assert data[1]["try_action"][0, 0] == 0.0


def test_load_data_handles_float_and_int_mix(tmp_path):
    csv_path = tmp_path / "mixed.csv"
    columns = _columns_for_dim(6)
    rows = []
    for i in range(4):
        row = []
        for d in range(6):
            row.extend([i + d, (i + d) * 0.5, 2, -1, 1, 1 / (d + 1)])
        rows.append(row)
    pd.DataFrame(rows, columns=columns).to_csv(csv_path, index=False)

    data = load_data(str(csv_path), group_size=2, n_try_max=3, P_NUM=6)
    assert len(data) == 2
    for sample in data:
        assert sample["action"].dtype == np.float32
        assert sample["reward"].dtype == np.float32
        assert sample["params"].dtype == np.float32


def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("not_exists_xyz_123.csv", group_size=2, n_try_max=3, P_NUM=6)


def test_generated_csv_header_consistency(tmp_path):
    csv_path = tmp_path / "header.csv"
    _write_dataset(csv_path, dim=6, group_size=2, groups=1)
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

    expected = _columns_for_dim(6)
    assert header == expected


def test_load_data_action_reward_nonnegative_when_input_nonnegative(tmp_path):
    csv_path = tmp_path / "nonneg.csv"
    _write_dataset(csv_path, dim=6, group_size=2, groups=2)
    data = load_data(str(csv_path), group_size=2, n_try_max=3, P_NUM=6)
    for sample in data:
        assert np.all(sample["action"] >= 0.0)
        assert np.all(sample["reward"] >= 0.0)


def test_load_data_type_field_length(tmp_path):
    csv_path = tmp_path / "types.csv"
    _write_dataset(csv_path, dim=6, group_size=2, groups=3)
    data = load_data(str(csv_path), group_size=2, n_try_max=3, P_NUM=6)
    for sample in data:
        assert isinstance(sample["type"], list)
        assert len(sample["type"]) == 6
        for t in sample["type"]:
            assert int(t) == 2


@pytest.mark.parametrize("p_num", [3, 4, 5, 6])
def test_load_data_supports_custom_p_num(tmp_path, p_num):
    csv_path = tmp_path / f"pnum_{p_num}.csv"
    _write_dataset(csv_path, dim=p_num, group_size=2, groups=2)
    data = load_data(str(csv_path), group_size=2, n_try_max=3, P_NUM=p_num)
    assert len(data) == 2
    for sample in data:
        assert sample["action"].shape == (2, p_num)
        assert sample["reward"].shape == (2, p_num)
        assert len(sample["type"]) == p_num
        assert sample["params"].shape == (p_num, 3)


def test_load_data_try_buffer_is_zero_initialized(tmp_path):
    csv_path = tmp_path / "zero_init.csv"
    _write_dataset(csv_path, dim=6, group_size=2, groups=1)
    sample = load_data(str(csv_path), group_size=2, n_try_max=5, P_NUM=6)[0]
    assert np.count_nonzero(sample["try_action"]) == 0
    assert np.count_nonzero(sample["try_reward"]) == 0
    assert sample["n_try"] == 0


def test_load_data_large_small_case(tmp_path):
    csv_path = tmp_path / "large_small.csv"
    _write_dataset(csv_path, dim=6, group_size=1, groups=50)
    data = load_data(str(csv_path), group_size=1, n_try_max=2, P_NUM=6)
    assert len(data) == 50
    for sample in data:
        assert sample["action"].shape == (1, 6)
        assert sample["reward"].shape == (1, 6)
