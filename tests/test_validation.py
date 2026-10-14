"""Tests for src/validation.py: the fixed train/test split."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.validation import (fixed_split_forecast, monthly_mae, score_table,
                            split_train_test)

HORIZON = 72
TEST_START = "2025-02-01"


def make_frame(hours=24 * 120, seed=0):
    """Made-up table: three inputs and a price that depends on them (last column)."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01 01:00", periods=hours, freq="h")
    df = pd.DataFrame(rng.normal(size=(hours, 3)), index=index, columns=["a", "b", "c"])
    df["price_mwh"] = 50 + 10 * df["a"] - 5 * df["b"] + rng.normal(scale=2, size=hours)
    return df


def test_training_and_test_periods_do_not_overlap_and_keep_the_gap():
    df = make_frame()
    train, test = split_train_test(df, TEST_START, HORIZON)
    assert train.index.max() <= pd.Timestamp(TEST_START) - pd.Timedelta(hours=HORIZON)
    assert test.index.min() >= pd.Timestamp(TEST_START)
    assert len(train.index.intersection(test.index)) == 0
    assert len(train) + len(test) < len(df)      # the gap hours are used by neither


def test_forecasts_do_not_depend_on_test_period_prices():
    """Change every price in the test period. The forecasts must not move,
    because the models were fitted only on the past."""
    df = make_frame()
    first = fixed_split_forecast(df, HORIZON, TEST_START)
    changed = df.copy()
    changed.loc[changed.index >= TEST_START, "price_mwh"] = 1e6
    second = fixed_split_forecast(changed, HORIZON, TEST_START)
    models = [c for c in first.columns if c != "actual"]
    pd.testing.assert_frame_equal(first[models], second[models])


def test_output_covers_the_whole_test_period_with_every_model():
    df = make_frame()
    out = fixed_split_forecast(df, HORIZON, TEST_START)
    assert list(out.columns) == ["actual", "Linear regression", "Decision tree", "SVR"]
    assert out.index.equals(df.index[df.index >= TEST_START])
    assert not out.isna().any().any()


def test_a_good_model_is_close_on_a_learnable_problem():
    out = fixed_split_forecast(make_frame(), HORIZON, TEST_START)
    scores = score_table(out, ["Linear regression"])
    assert scores.loc["Linear regression", "MAE"] < 4         # the noise level is about 2


def test_score_table_hand_calculated():
    frame = pd.DataFrame({"actual": [1.0, 2.0, 3.0], "f": [2.0, 2.0, 5.0]},
                         index=pd.date_range("2025-01-01", periods=3, freq="h"))
    scores = score_table(frame, ["f"])
    assert scores.loc["f", "MAE"] == pytest.approx(1.0)
    assert scores.loc["f", "RMSE"] == pytest.approx(np.sqrt(5 / 3))
    assert scores.loc["f", "Bias"] == pytest.approx(1.0)


def test_monthly_mae_splits_by_calendar_month():
    index = pd.to_datetime(["2025-01-31 23:00", "2025-02-01 00:00", "2025-02-01 01:00"])
    frame = pd.DataFrame({"actual": [0.0, 0.0, 0.0], "f": [1.0, 2.0, 4.0]}, index=index)
    result = monthly_mae(frame, "f")
    assert result.loc[pd.Period("2025-01")] == pytest.approx(1.0)
    assert result.loc[pd.Period("2025-02")] == pytest.approx(3.0)
