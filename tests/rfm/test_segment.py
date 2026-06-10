import pandas as pd
import pytest
from src.rfm.segment import compute_rfm, assign_season, flag_outlet


SNAPSHOT = pd.Timestamp("2026-06-10")


def make_transactions(**overrides) -> pd.DataFrame:
    base = {
        "customer_id": ["C1", "C2", "C3", "C4", "C5"],
        "order_date": [
            "2026-05-01",  # C1: recent
            "2026-03-01",  # C2: mid
            "2025-12-01",  # C3: old
            "2026-05-15",  # C4: recent, outlet buyer
            "2026-04-01",  # C5: recent
        ],
        "revenue": [200, 150, 80, 300, 120],
        "discount_rate": [0.10, 0.05, 0.20, 0.50, 0.15],
        "is_drop": [False, False, False, False, True],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_basic_rfm_runs():
    df = make_transactions()
    result = compute_rfm(df, snapshot_date=SNAPSHOT)
    assert set(["customer_id", "R", "F", "M", "segment"]).issubset(result.scores.columns)
    assert len(result.scores) == 5


def test_outlet_customer_flagged():
    # C4 has discount_rate=0.50 > threshold 0.40
    df = make_transactions()
    result = compute_rfm(df, snapshot_date=SNAPSHOT)
    c4 = result.scores[result.scores["customer_id"] == "C4"].iloc[0]
    assert c4["outlet_ratio"] > 0


def test_outlet_only_segment_assigned():
    # Build a customer with 100% outlet purchases
    df = pd.DataFrame({
        "customer_id": ["OUTLET"] * 3,
        "order_date": ["2026-05-01", "2026-04-01", "2026-03-01"],
        "revenue": [100, 100, 100],
        "discount_rate": [0.60, 0.55, 0.65],
        "is_drop": [False, False, False],
    })
    # Mix in one normal customer so quintile scoring has variance
    normal = make_transactions()
    combined = pd.concat([normal, df], ignore_index=True)
    result = compute_rfm(combined, snapshot_date=SNAPSHOT)
    outlet_row = result.scores[result.scores["customer_id"] == "OUTLET"].iloc[0]
    assert outlet_row["segment"] == "outlet_only"


def test_season_assignment():
    dates = pd.to_datetime(["2026-03-15", "2026-10-01", "2026-01-20"])
    seasons = [
        {"name": "SS", "months": [2, 3, 4, 5, 6, 7]},
        {"name": "AW", "months": [8, 9, 10, 11, 12, 1]},
    ]
    result = assign_season(pd.Series(dates), seasons)
    assert result.tolist() == ["SS", "AW", "AW"]


def test_net_revenue_strips_discount():
    df = make_transactions()
    result = compute_rfm(df, snapshot_date=SNAPSHOT)
    # C1: revenue=200, discount=0.10 → net=180
    c1 = result.scores[result.scores["customer_id"] == "C1"].iloc[0]
    assert abs(c1["monetary"] - 180.0) < 0.01


def test_segment_summary_non_empty():
    df = make_transactions()
    result = compute_rfm(df, snapshot_date=SNAPSHOT)
    assert len(result.segment_summary) > 0
    assert "total_revenue" in result.segment_summary.columns
