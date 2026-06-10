import numpy as np
import pandas as pd
import pytest
from src.profit_map.builder import (
    assign_profitability_band,
    assign_quadrant,
    compute_profit_score,
    build_profit_map,
    top_targets,
    QUADRANT_MAP,
)


def make_ltv(customers):
    rng = np.random.default_rng(1)
    tiers = ["platinum", "gold", "silver", "bronze"]
    return pd.DataFrame({
        "customer_id": customers,
        "predicted_ltv_12m": rng.uniform(50, 1000, len(customers)),
        "ltv_tier": [tiers[i % 4] for i in range(len(customers))],
    })


def make_cts(customers):
    rng = np.random.default_rng(2)
    revenue = rng.uniform(100, 600, len(customers))
    margin = revenue * rng.uniform(-0.1, 0.6, len(customers))
    cts = revenue - margin
    return pd.DataFrame({
        "customer_id": customers,
        "gross_revenue": revenue,
        "gross_margin": margin,
        "total_cost_to_serve": cts,
        "cts_pct_revenue": (cts / revenue * 100).round(1),
    })


def make_rfm(customers):
    rng = np.random.default_rng(3)
    segments = ["champions", "loyal", "at_risk", "lost"]
    return pd.DataFrame({
        "customer_id": customers,
        "rfm_score": rng.uniform(1, 5, len(customers)),
        "segment": [segments[i % 4] for i in range(len(customers))],
        "R": rng.integers(1, 6, len(customers)),
        "F": rng.integers(1, 6, len(customers)),
        "M": rng.integers(1, 6, len(customers)),
    })


CUSTOMERS = [f"C{i:03d}" for i in range(20)]


def test_profitability_band_negative():
    margin = pd.Series([-10.0, 50.0, 100.0, 200.0])
    bands = assign_profitability_band(margin)
    assert bands.iloc[0] == "negative"


def test_profitability_band_covers_all():
    margin = pd.Series(np.linspace(-50, 300, 40))
    bands = assign_profitability_band(margin)
    assert set(bands.unique()).issubset({"negative", "low", "medium", "high"})


def test_quadrant_protect_high_ltv_high_profit():
    ltv = pd.Series(["platinum", "gold"])
    profit = pd.Series(["high", "medium"])
    q = assign_quadrant(ltv, profit)
    assert (q == "PROTECT").all()


def test_quadrant_reduce_low_ltv_low_profit():
    ltv = pd.Series(["bronze", "silver"])
    profit = pd.Series(["low", "negative"])
    q = assign_quadrant(ltv, profit)
    assert (q == "REDUCE").all()


def test_quadrant_map_complete():
    tiers = ["platinum", "gold", "silver", "bronze"]
    bands = ["high", "medium", "low", "negative"]
    for t in tiers:
        for b in bands:
            assert (t, b) in QUADRANT_MAP, f"Missing quadrant for ({t}, {b})"


def test_profit_score_range():
    idx = pd.RangeIndex(10)
    ltv = pd.Series(np.linspace(100, 1000, 10), index=idx)
    margin = pd.Series(np.linspace(-50, 500, 10), index=idx)
    rfm = pd.Series(np.linspace(1, 5, 10), index=idx)
    score = compute_profit_score(ltv, margin, rfm)
    assert score.between(0, 100).all()


def test_build_profit_map_columns():
    ltv = make_ltv(CUSTOMERS)
    cts = make_cts(CUSTOMERS)
    rfm = make_rfm(CUSTOMERS)
    result = build_profit_map(ltv, cts, rfm)
    required = {"customer_id", "quadrant", "profit_score", "profitability_band",
                "predicted_ltv_12m", "gross_margin", "segment"}
    assert required.issubset(set(result.customer_map.columns))


def test_build_profit_map_row_count():
    ltv = make_ltv(CUSTOMERS)
    cts = make_cts(CUSTOMERS)
    rfm = make_rfm(CUSTOMERS)
    result = build_profit_map(ltv, cts, rfm)
    assert len(result.customer_map) == len(CUSTOMERS)


def test_quadrant_summary_has_action():
    ltv = make_ltv(CUSTOMERS)
    cts = make_cts(CUSTOMERS)
    rfm = make_rfm(CUSTOMERS)
    result = build_profit_map(ltv, cts, rfm)
    assert "action" in result.quadrant_summary.columns


def test_top_targets_returns_correct_quadrant():
    ltv = make_ltv(CUSTOMERS)
    cts = make_cts(CUSTOMERS)
    rfm = make_rfm(CUSTOMERS)
    result = build_profit_map(ltv, cts, rfm)
    for quadrant in ["PROTECT", "INVEST", "HARVEST", "REDUCE"]:
        targets = top_targets(result, quadrant=quadrant, n=5)
        if len(targets) > 0:
            assert (targets["quadrant"] == quadrant).all()


def test_profit_map_sorted_by_score():
    ltv = make_ltv(CUSTOMERS)
    cts = make_cts(CUSTOMERS)
    rfm = make_rfm(CUSTOMERS)
    result = build_profit_map(ltv, cts, rfm)
    scores = result.customer_map["profit_score"].values
    assert (scores[:-1] >= scores[1:]).all()
