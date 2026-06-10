import pandas as pd
import numpy as np
import pytest
from src.rfm.segment import compute_rfm
from src.cost_to_serve.calculator import (
    compute_cost_to_serve,
    fulfilment_cost,
    return_cost,
    service_cost,
    discount_cost,
    acquisition_cost,
)

SNAPSHOT = pd.Timestamp("2026-06-10")


def make_transactions(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    customers = [f"C{i:02d}" for i in range(10)]
    rows = []
    for cid in customers:
        for _ in range(rng.integers(1, 5)):
            days_ago = int(rng.integers(1, 300))
            rows.append({
                "customer_id": cid,
                "order_date": SNAPSHOT - pd.Timedelta(days=days_ago),
                "revenue": float(rng.integers(60, 400)),
                "discount_rate": float(rng.uniform(0, 0.65)),
                "is_drop": bool(rng.integers(0, 2)),
                "category": rng.choice(["tops", "dresses", "accessories", "outerwear"]),
                "quantity": int(rng.integers(1, 4)),
                "channel": rng.choice(["paid_social", "email", "organic"]),
            })
    return pd.DataFrame(rows)


def test_fulfilment_cost_outlet_cheaper():
    df = pd.DataFrame({
        "customer_id": ["A", "B"],
        "is_outlet": [False, True],
        "revenue": [100, 100],
    })
    cost = fulfilment_cost(df, cost_per_order=10.0, outlet_discount=0.30)
    assert cost["A"] == pytest.approx(10.0)
    assert cost["B"] == pytest.approx(7.0)


def test_return_cost_category_fallback():
    df = pd.DataFrame({
        "customer_id": ["A", "A", "B"],
        "revenue": [100, 100, 100],
        "category": ["dresses", "accessories", "tops"],
        "quantity": [1, 1, 1],
    })
    cost = return_cost(df, base_cost_per_return=10.0)
    # dresses (0.38) > accessories (0.10) → A has higher return cost per order
    assert cost["A"] > cost["B"]


def test_discount_cost_proportional():
    df = pd.DataFrame({
        "customer_id": ["A", "B"],
        "revenue": [200.0, 200.0],
        "discount_rate": [0.20, 0.50],
    })
    cost = discount_cost(df)
    assert cost["A"] == pytest.approx(40.0)
    assert cost["B"] == pytest.approx(100.0)


def test_acquisition_cost_amortises_to_zero():
    # Customer acquired >12 months ago should have fully amortised CAC
    df = pd.DataFrame({
        "customer_id": ["OLD", "NEW"],
        "order_date": [
            SNAPSHOT - pd.Timedelta(days=400),  # >12 months ago
            SNAPSHOT - pd.Timedelta(days=30),   # 1 month ago
        ],
        "revenue": [100, 100],
    })
    cost = acquisition_cost(df, cac_by_channel={}, default_cac=36.0,
                            snapshot_date=SNAPSHOT, amortisation_months=12)
    assert cost["OLD"] == pytest.approx(0.0)
    assert cost["NEW"] > 0


def test_service_cost_drops_more_expensive():
    df = pd.DataFrame({
        "customer_id": ["A", "A", "B", "B"],
        "is_drop": [True, True, False, False],
        "revenue": [100, 100, 100, 100],
    })
    cost = service_cost(df, service_contacts=None, cost_per_contact=6.50, drop_spike_factor=2.5)
    assert cost["A"] > cost["B"]


def test_compute_cost_to_serve_full_pipeline():
    txn = make_transactions()
    rfm = compute_rfm(txn, snapshot_date=SNAPSHOT).scores
    result = compute_cost_to_serve(txn, rfm, SNAPSHOT)

    expected_cols = {"customer_id", "cost_fulfilment", "cost_returns", "cost_service",
                     "cost_acquisition", "cost_discounts", "total_cost_to_serve",
                     "gross_revenue", "gross_margin", "segment"}
    assert expected_cols.issubset(set(result.customer_cts.columns))
    assert len(result.customer_cts) == txn["customer_id"].nunique()


def test_segment_cts_margin_pct_computed():
    txn = make_transactions()
    rfm = compute_rfm(txn, snapshot_date=SNAPSHOT).scores
    result = compute_cost_to_serve(txn, rfm, SNAPSHOT)
    assert "margin_pct" in result.segment_cts.columns
    assert result.segment_cts["margin_pct"].notna().all()


def test_total_cts_non_negative():
    txn = make_transactions()
    rfm = compute_rfm(txn, snapshot_date=SNAPSHOT).scores
    result = compute_cost_to_serve(txn, rfm, SNAPSHOT)
    assert (result.customer_cts["total_cost_to_serve"] >= 0).all()
