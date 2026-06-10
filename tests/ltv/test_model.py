import pytest
import numpy as np
import pandas as pd

pytest.importorskip("sklearn")

from src.rfm.segment import compute_rfm
from src.ltv.model import build_features, run_pipeline

SNAPSHOT = pd.Timestamp("2026-06-10")


def make_dataset(n: int = 40) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    customers = [f"C{i:03d}" for i in range(n)]
    rows = []
    for cid in customers:
        n_orders = rng.integers(1, 8)
        for _ in range(n_orders):
            days_ago = rng.integers(1, 365)
            rows.append({
                "customer_id": cid,
                "order_date": SNAPSHOT - pd.Timedelta(days=int(days_ago)),
                "revenue": float(rng.integers(50, 500)),
                "discount_rate": float(rng.uniform(0, 0.7)),
                "is_drop": bool(rng.integers(0, 2)),
                "category": rng.choice(["tops", "bottoms", "accessories", "outerwear"]),
            })
    txn = pd.DataFrame(rows)
    labels = pd.Series(
        rng.uniform(0, 800, size=n),
        index=customers,
        name="forward_revenue_12m",
    )
    return txn, labels


def test_build_features_shape():
    txn, _ = make_dataset()
    rfm = compute_rfm(txn, snapshot_date=SNAPSHOT).scores
    features = build_features(rfm, txn, SNAPSHOT)
    assert len(features) == rfm["customer_id"].nunique()
    assert "R" in features.columns
    assert "outlet_revenue_share" in features.columns
    assert features.isnull().sum().sum() == 0


def test_pipeline_returns_ltv_result():
    txn, labels = make_dataset()
    rfm = compute_rfm(txn, snapshot_date=SNAPSHOT).scores
    result = run_pipeline(rfm, txn, labels, SNAPSHOT)
    assert "predicted_ltv_12m" in result.predictions.columns
    assert "ltv_tier" in result.predictions.columns
    assert len(result.predictions) == rfm["customer_id"].nunique()


def test_ltv_predictions_non_negative():
    txn, labels = make_dataset()
    rfm = compute_rfm(txn, snapshot_date=SNAPSHOT).scores
    result = run_pipeline(rfm, txn, labels, SNAPSHOT)
    assert (result.predictions["predicted_ltv_12m"] >= 0).all()


def test_segment_ltv_covers_all_rfm_segments():
    txn, labels = make_dataset()
    rfm = compute_rfm(txn, snapshot_date=SNAPSHOT).scores
    result = run_pipeline(rfm, txn, labels, SNAPSHOT)
    rfm_segments = set(rfm["segment"].unique())
    ltv_segments = set(result.segment_ltv.index)
    assert rfm_segments == ltv_segments


def test_metrics_keys_present():
    txn, labels = make_dataset()
    rfm = compute_rfm(txn, snapshot_date=SNAPSHOT).scores
    result = run_pipeline(rfm, txn, labels, SNAPSHOT)
    assert "cv_mae_log" in result.metrics
    assert "train_r2" in result.metrics


def test_feature_importance_sorted():
    txn, labels = make_dataset()
    rfm = compute_rfm(txn, snapshot_date=SNAPSHOT).scores
    result = run_pipeline(rfm, txn, labels, SNAPSHOT)
    imp = result.feature_importance["importance"].values
    # Only assert ordering when importance values are non-trivial
    if imp.sum() > 0:
        assert (imp[:-1] >= imp[1:]).all(), "feature importance should be descending"
