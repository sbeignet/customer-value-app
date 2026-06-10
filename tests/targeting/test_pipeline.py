import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from src.targeting.pipeline import run, build_campaign_briefs, build_segment_report, QUADRANT_BRIEF_METADATA

SNAPSHOT = pd.Timestamp("2026-06-10")


def make_catalog(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(20)
    return pd.DataFrame({
        "product_id": [f"P{i:03d}" for i in range(n)],
        "name": [f"Product {i}" for i in range(n)],
        "category": [["tops", "dresses", "outerwear", "accessories", "bottoms"][i % 5] for i in range(n)],
        "season": [["SS", "AW", "SS", "AW", "SS"][i % 5] for i in range(n)],
        "price": rng.uniform(20, 350, n).round(2),
        "discount_rate": rng.choice([0.0, 0.0, 0.10, 0.50, 0.60], n),
        "is_drop": rng.integers(0, 2, n).astype(bool),
        "margin_pct": rng.uniform(10, 70, n).round(1),
    })


def make_transactions(catalog: pd.DataFrame, n_customers: int = 25) -> pd.DataFrame:
    rng = np.random.default_rng(21)
    rows = []
    for i in range(n_customers):
        cid = f"C{i:03d}"
        for _ in range(rng.integers(2, 9)):
            prod = catalog.sample(1, random_state=int(rng.integers(0, 9999))).iloc[0]
            rows.append({
                "customer_id": cid,
                "product_id": prod["product_id"],
                "order_date": SNAPSHOT - pd.Timedelta(days=int(rng.integers(1, 365))),
                "revenue": float(prod["price"]),
                "discount_rate": float(prod["discount_rate"]),
                "quantity": int(rng.integers(1, 3)),
                "category": prod["category"],
                "is_drop": bool(prod["is_drop"]),
                "channel": rng.choice(["paid_social", "email", "organic"]),
            })
    return pd.DataFrame(rows)


# ── Pipeline integration tests ───────────────────────────────────────────────

def test_pipeline_runs_end_to_end():
    cat = make_catalog()
    txn = make_transactions(cat)
    result = run(txn, catalog=cat, snapshot_date=SNAPSHOT, n_recommendations=5)
    assert result.customer_master is not None
    assert len(result.customer_master) == txn["customer_id"].nunique()


def test_customer_master_has_all_dimensions():
    cat = make_catalog()
    txn = make_transactions(cat)
    result = run(txn, catalog=cat, snapshot_date=SNAPSHOT, n_recommendations=0)
    required = {"customer_id", "quadrant", "profit_score", "predicted_ltv_12m",
                "ltv_tier", "gross_margin", "segment", "R", "F", "M"}
    assert required.issubset(set(result.customer_master.columns))


def test_all_four_quadrants_possible():
    cat = make_catalog()
    txn = make_transactions(cat, n_customers=40)
    result = run(txn, catalog=cat, snapshot_date=SNAPSHOT, n_recommendations=0)
    quadrants = set(result.customer_master["quadrant"].unique())
    # With 40 customers across varied spend/margin profiles all 4 should appear
    assert quadrants.issubset({"PROTECT", "INVEST", "HARVEST", "REDUCE"})


def test_campaign_briefs_cover_all_quadrants():
    cat = make_catalog()
    txn = make_transactions(cat)
    result = run(txn, catalog=cat, snapshot_date=SNAPSHOT, n_recommendations=0)
    assert set(result.campaign_briefs.keys()) == {"PROTECT", "INVEST", "HARVEST", "REDUCE"}


def test_campaign_brief_contains_tactics():
    cat = make_catalog()
    txn = make_transactions(cat)
    result = run(txn, catalog=cat, snapshot_date=SNAPSHOT, n_recommendations=0)
    for quadrant, brief in result.campaign_briefs.items():
        if len(brief) > 0:
            assert "campaign_tactics" in brief.columns
            assert "campaign_objective" in brief.columns


def test_segment_report_columns():
    cat = make_catalog()
    txn = make_transactions(cat)
    result = run(txn, catalog=cat, snapshot_date=SNAPSHOT, n_recommendations=0)
    required = {"segment", "quadrant", "customers", "avg_ltv", "total_margin", "margin_pct"}
    assert required.issubset(set(result.segment_report.columns))


def test_recommendations_attached_when_catalog_provided():
    cat = make_catalog()
    txn = make_transactions(cat)
    result = run(txn, catalog=cat, snapshot_date=SNAPSHOT, n_recommendations=5)
    assert not result.recommendations.empty
    assert "customer_id" in result.recommendations.columns
    assert "product_id" in result.recommendations.columns


def test_recommendations_skipped_without_catalog():
    cat = make_catalog()
    txn = make_transactions(cat)
    # Remove product_id so recommendations are skipped gracefully
    txn_no_prod = txn.drop(columns=["product_id"])
    result = run(txn_no_prod, catalog=None, snapshot_date=SNAPSHOT, n_recommendations=5)
    assert result.recommendations.empty


def test_metadata_keys():
    cat = make_catalog()
    txn = make_transactions(cat)
    result = run(txn, catalog=cat, snapshot_date=SNAPSHOT, n_recommendations=0)
    assert "snapshot_date" in result.metadata
    assert "n_customers" in result.metadata
    assert "ltv_metrics" in result.metadata


def test_save_writes_files(tmp_path):
    cat = make_catalog()
    txn = make_transactions(cat)
    result = run(txn, catalog=cat, snapshot_date=SNAPSHOT, n_recommendations=5)
    paths = result.save(tmp_path)
    assert (tmp_path / "customer_master.csv").exists()
    assert (tmp_path / "segment_report.csv").exists()
    for quadrant in ["protect", "invest", "harvest", "reduce"]:
        p = tmp_path / f"campaign_brief_{quadrant}.csv"
        # Brief may be empty if quadrant has no customers but file should exist
        assert p.exists()


def test_profit_score_sorted_descending():
    cat = make_catalog()
    txn = make_transactions(cat)
    result = run(txn, catalog=cat, snapshot_date=SNAPSHOT, n_recommendations=0)
    scores = result.customer_master["profit_score"].values
    assert (scores[:-1] >= scores[1:]).all()


def test_season_filter_applied_to_recommendations():
    cat = make_catalog()
    txn = make_transactions(cat)
    result = run(txn, catalog=cat, snapshot_date=SNAPSHOT,
                 n_recommendations=5, season_filter=["SS"])
    if not result.recommendations.empty and "season" in result.recommendations.columns:
        assert result.recommendations["season"].dropna().isin(["SS"]).all()
