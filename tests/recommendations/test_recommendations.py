import numpy as np
import pandas as pd
import pytest
from src.recommendations import collaborative, content_based
from src.recommendations.reranker import rerank, rerank_batch
from src.recommendations.engine import RecommendationEngine

SNAPSHOT = pd.Timestamp("2026-06-10")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_catalog(n_products: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(10)
    categories = ["tops", "dresses", "outerwear", "accessories", "bottoms"]
    seasons = ["SS", "AW", "SS", "AW", "SS"]
    return pd.DataFrame({
        "product_id": [f"P{i:03d}" for i in range(n_products)],
        "name": [f"Product {i}" for i in range(n_products)],
        "category": [categories[i % 5] for i in range(n_products)],
        "season": [seasons[i % 5] for i in range(n_products)],
        "price": rng.uniform(20, 350, n_products).round(2),
        "discount_rate": rng.choice([0.0, 0.0, 0.10, 0.50, 0.60], n_products),
        "is_drop": rng.integers(0, 2, n_products).astype(bool),
        "margin_pct": rng.uniform(10, 70, n_products).round(1),
    })


def make_transactions(catalog: pd.DataFrame, n_customers: int = 15) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    customers = [f"C{i:02d}" for i in range(n_customers)]
    rows = []
    for cid in customers:
        n_orders = int(rng.integers(2, 8))
        prods = rng.choice(catalog["product_id"], n_orders, replace=False)
        for pid in prods:
            prod = catalog[catalog["product_id"] == pid].iloc[0]
            rows.append({
                "customer_id": cid,
                "product_id": pid,
                "order_date": SNAPSHOT - pd.Timedelta(days=int(rng.integers(1, 300))),
                "revenue": float(prod["price"]),
                "discount_rate": float(prod["discount_rate"]),
                "quantity": int(rng.integers(1, 3)),
                "category": prod["category"],
                "is_drop": bool(prod["is_drop"]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Collaborative filtering
# ---------------------------------------------------------------------------

def test_collab_train_returns_model():
    cat = make_catalog()
    txn = make_transactions(cat)
    model = collaborative.train(txn, n_factors=10, iterations=5)
    assert model.backend in ("implicit", "svd")
    assert len(model.customer_enc.classes_) > 0
    assert len(model.product_enc.classes_) > 0


def test_collab_recommend_known_customer():
    cat = make_catalog()
    txn = make_transactions(cat)
    model = collaborative.train(txn, n_factors=10, iterations=5)
    cid = txn["customer_id"].iloc[0]
    recs = model.recommend(cid, n=5)
    assert len(recs) <= 5
    assert "product_id" in recs.columns
    assert "score" in recs.columns


def test_collab_recommend_unknown_customer():
    cat = make_catalog()
    txn = make_transactions(cat)
    model = collaborative.train(txn, n_factors=10, iterations=5)
    recs = model.recommend("UNKNOWN_XYZ", n=5)
    assert recs.empty


def test_collab_filter_already_purchased():
    cat = make_catalog()
    txn = make_transactions(cat)
    model = collaborative.train(txn, n_factors=10, iterations=5)
    cid = txn["customer_id"].iloc[0]
    purchased = set(txn[txn["customer_id"] == cid]["product_id"])
    recs = model.recommend(cid, n=10, filter_already_purchased=True)
    assert set(recs["product_id"]).isdisjoint(purchased)


def test_collab_season_filter():
    cat = make_catalog()
    txn = make_transactions(cat)
    model = collaborative.train(txn, n_factors=10, iterations=5)
    cid = txn["customer_id"].iloc[0]
    recs = model.recommend(cid, n=10, season_filter=["SS"], product_catalog=cat)
    if len(recs) > 0:
        ss_products = set(cat[cat["season"] == "SS"]["product_id"])
        assert set(recs["product_id"]).issubset(ss_products)


def test_collab_batch_recommend():
    cat = make_catalog()
    txn = make_transactions(cat)
    model = collaborative.train(txn, n_factors=10, iterations=5)
    customers = txn["customer_id"].unique()[:3].tolist()
    recs = model.recommend_batch(customers, n=5)
    assert "customer_id" in recs.columns
    assert set(recs["customer_id"].unique()).issubset(set(customers))


# ---------------------------------------------------------------------------
# Content-based
# ---------------------------------------------------------------------------

def test_content_train_returns_model():
    cat = make_catalog()
    model = content_based.train(cat)
    assert len(model.product_features) == len(cat)


def test_content_recommend_returns_unseen():
    cat = make_catalog()
    txn = make_transactions(cat)
    model = content_based.train(cat)
    cid = txn["customer_id"].iloc[0]
    purchased = set(txn[txn["customer_id"] == cid]["product_id"])
    recs = model.recommend(cid, txn, n=5, filter_already_purchased=True)
    if len(recs) > 0:
        assert set(recs["product_id"]).isdisjoint(purchased)


def test_content_recommend_unknown_customer():
    cat = make_catalog()
    txn = make_transactions(cat)
    model = content_based.train(cat)
    recs = model.recommend("GHOST", txn, n=5)
    assert recs.empty


def test_content_full_price_filter():
    cat = make_catalog()
    txn = make_transactions(cat)
    model = content_based.train(cat)
    cid = txn["customer_id"].iloc[0]
    recs = model.recommend(cid, txn, n=10, full_price_only=True)
    if len(recs) > 0:
        full_price_pids = set(cat[cat["discount_rate"] < 0.15]["product_id"])
        assert set(recs["product_id"]).issubset(full_price_pids)


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

def make_recs(n: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    return pd.DataFrame({
        "product_id": [f"P{i:03d}" for i in range(n)],
        "score": rng.uniform(0.1, 1.0, n),
        "rank": range(1, n + 1),
    })


def test_rerank_returns_same_products():
    recs = make_recs()
    reranked = rerank(recs, "INVEST")
    assert set(reranked["product_id"]) == set(recs["product_id"])


def test_rerank_invest_boosts_full_price():
    cat = make_catalog(20)
    recs = pd.DataFrame({
        "product_id": cat["product_id"].tolist(),
        "score": np.linspace(0.9, 0.1, len(cat)),
        "rank": range(1, len(cat) + 1),
    })
    invest = rerank(recs.copy(), "INVEST", cat)
    reduce = rerank(recs.copy(), "REDUCE", cat)

    full_price_pids = set(cat[cat["discount_rate"] < 0.40]["product_id"])
    invest_top5 = set(invest.head(5)["product_id"])
    reduce_top5 = set(reduce.head(5)["product_id"])

    # INVEST should have more full-price items in top-5 than REDUCE
    assert len(invest_top5 & full_price_pids) >= len(reduce_top5 & full_price_pids)


def test_rerank_batch_covers_all_customers():
    cat = make_catalog()
    txn = make_transactions(cat, n_customers=5)
    model = collaborative.train(txn, n_factors=10, iterations=5)
    customers = txn["customer_id"].unique().tolist()
    batch = model.recommend_batch(customers, n=5)
    quadrants = pd.Series("INVEST", index=customers)
    result = rerank_batch(batch, quadrants, cat)
    assert set(result["customer_id"].unique()) == set(batch["customer_id"].unique())


# ---------------------------------------------------------------------------
# Hybrid engine
# ---------------------------------------------------------------------------

def test_engine_build():
    cat = make_catalog()
    txn = make_transactions(cat)
    engine = RecommendationEngine.build(txn, cat, n_factors=10, iterations=5)
    assert engine.collab_model is not None
    assert engine.content_model is not None


def test_engine_recommend_single():
    cat = make_catalog()
    txn = make_transactions(cat)
    engine = RecommendationEngine.build(txn, cat, n_factors=10, iterations=5)
    cid = txn["customer_id"].iloc[0]
    recs = engine.recommend(cid, quadrant="INVEST", n=5)
    assert len(recs) <= 5
    assert "source" in recs.columns


def test_engine_recommend_for_segment():
    cat = make_catalog()
    txn = make_transactions(cat, n_customers=10)
    engine = RecommendationEngine.build(txn, cat, n_factors=10, iterations=5)
    customers = txn["customer_id"].unique()[:6].tolist()
    profit_df = pd.DataFrame({
        "customer_id": customers,
        "quadrant": ["INVEST", "PROTECT", "HARVEST", "REDUCE", "INVEST", "PROTECT"],
    })
    recs = engine.recommend_for_segment(customers, profit_df, n=5)
    assert "customer_id" in recs.columns
    assert "quadrant" in recs.columns
    assert len(recs) > 0
