"""
Profit-aware re-ranking layer.

Takes raw recommendation scores (from collaborative or content-based) and
re-scores them to align product suggestions with the customer's profit quadrant.

Quadrant-specific reranking logic
----------------------------------
PROTECT   → No change; surface best products regardless of margin.
            These customers buy at full price — trust the signal.

INVEST    → Boost full-price items, penalise outlet SKUs.
            Goal: migrate the customer from discount dependency to full-price.
            Also boost drop/limited items (higher excitement, higher margin).

HARVEST   → Boost high-margin products; penalise complex fulfilment
            (heavy items, bulky categories) to protect the margin.

REDUCE    → Boost clearance / outlet items only.
            These customers are unlikely to re-engage at full price;
            capture residual value through outlet without cannibalising
            full-price allocation.

Final score formula
-------------------
  reranked_score = base_score × relevance_weight + margin_bonus × margin_weight

  relevance_weight = 0.7 (preserve recommendation quality)
  margin_weight    = 0.3 (inject business objective)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


RELEVANCE_WEIGHT = 0.70
MARGIN_WEIGHT = 0.30

# Per-quadrant multipliers applied to product scores
# Keys: (quadrant, product attribute)
QUADRANT_BOOSTS: dict[str, dict] = {
    "PROTECT": {
        "full_price_boost": 1.0,
        "outlet_penalty": 1.0,
        "drop_boost": 1.1,
        "high_margin_boost": 1.0,
    },
    "INVEST": {
        "full_price_boost": 1.5,
        "outlet_penalty": 0.4,
        "drop_boost": 1.3,
        "high_margin_boost": 1.2,
    },
    "HARVEST": {
        "full_price_boost": 1.1,
        "outlet_penalty": 0.8,
        "drop_boost": 1.0,
        "high_margin_boost": 1.4,
    },
    "REDUCE": {
        "full_price_boost": 0.7,
        "outlet_penalty": 1.5,   # boost outlet for REDUCE (clearance value)
        "drop_boost": 0.8,
        "high_margin_boost": 0.9,
    },
}

HIGH_MARGIN_THRESHOLD = 0.45  # product margin_pct above this = high margin
OUTLET_DISCOUNT_THRESHOLD = 0.40


def _normalise(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else pd.Series(0.5, index=s.index)


def rerank(
    recommendations: pd.DataFrame,
    customer_quadrant: str,
    product_catalog: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Re-score and re-rank a recommendation list for a single customer.

    Parameters
    ----------
    recommendations  : DataFrame with columns product_id, score
    customer_quadrant: one of PROTECT / INVEST / HARVEST / REDUCE
    product_catalog  : optional DataFrame with product_id, margin_pct,
                       discount_rate (or is_outlet), is_drop

    Returns
    -------
    DataFrame with reranked_score and updated rank column.
    """
    recs = recommendations.copy()
    boosts = QUADRANT_BOOSTS.get(customer_quadrant, QUADRANT_BOOSTS["PROTECT"])

    # Normalise base score to [0, 1]
    recs["_norm_score"] = _normalise(recs["score"])

    # Default multiplier = 1.0
    recs["_multiplier"] = 1.0

    if product_catalog is not None:
        cat = product_catalog.set_index("product_id")
        recs = recs.set_index("product_id")

        # Full-price flag
        if "discount_rate" in cat.columns:
            is_outlet = cat["discount_rate"].fillna(0) >= OUTLET_DISCOUNT_THRESHOLD
        elif "is_outlet" in cat.columns:
            is_outlet = cat["is_outlet"].astype(bool)
        else:
            is_outlet = pd.Series(False, index=cat.index)

        is_full_price = ~is_outlet
        is_drop = cat.get("is_drop", pd.Series(False, index=cat.index)).astype(bool)

        if "margin_pct" in cat.columns:
            is_high_margin = cat["margin_pct"] >= HIGH_MARGIN_THRESHOLD * 100
        else:
            is_high_margin = pd.Series(False, index=cat.index)

        for pid in recs.index:
            if pid not in cat.index:
                continue
            m = 1.0
            if is_full_price.get(pid, False):
                m *= boosts["full_price_boost"]
            if is_outlet.get(pid, False):
                m *= boosts["outlet_penalty"]
            if is_drop.get(pid, False):
                m *= boosts["drop_boost"]
            if is_high_margin.get(pid, False):
                m *= boosts["high_margin_boost"]
            recs.at[pid, "_multiplier"] = m

        recs = recs.reset_index()

    # Margin bonus: normalised multiplier excess above 1.0
    recs["_margin_bonus"] = (_normalise(recs["_multiplier"]) - 0.5).clip(0)

    recs["reranked_score"] = (
        RELEVANCE_WEIGHT * recs["_norm_score"]
        + MARGIN_WEIGHT * recs["_margin_bonus"]
    )

    recs = (
        recs.drop(columns=["_norm_score", "_multiplier", "_margin_bonus"])
        .sort_values("reranked_score", ascending=False)
        .reset_index(drop=True)
    )
    recs["rank"] = recs.index + 1
    return recs


def rerank_batch(
    batch_recommendations: pd.DataFrame,
    customer_quadrants: pd.Series,
    product_catalog: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Re-rank recommendations for multiple customers.

    Parameters
    ----------
    batch_recommendations : long-format DataFrame with customer_id, product_id, score
    customer_quadrants    : Series indexed by customer_id → quadrant string
    product_catalog       : optional product metadata
    """
    results = []
    for cid, grp in batch_recommendations.groupby("customer_id"):
        quadrant = customer_quadrants.get(cid, "PROTECT")
        reranked = rerank(grp.drop(columns=["customer_id"]), quadrant, product_catalog)
        reranked.insert(0, "customer_id", cid)
        results.append(reranked)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
