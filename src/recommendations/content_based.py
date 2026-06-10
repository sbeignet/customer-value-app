"""
Content-based product recommendations.

Solves two problems collaborative filtering cannot:
  1. Cold-start customers (< MIN_INTERACTIONS purchases)
  2. New-season products with no purchase history yet

Similarity is computed over product attributes:
  - category (one-hot)
  - price tier (quantile bucket: budget / mid / premium / luxury)
  - season (SS / AW / all-season)
  - is_full_price (bool: discount_rate < 0.15)
  - is_drop (bool)

Fashion use-case: at the start of a new SS or AW season, collaborative
signals are sparse. Content-based fills the gap using affinity profiles
built from a customer's past purchases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


PRICE_TIER_BINS = [0, 30, 80, 200, np.inf]
PRICE_TIER_LABELS = ["budget", "mid", "premium", "luxury"]


def build_product_features(catalog: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Encode product catalog into a numeric feature matrix.

    catalog required columns: product_id, category, price
    catalog optional columns: season, is_drop, discount_rate
    """
    df = catalog.copy().set_index("product_id")

    features = pd.DataFrame(index=df.index)

    # Price tier
    features["price_tier"] = pd.cut(
        df["price"], bins=PRICE_TIER_BINS, labels=PRICE_TIER_LABELS, right=False
    ).astype(str)

    # Full-price flag
    if "discount_rate" in df.columns:
        features["is_full_price"] = (df["discount_rate"].fillna(0) < 0.15).astype(float)
    else:
        features["is_full_price"] = 1.0

    # Drop flag
    features["is_drop"] = df.get("is_drop", pd.Series(False, index=df.index)).astype(float)

    # Season (one-hot)
    season = df.get("season", pd.Series("unknown", index=df.index)).fillna("unknown")
    season_dummies = pd.get_dummies(season, prefix="season")
    features = pd.concat([features, season_dummies], axis=1)

    # Category (one-hot)
    category = df.get("category", pd.Series("unknown", index=df.index)).fillna("unknown")
    cat_dummies = pd.get_dummies(category, prefix="cat")
    features = pd.concat([features, cat_dummies], axis=1)

    # Price tier (one-hot, replace string column)
    tier_dummies = pd.get_dummies(features["price_tier"], prefix="tier")
    features = features.drop(columns=["price_tier"]).join(tier_dummies)

    # Normalise continuous features
    num_cols = ["is_full_price", "is_drop"]
    features[num_cols] = MinMaxScaler().fit_transform(features[num_cols].fillna(0))

    feature_matrix = features.fillna(0).values.astype("float32")
    return features, feature_matrix


def build_customer_profile(
    customer_id: str,
    transactions: pd.DataFrame,
    product_features: pd.DataFrame,
    outlet_discount_threshold: float = 0.40,
) -> np.ndarray | None:
    """
    Weighted average of purchased product feature vectors.
    Outlet purchases weighted at 0.3 (same logic as collaborative).
    Returns None if customer has no known product history.
    """
    df = transactions[transactions["customer_id"] == customer_id].copy()
    if df.empty or "product_id" not in df.columns:
        return None

    df = df[df["product_id"].isin(product_features.index)]
    if df.empty:
        return None

    if "discount_rate" in df.columns:
        df["weight"] = np.where(
            df["discount_rate"] >= outlet_discount_threshold,
            0.3,
            1.0,
        )
        df["weight"] *= np.log1p(df.get("quantity", 1) * df["revenue"])
    else:
        df["weight"] = np.log1p(df["revenue"])

    vecs = product_features.loc[df["product_id"]].values
    weights = df["weight"].values[:, np.newaxis]
    profile = (vecs * weights).sum(axis=0) / weights.sum()
    return profile.astype("float32")


@dataclass
class ContentBasedModel:
    product_features: pd.DataFrame
    feature_matrix: np.ndarray

    def recommend(
        self,
        customer_id: str,
        transactions: pd.DataFrame,
        n: int = 10,
        filter_already_purchased: bool = True,
        season_filter: list[str] | None = None,
        full_price_only: bool = False,
    ) -> pd.DataFrame:
        """
        Recommend products by cosine similarity to the customer's purchase profile.
        """
        profile = build_customer_profile(customer_id, transactions, self.product_features)
        if profile is None:
            return pd.DataFrame(columns=["product_id", "score", "rank"])

        sims = cosine_similarity(profile.reshape(1, -1), self.feature_matrix).flatten()
        product_ids = self.product_features.index.tolist()

        recs = pd.DataFrame({"product_id": product_ids, "score": sims})

        if filter_already_purchased:
            bought = transactions[transactions["customer_id"] == customer_id]["product_id"].unique()
            recs = recs[~recs["product_id"].isin(bought)]

        if season_filter:
            season_cols = [f"season_{s}" for s in season_filter if f"season_{s}" in self.product_features.columns]
            if season_cols:
                season_mask = self.product_features[season_cols].max(axis=1) > 0
                valid = self.product_features.index[season_mask.values]
                recs = recs[recs["product_id"].isin(valid)]

        if full_price_only and "is_full_price" in self.product_features.columns:
            full_price = self.product_features[self.product_features["is_full_price"] >= 0.85].index
            recs = recs[recs["product_id"].isin(full_price)]

        recs = recs.sort_values("score", ascending=False).head(n).reset_index(drop=True)
        recs["rank"] = recs.index + 1
        return recs


def train(catalog: pd.DataFrame) -> ContentBasedModel:
    """Build content-based model from product catalog."""
    product_features, feature_matrix = build_product_features(catalog)
    return ContentBasedModel(product_features=product_features, feature_matrix=feature_matrix)
