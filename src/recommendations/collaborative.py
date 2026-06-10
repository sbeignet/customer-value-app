"""
Collaborative filtering via matrix factorisation.

Two backends (auto-selected):
  1. implicit ALS  — fast, handles sparse purchase matrices well, production-grade
  2. sklearn TruncatedSVD — no extra deps, used as fallback

Fashion-specific design decisions:
  - Interaction weight = log1p(quantity × net_revenue), not raw purchase count.
    This prevents a single high-AOV transaction from dominating a frequent buyer.
  - Outlet purchases are down-weighted (factor 0.3): we do not want the model
    to learn that outlet SKUs are "similar" to full-price SKUs just because the
    same customer bought both.
  - Season filter on inference: only return products available in the target
    season (SS or AW) unless cross-season is explicitly enabled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

try:
    import implicit
    HAS_IMPLICIT = True
except ImportError:
    HAS_IMPLICIT = False

from sklearn.decomposition import TruncatedSVD


OUTLET_WEIGHT = 0.3   # down-weight for outlet interactions
MIN_INTERACTIONS = 2  # customers with fewer interactions are excluded from ALS


# ---------------------------------------------------------------------------
# Interaction matrix builder
# ---------------------------------------------------------------------------

def build_interaction_matrix(
    transactions: pd.DataFrame,
    outlet_discount_threshold: float = 0.40,
) -> tuple[csr_matrix, LabelEncoder, LabelEncoder]:
    """
    Build a sparse customer × product interaction matrix.

    Weight = log1p(quantity × net_revenue), outlet purchases × 0.3.

    Returns
    -------
    matrix        : (n_customers, n_products) sparse float32
    customer_enc  : fitted LabelEncoder for customer_id
    product_enc   : fitted LabelEncoder for product_id
    """
    df = transactions.copy()
    df = df[df["product_id"].notna() & df["customer_id"].notna()]

    if "discount_rate" in df.columns:
        df["net_revenue"] = df["revenue"] * (1 - df["discount_rate"].fillna(0))
        df["is_outlet"] = df["discount_rate"] >= outlet_discount_threshold
    else:
        df["net_revenue"] = df["revenue"]
        df["is_outlet"] = False

    qty = df.get("quantity", pd.Series(1, index=df.index))
    df["weight"] = np.log1p(qty * df["net_revenue"])
    df.loc[df["is_outlet"], "weight"] *= OUTLET_WEIGHT

    # Aggregate by customer × product
    agg = df.groupby(["customer_id", "product_id"])["weight"].sum().reset_index()

    customer_enc = LabelEncoder().fit(agg["customer_id"])
    product_enc = LabelEncoder().fit(agg["product_id"])

    rows = customer_enc.transform(agg["customer_id"])
    cols = product_enc.transform(agg["product_id"])
    data = agg["weight"].astype("float32").values

    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(customer_enc.classes_), len(product_enc.classes_)),
    )
    return matrix, customer_enc, product_enc


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

@dataclass
class CollaborativeModel:
    customer_enc: LabelEncoder
    product_enc: LabelEncoder
    matrix: csr_matrix
    _model: object = field(repr=False)
    backend: str = "svd"

    def recommend(
        self,
        customer_id: str,
        n: int = 10,
        filter_already_purchased: bool = True,
        season_filter: list[str] | None = None,
        product_catalog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Return top-N product recommendations for a single customer.

        Parameters
        ----------
        customer_id              : target customer
        n                        : number of recommendations
        filter_already_purchased : exclude products already bought
        season_filter            : list of seasons to restrict to, e.g. ["SS"]
        product_catalog          : DataFrame with product_id, season, margin_pct
                                   (used for season filtering and margin lookup)
        """
        if customer_id not in self.customer_enc.classes_:
            return pd.DataFrame(columns=["product_id", "score", "rank"])

        cidx = self.customer_enc.transform([customer_id])[0]

        if self.backend == "implicit":
            ids, scores = self._model.recommend(
                cidx, self.matrix[cidx],
                N=n * 3,  # over-fetch to allow post-filtering
                filter_already_liked=filter_already_purchased,
            )
            product_ids = self.product_enc.inverse_transform(ids)
            score_values = scores
        else:
            # SVD: score via dot product of user embedding × item embeddings
            user_vec = self._model.transform(self.matrix[cidx])
            item_vecs = self._model.components_.T
            raw_scores = (user_vec @ item_vecs.T).flatten()

            if filter_already_purchased:
                purchased_cols = self.matrix[cidx].indices
                raw_scores[purchased_cols] = -np.inf

            n_fetch = min(n * 3, len(raw_scores))
            top_idx = np.argpartition(raw_scores, -n_fetch)[-n_fetch:]
            top_idx = top_idx[np.argsort(raw_scores[top_idx])[::-1]]
            product_ids = self.product_enc.inverse_transform(top_idx)
            score_values = raw_scores[top_idx]

        recs = pd.DataFrame({"product_id": product_ids, "score": score_values})

        # Season filter
        if season_filter and product_catalog is not None:
            valid = product_catalog[product_catalog["season"].isin(season_filter)]["product_id"]
            recs = recs[recs["product_id"].isin(valid)]

        recs = recs.head(n).reset_index(drop=True)
        recs["rank"] = recs.index + 1

        # Attach catalog metadata if available
        if product_catalog is not None:
            recs = recs.merge(
                product_catalog[["product_id"] + [c for c in ["name", "category", "price", "season", "margin_pct"]
                                                   if c in product_catalog.columns]],
                on="product_id", how="left",
            )

        return recs

    def recommend_batch(
        self,
        customer_ids: list[str],
        n: int = 10,
        **kwargs,
    ) -> pd.DataFrame:
        """Recommend for multiple customers; returns long-format DataFrame."""
        results = []
        for cid in customer_ids:
            recs = self.recommend(cid, n=n, **kwargs)
            recs.insert(0, "customer_id", cid)
            results.append(recs)
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    transactions: pd.DataFrame,
    n_factors: int = 50,
    iterations: int = 30,
    regularisation: float = 0.01,
    outlet_discount_threshold: float = 0.40,
) -> CollaborativeModel:
    """
    Train collaborative filtering model.

    Parameters
    ----------
    transactions : customer_id, product_id, revenue, quantity (opt),
                   discount_rate (opt), season (opt)
    n_factors    : latent dimension
    iterations   : ALS iterations (or SVD components)
    """
    matrix, customer_enc, product_enc = build_interaction_matrix(
        transactions, outlet_discount_threshold
    )

    if HAS_IMPLICIT:
        # implicit expects (items × users) matrix
        model = implicit.als.AlternatingLeastSquares(
            factors=n_factors,
            iterations=iterations,
            regularization=regularisation,
            random_state=42,
        )
        model.fit(matrix.T)  # item_users format
        backend = "implicit"
    else:
        model = TruncatedSVD(n_components=min(n_factors, min(matrix.shape) - 1), random_state=42)
        model.fit(matrix)
        backend = "svd"

    return CollaborativeModel(
        customer_enc=customer_enc,
        product_enc=product_enc,
        matrix=matrix,
        _model=model,
        backend=backend,
    )
