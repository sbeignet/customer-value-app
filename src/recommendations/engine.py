"""
Hybrid recommendation engine.

Routing logic per customer:
  1. If ≥ MIN_INTERACTIONS purchases: collaborative filtering (primary)
  2. If < MIN_INTERACTIONS or cold-start: content-based (fallback)
  3. All results are profit-aware re-ranked based on quadrant

End-to-end usage:
    engine = RecommendationEngine.build(transactions, catalog)
    recs = engine.recommend_for_segment(
        customer_ids=invest_customers,
        profit_map=profit_map_result,
        season_filter=["SS"],
        n=10,
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.recommendations import collaborative, content_based
from src.recommendations.reranker import rerank_batch

MIN_INTERACTIONS = 3   # switch to content-based below this threshold


@dataclass
class RecommendationEngine:
    collab_model: collaborative.CollaborativeModel
    content_model: content_based.ContentBasedModel
    transactions: pd.DataFrame
    catalog: pd.DataFrame

    @classmethod
    def build(
        cls,
        transactions: pd.DataFrame,
        catalog: pd.DataFrame,
        n_factors: int = 50,
        iterations: int = 30,
    ) -> "RecommendationEngine":
        """Train both models and return a ready engine."""
        collab = collaborative.train(transactions, n_factors=n_factors, iterations=iterations)
        content = content_based.train(catalog)
        return cls(collab_model=collab, content_model=content,
                   transactions=transactions, catalog=catalog)

    def _customer_interaction_counts(self) -> pd.Series:
        return self.transactions.groupby("customer_id")["product_id"].nunique()

    def recommend(
        self,
        customer_id: str,
        quadrant: str = "PROTECT",
        n: int = 10,
        season_filter: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Single-customer recommendation with automatic collab/content routing
        and profit-aware re-ranking.
        """
        interaction_counts = self._customer_interaction_counts()
        n_interactions = interaction_counts.get(customer_id, 0)

        if n_interactions >= MIN_INTERACTIONS:
            recs = self.collab_model.recommend(
                customer_id, n=n * 2,
                season_filter=season_filter,
                product_catalog=self.catalog,
            )
            source = "collaborative"
        else:
            recs = self.content_model.recommend(
                customer_id, self.transactions, n=n * 2,
                season_filter=season_filter,
                full_price_only=(quadrant == "INVEST"),
            )
            source = "content_based"

        if recs.empty:
            return recs

        recs = rerank(recs, quadrant, self.catalog)
        recs["source"] = source
        return recs.head(n)

    def recommend_for_segment(
        self,
        customer_ids: list[str],
        profit_map,           # ProfitMapResult or DataFrame with customer_id, quadrant
        n: int = 10,
        season_filter: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Batch recommendations for a list of customers.
        Automatically routes collab vs content and applies profit-aware reranking.

        profit_map can be a ProfitMapResult (has .customer_map) or a plain DataFrame.
        """
        if hasattr(profit_map, "customer_map"):
            qmap = profit_map.customer_map.set_index("customer_id")["quadrant"]
        else:
            qmap = profit_map.set_index("customer_id")["quadrant"]

        interaction_counts = self._customer_interaction_counts()

        collab_customers = [
            c for c in customer_ids
            if interaction_counts.get(c, 0) >= MIN_INTERACTIONS
        ]
        content_customers = [
            c for c in customer_ids
            if interaction_counts.get(c, 0) < MIN_INTERACTIONS
        ]

        results = []

        # --- Collaborative batch ---
        if collab_customers:
            collab_recs = self.collab_model.recommend_batch(
                collab_customers, n=n * 2,
                season_filter=season_filter,
                product_catalog=self.catalog,
            )
            if not collab_recs.empty:
                reranked = rerank_batch(collab_recs, qmap, self.catalog)
                reranked["source"] = "collaborative"
                reranked = reranked.groupby("customer_id").head(n)
                results.append(reranked)

        # --- Content-based batch ---
        for cid in content_customers:
            quadrant = qmap.get(cid, "PROTECT")
            recs = self.content_model.recommend(
                cid, self.transactions, n=n * 2,
                season_filter=season_filter,
                full_price_only=(quadrant == "INVEST"),
            )
            if recs.empty:
                continue
            recs.insert(0, "customer_id", cid)
            results.append(recs)

        if not results:
            return pd.DataFrame()

        out = pd.concat(results, ignore_index=True)
        out["quadrant"] = out["customer_id"].map(qmap)
        return out.sort_values(["customer_id", "rank"])


# convenience re-export so callers can do: from src.recommendations.engine import rerank
from src.recommendations.reranker import rerank  # noqa: E402, F401
