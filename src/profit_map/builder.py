"""
Profit map: combines LTV predictions with cost-to-serve to classify
every customer on two axes and derive a prioritisation score.

Axes
----
  X — Predicted LTV tier      (platinum / gold / silver / bronze)
  Y — Current profitability   (high / medium / low / negative)

Quadrant logic
--------------
  ┌──────────────┬──────────────┐
  │  INVEST      │  PROTECT     │
  │  High LTV    │  High LTV    │
  │  Low profit  │  High profit │
  ├──────────────┼──────────────┤
  │  REDUCE CTS  │  HARVEST     │
  │  Low LTV     │  Low LTV     │
  │  Low profit  │  High profit │
  └──────────────┴──────────────┘

  PROTECT   → retention spend, VIP treatment, loyalty programme
  INVEST    → reduce cost-to-serve (returns mgmt, service deflection),
               upsell to full-price, move out of outlet
  HARVEST   → maintain margin, minimal incremental spend
  REDUCE    → churn candidates, suppress acquisition lookalike spend

Profit score (0–100)
--------------------
  Composite of: current margin contribution (40%) + predicted LTV (40%)
              + LTV trend potential (20%, based on RFM improvement headroom)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Quadrant assignment
# ---------------------------------------------------------------------------

LTV_TIER_RANK = {"platinum": 4, "gold": 3, "silver": 2, "bronze": 1}
PROFIT_THRESHOLDS = {"high": 0.60, "medium": 0.30, "low": 0.0}  # margin_pct percentile cuts


def assign_profitability_band(margin_pct: pd.Series) -> pd.Series:
    """
    Classify customers into profitability bands based on margin % percentiles.
    'negative' = margin_pct < 0 (cost-to-serve exceeds revenue).
    """
    bands = pd.Series("low", index=margin_pct.index)
    bands[margin_pct < 0] = "negative"
    p30 = margin_pct[margin_pct >= 0].quantile(0.30)
    p70 = margin_pct[margin_pct >= 0].quantile(0.70)
    bands[(margin_pct >= 0) & (margin_pct < p30)] = "low"
    bands[(margin_pct >= p30) & (margin_pct < p70)] = "medium"
    bands[margin_pct >= p70] = "high"
    return bands.rename("profitability_band")


QUADRANT_MAP = {
    ("platinum", "high"):  "PROTECT",
    ("platinum", "medium"): "PROTECT",
    ("platinum", "low"):   "INVEST",
    ("platinum", "negative"): "INVEST",
    ("gold", "high"):      "PROTECT",
    ("gold", "medium"):    "PROTECT",
    ("gold", "low"):       "INVEST",
    ("gold", "negative"):  "INVEST",
    ("silver", "high"):    "HARVEST",
    ("silver", "medium"):  "HARVEST",
    ("silver", "low"):     "REDUCE",
    ("silver", "negative"): "REDUCE",
    ("bronze", "high"):    "HARVEST",
    ("bronze", "medium"):  "HARVEST",
    ("bronze", "low"):     "REDUCE",
    ("bronze", "negative"): "REDUCE",
}

QUADRANT_ACTION = {
    "PROTECT": "Maximise retention; VIP treatment; loyalty programme; priority service",
    "INVEST":  "Reduce cost-to-serve; migrate from outlet to full-price; upsell drops",
    "HARVEST": "Maintain margin; automate service; limit incremental spend",
    "REDUCE":  "Suppress lookalike acquisition; churn candidates; outlet-only comms",
}


def assign_quadrant(ltv_tier: pd.Series, profitability_band: pd.Series) -> pd.Series:
    combined = list(zip(ltv_tier, profitability_band))
    return pd.Series(
        [QUADRANT_MAP.get((lt, pb), "REDUCE") for lt, pb in combined],
        index=ltv_tier.index,
        name="quadrant",
    )


# ---------------------------------------------------------------------------
# Profit score (0–100)
# ---------------------------------------------------------------------------

def compute_profit_score(
    predicted_ltv: pd.Series,
    gross_margin: pd.Series,
    rfm_score: pd.Series,
) -> pd.Series:
    """
    Composite 0–100 score:
      40% normalised predicted LTV
      40% normalised gross margin
      20% normalised RFM score (headroom proxy)
    """
    def minmax(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else pd.Series(0.5, index=s.index)

    score = (
        0.40 * minmax(predicted_ltv.reindex(rfm_score.index).fillna(0))
        + 0.40 * minmax(gross_margin.reindex(rfm_score.index).fillna(0))
        + 0.20 * minmax(rfm_score)
    ) * 100

    return score.clip(0, 100).round(1).rename("profit_score")


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

@dataclass
class ProfitMapResult:
    customer_map: pd.DataFrame        # one row per customer, all dimensions
    quadrant_summary: pd.DataFrame    # counts, avg metrics per quadrant
    segment_profit_summary: pd.DataFrame  # RFM segment × quadrant cross-tab


def build_profit_map(
    ltv_predictions: pd.DataFrame,   # from LTVResult.predictions
    customer_cts: pd.DataFrame,       # from CostToServeResult.customer_cts
    rfm_scores: pd.DataFrame,         # from RFMResult.scores
) -> ProfitMapResult:
    """
    Join LTV, cost-to-serve, and RFM into a unified profit map.

    Parameters
    ----------
    ltv_predictions : customer_id, predicted_ltv_12m, ltv_tier
    customer_cts    : customer_id, gross_margin, cts_pct_revenue, total_cost_to_serve
    rfm_scores      : customer_id, rfm_score, segment, R, F, M
    """
    # --- Join all three sources ---
    base = rfm_scores[["customer_id", "rfm_score", "segment", "R", "F", "M"]].copy()
    base = base.merge(
        ltv_predictions[["customer_id", "predicted_ltv_12m", "ltv_tier"]],
        on="customer_id", how="left",
    )
    base = base.merge(
        customer_cts[["customer_id", "gross_revenue", "gross_margin",
                      "total_cost_to_serve", "cts_pct_revenue"]],
        on="customer_id", how="left",
    )

    # --- Profitability band ---
    base["profitability_band"] = assign_profitability_band(base["gross_margin"].fillna(0))

    # --- Quadrant ---
    base["ltv_tier"] = base["ltv_tier"].fillna("bronze")
    base["quadrant"] = assign_quadrant(base["ltv_tier"], base["profitability_band"])
    base["quadrant_action"] = base["quadrant"].map(QUADRANT_ACTION)

    # --- Profit score ---
    base["profit_score"] = compute_profit_score(
        base.set_index("customer_id")["predicted_ltv_12m"],
        base.set_index("customer_id")["gross_margin"],
        base.set_index("customer_id")["rfm_score"],
    ).values

    # --- Quadrant summary ---
    qsummary = (
        base.groupby("quadrant")
        .agg(
            customers=("customer_id", "count"),
            pct_of_base=("customer_id", lambda x: round(len(x) / len(base) * 100, 1)),
            avg_predicted_ltv=("predicted_ltv_12m", "mean"),
            avg_gross_margin=("gross_margin", "mean"),
            total_gross_margin=("gross_margin", "sum"),
            avg_profit_score=("profit_score", "mean"),
            avg_cts=("total_cost_to_serve", "mean"),
        )
        .round(2)
        .sort_values("avg_predicted_ltv", ascending=False)
    )
    qsummary["action"] = qsummary.index.map(QUADRANT_ACTION)

    # --- Segment × quadrant cross-tab ---
    seg_quad = (
        base.groupby(["segment", "quadrant"])
        .agg(customers=("customer_id", "count"),
             avg_profit_score=("profit_score", "mean"),
             total_margin=("gross_margin", "sum"))
        .round(2)
        .reset_index()
    )

    return ProfitMapResult(
        customer_map=base.sort_values("profit_score", ascending=False),
        quadrant_summary=qsummary,
        segment_profit_summary=seg_quad,
    )


def top_targets(
    profit_map: ProfitMapResult,
    quadrant: str = "INVEST",
    n: int = 100,
) -> pd.DataFrame:
    """
    Return top-N customers in a given quadrant, ranked by profit_score.
    Typical use: pull INVEST customers for a win-back / full-price migration campaign.
    """
    return (
        profit_map.customer_map[profit_map.customer_map["quadrant"] == quadrant]
        .sort_values("profit_score", ascending=False)
        .head(n)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build customer profit map")
    parser.add_argument("--ltv", required=True, help="LTV predictions CSV")
    parser.add_argument("--cts", required=True, help="Cost-to-serve CSV")
    parser.add_argument("--rfm", required=True, help="RFM scores CSV")
    parser.add_argument("--output", default="outputs/profit_map.csv")
    args = parser.parse_args()

    ltv = pd.read_csv(args.ltv)
    cts = pd.read_csv(args.cts)
    rfm = pd.read_csv(args.rfm)

    result = build_profit_map(ltv, cts, rfm)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.customer_map.to_csv(args.output, index=False)
    print("\n=== Quadrant Summary ===")
    print(result.quadrant_summary.to_string())
    print("\n=== Segment × Quadrant ===")
    print(result.segment_profit_summary.to_string())
