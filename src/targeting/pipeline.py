"""
End-to-end targeting pipeline.

Orchestrates the full analytical stack in a single call:
  1. RFM segmentation       (src.rfm.segment)
  2. LTV prediction         (src.ltv.model)
  3. Cost-to-serve          (src.cost_to_serve.calculator)
  4. Profit map             (src.profit_map.builder)
  5. Recommendations        (src.recommendations.engine)

Outputs
-------
  - customer_master : one row per customer with all dimensions
  - campaign_briefs : one brief per quadrant with targets + recommended actions
  - segment_report  : executive summary across RFM × profit quadrant
  - recommendations : top-N product recommendations per customer (optional)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.rfm.segment import compute_rfm
from src.ltv.model import run_pipeline as run_ltv
from src.cost_to_serve.calculator import compute_cost_to_serve
from src.profit_map.builder import build_profit_map, top_targets
from src.recommendations.engine import RecommendationEngine


OUTPUT_DIR = Path(__file__).parents[2] / "outputs"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TargetingResult:
    customer_master: pd.DataFrame          # all dimensions per customer
    campaign_briefs: dict[str, pd.DataFrame]  # quadrant → brief DataFrame
    segment_report: pd.DataFrame           # exec summary
    recommendations: pd.DataFrame          # long-format product recs (may be empty)
    metadata: dict = field(default_factory=dict)

    def save(self, output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
        """Write all outputs to CSV files. Returns paths written."""
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        p = output_dir / "customer_master.csv"
        self.customer_master.to_csv(p, index=False)
        paths["customer_master"] = p

        p = output_dir / "segment_report.csv"
        self.segment_report.to_csv(p, index=False)
        paths["segment_report"] = p

        for quadrant, brief in self.campaign_briefs.items():
            p = output_dir / f"campaign_brief_{quadrant.lower()}.csv"
            brief.to_csv(p, index=False)
            paths[f"campaign_brief_{quadrant}"] = p

        if not self.recommendations.empty:
            p = output_dir / "recommendations.csv"
            self.recommendations.to_csv(p, index=False)
            paths["recommendations"] = p

        return paths

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("CUSTOMER VALUE PLATFORM — TARGETING SUMMARY")
        print("=" * 60)
        print(f"Total customers analysed : {len(self.customer_master):,}")
        print(f"Snapshot date            : {self.metadata.get('snapshot_date', 'N/A')}")
        print()
        print("── Quadrant Distribution ──────────────────────────────")
        for q in ["PROTECT", "INVEST", "HARVEST", "REDUCE"]:
            n = (self.customer_master["quadrant"] == q).sum()
            pct = n / len(self.customer_master) * 100
            bar = "█" * int(pct / 2)
            print(f"  {q:<10} {n:>5,}  ({pct:4.1f}%)  {bar}")
        print()
        print("── Segment Report ──────────────────────────────────────")
        print(self.segment_report.to_string(index=False))
        print()


# ---------------------------------------------------------------------------
# Campaign brief builder
# ---------------------------------------------------------------------------

BRIEF_COLUMNS = [
    "customer_id", "segment", "quadrant", "profit_score",
    "predicted_ltv_12m", "ltv_tier", "gross_revenue",
    "gross_margin", "cts_pct_revenue", "rfm_score", "R", "F", "M",
    "outlet_ratio", "profitability_band",
]

QUADRANT_BRIEF_METADATA = {
    "PROTECT": {
        "objective": "Maximise retention and lifetime value",
        "tactics": [
            "Enrol in VIP loyalty tier with exclusive early access",
            "Personalised recommendations: new season drops first",
            "Priority customer service (dedicated channel)",
            "Suppress discount / promotional comms to protect margin",
        ],
        "kpis": ["Retention rate", "Repeat purchase rate", "NPS", "Share of wallet"],
    },
    "INVEST": {
        "objective": "Migrate from outlet/discount dependency to full-price",
        "tactics": [
            "Full-price product recommendations (suppress outlet SKUs)",
            "Return reduction programme (size guide, fit AI, virtual try-on)",
            "Service deflection: chatbot for order queries, reduce contacts",
            "Upsell to limited drops to build full-price purchase habit",
        ],
        "kpis": ["Full-price revenue share", "Return rate", "Cost-to-serve reduction", "LTV growth"],
    },
    "HARVEST": {
        "objective": "Maintain margin with minimal incremental spend",
        "tactics": [
            "Automated email/SMS flows (no manual intervention)",
            "High-margin product recommendations (accessories, footwear)",
            "Annual reactivation campaign if recency > 90 days",
            "Exclude from paid acquisition lookalike audiences",
        ],
        "kpis": ["Gross margin per customer", "Automation rate", "CAC"],
    },
    "REDUCE": {
        "objective": "Capture residual value; suppress unprofitable spend",
        "tactics": [
            "Outlet-only communications (clearance, end-of-season sale)",
            "Remove from paid social lookalike seed audiences",
            "Suppress from new-season launch comms",
            "Win-back test: one re-engagement offer, if no response → churn",
        ],
        "kpis": ["Churn rate", "Outlet revenue recovered", "CAC suppressed"],
    },
}


def build_campaign_briefs(
    customer_master: pd.DataFrame,
    n_targets_per_quadrant: int = 500,
) -> dict[str, pd.DataFrame]:
    """
    Build one campaign brief per quadrant: top-N customers by profit_score
    with quadrant objective, tactics, and KPIs embedded as metadata columns.
    """
    briefs = {}
    available_cols = [c for c in BRIEF_COLUMNS if c in customer_master.columns]

    for quadrant, meta in QUADRANT_BRIEF_METADATA.items():
        cohort = (
            customer_master[customer_master["quadrant"] == quadrant]
            [available_cols]
            .sort_values("profit_score", ascending=False)
            .head(n_targets_per_quadrant)
            .copy()
        )
        cohort["campaign_objective"] = meta["objective"]
        cohort["campaign_tactics"] = " | ".join(meta["tactics"])
        cohort["campaign_kpis"] = " | ".join(meta["kpis"])
        briefs[quadrant] = cohort

    return briefs


# ---------------------------------------------------------------------------
# Segment report
# ---------------------------------------------------------------------------

def build_segment_report(customer_master: pd.DataFrame) -> pd.DataFrame:
    """Executive P&L summary: RFM segment × profit quadrant."""
    agg_spec: dict = {
        "customers": ("customer_id", "count"),
        "avg_profit_score": ("profit_score", "mean"),
        "avg_ltv": ("predicted_ltv_12m", "mean"),
        "total_revenue": ("gross_revenue", "sum"),
        "total_margin": ("gross_margin", "sum"),
    }
    for col, agg_name in [("cts_pct_revenue", "avg_cts_pct"), ("outlet_ratio", "avg_outlet_ratio")]:
        if col in customer_master.columns:
            agg_spec[agg_name] = (col, "mean")

    report = (
        customer_master.groupby(["segment", "quadrant"])
        .agg(**agg_spec)
        .round(2)
        .reset_index()
    )
    report["margin_pct"] = (
        report["total_margin"] / report["total_revenue"].replace(0, np.nan) * 100
    ).round(1)
    report["revenue_share_pct"] = (
        report["total_revenue"] / report["total_revenue"].sum() * 100
    ).round(1)
    return report.sort_values(["avg_profit_score"], ascending=False)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    transactions: pd.DataFrame,
    catalog: pd.DataFrame | None = None,
    ltv_labels: pd.Series | None = None,
    service_contacts: pd.DataFrame | None = None,
    snapshot_date: pd.Timestamp | None = None,
    n_recommendations: int = 10,
    n_targets_per_quadrant: int = 500,
    season_filter: list[str] | None = None,
    rfm_config_path: Optional[Path] = None,
    ltv_config_path: Optional[Path] = None,
    cost_config_path: Optional[Path] = None,
) -> TargetingResult:
    """
    Full customer value targeting pipeline.

    Parameters
    ----------
    transactions         : raw transaction log (see module docstrings for schema)
    catalog              : product catalog (required for recommendations)
    ltv_labels           : 12-month forward revenue per customer (for supervised LTV).
                           If None, uses historical revenue as a proxy label.
    service_contacts     : optional customer service contact log
    snapshot_date        : reference date; defaults to max(order_date) + 1 day
    n_recommendations    : top-N recommendations per customer (0 to skip)
    n_targets_per_quadrant: customers to include per campaign brief
    season_filter        : restrict recommendations to these seasons e.g. ["SS"]
    """
    txn = transactions.copy()
    txn["order_date"] = pd.to_datetime(txn["order_date"])

    if snapshot_date is None:
        snapshot_date = txn["order_date"].max() + pd.Timedelta(days=1)

    # ── Step 1: RFM ─────────────────────────────────────────────────────────
    kwargs = {}
    if rfm_config_path:
        kwargs["config_path"] = rfm_config_path
    rfm_result = compute_rfm(txn, snapshot_date=snapshot_date, **kwargs)
    rfm_scores = rfm_result.scores

    # ── Step 2: LTV ──────────────────────────────────────────────────────────
    if ltv_labels is None:
        # Proxy: historical net revenue as stand-in for forward revenue
        if "discount_rate" in txn.columns:
            net_rev = (txn["revenue"] * (1 - txn["discount_rate"].fillna(0)))
        else:
            net_rev = txn["revenue"]
        ltv_labels = net_rev.groupby(txn["customer_id"]).sum()

    ltv_kwargs = {}
    if ltv_config_path:
        ltv_kwargs["config_path"] = ltv_config_path
    ltv_result = run_ltv(rfm_scores, txn, ltv_labels, snapshot_date, **ltv_kwargs)

    # ── Step 3: Cost-to-serve ────────────────────────────────────────────────
    cts_kwargs = {}
    if cost_config_path:
        cts_kwargs["config_path"] = cost_config_path
    cts_result = compute_cost_to_serve(
        txn, rfm_scores, snapshot_date,
        service_contacts=service_contacts,
        **cts_kwargs,
    )

    # ── Step 4: Profit map ───────────────────────────────────────────────────
    profit_result = build_profit_map(
        ltv_result.predictions,
        cts_result.customer_cts,
        rfm_scores,
    )

    # ── Step 5: Recommendations ──────────────────────────────────────────────
    recs_df = pd.DataFrame()
    if n_recommendations > 0 and catalog is not None and "product_id" in txn.columns:
        engine = RecommendationEngine.build(txn, catalog)
        customer_ids = profit_result.customer_map["customer_id"].tolist()
        recs_df = engine.recommend_for_segment(
            customer_ids,
            profit_result,
            n=n_recommendations,
            season_filter=season_filter,
        )

    # ── Assemble customer master ─────────────────────────────────────────────
    master = profit_result.customer_map.copy()

    # ── Campaign briefs ──────────────────────────────────────────────────────
    briefs = build_campaign_briefs(master, n_targets_per_quadrant)

    # ── Segment report ───────────────────────────────────────────────────────
    report = build_segment_report(master)

    return TargetingResult(
        customer_master=master,
        campaign_briefs=briefs,
        segment_report=report,
        recommendations=recs_df,
        metadata={
            "snapshot_date": str(snapshot_date.date()),
            "n_customers": len(master),
            "n_transactions": len(txn),
            "ltv_metrics": ltv_result.metrics,
        },
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run full customer value targeting pipeline")
    parser.add_argument("--transactions", required=True)
    parser.add_argument("--catalog", default=None)
    parser.add_argument("--snapshot-date", default=None)
    parser.add_argument("--season-filter", nargs="*", default=None, help="e.g. SS AW")
    parser.add_argument("--n-recs", type=int, default=10)
    parser.add_argument("--n-targets", type=int, default=500)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    txn = pd.read_csv(args.transactions, parse_dates=["order_date"])
    cat = pd.read_csv(args.catalog) if args.catalog else None
    snap = pd.Timestamp(args.snapshot_date) if args.snapshot_date else None

    result = run(
        txn, catalog=cat, snapshot_date=snap,
        n_recommendations=args.n_recs,
        n_targets_per_quadrant=args.n_targets,
        season_filter=args.season_filter,
    )
    result.print_summary()
    paths = result.save(Path(args.output_dir))
    print("\nOutputs written:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
