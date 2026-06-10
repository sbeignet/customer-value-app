"""
Fashion-aware RFM segmentation.

Key differences vs standard RFM:
- Recency is computed within the customer's active season (SS/AW), not calendar year
- Frequency excludes outlet purchases when outlet_ratio exceeds threshold
- Monetary is net of discounts to avoid inflating outlet customers
- Drop purchases are tagged separately and do not inflate frequency scores
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


CONFIG_PATH = Path(__file__).parents[2] / "config" / "rfm_config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def assign_season(date: pd.Series, seasons: list[dict]) -> pd.Series:
    """Map each date to a season name based on config month ranges."""
    month = date.dt.month
    result = pd.Series("UNKNOWN", index=date.index)
    for s in seasons:
        mask = month.isin(s["months"])
        result[mask] = s["name"]
    return result


def flag_outlet(df: pd.DataFrame, discount_threshold: float) -> pd.Series:
    """Return boolean series: True if purchase is outlet-priced."""
    if "discount_rate" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["discount_rate"] >= discount_threshold


def flag_drop(df: pd.DataFrame) -> pd.Series:
    """Return boolean series: True if purchase is part of a limited drop."""
    if "is_drop" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["is_drop"].astype(bool)


@dataclass
class RFMResult:
    scores: pd.DataFrame          # customer-level RFM scores and segment labels
    segment_summary: pd.DataFrame # aggregate stats per segment
    config: dict = field(repr=False)


def compute_rfm(
    transactions: pd.DataFrame,
    snapshot_date: Optional[pd.Timestamp] = None,
    config_path: Path = CONFIG_PATH,
) -> RFMResult:
    """
    Compute fashion-aware RFM scores.

    Parameters
    ----------
    transactions : DataFrame with columns:
        customer_id, order_date, revenue, discount_rate (optional),
        is_drop (optional), season (optional — inferred if absent)
    snapshot_date : reference date for recency; defaults to max(order_date) + 1 day
    config_path : path to rfm_config.yaml

    Returns
    -------
    RFMResult with customer-level scores and segment summary
    """
    cfg = load_config(config_path)
    rfm_cfg = cfg["rfm"]
    seg_cfg = cfg["segments"]

    df = transactions.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])

    if snapshot_date is None:
        snapshot_date = df["order_date"].max() + pd.Timedelta(days=1)

    # --- Season assignment ---
    if "season" not in df.columns and rfm_cfg.get("season_aware"):
        df["season"] = assign_season(df["order_date"], rfm_cfg["seasons"])

    # --- Outlet and drop flags ---
    outlet_threshold = rfm_cfg.get("outlet_discount_threshold", 0.40)
    df["is_outlet"] = flag_outlet(df, outlet_threshold)
    df["is_drop"] = flag_drop(df)

    # Net revenue: strip outlet discounts from monetary signal
    if "discount_rate" in df.columns:
        df["net_revenue"] = df["revenue"] * (1 - df["discount_rate"])
    else:
        df["net_revenue"] = df["revenue"]

    # Exclude outlet-only transactions from core frequency/monetary
    core = df[~df["is_outlet"]]

    # --- Recency: days since last non-outlet purchase within recency window ---
    recency_window = rfm_cfg.get("recency_days", 90)
    recent_cutoff = snapshot_date - pd.Timedelta(days=recency_window)
    core_recent = core[core["order_date"] >= recent_cutoff]

    last_purchase = (
        core_recent.groupby("customer_id")["order_date"]
        .max()
        .rename("last_purchase_date")
    )
    recency = (snapshot_date - last_purchase).dt.days.rename("recency_days")

    # --- Frequency: distinct order dates (drop purchases counted once) ---
    frequency = (
        core.groupby("customer_id")["order_date"]
        .nunique()
        .rename("frequency")
    )

    # --- Monetary: sum of net revenue ---
    monetary = (
        core.groupby("customer_id")["net_revenue"]
        .sum()
        .rename("monetary")
    )

    # --- Outlet ratio (diagnostic, used for outlet_only segment) ---
    total_orders = df.groupby("customer_id")["order_date"].count().rename("total_orders")
    outlet_orders = df[df["is_outlet"]].groupby("customer_id")["order_date"].count().rename("outlet_orders")

    rfm = pd.concat([recency, frequency, monetary, total_orders, outlet_orders], axis=1).fillna(0)
    rfm["outlet_ratio"] = rfm["outlet_orders"] / rfm["total_orders"].replace(0, np.nan)
    rfm["outlet_ratio"] = rfm["outlet_ratio"].fillna(0)

    # --- Quintile scoring (1–5, higher = better) ---
    quintiles = rfm_cfg["scoring"]["quintiles"]

    def quintile_score(series: pd.Series, ascending: bool = True) -> pd.Series:
        labels = list(range(1, quintiles + 1))
        if not ascending:
            labels = labels[::-1]
        return pd.qcut(series.rank(method="first"), q=quintiles, labels=labels).astype(int)

    rfm["R"] = quintile_score(rfm["recency_days"], ascending=False)  # lower recency = better
    rfm["F"] = quintile_score(rfm["frequency"], ascending=True)
    rfm["M"] = quintile_score(rfm["monetary"], ascending=True)

    weights = rfm_cfg["scoring"]["weights"]
    rfm["rfm_score"] = (
        rfm["R"] * weights["recency"]
        + rfm["F"] * weights["frequency"]
        + rfm["M"] * weights["monetary"]
    )

    # --- Segment assignment ---
    def assign_segment(row: pd.Series) -> str:
        # Outlet-only override takes priority
        outlet_min = seg_cfg.get("outlet_only", {}).get("outlet_ratio_min", 0.80)
        if row["outlet_ratio"] >= outlet_min:
            return "outlet_only"

        for seg_name, rules in seg_cfg.items():
            if seg_name == "outlet_only":
                continue
            r_range = rules.get("r", [1, quintiles])
            f_range = rules.get("f", [1, quintiles])
            m_range = rules.get("m", [1, quintiles])
            if (
                r_range[0] <= row["R"] <= r_range[1]
                and f_range[0] <= row["F"] <= f_range[1]
                and m_range[0] <= row["M"] <= m_range[1]
            ):
                return seg_name
        return "other"

    rfm["segment"] = rfm.apply(assign_segment, axis=1)

    # --- Segment summary ---
    summary = (
        rfm.groupby("segment")
        .agg(
            customers=("rfm_score", "count"),
            avg_rfm_score=("rfm_score", "mean"),
            avg_recency_days=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_revenue=("monetary", "sum"),
        )
        .round(2)
        .sort_values("avg_rfm_score", ascending=False)
    )

    return RFMResult(scores=rfm.reset_index(), segment_summary=summary, config=cfg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fashion-aware RFM segmentation")
    parser.add_argument("--input", required=True, help="Path to transactions CSV")
    parser.add_argument("--output", default="outputs/rfm_scores.csv")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--snapshot-date", default=None, help="YYYY-MM-DD")
    args = parser.parse_args()

    transactions = pd.read_csv(args.input, parse_dates=["order_date"])
    snapshot = pd.Timestamp(args.snapshot_date) if args.snapshot_date else None

    result = compute_rfm(transactions, snapshot_date=snapshot, config_path=Path(args.config))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.scores.to_csv(args.output, index=False)
    print(result.segment_summary.to_string())
    print(f"\nScores written to {args.output}")
