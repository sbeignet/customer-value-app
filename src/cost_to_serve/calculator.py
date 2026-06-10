"""
Cost-to-serve (CtS) attribution per customer and segment.

Cost drivers modelled:
    1. Fulfilment cost       — picking, packing, shipping per order
    2. Return cost           — reverse logistics + restocking per returned item
    3. Customer service cost — service touchpoints (contacts, escalations)
    4. Acquisition cost      — amortised CAC over customer tenure
    5. Discount cost         — margin given away via promotions / outlet pricing

Fashion-specific adjustments:
    - Returns are significantly higher in fashion (avg 25-40%) and vary by
      category (dresses >> accessories) and channel (online >> in-store)
    - Outlet orders carry lower fulfilment cost (warehouse clearance) but
      higher markdown cost
    - Drop launches generate spike service contacts that must be attributed
      to the cohort that bought the drop

Output: per-customer CtS breakdown + segment-level P&L bridge
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from dataclasses import dataclass

CONFIG_PATH = Path(__file__).parents[2] / "config" / "cost_config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Individual cost driver calculators
# ---------------------------------------------------------------------------

def fulfilment_cost(
    transactions: pd.DataFrame,
    cost_per_order: float,
    outlet_discount: float = 0.30,
) -> pd.Series:
    """
    Shipping + pick/pack cost per customer.
    Outlet orders get a discount (warehouse clearance is cheaper to ship).
    """
    df = transactions.copy()
    outlet_mask = df.get("is_outlet", pd.Series(False, index=df.index))

    df["_fc"] = cost_per_order
    df.loc[outlet_mask, "_fc"] = cost_per_order * (1 - outlet_discount)

    return df.groupby("customer_id")["_fc"].sum().rename("cost_fulfilment")


def return_cost(
    transactions: pd.DataFrame,
    base_cost_per_return: float,
    category_multipliers: dict[str, float] | None = None,
) -> pd.Series:
    """
    Reverse logistics + restocking cost.
    Requires a 'returns' column (number of items returned per order).
    Falls back to estimated return rate per category if returns not recorded.
    """
    df = transactions.copy()
    cat_mult = category_multipliers or {}

    if "returns" in df.columns:
        df["_items_returned"] = df["returns"]
    elif "category" in df.columns:
        # Estimated return rates by category (fashion industry benchmarks)
        default_rates = {
            "dresses": 0.38, "tops": 0.28, "outerwear": 0.22,
            "bottoms": 0.25, "accessories": 0.10, "footwear": 0.32,
        }
        df["_return_rate"] = df["category"].map(default_rates).fillna(0.25)
        df["_items_returned"] = df.get("quantity", pd.Series(1, index=df.index)) * df["_return_rate"]
    else:
        df["_items_returned"] = df.get("quantity", pd.Series(1, index=df.index)) * 0.25

    # Apply category cost multiplier
    if "category" in df.columns:
        df["_mult"] = df["category"].map(cat_mult).fillna(1.0)
    else:
        df["_mult"] = 1.0

    df["_rc"] = df["_items_returned"] * base_cost_per_return * df["_mult"]
    return df.groupby("customer_id")["_rc"].sum().rename("cost_returns")


def service_cost(
    transactions: pd.DataFrame,
    service_contacts: pd.DataFrame | None,
    cost_per_contact: float,
    drop_spike_factor: float = 2.5,
) -> pd.Series:
    """
    Customer service cost. Uses actual contact log if available,
    otherwise estimates from order count + drop purchase indicator.

    service_contacts schema: customer_id, n_contacts
    """
    if service_contacts is not None and len(service_contacts) > 0:
        sc = service_contacts.groupby("customer_id")["n_contacts"].sum()
        return (sc * cost_per_contact).rename("cost_service")

    # Estimate: base 0.3 contacts per order, x2.5 for drop orders
    df = transactions.copy()
    df["_contacts"] = 0.3
    if "is_drop" in df.columns:
        df.loc[df["is_drop"].astype(bool), "_contacts"] = 0.3 * drop_spike_factor

    contacts = df.groupby("customer_id")["_contacts"].sum()
    return (contacts * cost_per_contact).rename("cost_service")


def acquisition_cost(
    transactions: pd.DataFrame,
    cac_by_channel: dict[str, float],
    default_cac: float,
    snapshot_date: pd.Timestamp,
    amortisation_months: int = 12,
) -> pd.Series:
    """
    Amortised customer acquisition cost (CAC).
    CAC is spread over amortisation_months from first purchase date.
    Customers beyond that window have fully amortised CAC (cost = 0).
    """
    df = transactions.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])

    first_purchase = df.groupby("customer_id")["order_date"].min()
    months_active = ((snapshot_date - first_purchase).dt.days / 30.44).clip(lower=0)

    channel = df.groupby("customer_id")["channel"].first() if "channel" in df.columns else pd.Series()
    cac = channel.map(cac_by_channel).fillna(default_cac) if len(channel) > 0 else pd.Series(default_cac, index=first_purchase.index)

    # Pro-rate: remaining months to amortise / total amortisation period
    remaining_fraction = ((amortisation_months - months_active) / amortisation_months).clip(0, 1)
    amortised_cac = (cac * remaining_fraction).rename("cost_acquisition")
    return amortised_cac.reindex(first_purchase.index).fillna(0)


def discount_cost(transactions: pd.DataFrame) -> pd.Series:
    """
    Total margin given away via discounts (revenue × discount_rate).
    This is the opportunity cost of promotional pricing.
    """
    if "discount_rate" not in transactions.columns:
        return pd.Series(0.0, index=transactions["customer_id"].unique(), name="cost_discounts")

    df = transactions.copy()
    df["_dc"] = df["revenue"] * df["discount_rate"].fillna(0)
    return df.groupby("customer_id")["_dc"].sum().rename("cost_discounts")


# ---------------------------------------------------------------------------
# Main calculator
# ---------------------------------------------------------------------------

@dataclass
class CostToServeResult:
    customer_cts: pd.DataFrame    # per-customer cost breakdown
    segment_cts: pd.DataFrame     # segment-level cost summary
    config: dict


def compute_cost_to_serve(
    transactions: pd.DataFrame,
    rfm_scores: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    service_contacts: pd.DataFrame | None = None,
    config_path: Path = CONFIG_PATH,
) -> CostToServeResult:
    """
    Compute full cost-to-serve per customer and aggregate by RFM segment.

    Parameters
    ----------
    transactions   : raw transaction log
    rfm_scores     : from compute_rfm().scores (for segment labels)
    snapshot_date  : reference date for CAC amortisation
    service_contacts : optional DataFrame with customer_id, n_contacts
    config_path    : path to cost_config.yaml
    """
    cfg = load_config(config_path)
    c = cfg["costs"]

    # Outlet flag (needed for fulfilment discount)
    df = transactions.copy()
    if "is_outlet" not in df.columns and "discount_rate" in df.columns:
        df["is_outlet"] = df["discount_rate"] >= cfg.get("outlet_discount_threshold", 0.40)

    fc = fulfilment_cost(df, c["fulfilment_per_order"], c.get("outlet_fulfilment_discount", 0.30))
    rc = return_cost(df, c["return_per_item"], c.get("category_return_multipliers"))
    sc = service_cost(df, service_contacts, c["service_per_contact"], c.get("drop_spike_factor", 2.5))
    ac = acquisition_cost(df, c.get("cac_by_channel", {}), c["default_cac"],
                          snapshot_date, c.get("cac_amortisation_months", 12))
    dc = discount_cost(df)

    cts = pd.concat([fc, rc, sc, ac, dc], axis=1).fillna(0)
    cts["total_cost_to_serve"] = cts[["cost_fulfilment", "cost_returns", "cost_service",
                                      "cost_acquisition", "cost_discounts"]].sum(axis=1)
    cts = cts.reset_index()  # brings customer_id back as column

    # Merge revenue and segment
    revenue = df.groupby("customer_id")["revenue"].sum().reset_index().rename(columns={"revenue": "gross_revenue"})
    cts = cts.merge(revenue, on="customer_id", how="left")
    cts = cts.merge(rfm_scores[["customer_id", "segment"]], on="customer_id", how="left")
    cts["gross_margin"] = cts["gross_revenue"] - cts["total_cost_to_serve"]
    cts["cts_pct_revenue"] = (cts["total_cost_to_serve"] / cts["gross_revenue"].replace(0, np.nan) * 100).round(1)

    # Segment summary
    seg = (
        cts.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            total_revenue=("gross_revenue", "sum"),
            total_cts=("total_cost_to_serve", "sum"),
            total_margin=("gross_margin", "sum"),
            avg_cts_per_customer=("total_cost_to_serve", "mean"),
            avg_cts_pct=("cts_pct_revenue", "mean"),
            avg_cost_fulfilment=("cost_fulfilment", "mean"),
            avg_cost_returns=("cost_returns", "mean"),
            avg_cost_service=("cost_service", "mean"),
            avg_cost_discounts=("cost_discounts", "mean"),
        )
        .round(2)
        .sort_values("total_margin", ascending=False)
    )
    seg["margin_pct"] = (seg["total_margin"] / seg["total_revenue"].replace(0, np.nan) * 100).round(1)

    return CostToServeResult(customer_cts=cts, segment_cts=seg, config=cfg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute cost-to-serve per customer")
    parser.add_argument("--transactions", required=True)
    parser.add_argument("--rfm-scores", required=True)
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--output", default="outputs/cost_to_serve.csv")
    args = parser.parse_args()

    txn = pd.read_csv(args.transactions, parse_dates=["order_date"])
    rfm = pd.read_csv(args.rfm_scores)
    snap = pd.Timestamp(args.snapshot_date)

    result = compute_cost_to_serve(txn, rfm, snap)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.customer_cts.to_csv(args.output, index=False)
    print("\n=== Segment Cost-to-Serve ===")
    print(result.segment_cts.to_string())
