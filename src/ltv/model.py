"""
ML-based Customer Lifetime Value (LTV) prediction.

Architecture:
- Features: RFM scores + season cohort + category affinity + tenure
- Model: LightGBM regressor (handles skewed revenue distributions well)
- Target: 12-month forward net revenue (log-transformed)
- Output: predicted LTV at customer and segment level

Fashion-specific considerations:
- Season cohort captures first-purchase season (SS vs AW buyers behave differently)
- Category affinity flags full-price vs outlet vs drop preference
- Tenure is measured in seasons, not calendar months
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

try:
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:  # catches OSError from missing libomp as well as ImportError
    HAS_LGBM = False

CONFIG_PATH = Path(__file__).parents[2] / "config" / "ltv_config.yaml"
MODEL_PATH = Path(__file__).parents[2] / "outputs" / "ltv_model.lgb"


def _require_deps():
    if not HAS_DEPS:
        raise ImportError("Install scikit-learn: pip install scikit-learn")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(
    rfm_scores: pd.DataFrame,
    transactions: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Combine RFM scores with transaction-derived behavioural features.

    Parameters
    ----------
    rfm_scores   : output of src.rfm.segment.compute_rfm (scores DataFrame)
    transactions : raw transaction log with customer_id, order_date, revenue,
                   discount_rate, category (optional), is_drop (optional)
    snapshot_date: reference date

    Returns
    -------
    Feature matrix indexed by customer_id
    """
    df = transactions.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])

    # --- Tenure in days ---
    first_purchase = df.groupby("customer_id")["order_date"].min()
    tenure = (snapshot_date - first_purchase).dt.days.rename("tenure_days")

    # --- Season cohort of first purchase ---
    first_month = first_purchase.dt.month
    season_cohort = first_month.map(lambda m: "SS" if m in [2,3,4,5,6,7] else "AW").rename("first_season")

    # --- Category affinity (share of revenue per category) ---
    cat_features = pd.DataFrame(index=df["customer_id"].unique())
    if "category" in df.columns:
        cat_pivot = (
            df.groupby(["customer_id", "category"])["revenue"]
            .sum()
            .unstack(fill_value=0)
        )
        cat_total = cat_pivot.sum(axis=1).replace(0, np.nan)
        cat_share = cat_pivot.div(cat_total, axis=0).add_prefix("cat_share_")
        cat_features = cat_share

    # --- Purchase type mix ---
    if "discount_rate" in df.columns:
        outlet_threshold = 0.40
        outlet_rev = df[df["discount_rate"] >= outlet_threshold].groupby("customer_id")["revenue"].sum()
        total_rev = df.groupby("customer_id")["revenue"].sum()
        outlet_share = (outlet_rev / total_rev).fillna(0).rename("outlet_revenue_share")
    else:
        outlet_share = pd.Series(0.0, index=df["customer_id"].unique(), name="outlet_revenue_share")

    if "is_drop" in df.columns:
        drop_rev = df[df["is_drop"]].groupby("customer_id")["revenue"].sum()
        drop_share = (drop_rev / total_rev).fillna(0).rename("drop_revenue_share")
    else:
        drop_share = pd.Series(0.0, index=df["customer_id"].unique(), name="drop_revenue_share")

    # --- Average order value ---
    aov = (
        df.groupby("customer_id")
        .apply(lambda g: g["revenue"].sum() / g["order_date"].nunique(), include_groups=False)
        .rename("avg_order_value")
    )

    # --- Inter-purchase time (avg days between orders) ---
    def avg_ipt(g):
        dates = g["order_date"].sort_values().unique()
        if len(dates) < 2:
            return np.nan
        return np.diff(dates).astype("timedelta64[D]").astype(float).mean()

    ipt = df.groupby("customer_id").apply(avg_ipt, include_groups=False).rename("avg_days_between_orders")

    # --- Assemble ---
    features = pd.concat(
        [rfm_scores.set_index("customer_id")[["R", "F", "M", "rfm_score", "recency_days",
                                               "frequency", "monetary", "outlet_ratio"]],
         tenure, season_cohort, outlet_share, drop_share, aov, ipt, cat_features],
        axis=1,
    )

    # Encode season cohort
    features["first_season"] = (features["first_season"] == "SS").astype(int)

    return features.fillna(0)


# ---------------------------------------------------------------------------
# LTV model
# ---------------------------------------------------------------------------

@dataclass
class LTVResult:
    predictions: pd.DataFrame       # customer_id, predicted_ltv_12m, ltv_segment
    segment_ltv: pd.DataFrame        # LTV stats per RFM segment
    feature_importance: pd.DataFrame
    metrics: dict
    model: object = field(repr=False, default=None)


def train(
    features: pd.DataFrame,
    labels: pd.Series,          # 12-month forward net revenue per customer
    config: Optional[dict] = None,
) -> tuple:
    """
    Train LightGBM LTV regressor. Returns (model, metrics).
    labels index must align with features index.
    """
    _require_deps()

    cfg = config or {}

    X = features
    # Log-transform target to handle revenue skew; clip to avoid log(0)
    y = np.log1p(labels.reindex(X.index).fillna(0))

    # Use LightGBM when available (faster, better on large data); fall back to sklearn
    if HAS_LGBM:
        params = {
            "objective": "regression",
            "metric": "mae",
            "n_estimators": cfg.get("n_estimators", 400),
            "learning_rate": cfg.get("learning_rate", 0.05),
            "num_leaves": cfg.get("num_leaves", 31),
            "min_child_samples": cfg.get("min_child_samples", 10),
            "subsample": cfg.get("subsample", 0.8),
            "colsample_bytree": cfg.get("colsample_bytree", 0.8),
            "reg_alpha": cfg.get("reg_alpha", 0.1),
            "reg_lambda": cfg.get("reg_lambda", 0.1),
            "random_state": 42,
            "verbose": -1,
        }
        model = lgb.LGBMRegressor(**params)
    else:
        model = HistGradientBoostingRegressor(
            max_iter=cfg.get("n_estimators", 200),
            learning_rate=cfg.get("learning_rate", 0.05),
            max_leaf_nodes=cfg.get("num_leaves", 31),
            min_samples_leaf=cfg.get("min_child_samples", 10),
            l2_regularization=cfg.get("reg_lambda", 0.1),
            random_state=42,
        )

    cv_mae = -cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error").mean()

    model.fit(X, y)
    y_pred = model.predict(X)
    metrics = {
        "cv_mae_log": round(cv_mae, 4),
        "train_r2": round(r2_score(y, y_pred), 4),
        "train_mae_log": round(mean_absolute_error(y, y_pred), 4),
    }
    return model, metrics


def predict(
    features: pd.DataFrame,
    rfm_scores: pd.DataFrame,
    model,
    n_ltv_segments: int = 4,
) -> LTVResult:
    """
    Generate LTV predictions and segment customers into LTV tiers.

    LTV tiers (default 4):
        platinum  — top 10%
        gold      — next 20%
        silver    — next 30%
        bronze    — bottom 40%
    """
    _require_deps()

    log_pred = model.predict(features)
    ltv_pred = np.expm1(log_pred)  # reverse log1p

    preds = pd.DataFrame({
        "customer_id": features.index,
        "predicted_ltv_12m": ltv_pred,
    })

    # LTV tier assignment
    tier_labels = ["bronze", "silver", "gold", "platinum"]
    tier_quantiles = [0.0, 0.40, 0.70, 0.90, 1.0]
    preds["ltv_tier"] = pd.cut(
        preds["predicted_ltv_12m"],
        bins=preds["predicted_ltv_12m"].quantile(tier_quantiles).values,
        labels=tier_labels,
        include_lowest=True,
    ).astype(str)

    # Merge with RFM segment for cross-tab
    preds = preds.merge(
        rfm_scores[["customer_id", "segment"]],
        on="customer_id",
        how="left",
    )

    # Segment-level LTV summary
    segment_ltv = (
        preds.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            avg_ltv=("predicted_ltv_12m", "mean"),
            median_ltv=("predicted_ltv_12m", "median"),
            total_ltv=("predicted_ltv_12m", "sum"),
            p90_ltv=("predicted_ltv_12m", lambda x: x.quantile(0.90)),
        )
        .round(2)
        .sort_values("avg_ltv", ascending=False)
    )

    # Feature importance (HistGradientBoosting doesn't expose feature_importances_)
    if hasattr(model, "feature_importances_"):
        imp_values = model.feature_importances_
    else:
        imp_values = np.zeros(len(features.columns))

    importance = pd.DataFrame({
        "feature": features.columns,
        "importance": imp_values,
    }).sort_values("importance", ascending=False)

    return LTVResult(
        predictions=preds,
        segment_ltv=segment_ltv,
        feature_importance=importance,
        metrics={},
        model=model,
    )


def run_pipeline(
    rfm_scores: pd.DataFrame,
    transactions: pd.DataFrame,
    labels: pd.Series,
    snapshot_date: pd.Timestamp,
    config_path: Path = CONFIG_PATH,
) -> LTVResult:
    """
    End-to-end: build features → train → predict → return LTVResult.

    Parameters
    ----------
    rfm_scores   : from compute_rfm().scores
    transactions : raw transaction log
    labels       : 12-month forward net revenue per customer_id (pd.Series)
    snapshot_date: reference date
    """
    _require_deps()

    cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f).get("ltv", {})

    features = build_features(rfm_scores, transactions, snapshot_date)

    # Align labels to feature index (customers with no forward revenue → 0)
    aligned_labels = labels.reindex(features.index).fillna(0)

    model, metrics = train(features, aligned_labels, cfg)
    result = predict(features, rfm_scores, model)
    result.metrics = metrics

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and predict LTV")
    parser.add_argument("--transactions", required=True)
    parser.add_argument("--rfm-scores", required=True)
    parser.add_argument("--labels", required=True, help="CSV: customer_id, forward_revenue_12m")
    parser.add_argument("--snapshot-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output", default="outputs/ltv_predictions.csv")
    args = parser.parse_args()

    txn = pd.read_csv(args.transactions, parse_dates=["order_date"])
    rfm = pd.read_csv(args.rfm_scores)
    lbl = pd.read_csv(args.labels).set_index("customer_id")["forward_revenue_12m"]
    snap = pd.Timestamp(args.snapshot_date)

    result = run_pipeline(rfm, txn, lbl, snap)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(args.output, index=False)
    print("\n=== Segment LTV ===")
    print(result.segment_ltv.to_string())
    print("\n=== Top Features ===")
    print(result.feature_importance.head(10).to_string())
    print("\n=== Metrics ===")
    print(result.metrics)
