# Customer Value AI App

A fashion-industry-specific customer value analytics platform built on **Data, AI, and Orchestration** to increase business performance by 30%+.

## Core Capabilities

| Module | Description |
|--------|-------------|
| **RFM Engine** | Fashion-specific RFM (season, drop, campaign-aware) |
| **LTV Model** | ML-based Lifetime Value at customer & segment level |
| **Cost-to-Serve** | Per-customer and per-segment cost attribution |
| **Profit Map** | Profitability scoring across the customer base |
| **Targeting** | High-LTV segment identification and activation |

## Why "Simple RFM Doesn't Cut It"

Standard RFM ignores fashion-specific buying patterns:
- **Seasonal drops** compress purchase cycles
- **Outlet behaviour** distorts frequency signals
- **Campaign attribution** inflates recency scores
- **Cross-season loyalty** is invisible to generic models

This platform adjusts for all of the above.

## Project Structure

```
├── data/               # Raw and processed data assets
├── notebooks/          # Exploratory analysis and model prototypes
├── src/
│   ├── rfm/            # Fashion-aware RFM segmentation
│   ├── ltv/            # ML LTV models
│   ├── cost_to_serve/  # Cost attribution logic
│   ├── profit_map/     # Profit map computation
│   └── targeting/      # Segment targeting and activation
├── outputs/            # Reports, exports, dashboards
└── templates/          # Prompt and analysis templates
```

## Business Model Coverage

- Fast fashion (high-frequency, low AOV)
- Season-based release (Spring/Summer, Autumn/Winter)
- Outlet / discount-driven sale cycles
- Drop-based release (limited edition, hype-driven)

## Getting Started

```bash
pip install -r requirements.txt
python src/rfm/segment.py --config config/rfm_config.yaml
```
