# Colonial Fresh — AI-Based Demand Forecasting & Inventory Management System

**MGT5ELS Assessment 3 — Sprint 2 (Conceptualisation & Design) prototype**

A functional demonstration of the proposed design for a fresh-food retail demand-forecasting
and inventory-decision-support dashboard, built on a fully **simulated** 2024–2025 dataset for
three baseline SKUs (Bananas, Apples, Strawberries) across six Colonial Fresh stores.

> **Academic integrity note:** the dataset (`data/Colonial_Fresh_Mock_Data_2024_2025_3_SKUs.xlsx`)
> is entirely simulated for this assessment. It is not real Colonial Fresh data and this
> application is not connected to any live business system.

## What this Sprint 2 prototype covers

1. Overview Dashboard (KPIs + filters for store, product, forecast period)
2. Product Movement Analysis (data-driven fast/moderate/slow classification)
3. Demand Forecasting (7-day moving-average baseline; optional scikit-learn linear-trend comparison)
4. Inventory Monitoring (Opening/Incoming/Sold/Closing/Wastage/Stockout)
5. Lead-Time Tracking (forecast during lead time)
6. Expiry Tracking (shelf life / expiry date / days remaining — labelled as simulated)
7. Demand-Based Reorder Recommendation (transparent formula, shown step-by-step)
8. Risk Dashboard (Normal / Stockout Risk / Overstock Risk / Expiry Risk, all SKUs × stores)
9. Solution Architecture (the Sprint 2 design write-up, embedded in the app)

## How to run it

1. Install Python 3.10+.
2. From this folder, install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Launch the dashboard:
   ```
   streamlit run app.py
   ```
4. Your browser will open automatically at `http://localhost:8501`.

The Excel dataset is already included in `data/`, so no extra setup is needed. If you replace
it with an updated file, keep the same filename and sheet name (`Mock_Data`).

## Project structure

```
colonial_fresh_dashboard/
├── app.py                  # Streamlit UI — all 9 dashboard sections
├── data_utils.py            # Loading, validation, cleaning
├── forecasting.py           # Baseline moving-average model + optional ML comparison model
├── inventory_analysis.py    # Movement analysis, lead-time, expiry, reorder, risk logic
├── requirements.txt
├── README.md
└── data/
    └── Colonial_Fresh_Mock_Data_2024_2025_3_SKUs.xlsx
```

## Key design decisions (for your lecturer discussion)

- **Forecasting:** the primary model is a simple, explainable 7-day moving average (as required
  by the brief). An optional scikit-learn `LinearRegression` trend model is shown side-by-side
  purely for comparison — it is **not** claimed to be more accurate, since no back-testing has
  been done in this sprint. `Simulated_True_Demand` is never used as a model input, to avoid
  data leakage; it is reserved for future evaluation.
- **Reorder formula:** `Recommended Order = Forecasted Demand During Lead Time + Safety Stock
  − Current Stock − Incoming Stock`, with `Safety Stock = 20% of Forecasted Demand During Lead
  Time`, floored at 0. Shown step-by-step in the app.
- **"Real-time" framing:** the dashboard treats the most recent date in the dataset as "today"
  to demonstrate real-time-style behaviour. Live POS/ERP integration is proposed as Sprint 3+
  future work, not claimed as already built.
- **Incoming stock (pipeline) estimate:** the dataset doesn't track pending purchase orders, so
  incoming stock for lead-time/reorder purposes is estimated from recent historical receipts.
  This is disclosed in the app.

## Testing performed (Sprint 2 scope)

All three baseline SKUs were tested against at least one store each, confirming that:
forecasts differ by product, inventory values update per store/SKU, lead time is incorporated
into the reorder calculation, expiry information displays correctly, risk status is calculated
per combination, and reorder quantities are calculated dynamically (including cases where the
recommended order is greater than zero, e.g. Apples at Doncaster and The Pines, and Strawberries
at The Pines, under Stockout Risk).
