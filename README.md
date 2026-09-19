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
9. **Live Simulation** (full 6-store × 3-SKU network, running forward day-by-day — see below)
10. Solution Architecture (the Sprint 2 design write-up, embedded in the app)

## Live Simulation tab

This is the most advanced piece of the prototype: a day-by-day simulation that runs the whole
demand → forecast → reorder → restock loop live, across all 6 stores and all 3 SKUs, starting
the day after the real historical data ends.

- Each simulated day's demand is generated from patterns learned from the real history (recent
  average sales, day-of-week seasonality, promotion effect size) plus random daily variation —
  it is **not** real or predicted future sales, purely a plausible synthetic continuation used to
  demonstrate the decision logic in motion.
- The exact same forecasting (`forecasting.py`), reorder formula and risk classification
  (`inventory_analysis.py`) used in the historical tabs are reused here, so the two parts of the
  app stay consistent.
- When the system recommends a reorder, it is automatically "placed" and arrives after that SKU's
  lead time — so you can watch stock deplete, a reorder trigger, and the restock arrive on
  schedule, all live.
- Controls: **Next Day** (manual step), **Reset Simulation**, and an **Auto-play** toggle with a
  speed slider.
- Two KPIs are shown side by side on purpose: the **Risk-flag rate** (a leading, forward-looking
  warning based on whether on-hand + on-order stock covers a full lead-time cycle under the 20%
  safety-stock policy) and the **actual demand-unmet rate** (a lagging, measured outcome — whether
  that day's real demand was actually fulfilled). In testing, the risk flag fires on ~99% of
  combo-days while actual unmet demand sits around ~13% — a genuine finding worth discussing with
  your lecturer: with only a 20% safety margin and 1–2 day lead times, this reorder policy keeps
  inventory lean enough that the leading-indicator flag fires far more often than real shortages
  occur. A natural Sprint 3 question: would a higher safety-stock percentage reduce false-positive
  risk flags, at the cost of holding more stock?

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
├── app.py                  # Streamlit UI — all dashboard sections, incl. Live Simulation tab
├── data_utils.py            # Loading, validation, cleaning
├── forecasting.py           # Baseline moving-average model + optional ML comparison model
├── inventory_analysis.py    # Movement analysis, lead-time, expiry, reorder, risk logic
├── simulation.py            # Live day-by-day simulation engine (full store network)
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
