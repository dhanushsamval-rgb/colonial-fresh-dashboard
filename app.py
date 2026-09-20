"""
app.py
------
AI-Based Demand Forecasting and Inventory Management System - Colonial Fresh
MGT5ELS Assessment 3 - Sprint 2 (Conceptualisation & Design) prototype.

Run with:  streamlit run app.py

This is a Sprint 2 functional DEMONSTRATION of the proposed design, built on
a fully SIMULATED dataset (2024-2025). It is not connected to any live
Colonial Fresh system. No business-improvement results are claimed - only
the mechanics of the proposed decision-support logic are demonstrated.
"""

import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_utils import load_and_prepare, get_stores, get_products, get_as_of_date
from forecasting import baseline_moving_average_forecast, linear_trend_forecast
from inventory_analysis import (
    movement_analysis,
    build_risk_table,
    build_full_snapshot_table,
    RECENT_WINDOW_DAYS,
)
import simulation

st.set_page_config(page_title="Colonial Fresh | Demand & Inventory Prototype", layout="wide")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
try:
    df, load_messages = load_and_prepare()
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()

STORES = get_stores(df)
PRODUCTS = get_products(df)
AS_OF_DATE = get_as_of_date(df)

# ---------------------------------------------------------------------------
# Header / disclaimer banner
# ---------------------------------------------------------------------------
st.title("🥑 Colonial Fresh — AI-Based Demand Forecasting & Inventory Management")
st.caption("MGT5ELS Assessment 3 · Sprint 2 Prototype — Conceptualisation & Design")

st.info(
    f"**Prototype notice:** This dashboard runs on a fully **simulated** dataset "
    f"({df['Date'].min().date()} to {df['Date'].max().date()}) built for academic purposes. "
    f"It is **not** real Colonial Fresh data and is **not** connected to any live system. "
    f"The most recent date in the dataset ({AS_OF_DATE.date()}) is treated as **'today'** "
    f"for all current-inventory, lead-time and expiry calculations, to demonstrate how a "
    f"real-time dashboard would behave. Live system integration is proposed as future work "
    f"beyond this Sprint 2 design.",
    icon="ℹ️",
)

if load_messages:
    with st.expander("Data validation & cleaning log"):
        for m in load_messages:
            st.write("•", m)

# ---------------------------------------------------------------------------
# Scope: this dashboard always shows ALL stores and ALL SKUs together (no
# filtering), so it can be presented end-to-end without pausing to change
# dropdowns. The only control is the forecast horizon, since that's a genuine
# "what-if" parameter rather than something that hides data.
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")
forecast_period = st.sidebar.selectbox("Forecast period (days)", [7, 14, 30], index=0)
st.sidebar.caption("Every tab below always shows **all 6 stores × all 3 SKUs** together — nothing is filtered or hidden.")

stores_selected = STORES
product_ids_selected = PRODUCTS["Product_ID"].tolist()
filtered_df = df
products_selected_df = PRODUCTS

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_movement, tab_forecast, tab_inventory, tab_leadtime, \
    tab_expiry, tab_reorder, tab_risk, tab_live, tab_architecture = st.tabs(
        [
            "1️⃣ Overview", "2️⃣ Movement Analysis", "3️⃣ Demand Forecasting",
            "4️⃣ Inventory Monitoring", "5️⃣ Lead-Time Tracking", "6️⃣ Expiry Tracking",
            "7️⃣ Reorder Recommendation", "8️⃣ Risk Dashboard", "🔴 Live Simulation",
            "9️⃣ Solution Architecture",
        ]
    )

# Pre-compute the risk table and the full all-store x all-SKU snapshot once (reused across tabs)
risk_table = build_risk_table(df, stores_selected, products_selected_df, AS_OF_DATE, forecast_period)
full_table = build_full_snapshot_table(df, STORES, PRODUCTS, AS_OF_DATE, forecast_period)


def _risk_style(val):
    colors = {"Normal": "background-color:#e6f4ea", "Stockout Risk": "background-color:#fce8e6",
              "Overstock Risk": "background-color:#fef3e2", "Expiry Risk": "background-color:#f3e8fd"}
    return colors.get(val, "")

# ===========================================================================
# TAB 1: OVERVIEW DASHBOARD
# ===========================================================================
with tab_overview:
    st.subheader("Overview Dashboard")
    st.caption(f"All 6 stores × all 3 SKUs · forecast period = {forecast_period} days · as of {AS_OF_DATE.date()}.")

    total_current_inventory = risk_table["Current_Stock"].sum()
    total_forecast_demand = risk_table["Forecast_Period_Demand"].sum()
    total_incoming = filtered_df[filtered_df["Date"] == AS_OF_DATE]["Incoming_Stock"].sum()
    recent_wastage = filtered_df[
        filtered_df["Date"] > AS_OF_DATE - pd.Timedelta(days=RECENT_WINDOW_DAYS)
    ]["Wastage"].sum()
    stockout_rows_recent = filtered_df[
        (filtered_df["Date"] > AS_OF_DATE - pd.Timedelta(days=RECENT_WINDOW_DAYS))
        & (filtered_df["Stockout"] == "Yes")
    ]
    products_attention = (risk_table["Risk"] != "Normal").sum()
    total_recommended_order = risk_table["Recommended_Order"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Current Inventory (units)", f"{total_current_inventory:,.0f}")
    c2.metric(f"Forecasted Demand (next {forecast_period}d)", f"{total_forecast_demand:,.0f}")
    c3.metric("Incoming Stock (today)", f"{total_incoming:,.0f}")
    c4.metric(f"Wastage (last {RECENT_WINDOW_DAYS}d)", f"{recent_wastage:,.0f}")

    c5, c6, c7 = st.columns(3)
    c5.metric(f"Stockout Occurrences (last {RECENT_WINDOW_DAYS}d)", f"{len(stockout_rows_recent):,}")
    c6.metric("Store × SKU Combos Needing Attention", f"{products_attention} / {len(risk_table)}")
    c7.metric("Total Recommended Reorder Qty", f"{total_recommended_order:,.0f} units")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Products / Stores Requiring Attention**")
        attention_df = risk_table[risk_table["Risk"] != "Normal"][
            ["Store", "SKU", "Product", "Risk", "Current_Stock", "Recommended_Order"]
        ].sort_values("Risk")
        if attention_df.empty:
            st.success("No stockout, overstock or expiry risks detected in the current scope.")
        else:
            st.dataframe(attention_df, use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("**Risk Mix (current scope)**")
        risk_counts = risk_table["Risk"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Count"]
        fig = px.pie(risk_counts, names="Risk", values="Count", hole=0.45,
                     color="Risk",
                     color_discrete_map={"Normal": "#2ca02c", "Stockout Risk": "#d62728",
                                          "Overstock Risk": "#ff7f0e", "Expiry Risk": "#9467bd"})
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB 2: PRODUCT MOVEMENT ANALYSIS
# ===========================================================================
with tab_movement:
    st.subheader("Product Movement Analysis")
    st.write(
        "Movement class is not assumed - it is **calculated** from average daily "
        "`Units_Sold` across the full 2024-2025 history for each SKU."
    )

    mv = movement_analysis(df)
    st.dataframe(
        mv.rename(columns={
            "Avg_Daily_Units_Sold": "Avg Daily Units Sold",
            "Max_Daily_Units_Sold": "Max Daily Units Sold",
            "Min_Daily_Units_Sold": "Min Daily Units Sold",
            "Calculated_Movement_Class": "Calculated Movement Class",
        }),
        use_container_width=True, hide_index=True,
    )

    fig = px.bar(
        mv, x="Product_Name", y="Avg_Daily_Units_Sold", color="Calculated_Movement_Class",
        text_auto=".1f", title="Average Daily Units Sold by Product (2024-2025 full history)",
        labels={"Avg_Daily_Units_Sold": "Avg Daily Units Sold", "Product_Name": "Product"},
        color_discrete_map={"Fast-moving": "#2ca02c", "Moderate-moving": "#ff7f0e", "Slow-moving": "#d62728"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Bananas are confirmed **fast-moving**, Apples **moderate-moving** and "
        "Strawberries **slow-moving** based on calculated average daily sales, "
        "consistent with the three-SKU baseline required by the assessment brief."
    )

# ===========================================================================
# TAB 3: DEMAND FORECASTING — all stores, all SKUs
# ===========================================================================
with tab_forecast:
    st.subheader("Demand Forecasting — All Stores × All SKUs")
    st.write(
        "**Baseline model (primary):** 7-day moving average of historical `Units_Sold`, "
        "projected forward as a flat daily rate. `Simulated_True_Demand` is **not** used "
        "as a model input, to avoid data leakage — it is reserved for later evaluation only. "
        "The chart below aggregates all 6 stores together for each SKU, so all three "
        "products are visible on one screen."
    )

    show_trend = st.checkbox(
        "Also show scikit-learn Linear Regression trend model (comparison only — not claimed superior)",
        value=False,
    )

    sku_colors = {"SKU001": "#2ca02c", "SKU002": "#ff7f0e", "SKU003": "#9467bd"}
    fig = go.Figure()
    for _, prod in PRODUCTS.iterrows():
        sku = prod["Product_ID"]
        # Network-wide historical: sum Units_Sold across all 6 stores, per day, for this SKU
        hist_net = (
            df[df["Product_ID"] == sku].groupby("Date")["Units_Sold"].sum().reset_index().tail(60)
        )
        fig.add_trace(go.Scatter(x=hist_net["Date"], y=hist_net["Units_Sold"], mode="lines",
                                  name=f"{prod['Product_Name']} — Historical (network)",
                                  line=dict(color=sku_colors.get(sku), width=2)))

        # Network-wide forecast: sum each store's baseline forecast for this SKU
        fc_dates, fc_totals = None, None
        for store in STORES:
            fc = baseline_moving_average_forecast(df, store, sku, AS_OF_DATE, forecast_period)
            if fc_dates is None:
                fc_dates = fc["forecast_dates"]
                fc_totals = list(fc["forecast_values"])
            else:
                fc_totals = [a + b for a, b in zip(fc_totals, fc["forecast_values"])]
        fig.add_trace(go.Scatter(x=fc_dates, y=fc_totals, mode="lines+markers",
                                  name=f"{prod['Product_Name']} — Forecast (network)",
                                  line=dict(color=sku_colors.get(sku), dash="dash")))

        if show_trend:
            tr_dates, tr_totals = None, None
            for store in STORES:
                tr = linear_trend_forecast(df, store, sku, AS_OF_DATE, forecast_period)
                if not tr["forecast_dates"]:
                    continue
                if tr_dates is None:
                    tr_dates = tr["forecast_dates"]
                    tr_totals = list(tr["forecast_values"])
                else:
                    tr_totals = [a + b for a, b in zip(tr_totals, tr["forecast_values"])]
            if tr_dates:
                fig.add_trace(go.Scatter(x=tr_dates, y=tr_totals, mode="lines",
                                          name=f"{prod['Product_Name']} — Linear Trend (comparison)",
                                          line=dict(color=sku_colors.get(sku), dash="dot", width=1)))

    fig.update_layout(title="Network-Wide Historical Sales vs Forecast (all stores combined, per SKU)",
                       xaxis_title="Date", yaxis_title="Units Sold (all 6 stores)",
                       legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Forecast summary — every store × SKU**")
    fc_display = full_table[["Store", "Product", "SKU", "Avg_Daily_Forecast", "Forecast_Period_Demand"]].rename(
        columns={"Avg_Daily_Forecast": "Avg Daily Forecast", "Forecast_Period_Demand": f"Forecast (next {forecast_period}d)"}
    )
    st.dataframe(fc_display, use_container_width=True, hide_index=True)

    totals_by_sku = full_table.groupby("Product")["Forecast_Period_Demand"].sum().reset_index()
    cols = st.columns(len(totals_by_sku))
    for col, (_, row) in zip(cols, totals_by_sku.iterrows()):
        col.metric(f"{row['Product']} — Network Total (next {forecast_period}d)", f"{row['Forecast_Period_Demand']:,.0f} units")

    if show_trend:
        st.caption(
            "Linear trend lines are shown for comparison only — no back-testing has been performed "
            "in this Sprint 2 prototype, so no accuracy claim is made about either model."
        )

# ===========================================================================
# TAB 4: INVENTORY MONITORING — all stores, all SKUs
# ===========================================================================
with tab_inventory:
    st.subheader("Inventory Monitoring — All Stores × All SKUs")
    st.caption(f"As of {AS_OF_DATE.date()} (the most recent date in the dataset).")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Opening Stock", f"{full_table['Opening_Stock'].sum():,.0f}")
    k2.metric("Total Incoming Today", f"{full_table['Incoming_Stock_Today'].sum():,.0f}")
    k3.metric("Total Units Sold Today", f"{full_table['Units_Sold_Today'].sum():,.0f}")
    k4.metric("Total Closing Stock", f"{full_table['Current_Stock'].sum():,.0f}")

    inv_display = full_table[[
        "Store", "Product", "SKU", "Opening_Stock", "Incoming_Stock_Today", "Units_Sold_Today",
        "Current_Stock", "Wastage_Today", "Stockout_Today", "Risk",
    ]].rename(columns={
        "Opening_Stock": "Opening Stock", "Incoming_Stock_Today": "Incoming Stock",
        "Units_Sold_Today": "Units Sold", "Current_Stock": "Closing Stock",
        "Wastage_Today": "Wastage", "Stockout_Today": "Stockout",
    })
    st.dataframe(inv_display.style.map(_risk_style, subset=["Risk"]), use_container_width=True, hide_index=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(full_table, x="Store", y="Current_Stock", color="Product", barmode="group",
                     title="Closing Stock by Store, split by SKU", labels={"Current_Stock": "Closing Stock"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(full_table, x="Store", y="Wastage_Today", color="Product", barmode="group",
                     title="Today's Wastage by Store, split by SKU", labels={"Wastage_Today": "Wastage"})
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB 5: LEAD-TIME TRACKING — all stores, all SKUs
# ===========================================================================
with tab_leadtime:
    st.subheader("Lead-Time Tracking — All Stores × All SKUs")
    st.caption("Incoming stock shown here is a SIMULATED ESTIMATE (average stock received over the "
               "last Lead-Time-Days days), since the dataset does not record pending purchase orders explicitly.")

    lt_display = full_table[[
        "Store", "Product", "SKU", "Lead_Time_Days", "Forecast_During_Lead_Time",
        "Current_Stock", "Incoming_Stock_Estimate",
    ]].rename(columns={
        "Lead_Time_Days": "Supplier Lead Time (days)", "Forecast_During_Lead_Time": "Forecasted Demand During Lead Time",
        "Current_Stock": "Current Stock", "Incoming_Stock_Estimate": "Incoming Stock (est.)",
    })
    st.dataframe(lt_display, use_container_width=True, hide_index=True)
    st.caption(f"As of {AS_OF_DATE.date()} · Forecast period = {forecast_period} days.")

    fig = px.bar(full_table, x="Store", y=["Current_Stock", "Incoming_Stock_Estimate", "Forecast_During_Lead_Time"],
                 barmode="group", facet_col="Product",
                 title="Current Stock vs Forecasted Demand During Lead Time, by Store & SKU",
                 labels={"value": "Units", "variable": "Metric"})
    st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB 6: EXPIRY TRACKING — all stores, all SKUs
# ===========================================================================
with tab_expiry:
    st.subheader("Expiry Tracking — All Stores × All SKUs")
    st.warning(
        "**Simulated/prototype values.** This dataset is not connected to Colonial Fresh's live "
        "operational systems. Days Remaining reflects the shelf life of the most recently received "
        "stock lot as of the dataset's latest date — a simplification appropriate for this Sprint 2 design.",
        icon="⚠️",
    )

    exp_display = full_table.copy()
    exp_display["Expiry Risk"] = exp_display["Days_To_Expiry"].apply(
        lambda d: "Expiry Risk" if d <= 2 else ("Watch" if d <= 4 else "OK")
    )
    exp_display = exp_display[[
        "Store", "Product", "SKU", "Shelf_Life_Days", "Expiry_Date", "Days_To_Expiry", "Expiry Risk",
    ]].rename(columns={"Shelf_Life_Days": "Shelf Life (days)", "Expiry_Date": "Expiry Date", "Days_To_Expiry": "Days Remaining"})

    def _expiry_style(val):
        colors = {"Expiry Risk": "background-color:#f3e8fd", "Watch": "background-color:#fef3e2", "OK": "background-color:#e6f4ea"}
        return colors.get(val, "")

    st.dataframe(exp_display.style.map(_expiry_style, subset=["Expiry Risk"]), use_container_width=True, hide_index=True)
    st.caption(f"As of {AS_OF_DATE.date()}")

# ===========================================================================
# TAB 7: DEMAND-BASED REORDER RECOMMENDATION — all stores, all SKUs
# ===========================================================================
with tab_reorder:
    st.subheader("Demand-Based Reorder Recommendation — All Stores × All SKUs")

    st.markdown("**Formula:** `Recommended Order = Forecasted Demand During Lead Time + Safety Stock − Current Stock − Incoming Stock`")
    st.markdown("**Safety Stock = 20% of Forecasted Demand During Lead Time**")
    st.caption(
        "The system recommends an order whenever available inventory (plus what's already inbound) "
        "is below the expected demand requirement during the replenishment period."
    )

    total_recommended = full_table["Recommended_Order"].sum()
    combos_needing_order = (full_table["Recommended_Order"] > 0).sum()
    m1, m2 = st.columns(2)
    m1.metric("Total Recommended Reorder (network)", f"{total_recommended:,.0f} units")
    m2.metric("Store × SKU combos needing an order", f"{combos_needing_order} / {len(full_table)}")

    reorder_display = full_table[[
        "Store", "Product", "SKU", "Forecast_During_Lead_Time", "Safety_Stock",
        "Current_Stock", "Incoming_Stock_Estimate", "Raw_Calculation", "Recommended_Order", "Risk",
    ]].rename(columns={
        "Forecast_During_Lead_Time": "Forecast During Lead Time", "Safety_Stock": "Safety Stock (20%)",
        "Current_Stock": "Current Stock", "Incoming_Stock_Estimate": "Incoming Stock (est.)",
        "Raw_Calculation": "Raw Calculation", "Recommended_Order": "Recommended Order",
    })
    st.dataframe(reorder_display.style.map(_risk_style, subset=["Risk"]), use_container_width=True, hide_index=True)

    if combos_needing_order > 0:
        st.success(
            f"{combos_needing_order} store × SKU combination(s) currently need a reorder — "
            f"see the highlighted rows above for the full step-by-step calculation."
        )
    else:
        st.success("No reorders are currently recommended anywhere in the network.")

# ===========================================================================
# TAB 8: RISK DASHBOARD
# ===========================================================================
with tab_risk:
    st.subheader("Risk Dashboard")
    st.caption(f"Dynamically calculated for all 6 stores × 3 SKUs · as of {AS_OF_DATE.date()} · "
               f"forecast period = {forecast_period} days.")

    display_cols = ["Store", "SKU", "Product", "Movement", "Current_Stock",
                     "Forecast_Period_Demand", "Lead_Time_Days", "Days_To_Expiry", "Risk", "Recommended_Order"]
    display_df = risk_table[display_cols].rename(columns={
        "Current_Stock": "Current Stock", "Forecast_Period_Demand": "Forecast",
        "Lead_Time_Days": "Lead Time (d)", "Days_To_Expiry": "Expiry (days left)",
        "Recommended_Order": "Recommended Order",
    })

    st.dataframe(
        display_df.style.map(_risk_style, subset=["Risk"]),
        use_container_width=True, hide_index=True,
    )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        wastage_by_sku = filtered_df.groupby("Product_Name")["Wastage"].sum().reset_index()
        fig = px.bar(wastage_by_sku, x="Product_Name", y="Wastage", title="Wastage by SKU (full history, current scope)",
                     labels={"Product_Name": "Product"}, color="Product_Name")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        stockout_counts = filtered_df[filtered_df["Stockout"] == "Yes"].groupby("Product_Name").size().reset_index(name="Stockout Days")
        fig = px.bar(stockout_counts, x="Product_Name", y="Stockout Days", title="Stockout Occurrences by SKU (full history, current scope)",
                     labels={"Product_Name": "Product"}, color="Product_Name")
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB: LIVE SIMULATION (full network — all 6 stores x 3 SKUs)
# ===========================================================================
with tab_live:
    st.subheader("Live Simulation — Full Store Network")
    st.info(
        "This simulates ongoing daily grocery operations **forward** from the end of the "
        "real historical dataset (i.e. from " + str(AS_OF_DATE.date()) + " onward). Each simulated "
        "day's demand is synthetically generated from patterns learned from the real 2024-2025 "
        "history for each Store+SKU (recent average daily sales, day-of-week seasonality, "
        "promotion effect size) plus random day-to-day variation — **it is not real sales data**. "
        "The same 7-day moving-average forecast, lead-time-aware reorder formula and risk "
        "classification used elsewhere in this app run automatically every simulated day, so you "
        "can watch demand-based reordering actually trigger and restocks actually arrive.",
        icon="🔴",
    )

    if "sim_state" not in st.session_state:
        st.session_state.sim_state = simulation.init_simulation_state(df, STORES, PRODUCTS, AS_OF_DATE, seed=None)
    if "sim_autoplay" not in st.session_state:
        st.session_state.sim_autoplay = False
    if "sim_speed" not in st.session_state:
        st.session_state.sim_speed = 1.5

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1.4, 1])
    with ctrl1:
        next_day_clicked = st.button("▶ Next Day", use_container_width=True)
    with ctrl2:
        reset_clicked = st.button("🔁 Reset Simulation", use_container_width=True)
    with ctrl3:
        st.session_state.sim_speed = st.slider("Auto-play speed (seconds/day)", 0.3, 4.0, st.session_state.sim_speed, 0.1)
    with ctrl4:
        st.session_state.sim_autoplay = st.toggle("Auto-play", value=st.session_state.sim_autoplay)

    if reset_clicked:
        st.session_state.sim_state = simulation.init_simulation_state(df, STORES, PRODUCTS, AS_OF_DATE, seed=None)
        st.session_state.sim_autoplay = False
        st.rerun()

    if next_day_clicked:
        st.session_state.sim_state = simulation.simulate_next_day(st.session_state.sim_state)

    sim_state = st.session_state.sim_state
    sim_log = sim_state["log"]

    st.markdown(f"**Simulated date:** {sim_state['sim_date'].date()}  ·  "
                f"**Days simulated:** {sim_state['days_simulated']}")

    if sim_log.empty:
        st.warning("No simulated days yet — click **Next Day** or turn on **Auto-play** to begin.")
    else:
        latest_date = sim_log["Date"].max()
        today_log = sim_log[sim_log["Date"] == latest_date]

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Network Stock (units)", f"{today_log['Closing_Stock'].sum():,.0f}")
        k2.metric("Units Sold Today", f"{today_log['Units_Sold'].sum():,.0f}")
        k3.metric("Wastage Today", f"{today_log['Wastage'].sum():,.0f}")
        k4.metric("Orders Placed Today", f"{(today_log['Recommended_Order'] > 0).sum()} combos")
        k5.metric("Units Reordered Today", f"{today_log['Recommended_Order'].sum():,.0f}")

        st.divider()
        st.markdown("**Today's snapshot — all stores × SKUs**")

        show_cols = ["Store", "Product_Name", "Opening_Stock", "Incoming_Stock", "Units_Sold",
                     "Closing_Stock", "Wastage", "Stockout", "Recommended_Order", "Risk"]
        st.dataframe(
            today_log[show_cols].rename(columns={"Product_Name": "Product"}).style.map(_risk_style, subset=["Risk"]),
            use_container_width=True, hide_index=True,
        )

        st.caption(
            "**Reading this table:** 'Risk' is a leading, forward-looking flag based on whether "
            "on-hand + already-ordered stock covers a full lead-time cycle under the 20% safety-stock "
            "policy. 'Stockout' shows whether that SPECIFIC day's actual demand was fully met. With a "
            "lean 20% safety margin and short 1-2 day lead times, 'Stockout Risk' can flag often as an "
            "early warning even while actual demand keeps getting met most days — the two metrics below "
            "make that gap visible, which is a genuine discussion point for Sprint 3 (e.g. would a higher "
            "safety-stock % reduce false-positive risk flags at the cost of holding more stock?)."
        )

        c1, c2 = st.columns(2)
        with c1:
            actual_stockout_rate = (sim_log["Stockout"] == "Yes").mean()
            st.metric("Actual demand-unmet rate (measured, all days simulated)", f"{actual_stockout_rate:.1%}")
        with c2:
            risk_flag_rate = (sim_log["Risk"] != "Normal").mean()
            st.metric("Risk-flag rate (leading indicator, all days simulated)", f"{risk_flag_rate:.1%}")

        st.divider()
        st.markdown("**Network trends across the simulation**")
        network_daily = sim_log.groupby("Date").agg(
            Total_Units_Sold=("Units_Sold", "sum"),
            Total_Forecast=("Avg_Daily_Forecast", "sum"),
            Total_Closing_Stock=("Closing_Stock", "sum"),
            Total_Wastage=("Wastage", "sum"),
        ).reset_index()

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=network_daily["Date"], y=network_daily["Total_Units_Sold"],
                                   name="Actual Units Sold (network)", line=dict(color="#1f77b4")))
        fig1.add_trace(go.Scatter(x=network_daily["Date"], y=network_daily["Total_Forecast"],
                                   name="Forecast (sum of daily rates, network)", line=dict(color="#2ca02c", dash="dash")))
        fig1.update_layout(title="Network-Wide Actual Demand vs Forecast (simulated days)",
                           xaxis_title="Date", yaxis_title="Units")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=network_daily["Date"], y=network_daily["Total_Closing_Stock"],
                                   name="Total Network Closing Stock", line=dict(color="#1f77b4")))
        fig2.update_layout(title="Total Network Stock On Hand Over Time", xaxis_title="Date", yaxis_title="Units")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Stock level by SKU, all stores combined**")
        stock_by_sku = sim_log.groupby(["Date", "Product_Name"])["Closing_Stock"].sum().reset_index()
        fig3 = px.line(stock_by_sku, x="Date", y="Closing_Stock", color="Product_Name",
                        title="Closing Stock Over Time by SKU (summed across all 6 stores)",
                        labels={"Closing_Stock": "Closing Stock", "Product_Name": "Product"})
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("**Reorders placed by store, all SKUs combined**")
        orders_by_store = sim_log.groupby(["Date", "Store"])["Recommended_Order"].sum().reset_index()
        orders_by_store = orders_by_store[orders_by_store["Recommended_Order"] > 0]
        if orders_by_store.empty:
            st.caption("No reorders placed yet.")
        else:
            fig4 = px.bar(orders_by_store, x="Date", y="Recommended_Order", color="Store",
                          title="Units Reordered per Day, by Store", labels={"Recommended_Order": "Units Reordered"})
            st.plotly_chart(fig4, use_container_width=True)

    if st.session_state.sim_autoplay:
        time.sleep(st.session_state.sim_speed)
        st.session_state.sim_state = simulation.simulate_next_day(st.session_state.sim_state)
        st.rerun()

# ===========================================================================
# TAB: SOLUTION ARCHITECTURE (Sprint 2 design explanation)
# ===========================================================================
with tab_architecture:
    st.subheader("Solution Architecture — Sprint 2 Design")
    st.markdown(
        """
This Sprint 2 prototype demonstrates the proposed end-to-end design. Sprint 3 will formally
develop and demonstrate the working prototype in full; this sprint focuses on **conceptualisation,
design and technology integration**.

**Data & processing pipeline:**

```
Excel Dataset (Colonial Fresh Mock Data)
        ↓
Data Cleaning & Preparation   (pandas: date conversion, type correction, missing-value handling, validation)
        ↓
Product Movement Analysis     (average daily Units_Sold per SKU → Fast/Moderate/Slow classification)
        ↓
Demand Forecasting            (7-day moving-average baseline; optional scikit-learn Linear Regression trend for comparison)
        ↓
Inventory Analysis            (Opening/Incoming/Sold/Closing/Wastage/Stockout monitoring)
        ↓
Lead-Time & Expiry Tracking   (supplier lead time, forecasted demand during lead time, shelf-life/expiry)
        ↓
Risk Detection                 (Normal / Stockout Risk / Overstock Risk / Expiry Risk classification)
        ↓
Demand-Based Reorder Calculation  (Forecast during lead time + 20% safety stock − current − incoming)
        ↓
Streamlit Dashboard             (Overview, Movement, Forecasting, Inventory, Lead-Time, Expiry, Reorder, Risk views)
```

**Technology stack:** Python, Pandas (data handling), Scikit-learn (optional comparison forecasting
model), Streamlit (interactive dashboard UI), Plotly (interactive charts).

**Why this design fits Sprint 2:** it is simple and fully explainable (baseline moving-average
forecast + transparent reorder formula), directly implements all six points raised in lecturer
feedback (three-SKU baseline, demand forecasting, stock dashboard, lead-time tracking, expiry
tracking, demand-based reorder), and is modular — each pipeline stage is a separate, testable
Python module (`data_utils.py`, `forecasting.py`, `inventory_analysis.py`), so it can be extended
in Sprint 3 without rebuilding the design.

**Potential benefits (proposed, not yet measured):** more consistent reorder decisions, earlier
visibility of stockout/overstock/expiry risk, and a single dashboard combining data that is
currently split across separate manual processes. These are proposed benefits for future
validation in Sprint 3 — no business-improvement results are claimed from this Sprint 2 prototype.

**Future development (beyond Sprint 2):** live POS/ERP integration in place of the Excel file,
batch-level (FEFO) expiry tracking, back-tested forecast accuracy comparison between the baseline
and ML models, and automated purchase-order generation from the reorder recommendation.
        """
    )

st.divider()
st.caption(
    "Colonial Fresh AI-Based Demand Forecasting and Inventory Management System — "
    "Sprint 2 academic prototype. Simulated data only. Not a live business system."
)
