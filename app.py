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

from data_utils import load_and_prepare, get_stores, get_products, get_as_of_date, filter_data
from forecasting import baseline_moving_average_forecast, linear_trend_forecast
from inventory_analysis import (
    movement_analysis,
    current_snapshot,
    lead_time_metrics,
    expiry_metrics,
    calculate_reorder,
    classify_risk,
    build_risk_table,
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
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

store_filter = st.sidebar.multiselect("Store", STORES, default=STORES)
product_filter = st.sidebar.multiselect(
    "Product", PRODUCTS["Product_ID"] + " – " + PRODUCTS["Product_Name"],
    default=list(PRODUCTS["Product_ID"] + " – " + PRODUCTS["Product_Name"]),
)
product_ids_selected = [p.split(" – ")[0] for p in product_filter] or PRODUCTS["Product_ID"].tolist()
stores_selected = store_filter or STORES

forecast_period = st.sidebar.selectbox("Forecast period (days)", [7, 14, 30], index=0)

st.sidebar.divider()
st.sidebar.caption("Detailed single-SKU sections (Forecasting, Inventory, Lead-Time, Expiry, Reorder) use the focus selectors below:")
focus_store = st.sidebar.selectbox(
    "Focus store",
    ["All Stores"] + STORES,
    index=0
)
focus_product_label = st.sidebar.selectbox(
    "Focus product", PRODUCTS["Product_ID"] + " – " + PRODUCTS["Product_Name"], index=0
)
focus_product_id = focus_product_label.split(" – ")[0]
focus_product_name = PRODUCTS.loc[PRODUCTS["Product_ID"] == focus_product_id, "Product_Name"].iloc[0]

filtered_df = filter_data(df, stores_selected, product_ids_selected)
products_selected_df = PRODUCTS[PRODUCTS["Product_ID"].isin(product_ids_selected)].reset_index(drop=True)

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

# Pre-compute the risk table once (used by Overview and Risk Dashboard tabs)
risk_table = build_risk_table(df, stores_selected, products_selected_df, AS_OF_DATE, forecast_period)

# ===========================================================================
# TAB 1: OVERVIEW DASHBOARD
# ===========================================================================
with tab_overview:
    st.subheader("Overview Dashboard")
    st.caption(f"Scope: {len(stores_selected)} store(s), {len(product_ids_selected)} product(s), "
               f"forecast period = {forecast_period} days, as of {AS_OF_DATE.date()}.")

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
# TAB 3: DEMAND FORECASTING
# ===========================================================================
with tab_forecast:
    st.subheader(f"Demand Forecasting — {focus_product_name} @ {focus_store}")
    st.write(
        "**Baseline model (primary):** 7-day moving average of historical `Units_Sold`, "
        "projected forward as a flat daily rate. `Simulated_True_Demand` is **not** used "
        "as a model input, to avoid data leakage — it is reserved for later evaluation only."
    )

    baseline = baseline_moving_average_forecast(df, focus_store, focus_product_id, AS_OF_DATE, forecast_period)
    trend = linear_trend_forecast(df, focus_store, focus_product_id, AS_OF_DATE, forecast_period)

    show_trend = st.checkbox(
        "Also show scikit-learn Linear Regression trend model (comparison only — not claimed superior)",
        value=False,
    )

    hist = baseline["historical"].tail(60)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Units_Sold"], mode="lines",
                              name="Historical Units Sold", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=baseline["forecast_dates"], y=baseline["forecast_values"], mode="lines+markers",
                              name=f"Baseline Forecast ({baseline.get('window_used',7)}-day MA)",
                              line=dict(color="#2ca02c", dash="dash")))
    if show_trend and trend["forecast_dates"]:
        fig.add_trace(go.Scatter(x=trend["forecast_dates"], y=trend["forecast_values"], mode="lines+markers",
                                  name="Linear Trend (scikit-learn) — comparison only",
                                  line=dict(color="#9467bd", dash="dot")))
    fig.update_layout(title="Historical Sales vs Forecast", xaxis_title="Date", yaxis_title="Units Sold",
                       legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Forecast period", f"{forecast_period} days")
    m2.metric("Avg daily forecast (baseline)", f"{baseline['avg_daily_rate']:.1f} units")
    m3.metric("Total forecasted demand", f"{baseline['avg_daily_rate']*forecast_period:.0f} units")

    with st.expander("Forecast values (baseline model)"):
        fc_df = pd.DataFrame({"Date": baseline["forecast_dates"], "Forecasted Units": baseline["forecast_values"]})
        st.dataframe(fc_df, use_container_width=True, hide_index=True)

    if show_trend:
        st.caption(
            f"Linear trend slope: {trend.get('trend_slope_per_day', 0):+.2f} units/day over the trailing "
            f"60 days. Shown for comparison only — no back-testing has been performed in this Sprint 2 "
            f"prototype, so no accuracy claim is made about either model."
        )

# ===========================================================================
# TAB 4: INVENTORY MONITORING
# ===========================================================================
with tab_inventory:
    st.subheader(f"Inventory Monitoring — {focus_product_name} @ {focus_store}")

    snap = current_snapshot(df, focus_store, focus_product_id, AS_OF_DATE)
    if not snap:
        st.warning("No data available for this store/product on the latest date.")
    else:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Opening Stock", snap["Opening_Stock"])
        c2.metric("Incoming Stock", snap["Incoming_Stock"])
        c3.metric("Units Sold", snap["Units_Sold"])
        c4.metric("Closing Stock", snap["Closing_Stock"])
        c5.metric("Wastage", snap["Wastage"])
        c6.metric("Stockout Today?", snap["Stockout"])

        baseline = baseline_moving_average_forecast(df, focus_store, focus_product_id, AS_OF_DATE, forecast_period)
        expected_demand = baseline["avg_daily_rate"] * forecast_period
        exp = expiry_metrics(df, focus_store, focus_product_id, AS_OF_DATE)
        # Risk uses lead-time demand (not period demand), consistent with the Risk Dashboard tab
        lt_tmp = lead_time_metrics(df, focus_store, focus_product_id, AS_OF_DATE, baseline["avg_daily_rate"])
        risk = classify_risk(lt_tmp["Current_Stock"], lt_tmp["Forecasted_Demand_During_Lead_Time"], exp.get("Days_Remaining", 999))

        st.divider()
        colx, coly = st.columns(2)
        with colx:
            st.metric(f"Available Stock vs Expected Demand (next {forecast_period}d)",
                      f"{snap['Closing_Stock']:.0f} vs {expected_demand:.0f}",
                      delta=f"{snap['Closing_Stock'] - expected_demand:+.0f} units")
        with coly:
            risk_color = {"Normal": "🟢", "Stockout Risk": "🔴", "Overstock Risk": "🟠", "Expiry Risk": "🟣"}
            st.metric("Inventory Status", f"{risk_color.get(risk,'')} {risk}")

        st.markdown("**Recent Inventory Trend (last 30 days)**")
        hist30 = df[(df["Store"] == focus_store) & (df["Product_ID"] == focus_product_id)
                    & (df["Date"] > AS_OF_DATE - pd.Timedelta(days=30))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist30["Date"], y=hist30["Closing_Stock"], name="Closing Stock", line=dict(color="#1f77b4")))
        fig.add_trace(go.Bar(x=hist30["Date"], y=hist30["Wastage"], name="Wastage", marker_color="#d62728", opacity=0.5, yaxis="y2"))
        fig.update_layout(
            yaxis=dict(title="Closing Stock"),
            yaxis2=dict(title="Wastage", overlaying="y", side="right"),
            title="Closing Stock & Wastage (last 30 days)",
        )
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB 5: LEAD-TIME TRACKING
# ===========================================================================
with tab_leadtime:
    st.subheader("Lead-Time Tracking")
    st.caption("Incoming stock shown here is a SIMULATED ESTIMATE (average stock received over the "
               "last Lead-Time-Days days), since the dataset does not record pending purchase orders explicitly.")

    lt_rows = []
    for _, prod in products_selected_df.iterrows():
        baseline = baseline_moving_average_forecast(df, focus_store, prod["Product_ID"], AS_OF_DATE, forecast_period)
        lt = lead_time_metrics(df, focus_store, prod["Product_ID"], AS_OF_DATE, baseline["avg_daily_rate"])
        lt_rows.append({
            "Product": prod["Product_Name"],
            "SKU": prod["Product_ID"],
            "Supplier Lead Time (days)": lt["Lead_Time_Days"],
            "Forecasted Demand During Lead Time": round(lt["Forecasted_Demand_During_Lead_Time"], 1),
            "Current Stock": lt["Current_Stock"],
            "Incoming Stock (est.)": round(lt["Incoming_Stock_Estimate"], 1),
        })
    lt_df = pd.DataFrame(lt_rows)
    st.dataframe(lt_df, use_container_width=True, hide_index=True)
    st.caption(f"Store: {focus_store} · Forecast period: {forecast_period} days · As of {AS_OF_DATE.date()}")

    fig = px.bar(lt_df, x="Product", y=["Current Stock", "Incoming Stock (est.)", "Forecasted Demand During Lead Time"],
                 barmode="group", title="Current Stock vs Forecasted Demand During Lead Time")
    st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB 6: EXPIRY TRACKING
# ===========================================================================
with tab_expiry:
    st.subheader("Expiry Tracking")
    st.warning(
        "**Simulated/prototype values.** This dataset is not connected to Colonial Fresh's live "
        "operational systems. Days Remaining reflects the shelf life of the most recently received "
        "stock lot as of the dataset's latest date — a simplification appropriate for this Sprint 2 design.",
        icon="⚠️",
    )

    exp_rows = []
    for _, prod in products_selected_df.iterrows():
        exp = expiry_metrics(df, focus_store, prod["Product_ID"], AS_OF_DATE)
        if not exp:
            continue
        days_rem = exp["Days_Remaining"]
        exp_risk = "Expiry Risk" if days_rem <= 2 else ("Watch" if days_rem <= 4 else "OK")
        exp_rows.append({
            "Product": prod["Product_Name"],
            "SKU": prod["Product_ID"],
            "Shelf Life (days)": exp["Shelf_Life_Days"],
            "Expiry Date": exp["Expiry_Date"].date(),
            "Days Remaining": days_rem,
            "Expiry Risk": exp_risk,
        })
    exp_df = pd.DataFrame(exp_rows)
    st.dataframe(exp_df, use_container_width=True, hide_index=True)
    st.caption(f"Store: {focus_store} · As of {AS_OF_DATE.date()}")

# ===========================================================================
# TAB 7: DEMAND-BASED REORDER RECOMMENDATION
# ===========================================================================
with tab_reorder:
    st.subheader(f"Demand-Based Reorder Recommendation — {focus_product_name} @ {focus_store}")

    baseline = baseline_moving_average_forecast(df, focus_store, focus_product_id, AS_OF_DATE, forecast_period)
    lt = lead_time_metrics(df, focus_store, focus_product_id, AS_OF_DATE, baseline["avg_daily_rate"])
    reorder = calculate_reorder(lt["Forecasted_Demand_During_Lead_Time"], lt["Current_Stock"], lt["Incoming_Stock_Estimate"])

    st.markdown("**Formula:** `Recommended Order = Forecasted Demand During Lead Time + Safety Stock − Current Stock − Incoming Stock`")
    st.markdown("**Safety Stock = 20% of Forecasted Demand During Lead Time**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecasted Demand During Lead Time", f"{lt['Forecasted_Demand_During_Lead_Time']:.1f}")
    c2.metric("Safety Stock (20%)", f"{reorder['Safety_Stock']:.1f}")
    c3.metric("Current Stock", f"{lt['Current_Stock']:.0f}")
    c4.metric("Incoming Stock (est.)", f"{lt['Incoming_Stock_Estimate']:.1f}")

    st.markdown(
        f"**Calculation:** {lt['Forecasted_Demand_During_Lead_Time']:.1f} + {reorder['Safety_Stock']:.1f} "
        f"− {lt['Current_Stock']:.0f} − {lt['Incoming_Stock_Estimate']:.1f} "
        f"= **{reorder['Raw_Calculation']:.1f}** → Recommended Order = **{reorder['Recommended_Order']:.0f} units**"
    )

    if reorder["Recommended_Order"] > 0:
        st.success(
            "The system recommends this order because available inventory is below the expected "
            "demand requirement during the replenishment period."
        )
    else:
        st.success(
            "No reorder is currently recommended — available inventory and incoming stock are "
            "sufficient to cover expected demand during the replenishment period."
        )

# ===========================================================================
# TAB 8: RISK DASHBOARD
# ===========================================================================
with tab_risk:
    st.subheader("Risk Dashboard")
    st.caption(f"Dynamically calculated for all selected stores × SKUs · as of {AS_OF_DATE.date()} · "
               f"forecast period = {forecast_period} days.")

    display_cols = ["Store", "SKU", "Product", "Movement", "Current_Stock",
                     "Forecast_Period_Demand", "Lead_Time_Days", "Days_To_Expiry", "Risk", "Recommended_Order"]
    display_df = risk_table[display_cols].rename(columns={
        "Current_Stock": "Current Stock", "Forecast_Period_Demand": "Forecast",
        "Lead_Time_Days": "Lead Time (d)", "Days_To_Expiry": "Expiry (days left)",
        "Recommended_Order": "Recommended Order",
    })

    def _risk_color(val):
        colors = {"Normal": "background-color:#e6f4ea", "Stockout Risk": "background-color:#fce8e6",
                  "Overstock Risk": "background-color:#fef3e2", "Expiry Risk": "background-color:#f3e8fd"}
        return colors.get(val, "")

    st.dataframe(
        display_df.style.map(_risk_color, subset=["Risk"]),
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

        def _risk_color_live(val):
            colors = {"Normal": "background-color:#e6f4ea", "Stockout Risk": "background-color:#fce8e6",
                      "Overstock Risk": "background-color:#fef3e2", "Expiry Risk": "background-color:#f3e8fd"}
            return colors.get(val, "")

        show_cols = ["Store", "Product_Name", "Opening_Stock", "Incoming_Stock", "Units_Sold",
                     "Closing_Stock", "Wastage", "Stockout", "Recommended_Order", "Risk"]
        st.dataframe(
            today_log[show_cols].rename(columns={"Product_Name": "Product"}).style.map(_risk_color_live, subset=["Risk"]),
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

        st.markdown(f"**Drill-down: {focus_product_name} @ {focus_store}** (using the focus selectors in the sidebar)")
        combo_log = sim_log[(sim_log["Store"] == focus_store) & (sim_log["Product_ID"] == focus_product_id)]
        if combo_log.empty:
            st.caption("No simulated days yet for this store/SKU.")
        else:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=combo_log["Date"], y=combo_log["Closing_Stock"], name="Closing Stock", line=dict(color="#1f77b4")))
            fig3.add_trace(go.Bar(x=combo_log["Date"], y=combo_log["Recommended_Order"], name="Order Placed", marker_color="#ff7f0e", opacity=0.6, yaxis="y2"))
            fig3.update_layout(
                yaxis=dict(title="Closing Stock"),
                yaxis2=dict(title="Order Placed", overlaying="y", side="right"),
                title=f"{focus_product_name} @ {focus_store}: Stock Level & Reorders (simulated)",
            )
            st.plotly_chart(fig3, use_container_width=True)

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
