# app/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import altair as alt

from src.data_cleaning import basic_clean_pipeline
from src.eda import delay_summary, numeric_distributions
from src.modeling import load_model, MODEL_PATH

st.set_page_config(
    page_title="Supply Chain Disruption",
    layout="wide"
)





@st.cache_data
def load_clean_data():
    return basic_clean_pipeline()

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Trained model not found. Run 'python -m src.modeling' first to train and save the model.")
        st.stop()
    return load_model(MODEL_PATH)

df = load_clean_data()
model = load_trained_model()

# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------
st.title("ForeSight")

st.markdown(
    """
This decision support dashboard analyzes historical **Life science shipments imported into the US**. The system estimates the risk that a shipment will be **delayed** relative to its scheduled delivery date.
"""
)

# st.markdown("---")

# ---------------------------------------------------------------------
# TABS (Tab 1 renamed to Executive Summary)
# ---------------------------------------------------------------------
tab1, tab2 = st.tabs(["Executive Summary", "Shipment Risk Check"])

# ---------------------------------------------------------------------
# TAB 1 – Executive Summary
# ---------------------------------------------------------------------
with tab1:
    st.subheader("Executive Summary")

    # Top-level metrics / cards
    total_shipments = len(df)
    overall_delay_rate = df["Delivery_Delayed"].mean()
    avg_delay_days = df["delivery_delay_days"].mean()

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric(
            "Total Shipments",
            f"{total_shipments:,}"
        )

    with col_b:
        st.metric(
            "Overall Delay Rate",
            f"{overall_delay_rate:.1%}"
        )

    with col_c:
        st.metric(
            "Average Delay (days)",
            f"{avg_delay_days:.1f}"
        )

    st.markdown("")
    st.markdown("")

    # delivered vs delayed shipments info
    delay_counts = df["Delivery_Delayed"].value_counts()
    on_time = int(delay_counts.get(0, 0))
    delayed = int(delay_counts.get(1, 0))
    total = on_time + delayed if (on_time + delayed) > 0 else 1
    overall_delay_rate = delayed / total

    summary_df = pd.DataFrame({
        "Status": ["On time", "Delayed"],
        "Count": [on_time, delayed],
        "Pct": [on_time/total, delayed/total]
    })

    color_scale = alt.Scale(
        domain=["On time", "Delayed"],
        range=["#2ecc71", "#e74c3c"]
    )

    base = alt.Chart(summary_df)

    donut = base.mark_arc(innerRadius=80).encode(
        theta=alt.Theta("Count:Q"),
        color=alt.Color("Status:N", scale=color_scale, legend=alt.Legend(title="")),
        tooltip=[
            alt.Tooltip("Status:N"),
            alt.Tooltip("Count:Q", title="Shipments", format=",.0f"),
            alt.Tooltip("Pct:Q", title="Share", format=".1%")
        ]
    ).properties(height=300, width=300)

    labels = base.mark_text(radius=120, size=12, color="#333").encode(
        theta=alt.Theta("Count:Q"),
        text=alt.Text("Pct:Q", format=".0%")
    )


    chart = (donut + labels).resolve_scale(color="independent")

    left, right = st.columns([1.2, 1.8])
    with left:
        st.markdown("### On-time vs Delayed Shipments")
        st.markdown(f"**Overall delayed:** {overall_delay_rate:.1%}")
        st.dataframe(
            summary_df.assign(Share=(summary_df["Pct"] * 100).round(1).astype(str) + "%")
                    .drop(columns=["Pct"])
                    .rename(columns={"Count": "Shipments"})
        )

    with right:
        st.altair_chart(chart, use_container_width=True)

    st.markdown("---")

    # --- Delay Hotspots (side-by-side table and chart) ---
    st.subheader("Delay Hotspots")

    summaries = delay_summary(df)

    # --- Export Countries by Delay Rate (improved) ---
    if "Country" in summaries:
        st.markdown("### Export Countries by Delay Rate")

        col_topk, _ = st.columns([1, 3])
        with col_topk:
            top_k = st.selectbox(
                "Show top N export countries by delay rate",
                options=[5, 10, 20, 30],
                index=1  # default 10
            )

        country_summary = summaries["Country"][["delay_rate", "n_shipments"]].reset_index()  # bring 'Country' out of index
        top_country = country_summary.head(top_k)

        # Compute overall delay rate as a reference line
        overall_delay_rate = df["Delivery_Delayed"].mean()

        # Side-by-side table and chart
        left, right = st.columns([1.2, 2])

        with left:
            st.dataframe(top_country.style.format({"delay_rate": "{:.1%}"}))

        with right:
            # Horizontal bar chart, sorted by delay rate, with tooltips and data labels
            base = alt.Chart(top_country).encode(
                y=alt.Y("Country:N", sort="-x", title=None),
                x=alt.X("delay_rate:Q", axis=alt.Axis(format="%", title="Delay rate")),
                tooltip=[
                    alt.Tooltip("Country:N", title="Export country"),
                    alt.Tooltip("delay_rate:Q", title="Delay rate", format=".1%"),
                    alt.Tooltip("n_shipments:Q", title="Shipments", format=",.0f"),
                ],
            )

            bars = base.mark_bar().encode(
                color=alt.Color(
                    "delay_rate:Q",
                    scale=alt.Scale(scheme="reds"),
                    legend=None,
                )
            ).properties(
                height=max(220, 26 * len(top_country)),  # adaptive height
            )

            labels = base.mark_text(
                align="left",
                dx=4,
                color="black"
            ).encode(
                text=alt.Text("delay_rate:Q", format=".1%")
            )

            ref_rule = alt.Chart(
                pd.DataFrame({"overall": [overall_delay_rate]})
            ).mark_rule(color="gray", strokeDash=[4, 4]).encode(
                x="overall:Q"
            )

            chart = (bars + labels + ref_rule).properties(width="container")

            st.altair_chart(chart, use_container_width=True)

            show_map = st.checkbox("Show world map view", value=False)

            if show_map:
                try:
                    import plotly.express as px
                    import pycountry

                    def to_iso3(name):
                        try:
                            return pycountry.countries.lookup(name).alpha_3
                        except Exception:
                            return None

                    map_df = top_country.copy()
                    map_df["iso3"] = map_df["Country"].apply(to_iso3)
                    map_df = map_df.dropna(subset=["iso3"])

                    fig = px.choropleth(
                        map_df,
                        locations="iso3",
                        color="delay_rate",
                        hover_name="Country",
                        hover_data={"delay_rate": ":.1%", "n_shipments": True, "iso3": False},
                        color_continuous_scale="Reds",
                        range_color=(map_df["delay_rate"].min(), map_df["delay_rate"].max()),
                        title=f"Top {top_k} Export Countries by Delay Rate",
                    )
                    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
                    st.plotly_chart(fig, use_container_width=True)

                except ModuleNotFoundError:
                    st.info("To enable the map, install: pip install plotly pycountry")

        # --- Shipment mode delay rates ---
        if "Shipment Mode" in summaries:
            st.markdown("### Delay Rate by Shipment Mode (On-time vs Delayed Share)")

            # Build counts by status
            status_df = (
                df.assign(Status=df["Delivery_Delayed"].map({0: "On time", 1: "Delayed"}))
                .dropna(subset=["Shipment Mode"])  # drop missing modes
                .groupby(["Shipment Mode", "Status"])
                .size()
                .reset_index(name="Count")
            )

            # Totals and shares
            totals = status_df.groupby("Shipment Mode")["Count"].sum().reset_index(name="Total")
            status_df = status_df.merge(totals, on="Shipment Mode", how="left")
            status_df["Share"] = status_df["Count"] / status_df["Total"]

            # Order modes by delayed share (most risky at top)
            delayed_share = status_df[status_df["Status"] == "Delayed"][["Shipment Mode", "Share"]]
            order_modes = delayed_share.sort_values("Share", ascending=False)["Shipment Mode"].tolist()

            # Side-by-side table and chart
            left2, right2 = st.columns([1.2, 2])

            with left2:
                # Compact summary table of delayed share and shipments
                delayed_tbl = delayed_share.merge(totals, on="Shipment Mode")
                delayed_tbl = delayed_tbl.rename(columns={"Share": "Delay rate", "Total": "Shipments"})
                st.dataframe(
                    delayed_tbl.style.format({"Delay rate": "{:.1%}", "Shipments": "{:,.0f}"})
                )

            with right2:
                # Executive colors: green for On time, red for Delayed
                color_scale = alt.Scale(domain=["On time", "Delayed"], range=["#2ecc71", "#e74c3c"])

                chart_stack = alt.Chart(status_df).mark_bar().encode(
                    y=alt.Y("Shipment Mode:N", sort=order_modes, title=None),
                    x=alt.X("Share:Q", axis=alt.Axis(format="%", title="Share of shipments")),
                    color=alt.Color("Status:N", scale=color_scale, legend=alt.Legend(title="", orient="top")),
                    tooltip=[
                        alt.Tooltip("Shipment Mode:N"),
                        alt.Tooltip("Status:N"),
                        alt.Tooltip("Count:Q", title="Shipments", format=",.0f"),
                        alt.Tooltip("Share:Q", title="Share", format=".1%"),
                    ]
                ).properties(
                    height=max(240, 28 * len(order_modes)),
                    width="container"
                )

                st.altair_chart(chart_stack, use_container_width=True)

        # --- Vendors with highest delay rate (min. 50 shipments) ---
        if "Vendor" in summaries:
            st.markdown("### Vendors with Highest Delay Rate (min. 50 shipments)")

            # Place the dropdown in a narrow column (same pattern as Export Countries)
            col_topk_v, _ = st.columns([1, 3])
            with col_topk_v:
                top_k_vendors = st.selectbox(
                    "Show top N vendors by delay rate",
                    options=[10, 20, 30],
                    index=0,
                    key="top_k_vendors"
                )

            vendor_df = (
                summaries["Vendor"]
                .reset_index()[["Vendor", "delay_rate", "n_shipments"]]
                .query("n_shipments >= 50")
                .sort_values("delay_rate", ascending=False)
            )

            top_vendors = vendor_df.head(top_k_vendors)

            if top_vendors.empty:
                st.info("No vendors meet the minimum 50-shipments criterion.")
            else:
                # Table and chart side by side
                left, right = st.columns([1.2, 2])

                with left:
                    st.dataframe(
                        top_vendors.style.format({
                            "delay_rate": "{:.1%}",
                            "n_shipments": "{:,.0f}"
                        })
                    )

                with right:
                    overall_delay_rate = df["Delivery_Delayed"].mean()

                    base = alt.Chart(top_vendors).encode(
                        y=alt.Y("Vendor:N", sort="-x", title=None),
                        x=alt.X("delay_rate:Q", axis=alt.Axis(format="%", title="Delay rate")),
                        tooltip=[
                            alt.Tooltip("Vendor:N"),
                            alt.Tooltip("delay_rate:Q", title="Delay rate", format=".1%"),
                            alt.Tooltip("n_shipments:Q", title="Shipments", format=",.0f"),
                        ],
                    )

                    bars = base.mark_bar().encode(
                        color=alt.Color("delay_rate:Q", scale=alt.Scale(scheme="reds"), legend=None)
                    ).properties(
                        height=max(240, 28 * len(top_vendors)),
                        width="container"
                    )

                    labels = base.mark_text(align="left", dx=4, color="#333").encode(
                        text=alt.Text("delay_rate:Q", format=".1%")
                    )

                    ref_rule = alt.Chart(pd.DataFrame({"overall": [overall_delay_rate]})).mark_rule(
                        color="gray", strokeDash=[4, 4]
                    ).encode(x="overall:Q")

                    chart = (bars + labels + ref_rule)
                    st.altair_chart(chart, use_container_width=True)

        # --- Product group delay rates ---
        if "Product Group" in summaries:
            st.markdown("### Delay Rate by Product Group")

            # Summary table (sorted by delay rate)
            pg_df = (
                summaries["Product Group"]
                .reset_index()[["Product Group", "delay_rate", "n_shipments"]]
                .sort_values("delay_rate", ascending=False)
            )

            # Build stacked composition data: On-time vs Delayed per product group
            status_df = (
                df.assign(Status=df["Delivery_Delayed"].map({0: "On time", 1: "Delayed"}))
                .dropna(subset=["Product Group"])
                .groupby(["Product Group", "Status"])
                .size()
                .reset_index(name="Count")
            )

            # Totals and shares
            totals = status_df.groupby("Product Group")["Count"].sum().reset_index(name="Total")
            status_df = status_df.merge(totals, on="Product Group", how="left")
            status_df["Share"] = status_df["Count"] / status_df["Total"]

            # Order groups by delayed share (highest first)
            delayed_share = status_df[status_df["Status"] == "Delayed"][["Product Group", "Share"]]
            order_groups = delayed_share.sort_values("Share", ascending=False)["Product Group"].tolist()

            # Side-by-side table and chart
            left_pg, right_pg = st.columns([1.2, 2])

            with left_pg:
                st.dataframe(
                    pg_df.rename(columns={"delay_rate": "Delay rate", "n_shipments": "Shipments"})
                        .style.format({"Delay rate": "{:.1%}", "Shipments": "{:,.0f}"})
                )

            with right_pg:
                color_scale = alt.Scale(domain=["On time", "Delayed"], range=["#2ecc71", "#e74c3c"])

                chart_stack = alt.Chart(status_df).mark_bar().encode(
                    y=alt.Y("Product Group:N", sort=order_groups, title=None),
                    x=alt.X("Share:Q", axis=alt.Axis(format="%", title="Share of shipments")),
                    color=alt.Color("Status:N", scale=color_scale, legend=alt.Legend(title="", orient="top")),
                    tooltip=[
                        alt.Tooltip("Product Group:N"),
                        alt.Tooltip("Status:N"),
                        alt.Tooltip("Count:Q", title="Shipments", format=",.0f"),
                        alt.Tooltip("Share:Q", title="Share", format=".1%"),
                    ]
                ).properties(
                    height=max(240, 32 * len(order_groups)),
                    width="container"
                )

                st.altair_chart(chart_stack, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 2 – Shipment Risk Check
# ---------------------------------------------------------------------
with tab2:
    st.subheader("Shipment Risk Check")

    st.markdown(
        """
All historical shipments here are **imports into the US**.  

Use this tool to estimate the **delay risk level** (Low / Medium / High) for a planned shipment, based on:
- **Export country (origin)**
- **Vendor** (filtered by export country)
- **Shipment mode**
- **Product group**
- Basic cost, quantity, weight and lead-time assumptions.
"""
    )

    col_form, col_info = st.columns([2, 1])

    with col_form:
        col1, col2 = st.columns(2)

        with col1:
            # Export country (origin)
            export_country = st.selectbox(
                "Export Country (origin)",
                sorted(df["Country"].dropna().unique())
            )

            # Filter vendors by export country
            vendors_for_country = (
                df.loc[df["Country"] == export_country, "Vendor"]
                .dropna()
                .unique()
            )
            vendor = st.selectbox(
                "Vendor",
                sorted(vendors_for_country)
            )

            shipment_mode = st.selectbox(
                "Shipment Mode",
                sorted(df["Shipment Mode"].dropna().unique())
            )
            product_group = st.selectbox(
                "Product Group",
                sorted(df["Product Group"].dropna().unique())
            )

        with col2:
            line_qty = st.number_input(
                "Line Item Quantity",
                min_value=1,
                value=1000,
                step=100
            )
            line_value = st.number_input(
                "Line Item Value (USD)",
                min_value=0.0,
                value=50000.0,
                step=1000.0
            )
            weight_kg = st.number_input(
                "Weight (Kilograms)",
                min_value=0.0,
                value=1000.0,
                step=10.0
            )
            freight_cost = st.number_input(
                "Freight Cost (USD)",
                min_value=0.0,
                value=5000.0,
                step=500.0
            )

        st.markdown("##### Lead Time Assumptions")

        col3, col4 = st.columns(2)
        with col3:
            po_to_schedule_days = st.number_input(
                "Planned PO-to-Schedule lead time (days)",
                min_value=-30,
                max_value=365,
                value=30
            )
        with col4:
            po_to_delivery_days = st.number_input(
                "Planned PO-to-Delivery lead time (days)",
                min_value=-30,
                max_value=365,
                value=60
            )

        show_debug = st.checkbox("Show debug info (for development)", value=False)

        def map_probability_to_risk_label(p: float):
            # Tuned thresholds for clearer separation
            if p < 0.15:
                return "Low"
            elif p < 0.3:
                return "Medium"
            else:
                return "High"

        from src.eda import delay_summary

        def simple_text_explanation(input_row: dict, df_local: pd.DataFrame) -> str:
            msgs = []
            summaries = delay_summary(df_local)

            if "Country" in summaries and input_row.get("Country") in summaries["Country"].index:
                row = summaries["Country"].loc[input_row["Country"]]
                dr = row["delay_rate"]
                msgs.append(f"- From **{input_row['Country']}**, about **{dr:.0%}** of shipments are delayed.")

            if "Shipment Mode" in summaries and input_row.get("Shipment Mode") in summaries["Shipment Mode"].index:
                row = summaries["Shipment Mode"].loc[input_row["Shipment Mode"]]
                dr = row["delay_rate"]
                msgs.append(f"- For **{input_row['Shipment Mode']}** shipments, about **{dr:.0%}** are delayed.")

            if "Vendor" in summaries and input_row.get("Vendor") in summaries["Vendor"].index:
                row = summaries["Vendor"].loc[input_row["Vendor"]]
                dr = row["delay_rate"]
                msgs.append(f"- For vendor **{input_row['Vendor']}**, about **{dr:.0%}** of shipments are delayed.")

            if not msgs:
                return (
                    "This risk level is based on patterns in historical shipments by export country, "
                    "vendor, and shipment mode."
                )
            else:
                return (
                    "This risk level is based on patterns seen in similar historical shipments:\n"
                    + "\n".join(msgs)
                )

        if st.button("Estimate delay risk"):
            from src.config import CATEGORICAL_COLS, NUMERIC_COLS_RAW

            input_dict = {
                "Country": export_country,
                "Shipment Mode": shipment_mode,
                "Product Group": product_group,
                "Vendor": vendor,
                "Line Item Quantity": line_qty,
                "Line Item Value": line_value,
                "Weight (Kilograms)": weight_kg,
                "Freight Cost (USD)": freight_cost,
                "delivery_delay_days": 0,  # unknown for planned shipments
                "po_to_schedule_days": po_to_schedule_days,
                "po_to_delivery_days": po_to_delivery_days,
            }

            # Fill remaining categorical/numeric features with simple defaults
            for col in CATEGORICAL_COLS:
                if col not in input_dict and col in df.columns:
                    input_dict[col] = df[col].mode(dropna=True)[0]

            for col in NUMERIC_COLS_RAW:
                if col not in input_dict and col in df.columns:
                    try:
                        med = float(df[col].median())
                    except Exception:
                        med = 0.0
                    input_dict[col] = med

            input_df = pd.DataFrame([input_dict])

            if show_debug:
                st.markdown("#### Debug – Input row sent to model")
                st.dataframe(input_df.T)

            proba = model.predict_proba(input_df)[0, 1]
            risk_label = map_probability_to_risk_label(proba)

            st.markdown(f"### Estimated delay risk: **{risk_label}**")

            if show_debug:
                st.markdown(f"**Raw model probability (for debugging):** {proba:.4f}")

            st.markdown("#### Explanation")
            explanation = simple_text_explanation(input_dict, df)
            st.markdown(explanation)

            st.info(
                "This is an approximate risk level based on historical export shipments to the US. "
                "It should be used as input for planning decisions, not as a guarantee."
            )

