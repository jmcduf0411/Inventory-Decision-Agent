"""
Supply Chain Analytics Inventory Decision Agent - Streamlit Application
========================================================================
Interactive web application for natural-language inventory decision analysis.
Supports:
1. Economic Order Quantity (EOQ)
2. Quantity Discount
3. Safety Stock and Reorder Point (Continuous Review)
4. Fixed-Period Review (Periodic Review Order-Up-To)
5. Newsvendor (Single-Period Stochastic)

Architecture:
Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain
"""

import os
import sys
import math
import json
from typing import Any, Dict, List, Optional
import streamlit as st
import pandas as pd
import altair as alt

# Ensure workspace root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inventory_agent import InventoryDecisionAgent, AgentResponse
import inventory_tools as tools


# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Supply Chain Inventory Decision Agent",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Pre-configured Sample Queries
# ---------------------------------------------------------------------------

SAMPLE_QUERIES = {
    "1. Economic Order Quantity (EOQ)": (
        "Our annual demand is 12,000 units. Ordering cost is $100 per order, and holding cost is $4 per unit per year. "
        "What is our optimal order quantity and how often should we place orders?"
    ),
    "2. Quantity Discount Model": (
        "Annual demand is 5,000 units, ordering cost is $49, and holding cost rate is 20%. "
        "Vendor offers tiers: 1 to 999 units at $5.00, 1000 to 1999 units at $4.80, and 2000+ units at $4.75. "
        "What quantity should we order?"
    ),
    "3. Safety Stock & Reorder Point": (
        "Average daily demand is 50 units with a standard deviation of 10 units. "
        "Supplier lead time is 9 days (constant). "
        "What safety stock and reorder point do we need for a 95% cycle service level?"
    ),
    "4. Fixed-Period Review (Order-Up-To)": (
        "We review inventory every 14 days with a lead time of 7 days. "
        "Average daily demand is 20 units with a standard deviation of 4 units. "
        "Target service level is 95%. "
        "Currently we have 120 on-hand, 50 on-order, and 10 backordered. "
        "What is our target order-up-to level and how many units should we order right now?"
    ),
    "5. Newsvendor (Single-Period)": (
        "We are selling holiday gift baskets for $100 each. Unit production cost is $40. "
        "Unsold baskets can be salvaged for $20. "
        "Demand follows a normal distribution with mean of 200 and standard deviation of 40. "
        "How many baskets should we produce to maximize expected profit?"
    ),
}


# ---------------------------------------------------------------------------
# Chart Generation Helpers
# ---------------------------------------------------------------------------

def generate_eoq_chart(res: tools.EOQResult) -> alt.Chart:
    """Generates cost curve trade-off chart for EOQ."""
    q_star = res.optimal_order_quantity
    q_min = max(10.0, q_star * 0.2)
    q_max = max(50.0, q_star * 2.5)
    step = (q_max - q_min) / 100.0

    quantities = []
    curr = q_min
    while curr <= q_max:
        quantities.append(curr)
        curr += step

    d = res.annual_demand
    s = res.order_cost
    h = res.holding_cost_per_unit

    data_rows = []
    for q in quantities:
        ord_c = (d / q) * s
        hold_c = (q / 2.0) * h
        tot_c = ord_c + hold_c
        data_rows.append({"Quantity": q, "Cost": ord_c, "Cost Component": "Annual Ordering Cost"})
        data_rows.append({"Quantity": q, "Cost": hold_c, "Cost Component": "Annual Holding Cost"})
        data_rows.append({"Quantity": q, "Cost": tot_c, "Cost Component": "Total Inventory Cost"})

    df = pd.DataFrame(data_rows)

    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("Quantity:Q", title="Order Quantity (Units)"),
            y=alt.Y("Cost:Q", title="Annual Cost ($)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "Cost Component:N",
                scale=alt.Scale(
                    domain=["Annual Ordering Cost", "Annual Holding Cost", "Total Inventory Cost"],
                    range=["#E67E22", "#2980B9", "#27AE60"]
                )
            ),
            tooltip=["Cost Component:N", alt.Tooltip("Quantity:Q", format=",.0f"), alt.Tooltip("Cost:Q", format="$,.2f")]
        )
        .properties(title=f"EOQ Cost Trade-off Curves (Optimal Q* = {q_star:,.1f})", height=320)
        .interactive()
    )

    q_df = pd.DataFrame([{"Q_star": q_star, "Label": f"Optimal Q* = {q_star:,.1f}"}])
    rule = alt.Chart(q_df).mark_rule(color="#E74C3C", strokeDash=[4, 4], strokeWidth=2).encode(x="Q_star:Q")
    return chart + rule


def generate_quantity_discount_chart(res: tools.QuantityDiscountResult) -> alt.Chart:
    """Bar chart comparing evaluated total costs across tiers."""
    rows = []
    for t in res.tiers_evaluated:
        max_str = f"{t.max_quantity:,.0f}" if t.max_quantity else "∞"
        tier_label = f"Tier {t.tier_number} (${t.unit_price:.2f})\n[{t.min_quantity:,.0f} - {max_str}]"
        is_best = (t.tier_number == res.best_tier.tier_number)
        rows.append({
            "Tier": tier_label,
            "Total Cost": t.total_annual_cost,
            "Purchase Cost": t.annual_purchase_cost,
            "Ordering Cost": t.annual_ordering_cost,
            "Holding Cost": t.annual_holding_cost,
            "Status": "Optimal (Best)" if is_best else "Alternative",
        })

    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Tier:N", title="Discount Tier", sort=None),
            y=alt.Y("Total Cost:Q", title="Total Annual Cost ($)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "Status:N",
                scale=alt.Scale(domain=["Optimal (Best)", "Alternative"], range=["#27AE60", "#95A5A6"])
            ),
            tooltip=[
                "Tier:N",
                alt.Tooltip("Total Cost:Q", format="$,.2f"),
                alt.Tooltip("Purchase Cost:Q", format="$,.2f"),
                alt.Tooltip("Ordering Cost:Q", format="$,.2f"),
                alt.Tooltip("Holding Cost:Q", format="$,.2f"),
            ]
        )
        .properties(title="Total Annual Cost Comparison Across Discount Tiers", height=300)
    )
    return chart


def generate_rop_chart(res: tools.SafetyStockROPResult) -> alt.Chart:
    """Normal distribution curve showing lead time demand mean, safety stock, and ROP."""
    mu = res.lead_time_demand_mean
    sigma = res.lead_time_demand_std
    rop = res.reorder_point

    x_min = max(0.0, mu - 3.8 * sigma)
    x_max = mu + 3.8 * sigma
    step = (x_max - x_min) / 150.0

    rows = []
    curr = x_min
    while curr <= x_max:
        pdf_val = (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((curr - mu) / sigma) ** 2)
        category = "Stockout Risk Zone" if curr > rop else "Protected Coverage"
        rows.append({"Lead Time Demand": curr, "Density": pdf_val, "Zone": category})
        curr += step

    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_area(opacity=0.65)
        .encode(
            x=alt.X("Lead Time Demand:Q", title="Demand During Lead Time (Units)"),
            y=alt.Y("Density:Q", title="Probability Density"),
            color=alt.Color(
                "Zone:N",
                scale=alt.Scale(domain=["Protected Coverage", "Stockout Risk Zone"], range=["#2980B9", "#E74C3C"])
            ),
            tooltip=[alt.Tooltip("Lead Time Demand:Q", format=",.1f"), alt.Tooltip("Density:Q", format=".5f"), "Zone:N"]
        )
        .properties(
            title=f"Lead Time Demand Distribution (Mean={mu:,.1f}, SS={res.safety_stock:,.1f}, ROP={rop:,.1f})",
            height=300
        )
    )

    lines_df = pd.DataFrame([
        {"Value": mu, "Marker": f"Mean Demand ({mu:,.0f})"},
        {"Value": rop, "Marker": f"Reorder Point ({rop:,.0f})"},
    ])
    rule = alt.Chart(lines_df).mark_rule(strokeDash=[4, 4], strokeWidth=2).encode(
        x="Value:Q",
        color=alt.Color("Marker:N", scale=alt.Scale(domain=[f"Mean Demand ({mu:,.0f})", f"Reorder Point ({rop:,.0f})"], range=["#2C3E50", "#C0392B"]))
    )
    return chart + rule


def generate_fixed_period_chart(res: tools.FixedPeriodReviewResult) -> alt.Chart:
    """Visualizes inventory components against Order-Up-To Level."""
    df = pd.DataFrame([
        {"Component": "On-Hand Stock", "Units": res.current_on_hand, "Type": "Current Inventory"},
        {"Component": "On-Order (Pipeline)", "Units": res.current_on_order, "Type": "Current Inventory"},
        {"Component": "Backorders", "Units": -res.current_backorders, "Type": "Liability"},
        {"Component": "Net Position (IP)", "Units": res.inventory_position, "Type": "Current Inventory"},
        {"Component": "Recommended Order (Q)", "Units": res.order_quantity, "Type": "Decision"},
        {"Component": "Target Level (S)", "Units": res.order_up_to_level, "Type": "Target"},
    ])
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Component:N", title="Inventory Element", sort=None),
            y=alt.Y("Units:Q", title="Units"),
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(
                    domain=["Current Inventory", "Liability", "Decision", "Target"],
                    range=["#3498DB", "#E74C3C", "#2ECC71", "#9B59B6"]
                )
            ),
            tooltip=["Component:N", alt.Tooltip("Units:Q", format=",.1f"), "Type:N"]
        )
        .properties(title=f"Periodic Review Position vs Order-Up-To Level (S = {res.order_up_to_level:,.1f})", height=300)
    )
    return chart


def generate_newsvendor_chart(res: tools.NewsvendorResult) -> alt.Chart:
    """Visualizes underage vs overage costs and expected financial breakdown."""
    cost_df = pd.DataFrame([
        {"Cost Type": "Cost of Underage (Cu: Lost Margin)", "Amount ($)": res.underage_cost},
        {"Cost Type": "Cost of Overage (Co: Leftover Loss)", "Amount ($)": res.overage_cost},
    ])
    cost_chart = (
        alt.Chart(cost_df)
        .mark_bar(size=40, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Cost Type:N", title="", sort=None),
            y=alt.Y("Amount ($):Q", title="Cost per Unit ($)"),
            color=alt.Color("Cost Type:N", scale=alt.Scale(range=["#E67E22", "#34495E"]), legend=None),
            tooltip=["Cost Type:N", alt.Tooltip("Amount ($):Q", format="$,.2f")]
        )
        .properties(title=f"Marginal Economic Costs (Critical Ratio = {res.critical_ratio*100:.1f}%)", height=280)
    )

    fin_df = pd.DataFrame([
        {"Metric": "Expected Revenue", "Amount ($)": res.expected_revenue, "Category": "Inflow"},
        {"Metric": "Expected Salvage Recovery", "Amount ($)": res.expected_salvage_recovery, "Category": "Inflow"},
        {"Metric": "Production / Purchase Cost", "Amount ($)": res.expected_cost, "Category": "Outflow"},
        {"Metric": "Expected Net Profit", "Amount ($)": res.expected_profit, "Category": "Profit"},
    ])
    fin_chart = (
        alt.Chart(fin_df)
        .mark_bar(size=35, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Metric:N", title="", sort=None),
            y=alt.Y("Amount ($):Q", title="Expected Dollars ($)"),
            color=alt.Color(
                "Category:N",
                scale=alt.Scale(domain=["Inflow", "Outflow", "Profit"], range=["#2980B9", "#E74C3C", "#27AE60"])
            ),
            tooltip=["Metric:N", alt.Tooltip("Amount ($):Q", format="$,.2f")]
        )
        .properties(title="Expected Financial Payoffs at Optimal Q*", height=280)
    )
    return cost_chart | fin_chart


def render_verifiable_solution(res: Any):
    """Renders the 8 verifiable calculation components in Streamlit."""
    if not hasattr(res, "inputs_extracted"):
        return

    with st.expander("🎓 Student Answer Verification (Step-by-Step Deterministic Audit)", expanded=True):
        st.caption("All numerical results below are generated deterministically by the Python operations research tool.")
        
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown("**1. Inputs Extracted**")
            st.json(res.inputs_extracted)
            st.markdown("**3. Formula**")
            st.code(res.formula, language="text")

        with col_v2:
            st.markdown("**2. Variable Definitions**")
            v_rows = [{"Symbol": f"${k}$", "Meaning": v} for k, v in getattr(res, "variable_definitions", {}).items()]
            st.dataframe(pd.DataFrame(v_rows), hide_index=True, use_container_width=True)
            st.markdown("**4. Formula with Substituted Values**")
            st.code(res.formula_substituted, language="text")

        st.markdown("**5. Intermediate Calculations**")
        inter_rows = []
        for ik, iv in getattr(res, "intermediate_calculations", {}).items():
            if ik != "tier_step_by_step":
                inter_rows.append({"Step / Metric": ik, "Value": f"{iv:,.4f}" if isinstance(iv, float) else str(iv)})
        st.dataframe(pd.DataFrame(inter_rows), hide_index=True, use_container_width=True)

        st.markdown("**6. Final Answer**")
        st.success(f"🎯 **Final Answer**: **{res.final_answer:,} units** (Optimal: {getattr(res, 'optimal_order_quantity', getattr(res, 'reorder_point', getattr(res, 'order_quantity', getattr(res, 'optimal_quantity', res.final_answer)))):.2f})")

        st.markdown("**7. Verification Checks**")
        for chk in getattr(res, "verification_checks", []):
            st.markdown(f"- ✅ {chk}")

        st.markdown("**8. Underlying Assumptions**")
        for a in getattr(res, "assumptions", []):
            st.markdown(f"- 📌 {a}")



# ---------------------------------------------------------------------------
# Streamlit Application Layout & Main Logic
# ---------------------------------------------------------------------------

def main():
    # Initialize Agent in Session State
    if "agent" not in st.session_state:
        st.session_state.agent = InventoryDecisionAgent()

    if "current_question" not in st.session_state:
        st.session_state.current_question = SAMPLE_QUERIES["1. Economic Order Quantity (EOQ)"]

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.title("📦 Inventory Agent")
        st.markdown("**Supply Chain Operations Research**")
        st.caption("Powered by 7-Step Decision Pipeline")

        st.markdown("---")
        st.subheader("💡 Load Sample Questions")
        selected_sample_label = st.selectbox(
            "Choose an inventory problem:",
            options=list(SAMPLE_QUERIES.keys()),
            index=0,
            help="Select a model scenario to automatically load its question into the input area."
        )

        if st.button("📥 Load Question", use_container_width=True):
            st.session_state.current_question = SAMPLE_QUERIES[selected_sample_label]
            st.rerun()

        st.markdown("---")
        st.subheader("⚙️ Supported Models")
        st.markdown("""
        1. **EOQ**: Classic lot sizing & cost balance
        2. **Quantity Discount**: Multi-tier price breaks
        3. **Safety Stock & ROP**: Continuous review $(R, Q)$
        4. **Fixed-Period Review**: Periodic review $(P, S)$
        5. **Newsvendor**: Perishable / seasonal single-period
        """)

        st.markdown("---")
        st.subheader("🔧 Display Settings")
        show_pipeline = st.checkbox("Show 7-Step Pipeline Trace", value=True)
        show_charts = st.checkbox("Show Interactive Visualizations", value=True)
        show_json = st.checkbox("Show Raw JSON API Response", value=False)

    # ---------------- Header ----------------
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1E3C72 0%, #2A5298 100%); padding: 22px 28px; border-radius: 10px; color: white; margin-bottom: 24px;">
        <h2 style="margin: 0; color: white;">📦 Supply Chain Analytics: Inventory Decision Agent</h2>
        <p style="margin: 6px 0 0 0; opacity: 0.9; font-size: 1.05rem;">
            Ask natural-language inventory questions to automatically identify models, extract operational parameters, execute exact calculations, verify constraints, and generate executive recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Architecture Banner
    st.markdown("""
    <div style="background-color: #F8F9FA; border: 1px solid #E9ECEF; border-radius: 8px; padding: 12px 18px; margin-bottom: 20px; font-size: 0.9rem;">
        <strong>Architecture Pipeline:</strong> 
        <code>Customer Question</code> ➔ 
        <code>Identify Model</code> ➔ 
        <code>Extract Inputs</code> ➔ 
        <code>Select Python Tool</code> ➔ 
        <code>Calculate</code> ➔ 
        <code>Verify</code> ➔ 
        <code>Explain</code>
    </div>
    """, unsafe_allow_html=True)

    # Main Tabs
    tab_agent, tab_manual, tab_reference = st.tabs([
        "💬 Natural Language Agent",
        "🎛️ Interactive Parameter Calculator",
        "📚 Mathematical Formulations Reference"
    ])

    # =========================================================================
    # TAB 1: Natural Language Agent
    # =========================================================================
    with tab_agent:
        st.markdown("### Enter Your Inventory Question")
        
        user_query = st.text_area(
            "Natural Language Inventory Question:",
            value=st.session_state.current_question,
            height=120,
            placeholder="e.g. Annual demand is 10,000 units, order cost is $50, holding cost is $2/unit/year. What is our EOQ?",
            help="Describe your inventory situation including demand, costs, lead times, or service levels."
        )

        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            run_button = st.button("🚀 Analyze & Decide", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🔄 Reset to Default Question", use_container_width=False):
                st.session_state.current_question = SAMPLE_QUERIES["1. Economic Order Quantity (EOQ)"]
                st.rerun()

        if run_button or user_query:
            if not user_query.strip():
                st.warning("Please enter a question to analyze.")
            else:
                agent: InventoryDecisionAgent = st.session_state.agent
                resp: AgentResponse = agent.ask(user_query)

                # 1. Pipeline Execution Trace Expander
                if show_pipeline:
                    with st.expander("🔍 7-Step Pipeline Execution Audit", expanded=True):
                        cols = st.columns(len(resp.pipeline_trace))
                        for idx, step in enumerate(resp.pipeline_trace):
                            with cols[idx]:
                                if step.status in ["SUCCESS", "SELECTED", "RECEIVED"]:
                                    st.success(f"**{step.name}**\n\n{step.summary}")
                                else:
                                    st.warning(f"**{step.name}**\n\n{step.summary}")

                # 2. Incomplete or Missing Inputs Check
                if not resp.success:
                    st.error(f"### ⚠️ Incomplete Inputs for {resp.model_display_name}")
                    st.markdown(resp.explanation)
                    if resp.extracted_inputs:
                        st.markdown("**Successfully Extracted So Far:**")
                        st.json(resp.extracted_inputs)
                else:
                    # 3. Success Metrics Banner
                    res = resp.raw_result
                    st.markdown("---")
                    st.markdown(f"### 📊 Decision Dashboard: {resp.model_display_name}")

                    # Model Specific KPI Metric Cards
                    if resp.identified_model == "eoq":
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Recommended Order (Q*)", f"{res.recommended_order_quantity:,} units", f"{res.optimal_order_quantity:.1f} exact")
                        m2.metric("Total Annual Inventory Cost", f"${res.annual_inventory_cost:,.2f}", "Holding + Ordering")
                        m3.metric("Annual Orders", f"{res.orders_per_year:.2f} orders", f"Every {res.cycle_time_days:.1f} days")
                        m4.metric("Cost Balance Delta", f"${res.cost_difference_holding_vs_ordering:.4f}", "H*Q/2 vs S*D/Q")

                    elif resp.identified_model == "quantity_discount":
                        best = res.best_tier
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Recommended Order Q*", f"{res.recommended_order_quantity:,} units", f"Tier {best.tier_number}")
                        m2.metric("Negotiated Unit Price", f"${best.unit_price:.2f}/unit", f"Base: ${res.tiers_evaluated[0].unit_price:.2f}")
                        m3.metric("Total Annual Cost", f"${res.total_annual_cost:,.2f}", f"Purch + Ord + Hold")
                        m4.metric("Annual Savings vs Base", f"${res.cost_savings_vs_base:,.2f}", "Saved", delta_color="normal")

                    elif resp.identified_model == "safety_stock_rop":
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Reorder Point (ROP)", f"{res.recommended_reorder_point:,} units", f"Trigger level")
                        m2.metric("Safety Stock (SS)", f"{res.recommended_safety_stock:,} units", f"Buffer")
                        m3.metric("Cycle Service Level", f"{res.service_level*100:.1f}%", f"Z = {res.z_score:.2f}")
                        m4.metric("Lead Time Demand (μ_L)", f"{res.lead_time_demand_mean:,.1f} units", f"σ_L = {res.lead_time_demand_std:.1f}")

                    elif resp.identified_model == "fixed_period_review":
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Target Level (S)", f"{res.recommended_order_up_to_level:,} units", "Order-Up-To")
                        m2.metric("Recommended Order (Q)", f"{res.recommended_order_quantity:,} units", "Place right now")
                        m3.metric("Safety Stock Buffer", f"{res.recommended_safety_stock:,} units", f"Z = {res.z_score:.2f}")
                        m4.metric("Current Net Position", f"{res.inventory_position:,.0f} units", f"On-hand: {res.current_on_hand:,.0f}")

                    elif resp.identified_model == "newsvendor":
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Optimal Production (Q*)", f"{res.recommended_quantity:,} units", f"{res.optimal_quantity:.1f} exact")
                        m2.metric("Critical Fractile (CR)", f"{res.critical_ratio*100:.1f}%", f"Underage ratio")
                        m3.metric("Expected Total Profit", f"${res.expected_profit:,.2f}", "Expected return")
                        m4.metric("In-Stock Probability", f"{res.in_stock_probability*100:.1f}%", f"Z = {res.z_score:.2f}")

                    # 4. Interactive Charts
                    if show_charts:
                        st.markdown("#### 📈 Operations Research Visualization")
                        if resp.identified_model == "eoq":
                            st.altair_chart(generate_eoq_chart(res), use_container_width=True)
                        elif resp.identified_model == "quantity_discount":
                            st.altair_chart(generate_quantity_discount_chart(res), use_container_width=True)
                        elif resp.identified_model == "safety_stock_rop":
                            st.altair_chart(generate_rop_chart(res), use_container_width=True)
                        elif resp.identified_model == "fixed_period_review":
                            st.altair_chart(generate_fixed_period_chart(res), use_container_width=True)
                        elif resp.identified_model == "newsvendor":
                            st.altair_chart(generate_newsvendor_chart(res), use_container_width=True)

                    # 5. Full Written Explanation
                    st.markdown("#### 📝 Executive Explanation & Managerial Insights")
                    st.markdown(resp.explanation)

                    # 6. Dedicated Interactive Verification Component
                    render_verifiable_solution(res)

                    # 7. Raw JSON Expander
                    if show_json:
                        with st.expander("🔧 Raw JSON Result Output"):
                            clean_dict = {
                                "model": resp.identified_model,
                                "tool": resp.selected_tool,
                                "inputs": resp.extracted_inputs,
                                "result": {k: v for k, v in res.__dict__.items() if k != "verification"},
                                "verification": {
                                    "is_valid": res.verification.is_valid,
                                    "checks": res.verification.checks_passed,
                                    "warnings": res.verification.warnings
                                } if hasattr(res, "verification") else None
                            }
                            st.json(clean_dict)

    # =========================================================================
    # TAB 2: Interactive Parameter Calculator
    # =========================================================================
    with tab_manual:
        st.markdown("### Direct Parameter Exploration")
        st.caption("Manually adjust input parameters for any of the 5 models to see immediate calculations.")

        model_choice = st.selectbox(
            "Select Model to Calculate Directly:",
            options=[
                "Economic Order Quantity (EOQ)",
                "Quantity Discount Model",
                "Safety Stock and Reorder Point",
                "Fixed-Period Review",
                "Newsvendor (Single-Period)",
            ]
        )

        if model_choice == "Economic Order Quantity (EOQ)":
            col1, col2, col3 = st.columns(3)
            with col1:
                d_in = st.number_input("Annual Demand (D):", min_value=1.0, value=12000.0, step=500.0)
            with col2:
                s_in = st.number_input("Order Setup Cost (S):", min_value=1.0, value=100.0, step=10.0)
            with col3:
                h_in = st.number_input("Holding Cost per Unit/Year (H):", min_value=0.01, value=4.0, step=0.5)

            if st.button("Calculate EOQ", type="primary"):
                calc_res = tools.calculate_eoq(demand=d_in, order_cost=s_in, holding_cost=h_in)
                st.success(f"**Recommended Order Quantity**: {calc_res.recommended_order_quantity:,} units")
                st.altair_chart(generate_eoq_chart(calc_res), use_container_width=True)
                render_verifiable_solution(calc_res)

        elif model_choice == "Quantity Discount Model":
            col1, col2, col3 = st.columns(3)
            with col1:
                qd_d = st.number_input("Annual Demand:", min_value=1.0, value=5000.0, step=500.0)
            with col2:
                qd_s = st.number_input("Order Cost:", min_value=1.0, value=49.0, step=5.0)
            with col3:
                qd_hr = st.number_input("Holding Cost Rate (%):", min_value=1.0, max_value=100.0, value=20.0, step=1.0)

            st.markdown("**Discount Tiers:**")
            t_df = pd.DataFrame([
                {"Min Qty": 1, "Max Qty": 999, "Price ($)": 5.00},
                {"Min Qty": 1000, "Max Qty": 1999, "Price ($)": 4.80},
                {"Min Qty": 2000, "Max Qty": None, "Price ($)": 4.75},
            ])
            st.dataframe(t_df, use_container_width=True)

            if st.button("Calculate Best Tier", type="primary"):
                tiers = [(1, 999, 5.00), (1000, 1999, 4.80), (2000, None, 4.75)]
                qd_res = tools.calculate_quantity_discount(demand=qd_d, order_cost=qd_s, discount_tiers=tiers, holding_rate=qd_hr)
                st.success(f"**Best Tier**: Tier {qd_res.best_tier.tier_number} | **Order Qty**: {qd_res.recommended_order_quantity:,} units at ${qd_res.best_tier.unit_price:.2f}/unit")
                st.altair_chart(generate_quantity_discount_chart(qd_res), use_container_width=True)
                render_verifiable_solution(qd_res)

        elif model_choice == "Safety Stock and Reorder Point":
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                rop_d = st.number_input("Daily Demand Mean:", min_value=1.0, value=50.0, step=5.0)
            with col2:
                rop_std = st.number_input("Daily Demand Std Dev:", min_value=0.0, value=10.0, step=1.0)
            with col3:
                rop_lt = st.number_input("Lead Time (Days):", min_value=1.0, value=9.0, step=1.0)
            with col4:
                rop_csl = st.slider("Cycle Service Level:", min_value=0.50, max_value=0.999, value=0.95, step=0.01)

            if st.button("Calculate Safety Stock & ROP", type="primary"):
                rop_res = tools.calculate_safety_stock_rop(demand_mean=rop_d, demand_std=rop_std, lead_time=rop_lt, service_level=rop_csl)
                st.success(f"**Reorder Point**: {rop_res.recommended_reorder_point:,} units | **Safety Stock**: {rop_res.recommended_safety_stock:,} units")
                st.altair_chart(generate_rop_chart(rop_res), use_container_width=True)
                render_verifiable_solution(rop_res)

        elif model_choice == "Fixed-Period Review":
            col1, col2, col3 = st.columns(3)
            with col1:
                fp_p = st.number_input("Review Period P (days):", min_value=1.0, value=14.0, step=1.0)
                fp_d = st.number_input("Daily Demand Mean (d):", min_value=1.0, value=20.0, step=5.0)
            with col2:
                fp_l = st.number_input("Lead Time L (days):", min_value=0.0, value=7.0, step=1.0)
                fp_std = st.number_input("Daily Demand Std Dev:", min_value=0.0, value=4.0, step=1.0)
            with col3:
                fp_oh = st.number_input("On-Hand Inventory:", value=120.0, step=10.0)
                fp_oo = st.number_input("On-Order Inventory:", value=50.0, step=10.0)
                fp_bo = st.number_input("Backorders:", value=10.0, step=5.0)

            if st.button("Calculate Order-Up-To Level", type="primary"):
                fp_res = tools.calculate_fixed_period_review(
                    review_period=fp_p, lead_time=fp_l, demand_mean=fp_d, demand_std=fp_std,
                    service_level=0.95, on_hand=fp_oh, on_order=fp_oo, backorders=fp_bo
                )
                st.success(f"**Target Level (S)**: {fp_res.recommended_order_up_to_level:,} units | **Order Now (Q)**: {fp_res.recommended_order_quantity:,} units")
                st.altair_chart(generate_fixed_period_chart(fp_res), use_container_width=True)
                render_verifiable_solution(fp_res)

        elif model_choice == "Newsvendor (Single-Period)":
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                nv_p = st.number_input("Selling Price ($):", min_value=1.0, value=100.0, step=5.0)
            with col2:
                nv_c = st.number_input("Unit Cost ($):", min_value=1.0, value=40.0, step=5.0)
            with col3:
                nv_s = st.number_input("Salvage Value ($):", min_value=0.0, value=20.0, step=5.0)
            with col4:
                nv_mu = st.number_input("Demand Mean:", min_value=1.0, value=200.0, step=20.0)
                nv_sigma = st.number_input("Demand Std Dev:", min_value=0.0, value=40.0, step=5.0)

            if st.button("Calculate Optimal Production", type="primary"):
                nv_res = tools.calculate_newsvendor(price=nv_p, cost=nv_c, salvage_value=nv_s, demand_mean=nv_mu, demand_std=nv_sigma)
                st.success(f"**Optimal Quantity (Q*)**: {nv_res.recommended_quantity:,} units | **Expected Profit**: ${nv_res.expected_profit:,.2f}")
                st.altair_chart(generate_newsvendor_chart(nv_res), use_container_width=True)
                render_verifiable_solution(nv_res)

    # =========================================================================
    # TAB 3: Mathematical Formulations Reference
    # =========================================================================
    with tab_reference:
        st.markdown("### 📚 Operations Research Mathematical Cheat-Sheet")
        
        st.markdown(r"""
        #### 1. Economic Order Quantity (EOQ)
        - **Optimal Lot Size**:
          $$Q^* = \sqrt{\frac{2 D S}{H}}$$
        - **Total Annual Cost**:
          $$TC(Q) = \frac{D}{Q}S + \frac{Q}{2}H + D \cdot C$$
        - **Order Frequency & Cycle Time**:
          $$N = \frac{D}{Q^*}, \quad T = \frac{Q^*}{D} \times \text{working\_days}$$

        ---

        #### 2. Quantity Discount Model
        - For each tier $j$ with unit price $C_j$ and holding cost $H_j = i \cdot C_j$:
          $$Q_j^* = \sqrt{\frac{2 D S}{H_j}}$$
        - If $Q_j^* < q_{\min, j}$, adjust candidate quantity to the price-break threshold $q_{\min, j}$.
        - Select candidate quantity minimizing total annual cost $TC(Q) = D \cdot C_j + \frac{D}{Q}S + \frac{Q}{2}H_j$.

        ---

        #### 3. Safety Stock & Reorder Point (Continuous Review $(s, Q)$)
        - **Constant Lead Time**:
          $$\mu_L = d \cdot L, \quad \sigma_L = \sigma_d \sqrt{L}, \quad SS = Z \cdot \sigma_L, \quad ROP = \mu_L + SS$$
        - **Stochastic Lead Time**:
          $$\sigma_{DL} = \sqrt{L \sigma_d^2 + d^2 \sigma_L^2}, \quad SS = Z \cdot \sigma_{DL}, \quad ROP = d \cdot L + SS$$
        - **Service Level**:
          $$Z = \Phi^{-1}(CSL)$$

        ---

        #### 4. Fixed-Period Review Model (Periodic Review $(P, S)$)
        - **Protection Interval**: $P + L$
        - **Demand over Protection Interval**: $\mu_{P+L} = d(P + L), \quad \sigma_{P+L} = \sigma_d \sqrt{P + L}$
        - **Target Order-Up-To Level**:
          $$S = d(P + L) + Z \cdot \sigma_d \sqrt{P + L}$$
        - **Order Quantity**:
          $$Q = \max(0, S - IP), \quad \text{where } IP = \text{on\_hand} + \text{on\_order} - \text{backorders}$$

        ---

        #### 5. Newsvendor Model (Single-Period Stochastic)
        - **Marginal Costs**:
          $$C_u = \text{Price} - \text{Cost} + \text{Penalty}, \quad C_o = \text{Cost} - \text{Salvage}$$
        - **Critical Fractile / Ratio**:
          $$CR = \frac{C_u}{C_u + C_o}$$
        - **Optimal Order**:
          $$Q^* = \mu + Z \sigma, \quad \text{where } Z = \Phi^{-1}(CR)$$
        - **Expected Payoffs using Standard Normal Loss Function** $L(z) = \phi(z) - z(1 - \Phi(z))$:
          $$E[\text{Shortage}] = \sigma \cdot L(Z), \quad E[\text{Leftovers}] = Q^* - \mu + E[\text{Shortage}], \quad E[\text{Sales}] = \mu - E[\text{Shortage}]$$
        """)


if __name__ == "__main__":
    main()
