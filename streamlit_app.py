"""
streamlit_app.py — Interactive Stock Price Prediction Application
Developed by Masum Ganvir (masumganvir2006@gmail.com)
Powered by 5 ML Models & 19 Technical Indicators
"""

import os
import sys
import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Ensure predictor module is accessible
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Stock-Prediction-System-Application', 'api'))

try:
    from predictor import predict_stock, get_stock_info, FEATURE_COLS
except ImportError:
    st.error("Failed to load predictor module. Please verify predictor.py is present.")
    st.stop()


# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockAI — Stock Price Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS Styling ────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark Glassmorphism Styling */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #111827 0%, #07090e 100%);
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, sans-serif;
    }

    /* Metric Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: rgba(19, 27, 46, 0.7);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 1.1rem;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.6);
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.2;
    }
    .kpi-delta-positive {
        font-size: 0.85rem;
        font-weight: 600;
        color: #10b981;
        margin-top: 0.25rem;
    }
    .kpi-delta-negative {
        font-size: 0.85rem;
        font-weight: 600;
        color: #ef4444;
        margin-top: 0.25rem;
    }

    /* Gradient Header */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #60a5fa 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        font-size: 0.75rem;
        font-weight: 700;
        border-radius: 9999px;
        margin-right: 0.5rem;
    }
    .badge-primary {
        background: rgba(99, 102, 241, 0.2);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.4);
    }
    .badge-success {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        color: #94a3b8;
        background-color: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [aria-selected="true"] {
        background-color: #6366f1 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Cached Helpers ───────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_stock_info(ticker: str):
    return get_stock_info(ticker)

@st.cache_data(ttl=1800, show_spinner=False)
def run_prediction_pipeline(ticker: str, days: int, period: str):
    return predict_stock(ticker, days, period)


# ─── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 StockAI v2.0")
    st.caption("AI-Powered Equity Price Forecasting")
    st.markdown("---")

    ticker_input = st.text_input("Stock Ticker Symbol", value="AAPL", max_chars=15).upper().strip()

    st.markdown("**Quick Suggestions:**")
    quick_cols = st.columns(3)
    if quick_cols[0].button("AAPL"):  ticker_input = "AAPL"
    if quick_cols[1].button("NVDA"):  ticker_input = "NVDA"
    if quick_cols[2].button("TSLA"):  ticker_input = "TSLA"
    
    quick_cols2 = st.columns(3)
    if quick_cols2[0].button("MSFT"): ticker_input = "MSFT"
    if quick_cols2[1].button("GOOGL"):ticker_input = "GOOGL"
    if quick_cols2[2].button("BTC-USD"): ticker_input = "BTC-USD"

    st.markdown("---")
    forecast_days = st.slider("Forecast Horizon (Days)", min_value=7, max_value=180, value=30, step=1)
    history_period = st.selectbox("Historical Training Period", options=["6mo", "1y", "2y", "5y"], index=2)

    selected_models = st.multiselect(
        "Display Forecast Models",
        options=["VotingEnsemble", "GradientBoosting", "RandomForest", "LinearRegression", "Ridge"],
        default=["VotingEnsemble", "GradientBoosting", "RandomForest"]
    )

    predict_btn = st.button("⚡ Run Prediction", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    **Developer:** Masum Ganvir  
    📧 [masumganvir2006@gmail.com](mailto:masumganvir2006@gmail.com)  
    🔗 [GitHub Profile](https://github.com/masumganvir)  
    ⭐ [Project Repo](https://github.com/masumganvir/stock_prediction_system)  
    """)
    st.caption("⚠️ For educational and research purposes only.")


# ─── Main Content Area ────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Intelligent Stock Price Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Multi-model regression powered by 19 quantitative technical indicators and live market data.</div>', unsafe_allow_html=True)

if not ticker_input:
    st.warning("Please enter a stock ticker symbol in the sidebar.")
    st.stop()

# Execution
with st.spinner(f"Fetching real-time market data & training 5 models for {ticker_input}…"):
    try:
        info = fetch_stock_info(ticker_input)
        result = run_prediction_pipeline(ticker_input, forecast_days, history_period)
    except Exception as e:
        st.error(f"Error fetching data or running prediction for **{ticker_input}**: {str(e)}")
        st.info("Tip: For Indian stocks, add `.NS` or `.BO` (e.g. `RELIANCE.NS`, `TCS.NS`). For crypto, use `-USD` (e.g. `BTC-USD`).")
        st.stop()


# ─── Header & Metadata ────────────────────────────────────────────────────────
current_price   = result['current_price']
ensemble_fc     = result['forecasts'].get('VotingEnsemble', [current_price])
final_pred      = ensemble_fc[-1]
price_diff      = final_pred - current_price
pct_change      = (price_diff / current_price) * 100
best_model_name = result['best_model']
best_r2         = result['metrics'][best_model_name]['r2']

company_name = info.get('name', ticker_input)
sector       = info.get('sector', 'Financial Asset')
industry     = info.get('industry', 'N/A')
currency     = info.get('currency', 'USD')

st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; margin-bottom:1rem;">
    <div>
        <h2 style="margin:0; font-size:1.8rem;">{company_name} <span style="color:#818cf8;">({ticker_input})</span></h2>
        <div style="margin-top:0.3rem;">
            <span class="badge badge-primary">{sector}</span>
            <span class="badge badge-primary">{industry}</span>
            <span class="badge badge-success">Currency: {currency}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Metric Cards ─────────────────────────────────────────────────────────────
delta_class = "kpi-delta-positive" if pct_change >= 0 else "kpi-delta-negative"
delta_sign  = "+" if pct_change >= 0 else ""

st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-card">
        <div class="kpi-label">Current Market Price</div>
        <div class="kpi-value">${current_price:,.2f}</div>
        <div class="kpi-delta-positive">Live Adjusted Close</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">{forecast_days}-Day Forecast (Ensemble)</div>
        <div class="kpi-value">${final_pred:,.2f}</div>
        <div class="{delta_class}">{delta_sign}{pct_change:.2f}% (${delta_sign}{price_diff:,.2f})</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Top Performing Model</div>
        <div class="kpi-value" style="font-size:1.25rem; color:#818cf8;">{best_model_name}</div>
        <div class="kpi-delta-positive">Lowest Testing Error</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Best Test Accuracy (R²)</div>
        <div class="kpi-value">{(best_r2 * 100):.1f}%</div>
        <div class="kpi-delta-positive">{best_r2:.4f} Score</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Quantitative Signals</div>
        <div class="kpi-value">{result['feature_count']}</div>
        <div class="kpi-delta-positive">Technical Indicators</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Tab Navigation ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Price Forecast & Trajectory",
    "🤖 Model Performance & Evaluation",
    "📊 Technical Indicators & Feature Importance",
    "🔭 Day-by-Day Forecast Table",
    "📄 Technical Architecture Report"
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: Forecast Chart
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader(f"📈 {ticker_input} Price Trajectory & {forecast_days}-Day Horizon Forecast")

    hist_dates  = result['historical']['dates']
    hist_prices = result['historical']['prices']
    fut_dates   = result['future_dates']

    fig_forecast = go.Figure()

    # Historical trace
    fig_forecast.add_trace(go.Scatter(
        x=hist_dates,
        y=hist_prices,
        mode='lines',
        name='Historical Close',
        line=dict(color='#94a3b8', width=2),
        hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))

    # Model colors
    color_map = {
        'VotingEnsemble':   '#8b5cf6',
        'GradientBoosting': '#10b981',
        'RandomForest':     '#3b82f6',
        'LinearRegression': '#f59e0b',
        'Ridge':            '#ec4899',
    }

    # Add selected model forecasts
    for m in selected_models:
        if m in result['forecasts']:
            fc_y = result['forecasts'][m]
            # Connect last historical point to first forecast point
            conn_x = [hist_dates[-1]] + fut_dates
            conn_y = [hist_prices[-1]] + fc_y

            fig_forecast.add_trace(go.Scatter(
                x=conn_x,
                y=conn_y,
                mode='lines+markers' if len(fut_dates) <= 15 else 'lines',
                name=f'{m} Forecast',
                line=dict(color=color_map.get(m, '#ffffff'), width=3 if m == 'VotingEnsemble' else 2,
                          dash='solid' if m == 'VotingEnsemble' else 'dash'),
                marker=dict(size=4),
                hovertemplate=f'{m}: ' + '$%{y:.2f}<br>Date: %{x}<extra></extra>'
            ))

    # Add vertical divider for forecast start
    fig_forecast.add_vline(
        x=hist_dates[-1],
        line_width=1.5,
        line_dash="dot",
        line_color="#e2e8f0",
        annotation_text="Today (Forecast Start)",
        annotation_position="top left",
        annotation_font_color="#cbd5e1"
    )

    fig_forecast.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)", rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(step="all", label="All")
            ])
        )),
        yaxis=dict(gridcolor="rgba(255,255,255,0.07)", title="Price (USD)", tickprefix="$")
    )

    st.plotly_chart(fig_forecast, use_container_width=True)

    # Forecast Range Insights
    all_fc_values = [v for m in selected_models if m in result['forecasts'] for v in result['forecasts'][m]]
    if all_fc_values:
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Min Projected Price", f"${min(all_fc_values):,.2f}")
        col_b.metric("Max Projected Price", f"${max(all_fc_values):,.2f}")
        col_c.metric("Ensemble Final Target", f"${final_pred:,.2f}")
        col_d.metric("Forecast Dispersion (Spread)", f"${(max(all_fc_values) - min(all_fc_values)):,.2f}")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: Model Performance & Comparison
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🤖 Model Evaluation Metrics on Out-of-Sample Test Set")

    metrics_df = pd.DataFrame([
        {
            "Model": m,
            "MAE ($)": f"${data['mae']:.2f}",
            "MSE": f"{data['mse']:.2f}",
            "RMSE ($)": f"${data['rmse']:.2f}",
            "Test R² Score": f"{data['r2']:.4f}",
            "Train R² Score": f"{data['train_r2']:.4f}",
            "Accuracy %": f"{(data['r2'] * 100):.2f}%"
        }
        for m, data in result['metrics'].items()
    ])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Test Set $R^2$ Score Comparison (Higher is better)**")
        r2_vals = [result['metrics'][m]['r2'] for m in result['metrics']]
        model_names = list(result['metrics'].keys())
        fig_r2 = go.Figure(go.Bar(
            x=r2_vals,
            y=model_names,
            orientation='h',
            marker_color=['#10b981' if m == best_model_name else '#6366f1' for m in model_names],
            text=[f"{v:.4f}" for v in r2_vals],
            textposition='inside'
        ))
        fig_r2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
            xaxis=dict(title="R² Score", range=[0, 1])
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    with col2:
        st.markdown("**Mean Absolute Error Comparison (Lower is better)**")
        mae_vals = [result['metrics'][m]['mae'] for m in result['metrics']]
        fig_mae = go.Figure(go.Bar(
            x=mae_vals,
            y=model_names,
            orientation='h',
            marker_color=['#38bdf8' if m == best_model_name else '#f43f5e' for m in model_names],
            text=[f"${v:.2f}" for v in mae_vals],
            textposition='inside'
        ))
        fig_mae.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
            xaxis=dict(title="MAE ($)")
        )
        st.plotly_chart(fig_mae, use_container_width=True)

    # Actual vs Predicted Test Plot
    st.markdown("---")
    st.markdown(f"**Test Set Actual vs. Predicted Prices ({best_model_name})**")
    actuals = result['test_actual']
    preds   = result['test_predicted'].get(best_model_name, [])

    fig_test = go.Figure()
    fig_test.add_trace(go.Scatter(y=actuals, mode='lines', name='Actual Test Prices', line=dict(color='#94a3b8', width=2)))
    fig_test.add_trace(go.Scatter(y=preds, mode='lines', name=f'{best_model_name} Predicted', line=dict(color='#10b981', width=2, dash='dot')))
    fig_test.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(title="Test Samples (Chronological)"),
        yaxis=dict(title="Price (USD)", tickprefix="$")
    )
    st.plotly_chart(fig_test, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: Technical Indicators & Feature Importance
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📊 Quantitative Signals & Feature Importance Breakdown")

    col_fi, col_snap = st.columns([1.2, 0.8])

    with col_fi:
        st.markdown("**Feature Importance via Gradient Boosting Impurity Reduction (ΔMDI)**")
        fi = result['feature_importance']
        top_features = list(fi.keys())[:12]
        top_weights  = [fi[k] * 100 for k in top_features]

        fig_fi = go.Figure(go.Bar(
            x=top_weights[::-1],
            y=top_features[::-1],
            orientation='h',
            marker=dict(color=top_weights[::-1], colorscale='Viridis'),
            text=[f"{w:.1f}%" for w in top_weights[::-1]],
            textposition='outside'
        ))
        fig_fi.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            margin=dict(l=20, r=40, t=10, b=20),
            height=400,
            xaxis=dict(title="Importance (%)")
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_snap:
        st.markdown("**Live Technical Indicator Snapshot**")
        snap = result.get('technical_snapshot', {})
        for name, val in snap.items():
            if val is not None:
                st.metric(name, f"{val:,.2f}")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4: Day-by-Day Forecast Table & CSV Download
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader(f"🔭 {forecast_days}-Day Day-by-Day Projection Table")

    table_rows = []
    prev_price = current_price

    for idx, (date, fc_price) in enumerate(zip(fut_dates, ensemble_fc), 1):
        day_diff = fc_price - prev_price
        day_pct  = (day_diff / prev_price) * 100
        trend    = "🟢 Bullish" if day_diff >= 0 else "🔴 Bearish"

        table_rows.append({
            "Day #": idx,
            "Date": date,
            "Predicted Price ($)": round(fc_price, 2),
            "Daily Change ($)": round(day_diff, 2),
            "Daily Change (%)": f"{day_pct:+.2f}%",
            "Projected Trend": trend
        })
        prev_price = fc_price

    df_table = pd.DataFrame(table_rows)

    csv_data = df_table.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Forecast as CSV",
        data=csv_data,
        file_name=f"{ticker_input}_forecast_{dt.datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        type="primary"
    )

    st.dataframe(df_table, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 5: Technical Architecture Report
# ──────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("📄 Model & System Architecture Report")

    st.markdown(r"""
    ### 1. Problem Formulation: Backward Shift Technique
    Stock forecasting is framed as a supervised regression task over forward horizon $h$:
    $$y_t = P_{t + h}$$
    Using `df['Adj_Close'].shift(-forecast_out)`:
    - Rows $t = 1 \dots (T - h)$ possess known pairs $(X_t, y_t)$, used for training and test evaluation.
    - The **last $h$ rows** ($t = T - h + 1 \dots T$) have no known future targets ($y_t = \\text{NaN}$). They represent the **current market regime ($X_{\\text{forecast}}$)**. Passing this matrix into our models generates future prices.

    ---

    ### 2. The 19 Engineered Features
    - **Trend (SMA & EMA):** `MA_5`, `MA_10`, `MA_20`, `MA_50`, `EMA_12`, `EMA_26`
    - **Momentum & Velocity:** `MACD`, `MACD Signal`, `RSI (14)`, `Momentum_5`, `ROC_10`
    - **Volatility:** `BB_upper`, `BB_lower`, `BB_width`, `Volatility_20`, `Price_MA20_dist`
    - **Volume & Liquidity:** `Volume`, `Volume_ratio` (relative to 10-day SMA)

    ---

    ### 3. The 5 Predictive Algorithms
    1. **Multiple Linear Regression (OLS):** Computes linear hyperplane minimizing squared residuals.
    2. **Ridge Regression:** Adds L2 shrinkage penalty to prevent instability from collinear moving averages.
    3. **Gradient Boosting Regressor (GBR):** Sequentially trains 300 decision trees to minimize residual loss gradients. Accurately detects non-linear mean reversion triggers.
    4. **Random Forest Regressor:** Averages 200 de-correlated bootstrap trees, reducing variance.
    5. **Voting Ensemble:** Blends non-linear tree models with regularized linear regression for the smoothest forecast curves.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "StockAI v2.0 · Developed by <strong>Masum Ganvir</strong> (masumganvir2006@gmail.com) · "
    "<a href='https://github.com/masumganvir/stock_prediction_system' target='_blank' style='color:#818cf8;'>GitHub Repository</a>"
    "</div>",
    unsafe_allow_html=True
)
