# 📈 Stock Price Prediction System

An AI-powered financial market forecasting web application that predicts future stock prices using **5 Machine Learning models** trained on real-time Yahoo Finance market data and **19 technical indicators**.

---

## 🚀 Key Features

- **🤖 5 Machine Learning Models:**
  - Multiple Linear Regression (Baseline)
  - Ridge Regression (L2 Regularization)
  - Random Forest Regressor (Bagging)
  - Gradient Boosting Regressor (Sequential Trees — **Best $R^2 \approx 0.97$**)
  - Voting Ensemble (Balanced synthesis of tree and linear regressors)

- **📊 19 Technical Indicators:**
  - **Trend:** Simple Moving Averages (`MA_5`, `MA_10`, `MA_20`, `MA_50`), Exponential Moving Averages (`EMA_12`, `EMA_26`)
  - **Momentum:** `MACD`, `MACD Signal`, `RSI (14)`, `Momentum (5-day)`, `Rate of Change (ROC_10)`
  - **Volatility:** Bollinger Bands (`Upper`, `Lower`, `Width`), `20-day Rolling Volatility`
  - **Volume & Liquidity:** Raw Volume, `Volume Ratio` (vs 10-day SMA)

- **⚡ Interactive Web Dashboard:**
  - Real-time stock search (`AAPL`, `MSFT`, `TSLA`, `NVDA`, `RELIANCE.NS`, `BTC-USD`, etc.)
  - Adjustable forecast horizon slider (7 to 180 days)
  - Interactive Chart.js graphs: Forecast Curves, Actual vs. Predicted, Residual Analysis
  - Technical indicator overlays (Bollinger Bands, MACD, RSI)
  - Day-by-Day forecast table with **CSV Export**
  - Downloadable/Printable quantitative prediction report

- **📡 Live Data Ingestion:**
  - Real-time market data retrieved on-the-fly via Yahoo Finance (`yfinance`).

---

## 🛠️ Tech Stack

- **Backend:** Python 3, Flask, Flask-CORS, REST API
- **Machine Learning & Data:** Scikit-Learn, Pandas, NumPy, yfinance
- **Frontend:** HTML5, Modern CSS3 (Dark Theme & Glassmorphism), JavaScript (ES6+), Chart.js

---

## 💻 Quick Start & Installation

### Option A: Launch Interactive Streamlit App (Recommended)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=masumganvir/stock_prediction_system&branch=main&mainModule=streamlit_app.py)

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run Streamlit Application:**
   ```bash
   streamlit run streamlit_app.py
   ```
3. **Open in Browser:**
   Visit `http://localhost:8501`

---

### Option B: Flask + HTML5/CSS3 Dashboard

1. **Navigate to App Directory:**
   ```bash
   cd Stock-Prediction-System-Application
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Start Flask Server:**
   ```bash
   python api/app.py
   ```
4. **Open in Browser:**
   Visit `http://127.0.0.1:5000`

---

## 📁 Project Structure

```text
stock_prediction_system/
├── Stock-Prediction-System-Application/
│   ├── api/
│   │   ├── app.py                # Flask REST API server
│   │   └── predictor.py          # Core ML pipeline (19 features, 5 models)
│   ├── dashboard/
│   │   ├── index.html            # Web dashboard interface
│   │   ├── style.css             # Glassmorphism styling & animations
│   │   └── app.js                # Frontend client & Chart.js logic
│   ├── app/                      # Django application (views, templates, models)
│   ├── core/                     # Django project configuration
│   ├── manage.py                 # Django management script
│   ├── requirements.txt          # Python dependencies
│   ├── Stock_Prediction_System.ipynb # Research & exploratory notebook
│   └── MODEL_PREDICTION_REPORT.md    # Detailed mathematical model report
├── .gitignore                    # Git ignore file
└── README.md                     # Project documentation
```

---

## 📈 Model Performance Benchmark (AAPL)

| Model | MAE ($) | RMSE ($) | Test $R^2$ Score | Status |
|---|---|---|---|---|
| **Gradient Boosting** | **\$3.26** | **\$4.50** | **0.974** | 🏆 Best Model |
| **Random Forest** | \$4.00 | \$5.53 | 0.961 | Excellent |
| **Voting Ensemble** | \$5.16 | \$6.81 | 0.941 | Recommended |
| **Linear Regression** | \$8.17 | \$11.31 | 0.838 | Baseline |
| **Ridge Regression** | \$9.17 | \$11.97 | 0.818 | Regularized |

---

## 👤 Author & Developer

- **Name:** Masum Ganvir
- **Email:** [masumganvir2006@gmail.com](mailto:masumganvir2006@gmail.com)
- **GitHub:** [@masumganvir](https://github.com/masumganvir)
- **Repository:** [masumganvir/stock_prediction_system](https://github.com/masumganvir/stock_prediction_system)

---

## ⚠️ Disclaimer

*This software is developed strictly for educational and research purposes. It does not constitute financial or investment advice. Always perform your own due diligence before making financial decisions.*
