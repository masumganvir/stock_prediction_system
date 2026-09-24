"""
predictor.py
Core ML prediction engine extracted from app/views.py predict() function.
Supports 5 models: LinearRegression (original), GradientBoosting, RandomForest, Ridge, VotingEnsemble
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS']      = '1'
os.environ['MKL_NUM_THREADS']      = '1'
os.environ['NUMEXPR_NUM_THREADS']  = '1'

import datetime as dt
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.linear_model import LinearRegression, Ridge
from sklearn import preprocessing, model_selection
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build 19 technical indicator features from raw OHLCV data."""
    feature_col = 'Close' if 'Close' in raw_df.columns else 'Adj Close'
    df = raw_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.rename(columns={feature_col: 'Adj_Close'}, inplace=True)
    df.dropna(inplace=True)

    df['MA_5']  = df['Adj_Close'].rolling(5).mean()
    df['MA_10'] = df['Adj_Close'].rolling(10).mean()
    df['MA_20'] = df['Adj_Close'].rolling(20).mean()
    df['MA_50'] = df['Adj_Close'].rolling(50).mean()

    df['EMA_12'] = df['Adj_Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Adj_Close'].ewm(span=26, adjust=False).mean()
    df['MACD']   = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    bb_std = df['Adj_Close'].rolling(20).std()
    df['BB_upper'] = df['MA_20'] + 2 * bb_std
    df['BB_lower'] = df['MA_20'] - 2 * bb_std
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / (df['MA_20'] + 1e-8)

    delta  = df['Adj_Close'].diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.rolling(14).mean()
    avg_l  = loss.rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + avg_g / (avg_l + 1e-8)))

    df['Momentum_5']    = df['Adj_Close'] - df['Adj_Close'].shift(5)
    df['ROC_10']        = df['Adj_Close'].pct_change(10) * 100
    df['Daily_Return']  = df['Adj_Close'].pct_change() * 100
    df['Volatility_20'] = df['Daily_Return'].rolling(20).std()
    df['Price_MA20_dist'] = (df['Adj_Close'] - df['MA_20']) / (df['MA_20'] + 1e-8) * 100

    if 'Volume' in df.columns:
        df['Volume_MA_10'] = df['Volume'].rolling(10).mean()
        df['Volume_ratio'] = df['Volume'] / (df['Volume_MA_10'] + 1)

    return df


FEATURE_COLS = [
    'Adj_Close', 'MA_5', 'MA_10', 'MA_20', 'MA_50',
    'EMA_12', 'EMA_26', 'MACD', 'MACD_signal',
    'BB_upper', 'BB_lower', 'BB_width', 'RSI',
    'Momentum_5', 'ROC_10', 'Volatility_20', 'Price_MA20_dist',
    'Volume_ratio', 'Volume',
]


def predict_stock(ticker: str, number_of_days: int = 30, period: str = '2y'):
    """
    Full prediction pipeline matching views.py predict() logic + enhancements.
    Returns dict with all model results, metrics and forecast.
    """
    ticker = ticker.upper().strip()

    # Download data (original approach from views.py)
    raw_df = yf.download(
        tickers=ticker, period=period, interval='1d',
        progress=False, auto_adjust=True
    )
    if raw_df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'")

    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)
    raw_df.index = pd.to_datetime(raw_df.index)

    # Build features
    df = _engineer_features(raw_df)

    # Determine available feature cols
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]

    forecast_out = int(number_of_days)
    df['Prediction'] = df['Adj_Close'].shift(-forecast_out)
    df_clean = df.dropna()

    df_features = df_clean[feat_cols + ['Prediction']].dropna()
    X_all = np.array(df_features[feat_cols])
    y_all = np.array(df_features['Prediction'])

    # Scale — original: preprocessing.scale(X)
    X_all_scaled = preprocessing.scale(X_all)

    X_forecast = X_all_scaled[-forecast_out:]
    X          = X_all_scaled[:-forecast_out]
    y          = y_all[:-forecast_out]

    # Split — original: train_test_split(X, y, test_size=0.2)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build models
    clf      = LinearRegression()
    gbr      = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                          max_depth=4, subsample=0.8, random_state=42)
    rfr      = RandomForestRegressor(n_estimators=200, max_depth=8,
                                      random_state=42, n_jobs=1)
    ridge    = Ridge(alpha=1.0)
    ensemble = VotingRegressor([('gbr', gbr), ('rfr', rfr), ('ridge', ridge)])

    models = {
        'LinearRegression': clf,
        'GradientBoosting': gbr,
        'RandomForest':     rfr,
        'Ridge':            ridge,
        'VotingEnsemble':   ensemble,
    }

    metrics    = {}
    forecasts  = {}
    test_preds = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred   = model.predict(X_test)
        fc       = model.predict(X_forecast).tolist()
        mae      = float(mean_absolute_error(y_test, y_pred))
        mse      = float(mean_squared_error(y_test, y_pred))
        rmse     = float(np.sqrt(mse))
        r2       = float(r2_score(y_test, y_pred))
        metrics[name]    = {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2,
                             'train_r2': float(model.score(X_train, y_train))}
        forecasts[name]  = fc
        test_preds[name] = y_pred.tolist()

    # Best model by R²
    best_name = max(metrics, key=lambda k: metrics[k]['r2'])

    # Historical prices (last 120 bars)
    hist = df_clean['Adj_Close'].dropna().tail(120)
    historical = {
        'dates':  [str(d.date()) for d in hist.index],
        'prices': [round(float(v), 4) for v in hist.values],
    }

    # Future dates
    today = dt.datetime.today()
    future_dates = [(today + dt.timedelta(days=i+1)).strftime('%Y-%m-%d')
                    for i in range(forecast_out)]

    # Feature importance (from GBR)
    fi = dict(zip(feat_cols, gbr.feature_importances_.tolist()))
    fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    # LR coefficients
    lr_coef = dict(zip(feat_cols, clf.coef_.tolist()))

    # Technical indicator snapshot (last row)
    last = df_clean.iloc[-1]
    snapshot = {}
    for col in ['Adj_Close','RSI','MACD','MACD_signal','BB_upper','BB_lower',
                'MA_20','MA_50','Volatility_20','Volume_ratio']:
        if col in df_clean.columns:
            v = last[col]
            snapshot[col] = round(float(v), 4) if pd.notna(v) else None

    # Test actual prices
    y_test_list = y_test.tolist()

    return {
        'ticker':          ticker,
        'period':          period,
        'forecast_days':   forecast_out,
        'data_rows':       int(len(X)),
        'train_samples':   int(len(X_train)),
        'test_samples':    int(len(X_test)),
        'feature_count':   len(feat_cols),
        'features':        feat_cols,
        'current_price':   round(float(df_clean['Adj_Close'].iloc[-1]), 4),
        'historical':      historical,
        'future_dates':    future_dates,
        'forecasts':       forecasts,
        'metrics':         metrics,
        'best_model':      best_name,
        'test_actual':     [round(v, 4) for v in y_test_list],
        'test_predicted':  {k: [round(v,4) for v in v_list]
                            for k, v_list in test_preds.items()},
        'feature_importance': fi_sorted,
        'lr_coefficients':    lr_coef,
        'technical_snapshot': snapshot,
        'timestamp':       dt.datetime.now().isoformat(),
    }


def get_stock_info(ticker: str):
    """Get basic stock metadata from yfinance."""
    try:
        info = yf.Ticker(ticker).info
        return {
            'ticker':      ticker.upper(),
            'name':        info.get('longName', ticker),
            'sector':      info.get('sector', 'N/A'),
            'industry':    info.get('industry', 'N/A'),
            'country':     info.get('country', 'N/A'),
            'currency':    info.get('currency', 'USD'),
            'market_cap':  info.get('marketCap'),
            'pe_ratio':    info.get('trailingPE'),
            'week52_high': info.get('fiftyTwoWeekHigh'),
            'week52_low':  info.get('fiftyTwoWeekLow'),
            'avg_volume':  info.get('averageVolume'),
            'description': info.get('longBusinessSummary', '')[:400],
        }
    except Exception:
        return {'ticker': ticker.upper(), 'name': ticker}
