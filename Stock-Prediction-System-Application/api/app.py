"""
app.py — Flask REST API for Stock Prediction System
Endpoints:
  GET  /api/health
  GET  /api/stock-info/<ticker>
  POST /api/predict
  GET  /api/report/<ticker>
"""

import os, sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS']      = '1'
os.environ['MKL_NUM_THREADS']      = '1'

# Add parent dir so predictor.py is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import json
import datetime as dt
import traceback

from predictor import predict_stock, get_stock_info

DASHBOARD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dashboard'))
app = Flask(__name__, static_folder=DASHBOARD_DIR, static_url_path='')
CORS(app)

# ─── Serve Frontend ──────────────────────────────────────────────────────────

@app.route('/')
def serve_index():
    return send_from_directory(DASHBOARD_DIR, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(DASHBOARD_DIR, path)


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status':    'ok',
        'timestamp': dt.datetime.now().isoformat(),
        'version':   '2.0',
        'models':    ['LinearRegression', 'GradientBoosting',
                      'RandomForest', 'Ridge', 'VotingEnsemble'],
    })


@app.route('/api/stock-info/<ticker>', methods=['GET'])
def stock_info(ticker):
    try:
        data = get_stock_info(ticker.upper())
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        body   = request.get_json(force=True) or {}
        ticker = body.get('ticker', 'AAPL').upper().strip()
        days   = int(body.get('days', 30))
        period = body.get('period', '2y')

        if days < 1 or days > 365:
            return jsonify({'success': False, 'error': 'days must be 1-365'}), 400

        result = predict_stock(ticker, days, period)
        return jsonify({'success': True, 'data': result})

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/report/<ticker>', methods=['GET'])
def report(ticker):
    """Returns a detailed model report as JSON."""
    try:
        days   = int(request.args.get('days',   30))
        period = request.args.get('period', '2y')
        result = predict_stock(ticker.upper(), days, period)

        report_data = {
            'ticker':        result['ticker'],
            'generated_at':  result['timestamp'],
            'data_summary': {
                'period':         result['period'],
                'total_rows':     result['data_rows'] + result['forecast_days'],
                'train_samples':  result['train_samples'],
                'test_samples':   result['test_samples'],
                'feature_count':  result['feature_count'],
                'features':       result['features'],
            },
            'model_metrics':          result['metrics'],
            'best_model':             result['best_model'],
            'feature_importance':     result['feature_importance'],
            'lr_coefficients':        result['lr_coefficients'],
            'technical_snapshot':     result['technical_snapshot'],
            'current_price':          result['current_price'],
            'forecast_summary': {
                name: {
                    'first_day':  round(fc[0], 4) if fc else None,
                    'last_day':   round(fc[-1], 4) if fc else None,
                    'pct_change': round((fc[-1] - result['current_price'])
                                       / result['current_price'] * 100, 4)
                                  if fc else None,
                }
                for name, fc in result['forecasts'].items()
            },
        }
        return jsonify({'success': True, 'data': report_data})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("  Stock Prediction API  —  v2.0")
    print("  http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)
