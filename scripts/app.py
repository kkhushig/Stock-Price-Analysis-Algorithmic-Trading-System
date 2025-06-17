from lab import (validation, fetchPreprocess, calculateEMA,
                              prepdataLSTM, LSTMModel, mockEMA,
                              mockLSTM, predictDates, predictPrices, metrics,
                              plotting)
from flask import Flask, request, jsonify, send_from_directory
import base64
import io
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from pandas.tseries.offsets import BDay
from flask_cors import CORS
from io import StringIO
import sys
import plotly
import plotly.graph_objects as go
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)


portfolios = {}

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/styles.css')
def styles():
    return send_from_directory('.', 'styles.css')

@app.route('/script.js')
def script():
    return send_from_directory('.', 'script.js')

@app.route('/add_stock', methods=['POST'])
def add_stock():
    print("Im here")
    data = request.json
    stock_name = data['stockName']
    stock_symbol = data['stockSymbol']
    
    if validation(stock_symbol):
        if stock_name not in portfolios:
            portfolios[stock_name] = []
        portfolios[stock_name].append({'symbol': stock_symbol})
        return jsonify({'success': True, 'message': f"{stock_name} ({stock_symbol}) stock added to portfolio."})
    else:
        return jsonify({'success': False, 'message': f"Invalid stock symbol: {stock_symbol}"})

@app.route('/remove_stock', methods=['POST'])
def remove_stock():
    data = request.json
    stockName = data['stockName']
    stock_symbol = data['stockSymbol']
    
    if stockName in portfolios:
        portfolios[stockName] = [stock for stock in portfolios[stockName] if stock['symbol'] != stock_symbol]
        return jsonify({'success': True, 'message': f"Stock {stockName} removed from portfolio."})
    else:
        return jsonify({'success': False, 'message': f"{stockName} not found in portfolio."})

@app.route('/view_portfolios', methods=['GET'])
def view_portfolios():
    return jsonify(portfolios)

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    data = request.json
    stock_name = data['stockName']
    start_date = data['startDate']
    end_date = data['endDate']
    initial_investment = float(data['initialInvestment'])
    window_size = int(data['windowSize'])
    algorithm = data['algorithm']
    num_prediction_days = int(data['numPredictionDays'])
    
    if stock_name not in portfolios:
        return jsonify({'success': False, 'message': f"{stock_name} not found in portfolio."})
    
    stocks = portfolios[stock_name]
    all_stock_data = pd.DataFrame()
    
    for stock in stocks:
        stock_data = fetchPreprocess(stock['symbol'], start_date, end_date)
        if stock_data is not None:
            all_stock_data = pd.concat([all_stock_data, stock_data], ignore_index=True)
    
    if all_stock_data.empty:
        return jsonify({'success': False, 'message': "No data fetched for the selected portfolio and date range."})
    
    old_stdout = sys.stdout
    sys.stdout = output = StringIO()
    
    if algorithm == "EMA":
        ema = calculateEMA(all_stock_data, window_size)
        ema_predictions = ema.tolist()
        
        buy_signals, sell_signals = [], []
        for i in range(1, len(all_stock_data)):
            if all_stock_data['close'].iloc[i] > ema[i] and all_stock_data['close'].iloc[i-1] <= ema[i-1]:
                buy_signals.append(i)
            elif all_stock_data['close'].iloc[i] < ema[i] and all_stock_data['close'].iloc[i-1] >= ema[i-1]:
                sell_signals.append(i)
        
        final_shares, portfolio_value = mockEMA(all_stock_data, initial_investment, window_size)
        
        # Predict future prices using EMA
        future_data = pd.DataFrame()
        future_data['date'] = pd.date_range(start=all_stock_data['date'].iloc[-1] + pd.Timedelta(days=1), periods=num_prediction_days)
        last_ema = ema_predictions[-1]
        future_prices = []
        for i in range(num_prediction_days):
            if i == 0:
                future_price = last_ema
            else:
                future_price = (all_stock_data['close'].iloc[-1] * (2 / (window_size + 1)) + 
                                last_ema * (1 - (2 / (window_size + 1))))
            future_prices.append(future_price)
            last_ema = future_price
        future_data['predicted_close'] = future_prices
    elif algorithm == "LSTM":
        lstm_lookback = window_size
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_close = scaler.fit_transform(all_stock_data['close'].values.reshape(-1, 1))
        X, y = prepdataLSTM(scaled_close, lstm_lookback)
        lstm_model = LSTMModel((X.shape[1], 1))
        lstm_model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        
        # Calculate LSTM predictions for historical data
        lstm_predictions = []
        buy_signals, sell_signals = [], []
        for i in range(lstm_lookback, len(all_stock_data)):
            lstm_input = scaled_close[i - lstm_lookback:i].reshape(1, lstm_lookback, 1)
            lstm_predict = lstm_model.predict(lstm_input, verbose=0)[0][0]
            lstm_predictions.append(scaler.inverse_transform([[lstm_predict]])[0][0])
            
            # Generate buy/sell signals
            if i > lstm_lookback:
                if lstm_predictions[-1] > all_stock_data['close'].iloc[i-1] and lstm_predictions[-2] <= all_stock_data['close'].iloc[i-2]:
                    buy_signals.append(i)
                elif lstm_predictions[-1] < all_stock_data['close'].iloc[i-1] and lstm_predictions[-2] >= all_stock_data['close'].iloc[i-2]:
                    sell_signals.append(i)
        
        # Pad the predictions with NaN for the first lstm_lookback days
        lstm_predictions = [np.nan] * lstm_lookback + lstm_predictions
        
        final_shares, portfolio_value = mockLSTM(all_stock_data, initial_investment, lstm_model, lstm_lookback)
        future_data = predictPrices(all_stock_data, lstm_model, lstm_lookback, num_prediction_days)

        # Generate buy/sell signals for LSTM
        buy_signals, sell_signals = [], []
        for i in range(lstm_lookback, len(all_stock_data)):
            lstm_input = scaled_close[i - lstm_lookback:i].reshape(1, lstm_lookback, 1)
            lstm_predict = lstm_model.predict(lstm_input, verbose=0)[0][0]
            if lstm_predict > scaled_close[i] and (i == lstm_lookback or lstm_model.predict(scaled_close[i-lstm_lookback-1:i-1].reshape(1, lstm_lookback, 1), verbose=0)[0][0] <= scaled_close[i-1]):
                buy_signals.append(i)
            elif lstm_predict < scaled_close[i] and (i == lstm_lookback or lstm_model.predict(scaled_close[i-lstm_lookback-1:i-1].reshape(1, lstm_lookback, 1), verbose=0)[0][0] >= scaled_close[i-1]):
                sell_signals.append(i)

    simulation_output = output.getvalue()
    sys.stdout = old_stdout
    
    total_return, annual_return, sharpe_ratio = metrics(all_stock_data, initial_investment, portfolio_value)
    
    # Create Plotly figure
    fig = go.Figure()

    # Plot close price
    fig.add_trace(go.Scatter(x=all_stock_data['date'], y=all_stock_data['close'], mode='lines', name='Close Price'))

    if algorithm == "EMA":
        fig.add_trace(go.Scatter(x=all_stock_data['date'], y=ema_predictions, mode='lines', name='EMA'))
    elif algorithm == "LSTM":
        fig.add_trace(go.Scatter(x=all_stock_data['date'], y=lstm_predictions, mode='lines', name='LSTM Prediction'))

    # Plot buy and sell signals
    fig.add_trace(go.Scatter(x=all_stock_data['date'].iloc[buy_signals], y=all_stock_data['close'].iloc[buy_signals],
                            mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', size=10, color='green')))
    fig.add_trace(go.Scatter(x=all_stock_data['date'].iloc[sell_signals], y=all_stock_data['close'].iloc[sell_signals],
                            mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', size=10, color='red')))

    # Plot future predictions
    fig.add_trace(go.Scatter(x=future_data['date'], y=future_data['predicted_close'], mode='lines', name='Future Predictions', line=dict(dash='dash')))

    fig.update_layout(title=f"{algorithm} Trading Signals", xaxis_title="Date", yaxis_title="Price")

    chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return json.dumps({
        'success': True,
        'metrics': {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio
        },
        'chart': chart_json,
        'simulation_output': simulation_output,
        'future_predictions': future_data.to_dict(orient='records'),
    }, cls=CustomJSONEncoder)

if __name__ == '__main__':
    app.run(debug=True)