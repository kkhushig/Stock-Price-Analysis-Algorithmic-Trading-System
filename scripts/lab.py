import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pandas.tseries.offsets import BDay

# Check if stock is valid
def validation(stockSymbol):
    try:
        stock = yf.Ticker(stockSymbol)
        stockInfo = stock.info
        if 'shortName' in stockInfo or 'longName' in stockInfo:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error occurred: {e}")
        return False

# Fetch and preprocess stock data
def fetchPreprocess(stockSymbol, SD, ED):
    try:
        stockData = yf.download(stockSymbol, start=SD, end=ED)
        
        if stockData.empty:
            print(f"No data found for {stockSymbol}")
            return None
        
        stockData.fillna(method='ffill', inplace=True)
        stockData['Daily Return'] = stockData['Adj Close'].pct_change().fillna(0)
        stockData.reset_index(inplace=True)
        stockData.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }, inplace=True)
        stockData['stock_symbol'] = stockSymbol
        stockData = stockData[['stock_symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'Daily Return']]
        
        return stockData
    
    except Exception as e:
        print(f"Error fetching data for {stockSymbol}: {e}")
        return None

# Calculate EMA
def calculateEMA(data, window):
    ema = data['close'].ewm(span=window, adjust=False).mean()
    return ema

# Prepare data for LSTM
def prepdataLSTM(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback - 1):
        X.append(data[i:(i + lookback), 0])
        y.append(data[i + lookback, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

# Create LSTM model
def LSTMModel(inputShape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=inputShape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def mockEMA(data, initial, EMAwindow):
    sharesHeld = 0
    balance = initial
    portfolioValue = initial
    portfolioValues = []

    ema = calculateEMA(data, EMAwindow)

    buying = []
    selling = []

    for i in range(len(data)):
        date = data['date'][i]
        
        if data['close'][i] > ema[i] and sharesHeld == 0:
            boughtShares = balance // data['close'][i]
            sharesHeld += boughtShares
            balance -= boughtShares * data['close'][i]
            print(f"Bought {boughtShares} shares on {date.strftime('%Y-%m-%d')} at {data['close'][i]}")
            buying.append(i)

        elif data['close'][i] < ema[i] and sharesHeld > 0:
            balance += sharesHeld * data['close'][i]
            print(f"Sold {sharesHeld} shares on {date.strftime('%Y-%m-%d')} at {data['close'][i]}")
            selling.append(i)
            sharesHeld = 0
        else:
            print(f"Holding {sharesHeld} shares on {date.strftime('%Y-%m-%d')}")

        portfolioValue = balance + sharesHeld * data['close'][i]
        portfolioValues.append(portfolioValue)

    # Plot the buy and sell signals, passing the portfolio values
    plotting(data, buying, selling, portfolioValues, title="EMA Trading Signals")

    return sharesHeld, portfolioValue

def mockLSTM(data, initial, modelLSTM, LSTMLookback):
    sharesHeld = 0
    balance = initial
    portfolioValue = initial
    portfolioValues = []

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaledClose = scaler.fit_transform(data['close'].values.reshape(-1, 1))

    buying = []
    selling = []

    for i in range(len(data)):
        date = data['date'][i]

        if i < LSTMLookback:
            portfolioValues.append(portfolioValue)
            continue

        LSTMinput = scaledClose[i - LSTMLookback:i].reshape(1, LSTMLookback, 1)
        LSTMpredict = modelLSTM.predict(LSTMinput, verbose=0)[0][0]
        LSTMsignal = LSTMpredict > scaledClose[i]

        if LSTMsignal and sharesHeld == 0:
            boughtShares = balance // data['close'][i]
            sharesHeld += boughtShares
            balance -= boughtShares * data['close'][i]
            print(f"Bought {boughtShares} shares on {date.strftime('%Y-%m-%d')} at {data['close'][i]}")
            buying.append(i)

        elif not LSTMsignal and sharesHeld > 0:
            balance += sharesHeld * data['close'][i]
            print(f"Sold {sharesHeld} shares on {date.strftime('%Y-%m-%d')} at {data['close'][i]}")
            selling.append(i)
            sharesHeld = 0
        else:
            print(f"Holding {sharesHeld} shares on {date.strftime('%Y-%m-%d')}")

        portfolioValue = balance + sharesHeld * data['close'][i]
        portfolioValues.append(portfolioValue)

    # Plot the buy and sell signals, passing the portfolio values
    plotting(data, buying, selling, portfolioValues, title="LSTM Trading Signals")

    return sharesHeld, portfolioValue

def predictDates(lastDate, noOfDays):
    futureDates = pd.date_range(start=lastDate + BDay(), periods=noOfDays, freq=BDay())
    return futureDates

# Predict future stock prices
def predictPrices(data, lstm_model, lstm_lookback, num_days):
    futureDates = predictDates(data['date'].iloc[-1], num_days)
    futurePrices = []

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaledClose = scaler.fit_transform(data['close'].values.reshape(-1, 1))

    for i in range(num_days):
        LSTMinput = scaledClose[-lstm_lookback:].reshape(1, lstm_lookback, 1)
        LSTMpredict = lstm_model.predict(LSTMinput, verbose=0)[0][0]
        futureprice = scaler.inverse_transform([[LSTMpredict]])[0][0]
        futurePrices.append(futureprice)

        # Update the scaled_close to include the new prediction
        scaledClose = np.append(scaledClose, LSTMpredict).reshape(-1, 1)

    futureData = pd.DataFrame({
        'date': futureDates,
        'predicted_close': futurePrices
    })

    return futureData

# Calculate metrics
def metrics(data, intial, portfolioValue):
    totalReturn = portfolioValue - intial
    annualReturn = (portfolioValue / intial) ** (252 / len(data)) - 1
    dailyReturns = data['Daily Return']
    SHARPE = np.sqrt(252) * dailyReturns.mean() / dailyReturns.std()
    
    print(f"\nTotal Return: ${totalReturn:.2f}")
    print(f"Annual Return: {annualReturn:.2%}")
    print(f"Sharpe Ratio: {SHARPE:.2f}")

    return round(totalReturn, 2), round(annualReturn, 2), round(SHARPE, 2)
 

# Enhanced function to plot stock price along with buy/sell signals and other useful insights (without shaded background)
def plotting(data, buy_signals, sell_signals, portfolio_values, ema_short=None, ema_long=None, title="Trading Signals"):
    fig, ax1 = plt.subplots(figsize=(14, 9))
    
    # Plot the stock price
    ax1.plot(data['date'], data['close'], label='Stock Price', color='blue', alpha=0.5)
    
    # Plot the short-term EMA, if provided
    if ema_short is not None:
        ax1.plot(data['date'], ema_short, label='Short-Term EMA', color='purple', linestyle='--', alpha=0.7)
    
    # Plot the long-term EMA, if provided
    if ema_long is not None:
        ax1.plot(data['date'], ema_long, label='Long-Term EMA', color='orange', linestyle='--', alpha=0.7)
    
    # Highlight buy and sell signals
    ax1.scatter(data['date'][buy_signals], data['close'][buy_signals], label='Buy Signal', marker='^', color='limegreen', lw=3)
    ax1.scatter(data['date'][sell_signals], data['close'][sell_signals], label='Sell Signal', marker='v', color='crimson', lw=3)

    # Set labels and title
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.set_title(title)
    ax1.grid(True)
    
    # Rotate date labels for better readability
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Add portfolio value subplot
    ax2 = ax1.twinx()  # Create another y-axis for the portfolio value
    ax2.plot(data['date'], portfolio_values, label='Portfolio Value', color='red', alpha=0.3, lw=2)
    ax2.set_ylabel("Portfolio Value")
    
    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.show()

# Main menu to manage portfolios and run simulation
if __name__ == "__main__":
    portfolios = {}
    selectAlgo = "EMA"
    
    while True:
        print("\nPortfolio Management Options:")
        print("1. Add stock to portfolio")
        print("2. Remove stock from portfolio")
        print("3. View portfolios")
        print("4. Select trading algorithm")
        print("5. Run trading simulation")
        print("6. Exit")
        choice = input("Enter your choice: ")
        
        if choice == '1':
            portfolioName = input("Enter portfolio name: ")
            stockSymbol = input("Enter stock symbol: ").upper()
            if not validation(stockSymbol):
                print(f"Invalid stock symbol: {stockSymbol}")
                continue
            if portfolioName not in portfolios:
                portfolios[portfolioName] = []
            if stockSymbol not in portfolios[portfolioName]:
                portfolios[portfolioName].append(stockSymbol)
                print(f"Stock {stockSymbol} added to portfolio {portfolioName}")
            else:
                print(f"Stock {stockSymbol} already exists in portfolio {portfolioName}")
        
        elif choice == '2':
            portfolioName = input("Enter portfolio name: ")
            stockSymbol = input("Enter stock symbol: ").upper()
            if portfolioName in portfolios and stockSymbol in portfolios[portfolioName]:
                portfolios[portfolioName].remove(stockSymbol)
                print(f"Stock {stockSymbol} removed from portfolio {portfolioName}")
            else:
                print(f"Stock {stockSymbol} not found in portfolio {portfolioName}")
        
        elif choice == '3':
            if not portfolios:
                print("No portfolios found.")
            else:
                for portfolioName, stocks in portfolios.items():
                    print(f"Portfolio: {portfolioName}, Stocks: {', '.join(stocks)}")
        
        elif choice == '4':
            print("\nAvailable Trading Algorithms:")
            print("1. EMA (Exponential Moving Average)")
            print("2. LSTM (Long Short-Term Memory)")
            chooseAlgo = input("Enter the number corresponding to your choice: ")
            if chooseAlgo == '1':
                selectAlgo = "EMA"
            elif chooseAlgo == '2':
                selectAlgo = "LSTM"
            else:
                print("Invalid choice. Defaulting to EMA.")
        
        elif choice == '5':
            if not portfolios:
                print("No portfolios available. Please add a portfolio and stocks first.")
                continue
            
            print("\nAvailable Portfolios:")
            for idx, p in enumerate(portfolios, start=1):
                print(f"{idx}. {p}")
            
            choosePortfolio = int(input("Enter the portfolio number to run the simulation: "))
            if choosePortfolio < 1 or choosePortfolio > len(portfolios):
                print("Invalid portfolio choice.")
                continue
            
            selectedPortfolio = list(portfolios.keys())[choosePortfolio - 1]
            stocks = portfolios[selectedPortfolio]
            
            SD = input("Enter start date (YYYY-MM-DD): ")
            ED = input("Enter end date (YYYY-MM-DD): ")
            initial = float(input("Enter initial investment amount: "))
            
            allStockData = pd.DataFrame()
            for stock in stocks:
                print(f"\nFetching data for {stock}...")
                stockData = fetchPreprocess(stock, SD, ED)
                if stockData is not None:
                    allStockData = pd.concat([allStockData, stockData], ignore_index=True)
            
            if allStockData.empty:
                print("No data fetched for the selected portfolio and date range.")
                continue
            
            print("\nPreprocessed Stock Data:")
            print(allStockData.head())
            
            if selectAlgo == "EMA":
                EMAwindow = int(input("Enter EMA window size: "))
                initialShares = 0
                finalShares, portfolioValue = mockEMA(allStockData, initial, EMAwindow)
            elif selectAlgo == "LSTM":
                LSTMlookback = int(input("Enter LSTM lookback window size: "))
                predictDays = int(input("Enter number of future days to predict: "))

                scaler = MinMaxScaler(feature_range=(0, 1))
                scaledClose = scaler.fit_transform(allStockData['close'].values.reshape(-1, 1))
                X, y = prepdataLSTM(scaledClose, LSTMlookback)
                LSTMmodel = LSTMModel((X.shape[1], 1))
                LSTMmodel.fit(X, y, epochs=50, batch_size=32, verbose=0)

                futureData = predictPrices(allStockData, LSTMmodel, LSTMlookback, predictDays)
                print("\nPredicted Future Data:")
                print(futureData)

                # Append future_data to the existing all_stock_data for further processing (if needed)
                allStockData = pd.concat([allStockData, futureData.rename(columns={'predicted_close': 'close'})], ignore_index=True)

                initialShares = 0
                finalShares, portfolioValue = mockLSTM(allStockData, initial, LSTMmodel, LSTMlookback)

            print(f"\nInitial Shares: {initialShares}")
            print(f"Final Shares: {finalShares}")
            metrics(allStockData, initial, portfolioValue)

        elif choice == '6':
            break
        
        else:
            print("Invalid choice, please try again.")