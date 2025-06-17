# scripts/fetch_data.py

import mysql.connector
import yfinance as yf
import pandas as pd
from datetime import datetime

# Connect to MySQL Database
def connect_db():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",  # Replace with your MySQL username
        password="B#@r@ti2324",  # Replace with your MySQL password
        database="stock_portfolio"
    )
    return conn

# Get list of portfolios
def get_portfolios():
    conn = connect_db()
    cursor = conn.cursor()

    try:
        cursor.execute('''
            SELECT DISTINCT portfolio_name FROM portfolios
        ''')
        portfolios = cursor.fetchall()
        return [p[0] for p in portfolios]
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return []
    finally:
        cursor.close()
        conn.close()

# Get stocks in a portfolio
def get_stocks_in_portfolio(portfolio_name):
    conn = connect_db()
    cursor = conn.cursor()

    try:
        cursor.execute('''
            SELECT stock_symbol FROM portfolios WHERE portfolio_name = %s
        ''', (portfolio_name,))
        stocks = cursor.fetchall()
        return [s[0] for s in stocks]
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return []
    finally:
        cursor.close()
        conn.close()

# Fetch and preprocess stock data
def fetch_and_preprocess(stock_symbol, start_date, end_date):
    try:
        # Fetch data from yFinance
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

        if stock_data.empty:
            print(f"No data found for {stock_symbol}")
            return None

        # Handle missing data by forward filling
        stock_data.fillna(method='ffill', inplace=True)

        # Calculate daily returns
        stock_data['Daily Return'] = stock_data['Adj Close'].pct_change().fillna(0)

        # Reset index to have 'Date' as a column
        stock_data.reset_index(inplace=True)

        # Rename columns to match database
        stock_data.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }, inplace=True)

        # Add stock symbol
        stock_data['stock_symbol'] = stock_symbol

        # Reorder columns
        stock_data = stock_data[['stock_symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'Daily Return']]

        return stock_data
    except Exception as e:
        print(f"Error fetching data for {stock_symbol}: {e}")
        return None

# Store preprocessed data into the database (optional)
def store_processed_data(stock_data):
    conn = connect_db()
    cursor = conn.cursor()

    try:
        for _, row in stock_data.iterrows():
            cursor.execute('''
                INSERT INTO stock_data (stock_symbol, date, open, high, low, close, adj_close, volume, daily_return)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (
                row['stock_symbol'],
                row['date'].strftime('%Y-%m-%d'),
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['adj_close'],
                row['volume'],
                row['Daily Return']
            ))
        conn.commit()
        print("Processed data stored successfully.")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        cursor.close()
        conn.close()

# Display processed data (optional)
def display_processed_data(stock_data):
    print("\n--- Processed Stock Data ---")
    print(stock_data.head())

# Main function
if __name__ == "__main__":
    portfolios = get_portfolios()

    if not portfolios:
        print("No portfolios available. Please add a portfolio first using manage_portfolio.py.")
        exit()

    print("\nAvailable Portfolios:")
    for idx, p in enumerate(portfolios, start=1):
        print(f"{idx}. {p}")
    
    print("\nOptions:")
    print("1. Select portfolios")
    print("2. Exclude specific portfolios")
    print("3. Select all portfolios")
    
    while True:
        try:
            option = int(input("\nEnter the number corresponding to your choice: "))
            if option in [1, 2, 3]:
                break
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    selected_portfolios = []
    
    if option == 1:
        # Select combinations of portfolios
        print("\nEnter the numbers of the portfolios you want to select for the combination (comma separated):")
        selected_indices = input().split(',')
        selected_portfolios = [portfolios[int(idx.strip()) - 1] for idx in selected_indices if 1 <= int(idx.strip()) <= len(portfolios)]
    
    elif option == 2:
        # Exclude specific portfolios
        print("\nEnter the numbers of the portfolios you want to exclude (comma separated):")
        excluded_indices = input().split(',')
        excluded_portfolios = [portfolios[int(idx.strip()) - 1] for idx in excluded_indices if 1 <= int(idx.strip()) <= len(portfolios)]
        selected_portfolios = [p for p in portfolios if p not in excluded_portfolios]
    
    elif option == 3:
        # Select all portfolios
        selected_portfolios = portfolios

    if not selected_portfolios:
        print("No portfolios selected.")
        exit()

    print(f"\nSelected Portfolios: {', '.join(selected_portfolios)}")

    # Get stocks in the selected portfolios
    all_stocks = set()
    for portfolio in selected_portfolios:
        stocks = get_stocks_in_portfolio(portfolio)
        all_stocks.update(stocks)

    if not all_stocks:
        print("No stocks found in the selected portfolios.")
        exit()

    print(f"Stocks in selected portfolios: {', '.join(all_stocks)}")

    # Input date range
    while True:
        start_date = input("Enter start date (YYYY-MM-DD): ")
        end_date = input("Enter end date (YYYY-MM-DD): ")

        # Validate date format
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
            if start_date > end_date:
                print("Start date must be earlier than end date.")
                continue
            break
        except ValueError:
            print("Invalid date format. Please enter dates in YYYY-MM-DD format.")

    # Fetch and preprocess data for each stock
    all_stock_data = pd.DataFrame()

    for stock in all_stocks:
        print(f"\nFetching data for {stock}...")
        stock_data = fetch_and_preprocess(stock, start_date, end_date)
        if stock_data is not None:
            all_stock_data = pd.concat([all_stock_data, stock_data], ignore_index=True)

    if all_stock_data.empty:
        print("No data fetched for the selected portfolios and date range.")
        exit()

    # Display the first few rows of the processed data
    display_processed_data(all_stock_data)

    # Ask user if they want to store the processed data
    while True:
        store_choice = input("\nDo you want to store the processed data in the database? (y/n): ").lower()
        if store_choice == 'y':
            # Ensure the processed_data table exists
            conn = connect_db()
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS stock_data (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        stock_symbol VARCHAR(10),
                        date DATE,
                        open FLOAT,
                        high FLOAT,
                        low FLOAT,
                        close FLOAT,
                        adj_close FLOAT,
                        volume BIGINT,
                        daily_return FLOAT
                    )
                ''')
                conn.commit()
            except mysql.connector.Error as err:
                print(f"Error creating stock_data table: {err}")
            finally:
                cursor.close()
                conn.close()

            store_processed_data(all_stock_data)
            break
        elif store_choice == 'n':
            print("Processed data not stored in the database.")
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    print("\nData fetching and processing completed.")

