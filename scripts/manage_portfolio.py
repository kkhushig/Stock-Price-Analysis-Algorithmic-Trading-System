import mysql.connector
import yfinance as yf

# Connect to MySQL Database
def connect_db():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",  # Change to your MySQL username if needed
        password="B#@r@ti2324",  # Change to your MySQL password if needed
        database="stock_portfolio"  # Ensure this matches your database name
    )
    return conn

# Check if stock is valid
def is_stock_valid(stock_symbol):
    try:
        stock = yf.Ticker(stock_symbol)
        stock_info = stock.info
        
        # Validate based on stock's name presence
        if 'shortName' in stock_info or 'longName' in stock_info:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error occurred: {e}")
        return False

# Add stock to portfolio
def add_stock_to_portfolio(portfolio_name, stock_symbol):
    if not is_stock_valid(stock_symbol):
        print(f"Invalid stock symbol: {stock_symbol}")
        return

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO portfolios (portfolio_name, stock_symbol)
        VALUES (%s, %s)
    ''', (portfolio_name, stock_symbol))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Stock {stock_symbol} added to portfolio {portfolio_name}")

# Remove stock from portfolio
def remove_stock_from_portfolio(portfolio_name, stock_symbol):
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute('''
        DELETE FROM portfolios WHERE portfolio_name = %s AND stock_symbol = %s
    ''', (portfolio_name, stock_symbol))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Stock {stock_symbol} removed from portfolio {portfolio_name}")

# View all portfolios
def view_portfolios():
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT portfolio_name, GROUP_CONCAT(stock_symbol), creation_date
        FROM portfolios
        GROUP BY portfolio_name, creation_date
    ''')

    portfolios = cursor.fetchall()
    conn.close()

    for portfolio in portfolios:
        print(f"Portfolio: {portfolio[0]}, Stocks: {portfolio[1]}, Created on: {portfolio[2]}")

# Main menu to manage portfolios
if __name__ == "__main__":
    while True:
        print("\nPortfolio Management Options:")
        print("1. Add stock to portfolio")
        print("2. Remove stock from portfolio")
        print("3. View portfolios")
        print("4. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            portfolio_name = input("Enter portfolio name: ")
            stock_symbol = input("Enter stock symbol: ").upper()
            add_stock_to_portfolio(portfolio_name, stock_symbol)
        elif choice == '2':
            portfolio_name = input("Enter portfolio name: ")
            stock_symbol = input("Enter stock symbol: ").upper()
            remove_stock_from_portfolio(portfolio_name, stock_symbol)
        elif choice == '3':
            view_portfolios()
        elif choice == '4':
            break
        else:
            print("Invalid choice, please try again.")

