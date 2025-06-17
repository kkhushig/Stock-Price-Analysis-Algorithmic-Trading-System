# 📈 Stock Price Analysis & Algorithmic Trading System

## 🧠 Overview
This project presents a full pipeline for **real-time stock price analysis and algorithmic trading**, developed in two integrated phases. It starts with robust data collection, preprocessing, and storage using a MySQL-backed system. It then progresses to modeling stock trends using classical (EMA) and advanced (LSTM) algorithms to simulate trading strategies in a realistic, mock trading environment.

The goal is to automate stock trading decision-making with minimal latency and high insight, using both statistical and machine learning techniques.

---

## 📚 Motivation
Manual stock analysis is error-prone and inefficient. With the growth of data availability and algorithmic trading, we sought to build an end-to-end tool that can:
- Track real-time stock trends
- Evaluate multiple strategies
- Simulate trades
- Offer visual insights via an interactive interface

---

## ⚙️ Technologies Used

### Backend
- **Python**
- **MySQL / phpMyAdmin** – for structured data storage
- **yfinance** – live stock data retrieval
- **pandas / numpy / scikit-learn** – preprocessing and metrics
- **TensorFlow / Keras** – LSTM modeling

### Frontend
- **HTML / CSS / JavaScript**
- **Flask (API layer)**
- **Plotly.js** – for interactive charts

---

## 🧱 System Architecture

### 🔹 Phase 1: Data Collection & Storage
- Set up a **MySQL database** to store portfolios and stock data.
- Fetch stock data using `yfinance`.
- Validate stock symbols before adding.
- Compute daily returns and clean missing values.
- Store cleaned data in tables (`portfolios`, `stock_data`).

### 🔹 Phase 2: Algorithmic Trading & Simulation

#### 📊 EMA Strategy
- Uses Exponential Moving Average crossover logic.
- Buy/sell signals triggered by crossing price lines.

#### 🤖 LSTM Forecasting
- Train LSTM on historical data.
- Predict future prices.
- Buy/sell based on future price vs. current price.

#### 💰 Mock Trading Environment
- Simulates trades with virtual balance.
- Logs positions and portfolio value dynamically.

#### 📈 Metrics Computed
- **Total Return**
- **Annualized Return**
- **Sharpe Ratio**

---

## 💻 GUI Features

- Add / remove stocks from portfolios
- View existing portfolios
- Select and run trading simulations (EMA or LSTM)
- Visualizations for:
  - Price vs. Signals
  - Portfolio growth
- Responsive frontend built using HTML + JS
- Charts powered by Plotly

---

## 📂 Project Structure

```
Stock-Trading-System/
├── scripts/
│   ├── fetch_data.py              
│   ├── manage_portfolio.py        
│   ├── trading_simulation.py      
│   └── model_utils.py             
├── frontend/
│   ├── index.html                 
│   ├── style.css                  
│   └── main.js                    
├── helper/
    ├── code_explanation.txt   
│   └── requirements.txt               
└── READ.md              
```

---

## 🛠️ How to Run

### 1. 📦 Setup MySQL & phpMyAdmin (Linux)
```bash
sudo apt install apache2 mysql-server php phpmyadmin
sudo ufw allow 'Apache Full'
sudo service apache2 restart
```
Then visit: `http://localhost/phpmyadmin` to access the GUI.

### 2. 🐍 Setup Python Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 🗄️ Initialize Database
Run the `schema.sql` file inside MySQL to create required tables.

### 4. 📉 Collect Stock Data
```bash
python scripts/fetch_data.py
```

### 5. 💼 Manage Portfolios
```bash
python scripts/manage_portfolio.py
```

### 6. 🚀 Run Simulations
```bash
python scripts/trading_simulation.py
```

---

## 🚀 Future Enhancements
- Add reinforcement learning-based trading agent
- Enable live trading via broker API
- Add email alerts for trade signals
- Integrate risk management strategies

---
