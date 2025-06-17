function showPopup(content) {
    document.getElementById('popupContent').innerHTML = content;
    document.getElementById('popup').style.display = 'block';
}

function closePopup() {
    document.getElementById('popup').style.display = 'none';
}

function showAddStockPopup() {
    const content = `
        <h2>Add Stock to Portfolio</h2>
        <input type="text" id="stockName" placeholder="Stock Name">
        <input type="text" id="stockSymbol" placeholder="Stock Symbol">
        <button onclick="addStock()">Submit</button>
    `;
    showPopup(content);
}

async function addStock() {
    const stockName = document.getElementById('stockName').value;
    const stockSymbol = document.getElementById('stockSymbol').value;
    
    try {
        const response = await fetch('http://127.0.0.1:5000/add_stock', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ stockName, stockSymbol }),
        });
        const result = await response.json();
        if (result.success) {
            alert(result.message);
            closePopup();
        } else {
            alert(result.message);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while adding the stock.');
    }
}

function showRemoveStockPopup() {
    const content = `
        <h2>Remove Stock from Portfolio</h2>
        <input type="text" id="removeStockName" placeholder="Stock Name">
        <input type="text" id="removeStockSymbol" placeholder="Stock Symbol">
        <button onclick="removeStock()">Submit</button>
    `;
    showPopup(content);
}

async function removeStock() {
    const stockName = document.getElementById('removeStockName').value;
    const stockSymbol = document.getElementById('removeStockSymbol').value;
    
    try {
        const response = await fetch('http://127.0.0.1:5000/remove_stock', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ stockName, stockSymbol }),
        });
        const result = await response.json();
        if (result.success) {
            alert(result.message);
            closePopup();
        } else {
            alert(result.message);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while removing the stock.');
    }
}

async function showViewPortfoliosPopup() {
    try {
        const response = await fetch('http://127.0.0.1:5000/view_portfolios');
        const portfolios = await response.json();
        let content = '<h2>View Portfolios</h2>';
        if (Object.keys(portfolios).length === 0) {
            content += '<p>No portfolios found</p>';
        } else {
            for (const [stockName, stocks] of Object.entries(portfolios)) {
                if (stocks.length === 0) {
                    content += '<p>No stocks in this portfolio</p>';
                } else {
                    stocks.forEach(stock => {
                        content += `<li>${stockName} (${stock.symbol})</li>`;
                    });
                }
            }
        }
        showPopup(content);
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while fetching portfolios.');
    }
}

function showTradingSimulationPopup() {
    const content = `
        <h2>Run Trading Simulation</h2>
        <input type="text" id="simulationPortfolio" placeholder="Stock Name">
        <input type="date" id="startDate" placeholder="Start Date">
        <input type="date" id="endDate" placeholder="End Date">
        <input type="number" id="initialInvestment" placeholder="Initial Investment">
        <input type="number" id="windowSize" placeholder="Window Size">
        <input type="number" id="numPredictionDays" placeholder="Number of Days to Predict">
        <select id="algorithm">
            <option value="EMA">EMA</option>
            <option value="LSTM">LSTM</option>
        </select>
        <button onclick="runSimulation()">Submit</button>
    `;
    showPopup(content);
}

async function runSimulation() {
    const stockName = document.getElementById('simulationPortfolio').value;
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    const initialInvestment = document.getElementById('initialInvestment').value;
    const windowSize = document.getElementById('windowSize').value;
    const numPredictionDays = document.getElementById('numPredictionDays').value;
    const algorithm = document.getElementById('algorithm').value;

    try {
        const response = await fetch('http://127.0.0.1:5000/run_simulation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                stockName,
                startDate,
                endDate,
                initialInvestment,
                windowSize,
                numPredictionDays,
                algorithm
            }),
        });
        const result = await response.json();
        if (result.success) {
            let content = `
                <h2>Simulation Results</h2>
                <h3>Metrics:</h3>
                <pre>
Total Return: $${result.metrics.total_return ? result.metrics.total_return.toFixed(2) : 'N/A'}
Annual Return: ${result.metrics.annual_return ? (result.metrics.annual_return * 100).toFixed(2) : 'N/A'}%
Sharpe Ratio: ${result.metrics.sharpe_ratio ? result.metrics.sharpe_ratio.toFixed(2) : 'N/A'}
                </pre>
                <h3>Trading Chart:</h3>
                <div id="chartContainer" style="width:100%;height:500px;"></div>
                <h3>Simulation Output:</h3>
                <pre>${result.simulation_output}</pre>
            `;

            if (result.future_predictions) {
                content += `
                    <h3>Future Predictions:</h3>
                    <table>
                        <tr>
                            <th>Date</th>
                            <th>Predicted Close</th>
                        </tr>
                        ${result.future_predictions.map(pred => `
                            <tr>
                                <td>${pred.date}</td>
                                <td>${pred.predicted_close ? pred.predicted_close.toFixed(2) : 'N/A'}</td>
                            </tr>
                        `).join('')}
                    </table>
                `;
            }

            showPopup(content);

            // Render the Plotly chart
            const chartData = JSON.parse(result.chart);
            Plotly.newPlot('chartContainer', chartData.data, chartData.layout);
        } else {
            alert(result.message);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while running the simulation.');
    }
}