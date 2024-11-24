# Portfolio Optimization Program Documentation

## Overview

This documentation describes a web application designed to compare the performance of two portfolio optimization methods: Classical Mean-Variance Optimization (MVO) and Random Forest-Enhanced Portfolio Optimization. The application fetches financial data, calculates the optimal portfolios, and visualizes the efficient frontiers for both methods.

The application allows users to:
- Enter stock tickers and a date range.
- Calculate and visualize the optimal portfolios based on historical and predicted returns.

The application is implemented using Flask, and uses various Python packages such as `pandas`, `yfinance`, `matplotlib`, `scipy.optimize`, and `sklearn`.

## Structure

### Flask Web Server
The Flask web server is the main entry point of the program. It provides two primary endpoints:
1. `"/"`: The main page that renders the HTML form for user inputs.
2. `"/generate_combined"`: A POST endpoint that processes user input to generate the Classical and Random Forest efficient frontiers.

The `index.html` is used for the user interface, and it contains form fields for inputting stock tickers and date ranges.

### Modules and Functions
Below is a description of each module and function in the program.

1. **Helper Functions**
   - **`get_stock_data(tickers, start_date, end_date)`**
     - **Purpose**: Fetches historical adjusted closing prices for given stock tickers between the specified dates.
     - **Return**: Returns daily returns as a Pandas DataFrame.
     - **Dependencies**: Uses the Yahoo Finance API (`yfinance`).

   - **`regularize_covariance_matrix(returns_df, lambda_reg)`**
     - **Purpose**: Regularizes the covariance matrix to avoid numerical instability issues during optimization.
     - **Return**: Regularized covariance matrix.

   - **`portfolio_performance(weights, returns_df, risk_free_rate)`**
     - **Purpose**: Calculates the annualized return and standard deviation of a portfolio.
     - **Return**: Tuple containing portfolio return and standard deviation.

   - **`negative_sharpe_ratio(weights, returns_df, risk_free_rate, regularization_strength)`**
     - **Purpose**: Calculates the negative Sharpe ratio, which is used for optimization to maximize the portfolio's Sharpe ratio.
     - **Return**: Negative Sharpe ratio with regularization to encourage diversification.

2. **Portfolio Optimization Functions**
   - **`calculate_sharpe_optimal_portfolio(returns_df, risk_free_rate)`**
     - **Purpose**: Finds the portfolio with the maximum Sharpe ratio.
     - **Return**: Portfolio return, volatility, Sharpe ratio, and asset weights.

   - **`calculate_min_variance_portfolio(returns_df)`**
     - **Purpose**: Finds the portfolio with the minimum variance.
     - **Return**: Portfolio return, volatility, and asset weights.

   - **`calculate_random_forest_max_sharpe_portfolio(returns_df)`**
     - **Purpose**: Uses a Random Forest Regressor to predict future returns and then finds the portfolio with the maximum Sharpe ratio.
     - **Return**: Portfolio return, volatility, Sharpe ratio, weights, and predicted returns.

   - **`calculate_min_variance_portfolio_with_predictions(returns_df, avg_predicted_returns)`**
     - **Purpose**: Uses predicted returns to calculate the portfolio with the minimum variance.
     - **Return**: Portfolio return, volatility, and asset weights.

3. **Efficient Frontier Plotting Functions**
   - **`plot_efficient_frontier(returns_df, max_sharpe, min_variance, method_name)`**
     - **Purpose**: Plots the efficient frontier using historical data.
     - **Return**: Image in memory to be displayed in the web page.

   - **`plot_efficient_frontier_with_predictions(returns_df, avg_predicted_returns, max_sharpe, min_variance, method_name)`**
     - **Purpose**: Plots the efficient frontier using predicted returns.
     - **Return**: Image in memory to be displayed in the web page.

## User Workflow
1. **Input Stock Tickers and Date Range**: The user inputs the stock tickers (e.g., `AAPL, MSFT, GOOGL`) and selects a start and end date for the data.
2. **Generate Combined Frontier**: On submitting the form, the backend fetches the stock data and runs the Classical and Random Forest portfolio optimizations.
3. **Display Results**: The optimized portfolios are presented in a graphical representation (efficient frontier) along with key metrics like portfolio weights, expected return, and risk.

## Error Handling
- **Error Popups**: If the user provides incorrect input (e.g., unavailable ticker, insufficient date range), an error is displayed.
- **Warnings for Predictive Power**: If the Random Forest model's predictive power is low, a warning is provided.
- **Logs**: Logging statements are added throughout the code to debug errors and warnings, such as failure in optimization.

## Enhancements
- **Parallel Processing**: To speed up calculations, multi-threading and the use of `ThreadPoolExecutor` allow for faster evaluation of portfolios during optimization.
- **Dynamic Error Messages**: The error message seen in the web application corresponds to the issue that occurred (e.g., incorrect ticker symbol).

## Dependencies
- **Python Libraries**:
  - `Flask`: To create the web interface.
  - `pandas`, `numpy`: To handle data operations.
  - `yfinance`: To fetch stock data.
  - `matplotlib`: To plot the efficient frontier.
  - `scipy.optimize`: To optimize portfolio allocations.
  - `sklearn`: To implement machine learning models and predict returns.
  - `ThreadPoolExecutor` (from `concurrent.futures`) and `multiprocessing`: For parallel processing to accelerate computations.

## Getting Started
1. **Install Requirements**: To install dependencies, use `pip install -r requirements.txt`.
2. **Run the Flask Application**: Use the command `python app.py` to start the server. The server will run in `debug` mode, and you can access the UI via `http://127.0.0.1:5000`.
3. **Navigate the UI**: Enter stock tickers, date range, and click "Generate Combined Frontier" to see the results.

## Important Notes
- **Model Accuracy**: The Random Forest predictions depend on the availability and quality of past data. A small date range or insufficient data might yield inaccurate predictions.
- **Risk-Free Rate**: The program uses a default risk-free rate of `0.01` to calculate Sharpe ratios.
- **Assumptions**: The program assumes that all stocks in the input have continuous historical data for the selected date range, and market conditions are normal.

## Future Improvements
- **GPU Acceleration**: Use GPU processing for even faster computation, especially for larger datasets.
- **Validation on Frontend**: Validate user input directly on the front end to prevent incorrect entries from being processed.
- **Additional Models**: Add other machine learning models, such as XGBoost, to compare with Random Forest for improved accuracy in return predictions.

