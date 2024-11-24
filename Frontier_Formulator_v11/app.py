from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # Use a backend suitable for a headless server
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import io
import base64
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

app = Flask(__name__)

# Configure logging for better error visibility
logging.basicConfig(level=logging.DEBUG)

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"An error occurred: {str(e)}")
    return jsonify({"error": str(e)}), 400

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Helper function to get stock data
def get_stock_data(tickers, start_date, end_date):
    """
    Fetches adjusted closing prices of the given stock tickers within the specified date range.
    
    Parameters:
        tickers (list): List of stock tickers to fetch data for.
        start_date (str): Start date for fetching data in 'YYYY-MM-DD' format.
        end_date (str): End date for fetching data in 'YYYY-MM-DD' format.

    Returns:
        DataFrame: Daily returns of the given stocks.
    """
    try:
        tickers = [ticker.strip().upper() for ticker in tickers]
        stock_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        if stock_data.empty:
            raise ValueError("No data found for the provided tickers and date range.")
        returns_df = stock_data.pct_change().dropna()
        logging.debug(f"Stock data loaded successfully for tickers: {tickers}")
        return returns_df
    except Exception as e:
        logging.error(f"Error in get_stock_data: {e}")
        raise

# Helper function to regularize covariance matrix
def regularize_covariance_matrix(returns_df, lambda_reg=1e-4):
    """
    Regularizes the covariance matrix to avoid instability issues.
    
    Parameters:
        returns_df (DataFrame): Dataframe containing daily returns of assets.
        lambda_reg (float): Regularization strength.

    Returns:
        DataFrame: Regularized covariance matrix.
    """
    cov_matrix = returns_df.cov() * 252
    identity_matrix = np.identity(len(cov_matrix))
    return cov_matrix + lambda_reg * identity_matrix  # Regularize the covariance matrix slightly

# Helper function to calculate portfolio performance
def portfolio_performance(weights, returns_df, risk_free_rate=0.01):
    """
    Calculates the annualized return and standard deviation of the portfolio.
    
    Parameters:
        weights (array): Weights of each asset in the portfolio.
        returns_df (DataFrame): Dataframe containing daily returns of assets.
        risk_free_rate (float): The risk-free rate used in the Sharpe ratio calculation.

    Returns:
        tuple: Portfolio return and portfolio standard deviation.
    """
    try:
        cov_matrix = regularize_covariance_matrix(returns_df)
        portfolio_return = np.sum(weights * returns_df.mean()) * 252
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        logging.debug(f"Portfolio performance calculated: return={portfolio_return}, std_dev={portfolio_std_dev}")
        return portfolio_return, portfolio_std_dev
    except Exception as e:
        logging.error(f"Error in portfolio_performance: {e}")
        raise

# Helper function to calculate the negative Sharpe ratio (used for optimization)
def negative_sharpe_ratio(weights, returns_df, risk_free_rate=0.01, regularization_strength=0.01):
    """
    Calculates the negative Sharpe ratio, adding regularization to encourage diversification.
    
    Parameters:
        weights (array): Weights of each asset in the portfolio.
        returns_df (DataFrame): Dataframe containing daily returns of assets.
        risk_free_rate (float): The risk-free rate used in the Sharpe ratio calculation.
        regularization_strength (float): L2 regularization strength.

    Returns:
        float: Negative Sharpe ratio with regularization.
    """
    try:
        portfolio_return, portfolio_std_dev = portfolio_performance(weights, returns_df, risk_free_rate)
        sharpe_ratio = -(portfolio_return - risk_free_rate) / portfolio_std_dev
        regularization = regularization_strength * np.sum(weights**2)  # Add L2 regularization to encourage diversification
        total_cost = sharpe_ratio + regularization
        logging.debug(f"Negative Sharpe ratio with regularization calculated: {total_cost}")
        return total_cost
    except Exception as e:
        logging.error(f"Error in negative_sharpe_ratio: {e}")
        raise

# Function to calculate the portfolio with the maximum Sharpe ratio
def calculate_sharpe_optimal_portfolio(returns_df, risk_free_rate=0.01):
    """
    Optimizes the portfolio to maximize the Sharpe ratio.
    
    Parameters:
        returns_df (DataFrame): Dataframe containing daily returns of assets.
        risk_free_rate (float): The risk-free rate used in the Sharpe ratio calculation.

    Returns:
        tuple: Portfolio return, portfolio volatility, Sharpe ratio, and weights.
    """
    try:
        num_assets = len(returns_df.columns)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights must sum to 1
        bounds = tuple((0, 1) for _ in range(num_assets))  # Weights must be between 0 and 1
        init_guess = num_assets * [1. / num_assets,]  # Initial guess of equal weights
        result = minimize(negative_sharpe_ratio, init_guess, args=(returns_df, risk_free_rate, 0.01), method='SLSQP', bounds=bounds, constraints=constraints)
        if not result.success:
            raise ValueError("Optimization did not converge for Sharpe optimal portfolio.")
        portfolio_return, portfolio_std_dev = portfolio_performance(result.x, returns_df, risk_free_rate)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
        logging.debug(f"Sharpe optimal portfolio calculated: return={portfolio_return}, volatility={portfolio_std_dev}, Sharpe ratio={sharpe_ratio}")
        return portfolio_return, portfolio_std_dev, sharpe_ratio, result.x
    except Exception as e:
        logging.error(f"Error in calculate_sharpe_optimal_portfolio: {e}")
        raise

# Function to calculate the minimum variance portfolio
def calculate_min_variance_portfolio(returns_df):
    """
    Optimizes the portfolio to minimize variance (risk).
    
    Parameters:
        returns_df (DataFrame): Dataframe containing daily returns of assets.

    Returns:
        tuple: Portfolio return, portfolio volatility, and weights.
    """
    try:
        num_assets = len(returns_df.columns)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights must sum to 1
        bounds = tuple((0, 1) for _ in range(num_assets))  # Weights must be between 0 and 1
        init_guess = num_assets * [1. / num_assets,]  # Initial guess of equal weights
        
        def portfolio_volatility(weights, returns_df):
            cov_matrix = regularize_covariance_matrix(returns_df)
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        result = minimize(portfolio_volatility, init_guess, args=(returns_df,), method='SLSQP', bounds=bounds, constraints=constraints)
        if not result.success:
            raise ValueError("Optimization did not converge for minimum variance portfolio.")
        portfolio_return, portfolio_std_dev = portfolio_performance(result.x, returns_df)
        logging.debug(f"Minimum variance portfolio calculated: return={portfolio_return}, volatility={portfolio_std_dev}")
        return portfolio_return, portfolio_std_dev, result.x
    except Exception as e:
        logging.error(f"Error in calculate_min_variance_portfolio: {e}")
        raise

# Function to calculate minimum variance portfolio using predicted returns
def calculate_min_variance_portfolio_with_predictions(returns_df, avg_predicted_returns):
    """
    Optimizes the portfolio to minimize variance using predicted returns.
    
    Parameters:
        returns_df (DataFrame): Dataframe containing daily returns of assets.
        avg_predicted_returns (array): Average predicted returns for each asset.

    Returns:
        tuple: Portfolio return, portfolio volatility, and weights.
    """
    try:
        num_assets = len(returns_df.columns)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights must sum to 1
        bounds = tuple((0, 1) for _ in range(num_assets))  # Weights must be between 0 and 1
        init_guess = num_assets * [1. / num_assets,]  # Initial guess of equal weights
        
        def portfolio_volatility(weights, cov_matrix):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        cov_matrix = regularize_covariance_matrix(returns_df)
        result = minimize(portfolio_volatility, init_guess, args=(cov_matrix,), method='SLSQP', bounds=bounds, constraints=constraints)
        if not result.success:
            raise ValueError("Optimization did not converge for minimum variance portfolio with predictions.")
        portfolio_return = np.sum(result.x * avg_predicted_returns) * 252
        portfolio_std_dev = portfolio_volatility(result.x, cov_matrix)
        logging.debug(f"Minimum variance portfolio with predictions calculated: return={portfolio_return}, volatility={portfolio_std_dev}")
        return portfolio_return, portfolio_std_dev, result.x
    except Exception as e:
        logging.error(f"Error in calculate_min_variance_portfolio_with_predictions: {e}")
        raise

# Function to calculate the portfolio with the maximum Sharpe ratio using Random Forest predictions
def calculate_random_forest_max_sharpe_portfolio(returns_df):
    """
    Uses Random Forest to predict future returns and optimizes the portfolio to maximize the Sharpe ratio.
    
    Parameters:
        returns_df (DataFrame): Dataframe containing daily returns of assets.

    Returns:
        tuple: Portfolio return, portfolio volatility, Sharpe ratio, weights, and predicted returns.
    """
    try:
        # Prepare features for prediction
        features = pd.DataFrame(index=returns_df.index)
        for ticker in returns_df.columns:
            features[f'{ticker}_lagged_1'] = returns_df[ticker].shift(1)  # 1-day lagged returns
            features[f'{ticker}_lagged_5'] = returns_df[ticker].shift(5)  # 5-day lagged returns
            features[f'{ticker}_lagged_10'] = returns_df[ticker].shift(10)  # 10-day lagged returns
            features[f'{ticker}_ema'] = returns_df[ticker].ewm(span=10, adjust=False).mean()  # 10-day EMA
            features[f'{ticker}_bollinger_high'] = returns_df[ticker].rolling(window=20).mean() + (returns_df[ticker].rolling(window=20).std() * 2)  # Upper Bollinger Band
            features[f'{ticker}_bollinger_low'] = returns_df[ticker].rolling(window=20).mean() - (returns_df[ticker].rolling(window=20).std() * 2)  # Lower Bollinger Band

        # Adding general features
        features['mean_return'] = returns_df.mean(axis=1)
        features['volatility'] = returns_df.std(axis=1)
        features['rolling_avg'] = returns_df.rolling(window=5).mean().mean(axis=1)
        features['momentum'] = returns_df.pct_change().rolling(window=5).sum().mean(axis=1)

        # Add Relative Strength Index (RSI) for each stock
        delta = returns_df.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs)).mean(axis=1)  # Average RSI across all assets

        features = features.dropna()

        y = returns_df.shift(-1).reindex(features.index).dropna()  # Predict next day's returns for each asset
        features = features.loc[y.index]  # Align X with y to ensure consistent indexing

        # Train-test split to evaluate the model
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

        # Using RandomForestRegressor for predicting returns for each asset
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)  # Use all available cores
        model.fit(X_train, y_train)

        # Cross-validation to ensure the model's predictive power
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)  # Parallelize cross-validation
        logging.debug(f"Cross-validation R^2 scores: {scores}")

        # Adjusting the threshold for the predictive power warning
        if scores.mean() < -0.5:  # Loosen the threshold to reduce false positives
            logging.warning("The predictive power of the model seems low. Consider revising features or model choice.")
            raise ValueError("The predictive power of the model is low. The selected date range may be too short, leading to insufficient data for accurate predictions.")

        # Predicting the returns for each stock (per asset)
        predicted_returns = model.predict(features)
        predicted_returns_df = pd.DataFrame(predicted_returns, index=features.index, columns=returns_df.columns)
        logging.debug(f"Predicted returns from model: {predicted_returns_df}")

        # Use the predicted returns for each asset independently to calculate each portfolio's performance
        num_assets = len(returns_df.columns)
        num_portfolios = 10000
        results = []
        cov_matrix = regularize_covariance_matrix(returns_df)

        def evaluate_portfolio(_):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            portfolio_return = np.sum(weights * predicted_returns_df.mean()) * 252  # Using mean predicted return per asset
            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - 0.01) / portfolio_std_dev
            return (sharpe_ratio, portfolio_return, portfolio_std_dev, weights)

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            results = list(executor.map(evaluate_portfolio, range(num_portfolios)))

        # Find the portfolio with the highest Sharpe ratio
        max_sharpe_portfolio = max(results, key=lambda x: x[0])
        max_sharpe_ratio, portfolio_return, portfolio_std_dev, weights = max_sharpe_portfolio

        logging.debug(f"Approximate Sharpe optimal portfolio calculated using predicted returns: return={portfolio_return}, volatility={portfolio_std_dev}, Sharpe ratio={max_sharpe_ratio}")
        return portfolio_return, portfolio_std_dev, max_sharpe_ratio, weights, predicted_returns_df.mean()
    except Exception as e:
        logging.error(f"Error in calculate_random_forest_max_sharpe_portfolio: {e}")
        raise

# Function to plot the efficient frontier
def plot_efficient_frontier(returns_df, max_sharpe, min_variance, method_name):
    """
    Generates and saves a plot of the efficient frontier for the given portfolio.
    
    Parameters:
        returns_df (DataFrame): Dataframe containing daily returns of assets.
        max_sharpe (tuple): Information of the maximum Sharpe ratio portfolio (return, volatility).
        min_variance (tuple): Information of the minimum variance portfolio (return, volatility).
        method_name (str): Name of the method used (e.g., "Classical").

    Returns:
        BytesIO: Plot saved in memory.
    """
    try:
        num_assets = len(returns_df.columns)
        num_portfolios = 10000
        results = np.zeros((3, num_portfolios))
        cov_matrix = regularize_covariance_matrix(returns_df)

        def generate_portfolio(i):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            portfolio_return, portfolio_std_dev = portfolio_performance(weights, returns_df)
            return portfolio_std_dev, portfolio_return, (portfolio_return - 0.01) / portfolio_std_dev

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            portfolios = list(executor.map(generate_portfolio, range(num_portfolios)))

        for i, (portfolio_std_dev, portfolio_return, sharpe_ratio) in enumerate(portfolios):
            results[0, i] = portfolio_std_dev
            results[1, i] = portfolio_return
            results[2, i] = sharpe_ratio

        plt.figure(figsize=(10, 7))
        plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='viridis', marker='o', alpha=0.5)
        plt.colorbar(label='Sharpe Ratio')

        plt.scatter(max_sharpe[1], max_sharpe[0], color='r', marker='*', s=200, label=f'{method_name} Max Sharpe Ratio')
        plt.scatter(min_variance[1], min_variance[0], color='b', marker='*', s=200, label=f'{method_name} Min Variance Portfolio')

        plt.title(f'{method_name} Efficient Frontier')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.legend()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        logging.debug(f"Efficient frontier plot generated for {method_name}")
        return img
    except Exception as e:
        logging.error(f"Error in plot_efficient_frontier: {e}")
        raise

# Function to plot the efficient frontier using predicted returns
def plot_efficient_frontier_with_predictions(returns_df, avg_predicted_returns, max_sharpe, min_variance, method_name):
    """
    Generates and saves a plot of the efficient frontier using predicted returns for the given portfolio.
    
    Parameters:
        returns_df (DataFrame): Dataframe containing daily returns of assets.
        avg_predicted_returns (array): Average predicted returns for each asset.
        max_sharpe (tuple): Information of the maximum Sharpe ratio portfolio (return, volatility).
        min_variance (tuple): Information of the minimum variance portfolio (return, volatility).
        method_name (str): Name of the method used (e.g., "Random Forest").

    Returns:
        BytesIO: Plot saved in memory.
    """
    try:
        num_assets = len(returns_df.columns)
        num_portfolios = 10000
        results = np.zeros((3, num_portfolios))
        cov_matrix = regularize_covariance_matrix(returns_df)

        def generate_portfolio(i):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            portfolio_return = np.sum(weights * avg_predicted_returns) * 252
            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return portfolio_std_dev, portfolio_return, (portfolio_return - 0.01) / portfolio_std_dev

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            portfolios = list(executor.map(generate_portfolio, range(num_portfolios)))

        for i, (portfolio_std_dev, portfolio_return, sharpe_ratio) in enumerate(portfolios):
            results[0, i] = portfolio_std_dev
            results[1, i] = portfolio_return
            results[2, i] = sharpe_ratio

        plt.figure(figsize=(10, 7))
        plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='viridis', marker='o', alpha=0.5)
        plt.colorbar(label='Sharpe Ratio')

        plt.scatter(max_sharpe[1], max_sharpe[0], color='r', marker='*', s=200, label=f'{method_name} Max Sharpe Ratio')
        plt.scatter(min_variance[1], min_variance[0], color='b', marker='*', s=200, label=f'{method_name} Min Variance Portfolio')

        plt.title(f'{method_name} Efficient Frontier')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.legend()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        logging.debug(f"Efficient frontier plot generated for {method_name}")
        return img
    except Exception as e:
        logging.error(f"Error in plot_efficient_frontier_with_predictions: {e}")
        raise

@app.route("/generate_combined", methods=["POST"])
def generate_combined():
    """
    Endpoint to generate the combined efficient frontier for Classical and Random Forest methods.
    
    Returns:
        JSON response containing the base64 encoded plots and portfolio data.
    """
    tickers = request.form.get("tickers").split(',')
    start_date = request.form.get("start_date")
    end_date = request.form.get("end_date")

    try:
        logging.debug(f"Generating combined frontier for tickers: {tickers}, start_date: {start_date}, end_date: {end_date}")
        returns_df = get_stock_data(tickers, start_date, end_date)

        classical_return, classical_vol, classical_sharpe, classical_weights = calculate_sharpe_optimal_portfolio(returns_df)
        classical_min_return, classical_min_vol, classical_min_weights = calculate_min_variance_portfolio(returns_df)

        rf_return, rf_vol, rf_sharpe, rf_weights, avg_predicted_returns = calculate_random_forest_max_sharpe_portfolio(returns_df)
        rf_min_return, rf_min_vol, rf_min_weights = calculate_min_variance_portfolio_with_predictions(returns_df, avg_predicted_returns)

        classical_img = plot_efficient_frontier(returns_df, (classical_return, classical_vol), (classical_min_return, classical_min_vol), "Classical")
        rf_img = plot_efficient_frontier_with_predictions(returns_df, avg_predicted_returns, (rf_return, rf_vol), (rf_min_return, rf_min_vol), "Random Forest")

        classical_img_base64 = base64.b64encode(classical_img.getvalue()).decode('utf-8')
        rf_img_base64 = base64.b64encode(rf_img.getvalue()).decode('utf-8')

        portfolio_data = {
            "classical_max_sharpe": {
                "weights": classical_weights.tolist(),
                "return": classical_return,
                "volatility": classical_vol,
                "sharpe_ratio": classical_sharpe
            },
            "classical_min_variance": {
                "weights": classical_min_weights.tolist(),
                "return": classical_min_return,
                "volatility": classical_min_vol
            },
            "rf_max_sharpe": {
                "weights": rf_weights.tolist(),
                "return": rf_return,
                "volatility": rf_vol,
                "sharpe_ratio": rf_sharpe
            },
            "rf_min_variance": {
                "weights": rf_min_weights.tolist(),
                "return": rf_min_return,
                "volatility": rf_min_vol
            }
        }

        logging.debug("Combined frontier and portfolio data generated successfully")
        return jsonify({
            "status": "success",
            "classical_plot": "data:image/png;base64," + classical_img_base64,
            "rf_plot": "data:image/png;base64," + rf_img_base64,
            "portfolio_data": portfolio_data
        })

    except ValueError as e:
        app.logger.error(f"ValueError occurred: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 400

    except Exception as e:
        app.logger.error(f"An unexpected error occurred: {str(e)}")
        return jsonify({"status": "error", "error": "An unexpected error occurred. Please check server logs for details."}), 500

if __name__ == "__main__":
    app.run(debug=True)
    