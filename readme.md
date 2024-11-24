# Portfolio Optimization Web Application

## Overview

This is a Python-based web application that compares the performance of Classical Mean-Variance Optimization (MVO) and Random Forest-Enhanced Portfolio Optimization for a selected set of stocks. Users can enter stock tickers and select a time period to generate efficient frontiers and see how the optimized portfolios perform.

## Features
- **Efficient Frontier Plotting**: Generates and displays the efficient frontier for both the Classical approach and the Random Forest-enhanced method.
- **Portfolio Optimization**: Calculates and visualizes the maximum Sharpe ratio and minimum variance portfolios for each method.
- **User-Friendly Interface**: A simple web interface where users can easily input stock tickers, select date ranges, and view results.
- **Machine Learning Integration**: Uses Random Forest Regression to forecast returns and construct optimized portfolios.
- **Automatic Browser Launch**: The web application automatically opens in the default browser after starting the server.

## Prerequisites

To use this program, you need the following:

- **Python 3.7 or higher**.
- Required Python packages listed in `requirements.txt`. You can install them with:
  ```
  pip install -r requirements.txt
  ```
- A working internet connection to download stock data from Yahoo Finance.

## Installation

1. **Clone the Repository**: Clone this repository to your local machine.
   ```
   git clone <repository_url>
   ```

2. **Install Dependencies**: Install the required Python packages using the command:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

1. **Use the Start-Up Script**:
   - Run the provided `start.bat` file. This script will create a virtual environment, install dependencies, and launch the web server automatically.
   - The web page will automatically open in your default browser.

2. **Alternative: Start the Flask Server Manually**:
   ```
   python app.py
   ```
   By default, the server will run in debug mode, accessible at `http://127.0.0.1:5000`.

3. **Use the Interface**:
   - Enter stock tickers (e.g., AAPL, MSFT, GOOGL).
   - Select a start date and an end date for the historical data.
   - Click on "Generate Combined Frontier" to visualize the results.

## User Guide

- **Input Requirements**: Enter valid stock tickers separated by commas (e.g., `AAPL, MSFT, GOOGL`). Ensure the tickers are supported by Yahoo Finance.
- **Date Range**: Select a date range long enough to ensure that the Random Forest model has sufficient data to train on (ideally several years).
- **Interpreting the Charts**: The web interface displays the efficient frontier and key portfolio metrics, such as weights, expected return, volatility, and Sharpe ratio for both the Classical and Random Forest-Enhanced approaches.

## Dependencies

- **Flask**: Web framework to serve the application.
- **Pandas & Numpy**: Data handling and numerical calculations.
- **Matplotlib**: Plotting the efficient frontier.
- **SciPy**: Portfolio optimization via numerical optimization.
- **Scikit-Learn**: Implements Random Forest Regressor for predicting returns.
- **Yahoo Finance (`yfinance`)**: Fetches historical stock data.

## Error Handling

- **Input Validation**: If the input tickers are invalid or if the selected date range yields insufficient data, the application will show an appropriate error message.
- **Predictive Accuracy Warning**: If the Random Forest model's predictive power is low, the application will alert the user that the selected time frame or dataset might be inadequate for reliable forecasting.
- **Setup Errors**: If there are issues during the setup, such as missing dependencies, detailed error messages will be shown, and the setup script will prompt the user to fix the issue.

## Known Limitations

- **Data Availability**: If Yahoo Finance data is unavailable or incomplete, the program might fail to produce accurate results.
- **Model Performance**: The Random Forest model may underperform for very short time frames or very few tickers due to insufficient data.

## Future Improvements

- **GPU Support**: Add GPU processing for faster Random Forest training and optimization.
- **Additional Optimization Models**: Integrate more advanced forecasting models, such as XGBoost or Neural Networks, to improve prediction accuracy.
- **Improved User Error Feedback**: Enhance the detail in user error messages to help identify specific input problems more easily.

## Contributing

Contributions are welcome! If you encounter issues or have suggestions for new features, feel free to create a pull request or open an issue on GitHub.

## License

This project is licensed under the MIT License - see the LICENSE file for details.