# Synthetic Data Generation and Portfolio Backtesting

This notebook (`synthetic.ipynb`) is designed to simulate a multi-asset financial market and backtest various portfolio allocation strategies on the generated synthetic data.

## Workflow

The notebook follows a two-stage process:

### 1. Synthetic Market Data Generation

- It begins by loading a real-world covariance matrix to ensure the simulated assets have realistic correlations.
- A custom function, `sim_mkt_data_highdim`, generates price paths for multiple assets using a stochastic differential equation (SDE). This simulation incorporates a time-varying drift component, creating a dynamic market environment.
- The output is a high-dimensional dataset representing the price evolution of multiple stocks over a long period (e.g., 12 years).

### 2. Rolling-Window Backtesting

- The synthetic data is then used in a monthly rolling-window backtesting framework.
- In each month, the simulation uses the preceding 10 years of data for training and the current month for testing.
- The core of the backtest is the comparison of three different portfolio management strategies:
    1.  **DRMV (Distributionally Robust Mean-Variance)**: A modern portfolio theory approach that accounts for uncertainty in the distribution of asset returns.
    2.  **Kara (Karatzas)**: A classical, non-robust portfolio optimization model that serves as a benchmark.
    3.  **DRBC (Distributionally Robust Bellman Control)**: A sophisticated strategy based on dynamic programming that explicitly models ambiguity in the market's drift. It constructs an "ambiguity set" around a reference model and optimizes for the worst-case scenario within that set.

## Objective

The primary goal of this notebook is to compare the performance of distributionally robust portfolio strategies (DRMV and DRBC) against a classical benchmark (Kara) in a controlled, simulated environment. By analyzing the wealth accumulation under each strategy, the research aims to demonstrate the potential benefits of robust optimization techniques in financial asset management.

## How to Run

1.  Ensure you have the necessary helper files (`helper.py`, `calculate_delta.py`, `drmv_riskfree.py`) and data files (`real_data_sigma.npy`, `df_date.csv`) in the same directory.
2.  Run the cells sequentially in the Jupyter Notebook `synthetic.ipynb`. The final cell will execute the main simulation and backtest, printing the progress.
3.  The output will be the wealth evolution lists for the three strategies, which can then be used for performance analysis and plotting.

## Parameters

sigma_real: is a real annualized volatility matrix (cholesky of cov matrix) using 2014-2024 stock data. Can consider to change the scale by dividing 2,3 etc.

r: interest rate

num_stocks: how many stocks we are investing in

plan_time: when optimizing, how far the terminal wealth we care? Can be we optimize for one month, and trade according for this month; or we optimize for terminal wealth after two months, yet trade for one month.

T and dt: that's paired. Currently T=1 means 1 year and dt=1/252 means one day. However, under current setting this may lead to very volatile stocks, making drbc and kara crash. Can consider T=1 means 10 years and dt=1/2520 means one day.