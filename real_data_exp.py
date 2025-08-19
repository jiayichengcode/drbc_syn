import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from helper import *
from calculate_delta import *
import sys
from sklearn.covariance import LedoitWolf
import os
from drmv_riskfree import *

def run_single_backtest_select_stocks(
    training_data: pd.DataFrame,
    selected_perms: np.ndarray,
    annual_target_return: float,
    r: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate DRMV weights for a single rebalancing period.
    """
    # Resample to monthly returns from daily data
    monthly_returns_train = training_data.pivot_table(
        index='date', columns='permno', values='ret'
    ).resample('M').apply(lambda x: (1 + x).prod() - 1).dropna()
    
    # Add risk-free asset
    risk_free_returns = (1 + r)**(1/12) - 1 + np.random.normal(0, 0.0003, len(monthly_returns_train))
    monthly_returns_train[0] = risk_free_returns
    
    perms_with_rf = np.append(selected_perms, 0)
    
    # Ensure columns are in the correct order
    monthly_returns_train = monthly_returns_train.reindex(columns=perms_with_rf)

    # Check for sufficient training data
    if monthly_returns_train.shape[0] < 118:
        # Return equal weights if not enough data
        new_weights = np.ones(len(selected_perms) + 1) / (len(selected_perms) + 1)
    else:
        new_weights = get_drmv_weights(monthly_returns_train, annual_target_return)
        
    return new_weights


def main(r=0.02, seed=42, start_date='2015-01-01', end_date='2024-12-31', beta=-3):
    # Step 1: Load data
    sp500_all = pd.read_csv('sp500_data.csv.gz')
    sp500_index = pd.read_csv('sp500_index.csv.gz')

    sp500_all['date'] = pd.to_datetime(sp500_all['date'])
    
    sp500_index.rename(columns={'caldt':'date', 'sprtrn':'ret'}, inplace=True)
    sp500_index['date'] = pd.to_datetime(sp500_index['date'])

    df = sp500_all
    
    # Sort by permno and date to ensure proper ordering for log return calculation
    df = df.sort_values(['permno', 'date'])
    
    # Calculate log returns for each stock
    df['log_ret'] = df.groupby('permno')['prc_adjusted'].transform(lambda x: np.log(x / x.shift(1)))
    
    # Remove rows with missing log returns (first observation for each stock)
    df = df.dropna(subset=['log_ret'])
    
    
    # Step 1: Find stocks with complete data from 2005-12-31 to 2015-01-01
    initial_start = pd.to_datetime(start_date) - relativedelta(years=10)    
    initial_end = pd.to_datetime(start_date) - pd.Timedelta(days=1)
    
    # Get stocks that have data in the initial period
    initial_period_data = df[(df['date'] >= initial_start) & (df['date'] <= initial_end)]
    
    # Count trading days in the initial period for validation
    total_trading_days = initial_period_data['date'].nunique()
    # print(f"Total trading days in initial period: {total_trading_days}")
    
    # Find stocks with sufficient data coverage (at least 80% of trading days)
    stock_coverage = initial_period_data.groupby('permno')['date'].nunique()
    min_required_days = int(total_trading_days)  # Require at least 80% coverage
    valid_stocks_initial = stock_coverage[stock_coverage >= min_required_days].index.tolist()
    
    # print(f"Stocks with sufficient data in initial period: {len(valid_stocks_initial)}")
    
    # Sample 20 stocks from those with complete initial data
    np.random.seed(seed)  # For reproducibility
    selected_stocks = np.sort(np.random.choice(valid_stocks_initial, 20, replace=False))
    
    print(f"Initially selected stocks: {selected_stocks}")
    
    # Step 2: Process monthly data starting from 2015-01-01
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    month_starts = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    current_stocks = selected_stocks.copy()
    kara_wealth_list = [1]
    drbc_wealth_list = [1]
    for i, current_month in enumerate(tqdm(month_starts)):
        # print(f"\nProcessing month {i+1}/{len(month_starts)}: {current_month}")
        
        # Define time windows
        train_start = current_month - relativedelta(years=10)
        train_end = current_month - pd.Timedelta(days=1)
        test_start = current_month
        test_end = month_starts[i+1] - pd.Timedelta(days=1) if i+1 < len(month_starts) else pd.to_datetime(end_date)
        
        # print(f"Previous 10 years: {train_start.date()} to {train_end.date()}")
        # print(f"Next month: {test_start.date()} to {test_end.date()}")
        
        prev_to_next_dates = df[(df['date'] >= train_start) & (df['date'] <= test_end)]['date'].nunique()
        
        # Efficiently find all stocks with 100% coverage using vectorized operations
        # Get data for the entire period (train + test)
        full_period_start = train_start
        full_period_end = test_end
        full_period_data = df[(df['date'] >= full_period_start) & (df['date'] <= full_period_end)]
        full_period_stock_dates = full_period_data.groupby('permno')['date'].nunique()
        all_valid_stocks = full_period_stock_dates[full_period_stock_dates >= prev_to_next_dates].index.values
        
        # Check which current stocks are still valid
        valid_current_stocks = np.intersect1d(current_stocks, all_valid_stocks)
        
        # print(f"Current stocks with 100% coverage: {len(valid_current_stocks)} out of {len(current_stocks)}")
        # print(f"Total stocks available with 100% coverage: {len(all_valid_stocks)}")
        
        # If we need to replace stocks to maintain 20 stocks
        stocks_needed = 20 - len(valid_current_stocks)
        
        if stocks_needed > 0:
            # print(f"Need to find {stocks_needed} replacement stocks")
            
            # Find replacement candidates (exclude currently valid stocks)
            replacement_candidates = np.setdiff1d(all_valid_stocks, valid_current_stocks)
            
            # print(f"Available replacement candidates: {len(replacement_candidates)}")
            # add to 20 stocks
            stocks_to_add = np.random.choice(replacement_candidates, stocks_needed, replace=False)
            # Use all available replacements, even if less than needed
            current_stocks = np.sort(np.concatenate([valid_current_stocks, stocks_to_add]))

        else:
            current_stocks = valid_current_stocks
            # print("All current stocks are valid, no replacement needed")
        
        # print(f"Final stock selection for this month: {current_stocks}")
        # print(f"Number of stocks: {len(current_stocks)}")
        
        # Get training data for the selected stocks
        pretrain_data = df[(df['date'] >= train_start) & (df['date'] <= train_end) & 
                          (df['permno'].isin(current_stocks))]
        length = len(pretrain_data) / len(current_stocks)
        prev_sigma_start_dt = (train_start - relativedelta(months=1))
        to_get_B_data = df[(df['date'] >= prev_sigma_start_dt) & (df['date'] <= train_end) & 
                          (df['permno'].isin(current_stocks))]
        _, matrix, _, _ = compute_annualized_matrix(to_get_B_data, start_month=train_start, end_month=train_end, permno_list=current_stocks)

        ret_matrix = pretrain_data.pivot(index='date', columns='permno', values='ret')
        ret_matrix = ret_matrix.fillna(0)
        
        lw = LedoitWolf()
        lw.fit(ret_matrix)
        # Calculate annualized covariance matrix
        cov = lw.covariance_ * 252
        sigma_mat = np.linalg.cholesky(cov)
        curr_data = df[(df['date'] <= test_end) & (df['date'] >= current_month)&(df['permno'].isin(current_stocks))]
        dt = 1/length
        t_list = np.linspace(0, 1, int(length)+1)
        price_st = curr_data.pivot(index='date', columns='permno', values='prc_adjusted').fillna(method='ffill').values
        curr_all_ret = price_st[-1] / price_st[0] - 1
        yt = St_to_Yt_vectorized(price_st[np.newaxis, :, :], price_st[0], sigma_mat, r, t_list[1:int(len(curr_data)/20)+1]) # can be t_list[0:len(curr_data)]
        k= solve_k_with_EL(matrix, r=r, sigma=sigma_mat, T=1/12, beta=beta, num_y=500, seed=seed)
        
        # calculate radius small delta
        var = calculate_z_var(T=0.25, r=r, sigma=sigma_mat, B_support=matrix, p_dist=np.ones(matrix.shape[0])/matrix.shape[0], beta=beta, k=k)
        small_delta_array = (np.random.normal(0, np.sqrt(var), size=100)**2)*(calculate_numerator(0.25, r, sigma_mat, matrix, np.ones(matrix.shape[0])/matrix.shape[0], beta=beta, k=k)/calculate_denominator(0.25, r, sigma_mat, matrix, np.ones(matrix.shape[0])/matrix.shape[0], beta=beta, k=k))
        small_delta = np.percentile(small_delta_array, 95)/120
        
        # calculate delta_B
        delta_B = compute_big_delta_star(matrix, r, 1/12, beta, small_delta, sigma_mat)
        kara_frac = pi_fraction_exact(t=t_list[0], Yt=yt[0][0], T=1/12, alpha=beta, r=r, sigma=sigma_mat,
                    joint_z_vectors=matrix, p_dist=np.ones(matrix.shape[0])/matrix.shape[0],
                    num_expectation_samples=5000, seed=seed)
        
        drbc_frac = pi_fraction_exact(t=t_list[0], Yt=yt[0][0], T=1/12, alpha=beta, r=r, sigma=sigma_mat,
                    joint_z_vectors=matrix-delta_B, p_dist=np.ones(matrix.shape[0])/matrix.shape[0],
                    num_expectation_samples=5000, seed=seed)
        month_r = np.power(1+r, 1/12)-1
        kara_wealth_list.append((kara_wealth_list[-1]*(1-kara_frac.sum())*month_r+np.dot(kara_frac, curr_all_ret)+1)*kara_wealth_list[-1])
        drbc_wealth_list.append((drbc_wealth_list[-1]*(1-drbc_frac.sum())*month_r+np.dot(drbc_frac, curr_all_ret)+1)*drbc_wealth_list[-1])
        
    return kara_wealth_list, drbc_wealth_list


if __name__ == "__main__":
    results = main()