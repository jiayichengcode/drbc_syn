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
import pickle

def sim_mkt_data_highdim(T, num_paths, 
                         sigma, s0, dt=1/2520, seed=1):
    """
    使用联合离散分布，模拟高维市场数据。

    参数:
        T (float): 总模拟时间 (例如，1.0 代表一年)。
        joint_z_vectors (ndarray): 预定义的场景向量，形状为 (m, dim)。
        p_dist (ndarray): 每个场景向量对应的概率，形状为 (m,)。
        num_paths (int): 要模拟的路径数量。
        sigma (ndarray): **波动率矩阵 σ**，形状为 (dim, dim)。
        s0 (float): 初始价格。  
        dt (float): 时间步长。

    返回:
        S (ndarray): 模拟的股价路径，形状 (num_paths, N+1, dim)。
        t_list (ndarray): 时间点列表，形状 (N+1,)。
        b_vectors (ndarray): 为每条路径选择的漂移向量，形状 (num_paths, dim)。
        W (ndarray): 模拟的多维布朗运动，形状 (num_paths, N+1, dim)。
    """
    dim = sigma.shape[0]
    N = int(T / dt)  # 时间步数量
    t_list = np.linspace(0, T, N + 1)
    np.random.seed(seed)
    # --- Bt=B0*(1+np.cos(2*np.pi*rand_k*t)) /2---
    # 抽取 m 个场景的索引
    # num_scenarios = joint_z_vectors.shape[0]
    # scenario_indices = np.arange(num_scenarios)
    # chosen_indices = np.random.choice(scenario_indices, p=p_dist, size=num_paths, replace=True)


    B0=0.07
    rand_k = np.random.normal(10, 20, sigma.shape[0]) # TODO: make k larger so fluctuate weekly or bi-weekly; can change to fixed numbers rather than random
    # generate b_vectors, finally shape is (N, dim)
    b_vectors = np.zeros((N, dim))
    
    # Create meshgrid for proper broadcasting: t (N,) and rand_k (dim,)
    # We use t_list[:-1] to get N time steps (excluding the last one)
    t_mesh, rand_k_mesh = np.meshgrid(t_list[:-1], rand_k, indexing='ij')
    # Now t_mesh and rand_k_mesh both have shape (N, dim)
    b_vectors = B0*(1 + 2*np.cos(2*np.pi*rand_k_mesh*t_mesh))/2

    # --- 2. 模拟多维布朗运动 W ---
    # 生成标准正态分布的增量
    
    normal_increments = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(num_paths, N, dim))
    
    W = np.zeros((num_paths, N + 1, dim))
    # 通过对增量进行累积求和来构建布朗运动路径
    W[:, 1:, :] = np.cumsum(normal_increments, axis=1)

    # --- 3. 模拟股价路径 S ---
    S = np.zeros((num_paths, N + 1, dim))

    S[:, 0, :] = s0 * np.ones((num_paths, dim))

    for i in range(N):
        # 提取当前状态
        current_S = S[:, i, :]
        
        # 布朗运动的增量 dW
        dW = W[:, i + 1, :] - W[:, i, :]
        
        # --- 计算 SDE 的增量 dS ---
        # 漂移项: b*dt
        drift_term = b_vectors[i] * dt
        
        # 波动率项: σ * dW
        # 使用矩阵乘法 (@)，并对 sigma 进行转置以匹配批量操作的维度
        # (num_paths, dim) @ (dim, dim) -> (num_paths, dim)
        vol_term = dW @ sigma.T
        
        # 逐元素乘法计算 dS
        dS = current_S * (drift_term + vol_term)
        
        # 更新下一时间步的价格
        S[:, i + 1, :] = current_S + dS
    
    # for each path, if any negative items in S, remove this path 
    for i in range(num_paths):
        if np.any(S[i] < 0):
            S = np.delete(S, i, axis=0)
            W = np.delete(W, i, axis=0)
    
    # remove the first row of S
    S = S[:, 1:, :]
    W = W[:, 1:, :]
    b_vectors = b_vectors[1:, :]
    t_list = t_list[1:]
    
    return S, t_list, b_vectors, W

def sim_data_to_df(prices, real_trade_dates, types=60):
    """
    Converts a 2D numpy array of prices into a long-format pandas DataFrame.

    Args:
        prices (np.ndarray): A 2D numpy array of shape (T, dim), where T is the
                             number of time periods and dim is the number of stocks.

    Returns:
        pd.DataFrame: A DataFrame with columns: 'date', 'permno', 'ret', and 'prc'.
                      'permno' is the stock identifier, from 1 to dim.
    """
    T, dim = prices.shape
    dates = real_trade_dates[-T:]
    permnos = range(1, dim + 1)

    # Create a wide DataFrame for prices
    df_prc = pd.DataFrame(prices, index=dates, columns=permnos)
    df_prc.index.name = 'date'
    df_prc.columns.name = 'permno'

    # Calculate returns
    df_ret = df_prc.pct_change()

    # Stack prices and returns to convert to long format
    # dropna=False is important to keep all price entries, even with NaN returns for the first day
    s_prc = df_prc.stack(dropna=False).rename('prc')
    s_ret = df_ret.stack(dropna=False).rename('ret')

    # Combine into a single DataFrame, aligning on the (date, permno) index
    df = pd.concat([s_ret, s_prc], axis=1)

    # Reset index to get 'date' and 'permno' as columns
    df = df.reset_index()
    
    # Reorder columns to the desired format
    df = df[['date', 'permno', 'ret', 'prc']]

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['type'] = (df.groupby('date').ngroup() % types) + 1
    df['prc_adjusted'] = df['prc']
    df['log_ret'] = df.groupby('permno')['prc'].transform(lambda x: np.log(x / x.shift(1)))
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def main_sim_new(input_df, r=0.02, seed=42, start_date='2024-01-01', end_date='2024-12-31',
                 beta=-3, num_stocks=20, plan_time=1/120, dt=1/2520, sigma_real=None):
    # Step 1: Load data
    
    df = input_df
    
    # Sort by permno and date to ensure proper ordering for log return calculation
    df = df.sort_values(['permno', 'date'])
    df['date'] = pd.to_datetime(df['date'])
    # Calculate log returns for each stock
    
    
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
    
    # Sample num_stocks stocks from those with complete initial data
    np.random.seed(seed)  # For reproducibility
    selected_stocks = np.sort(np.random.choice(valid_stocks_initial, num_stocks, replace=False))
    
    print(f"Initially selected stocks: {selected_stocks}")
    
    # Step 2: Process monthly data starting from 2015-01-01
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    month_starts = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    current_stocks = selected_stocks.copy()
    kara_wealth_list = [1]
    drbc_wealth_list = [1]
    drmv_wealth_list = [1]
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
        stocks_needed = num_stocks - len(valid_current_stocks)
        
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
        pretrain_data_drmv = pretrain_data.copy()
        pretrain_data_drmv['ret'] = (pretrain_data_drmv['ret']/dt)/252
        drmv_weights = run_single_backtest_select_stocks(
            training_data=pretrain_data_drmv,
            selected_perms=current_stocks,
            annual_target_return=0.105,
            r=r)
        length = len(pretrain_data) / len(current_stocks)
        prev_sigma_start_dt = (train_start - relativedelta(months=1))
        #prev_sigma_start_dt = (train_start - relativedelta(years=1))
        to_get_B_data = df[(df['date'] >= prev_sigma_start_dt) & (df['date'] <= train_end) & 
                          (df['permno'].isin(current_stocks))]
        matrix, _ = compute_annualized_matrix_type(to_get_B_data, sigma_real, dt=dt)
        # for dt=1/2520 ang T=1.2 case
        #matrix = matrix*10
        # ret_matrix = pretrain_data.pivot(index='date', columns='permno', values='ret')
        # ret_matrix = ret_matrix.fillna(0)
        
       
        # # use real sigma matrix (already annualized)
        # lw = LedoitWolf(store_precision=False, assume_centered=True)
        # lw.fit(ret_matrix)
        # cov = lw.covariance_ * 252
        sigma_mat = sigma_real #np.linalg.cholesky(cov) #sigma_real/9 # 
        curr_data = df[(df['date'] <= test_end) & (df['date'] >= current_month)&(df['permno'].isin(current_stocks))]
        dt = 1/length
        t_list = np.linspace(0, 1, int(1/dt))
        price_st = curr_data.pivot(index='date', columns='permno', values='prc_adjusted').fillna(method='ffill').values
        curr_all_ret = price_st[-1] / price_st[0] - 1
        yt = St_to_Yt_vectorized(price_st[np.newaxis, :, :], price_st[0], sigma_mat, r, t_list[1:int(len(curr_data)/20)+1]) # can be t_list[0:len(curr_data)]
        k= solve_k_with_EL(matrix, r=r, sigma=sigma_mat, T=plan_time, beta=beta, num_y=1000, seed=seed)

        # calculate radius small delta (using 1 year, represents by T=1)
        var = calculate_z_var(T=plan_time, r=r, sigma=sigma_mat, B_support=matrix, p_dist=np.ones(matrix.shape[0])/matrix.shape[0], beta=beta, k=k)
        small_delta_array = (np.random.normal(0, np.sqrt(var), size=100)**2)*(calculate_numerator(plan_time, r, sigma_mat, matrix, np.ones(matrix.shape[0])/matrix.shape[0], beta=beta, k=k)/calculate_denominator(plan_time, r, sigma_mat, matrix, np.ones(matrix.shape[0])/matrix.shape[0], beta=beta, k=k))
        small_delta = np.percentile(small_delta_array, 95)/40
        rng = np.random.default_rng(seed)
        # calculate delta_B (using 1 year, represents by T=1)
        delta_B = compute_big_delta_star(matrix, r, plan_time, beta, small_delta, sigma_mat, rng=rng)
        
        month_r = np.power(1+r, (1/12)/(dt*252))-1 
        # add month_r to curr_all_ret for drmv
        curr_ret_for_drmv = np.append(curr_all_ret, month_r)
        daily_kara = 1
        daily_drbc = 1
        daily_r = np.power(1+r, (1/252)/(dt*252))-1
        # last day not trade since no price for next day
        for j in range(1,curr_data['date'].nunique()):
            kara_frac_daily = pi_fraction_exact(t=j*dt, Yt=yt[0][j-1], T=plan_time, alpha=beta, r=r, sigma=sigma_mat,
                        joint_z_vectors=matrix, p_dist=np.ones(matrix.shape[0])/matrix.shape[0],
                        num_expectation_samples=5000, seed=seed)
            drbc_frac_daily = pi_fraction_exact(t=j*dt, Yt=yt[0][j-1], T=plan_time, alpha=beta, r=r, sigma=sigma_mat, 
                        joint_z_vectors=matrix+delta_B, p_dist=np.ones(matrix.shape[0])/matrix.shape[0],
                        num_expectation_samples=5000, seed=seed)
            daily_kara *= (1-kara_frac_daily.sum())*daily_r+np.dot(kara_frac_daily, price_st[j] / price_st[j-1] - 1)+1
            daily_drbc *= (1-drbc_frac_daily.sum())*daily_r+np.dot(drbc_frac_daily, price_st[j] / price_st[j-1] - 1)+1
            
            
        # kara_wealth_list.append((kara_wealth_list[-1]*(1-kara_frac.sum())*month_r+np.dot(kara_frac, curr_all_ret)+1)*kara_wealth_list[-1])
        # drbc_wealth_list.append((drbc_wealth_list[-1]*(1-drbc_frac.sum())*month_r+np.dot(drbc_frac, curr_all_ret)+1)*drbc_wealth_list[-1])
        drmv_wealth_list.append(drmv_wealth_list[-1]*(1+np.dot(drmv_weights, curr_ret_for_drmv)))
        kara_wealth_list.append(daily_kara*kara_wealth_list[-1])
        drbc_wealth_list.append(daily_drbc*drbc_wealth_list[-1])

    return kara_wealth_list, drbc_wealth_list, drmv_wealth_list

def sharpe_ratio(wealths, risk_free_rate=0.02, dt=1/2520):
    returns = np.diff(wealths, axis=0) / wealths[:-1] - 252*dt*risk_free_rate/(len(wealths)-1)
    return np.mean(returns) * np.sqrt(((len(wealths)-1)/dt)/252) / np.std(returns)

def main(r=0.02, num_stocks=20, std = 0.4, dt=1/2520, T=1.2, seed=1, plan_time=1/120):
    sigma = std*np.eye(num_stocks)
    df_date=pd.read_csv('df_date.csv')
    prices_gen, t_list_gen, b_vectors_gen, W_gen = sim_mkt_data_highdim(T=T, num_paths=2, s0=10, sigma=sigma, dt=dt, seed=seed)
    sim_df = sim_data_to_df(prices_gen[0], df_date['date'].unique())
    b = main_sim_new(sim_df, dt=dt, sigma_real=sigma, r=r, seed=seed, plan_time=plan_time)
    sharpe_list = [sharpe_ratio(b[0]), sharpe_ratio(b[1]), sharpe_ratio(b[2])]
    out_dict = {'kara': sharpe_ratio(b[0]), 'drbc': sharpe_ratio(b[1]), 'drmv': sharpe_ratio(b[2]), 'sharpe_list': sharpe_list}
    # save out_dict to pickle
    with open(f'res/out_dict_{num_stocks}_{std}_{dt}_{seed}.pkl', 'wb') as f:
        pickle.dump(out_dict, f)
        
if __name__ == '__main__':
    input_seed = sys.argv[1]
    T = 1.2*2520*float(1/10080)
    main(r=0.02, num_stocks=20, dt=float(1/10080), T=T, seed=int(input_seed), plan_time=1/480, std=0.3)
    
    