import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import sys
import warnings
import os
warnings.filterwarnings("ignore")

try:
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import default_converter, numpy2ri
    from rpy2.rinterface_lib.embedded import RRuntimeError

    # 加载 R 库
    ro.r('library(CompQuadForm)')

    # 定义 R 函数字符串
    R_UNROOT_FUNC_STRING = """
    get_quantile <- function(Lambda1, alpha) {
        # 定义一个内部函数来寻找根
        fun1 <- function(x) { davies(x, Lambda1)$Qq - alpha }
        
        # 使用 tryCatch 来处理 uniroot 可能发生的错误
        result <- tryCatch({
            # uniroot 在一个指定的区间内寻找函数的根 (即函数值为0的点)
            uniroot(fun1, c(1e-9, 100000))$root
        }, error = function(e) {
            # 如果 uniroot 失败 (例如，在区间内没有找到根)，则返回 NA
            # 这比让整个程序崩溃要好得多
            return(NA_real_)
        })
        
        return(result)
    }
    """
    
    # 步骤 1: 在 R 的全局环境中执行函数的定义
    ro.r(R_UNROOT_FUNC_STRING)
    
    # 步骤 2: 从 R 的全局环境中获取这个函数对象
    r_get_quantile = ro.globalenv['get_quantile']
    
    RPY2_AVAILABLE = True
    print("rpy2 setup and R function definition successful.")
    print("Successfully fetched 'get_quantile' function from R environment.")


except ImportError:
    print("="*60); print("CRITICAL: rpy2 is not installed. Please run 'pip install rpy2'."); print("="*60)
    RPY2_AVAILABLE = False
    r_get_quantile = None # 确保在失败时变量存在
except RRuntimeError as e:
    print("="*60); print("CRITICAL: An R runtime error occurred during setup."); print(f"Error: {e}")
    print("\nPlease check if R is installed and 'CompQuadForm' is installed in R."); print("="*60)
    RPY2_AVAILABLE = False
    r_get_quantile = None # 确保在失败时变量存在
except Exception as e:
    print("="*60); print("CRITICAL: An unexpected error occurred during rpy2 setup."); print(f"Error: {e}"); print("="*60)
    RPY2_AVAILABLE = False
    r_get_quantile = None # 确保在失败时变量存在



# =============================================================================
#
# Part 1: DRMV 算法核心 (DRMV Solver)
#
# =============================================================================
def _calculate_l0_eigenvalues(monthly_returns: pd.DataFrame, monthly_rho: float):
    # This function remains unchanged
    S1_np = monthly_returns.values; n, d = S1_np.shape
    s1_average = np.mean(S1_np, axis=0).reshape(d, 1)
    Acov = S1_np.T @ S1_np / n
    try: Acov_inv = np.linalg.inv(Acov)
    except np.linalg.LinAlgError: return None, None, None, None, None
    one_vec = np.ones((d, 1)); QR1 = (one_vec.T @ Acov_inv @ s1_average)[0, 0]; QR2 = (one_vec.T @ Acov_inv @ one_vec)[0, 0]; QR3 = (s1_average.T @ Acov_inv @ s1_average)[0, 0]
    QR_matrix = np.array([[QR3, QR1], [QR1, QR2]]); Mb_vector = np.array([[monthly_rho], [1]])
    try: lambda_c = np.linalg.inv(QR_matrix) @ Mb_vector
    except np.linalg.LinAlgError: return None, None, None, None, None
    lambda_1, lambda_2 = lambda_c[0, 0], lambda_c[1, 0]
    hat_phi = lambda_1 * (Acov_inv @ s1_average) + lambda_2 * (Acov_inv @ one_vec)
    CovZ0 = np.cov(S1_np.T); vec_RR_T_samples = np.einsum('ni,nj->nij', S1_np, S1_np).reshape(n, d * d)
    CovS_flat = np.cov(vec_RR_T_samples.T); CovS_tensor = CovS_flat.reshape(d, d, d, d)
    phi_outer = hat_phi @ hat_phi.T; CovYphi = np.einsum('lm,kjlm->kj', phi_outer, CovS_tensor)
    transform_matrix = np.eye(d) - one_vec @ hat_phi.T
    if abs(lambda_1) < 1e-9: return None, None, None, None, None
    CovZ1 = (transform_matrix @ CovYphi @ transform_matrix.T) / (lambda_1 ** 2)
    CovZ = CovZ0 + CovZ1; L0_eigenvalues = np.linalg.eig(CovZ)[0].real
    return L0_eigenvalues, s1_average, Acov, hat_phi, n

def _calculate_delta_rpy2(L0_eigenvalues: np.ndarray, monthly_returns: pd.DataFrame, s1_average: np.ndarray, confidence_level=0.95):
    """
    [REVISED] Uses the modern localconverter context manager for safe R calls.
    """
    n = monthly_returns.shape[0]
    try:
        sample_cov_inv = np.linalg.inv(np.cov(monthly_returns.values.T))
    except np.linalg.LinAlgError: return None
    mu_Sigma_inv_mu = s1_average.T @ sample_cov_inv @ s1_average
    parad = 4 * (1 - mu_Sigma_inv_mu[0,0])
    if parad <= 1e-6: return None
    
    alpha = 1.0 - confidence_level
    clipped_eigenvalues = np.clip(L0_eigenvalues, a_min=None, a_max=500)
    
    L0_quantile = None
    # 使用 localconverter 来创建临时的、安全的转换上下文
    with localconverter(default_converter + numpy2ri.converter) as cv:
        r_result = r_get_quantile(clipped_eigenvalues, alpha)
        # 检查 R 是否返回了 NA (失败标志)
        if r_result[0] is not ro.NA_Real:
            L0_quantile = r_result[0]

    if L0_quantile is None:
        return None
        
    delta = L0_quantile / n / parad
    return delta

def _solve_robust_portfolio(monthly_returns: pd.DataFrame, monthly_rho: float, delta: float, Acov: np.ndarray, s1_average: np.ndarray, Mphi: np.ndarray, confidence_level=0.95):
    # This function remains unchanged
    n, d = monthly_returns.shape; p_norm = 2
    Evar = np.sqrt(max(0, Mphi.T @ np.cov(monthly_returns.values.T) @ Mphi))[0,0]
    epsilon = 1.0 - confidence_level; lel = norm.ppf(epsilon)
    denominator = np.sqrt(n) * np.linalg.norm(Mphi, ord=p_norm) * np.sqrt(delta)
    if denominator < 1e-9: v0 = 1.0
    else: v0 = 1 - (lel * Evar) / denominator
    v0 = max(v0, 0)
    alpha_bar = monthly_rho - np.linalg.norm(Mphi, ord=p_norm) * np.sqrt(delta) * v0
    def objective_func(phi):
        phi = phi.reshape(d, 1); var_term = phi.T @ Acov @ phi - (phi.T @ s1_average)**2
        risk_term = np.sqrt(max(var_term, 0))
        regularization_term = np.sqrt(delta) * np.linalg.norm(phi, ord=p_norm)
        return risk_term + regularization_term
    def nonlinear_constraint_func(phi):
        phi = phi.reshape(d, 1)
        constraint_value = (phi.T @ s1_average) - np.sqrt(delta) * np.linalg.norm(phi, ord=p_norm) - alpha_bar
        return constraint_value[0,0]
    x0 = np.ones(d) / d; bounds = tuple((0, 1) for _ in range(d))
    eq_cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}; nlin_cons = {'type': 'ineq', 'fun': nonlinear_constraint_func}
    constraints = [eq_cons, nlin_cons]
    result = minimize(fun=objective_func, x0=x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'disp': False, 'ftol': 1e-9})
    return result.x if result.success else np.ones(d) / d

def get_drmv_weights(monthly_returns_df: pd.DataFrame, annual_target_return: float):
    # This function remains unchanged
    monthly_rho = (1 + annual_target_return)**(1/12) - 1; d = monthly_returns_df.shape[1]
    l0_eig, mu_n, E_rr, phi_n, n_samples = _calculate_l0_eigenvalues(monthly_returns_df, monthly_rho)
    if l0_eig is None: return np.ones(d) / d
    delta = _calculate_delta_rpy2(l0_eig, monthly_returns_df, mu_n)
    if delta is None or delta < 0: return np.ones(d) / d
    weights = _solve_robust_portfolio(monthly_returns_df, monthly_rho, delta, E_rr, mu_n, phi_n)
    return weights

# =============================================================================
#
# Part 2 & 3: Backtesting Engine & Main Runner
# (These parts remain unchanged from the previous correct version)
#
# =============================================================================
def run_single_backtest(
    all_data_df: pd.DataFrame, 
    n_stocks: int,
    transaction_cost: float,
    annual_target_return: float,
    backtest_start_date: str,
    backtest_end_date: str, 
    all_seed: int = 42
):
    rebalancing_dates = pd.date_range(start=backtest_start_date, end=backtest_end_date, freq='MS')
    wealth = 1.0
    current_weights = np.array([])
    current_perms = []
    wealth_history = {rebalancing_dates[0] - pd.DateOffset(days=1): wealth}

    for num, i in enumerate(range(len(rebalancing_dates))):
        rebal_date = rebalancing_dates[i]
        
        if i > 0:
            prev_rebal_date = rebalancing_dates[i-1]
            if len(current_perms) > 0:
                oos_data = all_data_df[(all_data_df['permno'].isin(current_perms)) & (all_data_df['date'] >= prev_rebal_date) & (all_data_df['date'] < rebal_date)]
                monthly_returns_oos = oos_data.pivot_table(index='date', columns='permno', values='ret').resample('M').apply(lambda x: (1 + x).prod() - 1)
                if monthly_returns_oos.empty:
                    portfolio_return = 0
                else:
                    actual_returns_series = monthly_returns_oos.iloc[0].reindex(current_perms)
                    actual_returns = np.nan_to_num(actual_returns_series.values, nan=-1.0)
                    portfolio_return = np.dot(current_weights, actual_returns)
            else:
                portfolio_return = 0
            wealth *= (1 + portfolio_return)
            if portfolio_return != -1 and len(current_perms) > 0:
                 weights_after_return = current_weights * (1 + actual_returns) / (1 + portfolio_return)
            else:
                 weights_after_return = np.zeros_like(current_weights)
        else:
            weights_after_return = np.array([])
        
        training_start_date = rebal_date - relativedelta(years=10)
        future_date = rebal_date + relativedelta(months=1)
        active_stocks = all_data_df[(all_data_df['start'] <= rebal_date) & (all_data_df['ending'] >= future_date)]
        eligible_stocks = []
        for permno in active_stocks['permno'].unique():
            if all_data_df[all_data_df['permno'] == permno]['date'].min() <= training_start_date:
                eligible_stocks.append(permno)
        
        if len(eligible_stocks) < n_stocks:
            new_weights = np.array([])
            selected_perms = []
        else:
            np.random.seed(all_seed*1000+num)  # Ensure reproducibility
            selected_perms = np.random.choice(eligible_stocks, n_stocks, replace=False)
            training_data = all_data_df[(all_data_df['permno'].isin(selected_perms)) & (all_data_df['date'] >= training_start_date) & (all_data_df['date'] < rebal_date)]
            monthly_returns_train = training_data.pivot_table(index='date', columns='permno', values='ret').resample('M').apply(lambda x: (1 + x).prod() - 1).dropna()
            if monthly_returns_train.shape[0] < 118:
                new_weights = np.ones(n_stocks) / n_stocks
            else:
                monthly_returns_train = monthly_returns_train[selected_perms]
                new_weights = get_drmv_weights(monthly_returns_train, annual_target_return)
        
        old_holdings = pd.Series(weights_after_return, index=current_perms, name='old').fillna(0)
        new_target = pd.Series(new_weights, index=selected_perms, name='new').fillna(0)
        combined_df = pd.concat([old_holdings, new_target], axis=1).fillna(0)
        turnover = np.sum(np.abs(combined_df['new'] - combined_df['old']))
        cost = wealth * turnover * transaction_cost
        wealth -= cost
        
        wealth_history[rebal_date] = wealth
        current_weights = new_weights
        current_perms = selected_perms
        
    return pd.Series(wealth_history, name='Wealth')

if __name__ == "__main__":
    # sys input arguments for the script
    if len(sys.argv) > 1:
        try:
            N_EXPERIMENTS = int(sys.argv[1])
            seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
        except ValueError as e:
            print(f"Error parsing command line arguments: {e}")
            exit()
    N_STOCKS = 20
    TRANSACTION_COST = 0
    ANNUAL_TARGET_RETURN = 0.10
    BACKTEST_START = '2000-01-01'
    BACKTEST_END = '2017-01-01'
    DATA_FILE = 'sp500_data.csv.gz'

    try:
        print(f"Loading data from {DATA_FILE}...")
        df = pd.read_csv(DATA_FILE)
        for col in ['date', 'start', 'ending']:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        df.dropna(subset=['date', 'start', 'ending', 'permno', 'ret'], inplace=True)
        print("Data loaded and preprocessed successfully.")
    except FileNotFoundError:
        print(f"Error: Data file '{DATA_FILE}' not found.")
        exit()

    all_wealth_curves = []
    print(f"\nStarting {N_EXPERIMENTS} backtesting experiments...")
    for i in tqdm(range(N_EXPERIMENTS), desc="Running Experiments"):
        wealth_series = run_single_backtest(
            all_data_df=df, n_stocks=N_STOCKS, transaction_cost=TRANSACTION_COST,
            annual_target_return=ANNUAL_TARGET_RETURN, backtest_start_date=BACKTEST_START,
            backtest_end_date=BACKTEST_END, all_seed=seed 
        )
        all_wealth_curves.append(wealth_series)

    print("All experiments completed.")
    
    if not all_wealth_curves: print("No wealth curves were generated.")
    else:
        results_df = pd.concat(all_wealth_curves, axis=1)
        results_df.to_csv(f'res/drmv_backtest_results_{seed}_n{N_STOCKS}.csv', index=True)
        print(f"Results saved to 'res/drmv_backtest_results_{seed}_n{N_STOCKS}.csv'.")