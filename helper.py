import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import warnings
from numba import njit
import os
from drmv_riskfree import get_drmv_weights
warnings.filterwarnings("ignore")

def L_t_vectorized(t, z, y, r, sigma):
    """
    计算高维向量化的 L_t(t, z, y)。

    这个函数可以处理z和y的多种输入情况:
    - z: 单个向量 (dim,) 或 多个向量 (n_samples, dim)
    - y: 单个向量 (dim,) 或 多个向量 (m_samples, dim)

    参数:
        t (float): 时间参数。
        z (ndarray): 形状为 (dim,) 或 (n_samples, dim) 的数组。
        y (ndarray): 形状为 (dim,) 或 (m_samples, dim) 的数组。
        r (float or ndarray): 无风险利率。可以是标量或形状为 (dim,) 的向量。
        sigma (ndarray): 协方差矩阵，形状为 (dim, dim)。

    返回:
        根据输入y和z的维度，返回一个标量或一个数组。
        - z(1,d), y(1,d) -> scalar
        - z(n,d), y(1,d) -> (n,)
        - z(1,d), y(m,d) -> (m,)
        - z(n,d), y(m,d) -> (n, m)
    """
    # 确保 y 和 z 至少是二维的，方便统一处理
    original_z_dim = np.ndim(z)
    original_y_dim = np.ndim(y)
    
    y = np.atleast_2d(y)  # 形状变为 (m_samples, dim)
    z = np.atleast_2d(z)  # 形状变为 (n_samples, dim)

    if t == 0.0:
        # L_t 在 t=0 时为 1
        # 我们需要根据 z 和 y 的样本数决定输出的形状
        n_samples = z.shape[0]
        m_samples = y.shape[0]
        # 创建一个 (n_samples, m_samples) 的矩阵
        # 如果原始输入是1D，最后会通过 squeeze() 降维
        ones = np.ones((n_samples, m_samples))
        if original_z_dim == 1 and original_y_dim == 1:
            return 1.0
        if original_z_dim > 1 and original_y_dim == 1:
            return ones[:, 0]
        if original_z_dim == 1 and original_y_dim > 1:
            return ones[0, :]
        return ones


    # 预计算 sigma 的逆
    sigma_inv = np.linalg.inv(sigma)
    
    # z - r，支持 r 是标量或向量
    diff = z - r  # 形状 (n_samples, dim)
    
    # A = (z - r) @ sigma_inv
    A = diff @ sigma_inv # 形状 (n_samples, dim)
    
    # standardize A
    #A = A / np.linalg.norm(A, axis=1, keepdims=True)  # 归一化每个样本的 A
    
    # 计算点积项: A @ y.T
    # A: (n_samples, dim), y.T: (dim, m_samples) -> dot_term: (n_samples, m_samples)
    dot_term = A @ y.T
    
    # 计算平方范数项 ||A||^2
    # A**2 是元素级平方，沿维度1求和得到每个样本的范数平方
    norm_sq = np.sum(A**2, axis=1) # 形状 (n_samples,)
    
    # 计算指数项
    # norm_sq[:, np.newaxis] 将 (n_samples,) 变形为 (n_samples, 1)
    # 这允许它与 (n_samples, m_samples) 的 dot_term 进行广播相减
    exponent = dot_term - 0.5 * t * norm_sq[:, np.newaxis]
    # For each column, keep values between the 2nd and 98th percentile
    if exponent.shape[0] > 1: # Percentiles are meaningful only for >1 sample
        p2 = np.percentile(exponent, 5, axis=0, keepdims=True)
        p98 = np.percentile(exponent, 95, axis=0, keepdims=True)
        
        # Create a mask for values within the percentile range
        mask = (exponent >= p2) & (exponent <= p98)
        
        # Replace values outside the range with -inf so they become 0 after exp()
        exponent = np.where(mask, exponent, -np.inf)
    result = np.exp(exponent)
    
    # 根据原始输入维度，调整输出形状，使其更符合直觉
    if original_z_dim == 1 and original_y_dim == 1:
        return result[0, 0] # 返回标量
    if original_z_dim > 1 and original_y_dim == 1:
        return result[:, 0] # 返回 (n_samples,) 数组
    if original_z_dim == 1 and original_y_dim > 1:
        return result[0, :] # 返回 (m_samples,) 数组
    
    return result # 返回 (n_samples, m_samples) 矩阵


def F_tilde_p_exact(t, y, joint_z_vectors, p_dist, r, sigma):
    """
    精确计算 F_tilde_p(t, y) = E_P[L_t(t, Z, y)]。

    参数:
        t (float): 时间参数。
        y (ndarray): 形状为 (dim,) 或 (n_y, dim) 的向量。
        joint_z_vectors (ndarray): 预先生成的m个n维向量，形状为 (m, dim)。
        p_dist (ndarray): 对应每个z向量的概率，形状为 (m,)。
        r, sigma: 模型参数。

    返回:
        F_tilde_p 的精确值。
    """
    # 1. 一次性计算所有m个z向量的L_t值。
    # L_t_vectorized 会返回一个形状为 (m,) 或 (m, n_y) 的数组。
    l_values = L_t_vectorized(t, joint_z_vectors, y, r, sigma)
    
    # 2. 计算加权和（即期望）
    # 如果y是单个向量，l_values是(m,)，p_dist是(m,)，结果是标量。
    # 如果y是多个向量，l_values是(m, n_y)，p_dist是(m,)，
    # p_dist @ l_values (点积) 等价于 sum(p_i * l_i) for each y column
    # 结果是一个 (n_y,) 的数组。
    return p_dist @ l_values

def grad_F_tilde_p_exact(t, y, joint_z_vectors, p_dist, r, sigma):
    """
    精确计算 ∇_y F_tilde_p(t, y) = E_P[∇_y L_t(t, Z, y)]。
    """
    # 确保 y 至少是 2D
    original_y_dim = np.ndim(y)
    y = np.atleast_2d(y) # Shape (n_y, dim)
    dim = y.shape[1]
    
    # 1. 预计算所有 z 向量的梯度乘数 A(Z)
    sigma_inv = np.linalg.inv(sigma) # 假设sigma是波动率矩阵
    A = (joint_z_vectors - r) @ sigma_inv  # 形状 (m, dim)

    # 2. 计算 L_t 值
    l_values = L_t_vectorized(t, joint_z_vectors, y, r, sigma) # 形状 (m, n_y)
    
    # 3. 计算梯度的期望 E_P[L_t * A]
    # 我们需要为每个 y_j 计算 sum_i [p_dist[i] * l_values[i, j] * A[i, :]]
    # 这可以通过一个高效的矩阵乘法完成。
    # 首先，用概率对L值进行加权
    weighted_l_values = l_values * p_dist[:, np.newaxis] # 形状 (m, n_y)
    
    # 然后，计算加权后的L值与梯度乘数A的期望
    # weighted_l_values.T: (n_y, m) @ A: (m, dim) -> grad: (n_y, dim)
    grad = weighted_l_values.T @ A
    
    # 如果原始输入y是1D，则返回1D结果
    if original_y_dim == 1:
        return grad[0]
    return grad

def V_tilde_p_mixed(T, alpha, x0, r, sigma, 
                    joint_z_vectors, p_dist,
                    num_y_samples=10000, seed=None):
    """
    使用混合方法计算高维 V_tilde_p。
    内部对P的期望是精确的，外部对y的期望是蒙特卡洛。
    """
    if seed is not None:
        np.random.seed(seed)
    
    dim = joint_z_vectors.shape[1]
    
    # 1. 外层蒙特卡洛：生成 y 的样本
    mean_y = np.zeros(dim)
    cov_y = T * np.eye(dim) # 假设sigma是年化的，所以T=1
    y_samples = np.random.multivariate_normal(mean_y, cov_y, size=num_y_samples)
    
    # 2. 对所有 y 样本，一次性计算 F_tilde_p 的值 (使用我们的新精确函数)
    F_vals = F_tilde_p_exact(
        t=T, y=y_samples, joint_z_vectors=joint_z_vectors, 
        p_dist=p_dist, r=r, sigma=sigma
    )
    
    # 3. 根据公式计算 V_tilde_p (这部分不变)
    F_powered = F_vals ** (1 / (1 - alpha))
    mean_F_powered = np.mean(F_powered)
    term1 = ((x0 * np.exp(r * T)) ** alpha) / alpha
    term2 = mean_F_powered ** (1 - alpha)
    
    return term1 * term2

def St_to_Yt_vectorized(St, s0, sigma, r, t_list):
    """
    将多维股价路径 St 转换为其对应的底层多维标准布朗运动 Yt。

    此函数是完全向量化的，可以处理多个路径和多个时间步。

    参数:
        St (ndarray): 股价路径。
                      形状为 (num_paths, num_timesteps, dim)。
        s0 (ndarray): 初始股价向量。形状为 (dim,)。
        sigma (ndarray): 波动率矩阵。形状为 (dim, dim)。
        r (float or ndarray): 无风险利率。可以是标量或形状为 (dim,) 的向量。
        t_list (ndarray): 时间点列表。形状为 (num_timesteps,)。

    返回:
        ndarray: 对应的多维标准布朗运动路径 Yt。
                 形状与 St 相同 (num_paths, num_timesteps, dim)。
    """
    # 确保输入是 numpy array
    St = np.asarray(St)
    s0 = np.asarray(s0)
    sigma = np.asarray(sigma)
    r = np.asarray(r)
    t_list = np.asarray(t_list)

    # 1. 计算对数回报项: log(St / s0)
    # s0 (dim,) 会被广播到 St (num_paths, num_timesteps, dim) 的形状
    log_returns = np.log(St / s0)

    # 2. 计算 sigma 的逆矩阵
    sigma_inv = np.linalg.inv(sigma)

    # 3. 计算转换后的回报项: sigma_inv @ log_returns
    # 我们需要对每个路径和时间点的 (dim,) 向量左乘 sigma_inv (dim, dim)
    # 使用 np.einsum 或 @ 运算符可以高效完成此操作
    # 'pij,kj->pik' 的意思是：对于每个 p 和 i，计算 sum_j(A[p,i,j] * B[k,j])
    # 这等价于 (sigma_inv @ log_returns[p,i,:].T).T
    # 更现代的写法是直接使用 @，它会自动处理最后的维度
    # log_returns 的形状是 (..., dim), sigma_inv.T 的形状是 (dim, dim)
    # (log_returns @ sigma_inv.T) 的形状是 (..., dim)
    transformed_returns = log_returns @ sigma_inv.T

    # 4. 计算漂移修正项
    # a) 计算每项资产的方差: diag(sigma @ sigma.T)
    variances = np.diag(sigma @ sigma.T)  # 形状 (dim,)
    
    # b) 计算括号内的向量: r - 0.5 * variances
    # r (标量或向量) 会被广播到 variances (dim,) 的形状
    drift_vector = r - 0.5 * variances  # 形状 (dim,)
    
    # c) 计算完整的漂移乘数: sigma_inv @ drift_vector
    drift_multiplier = sigma_inv @ drift_vector  # 形状 (dim,)
    
    # d) 将漂移乘数与时间结合
    # 我们需要将 t_list (num_timesteps,) 和 drift_multiplier (dim,) 结合
    # 得到一个 (num_timesteps, dim) 的矩阵。使用外积最合适。
    drift_correction = np.outer(t_list, drift_multiplier) # 形状 (num_timesteps, dim)

    # 5. 从回报项中减去漂移修正项
    # drift_correction (num_timesteps, dim) 会被广播到
    # transformed_returns (num_paths, num_timesteps, dim) 的形状
    Yt = transformed_returns - drift_correction
    
    return Yt

def pi_fraction_exact(t, Yt, T, alpha, r, sigma, 
                      joint_z_vectors, p_dist,
                      num_expectation_samples=10000, seed=None):
    """
    使用混合方法计算高维最优投资组合分数向量 (pi_t / X_t)。
    对 P 的期望是精确的，对 z 的期望是蒙特卡洛。

    参数:
        t (float): 当前时间点。
        Yt (ndarray): t时刻的底层多维布朗运动状态向量，形状为 (dim,)。
        T, alpha, r, sigma（波动率矩阵）: 模型参数。
        joint_z_vectors (ndarray): 预先生成的m个n维向量，形状为 (m, dim)。
        p_dist (ndarray): 对应每个z向量的概率，形状为 (m,)。
                        这应该是优化后的分布 Q*。
        num_expectation_samples (int): 用于近似 E_z[...] 的外层蒙特卡洛样本数。
        seed (int, optional): 随机种子。

    返回:
        ndarray: 最优投资组合分数向量，形状为 (dim,)。
    """
    if seed is not None:
        np.random.seed(seed)
        
    dim = Yt.shape[0]
    
    # 1. 外层蒙特卡洛：生成 z ~ N(0, (T-t)*I) 的样本
    mean_z_norm = np.zeros(dim)
    time_remaining = max(T - t, 0)
    cov_z_norm = time_remaining * np.eye(dim)
    z_norm_samples = np.random.multivariate_normal(mean_z_norm, cov_z_norm, size=num_expectation_samples)
    
    # 2. 构造 F 和 grad_F 的输入参数: y_final = z_norm + Yt
    y_final_samples = z_norm_samples + Yt
    
    # 3. 一次性计算所有 y_final_samples 的 F 和 grad_F 的值
    #    这里调用的是我们新的、基于精确求和的函数。
    F_vals = F_tilde_p_exact(
        t=T, y=y_final_samples, joint_z_vectors=joint_z_vectors, 
        p_dist=p_dist, r=r, sigma=sigma
    ) # 返回形状 (num_expectation_samples,)
    
    grad_F_vals = grad_F_tilde_p_exact(
        t=T, y=y_final_samples, joint_z_vectors=joint_z_vectors, 
        p_dist=p_dist, r=r, sigma=sigma
    ) # 返回形状 (num_expectation_samples, dim)

    # 4. 计算分子向量 (这部分逻辑不变)
    F_powered_num = F_vals ** (alpha / (1 - alpha))
    numerator_samples = grad_F_vals * F_powered_num[:, np.newaxis]
    numerator_vec = np.mean(numerator_samples, axis=0) # 形状 (dim,)
    
    # 5. 计算分母标量 (这部分逻辑不变)
    F_powered_den = F_vals ** (1 / (1 - alpha))
    denominator_scalar = (1 - alpha) * np.mean(F_powered_den)
    
    # 6. 组合最终结果
    if denominator_scalar == 0 or not np.isfinite(denominator_scalar):
        return np.zeros(dim)
        

    sigma_inv = np.linalg.inv(sigma)
    # cov_matrix_inv = sigma_inv.T @ sigma_inv
    
    pi_fraction = (sigma_inv @ numerator_vec) / denominator_scalar
    
    return pi_fraction

@njit
def utility_function_array(x_array, beta=-1):
    """
    Numba-friendly version: 1/beta * (x^beta)
    """
    if beta == 0:
        return np.log(x_array)
    else:
        return (1/beta) * (x_array ** beta)    
    
def I(y_array, beta=-1):
    """
    计算逆边际效用函数 I(y) = (u')⁻¹(y)。
    
    该函数是向量化的，可以直接处理 NumPy 数组。
    它基于 u(x) = x^β / β, 其导数 u'(x) = sign(x)|x|^(β-1)。

    参数:
        y_array (ndarray): 输入的数组或标量。
        beta (float): 效用函数的参数。

    返回:
        ndarray: 计算结果，与 y_array 形状相同。
    """
    # 为避免混淆，使用浮点数进行计算
    beta = float(beta)
    
    # 预先计算指数，这是关键部分
    exponent = 1.0 / (beta - 1.0)
    
    # 应用公式: I(y) = sign(y) * |y|^exponent
    # np.sign 和 np.abs 都是向量化函数
    return y_array ** exponent

def g(x_array, r, T, k=0.06, beta=-1):
    """
    计算 g(x) = I(k * exp(-r*T) / x)。

    此函数是向量化的，可以直接处理 NumPy 数组。

    参数:
        x_array (ndarray): 输入的数组或标量。
        k (float): 常数参数。
        r (float): 常数参数 (例如，无风险利率)。
        T (float): 常数参数 (例如，时间周期)。
        beta (float): 效用函数的参数。

    返回:
        ndarray: g(x) 的计算结果，与 x_array 形状相同。
    """
    # 1. 计算 I 函数的参数
    # k * exp(-rT) 是一个与 x 无关的常数
    constant_term = k * np.exp(-r * T)
    
    # 除以 x_array
    y_argument = constant_term / x_array
    
    # 2. 调用 I 函数
    return I(y_argument, beta)

def grad_I(y_array, beta=-1):
    """
    计算 I(y) 的导数, dI/dy。
    
    该函数是向量化的。
    I(y) = y^(1/(beta-1))
    dI/dy = (1/(beta-1)) * y^((1/(beta-1)) - 1)

    参数:
        y_array (ndarray): 输入的数组或标量。
        beta (float): 效用函数的参数。

    返回:
        ndarray: 导数的计算结果，与 y_array 形状相同。
    """
    # 确保使用浮点数进行计算
    beta = float(beta)
    
    # 检查 beta 是否为 1，以避免除以零
    if beta == 1.0:
        # 在这种情况下，效用函数是对数效用 u(x) = log(x)，
        # u'(x) = 1/x, I(y) = 1/y, I'(y) = -1/y^2
        return -1.0 / (y_array ** 2)

    # 计算指数
    exponent = 1.0 / (beta - 1.0)
    
    # 应用求导法则
    return exponent * (y_array ** (exponent - 1.0))

def grad_g(x_array, r, T, k=0.06, beta=-1):
    """
    计算 g(x) 的导数, dg/dx。
    g(x) = I(k * exp(-r*T) / x)

    使用链式法则: dg/dx = I'(y) * dy/dx, 其中 y = k * exp(-r*T) / x.
    dy/dx = -k * exp(-r*T) / x^2.

    此函数是向量化的。

    参数:
        x_array (ndarray): 输入的数组或标量。
        k (float): 常数参数。
        r (float): 常数参数 (例如，无风险利率)。
        T (float): 常数参数 (例如，时间周期)。
        beta (float): 效用函数的参数。

    返回:
        ndarray: g(x) 的导数计算结果，与 x_array 形状相同。
    """
    # 1. 计算 I 函数的参数 y(x)
    constant_term = k * np.exp(-r * T)
    y_argument = constant_term / x_array
    
    # 2. 计算 I'(y) 的值
    grad_I_value = grad_I(y_argument, beta)
    
    # 3. 计算 y(x) 对 x 的导数 dy/dx
    dy_dx = -constant_term / (x_array ** 2)
    
    # 4. 应用链式法则
    return grad_I_value * dy_dx

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

def compute_annualized_matrix(
    df: pd.DataFrame,
    start_month: str = "2015-01-01",
    end_month: str = "2024-12-31",
    sample_size: int = 20,
    permno_list: list = None,
    trading_days_per_year: int = 252,
):
    """
    Build a matrix of monthly adjusted-and-annualized series for sampled stocks using daily simple returns.

    Inputs
    - df: DataFrame with columns ['permno', 'date', 'ret'] where 'ret' is the daily simple return (e.g., 0.01 for 1%)
    - start_month, end_month: inclusive date strings 'YYYY-MM-DD'
    - sample_size: number of stocks to randomly sample among those with full monthly coverage
    - random_seed: RNG seed for reproducible sampling
    - trading_days_per_year: used to annualize daily mean return and daily variance

    Output
    - matrix_df: DataFrame shape (num_months x sample_size), index: Periods (M), columns: permnos
                 Values: trading_days_per_year * mean_daily_ret + 0.5 * previous_annual_sigma2
    - matrix: numpy array of shape (num_months, sample_size)
    - sampled_permnos: list of sampled permnos in column order
    - months: PeriodIndex of months from start_month to end_month (M)
    """
    # Basic cleaning and filtering
    df = df[['permno', 'date', 'ret']].copy()
    df = df.dropna(subset=['permno', 'date', 'ret'])
    df['date'] = pd.to_datetime(df['date'])
    df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
    df = df.dropna(subset=['ret'])
    df = df[np.isfinite(df['ret'])]
    df = df.drop_duplicates(subset=['permno', 'date'], keep='last')
    df = df.sort_values(['permno', 'date'])

    # Parse input months and boundaries
    start_dt = pd.to_datetime(start_month)
    end_dt = pd.to_datetime(end_month)

    # Include previous month for prev sigma^2
    prev_sigma_start_dt = (start_dt - relativedelta(months=1)).replace(day=1)

    # Filter daily data: from previous month start through end month end-date
    end_date_filter = pd.Period(end_dt, freq='M').end_time.normalize()
    df = df[(df['date'] >= prev_sigma_start_dt) & (df['date'] <= end_date_filter)]

    # Month key
    df['month'] = df['date'].dt.to_period('M')

    # Monthly mean and variance of daily simple returns
    monthly_stats = (
        df.groupby(['permno', 'month'], as_index=False)
          .agg(mean_daily_ret=('ret', 'mean'),
               var_daily=('ret', lambda x: x.var(ddof=1)))
    )
    monthly_stats['sigma2_annual'] = monthly_stats['var_daily'] * trading_days_per_year

    # Previous month's annualized sigma^2
    monthly_stats['prev_sigma2_annual'] = (
        monthly_stats.groupby('permno', group_keys=False)['sigma2_annual'].shift(1)
    )

    # Target months (inclusive) as monthly Periods
    start_period = pd.Period(start_dt, freq='M')
    end_period = pd.Period(end_dt, freq='M')
    months = pd.period_range(start=start_period, end=end_period, freq='M')

    panel = monthly_stats[monthly_stats['month'].isin(months)].copy()



    sampled_permnos = permno_list

    # Annualized series using daily returns:
    # annualized_value_t = trading_days_per_year * mean_daily_ret_t + 0.5 * prev_sigma2_annual_{t}
    selected = panel[panel['permno'].isin(sampled_permnos)].copy()
    selected['annualized'] = (
        trading_days_per_year * selected['mean_daily_ret'] + 0.5 * selected['prev_sigma2_annual']
    )

    # Pivot to (num_months x sample_size) matrix in the sampled permno order
    matrix_df = (
        selected.pivot(index='month', columns='permno', values='annualized')
                .loc[months, sampled_permnos]
    )
    matrix = matrix_df.to_numpy()

    return matrix_df, matrix, sampled_permnos, months


def james_stein_mean(ret_matrix):
    """
    Calculates the James-Stein shrinkage estimator for the mean of asset returns.

    Parameters:
    ret_matrix (np.ndarray): A T x p matrix of asset returns,
                             where T is the number of time periods and
                             p is the number of assets.

    Returns:
    np.ndarray: A p-dimensional vector of shrunk mean returns.
    """
    p = ret_matrix.shape[1]
    T = ret_matrix.shape[0]

    if p <= 2:
        # James-Stein estimator is not applicable for p <= 2, returns sample mean.
        return ret_matrix.mean(axis=0)

    # Sample mean of each asset
    sample_mean = ret_matrix.mean(axis=0)

    # Grand mean (mean of sample means)
    grand_mean = sample_mean.mean()

    # Sum of squared differences from the grand mean
    ssq_diff = np.sum((sample_mean - grand_mean)**2)
    
    if ssq_diff == 0:
        # If all sample means are the same, no shrinkage is needed.
        return sample_mean

    # Shrinkage intensity
    # The (p-2)/T term comes from theory.
    # We are shrinking the sample means towards the grand mean.
    delta = (p - 2) / T / ssq_diff
    
    # Cap shrinkage intensity at 1
    delta = min(1.0, delta)

    # James-Stein shrunk mean
    shrunk_mean = (1 - delta) * sample_mean + delta * grand_mean

    return shrunk_mean



def solve_k_with_EL(
    z_matrix: np.ndarray,
    r: float = 0.06,
    sigma: np.ndarray | None = None,
    T: float = 1.0,
    beta: float = -3.0,
    num_y: int = 500,
    seed: int = 42,
):
    """
    - z_matrix: shape (120, 20) monthly vectors (each row is one 20-d vector)
    - r: risk-free rate (annualized)
    - sigma: volatility matrix used in L_t (defaults to identity if None)
    - T: time horizon
    - beta: utility parameter used by I
    - num_y: number of N(0, I) samples for y
    - seed: RNG seed
    Returns: (k_opt, objective_at_k, EL_per_z, y_samples)
    """
    n_z, dim = z_matrix.shape
    if dim != 20:
        raise ValueError(f"Expected 20-dim z vectors; got {dim}")

    if sigma is None:
        sigma = np.eye(dim)

    rng = np.random.default_rng(seed)
    y_samples = rng.standard_normal((num_y, dim))

    # Compute EL_i = E_y[L_t(1, z_i, y)]
    # L_t_vectorized supports broadcasting: z -> (n_z, d), y -> (num_y, d) returns (n_z, num_y)
    L_vals = L_t_vectorized(t=T, z=z_matrix, y=y_samples, r=r, sigma=sigma)  # (n_z, num_y)
    EL_per_z = L_vals.mean(axis=0)  # (num_y,)

    # Objective: mse(mean_i I(k*exp(-rT)/EL_i) - exp(rT))
    target = np.exp(r * T)

    # Analytic initial guess (if beta != 1): mean I(k*c/EL_i) == target --> solve for k
    # I(x) = x^(1/(beta-1)) => let p = 1/(beta-1):
    # mean((k*c/EL_i)^p) = target  =>  k^p * c^p * mean(EL_i^-p) = target
    # k = [target / (c^p * mean(EL_i^-p))]^(1/p), c = exp(-rT)
    p = 1.0 / (beta - 1.0) if beta != 1.0 else None
    c = np.exp(-r * T)
    if p is not None:
        mean_EL_neg_p = np.mean(EL_per_z ** (-p))
        k_guess = (target / (c**p * mean_EL_neg_p)) ** (1.0 / p)
        # guardrails
        if not np.isfinite(k_guess) or k_guess <= 0:
            k_guess = 1.0
    else:
        k_guess = 1.0

    def objective(k: float) -> float:
        vals = I((k * c) / EL_per_z, beta=beta)
        return (np.mean(vals) - target) ** 2

    # Optimize k >= 0 using bounded scalar minimization
    res = minimize_scalar(objective, bounds=(1e-10, 1e10), method='bounded', options={'xatol': 1e-10, 'maxiter': 1000})
    k_opt = res.x
    return k_opt


def compute_annualized_matrix_new(
    df: pd.DataFrame,
    start_year: str = "2015",
    end_year: str = "2024",
    sample_size: int = 20,
    permno_list: list = None,
    trading_days_per_year: int = 252,
):
    """
    Build a matrix of annual adjusted-and-annualized series for sampled stocks using daily simple returns.
    Uses annual aggregation for more robust estimation.

    Inputs
    - df: DataFrame with columns ['permno', 'date', 'ret'] where 'ret' is the daily simple return (e.g., 0.01 for 1%)
    - start_year, end_year: inclusive year strings (e.g., '2015', '2024')
    - sample_size: number of stocks to randomly sample among those with full annual coverage
    - permno_list: list of specific permnos to use (if provided, sample_size is ignored)
    - trading_days_per_year: used to annualize daily mean return and daily variance

    Output
    - matrix_df: DataFrame shape (num_years x sample_size), index: years, columns: permnos
                 Values: trading_days_per_year * annual_mean_daily_ret + 0.5 * previous_annual_sigma2
    - matrix: numpy array of shape (num_years, sample_size)
    - sampled_permnos: list of sampled permnos in column order
    - years: list of years from start_year to end_year
    """
    # Basic cleaning and filtering
    df = df[['permno', 'date', 'ret']].copy()
    df = df.dropna(subset=['permno', 'date', 'ret'])
    df['date'] = pd.to_datetime(df['date'])
    df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
    df = df.dropna(subset=['ret'])
    df = df[np.isfinite(df['ret'])]
    df = df.drop_duplicates(subset=['permno', 'date'], keep='last')
    df = df.sort_values(['permno', 'date'])

    # Parse input years and boundaries
    start_year_int = int(start_year)
    end_year_int = int(end_year)

    # Include previous year for prev sigma^2
    prev_year_start = start_year_int - 1

    # Filter daily data: from previous year through end year
    start_date_filter = pd.to_datetime(f'{prev_year_start}-01-01')
    end_date_filter = pd.to_datetime(f'{end_year_int}-12-31')
    df = df[(df['date'] >= start_date_filter) & (df['date'] <= end_date_filter)]

    # Year key
    df['year'] = df['date'].dt.year

    # Annual mean and variance of daily simple returns
    annual_stats = (
        df.groupby(['permno', 'year'], as_index=False)
          .agg(mean_daily_ret=('ret', 'mean'),
               var_daily=('ret', lambda x: x.var(ddof=1)))
    )
    annual_stats['sigma2_annual'] = annual_stats['var_daily'] * trading_days_per_year

    # Previous year's annualized sigma^2
    annual_stats['prev_sigma2_annual'] = (
        annual_stats.groupby('permno', group_keys=False)['sigma2_annual'].shift(1)
    )

    # Target years (inclusive)
    years = list(range(start_year_int, end_year_int + 1))

    panel = annual_stats[annual_stats['year'].isin(years)].copy()

    sampled_permnos = permno_list

    # Annualized series using daily returns:
    # annualized_value_t = trading_days_per_year * mean_daily_ret_t + 0.5 * prev_sigma2_annual_{t}
    selected = panel[panel['permno'].isin(sampled_permnos)].copy()
    selected['annualized'] = (
        trading_days_per_year * selected['mean_daily_ret'] + 0.5 * selected['prev_sigma2_annual']
    )

    # Pivot to (num_years x sample_size) matrix in the sampled permno order
    matrix_df = (
        selected.pivot(index='year', columns='permno', values='annualized')
                .loc[years, sampled_permnos]
    )
    matrix = matrix_df.to_numpy()

    return matrix_df, matrix, sampled_permnos, years

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

def compute_annualized_matrix_type(
    df: pd.DataFrame,
    sigma: np.ndarray,
    permno_list: list = None,
    trading_days_per_year: int = 252,
    dt: float = 1/2520,
):
    assert 'type' in df.columns, "df must have a 'type' column"
    if permno_list is not None:
        df = df[df['permno'].isin(permno_list)]
    df = df.sort_values(['permno', 'date'])
    df['date'] = pd.to_datetime(df['date'])
    df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
    mean_type_logret = df.groupby(['type','permno'])['log_ret'].mean()
    # make it a matrix of shape (num_types, num_permnos)
    matrix = mean_type_logret.unstack().to_numpy()*1/dt
    matrix += (np.diag(sigma@sigma.T))/2
    n = len(df[(df['type']==1)&(df['permno']==df['permno'].iloc[0])])
    return matrix, n
    
    
