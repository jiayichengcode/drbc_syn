import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import warnings
from numba import njit
import os
from helper import *
warnings.filterwarnings("ignore")

def grad_b_LT_vectorized(t, z, y, r, sigma):
    """
    计算 L_T(z, y) 关于其第一个参数 z (在公式中为 b) 的梯度, 即 ∇_b L_T。
    
    此版本经过修正，不再假设波动率矩阵 sigma 是对称的。
    推导公式: ∇_z L_t = L_t * ( y @ (sigma_inv).T - t * A @ sigma_inv )
    其中 A = (z - r) @ sigma_inv。
    
    - z: 形状为 (n, dim) 的数组
    - y: 形状为 (m, dim) 的数组
    
    返回: 
        一个形状为 (n, m, dim) 的张量。
    """
    # 确保输入至少是二维数组
    z = np.atleast_2d(z)
    y = np.atleast_2d(y)
    n_samples_z, dim = z.shape
    n_samples_y = y.shape[0]

    if t == 0.0:
        return np.zeros((n_samples_z, n_samples_y, dim))

    # --- 预计算 ---
    # 计算波动率矩阵的逆
    sigma_inv = np.linalg.inv(sigma)
    # 计算逆的转置，因为 sigma 不对称，所以 sigma_inv.T != sigma_inv
    sigma_inv_T = sigma_inv.T
    
    # 计算中间变量 A = (z - r) @ sigma_inv
    A = (z - r) @ sigma_inv  # 形状 (n, dim)
    
    # --- 核心梯度计算 ---
    # 1. 计算梯度公式中的 L_t(z, y) 部分
    l_values = L_t_vectorized(t, z, y, r, sigma) # 形状 (n, m)

    # 2. 分别计算梯度指数部分的两个项
    # 第一项: y @ (sigma_inv).T
    # y (m, dim) @ sigma_inv_T (dim, dim) -> term1 (m, dim)
    term1 = y @ sigma_inv_T
    
    # 第二项: t * A @ sigma_inv
    # A (n, dim) @ sigma_inv (dim, dim) -> term2 (n, dim)
    term2 = (t * A) @ sigma_inv

    # 3. 将两项合并: term1 - term2
    # 为了相减，需要利用广播机制
    # term1 (m, dim) -> (1, m, dim)
    # term2 (n, dim) -> (n, 1, dim)
    # 结果 grad_exponent_factor 的形状为 (n, m, dim)
    grad_exponent_factor = term1.reshape(1, n_samples_y, dim) - term2.reshape(n_samples_z, 1, dim)

    # 4. 最后，将 L_t 的值作为缩放因子乘到每个梯度向量上
    l_values_reshaped = l_values.reshape(n_samples_z, n_samples_y, 1)
    final_grad = l_values_reshaped * grad_exponent_factor
    
    return final_grad


def calculate_numerator(T, r, sigma, B_support, p_dist, m0=500, k=0.4, beta=-1, seed=None):
    """
    计算图片中公式所示的值。此函数本身不变，但它现在调用的是修正后的梯度函数。

    参数:
        T, r, sigma, beta, k: 模型和效用函数相关的参数。
        B_support (ndarray): 先验分布 B 的支撑集 (即 B_j 向量)，形状为 (n, dim)。
        p_dist (ndarray): 先验分布 B 中每个向量对应的概率 (P_n)，形状为 (n,)。
        m0 (int): 从正态分布中抽取的 Y 样本的数量。
        seed (int, optional): 用于生成随机数的种子，以保证结果可复现。
    """
    if seed is not None:
        np.random.seed(seed)
        
    n, dim = B_support.shape
    
    # 步骤 1: 生成 Y 的样本
    mean_y = np.zeros(dim)
    cov_y = T * np.eye(dim)
    Y_samples = np.random.multivariate_normal(mean_y, cov_y, size=m0)

    # 步骤 2: 计算 g'( E_Pn[L_T(B, Y_i)] )
    F_vals = F_tilde_p_exact(T, Y_samples, B_support, p_dist, r, sigma)
    g_prime_vals = grad_g(F_vals, r, T, k, beta)
    
    # 步骤 3: 计算梯度 (调用修正后的新函数)
    grad_L_tensor = grad_b_LT_vectorized(T, B_support, Y_samples, r, sigma)
    
    # 步骤 4: 计算L2范数内的项
    term_to_norm = g_prime_vals.reshape(1, m0, 1) * grad_L_tensor
    
    # 步骤 5: 计算L2范数的平方
    norm_squared_matrix = np.sum(term_to_norm**2, axis=2)
    
    # 步骤 6 & 7: 求和与归一化
    total_sum = np.sum(norm_squared_matrix)
    final_result = total_sum / (m0 * n)
    
    return final_result

def calculate_denominator(T, r, sigma, B_support, p_dist, beta=-1, k=0.4, m1=500, m2=500, seed=None):
    """
    根据图片中的公式计算第二个量。
    
    此版本使用了 np.einsum 来正确并高效地实现三重嵌套求和与点积的计算，
    修正了之前版本中错误的简化逻辑。

    参数:
        T, r, sigma, beta, k: 模型和效用函数相关的参数。
        B_support (ndarray): 先验分布 B 的支撑集 (B_k)，形状为 (n, dim)。
        p_dist (ndarray): 先验分布的概率 (P_n)，形状为 (n,)。
        m1 (int): 第一组 Y 样本的数量。
        m2 (int): 第二组 Y 样本的数量。
        seed (int, optional): 随机数种子，用于复现结果。

    返回:
        float: 公式计算出的标量值。
    """
    if seed is not None:
        np.random.seed(seed)
        
    n, dim = B_support.shape
    
    # --- 步骤 1 & 2 & 3: 与之前相同，预计算所有需要的 g' 和 ∇L ---
    mean_y = np.zeros(dim)
    cov_y = T * np.eye(dim)
    Y1_samples = np.random.multivariate_normal(mean_y, cov_y, size=m1) # (m1, dim)
    Y2_samples = np.random.multivariate_normal(mean_y, cov_y, size=m2) # (m2, dim)

    F1_vals = F_tilde_p_exact(T, Y1_samples, B_support, p_dist, r, sigma) # (m1,)
    g_prime1_vals = grad_g(F1_vals, r, T, k, beta) # (m1,)
    
    F2_vals = F_tilde_p_exact(T, Y2_samples, B_support, p_dist, r, sigma) # (m2,)
    g_prime2_vals = grad_g(F2_vals, r, T, k, beta) # (m2,)

    grad_L1_tensor = grad_b_LT_vectorized(T, B_support, Y1_samples, r, sigma) # (n, m1, dim)
    grad_L2_tensor = grad_b_LT_vectorized(T, B_support, Y2_samples, r, sigma) # (n, m2, dim)

    vectors_A = g_prime1_vals.reshape(1, m1, 1) * grad_L1_tensor # (n, m1, dim)
    vectors_B = g_prime2_vals.reshape(1, m2, 1) * grad_L2_tensor # (n, m2, dim)
    
    # --- 步骤 4: 使用 np.einsum 正确并高效地计算三重求和 ---
    # 这是此函数的关键修正点。
    # 我们要计算: Σ_i Σ_j Σ_k Σ_d (A_{kid} * B_{kjd})
    # np.einsum 的字符串 'kid,kjd->' 完美地表达了这个操作:
    # 'kid' -> 第一个操作数 vectors_A 的索引 (k=n, i=m1, d=dim)
    # 'kjd' -> 第二个操作数 vectors_B 的索引 (k=n, j=m2, d=dim)
    # '->'  -> 输出。因为右边是空的，意味着对所有出现在左边但没在右边的
    #          索引 (k, i, j, d) 进行求和，最终得到一个标量。
    total_sum = np.einsum('kid,kjd->', vectors_A, vectors_B)
    
    # --- 步骤 5: 应用最终的归一化因子 ---
    final_result = total_sum / (n * m1 * m2)
    
    return final_result**2

def cov_p_vectorized(l1_values_vs_B, l2_values_vs_B, p_dist):
    """
    直接且向量化地计算 L_T(B,Y₁) 和 L_T(B,Y₂) 在概率测度 P* 下的协方差。

    该函数接收两组 L_T 的值，每一组都对应于多个蒙特卡洛样本。

    参数:
        l1_values_vs_B (ndarray): L_T(B, Y₁) 的值。形状为 (n, m)，
                                  n 是先验 B 的支撑点数量, 
                                  m 是蒙特卡洛样本数量。
        l2_values_vs_B (ndarray): L_T(B, Y₂) 的值。形状为 (n, m)。
        p_dist (ndarray): 先验分布 B 的概率，形状为 (n,)。

    返回:
        ndarray: 一个包含 m 个协方差标量的向量，形状为 (m,)。
    """
    # 验证 p_dist 是一个合法的概率分布
    # np.testing.assert_almost_equal(np.sum(p_dist), 1.0)
    
    # 步骤 1: 计算 E_P*[L₁] 和 E_P*[L₂]
    # p_dist (n,) @ l1_values_vs_B (n, m) -> 结果 (m,)
    # 结果向量的第 i 个元素是 E_P*[L_T(B, Y₁ᵢ)]
    e_l1 = p_dist @ l1_values_vs_B
    e_l2 = p_dist @ l2_values_vs_B
    
    # 步骤 2: 计算 E_P*[L₁ * L₂]
    # 首先，计算 L₁ 和 L₂ 对应元素的乘积
    l1_l2_product = l1_values_vs_B * l2_values_vs_B # 形状 (n, m)
    # 然后，计算这个乘积在 P* 下的期望
    e_l1l2 = p_dist @ l1_l2_product # 形状 (m,)
    
    # 步骤 3: 应用协方差公式
    covariances = e_l1l2 - e_l1 * e_l2
    
    return covariances

def calculate_z_var(
    T, r, sigma,  
    B_support, p_dist, 
    num_mc_samples=500, beta=-1, k=0.4, seed=None
):
    """
    根据图片公式计算期望值（重构版）。
    此版本调用一个新的辅助函数来直接计算协方差，使逻辑更清晰。

    参数:
        (同之前版本)
    """
    if seed is not None:
        np.random.seed(seed)
        
    n, dim = B_support.shape
    
    # --- 步骤 1: 生成蒙特卡洛样本 ---
    mean_y = np.zeros(dim)
    cov_y = T * np.eye(dim)
    Y1_samples = np.random.multivariate_normal(mean_y, cov_y, size=num_mc_samples) # (m, dim)
    Y2_samples = np.random.multivariate_normal(mean_y, cov_y, size=num_mc_samples) # (m, dim)
    
    # --- 步骤 2: 为所有样本计算 L_T(B_k, Y_i) 的值 ---
    l1_tensor = L_t_vectorized(T, B_support, Y1_samples, r, sigma) # 形状 (n, m)
    l2_tensor = L_t_vectorized(T, B_support, Y2_samples, r, sigma) # 形状 (n, m)
    
    # --- 步骤 3: 直接调用新函数计算所有样本的协方差 ---
    covariances = cov_p_vectorized(l1_tensor, l2_tensor, p_dist) # 形状 (m,)
    
    # --- 步骤 4: 计算 g'(α(Y)) 的值 ---
    # α(Y) = E_P*[L(B, Y)]
    alpha_y1 = p_dist @ l1_tensor # 形状 (m,)
    alpha_y2 = p_dist @ l2_tensor # 形状 (m,)
    g_prime1_values = grad_g(alpha_y1, r, T, k, beta) # 形状 (m,)
    g_prime2_values = grad_g(alpha_y2, r, T, k, beta) # 形状 (m,)
    
    # --- 步骤 5: 计算括号内的完整表达式 ---
    integrand_values = g_prime1_values * g_prime2_values * covariances # 形状 (m,)
    
    # --- 步骤 6: 计算期望 E[...] ---
    final_result = np.mean(integrand_values)
    
    return final_result

def compute_big_delta_star(B, r, T, alpha, delta, sigma_matrix, m=200, rng=None):
    """
    Compute the empirical-shift direction Delta^*(B) using Monte Carlo over Y ~ N(0, T I).

    Parameters
    ----------
    B : array-like, shape (n, d)
        Input samples representing the empirical law P0 = Pn.
    r : float
        Risk-free rate.
    T : float
        Time horizon.
    alpha : float
        Robustness parameter.
    delta : float
        Ambiguity radius used in Delta*.
    sigma_matrix : array-like, shape (d, d)
        Invertible diffusion matrix.
    m : int, optional (default=200)
        Number of Monte Carlo draws for Y.
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility. If None, a new Generator() is used.

    Returns
    -------
    Delta : ndarray, shape (n, d)
        The Delta^*(B) shifts for each row of B.
    """
    B = np.asarray(B, dtype=np.float64)
    if B.ndim != 2:
        raise ValueError("B must be a 2D array of shape (n, d)")
    n, d = B.shape

    if rng is None:
        rng = np.random.default_rng()

    sigma_matrix = np.asarray(sigma_matrix, dtype=np.float64)
    sigma_inv = np.linalg.inv(sigma_matrix)

    ones = np.ones(d, dtype=np.float64)
    expo = alpha / (1.0 - alpha)

    # Local stable log-sum-exp utilities
    def _logsumexp(a, axis=None):
        a = np.asarray(a, dtype=np.float64)
        amax = np.max(a, axis=axis, keepdims=True)
        out = amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))
        if axis is None:
            return out.squeeze()
        return np.squeeze(out, axis=axis)

    def _logmeanexp(a, axis=None):
        a = np.asarray(a, dtype=np.float64)
        lse = _logsumexp(a, axis=axis)
        if axis is None:
            N = a.size
        else:
            N = a.shape[axis]
        return lse - np.log(N)

    # Monte Carlo draws Y ~ N(0, T I)
    Y = rng.normal(loc=0.0, scale=np.sqrt(T), size=(m, d)).astype(np.float64)
    # make >95 percentile of Y to be 0
    Y = np.where(Y > 1.96*np.sqrt(T), 0, Y)
    # make < 5 percentile of Y to be 0
    Y = np.where(Y < -1.96*np.sqrt(T), 0, Y)
    # v_i = sigma^{-1} (B_i - r*1)
    V = (B - r * ones) @ sigma_inv.T                  # (n, d)
    V_norm2 = np.sum(V * V, axis=1)                   # (n,)

    # log L_T(B_i, Y_j)
    logL = V @ Y.T - 0.5 * (V_norm2[:, None]) * T     # (n, m)

    # M(Y_j) = ( E_{P0}[L_T(B, Y_j)] )^(alpha/(1-alpha))
    log_mean_L_over_i = _logmeanexp(logL, axis=0)     # (m,)
    M = np.exp(expo * log_mean_L_over_i)              # (m,)

    # H(B_i) computation via weights
    W = np.exp(logL) * (M[None, :]) / float(m)        # (n, m)
    WY = W @ Y                                        # (n, d)
    s = np.sum(W, axis=1)                             # (n,)
    S = WY - (T * s)[:, None] * V                     # (n, d)
    H = S @ sigma_inv                                 # (n, d)

    # Denominator: sqrt( E^{P0}[ ||H(B)||_2^2 ] )
    H_norm2 = np.sum(H * H, axis=1)
    denom = np.sqrt(np.mean(H_norm2))

    if denom == 0.0 or not np.isfinite(denom):
        Delta = np.zeros_like(B)
    else:
        Delta = np.sqrt(delta) * H / denom

    return Delta