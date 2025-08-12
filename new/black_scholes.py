# ibx_pricing/black_scholes.py
from __future__ import annotations
from dataclasses import dataclass
from math import log, sqrt, exp, erf, isfinite, pi
from typing import Literal, Optional, Tuple
import numpy as np
from numpy.typing import ArrayLike

Right = Literal["C","P"]

def _norm_cdf(x):
    # 0.5 * (1 + erf(x / sqrt(2)))
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def _norm_pdf(x):
    return (1.0 / sqrt(2.0*pi)) * np.exp(-0.5 * x * x)

@dataclass(frozen=True)
class BSInputs:
    S: float
    K: float
    T: float
    r: float
    q: float
    sigma: float
    right: Right

def _d1_d2(S: ArrayLike, K: ArrayLike, r: ArrayLike, q: ArrayLike, sigma: ArrayLike, T: ArrayLike):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    r = np.asarray(r, dtype=float)
    q = np.asarray(q, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    T = np.asarray(T, dtype=float)
    vol_sqrt_T = sigma * np.sqrt(np.maximum(T, 0.0))
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / np.where(vol_sqrt_T==0, np.nan, vol_sqrt_T)
        d2 = d1 - vol_sqrt_T
    return d1, d2

def price(S: ArrayLike, K: ArrayLike, r: ArrayLike, q: ArrayLike, sigma: ArrayLike, T: ArrayLike, right: Right):
    d1, d2 = _d1_d2(S,K,r,q,sigma,T)
    df_r = np.exp(-np.asarray(r)*np.asarray(T))
    df_q = np.exp(-np.asarray(q)*np.asarray(T))
    if right == "C":
        return df_q * np.asarray(S) * _norm_cdf(d1) - df_r * np.asarray(K) * _norm_cdf(d2)
    else:
        return df_r * np.asarray(K) * _norm_cdf(-d2) - df_q * np.asarray(S) * _norm_cdf(-d1)

def delta(S,K,r,q,sigma,T,right: Right):
    d1, _ = _d1_d2(S,K,r,q,sigma,T)
    df_q = np.exp(-np.asarray(q)*np.asarray(T))
    if right == "C":
        return df_q * _norm_cdf(d1)
    else:
        return -df_q * _norm_cdf(-d1)

def gamma(S,K,r,q,sigma,T):
    d1, _ = _d1_d2(S,K,r,q,sigma,T)
    df_q = np.exp(-np.asarray(q)*np.asarray(T))
    denom = (np.asarray(S)*np.asarray(sigma)*np.sqrt(np.asarray(T)))
    return df_q * _norm_pdf(d1) / denom

def vega(S,K,r,q,sigma,T):
    d1, _ = _d1_d2(S,K,r,q,sigma,T)
    df_q = np.exp(-np.asarray(q)*np.asarray(T))
    return df_q * np.asarray(S) * _norm_pdf(d1) * np.sqrt(np.asarray(T))

def theta(S,K,r,q,sigma,T,right: Right):
    d1, d2 = _d1_d2(S,K,r,q,sigma,T)
    df_r = np.exp(-np.asarray(r)*np.asarray(T))
    df_q = np.exp(-np.asarray(q)*np.asarray(T))
    term1 = - df_q * np.asarray(S) * _norm_pdf(d1) * (np.asarray(sigma)/(2*np.sqrt(np.asarray(T))))
    if right == "C":
        term2 = q * df_q * np.asarray(S) * _norm_cdf(d1)
        term3 = - r * df_r * np.asarray(K) * _norm_cdf(d2)
    else:
        term2 = q * df_q * np.asarray(S) * _norm_cdf(-d1)
        term3 = - r * df_r * np.asarray(K) * _norm_cdf(-d2)
    return term1 - term2 + term3

def rho(S,K,r,q,sigma,T,right: Right):
    _, d2 = _d1_d2(S,K,r,q,sigma,T)
    df_r = np.exp(-np.asarray(r)*np.asarray(T))
    if right == "C":
        return np.asarray(T) * df_r * np.asarray(K) * _norm_cdf(d2)
    else:
        return -np.asarray(T) * df_r * np.asarray(K) * _norm_cdf(-d2)

def implied_vol_from_price(target_price: float, S: float, K: float, r: float, q: float, T: float, right: Right,
                           tol: float = 1e-7, max_iter: int = 100, bracket: Tuple[float,float] = (1e-6, 5.0)) -> Optional[float]:
    """Find sigma such that BS price matches target_price.
    Uses bracketed Newton with fallback bisection; robust for tiny vega."""
    low, high = bracket
    def f(sig):
        return float(price(S,K,r,q,sig,T,right) - target_price)
    f_low, f_high = f(low), f(high)
    if np.isnan(f_low) or np.isnan(f_high):
        return None
    if f_low * f_high > 0:
        # expand high if needed
        for h in (10.0, 20.0, 50.0):
            f_high = f(h)
            if isfinite(f_high) and f_low * f_high <= 0:
                high = h
                break
        else:
            return None

    sigma = 0.5 * (low + high)
    for _ in range(max_iter):
        v = float(vega(S,K,r,q,sigma,T))
        if v > 1e-12 and isfinite(v):
            p = float(price(S,K,r,q,sigma,T,right))
            step = (p - target_price) / v
            cand = sigma - step
            if low < cand < high:
                sigma = cand
        diff = f(sigma)
        if abs(diff) < tol:
            return float(max(sigma, 0.0))
        if diff > 0:
            high = sigma
        else:
            low = sigma
        sigma = 0.5 * (low + high)
    return float(max(sigma, 0.0))