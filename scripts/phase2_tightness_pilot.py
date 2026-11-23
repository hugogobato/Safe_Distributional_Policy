#!/usr/bin/env python3
"""
WP-2.2: Analytical & Toy 1D Non-Vacuity Tightness Pilot (LCT-LTT vs. Two-Arm OPRA/UnO + LTT)

Strictly respects 4 GiB RAM budget (OMP/OPENBLAS/MKL threads = 1, max_workers <= 2).
Implements:
  1. LCT-LTT-EB: Localized Comparative Tail-Risk LTT with Maurer-Pontil Empirical-Bernstein UCBs
     (exact breakpoint supremum over [eta_lo, eta_hi] for both capped CVaR and uncapped Q_beta).
  2. LCT-LTT-Betting: Localized Comparative Tail-Risk LTT with Waudby-Smith & Ramdas (2024)
     aGRAPA betting-martingale tests (deterministic-cell envelope for capped CVaR, exact breakpoints
     for uncapped Q_beta).
  3. LCT-LTT-OracleEta0-Betting (diagnostic for Risk R4: isolates cost of baseline-VaR CI width).
  4. B1-OPRA: Two-arm Huang et al. (NeurIPS 2021, Thm 5.1 Eq. 4 Bernstein CDF bound + tight
     envelope integration) + Fixed-Sequence LTT.
  5. B1-TwoArm-Betting: Oracle-tightened separate-arm betting-martingale bounds +
     Fixed-Sequence LTT, with oracle VaR for the capped-CVaR diagnostic.

Outputs:
  - results/phase2_pilot/tightness_check.csv
  - results/phase2_pilot/pilot_diagnostics.json
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import math
import time
import resource
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import beta as beta_dist, norm
from scipy.optimize import brentq


# =============================================================================
# 1. Analytical Ground Truth for Semicontinuous Log-Normal Hurdle DGP
# =============================================================================
# Baseline arm A=0 (pi_0 = 0):
#   P(L(0) = 0 | X) = 0.40
#   L(0) | L(0) > 0 ~ LogNormal(mu0=1.60, sigma0=0.85)
# Candidate arm A=1 (pi_hat = 1):
#   On safe region S* = [0, p_star]:
#     P(L(1) = 0 | X in S*) = 0.55
#     L(1) | L(1) > 0, X in S* ~ LogNormal(mu_safe=1.00, sigma_safe=0.70)
#   On unsafe region (p_star, 1]:
#     P(L(1) = 0 | X not in S*) = 0.35
#     L(1) | L(1) > 0, X not in S* ~ LogNormal(mu_unsafe=2.15, sigma_unsafe=1.00)
# Propensity: e(1 | x) = 0.25 + 0.50 * x in [0.25, 0.75], so kappa = 0.25.
# Oracle score: s*(x) = 1 - x, so S_lambda = [0, p_lam] with p_lam = 1 - lambda.

DGP_PARAMS = {
    "p0_zero": 0.40,
    "mu0": 1.60,
    "sigma0": 0.85,
    "p1_safe_zero": 0.55,
    "mu1_safe": 1.00,
    "sigma1_safe": 0.70,
    "p1_unsafe_zero": 0.35,
    "mu1_unsafe": 2.15,
    "sigma1_unsafe": 1.00,
    "kappa": 0.25,
}


def _ln_cdf(t, p_zero, mu, sigma):
    if t < 0.0:
        return 0.0
    if t == 0.0:
        return p_zero
    return p_zero + (1.0 - p_zero) * norm.cdf((math.log(t) - mu) / sigma)


def _ln_capped_excess(eta, B, p_zero, mu, sigma):
    """Exact E[(min(L, B) - eta)_+] for hurdle log-normal, 0 <= eta <= B."""
    if eta >= B:
        return 0.0
    if eta <= 0.0:
        # E[min(L, B)]
        z_B = (math.log(B) - mu) / sigma
        part1 = math.exp(mu + 0.5 * sigma * sigma) * norm.cdf(z_B - sigma)
        part2 = B * (1.0 - norm.cdf(z_B))
        return (1.0 - p_zero) * (part1 + part2)
    z_eta = (math.log(eta) - mu) / sigma
    z_B = (math.log(B) - mu) / sigma
    # E[L * 1{eta < L <= B}]
    term_L = math.exp(mu + 0.5 * sigma * sigma) * (norm.cdf(z_B - sigma) - norm.cdf(z_eta - sigma))
    # -eta * P(eta < L <= B)
    term_eta = -eta * (norm.cdf(z_B) - norm.cdf(z_eta))
    # (B - eta) * P(L > B)
    term_B = (B - eta) * (1.0 - norm.cdf(z_B))
    return (1.0 - p_zero) * (term_L + term_eta + term_B)


def mixture_weights(p_lam, p_star, null_mode=False):
    """Weights (w_base, w_safe, w_unsafe) for gated policy pi_lambda."""
    if null_mode:
        # Null boundary: treated arm is slightly worse everywhere (mu0 + 0.002)
        return (1.0 - p_lam, 0.0, p_lam)
    w_safe = min(p_lam, p_star)
    w_unsafe = max(0.0, p_lam - p_star)
    w_base = 1.0 - p_lam
    return (w_base, w_safe, w_unsafe)


def true_cdf(t, p_lam, p_star, cap_B=None, null_mode=False):
    if cap_B is not None and t >= cap_B:
        return 1.0
    w_b, w_s, w_u = mixture_weights(p_lam, p_star, null_mode=null_mode)
    c_b = _ln_cdf(t, DGP_PARAMS["p0_zero"], DGP_PARAMS["mu0"], DGP_PARAMS["sigma0"])
    if null_mode:
        c_u = _ln_cdf(t, DGP_PARAMS["p0_zero"], DGP_PARAMS["mu0"] + 0.002, DGP_PARAMS["sigma0"])
        return w_b * c_b + w_u * c_u
    c_s = _ln_cdf(t, DGP_PARAMS["p1_safe_zero"], DGP_PARAMS["mu1_safe"], DGP_PARAMS["sigma1_safe"])
    c_u = _ln_cdf(t, DGP_PARAMS["p1_unsafe_zero"], DGP_PARAMS["mu1_unsafe"], DGP_PARAMS["sigma1_unsafe"])
    return w_b * c_b + w_s * c_s + w_u * c_u


def true_quantile(beta, p_lam, p_star, cap_B=None, null_mode=False):
    if true_cdf(0.0, p_lam, p_star, cap_B=cap_B, null_mode=null_mode) >= beta:
        return 0.0
    hi = cap_B if cap_B is not None else 500.0
    while cap_B is None and true_cdf(hi, p_lam, p_star, cap_B=None, null_mode=null_mode) < beta:
        hi *= 2.0
    if cap_B is not None and true_cdf(cap_B - 1e-12, p_lam, p_star, cap_B=cap_B, null_mode=null_mode) < beta:
        return float(cap_B)
    return float(brentq(lambda t: true_cdf(t, p_lam, p_star, cap_B=cap_B, null_mode=null_mode) - beta, 1e-12, hi, xtol=1e-11))


def true_capped_cvar(beta, p_lam, p_star, cap_B, null_mode=False):
    eta = true_quantile(beta, p_lam, p_star, cap_B=cap_B, null_mode=null_mode)
    w_b, w_s, w_u = mixture_weights(p_lam, p_star, null_mode=null_mode)
    ex_b = _ln_capped_excess(eta, cap_B, DGP_PARAMS["p0_zero"], DGP_PARAMS["mu0"], DGP_PARAMS["sigma0"])
    if null_mode:
        ex_u = _ln_capped_excess(eta, cap_B, DGP_PARAMS["p0_zero"], DGP_PARAMS["mu0"] + 0.002, DGP_PARAMS["sigma0"])
        ex = w_b * ex_b + w_u * ex_u
    else:
        ex_s = _ln_capped_excess(eta, cap_B, DGP_PARAMS["p1_safe_zero"], DGP_PARAMS["mu1_safe"], DGP_PARAMS["sigma1_safe"])
        ex_u = _ln_capped_excess(eta, cap_B, DGP_PARAMS["p1_unsafe_zero"], DGP_PARAMS["mu1_unsafe"], DGP_PARAMS["sigma1_unsafe"])
        ex = w_b * ex_b + w_s * ex_s + w_u * ex_u
    return float(eta + ex / (1.0 - beta))


def generate_data(n, p_star, rng, null_mode=False):
    X = rng.uniform(0.0, 1.0, size=n)
    e1 = 0.25 + 0.50 * X
    A = (rng.uniform(0.0, 1.0, size=n) < e1).astype(np.int8)
    e_obs = np.where(A == 1, e1, 1.0 - e1)

    L = np.zeros(n, dtype=np.float64)
    # Arm 0
    m0 = A == 0
    n0 = int(m0.sum())
    if n0 > 0:
        pos0 = rng.uniform(0.0, 1.0, size=n0) >= DGP_PARAMS["p0_zero"]
        vals0 = np.zeros(n0)
        n0_pos = int(pos0.sum())
        if n0_pos > 0:
            vals0[pos0] = np.exp(rng.normal(DGP_PARAMS["mu0"], DGP_PARAMS["sigma0"], size=n0_pos))
        L[m0] = vals0

    # Arm 1
    if null_mode:
        m1 = A == 1
        n1 = int(m1.sum())
        if n1 > 0:
            pos1 = rng.uniform(0.0, 1.0, size=n1) >= DGP_PARAMS["p0_zero"]
            vals1 = np.zeros(n1)
            n1_pos = int(pos1.sum())
            if n1_pos > 0:
                vals1[pos1] = np.exp(rng.normal(DGP_PARAMS["mu0"] + 0.002, DGP_PARAMS["sigma0"], size=n1_pos))
            L[m1] = vals1
    else:
        m1s = (A == 1) & (X <= p_star)
        n1s = int(m1s.sum())
        if n1s > 0:
            pos = rng.uniform(0.0, 1.0, size=n1s) >= DGP_PARAMS["p1_safe_zero"]
            vals = np.zeros(n1s)
            np_ = int(pos.sum())
            if np_ > 0:
                vals[pos] = np.exp(rng.normal(DGP_PARAMS["mu1_safe"], DGP_PARAMS["sigma1_safe"], size=np_))
            L[m1s] = vals

        m1u = (A == 1) & (X > p_star)
        n1u = int(m1u.sum())
        if n1u > 0:
            pos = rng.uniform(0.0, 1.0, size=n1u) >= DGP_PARAMS["p1_unsafe_zero"]
            vals = np.zeros(n1u)
            np_ = int(pos.sum())
            if np_ > 0:
                vals[pos] = np.exp(rng.normal(DGP_PARAMS["mu1_unsafe"], DGP_PARAMS["sigma1_unsafe"], size=np_))
            L[m1u] = vals

    return X, A, e1, e_obs, L


# =============================================================================
# 2. Core Finite-Sample Confidence Bounds (Empirical-Bernstein & aGRAPA Betting)
# =============================================================================

def eb_ucb_from_moments(mean, var, n, range_width, delta):
    log_term = math.log(2.0 / delta)
    return mean + np.sqrt(2.0 * np.maximum(0.0, var) * log_term / n) + (7.0 * range_width * log_term) / (3.0 * (n - 1))


def betting_reject_upper(Z, a, b, tau, delta, c=0.75):
    """
    Exact finite-sample level-delta test of H0: E[Z] >= tau for Z in [a, b]
    using Waudby-Smith & Ramdas (2024) predictable aGRAPA supermartingale + Ville's inequality.
    Returns True iff H0 is rejected (certifying E[Z] < tau).
    """
    if b <= a:
        return bool(a <= tau)
    m0 = (tau - a) / (b - a)
    if m0 >= 1.0:
        return True
    if m0 <= 0.0:
        return False
    n = Z.shape[0]
    U = (Z - a) / (b - a)
    S1 = np.empty(n, dtype=np.float64)
    S2 = np.empty(n, dtype=np.float64)
    S1[0] = 0.0
    S2[0] = 0.0
    np.cumsum(U[:-1], out=S1[1:])
    np.cumsum(U[:-1] * U[:-1], out=S2[1:])
    idx = np.arange(1, n + 1, dtype=np.float64)
    mu_hat = (m0 + S1) / idx
    var_hat = np.maximum(1e-12, (0.0625 + S2 - 2.0 * mu_hat * S1 + (idx - 1.0) * mu_hat * mu_hat) / idx)
    diff = m0 - mu_hat
    lam = np.clip(diff / (var_hat + diff * diff), 0.0, c / (1.0 - m0))
    log_terms = np.log1p(lam * (m0 - U))
    max_log_w = float(np.max(np.cumsum(log_terms)))
    return max_log_w >= math.log(1.0 / delta)


def betting_reject_lower(Z, a, b, tau, delta, c=0.75):
    """Exact finite-sample level-delta test of H0: E[Z] <= tau (certifying E[Z] > tau)."""
    return betting_reject_upper(-Z, -b, -a, -tau, delta, c=c)


BET_FRACTIONS = np.array([0.005, 0.015, 0.03, 0.06, 0.10, 0.20, 0.40, 0.70, 0.95])


def monotone_betting_reject_upper(Z, a, b, tau, delta):
    """Fixed-time convex mixture of valid bets, increasing in tau and decreasing in Z."""
    if tau >= b:
        return True
    if tau <= a:
        return False
    # Zero-valued survival observations (or values at a) share one factor.
    scaled = (tau - Z[Z != a]) / (b - tau)
    n_at_a = Z.size - scaled.size
    log_wealth = np.log1p(BET_FRACTIONS[:, None] * scaled).sum(axis=1)
    log_wealth += n_at_a * np.log1p(BET_FRACTIONS * (tau - a) / (b - tau))
    return bool(logsumexp(log_wealth) >= math.log(len(BET_FRACTIONS) / delta))


def betting_ucb_bisect(Z, a, b, delta):
    """Upper confidence bound by monotone inversion, rounded outward."""
    lo, hi = float(a), float(b)
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if monotone_betting_reject_upper(Z, a, b, mid, delta):
            hi = mid
        else:
            lo = mid
    return hi


def betting_lcb_bisect(Z, a, b, delta):
    return -betting_ucb_bisect(-Z, -b, -a, delta)


# =============================================================================
# 3. Exact Baseline-VaR CI [eta_lo, eta_hi] via Lemma 1 (Survival Complement)
# =============================================================================

def baseline_var_ci_eb(L, w0, beta, delta1, kappa, cap_B=None):
    """
    Exact finite-sample (1 - delta1) CI [eta_lo, eta_hi] for Q_beta(L(pi_0)) via Lemma 1
    using survival indicators Z_i(t) = w0_i * 1{L_i > t} in [0, 1/kappa] and Empirical Bernstein.
    """
    n = L.shape[0]
    target_S = 1.0 - beta
    R = 1.0 / kappa
    sup_T = float(cap_B) if cap_B is not None else float("inf")

    mask = w0 > 0.0
    L0 = L[mask]
    w0_pos = w0[mask]
    if L0.size == 0:
        return 0.0, sup_T

    # Distinct observed baseline losses sorted ascending: v_1 < ... < v_K
    order = np.argsort(L0)
    Ls = L0[order]
    ws = w0_pos[order]
    # Group ties so indicators 1{L > v_k} are exact
    uniq_vals, first_idx = np.unique(Ls, return_index=True)
    # Candidate thresholds: v_0 = 0 (if 0 not in uniq_vals) plus uniq_vals
    # For each threshold v_k, sum_{L_i > v_k} w_i and sum_{L_i > v_k} w_i^2
    # Compute suffix sums from largest downward
    m = Ls.size
    suf_w = np.empty(m + 1, dtype=np.float64)
    suf_w2 = np.empty(m + 1, dtype=np.float64)
    suf_w[m] = 0.0
    suf_w2[m] = 0.0
    suf_w[:m] = np.cumsum(ws[::-1])[::-1]
    suf_w2[:m] = np.cumsum((ws * ws)[::-1])[::-1]

    # Index of first element strictly greater than uniq_vals[k]
    after_idx = np.empty_like(first_idx)
    after_idx[:-1] = first_idx[1:]
    after_idx[-1] = m

    # Build grid v_grid starting at 0.0
    if uniq_vals[0] > 0.0:
        v_grid = np.concatenate(([0.0], uniq_vals))
        s_w = np.concatenate(([suf_w[0]], suf_w[after_idx]))
        s_w2 = np.concatenate(([suf_w2[0]], suf_w2[after_idx]))
    else:
        v_grid = uniq_vals
        s_w = suf_w[after_idx]
        s_w2 = suf_w2[after_idx]

    mean_S = s_w / n
    var_S = np.maximum(0.0, (s_w2 - n * mean_S * mean_S) / (n - 1))
    rad = eb_ucb_from_moments(0.0, var_S, n, R, delta1 / 2.0)
    S_hi = mean_S + rad
    S_lo = mean_S - rad

    # eta_lo = inf{v_k : S_lo(v_k) <= 1 - beta}
    idx_lo = np.where(S_lo <= target_S)[0]
    eta_lo = float(v_grid[idx_lo[0]]) if idx_lo.size > 0 else sup_T

    # eta_hi = smallest v_k such that S_hi(v_j) <= 1 - beta for ALL j >= k
    # Equivalently, find largest j with S_hi(v_j) > 1 - beta; then eta_hi = v_{j+1} (or sup_T if j = last)
    bad = np.where(S_hi > target_S)[0]
    if bad.size == 0:
        eta_hi = float(v_grid[0])
    else:
        last_bad = int(bad[-1])
        if last_bad + 1 < v_grid.size:
            eta_hi = float(v_grid[last_bad + 1])
        else:
            eta_hi = sup_T

    if cap_B is not None:
        eta_lo = min(eta_lo, float(cap_B))
        eta_hi = min(eta_hi, float(cap_B))
    return eta_lo, eta_hi


def baseline_var_ci_betting(L, w0, beta, delta1, kappa, cap_B=None):
    """Invert monotone fixed-stake tests at the exact survival breakpoints."""
    target_S = 1.0 - beta
    b_max = 1.0 / kappa
    support_end = float(cap_B) if cap_B is not None else float("inf")
    v_grid = np.unique(np.concatenate(([0.0], L[w0 > 0.0])))

    def endpoint(lower):
        lo, hi = 0, v_grid.size
        while lo < hi:
            mid = (lo + hi) // 2
            z = np.where(L > v_grid[mid], w0, 0.0)
            if lower:
                crossed = not monotone_betting_reject_upper(
                    -z, -b_max, 0.0, -target_S, delta1 / 2.0
                )
            else:
                crossed = monotone_betting_reject_upper(
                    z, 0.0, b_max, target_S, delta1 / 2.0
                )
            if crossed:
                hi = mid
            else:
                lo = mid + 1
        return float(v_grid[lo]) if lo < v_grid.size else support_end

    return endpoint(lower=True), endpoint(lower=False)


# =============================================================================
# 4. Localized Comparative Tests (CVaR capped & Q_beta uncapped)
# =============================================================================

def lct_cvar_eb_certifies(L, dw, eta_lo, eta_hi, beta, delta2, kappa, cap_B, eps=0.0):
    """Exact breakpoint supremum over [eta_lo, eta_hi] for Empirical-Bernstein CVaR UCB."""
    n = L.shape[0]
    active = dw != 0.0
    La = L[active]
    dwa = dw[active]
    tau = (1.0 - beta) * eps

    # Breakpoints: {eta_lo, eta_hi} union ({La} intersect (eta_lo, eta_hi))
    in_band = La[(La > eta_lo) & (La < eta_hi)]
    if in_band.size > 0:
        bpts = np.unique(np.concatenate(([eta_lo, eta_hi], in_band)))
    else:
        bpts = np.array([eta_lo, eta_hi], dtype=np.float64)

    # Vectorized exact moments across all breakpoints via descending sort
    order = np.argsort(La)[::-1]
    Ls = La[order]
    ds = dwa[order]
    d2s = ds * ds

    # Prefix sums of d, d*L, d^2, d^2*L, d^2*L^2 for Ls > eta
    c_d = np.concatenate(([0.0], np.cumsum(ds)))
    c_dL = np.concatenate(([0.0], np.cumsum(ds * Ls)))
    c_d2 = np.concatenate(([0.0], np.cumsum(d2s)))
    c_d2L = np.concatenate(([0.0], np.cumsum(d2s * Ls)))
    c_d2L2 = np.concatenate(([0.0], np.cumsum(d2s * Ls * Ls)))

    # Number of active elements with Ls > eta
    # Since Ls is descending, -Ls is ascending; searchsorted(-Ls, -eta, side='left') gives count > eta
    cnt = np.searchsorted(-Ls, -bpts, side="left")
    sum_Z = c_dL[cnt] - bpts * c_d[cnt]
    sum_Z2 = c_d2L2[cnt] - 2.0 * bpts * c_d2L[cnt] + (bpts * bpts) * c_d2[cnt]

    mean_Z = sum_Z / n
    var_Z = np.maximum(0.0, (sum_Z2 - n * mean_Z * mean_Z) / (n - 1))
    R = 2.0 * (cap_B - bpts) / kappa
    ucbs = eb_ucb_from_moments(mean_Z, var_Z, n, R, delta2)
    return bool(np.max(ucbs) <= tau)


def lct_cvar_betting_certifies(L, dw, eta_lo, eta_hi, beta, delta2, kappa, cap_B, eps=0.0, n_cells=250):
    """
    Exact finite-sample Betting-Martingale CVaR test via deterministic cell-envelope grid
    u_0 < u_1 < ... < u_G on [0, cap_B] (zero discretization error, zero union bound over G).
    """
    tau = (1.0 - beta) * eps
    u_grid = np.linspace(0.0, cap_B, n_cells + 1)
    # Cells [u_{g-1}, u_g] intersecting [eta_lo, eta_hi]
    g_start = min(n_cells, int(np.searchsorted(u_grid, eta_lo, side="right")))
    g_end = min(n_cells, int(np.searchsorted(u_grid, eta_hi, side="right")))

    dw_pos = np.maximum(dw, 0.0)
    dw_neg = np.maximum(-dw, 0.0)

    # Order cells by descending empirical mean so failures exit on 1st check
    cells = list(range(g_start, g_end + 1))
    if len(cells) > 1:
        means = []
        for g in cells:
            u_lo, u_hi = u_grid[g - 1], u_grid[g]
            Z = dw_pos * np.maximum(0.0, L - u_lo) - dw_neg * np.maximum(0.0, L - u_hi)
            means.append(float(np.mean(Z)))
        cells = [c for _, c in sorted(zip(means, cells), reverse=True)]

    for g in cells:
        u_lo, u_hi = u_grid[g - 1], u_grid[g]
        Z = dw_pos * np.maximum(0.0, L - u_lo) - dw_neg * np.maximum(0.0, L - u_hi)
        a = -(cap_B - u_hi) / kappa
        b = (cap_B - u_lo) / kappa
        if not betting_reject_upper(Z, a, b, tau, delta2):
            return False
    return True


def lct_cvar_betting_oracle_eta_certifies(L, dw, eta0_true, beta, delta, kappa, cap_B, eps=0.0):
    """Diagnostic (Risk R4): Localized Betting CVaR using known oracle eta0* (no CI width cost)."""
    Z = dw * np.maximum(0.0, L - eta0_true)
    bound = (cap_B - eta0_true) / kappa
    return betting_reject_upper(Z, -bound, bound, (1.0 - beta) * eps, delta)


def lct_q_eb_certifies(L, dw, eta_lo, eta_hi, delta2, kappa):
    """Exact piecewise-constant supremum over [eta_lo, eta_hi] for uncapped Q_beta (eps=0, EB)."""
    n = L.shape[0]
    active = dw != 0.0
    La = L[active]
    dwa = dw[active]

    if np.isfinite(eta_hi):
        in_band = La[(La > eta_lo) & (La <= eta_hi)]
    else:
        in_band = La[La > eta_lo]
    endpoints = [eta_lo, eta_hi] if np.isfinite(eta_hi) else [eta_lo]
    bpts = np.unique(np.concatenate((endpoints, in_band)))

    order = np.argsort(La)[::-1]
    Ls = La[order]
    ds = dwa[order]
    d2s = ds * ds
    c_d = np.concatenate(([0.0], np.cumsum(ds)))
    c_d2 = np.concatenate(([0.0], np.cumsum(d2s)))

    cnt = np.searchsorted(-Ls, -bpts, side="left")
    mean_Z = c_d[cnt] / n
    var_Z = np.maximum(0.0, (c_d2[cnt] - n * mean_Z * mean_Z) / (n - 1))
    ucbs = eb_ucb_from_moments(mean_Z, var_Z, n, 2.0 / kappa, delta2)
    return bool(np.max(ucbs) <= 0.0)


def lct_q_betting_certifies(L, dw, eta_lo, eta_hi, delta2, kappa):
    """Exact piecewise-constant supremum over [eta_lo, eta_hi] for uncapped Q_beta (eps=0, Betting)."""
    active = dw != 0.0
    La = L[active]
    dwa = dw[active]

    if np.isfinite(eta_hi):
        in_band = La[(La > eta_lo) & (La <= eta_hi)]
    else:
        in_band = La[La > eta_lo]
    endpoints = [eta_lo, eta_hi] if np.isfinite(eta_hi) else [eta_lo]
    bpts = np.unique(np.concatenate((endpoints, in_band)))

    # Order breakpoints by descending empirical mean (short-circuit on first failure)
    order = np.argsort(La)[::-1]
    Ls = La[order]
    ds = dwa[order]
    c_d = np.concatenate(([0.0], np.cumsum(ds)))
    cnt = np.searchsorted(-Ls, -bpts, side="left")
    emp_sums = c_d[cnt]
    check_order = np.argsort(emp_sums)[::-1]

    bound = 1.0 / kappa
    for idx_b in check_order:
        eta = bpts[idx_b]
        Z = np.where(L > eta, dw, 0.0)
        if not betting_reject_upper(Z, -bound, bound, 0.0, delta2):
            return False
    return True


def lct_q_betting_oracle_eta_certifies(L, dw, eta0_true, delta, kappa):
    """Diagnostic (Risk R4): Localized Betting Q_beta using known oracle eta0*."""
    Z = np.where(L > eta0_true, dw, 0.0)
    bound = 1.0 / kappa
    return betting_reject_upper(Z, -bound, bound, 0.0, delta)


# =============================================================================
# 5. Baselines B1-OPRA (Huang et al. 2021) & B1-TwoArm-Betting (UnO-Tight)
# =============================================================================

def _cvar_from_step_cdf(vals, cdf_vals, beta, cap_B):
    """
    Exact CVaR_beta(F) = (1/(1-beta)) * int_beta^1 Q_u(F) du
    for right-continuous step CDF on [0, cap_B] with jumps at vals and cdf_vals[-1] = 1.
    """
    # Clip and ensure monotone non-decreasing ending at 1 at cap_B
    F = np.clip(cdf_vals, 0.0, 1.0)
    F = np.maximum.accumulate(F)
    # Integrate (1 - max(F(t), beta)) dt from 0 to cap_B, divided by (1 - beta), plus 0
    # Exact identity: CVaR_beta(F) = (1/(1-beta)) * int_0^{cap_B} (1 - max(F(t), beta)) dt
    # (for L in [0, cap_B], since int_beta^1 Q_u du = int_0^{cap_B} (1 - max(F(t), beta)) dt!)
    # Verify: for F = point mass at c, max(F(t), beta) = beta on [0, c) and 1 on [c, B]
    # => int = c * (1 - beta) => / (1 - beta) = c. Exact!
    if vals.size == 0:
        return float(cap_B)
    # Construct right-continuous step intervals: [0, vals[0]), [vals[0], vals[1]), ..., [vals[-1], cap_B]
    edges = np.concatenate(([0.0], vals, [cap_B]))
    # CDF level on each interval: 0 on [0, vals[0]), F[0] on [vals[0], vals[1]), ..., F[-1] on [vals[-1], cap_B]
    levels = np.concatenate(([0.0], F))
    widths = np.maximum(0.0, np.diff(edges))
    integrand = 1.0 - np.maximum(levels, beta)
    return float(np.sum(widths * integrand) / (1.0 - beta))


def opra_cdf_eps(n, w_max, w2_exact, delta_arm):
    """Huang et al. (NeurIPS 2021) Thm 5.1 Eq. (3) & Eq. (4) — take tighter of the two."""
    log_term = math.log(4.0 / delta_arm)
    eps1 = math.sqrt(8.0 * w_max * w_max * log_term / n)
    eps2 = (4.0 * w_max * log_term) / n + 2.0 * math.sqrt(2.0 * w2_exact * log_term / n)
    return min(eps1, eps2)


def opra_cdf_steps(L, w):
    """Return sorted distinct positive-weight losses and clipped IS CDF values."""
    n = L.shape[0]
    mask = w > 0.0
    Lp = np.concatenate(([0.0], L[mask]))
    wp = np.concatenate(([0.0], w[mask]))
    order = np.argsort(Lp)
    Ls = Lp[order]
    ws = wp[order]
    uniq, inv = np.unique(Ls, return_inverse=True)
    w_agg = np.bincount(inv, weights=ws)
    cdf = np.minimum(1.0, np.cumsum(w_agg) / n)
    return uniq, cdf


def b1_opra_cvar_certifies(L, w_lam, w0, eps_lam, eps_0, beta, cap_B, tol_eps=0.0):
    v_lam, F_lam = opra_cdf_steps(L, w_lam)
    v_0, F_0 = opra_cdf_steps(L, w0)
    # Lower envelope F_lam^- gives upper bound on CVaR(pi_lam)
    cvar_lam_ub = _cvar_from_step_cdf(v_lam, np.maximum(0.0, F_lam - eps_lam), beta, cap_B)
    # Upper envelope F_0^+ gives lower bound on CVaR(pi_0)
    cvar_0_lb = _cvar_from_step_cdf(v_0, np.minimum(1.0, F_0 + eps_0), beta, cap_B)
    return bool(cvar_lam_ub <= cvar_0_lb + tol_eps)


def b1_opra_q_certifies(L, w_lam, w0, eps_lam, eps_0, beta, tol_eps=0.0):
    if beta + eps_lam > 1.0:
        return False
    v_lam, F_lam = opra_cdf_steps(L, w_lam)
    v_0, F_0 = opra_cdf_steps(L, w0)
    # Upper bound on Q_beta(pi_lam): inf{t : F_lam(t) >= beta + eps_lam}
    idx_lam = np.where(F_lam >= beta + eps_lam)[0]
    if idx_lam.size == 0:
        return False
    q_lam_ub = float(v_lam[idx_lam[0]])
    # Lower bound on Q_beta(pi_0): inf{t : F_0(t) >= beta - eps_0}
    target0 = beta - eps_0
    if target0 <= 0.0:
        q_0_lb = 0.0
    else:
        idx_0 = np.where(F_0 >= target0)[0]
        q_0_lb = float(v_0[idx_0[0]]) if idx_0.size > 0 else float("inf")
    return bool(q_lam_ub <= q_0_lb + tol_eps)


def b1_twoarm_betting_cvar_certifies(L, w_lam, w0, eta_lam_oracle, eta0_oracle, beta, delta, kappa, cap_B, tol_eps=0.0):
    """
    Separate-policy CVaR betting diagnostic with both population VaRs supplied.
    It cannot be deployed when these VaRs are unknown.
    """
    Z_lam = w_lam * np.maximum(0.0, L - eta_lam_oracle)
    Z_0 = w0 * np.maximum(0.0, L - eta0_oracle)
    ub_ex_lam = betting_ucb_bisect(Z_lam, 0.0, (cap_B - eta_lam_oracle) / kappa, delta / 2.0)
    lb_ex_0 = betting_lcb_bisect(Z_0, 0.0, (cap_B - eta0_oracle) / kappa, delta / 2.0)
    cvar_lam_ub = eta_lam_oracle + ub_ex_lam / (1.0 - beta)
    cvar_0_lb = eta0_oracle + lb_ex_0 / (1.0 - beta)
    return bool(cvar_lam_ub <= cvar_0_lb + tol_eps)


def b1_twoarm_betting_q_certifies(L, w_lam, eta0_lo, beta, delta, kappa):
    """Compare separately certified candidate upper and baseline lower quantiles."""
    if not np.isfinite(eta0_lo):
        return False
    z = np.where(L > eta0_lo, w_lam, 0.0)
    return monotone_betting_reject_upper(z, 0.0, 1.0 / kappa, 1.0 - beta, delta / 2.0)


# =============================================================================
# 6. Single-Cell Monte Carlo Runner
# =============================================================================

# Exact second moment E[1/e(a|X)] for e(1|x) = 0.25 + 0.5*x on [0, 1]
def exact_w2_policy(p_lam):
    """Exact E[w_lambda(X, A)^2] = int_0^{p_lam} 1/(0.25+0.5x) dx + int_{p_lam}^1 1/(0.75-0.5x) dx."""
    t1 = 2.0 * math.log((0.25 + 0.5 * p_lam) / 0.25) if p_lam > 0 else 0.0
    t2 = 2.0 * math.log((0.75 - 0.5 * p_lam) / 0.25) if p_lam < 1.0 else 0.0
    return t1 + t2


def run_cell(task):
    """Run 200 MC draws for a single (n_cal, beta, p_star, cap_B, null_mode, tag) cell."""
    n_cal, beta, p_star, cap_B, null_mode, tag, n_draws = task
    t0 = time.perf_counter()
    kappa = DGP_PARAMS["kappa"]
    delta = 0.05
    delta1 = 0.01  # 20% for baseline-VaR CI, 80% for localized test
    delta2 = delta - delta1

    # Main sequence starts at 0.05. The paired sensitivity starts at 0.15.
    p_grid = np.arange(1, 21) / 20
    if tag in {"order_15pct", "null_order_15pct"}:
        p_grid = p_grid[2:]
    M = p_grid.size

    # Precompute EXACT analytical ground truth for every grid point
    cvar0_true = true_capped_cvar(beta, 0.0, p_star, cap_B, null_mode=null_mode)
    q0_true = true_quantile(beta, 0.0, p_star, cap_B=None, null_mode=null_mode)
    eta0_cap_true = true_quantile(beta, 0.0, p_star, cap_B=cap_B, null_mode=null_mode)

    cvar_safe = np.zeros(M, dtype=bool)
    q_safe = np.zeros(M, dtype=bool)
    eta_lam_cap_oracles = np.zeros(M, dtype=np.float64)
    for m, p_lam in enumerate(p_grid):
        cvar_m = true_capped_cvar(beta, float(p_lam), p_star, cap_B, null_mode=null_mode)
        q_m = true_quantile(beta, float(p_lam), p_star, cap_B=None, null_mode=null_mode)
        cvar_safe[m] = (cvar_m <= cvar0_true + 1e-12)
        q_safe[m] = (q_m <= q0_true + 1e-12)
        eta_lam_cap_oracles[m] = true_quantile(beta, float(p_lam), p_star, cap_B=cap_B, null_mode=null_mode)

    # Precompute OPRA exact CDF radius for each p_lam (delta/2 per arm)
    w_max = 1.0 / kappa
    eps_opra_0 = opra_cdf_eps(n_cal, w_max, exact_w2_policy(0.0), delta / 2.0)
    eps_opra_lam = np.array([opra_cdf_eps(n_cal, w_max, exact_w2_policy(float(p)), delta / 2.0) for p in p_grid])

    methods = [
        "LCT_LTT_EB",
        "LCT_LTT_Betting",
        "LCT_LTT_OracleEta0_Betting",
        "B1_OPRA",
        "B1_TwoArm_Betting",
    ]
    cvar_dep = {k: np.zeros(n_draws) for k in methods}
    cvar_viol = {k: np.zeros(n_draws, dtype=bool) for k in methods}
    q_dep = {k: np.zeros(n_draws) for k in methods}
    q_viol = {k: np.zeros(n_draws, dtype=bool) for k in methods}
    ci_widths_cap = np.zeros(n_draws)
    ci_widths_uncap = np.zeros(n_draws)
    ci_cov_cap = np.zeros(n_draws, dtype=bool)
    ci_cov_uncap = np.zeros(n_draws, dtype=bool)

    for seed in range(n_draws):
        # Pair caps and orderings on the same simulated log within each (n, beta, p_star).
        ss = np.random.SeedSequence([
            seed, n_cal, int(round(beta * 1000)), int(round(p_star * 1000)),
            500, int(null_mode)
        ])
        rng = np.random.default_rng(ss)
        X, A, e1, _, L_uncap = generate_data(n_cal, p_star, rng, null_mode=null_mode)
        L_cap = np.minimum(L_uncap, cap_B)
        w0 = np.where(A == 0, 1.0 / (1.0 - e1), 0.0)
        w1 = np.where(A == 1, 1.0 / e1, 0.0)

        # Baseline-VaR CIs (level 1 - delta1)
        eta_lo_eb_cap, eta_hi_eb_cap = baseline_var_ci_eb(L_cap, w0, beta, delta1, kappa, cap_B=cap_B)
        eta_lo_bt_cap, eta_hi_bt_cap = baseline_var_ci_betting(L_cap, w0, beta, delta1, kappa, cap_B=cap_B)
        eta_lo_eb_q, eta_hi_eb_q = baseline_var_ci_eb(L_uncap, w0, beta, delta1, kappa, cap_B=None)
        eta_lo_bt_q, eta_hi_bt_q = baseline_var_ci_betting(L_uncap, w0, beta, delta1, kappa, cap_B=None)

        ci_widths_cap[seed] = eta_hi_bt_cap - eta_lo_bt_cap
        ci_widths_uncap[seed] = eta_hi_bt_q - eta_lo_bt_q if np.isfinite(eta_hi_bt_q) else float("nan")
        ci_cov_cap[seed] = (eta_lo_bt_cap <= eta0_cap_true <= eta_hi_bt_cap)
        ci_cov_uncap[seed] = (eta_lo_bt_q <= q0_true <= eta_hi_bt_q)

        # For B1-TwoArm-Betting Q_beta: one-sided delta/2 lower bound on Q_beta(pi_0) => delta1 = delta
        eta0_lo_b1, _ = baseline_var_ci_betting(L_uncap, w0, beta, delta, kappa, cap_B=None)

        # Fixed-sequence LTT scan over p_grid for each method & risk functional
        def run_fseq(cert_fn, safe_mask):
            best_m = -1
            rejected_unsafe = False
            for m in range(M):
                if cert_fn(m):
                    best_m = m
                    rejected_unsafe |= not bool(safe_mask[m])
                else:
                    break
            if best_m == -1:
                return 0.0, False
            return float(p_grid[best_m]), rejected_unsafe

        def get_dw(m):
            in_S = X <= p_grid[m]
            return np.where(in_S, w1 - w0, 0.0)

        def get_wlam(m):
            in_S = X <= p_grid[m]
            return np.where(in_S, w1, w0)

        # --- Capped CVaR ---
        cvar_dep["LCT_LTT_EB"][seed], cvar_viol["LCT_LTT_EB"][seed] = run_fseq(
            lambda m: lct_cvar_eb_certifies(L_cap, get_dw(m), eta_lo_eb_cap, eta_hi_eb_cap, beta, delta2, kappa, cap_B),
            cvar_safe,
        )
        cvar_dep["LCT_LTT_Betting"][seed], cvar_viol["LCT_LTT_Betting"][seed] = run_fseq(
            lambda m: lct_cvar_betting_certifies(L_cap, get_dw(m), eta_lo_bt_cap, eta_hi_bt_cap, beta, delta2, kappa, cap_B),
            cvar_safe,
        )
        cvar_dep["LCT_LTT_OracleEta0_Betting"][seed], cvar_viol["LCT_LTT_OracleEta0_Betting"][seed] = run_fseq(
            lambda m: lct_cvar_betting_oracle_eta_certifies(L_cap, get_dw(m), eta0_cap_true, beta, delta, kappa, cap_B),
            cvar_safe,
        )
        cvar_dep["B1_OPRA"][seed], cvar_viol["B1_OPRA"][seed] = run_fseq(
            lambda m: b1_opra_cvar_certifies(L_cap, get_wlam(m), w0, eps_opra_lam[m], eps_opra_0, beta, cap_B),
            cvar_safe,
        )
        cvar_dep["B1_TwoArm_Betting"][seed], cvar_viol["B1_TwoArm_Betting"][seed] = run_fseq(
            lambda m: b1_twoarm_betting_cvar_certifies(
                L_cap, get_wlam(m), w0, eta_lam_cap_oracles[m], eta0_cap_true, beta, delta, kappa, cap_B
            ),
            cvar_safe,
        )

        # --- Uncapped Q_beta ---
        q_dep["LCT_LTT_EB"][seed], q_viol["LCT_LTT_EB"][seed] = run_fseq(
            lambda m: lct_q_eb_certifies(L_uncap, get_dw(m), eta_lo_eb_q, eta_hi_eb_q, delta2, kappa),
            q_safe,
        )
        q_dep["LCT_LTT_Betting"][seed], q_viol["LCT_LTT_Betting"][seed] = run_fseq(
            lambda m: lct_q_betting_certifies(L_uncap, get_dw(m), eta_lo_bt_q, eta_hi_bt_q, delta2, kappa),
            q_safe,
        )
        q_dep["LCT_LTT_OracleEta0_Betting"][seed], q_viol["LCT_LTT_OracleEta0_Betting"][seed] = run_fseq(
            lambda m: lct_q_betting_oracle_eta_certifies(L_uncap, get_dw(m), q0_true, delta, kappa),
            q_safe,
        )
        q_dep["B1_OPRA"][seed], q_viol["B1_OPRA"][seed] = run_fseq(
            lambda m: b1_opra_q_certifies(L_uncap, get_wlam(m), w0, eps_opra_lam[m], eps_opra_0, beta),
            q_safe,
        )
        q_dep["B1_TwoArm_Betting"][seed], q_viol["B1_TwoArm_Betting"][seed] = run_fseq(
            lambda m: b1_twoarm_betting_q_certifies(L_uncap, get_wlam(m), eta0_lo_b1, beta, delta, kappa),
            q_safe,
        )

    elapsed = time.perf_counter() - t0
    oracle_max_cvar = float(p_grid[np.where(cvar_safe)[0][-1]]) if np.any(cvar_safe) else 0.0
    oracle_max_q = float(p_grid[np.where(q_safe)[0][-1]]) if np.any(q_safe) else 0.0

    rows = []
    for risk_name, dep_dict, viol_dict, oracle_p in [
        ("CVaR_capped", cvar_dep, cvar_viol, oracle_max_cvar),
        ("Q_beta_uncapped", q_dep, q_viol, oracle_max_q),
    ]:
        for meth in methods:
            d = dep_dict[meth]
            v = viol_dict[meth]
            rows.append({
                "tag": tag,
                "null_mode": bool(null_mode),
                "risk": risk_name,
                "method": meth,
                "n_cal": n_cal,
                "beta": beta,
                "p_star": p_star,
                "cap_B": round(float(cap_B), 3),
                "oracle_max_safe_p": oracle_p,
                "n_draws": n_draws,
                "fwer": float(np.mean(v)),
                "fwer_cp95_upper": (
                    float(beta_dist.ppf(0.95, v.sum() + 1, n_draws - v.sum()))
                    if v.sum() < n_draws else 1.0
                ),
                "fwer_se": float(np.std(v, ddof=1) / math.sqrt(n_draws)),
                "mean_deploy_se": float(np.std(d, ddof=1) / math.sqrt(n_draws)),
                "mean_deploy": float(np.mean(d)),
                "median_deploy": float(np.median(d)),
                "iqr_deploy": float(np.percentile(d, 75) - np.percentile(d, 25)),
                "frac_nontrivial": float(np.mean(d > 0.0)),
                "ci_cov_baseline_var": float(np.mean(ci_cov_cap if risk_name == "CVaR_capped" else ci_cov_uncap)),
                "ci_width_baseline_var": float(np.nanmean(ci_widths_cap if risk_name == "CVaR_capped" else ci_widths_uncap)),
                "cell_wall_sec": round(elapsed, 2),
            })
    return rows


def main():
    t_start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=Path("results/phase2_pilot"))
    args = parser.parse_args()
    n_draws = args.draws
    if n_draws < 2:
        parser.error("--draws must be at least 2")
    out_dir = args.out_dir
    csv_path = out_dir / "tightness_check.csv"
    diag_path = out_dir / "pilot_diagnostics.json"
    if csv_path.exists() or diag_path.exists():
        parser.error(f"refusing to overwrite pilot outputs in {out_dir}")

    # Baseline Q_0.99 for cap sensitivity sweep
    q99_base = true_quantile(0.99, 0.0, 0.35, cap_B=None, null_mode=False)
    print(f"Baseline L(0) analytical quantiles: Q80={true_quantile(0.80,0,0.35):.3f}, "
          f"Q90={true_quantile(0.90,0,0.35):.3f}, Q95={true_quantile(0.95,0,0.35):.3f}, Q99={q99_base:.3f}")

    tasks = []
    # 1. Main 3 x 3 x 3 grid (B = 50)
    for n_cal in [2000, 5000, 10000]:
        for beta in [0.80, 0.90, 0.95]:
            for p_star in [0.15, 0.35, 0.60]:
                tasks.append((n_cal, beta, p_star, 50.0, False, "main_grid", n_draws))

    # 2. Null-boundary FWER verification (n_cal = 5000, beta in {0.80, 0.90, 0.95})
    for beta in [0.80, 0.90, 0.95]:
        tasks.append((5000, beta, 0.0, 50.0, True, "null_fwer", n_draws))

    # 3. Cap-B sensitivity at focal cell (n_cal=5000, beta=0.90, p_star=0.35)
    for mult, label in [(1.0, "cap_1xQ99"), (2.0, "cap_2xQ99"), (5.0, "cap_5xQ99")]:
        tasks.append((5000, 0.90, 0.35, mult * q99_base, False, label, n_draws))

    # Paired sensitivity: skip signal-poor first tests, decided before inspecting draws.
    for n_cal in [5000, 10000]:
        tasks.append((n_cal, 0.90, 0.35, 50.0, False, "order_15pct", n_draws))
        tasks.append((n_cal, 0.90, 0.0, 50.0, True, "null_order_15pct", n_draws))

    print(f"Launching {len(tasks)} cells x {n_draws} draws using {args.workers} worker(s).", flush=True)
    all_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(run_cell, tasks), 1):
            all_rows.extend(res)
            if i % 6 == 0 or i == len(tasks):
                print(f"  Completed {i}/{len(tasks)} cells ({time.perf_counter() - t_start:.1f}s elapsed)", flush=True)

    df = pd.DataFrame(all_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, mode="x")

    # Peak RSS across parent + children (ru_maxrss is in KB on Linux)
    rss_self_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    rss_child_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
    total_sec = time.perf_counter() - t_start

    diag = {
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "total_wall_sec": round(total_sec, 2),
        "peak_rss_parent_mb": round(rss_self_mb, 2),
        "peak_rss_child_mb": round(rss_child_mb, 2),
        "workers": args.workers,
        "upper_bound_peak_aggregate_rss_mib": round(rss_self_mb + args.workers * rss_child_mb, 2),
        "n_rows": int(len(df)),
        "any_nan_numeric": bool(df.drop(columns=["ci_width_baseline_var"]).isna().any().any()),
        "baseline_q99": round(q99_base, 4),
    }
    with diag_path.open("x") as f:
        json.dump(diag, f, indent=2)

    print("Saved:", csv_path)
    print("Diagnostics:", json.dumps(diag, indent=2))

    # Print key summary tables
    print("\n=== NULL-BOUNDARY FWER VERIFICATION (n_cal=5000, delta=0.05) ===")
    sub_null = df[df["tag"] == "null_fwer"][["risk", "beta", "method", "fwer", "mean_deploy", "ci_cov_baseline_var"]]
    print(sub_null.to_string(index=False))

    print("\n=== FOCAL GATE G2 CELL: n_cal=5000, beta=0.90, p_star=0.35 (B=50) ===")
    sub_focal = df[(df["tag"] == "main_grid") & (df["n_cal"] == 5000) & (df["beta"] == 0.90) & (df["p_star"] == 0.35)][
        ["risk", "method", "oracle_max_safe_p", "fwer", "mean_deploy", "median_deploy", "frac_nontrivial"]
    ]
    print(sub_focal.to_string(index=False))

    print("\n=== MAIN GRID PIVOT (mean_deploy) AT p_star=0.35 ===")
    piv = df[(df["tag"] == "main_grid") & (df["p_star"] == 0.35)].pivot_table(
        index=["risk", "beta", "n_cal"], columns="method", values="mean_deploy"
    )
    print(piv.round(4).to_string())

    print("\n=== CAP-B SENSITIVITY AT n_cal=5000, beta=0.90, p_star=0.35 (CVaR_capped) ===")
    sub_cap = df[(df["tag"].str.startswith("cap_")) & (df["risk"] == "CVaR_capped")][
        ["tag", "cap_B", "method", "fwer", "mean_deploy", "frac_nontrivial"]
    ]
    print(sub_cap.to_string(index=False))


if __name__ == "__main__":
    main()
