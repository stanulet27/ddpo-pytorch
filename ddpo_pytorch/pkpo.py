"""Pass-at-k Policy Optimization (PKPO) reward transforms.

Implements Listing 1 from Walder et al., Pass@K Policy Optimization.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _m_normed(N: int, K: int, i: int, j: int) -> float:
    if i == j and i >= K - 1:
        return float(
            K
            / (N - K + 1)
            * np.prod(np.arange(i - K + 2, i + 1) / np.arange(N - K + 2, N + 1))
        )
    if j > i and j >= K - 1 and K >= 2:
        return float(
            K
            / (N - K + 1)
            * (K - 1)
            / N
            * np.prod(np.arange(j - K + 2, j) / np.arange(N - K + 2, N))
        )
    return 0.0


def _m_diagonal(N: int, K: int) -> np.ndarray:
    return np.array([_m_normed(N, K, i, i) for i in range(N)])


def rho(g: np.ndarray, K: int) -> float:
    """Estimated pass@k proxy (Equation 12)."""
    g = np.asarray(g, dtype=float)
    return float((np.sort(g) * _m_diagonal(len(g), K)).sum())


def _delta(N: int, K: int, i: int) -> float:
    return _m_normed(N, K, i, i + 1) - _m_normed(N, K, i + 1, i + 1)


def _deltas(N: int, K: int) -> np.ndarray:
    return np.array([_delta(N - 1, K, i) for i in range(N - 2)])


def _sorted_apply(func: Callable) -> Callable:
    def inner(x: np.ndarray, *args, **kwargs) -> np.ndarray:
        i_sort = np.argsort(x)
        func_x = np.zeros_like(x, dtype=float)
        func_x[i_sort] = func(x[i_sort], *args, **kwargs)
        return func_x

    return inner


@_sorted_apply
def s(g: np.ndarray, K: int) -> np.ndarray:
    """Equation 19."""
    N = len(g)
    c = g * _m_diagonal(N, K)
    c[: (N - 1)] += g[1:] * _deltas(N + 1, K)
    return np.cumsum(c[::-1])[::-1]


@_sorted_apply
def _b(g: np.ndarray, K: int) -> np.ndarray:
    N = len(g)
    w = (_m_diagonal(N - 1, K) * np.arange(1, N)).astype(float)
    w[1:] += _deltas(N, K) * np.arange(1, N - 1)
    c1 = np.array([(w * g[1:]).sum()])
    c2 = (g[:-1] - g[1:]) * w
    return np.cumsum(np.concatenate((c1, c2)))


def sloo(g: np.ndarray, K: int) -> np.ndarray:
    """Equation 29."""
    g = np.asarray(g, dtype=float)
    return s(g, K) - _b(g, K) / (len(g) - 1)


def sloo_minus_one(g: np.ndarray, K: int) -> np.ndarray:
    """Equation 33 — recommended PKPO estimator."""
    g = np.asarray(g, dtype=float)
    n = len(g)
    if K < 1 or K > n:
        raise ValueError(f"Require 1 <= K <= len(g), got K={K}, len(g)={n}")
    if K == 1:
        # Paper: k_opt=1 is the baseline (no multivariate PKPO transform).
        return g.copy()
    return s(g, K) - _b(g, K - 1) * K / (K - 1) / n


def apply_pkpo_to_groups(
    rewards: np.ndarray,
    group_ids: np.ndarray,
    k: int,
) -> np.ndarray:
    """Apply sloo_minus_one independently to each prompt group."""
    rewards = np.asarray(rewards, dtype=float)
    group_ids = np.asarray(group_ids)
    effective = np.empty_like(rewards)
    for gid in np.unique(group_ids):
        mask = group_ids == gid
        effective[mask] = sloo_minus_one(rewards[mask], k)
    return effective


def mean_rho_per_group(
    rewards: np.ndarray,
    group_ids: np.ndarray,
    k: int,
) -> float:
    """Mean rho(g, k) across prompt groups (monitoring)."""
    rewards = np.asarray(rewards, dtype=float)
    group_ids = np.asarray(group_ids)
    values = []
    for gid in np.unique(group_ids):
        mask = group_ids == gid
        values.append(rho(rewards[mask], k))
    return float(np.mean(values)) if values else 0.0
