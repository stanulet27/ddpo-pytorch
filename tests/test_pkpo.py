import numpy as np
import pytest

from ddpo_pytorch.pkpo import (
    apply_pkpo_to_groups,
    rho,
    sloo_minus_one,
)


def test_sloo_minus_one_length_matches_input():
    g = np.array([0.1, 0.5, 0.9, 0.2])
    out = sloo_minus_one(g, K=2)
    assert out.shape == g.shape


def test_sloo_minus_one_k1_is_identity():
    g = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(sloo_minus_one(g, K=1), g)


def test_sloo_minus_one_sort_invariance():
    g = np.array([0.2, 0.9, 0.1, 0.5])
    out1 = sloo_minus_one(g, K=2)
    perm = np.array([2, 1, 3, 0])
    out2 = np.empty_like(out1)
    out2[perm] = sloo_minus_one(g[perm], K=2)
    np.testing.assert_allclose(out1, out2, rtol=1e-10)


def test_apply_pkpo_to_groups():
    rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    group_ids = np.array([0, 0, 0, 1, 1, 1])
    effective = apply_pkpo_to_groups(rewards, group_ids, k=2)
    np.testing.assert_array_equal(
        effective[:3], sloo_minus_one(rewards[:3], K=2)
    )
    np.testing.assert_array_equal(
        effective[3:], sloo_minus_one(rewards[3:], K=2)
    )


def test_rho_bounds_binary():
    g = np.array([0.0, 0.0, 1.0, 1.0])
    assert 0.0 <= rho(g, K=2) <= 1.0


def test_invalid_k_raises():
    with pytest.raises(ValueError):
        sloo_minus_one(np.array([1.0, 2.0]), K=3)
