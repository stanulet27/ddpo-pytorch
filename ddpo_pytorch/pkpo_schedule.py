"""k-annealing schedule for PKPO training."""


def current_pkpo_k(epoch: int, pkpo_config) -> int:
    """Return the pass@k target for this epoch (clamped to [1, n])."""
    n = int(pkpo_config.n)
    if not pkpo_config.anneal_k:
        k = int(pkpo_config.k)
    else:
        anneal_epochs = int(pkpo_config.anneal_epochs)
        t = min(float(epoch) / anneal_epochs, 1.0)
        k = int(round(pkpo_config.k_start + t * (pkpo_config.k_end - pkpo_config.k_start)))
    return max(1, k)
