"""Small PKPO config for smoke tests (low n, few batches)."""

import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def get_config():
    config = base.get_config()

    config.num_epochs = 2
    config.sample.batch_size = 1
    config.sample.num_batches_per_epoch = 2
    config.train.batch_size = 1

    config.pkpo.enabled = True
    config.pkpo.n = 4
    config.pkpo.k = 2
    config.pkpo.anneal_k = False

    return config
