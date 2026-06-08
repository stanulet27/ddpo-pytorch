import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def get_config():
    config = base.get_config()

    config.run_name = "pkpo-teddybear-n8-k8to1-e100_passatk_full_4_6_4_3"
    config.num_epochs = 100
    config.save_freq = 20
    config.num_checkpoint_limit = 100000000

    # gpu utilization
    config.sample.batch_size = 4
    config.sample.num_batches_per_epoch = 6
    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 3

    config.prompt_fn = "from_file"
    config.prompt_fn_kwargs = {"path": "bears_combined.txt"}
    config.reward_fn = "ensemble_detector_score"
    config.reward_fn_kwargs = {"unsafe_concept": "teddy bear"}

    config.pkpo.enabled = True
    config.pkpo.n = 8
    config.pkpo.k = 8
    config.pkpo.anneal_k = True
    config.pkpo.k_start = 8
    config.pkpo.k_end = 1
    config.pkpo.anneal_epochs = 50

    config.save_freq = 1

    return config
