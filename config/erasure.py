import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def prompt_image_alignment():
    config = base.get_config()

    config.pretrained.model = "CompVis/stable-diffusion-v1-4"

    config.num_epochs = 100
    config.use_lora = True
    config.save_freq = 1
    config.num_checkpoint_limit = 100000000

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 6

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 6

    # prompting
    config.prompt_fn = "bears_combined"
    config.prompt_fn_kwargs = {}

    # rewards
    config.reward_fn = "ensemble_detector_score"
    config.reward_fn_kwargs = {"unsafe_concept": "teddy bear"}

    config.per_prompt_stat_tracking = {
        "buffer_size": 16,
        "min_count": 16,
    }


    return config


def get_config():
    return prompt_image_alignment()
