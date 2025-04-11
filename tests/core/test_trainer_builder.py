"""
unit tests for axolotl.core.trainer_builder
"""

import pytest

from axolotl.core.trainer_builder import HFCausalTrainerBuilder, HFRLTrainerBuilder
from axolotl.utils.callbacks import FractionalEpochCallback
from axolotl.utils.config import normalize_config
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_model, load_tokenizer


@pytest.fixture(name="cfg")
def fixture_cfg():
    cfg = DictDefault(
        {
            "base_model": "HuggingFaceTB/SmolLM2-135M",
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.00005,
            "save_steps": 100,
            "output_dir": "./model-out",
            "warmup_steps": 10,
            "gradient_checkpointing": False,
            "optimizer": "adamw_torch_fused",
            "sequence_len": 2048,
            "rl": True,
            "adam_beta1": 0.998,
            "adam_beta2": 0.9,
            "adam_epsilon": 0.00001,
            "dataloader_num_workers": 1,
            "dataloader_pin_memory": True,
            "model_config_type": "llama",
            "special_tokens": {
                "pad_token": "<|endoftext|>",
            },
        }
    )

    normalize_config(cfg)

    return cfg


@pytest.fixture(name="tokenizer")
def fixture_tokenizer(cfg):
    return load_tokenizer(cfg)


@pytest.fixture(name="model")
def fixture_model(cfg, tokenizer):
    return load_model(cfg, tokenizer)


class TestHFRLTrainerBuilder:
    """
    TestCase class for DPO trainer builder
    """

    def test_build_training_arguments(self, cfg, model, tokenizer):
        builder = HFRLTrainerBuilder(cfg, model, tokenizer)
        training_arguments = builder.build_training_arguments(100)
        assert training_arguments.adam_beta1 == 0.998
        assert training_arguments.adam_beta2 == 0.9
        assert training_arguments.adam_epsilon == 0.00001
        assert training_arguments.dataloader_num_workers == 1
        assert training_arguments.dataloader_pin_memory is True


class TestHFCausalTrainerBuilder:
    """
    TestCase class for Causal trainer builder
    """

    def test_fractional_epoch_callback(self, cfg, model, tokenizer):
        # Modify config to use save_fractions
        cfg.save_fractions = [0.01, 0.1, 0.5, 1.0]
        cfg.num_epochs = 2
        
        # Remove save_steps if it exists
        if hasattr(cfg, "save_steps"):
            delattr(cfg, "save_steps")
        
        builder = HFCausalTrainerBuilder(cfg, model, tokenizer)
        
        # Total steps for 2 epochs
        total_steps = 100
        
        # Get callbacks and check for FractionalEpochCallback
        callbacks = builder.get_callbacks(total_num_steps=total_steps)
        
        # Find the FractionalEpochCallback instance
        fraction_callback = None
        for callback in callbacks:
            if isinstance(callback, FractionalEpochCallback):
                fraction_callback = callback
                break
        
        assert fraction_callback is not None, "FractionalEpochCallback not found in callbacks"
        assert fraction_callback.save_fractions == [0.01, 0.1, 0.5, 1.0]
        assert fraction_callback.steps_per_epoch == 50  # 100 steps / 2 epochs
        
        # Check if next checkpoint steps are calculated correctly
        assert fraction_callback.next_checkpoint_step[0.01] == 0  # 0.01 * 50 = 0.5, rounded to 0
        assert fraction_callback.next_checkpoint_step[0.1] == 5  # 0.1 * 50 = 5
        assert fraction_callback.next_checkpoint_step[0.5] == 25  # 0.5 * 50 = 25
        assert fraction_callback.next_checkpoint_step[1.0] == 50  # 1.0 * 50 = 50
