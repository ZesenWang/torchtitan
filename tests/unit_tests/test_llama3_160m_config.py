# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import ConfigManager
from torchtitan.models.llama3 import llama3_configs, Llama3Model
from torchtitan.trainer import Trainer


def test_llama3_160m_model_config():
    config = llama3_configs["160M"](attn_backend="flex")
    layer = config.layers[0]

    assert config.dim == 768
    assert config.vocab_size == 32000
    assert config.enable_weight_tying
    assert config.tok_embeddings.num_embeddings == 32000
    assert config.tok_embeddings.embedding_dim == 768
    assert config.lm_head.in_features == 768
    assert config.lm_head.out_features == 32000
    assert len(config.layers) == 18
    assert layer.attention.n_heads == 12
    assert layer.attention.n_kv_heads is None
    assert layer.feed_forward.w1.out_features == 2048
    assert layer.feed_forward.w2.in_features == 2048
    assert layer.feed_forward.w3.out_features == 2048
    assert layer.attention.rope.dim == 64
    assert layer.attention.rope.max_seq_len == 131072
    assert layer.attention.rope.theta == 500000
    assert layer.attention.rope.scaling == "llama"


def test_llama3_160m_weight_tying():
    model = Llama3Model(llama3_configs["160M"](attn_backend="flex"))

    assert model.tok_embeddings.weight is model.lm_head.weight
    assert sum(param.numel() for param in model.parameters()) == 152006400


def test_llama3_160m_trainer_config():
    config = ConfigManager().parse_args(
        ["--module", "llama3", "--config", "llama3_160m"]
    )
    assert isinstance(config, Trainer.Config)
    assert config.model_spec is not None
    optimizer_kwargs = config.optimizer.param_groups[0].optimizer_kwargs

    assert config.model_spec.name == "llama3"
    assert config.model_spec.flavor == "160M"
    assert config.hf_assets_path == "./assets/hf/Llama-2-70b-hf"
    assert optimizer_kwargs == {
        "lr": 5e-3,
        "betas": (0.975, 0.9),
        "eps": 1e-8,
        "weight_decay": 0.1,
    }
    assert config.lr_scheduler.warmup_steps == 3662
    assert config.lr_scheduler.decay_type == "cosine"
    assert config.lr_scheduler.min_lr_factor == 0.0
    assert config.training.local_batch_size == 8
    assert config.training.global_batch_size == 128
    assert config.training.seq_len == 2048
    assert config.training.max_norm == 1.0
    assert config.training.steps == 36622
    assert config.dataloader.dataset == "c4"
    assert config.validator.enable
    assert config.validator.freq == 500
    assert config.validator.steps == 1200
    assert config.validator.dataloader.dataset == "c4_validation"
