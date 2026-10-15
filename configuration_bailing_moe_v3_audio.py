# Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BailingMoE V3 Audio model configuration."""

import os
from typing import Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class BailingMoeV3Config(PretrainedConfig):

    def __init__(
        self,
        vocab_size=157184,
        hidden_size=2048,
        intermediate_size=5120,
        num_hidden_layers=20,
        num_attention_heads=16,
        num_key_value_heads=4,
        hidden_act="silu",
        use_qkv_bias=False,  # bailing only
        use_bias=False,  # bailing only
        rms_norm_eps=1e-06,
        tie_word_embeddings=False,  # PretrainedConfig key, here change default value.
        embedding_dropout=0.0,
        attention_dropout=0.0,
        output_dropout=0.0,
        initializer_range=0.02,
        max_position_embeddings=32768,
        rope_theta=600000.0,
        use_cache=True,
        max_window_layers=20,
        rope_scaling=None,
        pad_token_id=156892,
        eos_token_id=156892,
        num_experts=256,
        num_shared_experts=1,
        num_experts_per_tok=8,
        n_group=8,
        topk_group=4,
        moe_intermediate_size=512,
        moe_shared_expert_intermediate_size=512,
        expert_swiglu_limit_list=None,
        share_expert_swiglu_limit_list=None,
        first_k_dense_replace=1,
        head_dim=128,
        output_router_logits=False,
        use_qk_norm=True,
        num_nextn_predict_layers=0,
        mtp_loss_scaling_factor=0,
        moe_router_enable_expert_bias=True,
        routed_scaling_factor=1.0,
        layer_group_size=5,
        kv_lora_rank=512,
        q_lora_rank=None,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        rope_interleave=True,
        score_function="sigmoid",
        scoring_func="sigmoid",
        seq_aux=True,
        topk_method="noaux_tc",
        router_dtype="fp32",
        gated_attention_proj_granularity_type=None,
        no_kda_lora=False,
        kda_safe_gate=False,
        kda_lower_bound=None,
        short_conv_kernel_size=4,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.use_qkv_bias = use_qkv_bias
        self.use_bias = use_bias
        self.rms_norm_eps = rms_norm_eps
        self.embedding_dropout = embedding_dropout
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_loss_scaling_factor = mtp_loss_scaling_factor
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        self.max_window_layers = max_window_layers
        self.head_dim = head_dim or self.hidden_size // self.num_attention_heads
        self.rope_scaling = rope_scaling
        self.use_qk_norm = use_qk_norm
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.routed_scaling_factor = routed_scaling_factor

        # MoE configs
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        self.expert_swiglu_limit_list = expert_swiglu_limit_list
        self.share_expert_swiglu_limit_list = share_expert_swiglu_limit_list
        self.first_k_dense_replace = first_k_dense_replace
        self.output_router_logits = output_router_logits

        # Linear configs
        self.layer_group_size = layer_group_size
        # mla
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim

        self.score_function = score_function
        self.scoring_func = scoring_func
        self.seq_aux = seq_aux
        self.topk_method = topk_method
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.rope_interleave = rope_interleave
        self.router_dtype = router_dtype
        self.gated_attention_proj_granularity_type = gated_attention_proj_granularity_type
        self.no_kda_lora = no_kda_lora
        self.kda_safe_gate = kda_safe_gate
        self.kda_lower_bound = kda_lower_bound
        self.short_conv_kernel_size = short_conv_kernel_size
        super().__init__(
            pad_token_id=pad_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )


class BailingMoeV3AudioEncoderConfig(PretrainedConfig):
    model_type = "bailing_moe_v3_audio_encoder"

    def __init__(
        self,
        n_mels=128,
        n_audio_ctx=3000,
        n_audio_state=1280,
        n_audio_head=20,
        n_audio_layer=32,
        downsampler_kernel_size_list=None,
        downsampler_stride_list=None,
        downsampler_padding_list=None,
        whisper_chunk_size=3000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_mels = n_mels
        self.n_audio_ctx = n_audio_ctx
        self.n_audio_state = n_audio_state
        self.n_audio_head = n_audio_head
        self.n_audio_layer = n_audio_layer
        self.downsampler_kernel_size_list = downsampler_kernel_size_list or [3]
        self.downsampler_stride_list = downsampler_stride_list or [2]
        self.downsampler_padding_list = downsampler_padding_list or [
            kernel // 2 for kernel in self.downsampler_kernel_size_list
        ]
        self.whisper_chunk_size = whisper_chunk_size

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if "audio_config" in config_dict:
            config_dict = config_dict["audio_config"]
        return cls.from_dict(config_dict, **kwargs)


class BailingMoeV3AudioConfig(PretrainedConfig):
    model_type = "bailing_moe_v3_audio"

    def __init__(
        self,
        text_config=None,
        audio_config=None,
        audio_token_id=157168,
        norm_query_embeds=False,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config = BailingMoeV3Config(**text_config)
        elif text_config is None:
            text_config = BailingMoeV3Config()

        if isinstance(audio_config, dict):
            audio_config = BailingMoeV3AudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = BailingMoeV3AudioEncoderConfig()

        self.text_config = text_config
        self.audio_config = audio_config
        self.audio_token_id = audio_token_id
        self.norm_query_embeds = norm_query_embeds

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "BailingMoeV3AudioConfig":
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )
        config_dict["model_type"] = cls.model_type
        return cls.from_dict(config_dict, **kwargs)
