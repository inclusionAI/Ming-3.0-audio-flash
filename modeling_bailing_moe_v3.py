# coding=utf-8
# Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch BailingMoE model."""

import math
import warnings
from typing import List, Optional, Tuple, Union, Callable
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
try:
    from transformers.utils.import_utils import is_torch_fx_available
except ImportError:
    def is_torch_fx_available():
        return False
from .configuration_bailing_moe_v3 import BailingMoeV3Config
from transformers.generation.utils import GenerationMixin
from dataclasses import dataclass
from transformers.utils import ModelOutput
from transformers import DynamicLayer
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.deprecation import deprecate_kwarg
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla
from fla.ops.simple_gla.chunk import chunk_simple_gla
from einops import rearrange, repeat


try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda, fused_recurrent_kda

    from fla.ops.utils.index import prepare_cu_seqlens_from_mask, prepare_lens_from_mask
    from fla.utils import tensor_cache
except ImportError:
    raise ImportError("Plese run `pip install -U fla-core`")

# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BailingMoeV3Config"


def roll_tensor(tensor, shifts=-1, dims=-1, fill_value=0):
    """Roll the tensor input along the given dimension(s).
    Inserted elements are set to be 0.0.
    """
    rolled_tensor = torch.roll(tensor, shifts=shifts, dims=dims)
    rolled_tensor.select(dims, shifts).fill_(fill_value)
    return rolled_tensor, rolled_tensor.sum()


@dataclass
class MoEV3CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs as well as Mixture of Expert's router hidden
    states terms, to train a MoE model.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        z_loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            z_loss for the sparse modules.
        aux_loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            aux_loss for the sparse modules.
        router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.
            Router logits of the encoder model, useful to compute the auxiliary loss and the z_loss for the sparse
            modules.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    z_loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    router_logits: Optional[tuple[torch.FloatTensor]] = None
    mtp_loss: Optional[torch.FloatTensor] = None
    mtp_logits: Optional[tuple[torch.FloatTensor, ...]] = None


class MoeV3ModelOutputWithPast(MoeModelOutputWithPast):

    def __init__(self, mtp_hidden_states=None, **kwargs):
        super().__init__(**kwargs)
        self.mtp_hidden_states = mtp_hidden_states


def index_first_axis(x, indices):
    other_shape = x.shape[1:]
    second_dim = other_shape.numel()
    return torch.gather(
        rearrange(x, "b ... -> b (...)"),
        0,
        repeat(indices, "z -> z d", d=second_dim),
    ).reshape(-1, *other_shape)


def index_put_first_axis(x, indices, first_axis_dim):
    y = torch.zeros(first_axis_dim, *x.shape[1:], device=x.device, dtype=x.dtype)
    y[indices] = x
    # y.scatter_(0, repeat(indices, 'z -> z d', d=x.shape[1]), x)
    return y


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.LongTensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    output = index_put_first_axis(hidden_states, indices, batch_size * seq_len)
    return rearrange(output, "(b s) ... -> b s ...", b=batch_size)


@tensor_cache
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.BailingMoeV3.modeling_BailingMoeV3._prepare_4d_attention_mask` is deprecated and will be removed in v4.37. Use `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask"
    )
    return _prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    warnings.warn(
        "Calling `transformers.models.BailingMoeV3.modeling_BailingMoeV3._make_causal_mask` is deprecated and will be removed in v4.37. Use `transformers.models.BailingMoeV3.modeling_BailingMoeV3.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape, dtype=dtype, device=device, past_key_values_length=past_key_values_length
    )


class BailingMoeV3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        BailingMoeV3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class BailingMoeV3GroupRMSNorm(nn.Module):
    def __init__(self, hidden_size, group_norm_size, eps=1e-6):
        """
        BailingMoeV3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.group_norm_size = group_norm_size
        assert hidden_size % group_norm_size == 0, "hidden_size must be divisible by group_norm_size"
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        input_shape = hidden_states.size()
        group_input_shape = input_shape[:-1] + (self.group_norm_size, input_shape[-1] // self.group_norm_size)
        hidden_states = hidden_states.view(group_input_shape)
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype).view(input_shape)


ALL_LAYERNORM_LAYERS.append(BailingMoeV3RMSNorm)


def _compute_default_rope_parameters(config, device=None, seq_len=None):
    # Transformers 5 no longer includes the unscaled initializer in its registry.
    dim = int(config.head_dim * getattr(config, "partial_rotary_factor", 1.0))
    inv_freq = 1.0 / (
        config.rope_theta ** (torch.arange(0, dim, 2, device=device).float() / dim)
    )
    return inv_freq, 1.0


class BailingMoeV3RotaryEmbedding(nn.Module):
    def __init__(self, config: BailingMoeV3Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = deepcopy(config)
        self.config.head_dim = config.qk_rope_head_dim
        self.config.partial_rotary_factor = 1.0
        self.rope_init_fn = (
            _compute_default_rope_parameters
            if self.rope_type == "default"
            else ROPE_INIT_FUNCTIONS[self.rope_type]
        )

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class BailingMoeV3MLP(nn.Module):
    def __init__(
        self,
        config: BailingMoeV3Config,
        intermediate_size: int,
        swiglu_limit: Optional[float] = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.swiglu_limit = swiglu_limit

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        if self.swiglu_limit is not None:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(gate * up)


class BailingMoeV3Gate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts

        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.num_experts, self.gating_dim)))
        self.routed_scaling_factor = config.routed_scaling_factor

        self.register_buffer("expert_bias", torch.zeros((self.num_experts)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def group_limited_topk(
        self,
        scores: torch.Tensor,
    ):
        num_tokens, _ = scores.size()
        # Organize the experts into groups
        group_scores = scores.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)

        # Mask the experts based on selection groups
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
            .reshape(num_tokens, -1)
        )

        masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
        probs, top_indices = torch.topk(masked_scores, k=self.top_k, dim=-1)

        return probs, top_indices

    def forward(self, hidden_states):
        # compute gating score
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))

        scores = torch.sigmoid(logits.float()).type_as(logits)

        scores_for_routing = scores + self.expert_bias
        _, topk_idx = self.group_limited_topk(scores_for_routing)

        scores = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)

        topk_weight = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.top_k > 1 else scores
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight, logits


class BailingMoeV3SparseMoeBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    @staticmethod
    def _get_swiglu_limit(limit_list, layer_idx):
        if limit_list is None or layer_idx < 0 or layer_idx >= len(limit_list):
            return None
        limit = limit_list[layer_idx]
        return None if limit == 0 else limit

    def __init__(self, config: BailingMoeV3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.expert_swiglu_limit = self._get_swiglu_limit(
            getattr(config, "expert_swiglu_limit_list", None), layer_idx
        )
        self.share_expert_swiglu_limit = self._get_swiglu_limit(
            getattr(config, "share_expert_swiglu_limit_list", None), layer_idx
        )
        self._setup_experts()
        self.gate = BailingMoeV3Gate(config)
        if config.num_shared_experts is not None:
            self.shared_experts = BailingMoeV3MLP(
                config=config,
                intermediate_size=config.moe_shared_expert_intermediate_size * config.num_shared_experts,
                swiglu_limit=self.share_expert_swiglu_limit,
            )

    def _setup_experts(self):
        self.experts = nn.ModuleList(
            [
                BailingMoeV3MLP(
                    config=self.config,
                    intermediate_size=self.config.moe_intermediate_size,
                    swiglu_limit=self.expert_swiglu_limit,
                )
                for _ in range(self.config.num_experts)
            ]
        )

    def forward(self, hidden_states):
        identity = hidden_states
        bsz, seq_len, h = hidden_states.shape
        topk_idx, topk_weight, router_logits = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.to(hidden_states.dtype).view(bsz, seq_len, h)
        else:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(bsz, seq_len, h)
        if self.config.num_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y, (router_logits.view(bsz, seq_len, -1), topk_idx.view(bsz, seq_len, -1))

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()
        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out.to(x.device))
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int, head_first: bool = True) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). If head_first is True, the hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    if head_first:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    else:
        batch, slen, num_key_value_heads, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
        return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


def repeat_kv2(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv2(key, module.num_key_value_groups)
    value_states = repeat_kv2(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def apply_rotary_pos_emb_interleave(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    r"""
    TODO let's just use the original freqcis computation to not have the view
    transpose + reshape! This is not optimized!
    Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    def rotate_pairs(x):
        half_dim = x.shape[-1] // 2
        x_pairs = x.reshape(*x.shape[:-1], half_dim, 2)
        x_even, x_odd = x_pairs.unbind(dim=-1)
        pair_cos = cos[..., :half_dim]
        pair_sin = sin[..., :half_dim]
        return torch.stack(
            (
                x_even * pair_cos - x_odd * pair_sin,
                x_odd * pair_cos + x_even * pair_sin,
            ),
            dim=-1,
        ).flatten(-2)

    return rotate_pairs(q), rotate_pairs(k)


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class BailingMoeV3MultiLatentAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: BailingMoeV3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.gated_attention_proj_granularity_type = config.gated_attention_proj_granularity_type

        self.is_causal = True
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.use_qkv_bias)
            self.q_a_layernorm = BailingMoeV3RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.use_qkv_bias,
        )
        self.kv_a_layernorm = BailingMoeV3RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        if self.gated_attention_proj_granularity_type is None:
            self.g_proj = None
        elif self.gated_attention_proj_granularity_type == "head_wise":
            self.g_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        elif self.gated_attention_proj_granularity_type == "element_wise":
            self.g_proj = nn.Linear(config.hidden_size, self.num_heads * self.v_head_dim, bias=False)

        self.dense = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.use_qkv_bias,
        )

        self.scaling = self.qk_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            if mscale_all_dim:
                scaling_factor = self.config.rope_scaling["factor"]
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scaling = self.scaling * mscale * mscale

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:

        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings  # tptest
        if self.config.rope_interleave:  # support using interleaved weights for efficiency
            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            x = 1 / 0
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS.get(
                self.config._attn_implementation, eager_attention_forward
            )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        if self.g_proj is not None:
            gate = self.g_proj(hidden_states)
            gate = F.sigmoid(gate.float()).type_as(hidden_states)
            if self.gated_attention_proj_granularity_type == "head_wise":
                attn_output = attn_output * gate[:, :, :, None]
            else:
                attn_output = attn_output * gate.view(batch_size, seq_length, self.num_heads, self.v_head_dim)

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.dense(attn_output)
        return attn_output, attn_weights, past_key_values


class BailingMoeV3KimiDeltaAttention(nn.Module):
    def __init__(self, config: BailingMoeV3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.mode = "chunk"

        self.hidden_size = config.hidden_size
        self.conv_size = config.short_conv_kernel_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.head_k_dim = self.head_dim
        self.num_k_heads = self.num_heads
        self.no_kda_lora = config.no_kda_lora
        self.safe_gate = config.kda_safe_gate
        self.lower_bound = config.kda_lower_bound

        self.layer_idx = layer_idx

        assert self.mode in ['chunk', 'fused_recurrent'], f"Not suppoerted mode `{self.mode}`."

        projection_k_size = self.head_k_dim * self.num_k_heads
        projection_size = self.head_dim * self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

        self.q_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation='silu',
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation='silu',
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=projection_size,
            kernel_size=self.conv_size,
            activation='silu',
        )

        self.A_log = torch.nn.Parameter(torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)))

        if self.no_kda_lora:
            self.f_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        else:
            self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
            self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.dt_bias = nn.Parameter(torch.empty(projection_size, dtype=torch.float32))

        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        if self.no_kda_lora:
            self.g_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        else:
            self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.o_norm = FusedRMSNormGated(self.head_dim, eps=config.rms_norm_eps, activation='sigmoid')
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value=None,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        attention_mask = None
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                attention_mask = kwargs.get("padding_mask")

            if attention_mask is not None and attention_mask.dim() != 2:
                raise ValueError(
                    "attention_mask must be a 0-1 matrix of shape [batch_size, seq_len] "
                    "(0 = padding). 3D masks are not supported here.",
                )
        use_cache = past_key_value is not None
        batch_size, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if q_len <= 64 else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        cu_seqlens = kwargs.get('cu_seqlens')
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = _get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        conv_state_q, conv_state_k, conv_state_v = None, None, None
        recurrent_state = None

        if past_key_value is not None and isinstance(past_key_value, Cache):
            # ensure the cache list is long enough
            while len(past_key_value.layers) <= self.layer_idx:
                past_key_value.layers.append(DynamicLayer())

            if past_key_value.layers[self.layer_idx].keys is not None:
                recurrent_state = past_key_value.layers[self.layer_idx].keys
                # ensure recurrent_state is on the same device as hidden_states
                if recurrent_state.device != hidden_states.device:
                    recurrent_state = recurrent_state.to(hidden_states.device).contiguous()

            if past_key_value.layers[self.layer_idx].values is not None:
                conv_state_q, conv_state_k, conv_state_v = past_key_value.layers[self.layer_idx].values

        q, conv_state_q = self.q_conv1d(
            x=self.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        k, conv_state_k = self.k_conv1d(
            x=self.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        v, conv_state_v = self.v_conv1d(
            x=self.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )

        if self.no_kda_lora:
            g = self.f_proj(hidden_states)
        else:
            g = self.f_b_proj(self.f_a_proj(hidden_states))

        beta = self.b_proj(hidden_states).float().sigmoid()

        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)
        g = rearrange(g, '... (h d) -> ... h d', d=self.head_dim)

        if mode == 'chunk':
            o, recurrent_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                safe_gate=self.safe_gate,
                lower_bound=self.lower_bound,
                cu_seqlens=cu_seqlens,
            )
        else:
            o, recurrent_state = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                lower_bound=self.lower_bound,
                cu_seqlens=cu_seqlens,
            )

        if use_cache and past_key_value is not None and isinstance(past_key_value, Cache):
            target_device = None
            for cache in past_key_value.layers:
                if cache.keys is not None:
                    target_device = cache.keys.device
                    break
            if target_device is None:
                target_device = recurrent_state.device

            # move to target device
            if recurrent_state.device != target_device:
                recurrent_state = recurrent_state.to(target_device)

            past_key_value.layers[self.layer_idx].keys = recurrent_state
            past_key_value.layers[self.layer_idx].values = (conv_state_q, conv_state_k, conv_state_v)

        if self.no_kda_lora:
            g = self.g_proj(hidden_states)
        else:
            g = self.g_b_proj(self.g_a_proj(hidden_states))
        g = rearrange(g, '... (h d) -> ... h d', d=self.head_dim)
        o = self.o_norm(o, g)

        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_value


class BailingMoeV3MTPLayer(nn.Module):
    def __init__(self, config: BailingMoeV3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = BailingMoeV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.enorm = BailingMoeV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.post_attention_layernorm = BailingMoeV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = BailingMoeV3MultiLatentAttention(config=config, layer_idx=layer_idx)
        self.mlp = BailingMoeV3SparseMoeBlock(config, layer_idx=layer_idx)

        self.hnorm = BailingMoeV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.final_layernorm = BailingMoeV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_embeds,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        input_embeds = self.enorm(input_embeds)
        hidden_states = self.hnorm(hidden_states)
        hidden_states = self.eh_proj(torch.cat([input_embeds, hidden_states], dim=-1))
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        hidden_states = residual + hidden_states.to(residual.device)
        hidden_states = self.final_layernorm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class BailingMoeV3DecoderLayer(nn.Module):
    def __init__(self, config: BailingMoeV3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_layer_type = (
            "attention"
            if (layer_idx + 1) % config.layer_group_size == 0
            or layer_idx >= config.num_hidden_layers // config.layer_group_size * config.layer_group_size
            else "linear_attention"
        )

        if self.attention_layer_type == "attention":
            self.attention = BailingMoeV3MultiLatentAttention(config=config, layer_idx=layer_idx)
        else:
            self.attention = BailingMoeV3KimiDeltaAttention(config=config, layer_idx=layer_idx)

        self.mlp = (
            BailingMoeV3SparseMoeBlock(config, layer_idx=layer_idx)
            if (config.num_experts is not None and layer_idx >= config.first_k_dense_replace)
            else BailingMoeV3MLP(config=config, intermediate_size=config.intermediate_size)
        )
        self.input_layernorm = BailingMoeV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = BailingMoeV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*):
                cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        if self.attention_layer_type == "attention":
            hidden_states, self_attn_weights, present_key_value = self.attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_value,
                use_cache=use_cache,
                cache_position=cache_position,  #
                position_embeddings=position_embeddings,  #
                **kwargs,
            )
        else:
            batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
            device = hidden_states.device

            if attention_mask is None:
                # if attention_mask is None, create a full mask
                attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int32, device=device)
            elif attention_mask.dim() == 4 and attention_mask.shape[1] == 1:
                attention_mask = attention_mask[:, 0, -1, :].to(torch.int32)
                attention_mask = (attention_mask > -1e4).to(torch.int32)
            elif attention_mask.dim() == 2:
                attention_mask = attention_mask.to(torch.int32)
            else:
                raise ValueError(f"Unsupported mask dimension: {attention_mask.shape}")

            hidden_states, self_attn_weights, present_key_value = self.attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                position_ids=position_ids,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        hidden_states = residual + hidden_states.to(residual.device)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


BAILINGMOEV3_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`BailingMoeV3Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare BailingMoeV3 Model outputting raw hidden-states without any specific head on top.",
    BAILINGMOEV3_START_DOCSTRING,
)
class BailingMoeV3PreTrainedModel(PreTrainedModel):
    config_class = BailingMoeV3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BailingMoeV3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


BAILINGMOEV3_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).
            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.
            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.
            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.
            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.
            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare BailingMoeV3 Model outputting raw hidden-states without any specific head on top.",
    BAILINGMOEV3_START_DOCSTRING,
)
class BailingMoeV3Model(BailingMoeV3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`BailingMoeV3DecoderLayer`]
    Args:
        config: BailingMoeV3Config
    """

    def __init__(self, config: BailingMoeV3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.num_nextn_predict_layers = config.num_nextn_predict_layers

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = []
        for layer_idx in range(config.num_hidden_layers + config.num_nextn_predict_layers):
            layer_cls = BailingMoeV3DecoderLayer if layer_idx < config.num_hidden_layers else BailingMoeV3MTPLayer
            self.layers.append(layer_cls(config, layer_idx))

        self.layers = nn.ModuleList(self.layers)

        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = BailingMoeV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = BailingMoeV3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

    @add_start_docstrings_to_model_forward(BAILINGMOEV3_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, MoeV3ModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # Recurrent KDA states do not encode sequence length. Use the first
        # softmax attention cache for positions as well as the causal mask.
        softmax_attention_layer_id = min(self.config.layer_group_size - 1, self.config.num_hidden_layers - 1)
        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length(layer_idx=softmax_attention_layer_id)
                if past_key_values is not None else 0
            )
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        past_seen_tokens = (
            past_key_values.get_seq_length(layer_idx=softmax_attention_layer_id) if past_key_values is not None else 0
        )

        if position_ids is None:
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
            position_ids = position_ids.unsqueeze(0)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_seen_tokens,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_seen_tokens
            )

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None
        layers = self.layers[: -self.num_nextn_predict_layers] if self.num_nextn_predict_layers > 0 else self.layers
        mtp_layers = self.layers[-self.num_nextn_predict_layers :] if self.num_nextn_predict_layers > 0 else None

        # tptest miss causal_mask = create_causal_mask(

        for decoder_layer in layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    cache_position,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    cache_position=cache_position,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits and layer_outputs[-1] is not None:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)
        main_hidden_states = hidden_states

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (main_hidden_states,)

        mtp_hidden_states = None

        if mtp_layers and input_ids is not None:
            for decoder_layer in mtp_layers:
                input_ids, _ = roll_tensor(input_ids, shifts=-1, dims=-1)
                inputs_embeds = self.word_embeddings(input_ids)

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        inputs_embeds,
                        hidden_states,
                        attention_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        output_router_logits,
                        use_cache,
                        position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        inputs_embeds,
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        output_router_logits=output_router_logits,
                        use_cache=use_cache,
                        position_embeddings=position_embeddings,
                    )
                if mtp_hidden_states is None:
                    mtp_hidden_states = []
                hidden_states = layer_outputs[0]
                mtp_hidden_states.append(hidden_states)

                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

                if output_router_logits and layer_outputs[-1] is not None:
                    all_router_logits += (layer_outputs[-1],)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache
        if not return_dict:
            return tuple(
                v
                for v in [main_hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )
        return MoeV3ModelOutputWithPast(
            last_hidden_state=main_hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            mtp_hidden_states=mtp_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


class BailingMoeV3ForCausalLM(BailingMoeV3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.word_embeddings.weight"}

    def __init__(self, config: BailingMoeV3Config):
        super().__init__(config)
        self.model = BailingMoeV3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.num_nextn_predict_layers = config.num_nextn_predict_layers
        self.mtp_loss_scaling_factor = config.mtp_loss_scaling_factor

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.word_embeddings

    def set_input_embeddings(self, value):
        self.model.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(BAILINGMOEV3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MoEV3CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, MoEV3CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        Example:
        ```python
        >>> from transformers import AutoTokenizer
        >>> model = BailingMoeV3ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            **kwargs,
        )

        loss = None
        all_mtp_loss = None
        aux_loss = None
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if labels is not None:
            loss = self.loss_function(logits, labels, self.config.vocab_size, **kwargs)

        all_mtp_logits = None
        if self.num_nextn_predict_layers > 0:
            mtp_hidden_states = outputs.mtp_hidden_states
            shift_labels_mtp = None
            for i in range(self.num_nextn_predict_layers):
                mtp_hidden_states = mtp_hidden_states[i]
                mtp_logits = self.lm_head(mtp_hidden_states).float()
                if all_mtp_logits is None:
                    all_mtp_logits = []
                all_mtp_logits.append(mtp_logits)
                if labels is not None:
                    if shift_labels_mtp is None:
                        shift_labels_mtp = labels.clone()
                    shift_labels_mtp, _ = roll_tensor(shift_labels_mtp, shifts=-1, dims=-1, fill_value=-100)
                    mtp_logits_ = mtp_logits.view(-1, self.config.vocab_size)
                    mtp_loss = self.loss_function(
                        mtp_logits_, shift_labels_mtp.to(mtp_logits_.device).view(-1), self.config.vocab_size, **kwargs
                    )
                    if loss is not None:
                        loss += self.mtp_loss_scaling_factor * mtp_loss
                    else:
                        loss = self.mtp_loss_scaling_factor * mtp_loss

                    if all_mtp_loss is None:
                        all_mtp_loss = []
                    all_mtp_loss.append(mtp_loss)

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoEV3CausalLMOutputWithPast(
            loss=loss,
            mtp_loss=all_mtp_loss,
            aux_loss=aux_loss,
            logits=logits,
            mtp_logits=all_mtp_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )
