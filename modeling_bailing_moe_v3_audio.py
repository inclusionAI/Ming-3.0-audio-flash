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
"""PyTorch BailingMoE V3 Audio model."""

import os
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.cache_utils import Cache
from transformers.modeling_utils import PreTrainedModel
try:
    from transformers.utils.import_utils import is_torch_fx_available
except ImportError:
    def is_torch_fx_available():
        return False
from .configuration_bailing_moe_v3_audio import (
    BailingMoeV3AudioConfig,
    BailingMoeV3AudioEncoderConfig,
)
from transformers.generation.utils import GenerationMixin
from dataclasses import dataclass
from transformers.utils import ModelOutput
import whisper

# OpenAI Whisper casts LayerNorm inputs to float32. Hugging Face loads this
# encoder's parameters in the requested model dtype, so that implementation
# fails for BF16 weights. Keep inputs and parameters in the same dtype, as in
# the LingV3 Audio reference torch client.
whisper.model.LayerNorm.forward = lambda self, x: F.layer_norm(
    x, self.normalized_shape, self.weight, self.bias, self.eps
)



class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class WhisperEncoderForBailing(nn.Module):
    """Whisper encoder wrapper for audio encoding in BailingMM inference.

    Unlike the original Whisper encoder, this version does not perform 30-second padding
    and returns output lengths along with the encoded features.
    """

    def __init__(self, n_mels: int, n_audio_ctx: int, n_audio_state: int, n_audio_head: int, n_audio_layer: int):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_audio_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_audio_state, n_audio_state, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = nn.Parameter(torch.zeros(n_audio_ctx, n_audio_state))
        self.blocks = nn.ModuleList(
            [whisper.model.ResidualAttentionBlock(n_audio_state, n_audio_head) for _ in range(n_audio_layer)]
        )
        self.ln_post = nn.LayerNorm(n_audio_state)
        self.audio_emb_dim = n_audio_state

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor):
        """Forward pass.

        Args:
            x: Mel spectrogram, shape (B, T, d)
            input_lengths: Original lengths before mel, shape (B,)
        Returns:
            Tuple of (encoded features, output_lengths).
        """
        x = x.transpose(1, 2)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        positional_embedding = self.positional_embedding[: x.shape[1], :]
        assert x.shape[1:] == positional_embedding.shape, "incorrect audio shape"
        x = (x + positional_embedding).to(x.dtype)
        for block in self.blocks:
            x = block(x)
        x = self.ln_post(x)
        output_lengths = input_lengths
        output_lengths = (
            output_lengths + 2 * (self.conv1.kernel_size[0] // 2) - self.conv1.kernel_size[0]
        ) // self.conv1.stride[0] + 1
        output_lengths = (
            output_lengths + 2 * (self.conv2.kernel_size[0] // 2) - self.conv2.kernel_size[0]
        ) // self.conv2.stride[0] + 1
        return x, output_lengths

    @classmethod
    def from_pretrained(cls, audio_config: BailingMoeV3AudioEncoderConfig):
        return cls(
            n_mels=audio_config.n_mels,
            n_audio_ctx=audio_config.n_audio_ctx,
            n_audio_state=audio_config.n_audio_state,
            n_audio_head=audio_config.n_audio_head,
            n_audio_layer=audio_config.n_audio_layer,
        )


from .modeling_bailing_moe_v3 import BailingMoeV3Model


@dataclass
class BailingMoeV3AudioCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[tuple[torch.FloatTensor]] = None
    aux_loss: Optional[torch.FloatTensor] = None


class BailingMoeV3AudioPreTrainedModel(PreTrainedModel):
    config_class = BailingMoeV3AudioConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BailingMoeV3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range
        else:
            std = getattr(self.config.get_text_config(), "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class BailingMoeV3AudioForConditionalGeneration(BailingMoeV3AudioPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.word_embeddings.weight"}

    def __init__(self, config: BailingMoeV3AudioConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BailingMoeV3Model(config.text_config)
        # Audio encoder lives under self.model to match checkpoint key "model.audio.*"
        self.model.audio = WhisperEncoderForBailing.from_pretrained(config.audio_config)

        mlp_modules_audio = []
        audio_feature_size = config.audio_config.n_audio_state
        for kernel_size, stride in zip(
            config.audio_config.downsampler_kernel_size_list, config.audio_config.downsampler_stride_list
        ):
            mlp_modules_audio.append(torch.nn.Conv1d(
                audio_feature_size, config.text_config.hidden_size,
                kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
            ))
            mlp_modules_audio.append(nn.GELU())
            audio_feature_size = config.text_config.hidden_size
        mlp_modules_audio.append(Transpose(-1, -2))
        mlp_modules_audio.append(nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size))
        self.linear_proj_audio = nn.Sequential(*mlp_modules_audio)

        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _get_audio_features(self, audio_feats, audio_feats_lengths):
        chunk_size = getattr(self.config.audio_config, "whisper_chunk_size", 3000)
        if audio_feats.ndim != 3 or audio_feats.shape[0] != 1:
            raise ValueError("audio_feats must have shape [1, mel_frames, n_mels]")
        if audio_feats_lengths is None or audio_feats_lengths.ndim != 2:
            raise ValueError("audio_feats_lengths must have shape [1, num_audios]")
        if chunk_size <= 0:
            raise ValueError("whisper_chunk_size must be positive")
        valid_lengths = [int(length) for length in audio_feats_lengths[0].tolist() if int(length) > 0]
        if not valid_lengths:
            raise ValueError("audio_feats_lengths must contain at least one positive length")

        total_length = sum(valid_lengths)
        if audio_feats.shape[1] < total_length:
            raise ValueError(
                f"audio_feats length {audio_feats.shape[1]} is shorter than requested audio length {total_length}"
            )

        segments = audio_feats[0, :total_length].split(valid_lengths)
        chunks = []
        for segment in segments:
            chunks.extend(segment.split(chunk_size))

        input_lengths = torch.tensor([chunk.shape[0] for chunk in chunks], dtype=torch.long, device=audio_feats.device)
        chunk_batch = torch.nn.utils.rnn.pad_sequence(chunks, batch_first=True)

        # Limit Whisper activations for long recordings. An unset value keeps
        # the original single-batch path; each chunk has independent context.
        micro_batch = int(os.environ.get("HF_SFT_WHISPER_MICRO_BATCH", "0"))
        if micro_batch < 0:
            raise ValueError("HF_SFT_WHISPER_MICRO_BATCH must be nonnegative")
        micro_batch = micro_batch or len(chunks)
        projected_chunks = []
        for start in range(0, len(chunks), micro_batch):
            end = start + micro_batch
            encoded, enc_lengths = self.model.audio(
                chunk_batch[start:end], input_lengths[start:end]
            )[:2]
            projected = self.linear_proj_audio(encoded.transpose(-1, -2))
            proj_lens = enc_lengths
            for module in self.linear_proj_audio:
                if isinstance(module, nn.Conv1d):
                    proj_lens = (proj_lens + 2 * module.padding[0] - module.kernel_size[0]) // module.stride[0] + 1
            projected_chunks.extend(
                projected[i, :proj_lens[i]] for i in range(projected.shape[0])
            )
        audio_embeds = torch.cat(projected_chunks, dim=0).unsqueeze(0)
        if self.config.norm_query_embeds:
            audio_embeds = F.normalize(audio_embeds, dim=2)
        return audio_embeds

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None,
        inputs_embeds=None, cache_position=None, **kwargs
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        # Audio is injected only during prefill. Keeping it on decode steps
        # needlessly reruns the Whisper encoder for every generated token.
        is_decode_step = (
            cache_position is not None
            and cache_position.numel() > 0
            and cache_position[0].item() > 0
        )
        # Transformers 5 slices cached decoding with next_sequence_length
        # instead of passing cache_position to remote models.
        is_decode_step = is_decode_step or (
            past_key_values is not None
            and kwargs.get("next_sequence_length") is not None
            and not kwargs.get("is_first_iteration", False)
        )
        if is_decode_step:
            model_inputs.pop("audio_feats", None)
            model_inputs.pop("audio_feats_lengths", None)
        return model_inputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        audio_feats=None,
        audio_feats_lengths=None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if audio_feats is not None:
            audio_embeds = self._get_audio_features(audio_feats, audio_feats_lengths)
            audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            num_audio_tokens = int((input_ids == self.config.audio_token_id).sum())
            if num_audio_tokens != audio_embeds.shape[1]:
                raise ValueError(
                    "audio placeholder count does not match encoded features: "
                    f"{num_audio_tokens} != {audio_embeds.shape[1]}"
                )
            audio_mask = (input_ids == self.config.audio_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return BailingMoeV3AudioCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )
