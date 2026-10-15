"""Audio processor for BailingMoE V3 Audio Hugging Face inference."""

from pathlib import Path

import torch
import torchaudio
import soundfile as sf
import whisper
from transformers import AutoTokenizer, BatchFeature
from transformers.processing_utils import ProcessorMixin

from .configuration_bailing_moe_v3_audio import BailingMoeV3AudioConfig


class BailingMoeV3AudioProcessor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    ASR_PROMPT = (
        "Please recognize the language of this speech and transcribe it, "
        "placing a `\t` between the identified language and the transcribe."
    )

    def __init__(self, tokenizer, audio_config=None):
        super().__init__(tokenizer)
        self.audio_config = audio_config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        config = BailingMoeV3AudioConfig.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        return cls(tokenizer, config.audio_config)

    def _load_audio(self, audio):
        if isinstance(audio, (str, Path)):
            samples, sample_rate = sf.read(str(audio), dtype="float32", always_2d=True)
            waveform = torch.from_numpy(samples.T.copy())
        elif isinstance(audio, tuple) and len(audio) == 2:
            waveform, sample_rate = audio
            waveform = torch.as_tensor(waveform)
        else:
            raise TypeError("audio must be a path or a (waveform, sample_rate) tuple")
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError("waveform must have shape [samples] or [channels, samples]")
        waveform = waveform.float()
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        return waveform.mean(dim=0)

    def _num_audio_tokens(self, mel_length):
        config = self.audio_config
        chunk_size = getattr(config, "whisper_chunk_size", 3000)
        kernels = config.downsampler_kernel_size_list
        strides = config.downsampler_stride_list
        paddings = getattr(config, "downsampler_padding_list", None)
        paddings = paddings or [kernel // 2 for kernel in kernels]

        total = 0
        for offset in range(0, mel_length, chunk_size):
            length = min(chunk_size, mel_length - offset)
            length = (length + 2 - 3) // 2 + 1
            for kernel, stride, padding in zip(kernels, strides, paddings):
                length = (length + 2 * padding - kernel) // stride + 1
            total += length
        return total

    def __call__(self, audio, text=None, return_tensors="pt"):
        messages = None
        if isinstance(audio, list) and audio and isinstance(audio[0], dict):
            messages = audio
            audio_items = [
                item["audio"]
                for message in messages
                if isinstance(message.get("content"), list)
                for item in message["content"]
                if item.get("type") == "audio"
            ]
            if len(audio_items) != 1:
                raise ValueError("exactly one audio item is required")
            audio = audio_items[0]
        waveform = self._load_audio(audio)
        n_mels = getattr(self.audio_config, "n_mels", 128)
        mel = whisper.log_mel_spectrogram(waveform, n_mels=n_mels)
        mel = mel.transpose(0, 1).unsqueeze(0).contiguous()
        mel_length = mel.shape[1]

        audio_start = "<|audio_start|>"
        audio_token = "<|audio_pad|>"
        audio_end = "<|audio_end|>"
        if messages is None:
            prompt = (
                audio_start
                + audio_token * self._num_audio_tokens(mel_length)
                + audio_end
                + (text or self.ASR_PROMPT)
            )
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            placeholder = audio_start + audio_token + audio_end
            if prompt.count(placeholder) != 1:
                raise ValueError("chat prompt must contain exactly one audio placeholder")
            prompt = prompt.replace(
                placeholder,
                audio_start + audio_token * self._num_audio_tokens(mel_length) + audio_end,
            )
        inputs = self.tokenizer(prompt, return_tensors=return_tensors)
        inputs.pop("token_type_ids", None)
        inputs["audio_feats"] = mel
        inputs["audio_feats_lengths"] = torch.tensor([[mel_length]], dtype=torch.long)
        return BatchFeature(inputs)


__all__ = ["BailingMoeV3AudioProcessor"]
