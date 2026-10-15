"""Inference entry point for the talker weights bundled with this checkpoint."""

from pathlib import Path

import torch
import torchaudio

from .AudioVAE.modeling_audio_vae import AudioVAE
from .modeling_bailing_talker import BailingTalker2


class Talker:
    def __init__(self, model_path=None, device="cuda:0", stream=False):
        model_path = Path(model_path or Path(__file__).resolve().parent.parent).resolve()
        talker_path = model_path / "talker"
        if not talker_path.is_dir():
            raise FileNotFoundError(f"Talker directory not found: {talker_path}")
        self.device = torch.device(device)
        self.model = BailingTalker2.from_pretrained(
            talker_path, dtype=torch.bfloat16
        ).eval().to(self.device)
        self.model.use_vllm = False
        self.vae = AudioVAE.from_pretrained(
            talker_path / "vae", dtype=torch.bfloat16
        ).eval().to(self.device)
        self.stream = stream

    @torch.inference_mode()
    def generate(self, text, voice_name="DB30", output_wav_path=None):
        chunks = [
            speech for speech, _, _, _ in self.model.omni_audio_generation(
                prompt="Please generate speech based on the following description.\n",
                tts_text=text,
                voice_name=voice_name,
                max_length=50,
                audio_detokenizer=self.vae,
                stream=self.stream,
            )
        ]
        if not chunks:
            raise RuntimeError("Talker produced no audio")
        waveform = torch.cat(chunks, dim=-1)
        if output_wav_path is not None:
            output_path = Path(output_wav_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(output_path), waveform.cpu(), self.vae.config.sample_rate)
        return waveform, self.vae.config.sample_rate
