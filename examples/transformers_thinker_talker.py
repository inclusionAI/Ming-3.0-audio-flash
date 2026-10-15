"""Run the Transformers Thinker -> native Ming Talker pipeline."""

from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor


DEFAULT_MODEL = os.getenv("MING_MODEL", "inclusionAI/Ming-3.0-audio-flash")
DEFAULT_SYSTEM_PROMPT = (
    "You are a voice assistant. Answer questions or execute commands based on "
    "the audio content. Return plain text without Markdown."
)
DEFAULT_AUDIO_PROMPT = "Answer the question or follow the instruction in the audio."
TALKER_DEPENDENCIES = (
    "onnxruntime",
    "x_transformers",
    "diffusers",
    "inflect",
    "pynini",
    "fasttext",
)


def resolve_model_path(model: str) -> Path:
    local_path = Path(model).expanduser()
    if local_path.is_dir():
        return local_path.resolve()
    return Path(snapshot_download(repo_id=model)).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Local model directory or Hugging Face model ID",
    )
    parser.add_argument(
        "--talker-model",
        type=Path,
        help="Talker directory; defaults to <model>/talker",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="User text sent to the Thinker")
    source.add_argument("--audio", type=Path, help="User audio sent to the Thinker")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_AUDIO_PROMPT,
        help="Instruction accompanying --audio",
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--prompt-as-system",
        action="store_true",
        help="Use --prompt as the system message and send only audio as user content",
    )
    voice = parser.add_mutually_exclusive_group()
    voice.add_argument("--ip-name", help="IP voice, for example 灵小柔")
    voice.add_argument("--ref-audio", type=Path, help="Reference WAV for voice cloning")
    voice.add_argument("--ref-audio-base64", help="Base64-encoded reference WAV")
    voice.add_argument(
        "--ref-audio-base64-file",
        type=Path,
        help="File containing a base64-encoded reference WAV",
    )
    parser.add_argument(
        "--prompt-text",
        help="Transcript of the reference audio; enables continuation mode",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/answer.wav"))
    parser.add_argument("--thinker-device-map", default="auto")
    parser.add_argument(
        "--thinker-max-memory",
        default="40GiB",
        help="Per-GPU limit when --thinker-device-map=auto",
    )
    parser.add_argument("--talker-device", help="Defaults to the last visible GPU")
    parser.add_argument("--max-thinker-tokens", type=int, default=256)
    parser.add_argument("--max-talker-steps", type=int, default=1000)
    parser.add_argument(
        "--language",
        help="Talker output language; detected from generated text by default",
    )
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.audio is not None and not args.audio.is_file():
        parser.error(f"audio file does not exist: {args.audio}")
    if args.ref_audio is not None and not args.ref_audio.is_file():
        parser.error(f"reference audio file does not exist: {args.ref_audio}")
    if args.ref_audio_base64_file is not None and not args.ref_audio_base64_file.is_file():
        parser.error(
            f"base64 reference audio file does not exist: {args.ref_audio_base64_file}"
        )
    if args.prompt_text and not any(
        (args.ref_audio, args.ref_audio_base64, args.ref_audio_base64_file)
    ):
        parser.error("--prompt-text requires reference audio")
    return args


def load_reference_audio(args: argparse.Namespace) -> dict | None:
    if args.ref_audio:
        raw = args.ref_audio.read_bytes()
    elif args.ref_audio_base64 or args.ref_audio_base64_file:
        encoded = (
            args.ref_audio_base64
            or args.ref_audio_base64_file.read_text(encoding="utf-8").strip()
        )
        if encoded.startswith("data:"):
            encoded = encoded.partition(",")[2]
        raw = base64.b64decode(encoded)
    else:
        return None
    samples, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    return {"samples": np.mean(samples, axis=1), "sample_rate": sample_rate}


def build_thinker_inputs(processor, args: argparse.Namespace):
    if args.audio:
        audio_content = {"type": "audio", "audio": str(args.audio.resolve())}
        if args.prompt_as_system:
            messages = [
                {"role": "system", "content": args.prompt},
                {"role": "user", "content": [audio_content]},
            ]
        else:
            messages = [
                {"role": "system", "content": args.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": args.prompt},
                        audio_content,
                    ],
                },
            ]
        return processor(messages)
    prompt = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": args.text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor.tokenizer(prompt, return_tensors="pt")
    inputs.pop("token_type_ids", None)
    return inputs


def load_talker_class(talker_path: Path):
    missing = [
        dependency
        for dependency in TALKER_DEPENDENCIES
        if importlib.util.find_spec(dependency) is None
    ]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            f"Missing native Talker dependencies: {names}. "
            "Install requirements-transformers.txt before running this example."
        )
    sys.path.insert(0, str(talker_path.parent))
    from talker import MingTalker

    return MingTalker


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model)
    talker_path = (args.talker_model or model_path / "talker").expanduser().resolve()
    if not talker_path.is_dir():
        raise FileNotFoundError(f"Talker directory does not exist: {talker_path}")
    MingTalker = load_talker_class(talker_path)

    reference = load_reference_audio(args)
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("Thinker and Talker inference requires at least one CUDA GPU")
    talker_device = args.talker_device or f"cuda:{device_count - 1}"

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    thinker_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    max_memory = {index: args.thinker_max_memory for index in range(device_count)}
    thinker = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=thinker_config,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=args.thinker_device_map,
        max_memory=max_memory if args.thinker_device_map == "auto" else None,
        attn_implementation="eager",
    ).eval()
    inputs = build_thinker_inputs(processor, args).to(thinker.device)
    if "audio_feats" in inputs:
        inputs["audio_feats"] = inputs["audio_feats"].to(thinker.dtype)
    with torch.inference_mode():
        output = thinker.generate(
            **inputs,
            max_new_tokens=args.max_thinker_tokens,
            do_sample=False,
        )
    generated = processor.decode(
        output[0, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()
    print(f"Thinker: {generated}", flush=True)

    talker = MingTalker.from_pretrained(
        talker_path,
        dtype=torch.bfloat16,
        device=talker_device,
    )
    audio = talker.synthesize(
        generated,
        ip_name=args.ip_name or "灵小柔",
        reference=reference,
        prompt_text=args.prompt_text,
        language=args.language,
        max_steps=args.max_talker_steps,
        seed=args.seed,
    )
    waveform = audio["audio"].squeeze().numpy()
    sample_rate = int(audio["sr"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, waveform, sample_rate)
    print(
        f"Saved {args.output} ({len(waveform) / sample_rate:.2f}s, {sample_rate} Hz)",
        flush=True,
    )


if __name__ == "__main__":
    main()
