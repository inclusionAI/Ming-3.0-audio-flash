"""Run Ming-3.0-audio-flash directly with Transformers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoProcessor


ASR_PROMPT = (
    "Please recognize the language of this speech and transcribe it with "
    "appropriate punctuation, placing a tab between the identified language "
    "and the transcription."
)
QA_PROMPT = "Answer the question in the audio. Return plain text without Markdown."
DEFAULT_MODEL = os.getenv("MING_MODEL", "inclusionAI/Ming-3.0-audio-flash")


def resolve_model_path(model: str) -> Path:
    local_path = Path(model).expanduser()
    if local_path.is_dir():
        return local_path.resolve()
    return Path(snapshot_download(repo_id=model)).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Input audio file")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Local model directory or Hugging Face model ID",
    )
    parser.add_argument("--task", choices=("asr", "qa"), default="qa")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Override the task-specific instruction",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable detailed thinking through the model chat template",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default=None,
    )
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"audio file does not exist: {args.audio}")
    return args


def process_messages(processor, messages, enable_thinking: bool):
    if not enable_thinking:
        return processor(messages)

    apply_chat_template = processor.tokenizer.apply_chat_template

    def apply_thinking_template(*args, **kwargs):
        kwargs["enable_thinking"] = True
        return apply_chat_template(*args, **kwargs)

    processor.tokenizer.apply_chat_template = apply_thinking_template
    try:
        return processor(messages)
    finally:
        processor.tokenizer.apply_chat_template = apply_chat_template


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model)

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    model_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs,
    ).eval()

    audio_path = str(args.audio.resolve())
    if args.task == "asr":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt or ASR_PROMPT},
                    {"type": "audio", "audio": audio_path},
                ],
            }
        ]
    else:
        messages = [
            {"role": "system", "content": args.prompt or QA_PROMPT},
            {
                "role": "user",
                "content": [{"type": "audio", "audio": audio_path}],
            },
        ]

    inputs = process_messages(processor, messages, args.thinking).to(model.device)
    inputs["audio_feats"] = inputs["audio_feats"].to(model.dtype)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    response = processor.decode(generated_ids, skip_special_tokens=True).strip()
    print(response)


if __name__ == "__main__":
    main()
