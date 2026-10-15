"""Run Ming-3.0-audio-flash directly with Transformers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoProcessor


DEFAULT_SYSTEM_PROMPT = (
    "You are a voice assistant. Answer questions or execute commands based on "
    "the audio content. Return plain text without Markdown."
)
DEFAULT_PROMPT = "Answer the question or follow the instruction in the audio."
ASR_PROMPT = (
    "Please recognize the language of this speech and transcribe it, placing a ‘\\t’ "
    "between the identified language and the transcribe."
)
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
    parser.add_argument(
        "--task",
        choices=("chat", "asr"),
        default="chat",
        help="Use ASR-specific message formatting or general audio chat",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Instruction accompanying the audio",
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--prompt-as-system",
        action="store_true",
        help="Use --prompt as the system message and send only audio as user content",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=None,
        help="Prevent repeated n-grams; use 10 for validated ASR evaluation",
    )
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
    if args.prompt is None:
        args.prompt = ASR_PROMPT if args.task == "asr" else DEFAULT_PROMPT
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


def build_messages(args: argparse.Namespace, audio_path: str) -> list[dict]:
    audio_content = {"type": "audio", "audio": audio_path}
    if args.task == "asr":
        return [
            {
                "role": "user",
                "content": [
                    audio_content,
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]
    if args.prompt_as_system:
        return [
            {"role": "system", "content": args.prompt},
            {"role": "user", "content": [audio_content]},
        ]
    return [
        {"role": "system", "content": args.system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": args.prompt},
                audio_content,
            ],
        },
    ]


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

    messages = build_messages(args, str(args.audio.resolve()))

    inputs = process_messages(processor, messages, args.thinking).to(model.device)
    inputs["audio_feats"] = inputs["audio_feats"].to(model.dtype)
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
    }
    if args.no_repeat_ngram_size is not None:
        generation_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    response = processor.decode(generated_ids, skip_special_tokens=True).strip()
    print(response)


if __name__ == "__main__":
    main()
