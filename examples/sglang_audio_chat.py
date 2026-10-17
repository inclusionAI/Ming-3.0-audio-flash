"""Send an audio request to a Ming model served by SGLang."""

from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
from typing import Any

import requests


ASR_PROMPT = (
    "Please recognize the language of this speech and transcribe it, placing a "
    "‘\\t’ between the identified language and the transcribe."
)
DEFAULT_PROMPT = "Answer the question or follow the instruction in the audio."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Input audio file")
    parser.add_argument("--task", choices=("chat", "asr"), default="chat")
    parser.add_argument("--prompt", help="Instruction placed after the audio")
    parser.add_argument("--system-prompt")
    parser.add_argument(
        "--base-url",
        default=os.getenv("MING_SGLANG_BASE_URL", "http://localhost:8008/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("MING_SGLANG_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("MING_SGLANG_MODEL"))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    return parser.parse_args()


def headers(api_key: str) -> dict[str, str]:
    result = {"Content-Type": "application/json"}
    if api_key:
        result["Authorization"] = f"Bearer {api_key}"
    return result


def resolve_model(base_url: str, api_key: str, timeout: float) -> str:
    response = requests.get(
        f"{base_url.rstrip('/')}/models",
        headers=headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    models = response.json().get("data", [])
    if not models:
        raise RuntimeError("SGLang returned no models from /v1/models")
    return str(models[0]["id"])


def audio_content(path: Path) -> dict[str, Any]:
    audio_format = path.suffix.lower().lstrip(".") or "wav"
    if audio_format == "wave":
        audio_format = "wav"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "input_audio",
        "input_audio": {"data": encoded, "format": audio_format},
    }


def main() -> None:
    args = parse_args()
    if not args.audio.is_file():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    base_url = args.base_url.rstrip("/")
    model = args.model or resolve_model(base_url, args.api_key, args.timeout)
    prompt = args.prompt or (ASR_PROMPT if args.task == "asr" else DEFAULT_PROMPT)

    messages: list[dict[str, Any]] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                audio_content(args.audio),
                {"type": "text", "text": prompt},
            ],
        }
    )

    response = requests.post(
        f"{base_url}/chat/completions",
        headers=headers(args.api_key),
        json={
            "model": model,
            "messages": messages,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        timeout=args.timeout,
    )
    response.raise_for_status()
    result = response.json()
    choice = result["choices"][0]
    print(choice["message"]["content"])
    if choice.get("finish_reason") == "length":
        print(
            "[warning] Output reached --max-tokens; increase it for a complete "
            "response.",
            flush=True,
        )


if __name__ == "__main__":
    main()
