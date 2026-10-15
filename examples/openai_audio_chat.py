"""Send an audio request to a Ming-3.0-audio-flash vLLM-Omni server."""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
from pathlib import Path

from openai import OpenAI


DEFAULT_SYSTEM_PROMPT = (
    "You are a voice assistant. Answer questions or execute commands based on "
    "the audio content. Return plain text without Markdown."
)


def audio_data_url(path: Path) -> str:
    media_type, _ = mimetypes.guess_type(path.name)
    if media_type is None or not media_type.startswith("audio/"):
        media_type = "audio/wav"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def resolve_model(client: OpenAI, requested_model: str | None) -> str:
    if requested_model:
        return requested_model
    models = client.models.list().data
    if not models:
        raise RuntimeError("The server returned no models; pass --model explicitly")
    return models[0].id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Input audio file")
    parser.add_argument(
        "--prompt",
        default="Answer the question or follow the instruction in the audio.",
        help="Instruction accompanying the audio",
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--prompt-as-system",
        action="store_true",
        help="Use --prompt as the system message and send only audio as user content",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("MING_BASE_URL", "http://localhost:8091/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("MING_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.getenv("MING_MODEL"))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=None,
        help="Prevent repeated n-grams; use 10 for validated ASR evaluation",
    )
    parser.add_argument(
        "--audio-response",
        action="store_true",
        help="Request both text and synthesized speech",
    )
    parser.add_argument("--voice-name", default="DB30")
    parser.add_argument(
        "--output-audio",
        type=Path,
        default=Path("outputs/answer.wav"),
    )
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"audio file does not exist: {args.audio}")
    return args


def main() -> None:
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    model = resolve_model(client, args.model)
    modalities = ["text", "audio"] if args.audio_response else ["text"]
    audio_content = {
        "type": "audio_url",
        "audio_url": {"url": audio_data_url(args.audio)},
    }
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

    request = {
        "model": model,
        "messages": messages,
        "modalities": modalities,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    extra_body = {}
    if args.no_repeat_ngram_size is not None:
        extra_body["sampling_params_list"] = [
            {"extra_args": {"no_repeat_ngram_size": args.no_repeat_ngram_size}}
        ]
    if args.audio_response:
        extra_body["additional_information"] = {"voice_name": args.voice_name}
    if extra_body:
        request["extra_body"] = extra_body

    response = client.chat.completions.create(**request)
    message = response.choices[0].message
    if message.content:
        print(message.content)

    if args.audio_response:
        if message.audio is None or not message.audio.data:
            raise RuntimeError("The server returned no audio; use the full deploy profile")
        args.output_audio.parent.mkdir(parents=True, exist_ok=True)
        args.output_audio.write_bytes(base64.b64decode(message.audio.data))
        print(f"Saved audio response to {args.output_audio}")


if __name__ == "__main__":
    main()
