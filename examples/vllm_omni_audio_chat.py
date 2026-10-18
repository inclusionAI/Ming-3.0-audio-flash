"""Send an audio request to a Ming-3.0-audio vLLM-Omni server."""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import sys
from pathlib import Path

from openai import OpenAI


DEFAULT_SYSTEM_PROMPT = (
    "You are a voice assistant. Answer questions or execute commands based on "
    "the audio content. Return plain text without Markdown."
)
DEFAULT_PROMPT = "Answer the question or follow the instruction in the audio."
ASR_PROMPT = (
    "Please recognize the language of this speech and transcribe it, placing a ‘\\t’ "
    "between the identified language and the transcribe."
)


def audio_data_url(path: Path) -> str:
    media_type, _ = mimetypes.guess_type(path.name)
    if media_type is None or not media_type.startswith("audio/"):
        media_type = "audio/wav"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def talker_options(args: argparse.Namespace) -> dict[str, str]:
    if args.ref_audio is None:
        options = {"voice_mode": "ip"}
        if args.ip_name:
            options["ip_name"] = args.ip_name
        return options
    return {
        "ref_audio": base64.b64encode(args.ref_audio.read_bytes()).decode("ascii"),
        "prompt_text": args.prompt_text,
    }


def request_extra_body(args: argparse.Namespace) -> dict[str, object]:
    extra_body: dict[str, object] = {
        "chat_template_kwargs": {"enable_thinking": args.thinking},
        "top_k": args.top_k,
    }
    if args.no_repeat_ngram_size is not None:
        extra_body["sampling_params_list"] = [
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_tokens": args.max_tokens,
                "extra_args": {"no_repeat_ngram_size": args.no_repeat_ngram_size},
            }
        ]
    if args.audio_response:
        extra_body["additional_information"] = talker_options(args)
    return extra_body


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
    parser.add_argument(
        "--base-url",
        default=os.getenv("MING_BASE_URL", "http://localhost:8091/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("MING_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.getenv("MING_MODEL"))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable detailed thinking through the model chat template",
    )
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
    voice = parser.add_mutually_exclusive_group()
    voice.add_argument(
        "--ip-name",
        help="Built-in voice name; omit it to select by response language",
    )
    voice.add_argument(
        "--ref-audio",
        type=Path,
        help="Reference WAV for voice cloning; omit it to use automatic IP voices",
    )
    parser.add_argument(
        "--prompt-text",
        help="Transcript of --ref-audio, required for voice cloning",
    )
    parser.add_argument(
        "--output-audio",
        type=Path,
        default=Path("outputs/answer.wav"),
    )
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"audio file does not exist: {args.audio}")
    if args.ref_audio is not None and not args.ref_audio.is_file():
        parser.error(f"reference audio file does not exist: {args.ref_audio}")
    if args.ref_audio is not None and not args.prompt_text:
        parser.error("--prompt-text is required with --ref-audio")
    if args.prompt_text and args.ref_audio is None:
        parser.error("--prompt-text requires --ref-audio")
    if args.task == "asr" and args.prompt_as_system:
        parser.error("--prompt-as-system cannot be used with --task asr")
    if args.prompt is None:
        args.prompt = ASR_PROMPT if args.task == "asr" else DEFAULT_PROMPT
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
    if args.task == "asr":
        messages = [
            {
                "role": "user",
                "content": [
                    audio_content,
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]
    elif args.prompt_as_system:
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
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    extra_body = request_extra_body(args)
    if extra_body:
        request["extra_body"] = extra_body

    response = client.chat.completions.create(**request)
    choice = response.choices[0]
    message = choice.message
    if message.content:
        print(message.content)
    if choice.finish_reason == "length":
        print(
            "Warning: response reached --max-tokens and may be incomplete.",
            file=sys.stderr,
        )

    if args.audio_response:
        if message.audio is None or not message.audio.data:
            raise RuntimeError(
                "The server returned no audio; use the full deploy profile"
            )
        args.output_audio.parent.mkdir(parents=True, exist_ok=True)
        args.output_audio.write_bytes(base64.b64decode(message.audio.data))
        print(f"Saved audio response to {args.output_audio}")


if __name__ == "__main__":
    main()
