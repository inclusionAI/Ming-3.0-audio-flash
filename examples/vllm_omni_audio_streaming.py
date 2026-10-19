"""Stream text and synthesized speech from a Ming vLLM-Omni server."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import statistics
import time
import wave
from pathlib import Path

import requests


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant.\n"
    "Listen to the user's audio and respond to their question or request.\n"
    "Answer the question or carry out the requested task. For lists, provide "
    "several concrete examples; for why/how questions, explain the main reason "
    "or method. Follow the user's explicit instructions, including any "
    "requirements for format, wording, and length. Answer directly and keep "
    "your response concise but complete. Do not repeat or transcribe the "
    "request unless asked to do so. Before responding, silently check that "
    "your answer satisfies the requested constraints."
)


def talker_options(args: argparse.Namespace) -> dict[str, str]:
    if args.ref_audio is None:
        return {}
    return {
        "voice_mode": "reference",
        "ref_audio": base64.b64encode(args.ref_audio.read_bytes()).decode("ascii"),
        "prompt_text": args.prompt_text,
    }


def output_audio_options(
    args: argparse.Namespace, audio_format: str
) -> dict[str, object]:
    options: dict[str, object] = {"format": audio_format}
    if args.voice is not None:
        options["voice"] = {"id": args.voice}
    return options


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Input audio file")
    parser.add_argument(
        "--base-url",
        default=os.getenv("MING_BASE_URL", "http://localhost:8091/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("MING_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.getenv("MING_MODEL"))
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
    )
    voice = parser.add_mutually_exclusive_group()
    voice.add_argument(
        "--voice",
        help="IP voice name or registered voice ID; defaults to the server's 灵小柔 voice",
    )
    voice.add_argument(
        "--ref-audio",
        type=Path,
        help="Reference WAV for voice cloning; omit it to use an IP or registered voice",
    )
    parser.add_argument(
        "--prompt-text",
        help="Transcript of --ref-audio, required for voice cloning",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--output-audio", type=Path, default=Path("outputs/stream_answer.wav")
    )
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"audio file does not exist: {args.audio}")
    if args.ref_audio is not None and not args.ref_audio.is_file():
        parser.error(f"reference audio file does not exist: {args.ref_audio}")
    if args.ref_audio is not None and not args.prompt_text:
        parser.error("--prompt-text is required with --ref-audio")
    if args.prompt_text and args.ref_audio is None:
        parser.error("--prompt-text requires --ref-audio")
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    return args


def api_url(base_url: str, endpoint: str) -> str:
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return f"{base_url}/{endpoint.lstrip('/')}"


def resolve_model(
    session: requests.Session, base_url: str, requested_model: str | None
) -> str:
    if requested_model:
        return requested_model
    response = session.get(api_url(base_url, "models"), timeout=30)
    response.raise_for_status()
    models = response.json().get("data", [])
    if not models:
        raise RuntimeError("The server returned no models; pass --model explicitly")
    return models[0]["id"]


def stream_once(
    session: requests.Session,
    args: argparse.Namespace,
    model: str,
    audio_data_url: str,
) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": args.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": audio_data_url}}
                ],
            },
        ],
        "modalities": ["text", "audio"],
        "audio": output_audio_options(args, "pcm"),
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "stream": True,
    }
    controls = talker_options(args)
    if controls:
        payload["additional_information"] = controls
    start = time.perf_counter()
    response = session.post(
        api_url(args.base_url, "chat/completions"),
        json=payload,
        stream=True,
        timeout=(30, 600),
    )
    response.raise_for_status()

    pcm_parts = []
    arrivals = []
    text_parts = []
    sample_rate = 44100
    channels = 1
    metrics = {"ttft_ms": None, "ttfa_ms": None, "done_ms": None, "finish_reason": None}

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data:"):
            continue
        elapsed_ms = (time.perf_counter() - start) * 1000
        data = raw_line.removeprefix("data:").strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if chunk.get("error"):
            raise RuntimeError(f"server returned error: {chunk['error']}")
        for choice in chunk.get("choices", []):
            if choice.get("finish_reason"):
                metrics["finish_reason"] = choice["finish_reason"]
                metrics["done_ms"] = elapsed_ms
            metadata = choice.get("audio_metadata")
            content = (choice.get("delta") or {}).get("content")
            if metadata is not None:
                sample_rate = metadata.get("sample_rate_hz") or sample_rate
                channels = metadata.get("channels") or channels
                if content:
                    pcm_parts.append(base64.b64decode(content))
                    arrivals.append(elapsed_ms)
                    if metrics["ttfa_ms"] is None:
                        metrics["ttfa_ms"] = elapsed_ms
            elif isinstance(content, str) and content:
                if metrics["ttft_ms"] is None:
                    metrics["ttft_ms"] = elapsed_ms
                text_parts.append(content)
                print(content, end="", flush=True)
    print()

    if metrics["done_ms"] is None:
        metrics["done_ms"] = (time.perf_counter() - start) * 1000
        metrics["finish_reason"] = metrics["finish_reason"] or "stream ended"
    audio = b"".join(pcm_parts)
    if not audio:
        raise RuntimeError("The server returned no audio; use the full deploy profile")
    args.output_audio.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output_audio), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(audio)
    metrics["audio_seconds"] = len(audio) / (2 * channels * sample_rate)
    metrics["arrivals"] = arrivals
    metrics["text"] = "".join(text_parts)
    return metrics


def print_metrics(metrics: dict, output_audio: Path) -> None:
    ttft = f"{metrics['ttft_ms']:.0f} ms" if metrics["ttft_ms"] is not None else "-"
    ttfa = f"{metrics['ttfa_ms']:.0f} ms" if metrics["ttfa_ms"] is not None else "-"
    print(f"TTFT: {ttft}")
    print(f"TTFA: {ttfa}")
    print(f"Total: {metrics['done_ms']:.0f} ms ({metrics['finish_reason']})")
    arrivals = metrics["arrivals"]
    if len(arrivals) >= 2:
        gaps = [end - start for start, end in zip(arrivals, arrivals[1:])]
        print(
            "Audio chunk gaps: "
            f"min={min(gaps):.0f} p50={statistics.median(gaps):.0f} "
            f"mean={statistics.mean(gaps):.0f} max={max(gaps):.0f} ms"
        )
    if metrics["audio_seconds"] > 0:
        rtf = metrics["done_ms"] / 1000 / metrics["audio_seconds"]
        print(f"Audio: {metrics['audio_seconds']:.2f}s, RTF={rtf:.2f}")
        print(f"Saved audio response to {output_audio}")
    else:
        print("Audio: no audio received")


def print_aggregate(all_runs: list[dict]) -> None:
    print("Aggregate:")
    for name, key in (("TTFT", "ttft_ms"), ("TTFA", "ttfa_ms"), ("Total", "done_ms")):
        values = [metrics[key] for metrics in all_runs if metrics[key] is not None]
        if values:
            print(
                f"  {name}: mean={statistics.mean(values):.0f} "
                f"p50={statistics.median(values):.0f} "
                f"min={min(values):.0f} max={max(values):.0f} ms"
            )


def main() -> None:
    args = parse_args()
    media_type, _ = mimetypes.guess_type(args.audio.name)
    if media_type is None or not media_type.startswith("audio/"):
        media_type = "audio/wav"
    encoded_audio = base64.b64encode(args.audio.read_bytes()).decode("ascii")
    audio_data_url = f"data:{media_type};base64,{encoded_audio}"
    headers = {"Authorization": f"Bearer {args.api_key}"}
    with requests.Session() as session:
        session.headers.update(headers)
        model = resolve_model(session, args.base_url, args.model)
        all_runs = []
        for run_index in range(args.runs):
            print(f"Run {run_index + 1}/{args.runs}")
            metrics = stream_once(session, args, model, audio_data_url)
            print_metrics(metrics, args.output_audio)
            all_runs.append(metrics)
        if len(all_runs) > 1:
            print_aggregate(all_runs)


if __name__ == "__main__":
    main()
