# Example recipes

The repository includes samples for speech recognition, meeting understanding,
music understanding, and tool calling. Their metadata is documented in
[`audio/README.md`](audio/README.md).

Run all commands from the repository root and set `MODEL_PATH` to the downloaded
or local model directory.

The default Flash vLLM-Omni profiles use TP=4 across four GPUs for the 128K
context length. The full speech-output profile reuses GPU 3 for the Talker.
The four-GPU profiles may OOM on particularly long audio inputs; when that
happens, switch to the eight-GPU fallback profiles in `deploy/*_8gpu.yaml`
(TP=8 across GPUs 0–7; the Talker reuses GPU 7).

Start the dedicated ASR profile for speech recognition:

```bash
MING_OUTPUT_MODE=asr ./scripts/serve_vllm_omni.sh "$MODEL_PATH"
```

Flash and Tiny use the same inference client. To run Tiny ASR, point
`MODEL_PATH` to the Tiny checkpoint and select the Tiny server profile:

```bash
export MODEL_PATH=inclusionAI/Ming-3.0-audio-tiny
MING_MODEL_VARIANT=tiny MING_OUTPUT_MODE=asr ./scripts/serve_vllm_omni.sh "$MODEL_PATH"
```

For the remaining text-output examples, start the general text profile instead:

```bash
MING_OUTPUT_MODE=text ./scripts/serve_vllm_omni.sh "$MODEL_PATH"
```

## Speech recognition

```bash
python examples/vllm_omni_audio_chat.py examples/audio/introduce_ant_group.wav \
  --task asr \
  --prompt 'Please recognize the language of this speech and transcribe it, placing a `\t` between the identified language and the transcribe.' \
  --max-tokens 256 \
  --no-repeat-ngram-size 10
```

## Meeting transcription and summary

```bash
python examples/vllm_omni_audio_chat.py examples/audio/meeting.mp3 \
  --prompt "Transcribe this meeting, identify the speakers when possible, and summarize decisions and action items." \
  --max-tokens 32768
```

## Thinking mode

```bash
python examples/vllm_omni_audio_chat.py examples/audio/meeting.mp3 \
  --prompt "Analyze the meeting and summarize the key decisions and action items." \
  --max-tokens 32768 \
  --thinking
```

## Music understanding

```bash
python examples/vllm_omni_audio_chat.py examples/audio/music.mp3 \
  --prompt "Please transcribe the lyrics of this song and carefully describe its vocals, arrangement, instrumentation, style, genre, and other relevant aspects."
```

## Spoken response

Restart the server with the full speech-output profile:

```bash
MING_OUTPUT_MODE=audio ./scripts/serve_vllm_omni.sh "$MODEL_PATH"
python examples/vllm_omni_audio_chat.py examples/audio/introduce_ant_group.wav \
  --audio-response \
  --output-audio outputs/answer.wav
```

For streaming speech output and latency metrics:

```bash
python examples/vllm_omni_audio_streaming.py examples/audio/introduce_ant_group.wav \
  --output-audio outputs/stream_answer.wav
```

vLLM-Omni uses the built-in IP voice `灵小柔` by default. Pass `--voice` to
select another voice from the [complete built-in voice list](../docs/voices.md),
or a registered voice ID returned by `GET /v1/audio/voices`. Both clients send
Chat Completions `audio.voice: {"id": "..."}`. Voice selection cannot be combined
with `--ref-audio`.
The model directory also provides `灵楷` and `US_James` as preset reference
voices under `voice_presets/voice_audio/`, with matching transcripts in
`voice_presets/voice_prompts.json`. To use one of these presets, or another
reference voice, pass its WAV file and exact transcript:

```bash
python examples/vllm_omni_audio_chat.py examples/audio/introduce_ant_group.wav \
  --audio-response \
  --ref-audio path/to/reference.wav \
  --prompt-text "<exact transcript of the reference audio>" \
  --output-audio outputs/cloned_answer.wav
```

## Direct Transformers inference

Use a separate Transformers 5.14.1 environment with `requirements-transformers.txt` installed:

```bash
python examples/transformers_audio.py examples/audio/introduce_ant_group.wav \
  --model "$MODEL_PATH" \
  --task asr \
  --prompt 'Please recognize the language of this speech and transcribe it, placing a `\t` between the identified language and the transcribe.' \
  --max-new-tokens 256 \
  --no-repeat-ngram-size 10
python examples/transformers_audio.py examples/audio/introduce_ant_group.wav \
  --model "$MODEL_PATH"
python examples/transformers_audio.py examples/audio/meeting.mp3 \
  --model "$MODEL_PATH" \
  --prompt "Transcribe this meeting, identify the speakers when possible, and summarize decisions and action items." \
  --max-new-tokens 32768
python examples/transformers_audio.py examples/audio/meeting.mp3 \
  --model "$MODEL_PATH" \
  --prompt "Analyze the meeting and summarize the key decisions and action items." \
  --max-new-tokens 32768 \
  --thinking
python examples/transformers_audio.py examples/audio/music.mp3 \
  --model "$MODEL_PATH" \
  --prompt "Please transcribe the lyrics of this song and carefully describe its vocals, arrangement, instrumentation, style, genre, and other relevant aspects."
```

For end-to-end Thinker and native Talker inference:

```bash
python examples/transformers_thinker_talker.py \
  --model "$MODEL_PATH" \
  --audio examples/audio/introduce_ant_group.wav \
  --voice US_Stella \
  --thinker-max-memory 72GiB \
  --output outputs/answer.wav
```

## SGLang inference

Install and start the SGLang audio branch from the repository root:

```bash
python3 -m venv .venv-sglang
source .venv-sglang/bin/activate
pip install -r requirements-sglang.txt
./scripts/serve_sglang.sh "$MODEL_PATH"
```

For the adjacent `sglang_ling_v3` source checkout, replace the requirements
installation with `pip install -e ../sglang_ling_v3/python`.

Then send an ASR request to its OpenAI-compatible endpoint:

```bash
python examples/sglang_audio_chat.py examples/audio/introduce_ant_group.wav \
  --task asr \
  --max-tokens 256
```

SGLang currently provides text output for these examples; spoken output uses
the vLLM-Omni or direct Transformers paths above.
