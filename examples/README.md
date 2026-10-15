# Example recipes

The repository includes `audio/introduce_ant_group.wav` for quick validation.
Its reference transcript and technical metadata are documented in
[`audio/README.md`](audio/README.md). Verified model outputs will be added after
the final release checkpoint is fixed.

Run all commands from the repository root and set `MODEL_PATH` to the downloaded
or local model directory.

Start a text-output vLLM-Omni server before running these commands:

```bash
MING_OUTPUT_MODE=text ./scripts/serve.sh "$MODEL_PATH"
```

## Speech recognition

```bash
python examples/vllm_omni_audio_chat.py examples/audio/introduce_ant_group.wav \
  --prompt "Recognize the language and transcribe the speech with appropriate punctuation." \
  --prompt-as-system \
  --max-tokens 256 \
  --no-repeat-ngram-size 10
```

## Meeting transcription and summary

```bash
python examples/vllm_omni_audio_chat.py examples/audio/meeting.mp3 \
  --prompt "Transcribe this meeting, identify the speakers when possible, and summarize decisions and action items." \
  --max-tokens 256
```

## Thinking mode

```bash
python examples/vllm_omni_audio_chat.py examples/audio/meeting.mp3 \
  --prompt "Analyze the meeting and summarize the key decisions and action items." \
  --max-tokens 256 \
  --thinking
```

## Music understanding

```bash
python examples/vllm_omni_audio_chat.py examples/audio/music.mp3 \
  --prompt "Analyze the genre, instruments, rhythm, mood, and structure of this music."
```

## Spoken response

Restart the server with the full speech-output profile:

```bash
MING_OUTPUT_MODE=audio ./scripts/serve.sh "$MODEL_PATH"
python examples/vllm_omni_audio_chat.py examples/audio/introduce_ant_group.wav \
  --audio-response \
  --output-audio outputs/answer.wav
```

For streaming speech output and latency metrics:

```bash
python examples/vllm_omni_audio_streaming.py examples/audio/introduce_ant_group.wav \
  --voice-name DB30 \
  --output-audio outputs/stream_answer.wav
```

## Direct Transformers inference

Use a separate Transformers 5.14.1 environment with `requirements-transformers.txt` installed:

```bash
python examples/transformers_audio.py examples/audio/introduce_ant_group.wav \
  --model "$MODEL_PATH" \
  --task asr
python examples/transformers_audio.py examples/audio/introduce_ant_group.wav \
  --model "$MODEL_PATH" \
  --task qa \
  --prompt "Answer the question in the audio."
python examples/transformers_audio.py examples/audio/meeting.mp3 \
  --model "$MODEL_PATH" \
  --task qa \
  --prompt "Analyze the meeting and summarize the key decisions and action items." \
  --thinking
```
