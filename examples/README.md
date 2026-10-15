# Example recipes

The repository includes `audio/introduce_ant_group.wav` for quick validation.
Its reference transcript and technical metadata are documented in
[`audio/README.md`](audio/README.md). Verified model outputs will be added after
the final release checkpoint is fixed.

Start a text-output vLLM-Omni server before running these commands:

```bash
MING_OUTPUT_MODE=text ./scripts/serve.sh "$MODEL_PATH"
```

## Speech recognition

```bash
python examples/openai_audio_chat.py examples/audio/introduce_ant_group.wav \
  --prompt "Recognize the language and transcribe the speech with appropriate punctuation."
```

## Meeting transcription and summary

```bash
python examples/openai_audio_chat.py meeting.wav \
  --prompt "Transcribe this meeting, identify the speakers when possible, and summarize decisions and action items."
```

## Audio captioning

```bash
python examples/openai_audio_chat.py scene.wav \
  --prompt "Describe the speech, background sounds, and acoustic environment in this recording."
```

## Music understanding

```bash
python examples/openai_audio_chat.py music.wav \
  --prompt "Analyze the genre, instruments, rhythm, mood, and structure of this music."
```

## Audio reasoning

```bash
python examples/openai_audio_chat.py reasoning.wav \
  --prompt "Answer the question in the audio. Use the audio evidence to reason, but output only the final answer."
```

## Spoken response

Restart the server with the full speech-output profile:

```bash
MING_OUTPUT_MODE=audio ./scripts/serve.sh "$MODEL_PATH"
python examples/openai_audio_chat.py spoken_question.wav \
  --audio-response \
  --output-audio outputs/answer.wav
```

## Direct Transformers inference

Use a separate environment with `requirements-hf.txt` installed:

```bash
python examples/transformers_audio.py examples/audio/introduce_ant_group.wav --task asr
python examples/transformers_audio.py spoken_question.wav \
  --task qa \
  --prompt "Answer the question in the audio."
```
