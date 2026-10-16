# Example recipes

The repository includes samples for speech recognition, meeting understanding,
music understanding, and tool calling. Their metadata is documented in
[`audio/README.md`](audio/README.md).

Run all commands from the repository root and set `MODEL_PATH` to the downloaded
or local model directory.

Start the dedicated ASR profile for speech recognition:

```bash
MING_OUTPUT_MODE=asr ./scripts/serve.sh "$MODEL_PATH"
```

For the remaining text-output examples, start the general text profile instead:

```bash
MING_OUTPUT_MODE=text ./scripts/serve.sh "$MODEL_PATH"
```

## Speech recognition

```bash
python examples/vllm_omni_audio_chat.py examples/audio/introduce_ant_group.wav \
  --task asr \
  --prompt "Please recognize the language of this speech and transcribe it, placing a ‘\t’ between the identified language and the transcribe." \
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
MING_OUTPUT_MODE=audio ./scripts/serve.sh "$MODEL_PATH"
python examples/vllm_omni_audio_chat.py examples/audio/introduce_ant_group.wav \
  --audio-response \
  --output-audio outputs/answer.wav
```

For streaming speech output and latency metrics:

```bash
python examples/vllm_omni_audio_streaming.py examples/audio/introduce_ant_group.wav \
  --output-audio outputs/stream_answer.wav
```

The Talker automatically selects a built-in IP voice for the response language.
The English IP voice is `US_Stella`. To use a cloned voice instead:

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
  --prompt "Please recognize the language of this speech and transcribe it, placing a ‘\t’ between the identified language and the transcribe." \
  --max-new-tokens 256 \
  --no-repeat-ngram-size 10
python examples/transformers_audio.py examples/audio/introduce_ant_group.wav \
  --model "$MODEL_PATH" \
  --prompt "Answer the question or follow the instruction in the audio."
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
  --prompt "Answer the question or follow the instruction in the audio." \
  --ip-name US_Stella \
  --thinker-max-memory 72GiB \
  --output outputs/answer.wav
```
