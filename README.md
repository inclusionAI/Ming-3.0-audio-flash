# Ming-3.0-audio-flash

Ming-3.0-audio-flash is a native audio model for real-world speech scenarios. It supports speech recognition, long-audio understanding, speech and non-speech audio perception, audio reasoning, end-to-end spoken interaction, and tool-oriented workflows.

## Core capabilities

- **Complex speech recognition:** Chinese and English speech across dialects, accents, noisy environments, and multi-speaker meetings.
- **Long-audio understanding:** Audio comprehension for recordings up to hour-long durations.
- **Multi-dimensional perception:** Fine-grained understanding of speech, non-speech sounds, and music.
- **Audio reasoning:** Analysis and reasoning directly over audio content.
- **Audio interaction and execution:** End-to-end spoken dialogue and tool-oriented task execution.

This repository provides two inference paths:

- **vLLM-Omni** for an OpenAI-compatible serving API, including optional speech output.
- **Transformers** for direct, text-output inference from a local model directory or Hugging Face model ID.

## Repository layout

```text
configuration_bailing_moe_v3*.py
                         Model configuration
modeling_bailing_moe_v3*.py
                         Core model implementation
processing_bailing_moe_v3_audio.py
                         Audio processor implementation
talker/                  Talker and AudioVAE source code
cookbook.ipynb            Interactive inference walkthrough
deploy/                  vLLM-Omni deployment profiles
examples/
  audio/                  Bundled example audio and metadata
  README.md              Task-specific prompt recipes
  openai_audio_chat.py   OpenAI-compatible audio client
  transformers_audio.py  Direct Transformers inference
scripts/
  serve.sh               vLLM-Omni server launcher
requirements.txt         Client dependencies
requirements-hf.txt      Direct Transformers dependencies
requirements-talker.txt  Direct Talker dependencies
LICENSE                  MIT license
```

## Model preparation

The default model ID is `inclusionAI/Ming-3.0-audio-flash`. Download it from Hugging Face:

```bash
pip install huggingface-hub
huggingface-cli download inclusionAI/Ming-3.0-audio-flash \
  --local-dir inclusionAI/Ming-3.0-audio-flash
```

For users in mainland China, ModelScope is recommended:

```bash
pip install modelscope
modelscope download --model inclusionAI/Ming-3.0-audio-flash \
  --local_dir inclusionAI/Ming-3.0-audio-flash \
  --revision master
```

Set the downloaded checkpoint path:

```bash
export MODEL_PATH=/path/to/Ming-3.0-audio-flash
```

The model directory must include the tokenizer, audio processor assets, model configuration files, and checkpoint shards. The direct Transformers example uses the model implementation tracked in this repository. Speech output additionally requires the complete `talker/` model directory.

As with Ming-Flash-Omni-2.0, this GitHub repository tracks the Talker implementation while the model repository stores its large runtime assets. Keep `talker/model.safetensors`, `talker/vae/model.safetensors`, `talker/campplus.onnx`, the Talker tokenizer, text-normalization FST files, and voice prompt audio in the Hugging Face or ModelScope checkpoint. See [`talker/README.md`](talker/README.md) for the expected layout.

## vLLM-Omni serving

Install a vLLM-Omni build that includes the `bailing_moe_v3_audio` pipeline. The following versions were used during development:

```text
vllm 0.30.0
vllm-omni 0.1.dev5
```

### Text output

Text-only serving is suitable for ASR, audio understanding, and audio reasoning:

```bash
MING_OUTPUT_MODE=text ./scripts/serve.sh "$MODEL_PATH"
```

### Text and speech output

Use the full profile for end-to-end spoken responses:

```bash
MING_OUTPUT_MODE=audio ./scripts/serve.sh "$MODEL_PATH"
```

The included profiles use four GPUs for the thinker. The full profile also places the talker on the fourth GPU. Adjust `devices`, `gpu_memory_utilization`, and sequence limits in `deploy/*.yaml` for your hardware.

### Call the API

Install the client dependency:

```bash
pip install -r requirements.txt
```

Audio understanding with a text response:

```bash
python examples/openai_audio_chat.py examples/audio/introduce_ant_group.wav \
  --prompt "Transcribe the speech and summarize the key points."
```

Audio interaction with both text and speech responses:

```bash
python examples/openai_audio_chat.py spoken_question.wav \
  --audio-response \
  --voice-name DB30 \
  --output-audio outputs/answer.wav
```

The client defaults to `http://localhost:8091/v1` and automatically uses the first model returned by `/v1/models`. Override these settings with `--base-url`, `--api-key`, and `--model`, or with the corresponding `MING_BASE_URL`, `MING_API_KEY`, and `MING_MODEL` environment variables.

## Transformers inference

For direct text-output inference, install the validated dependencies:

```bash
pip install -r requirements-hf.txt
```

Speech recognition:

```bash
python examples/transformers_audio.py examples/audio/introduce_ant_group.wav \
  --model "$MODEL_PATH" \
  --task asr
```

Audio question answering:

```bash
python examples/transformers_audio.py spoken_question.wav \
  --model "$MODEL_PATH" \
  --task qa \
  --prompt "Answer the question in the audio."
```

The Transformers path currently accepts one audio file per request and produces text. Use vLLM-Omni with the full deployment profile when speech output is required.

More task-specific commands are available in [`examples/README.md`](examples/README.md).
For an interactive walkthrough, open [`cookbook.ipynb`](cookbook.ipynb) from the repository root.

## API request format

The OpenAI-compatible endpoint accepts audio as a base64 data URL:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this audio."},
            {
                "type": "audio_url",
                "audio_url": {"url": "data:audio/wav;base64,..."},
            },
        ],
    }
]
```

Set `modalities=["text"]` for text-only output or `modalities=["text", "audio"]` for a spoken response.
