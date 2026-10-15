# Talker source

This directory tracks the Python implementation of the Ming-3.0-audio-flash
Talker, AudioVAE, flow model, and text normalization pipeline. Large model and
runtime assets are intentionally not stored in this GitHub repository.

The complete Hugging Face or ModelScope checkpoint must retain this layout:

```text
talker/
  model.safetensors
  campplus.onnx
  llm/
  vae/
    config.json
    model.safetensors
  data/
    voice_name.json
    ... prompt audio files ...
  talker_tn/
    *_tagger.fst
    *_verbalizer.fst
```

Install `requirements-talker.txt` when using the Talker implementation
directly. The recommended end-to-end serving path remains vLLM-Omni, which
loads these runtime assets from `<MODEL_PATH>/talker`.
