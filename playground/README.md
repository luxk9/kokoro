# Kokoro Playground

A tiny local web UI for trying out Kokoro: type text, pick a voice, adjust speed, play / download the WAV, and see the phonemes and generation speed.

```bash
# from the repo root
pip install -e . flask soundfile
# needed for non-English voices (and English out-of-dictionary fallback)
sudo apt-get install espeak-ng     # macOS: brew install espeak-ng

python playground/app.py
# open http://localhost:7860   (set PORT=... to change)
```

The first run downloads the model (~330 MB) and voices from Hugging Face. Runs on CPU fine; uses CUDA automatically if available.
