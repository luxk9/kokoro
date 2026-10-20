"""Minimal local web playground for trying out Kokoro TTS.

Run:  python playground/app.py   then open http://localhost:7860
"""
import io
import os
import threading
import time
from urllib.parse import quote

import numpy as np
import soundfile as sf
import torch
from flask import Flask, jsonify, request, send_file, send_from_directory

from kokoro import KModel, KPipeline

REPO_ID = 'hexgrad/Kokoro-82M'
SAMPLE_RATE = 24000
MAX_CHARS = 5000

# Voice id prefix letter == pipeline lang_code. Non-English voices need espeak-ng installed.
VOICES = {
    'American English': ['af_heart', 'af_bella', 'af_nicole', 'af_aoede', 'af_kore', 'af_sarah', 'af_nova',
                         'af_sky', 'af_alloy', 'af_jessica', 'af_river', 'am_michael', 'am_fenrir', 'am_puck',
                         'am_echo', 'am_eric', 'am_liam', 'am_onyx', 'am_santa', 'am_adam'],
    'British English': ['bf_emma', 'bf_isabella', 'bf_alice', 'bf_lily', 'bm_george', 'bm_fable', 'bm_lewis',
                        'bm_daniel'],
    'Spanish': ['ef_dora', 'em_alex', 'em_santa'],
    'French': ['ff_siwis'],
    'Hindi': ['hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi'],
    'Italian': ['if_sara', 'im_nicola'],
    'Brazilian Portuguese': ['pf_dora', 'pm_alex', 'pm_santa'],
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = KModel(repo_id=REPO_ID).to(device).eval()
pipelines = {}
lock = threading.Lock()


def get_pipeline(lang_code):
    if lang_code not in pipelines:
        pipelines[lang_code] = KPipeline(lang_code=lang_code, repo_id=REPO_ID, model=model)
        if lang_code == 'a':
            pipelines[lang_code].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
        elif lang_code == 'b':
            pipelines[lang_code].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'
    return pipelines[lang_code]


app = Flask(__name__)


@app.get('/')
def index():
    return send_from_directory(os.path.dirname(__file__), 'index.html')


@app.get('/api/voices')
def voices():
    return jsonify({'voices': VOICES, 'device': device})


@app.post('/api/tts')
def tts():
    body = request.get_json(force=True) or {}
    text = (body.get('text') or '').strip()
    voice = body.get('voice') or 'af_heart'
    speed = float(body.get('speed') or 1.0)
    if not text:
        return jsonify({'error': 'text is empty'}), 400
    if len(text) > MAX_CHARS:
        return jsonify({'error': f'text is longer than {MAX_CHARS} characters'}), 400
    if not any(voice in v for v in VOICES.values()):
        return jsonify({'error': f'unknown voice {voice}'}), 400
    speed = min(max(speed, 0.5), 2.0)

    start = time.perf_counter()
    try:
        with lock:
            pipeline = get_pipeline(voice[0])
            chunks, phonemes = [], []
            for result in pipeline(text, voice=voice, speed=speed):
                if result.audio is not None:
                    chunks.append(result.audio.cpu().numpy())
                    phonemes.append(result.phonemes)
    except Exception as e:  # surface G2P / download errors to the UI instead of a bare 500
        return jsonify({'error': f'{type(e).__name__}: {e}'}), 500
    if not chunks:
        return jsonify({'error': 'no audio was produced'}), 500

    audio = np.concatenate(chunks)
    elapsed = time.perf_counter() - start
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format='WAV')
    buf.seek(0)
    resp = send_file(buf, mimetype='audio/wav', download_name=f'kokoro_{voice}.wav')
    resp.headers['X-Gen-Seconds'] = f'{elapsed:.3f}'
    resp.headers['X-Audio-Seconds'] = f'{len(audio) / SAMPLE_RATE:.3f}'
    resp.headers['X-Phonemes'] = quote(' / '.join(phonemes))
    return resp


if __name__ == '__main__':
    get_pipeline('a')  # warm up so the first request isn't slow
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 7860)), threaded=True)
