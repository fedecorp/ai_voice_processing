# The Science Behind AI Voice Processing
### A Hands-On Workshop · iw2026

> From raw sound waves to neural voice assistants and deepfake detection — built from scratch in Python.

This workshop walks through the full stack of modern AI voice processing in five progressive labs. Each notebook builds on the previous one, taking you from the physics of sound all the way to biometric speaker verification and watermarking. No black boxes — every algorithm is implemented and visualized step by step.

---

## What You'll Build

- A **real-time voice assistant** that hears you, thinks, and talks back — running either fully local or cloud-powered
- A **biometric speaker verification system** backed by a SQLite database and 512-dimensional voice embeddings
- A **deepfake audio detector** that compares classical watermarking against Meta's neural AudioSeal
- Neural networks that **classify animal sounds** and **visualize their own synaptic weights** as they train

---

## Setup

> Before the workshop, run through [Notebook 0](notebooks/sound_processing_0.ipynb) to get your environment ready.

**1. Clone the repo and create a virtual environment**
```bash
py -m venv .venv
.\.venv\Scripts\activate.bat      # Windows CMD
# source .venv/bin/activate       # macOS / Linux
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Get a Google Gemini API key**  
Head to [aistudio.google.com/api-keys](https://aistudio.google.com/api-keys), create a key, and store it somewhere safe. It goes into the notebooks as `GOOGLE_API_KEY`. Keys typically start with `AIza...`

**4. Pre-download the local LLM** *(run once, requires internet)*  
Open [Notebook 0](notebooks/sound_processing_0.ipynb) and execute the single setup cell — it caches SmolLM2-360M to your hard drive so Part 4 can run fully offline.

---

## Chapters

### [Part 1 — The Anatomy of Sound](notebooks/sound_processing_1.ipynb)

Start here if you've never thought about what a sound *is* as data.

You'll synthesize a pure sine wave from scratch, load a real MP3, and pull it apart with every major visualization tool in the audio toolkit:

- **Waveform (Oscillogram)** — full view and zoomed in to see individual cycles
- **Spectrogram** — STFT on a logarithmic frequency scale (the same view used by audio engineers and speech researchers)
- **Amplitude & Frequency Histograms** — spotting clipping, dominant frequencies, and signal energy distribution
- **Frequency Filters** — low-pass, high-pass, and band-pass implemented with Butterworth filters, each played back so you can hear the difference

> *Classroom prompt:* Record your own voice. Can you identify silences, plosives ("T", "K"), and vowels in the spectrogram?

---

### [Part 2 — Fourier, Noise & Voice Fingerprints](notebooks/sound_processing_2.ipynb)

The mathematics that powers almost every audio AI system.

You'll build intuition for the Fourier Transform by decomposing a noisy composite signal into its constituent frequencies, then reassembling it — proving that the transform is fully reversible. Then you'll capture your own voice and compute the features that speech-recognition AIs actually use.

**Key topics:**
- **Frequency filtering** with `scipy` (low / high / band-pass Butterworth)
- **Noise reduction** — one-liner with `noisereduce` vs. Wiener filter
- **Fourier Transform pipeline** — generate a mystery signal → add white Gaussian noise → decompose with FFT → threshold → reconstruct (denoised)
- **Spectral leakage & windowing** — why sharp signal edges corrupt frequency plots and how Hann windows fix it
- **Mel Spectrogram** — warping the frequency axis to match human hearing
- **MFCCs from your own live voice recording** — extract the 13-coefficient blueprint that AI uses as a voice fingerprint, saved to disk for use in later notebooks

> **Bonus:** Your saved MFCC vectors are the foundation for the biometric system built in Part 5.

---

### [Part 3 — Teaching a Machine to Hear](notebooks/sound_processing_3.ipynb)

Turn audio files into a working classifier — and watch the neural network's brain form in real time.

The notebook processes a folder of dog and elephant sounds through a full ML pipeline, from raw bytes to live inference. Every transformation is plotted so the data processing choices are visible, not hidden inside a library call.

**The four-step pipeline:**
1. **Audio Intake & Trim** — silence removal with `librosa.effects.trim`
2. **Length Enforcement** — reflection-padding short clips to a fixed 5-second window
3. **Amplitude Normalization** — neutralizing microphone-distance bias
4. **MFCC Feature Extraction** — 20-coefficient acoustic fingerprints → 4,320 features per clip

**What gets trained and visualized:**
- **PCA cluster map** — project 4,320 features down to 2D and check whether dogs and elephants are already separable before any model sees them
- **Neural network brain diagram** — every synapse colored by weight sign and thickness by magnitude, drawn after training
- **PCA-optimized network** — re-train with just 2 PCA coordinates instead of 4,320; the input layer shrinks from a wall of neurons to two nodes
- **Live Inference Arena** — drop in an unseen sound file and watch the probability scores

**Unsupervised bonus:**
- **K-Means clustering** — the algorithm groups the sounds without ever seeing the labels, then you compare its groupings against the ground truth
- **Elbow Method** — find the natural number of clusters from data geometry, without peeking at labels

> *The width of your input data dictates the size of the brain's front door.*

---

### [Part 4 — Building a Voice Assistant](notebooks/sound_processing_4.ipynb)

Assemble a complete voice assistant from three independently swappable components — and then flip a single boolean to run it entirely offline.

```
Microphone → [ Ear: STT ] → [ Brain: LLM ] → [ Voice: TTS ] → Speaker
```

**The three components:**
| Stage | Cloud mode | Local mode |
|---|---|---|
| **Ear** (Speech-to-Text) | Google Gemini 2.5 Flash (audio upload) | Google Gemini 2.5 Flash |
| **Brain** (LLM) | Google Gemini 2.5 Flash | SmolLM2-360M-Instruct (on-device CPU) |
| **Voice** (Text-to-Speech) | Microsoft Neural TTS via `edge-tts` | Same |

Set `USE_LOCAL_CPU = True` in Step 1 and re-run — the Brain switches to the locally cached model with no other changes needed.

**What you'll discover:**
- The batch sequential architecture has an unavoidable latency floor of ~4–6 seconds
- **Streaming cascaded** approach (Deepgram + streaming LLM + Cartesia) brings that down to ~500ms by running all three stages concurrently
- **Native audio-to-audio** models (Gemini Live, Moshi, Ultravox) bypass the text bottleneck entirely and natively understand tone, laughter, and interruptions

> **Bonus:** A standalone `scripts/live_voice_assistant.py` uses the Gemini Multimodal Live API for a real-time conversation loop — run outside Jupyter to avoid async event-loop collisions.

---

### [Part 5 — Speaker Verification & Deepfakes](notebooks/sound_processing_5.ipynb)

> ⚠️ *This notebook is for educational purposes only. The techniques demonstrated here are used in real-world security systems and deepfake research — not for misuse.*

The capstone. Seven interconnected sections that build a production-style biometric system, attack it with synthetic audio, and then explore how to make that audio traceable.

**Part 1 — Building the embedding space**
- Download LibriSpeech `test-clean` and curate 10 speakers × 3 utterances
- Extract MFCCs; visualize how different speakers leave different heatmap patterns
- Upgrade to **WavLM-Base-Plus-SV** (Microsoft) — 512-dimensional transformer embeddings that pack far more speaker identity than MFCCs alone
- Plot the speaker-embedding **cosine similarity matrix** and watch same-speaker scores cluster near 1.0

**Part 2 — Enrolling your own voice**
- Record 3 clips of ~10 seconds each
- Extract your WavLM embeddings and drop yourself onto the similarity map — find where you cluster relative to the 10 LibriSpeech speakers

**Part 3 — Persisting to a biometric database**
- Design and populate a **SQLite schema** for enrolled identities
- Enrol all 10 speakers under fictional names; add yourself
- Visualize the raw 512-dimensional fingerprints as a full-width heatmap

**Part 4 — Calibrating the verification threshold**
- Compute genuine-pair and impostor-pair score distributions
- Plot **FAR vs. FRR curves** and understand the Equal Error Rate (EER) tradeoff
- Two threshold choices: security-first (minimize false accepts) vs. convenience-first (minimize false rejects)

**Parts 5 & 6 — The `verify()` function and live battle demo**
- Call `verify(audio_clip)` → get a ranked top-3 match list with confidence scores
- Record a fresh clip and try to authenticate; see what happens when the system is wrong and why

**Part 7 — Deepfakes & Watermarking**
- Load pre-generated **synthetic voice clips** and run them through the biometric system — observe how deepfakes exploit the enrollment pipeline
- **Classical watermarking:** embed a narrow-band frequency signature into the audio; detect it automatically; then destroy it in two lines with a notch filter
- **Neural watermarking with AudioSeal (Meta, 2024):** embed an imperceptible learned watermark; survive the same attacks that erased the classical one; visualize the difference in the frequency domain

---

## Tech Stack

| Purpose | Library |
|---|---|
| Audio I/O & DSP | `librosa`, `sounddevice`, `soundfile`, `scipy` |
| Noise reduction | `noisereduce` |
| Machine learning | `scikit-learn` (MLP, PCA, K-Means) |
| Transformer models | `transformers` (SmolLM2, WavLM) |
| LLM & STT cloud | `google-generativeai` (Gemini 2.5 Flash) |
| TTS | `edge-tts` (Microsoft Neural voices) |
| Neural watermarking | `audioseal` (Meta, 2024) |
| Biometric storage | `sqlite3` |
| Visualization | `matplotlib` |

