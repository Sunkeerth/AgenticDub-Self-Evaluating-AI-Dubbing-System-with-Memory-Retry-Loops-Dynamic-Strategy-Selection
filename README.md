<div align="center">

<img src="https://img.shields.io/badge/Agentic-AI-blueviolet?style=for-the-badge&logo=openai&logoColor=white" />
<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" />
<img src="https://img.shields.io/badge/Whisper-OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" />
<img src="https://img.shields.io/badge/Groq-LLaMA%203.3-00A67E?style=for-the-badge" />
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />

# 🎙️ Agentic AI Dubbing System

### *Plan → Transcribe → Translate → Synthesize → Evaluate → Retry → Learn*

**An end-to-end multi-agent pipeline that dubs speech from any language into any other language — autonomously, with self-evaluation, automatic retries, and persistent memory that improves with every run.**

[🎬 Watch Demo](#-demo) · [🚀 Quick Start](#-quick-start) · [🧠 How It Works](#-how-it-works) · [🏗️ Architecture](#️-architecture) · [📦 Tech Stack](#-tech-stack)

---

</div>

## 🎬 Demo

> **👇 Click the thumbnail below to watch the full demo walkthrough**

[![Demo Video](https://img.shields.io/badge/▶%20Watch%20Demo-YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://YOUR_VIDEO_LINK_HERE)

<!--
  🔴 REPLACE the link above with your actual demo video URL
  Example: https://youtu.be/YOUR_VIDEO_ID
-->

```
📽️  Demo covers:
  ✅ Upload Kannada audio via Gradio UI
  ✅ Watch the 5-agent pipeline execute in real-time
  ✅ See ASR self-evaluation scores live
  ✅ See LLaMA 3.3 translate with strategy selection
  ✅ Hear the English dubbed output via Microsoft Neural TTS
  ✅ Observe memory learning across multiple runs
```

---

## ✨ What Makes This Agentic?

Unlike a normal pipeline that blindly executes steps in sequence, this system **thinks, judges its own output, and corrects itself** without any human involvement after the initial audio upload.

| 🧠 Agent Property | What It Means | How This System Does It |
|---|---|---|
| **Autonomy** | Works without step-by-step human guidance | Upload once → full dubbing pipeline completes on its own |
| **Reasoning** | Thinks before acting | Orchestrator reads memory, selects the best strategy before starting |
| **Self-Evaluation** | Scores its own output | Every agent produces a 0–1 confidence score after each action |
| **Self-Correction** | Fixes mistakes without being told | Auto-retries with a different strategy when quality is below threshold |
| **Memory & Learning** | Improves from experience | Writes results to JSON memory; reads it next run to start smarter |

---

## 🚀 Quick Start

### Run in Google Colab (Recommended)

> No local setup required. GPU available for free.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/BEST_AGENTIC.ipynb)

**Steps:**
1. Open the notebook in Colab
2. Set your free Groq API key in **Cell 2** (get one at [console.groq.com](https://console.groq.com) — no credit card needed)
3. Run **Cell 1** (install) → **Cell 2** (config) → **Cell 3** (memory) → **Cell 4** (models) → **Cell 5** (agents) → **Cell 6** (orchestrator) → **Cell 7** (UI)
4. Click the Gradio public link, upload your audio, and hit **🚀 Run Agentic Pipeline**

### Local Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# Install system dependency
sudo apt-get install -y ffmpeg   # Linux
brew install ffmpeg              # macOS

# Install Python packages
pip install numpy==1.26.4 --force-reinstall
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install openai-whisper groq edge-tts pydub soundfile scipy gradio nest_asyncio rich jiwer

# Launch
jupyter notebook BEST_AGENTIC.ipynb
```

---

## 🧠 How It Works

### The Full Pipeline — Step by Step

```
You upload audio
       │
       ▼
┌─────────────────────┐
│  Orchestrator Agent │  ← Reads memory, picks best strategy
│  (The Boss)         │  ← Plans before any work starts
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    ASR Agent        │  ← Whisper transcribes audio → Kannada text
│  (Ear of system)    │  ← Self-scores: confidence, language match,
└────────┬────────────┘     speech rate, transcript length
         │
    Score ≥ 0.55?
    NO → retry with bigger Whisper model
    YES ↓
         ▼
┌─────────────────────┐
│ Translation Agent   │  ← Groq LLaMA 3.3 translates → English
│  (Brain of system)  │  ← Picks from: Literary / Literal / TTS-Optimised
└────────┬────────────┘  ← LLM self-judges translation quality (out of 10)
         │
         ▼
┌─────────────────────┐
│   Voice Agent       │  ← Microsoft Neural TTS synthesizes English audio
│  (Mouth of system)  │  ← Validates duration, silence level, file integrity
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Final Evaluation   │  ASR×0.30 + Translation×0.40 + Voice×0.30
│  Score ≥ 0.65?      │
└────────┬────────────┘
         │
    NO → retry (max 3x, keep best)
    YES ↓
         ▼
┌─────────────────────┐
│   Memory Agent      │  ← Saves job record, strategy, scores to JSON
│  (Long-term memory) │  ← Next run starts with the winning strategy
└─────────────────────┘
         │
         ▼
   Download dubbed audio ✅
```

### Quality Scoring Formula

```
Final Score = (ASR Score × 0.30) + (Translation Score × 0.40) + (Voice Score × 0.30)

≥ 0.80  →  Excellent  ✅
≥ 0.65  →  Good        ✅ (accepted)
≥ 0.50  →  Acceptable  ⚠️ (retry attempted)
< 0.50  →  Poor        ❌ (retry triggered)
```

---

## 🏗️ Architecture

### Agent Roles

#### 🎯 Orchestrator Agent
The central coordinator. Wakes up before any work begins, queries memory for the best historical strategy for the given language pair, and then drives the entire pipeline. After all agents finish, it computes the combined score and decides: **accept**, **retry**, or **keep best attempt**.

#### 🎤 ASR Agent (Automatic Speech Recognition)
Uses **OpenAI Whisper** locally on GPU. Self-evaluates on:
- Whisper's internal `avg_logprob` confidence score
- Language detection match (did it detect the right language?)
- Speech rate plausibility (is transcript suspiciously short for audio length?)
- Repetition detection (hallucination check)

On low-confidence results, escalates from `tiny` → `base` → `small` → `medium` model automatically.

#### 🌐 Translation Agent
Uses **Groq's free LLaMA 3.3 70B** API. Has three switchable strategies:

| Strategy | Best For | Description |
|---|---|---|
| `literary` | Stories, scripts, drama | Preserves emotional tone and dramatic voice |
| `literal` | Factual, technical content | Word-for-word accuracy |
| `tts_optimised` | Any speech | Short clear sentences for natural-sounding TTS |

After translating, it sends the result back to the LLM to **self-evaluate** on completeness, fluency, and speech suitability (scored out of 10).

#### 🗣️ Voice Agent
Uses **Microsoft Edge TTS** (Neural voices via `edge-tts`). Features:
- Per-language neural voice selection (14+ languages supported)
- LRU cache — never re-synthesizes identical text
- Long-text chunking at sentence boundaries with natural pauses
- Post-synthesis validation: duration check, silence/RMS check, file integrity

#### 🧠 Memory Agent
Persistent JSON store at `/content/dub_outputs/agent_memory.json`. Records:
- Every completed job (language pair, strategy, all three scores, success/fail, attempt count)
- Per-language-pair strategy performance averages
- Aggregate success rate across all jobs

On next run, `best_strategy(src, tgt)` returns the highest-average-scoring strategy automatically.

---

## 📦 Tech Stack

| Component | Library / Service | Purpose |
|---|---|---|
| Speech Recognition | `openai-whisper` (tiny/small/medium) | Kannada (and 99 other languages) → text |
| LLM Translation | Groq API · LLaMA 3.3 70B | Text translation + self-evaluation |
| Text-to-Speech | Microsoft `edge-tts` Neural voices | High-quality multilingual synthesis |
| Audio Processing | `pydub`, `soundfile`, `ffmpeg` | Format conversion, chunking, merging |
| Deep Learning Runtime | `torch`, `torchaudio` (CUDA 11.8) | GPU inference for Whisper |
| UI | `gradio >= 4.0` | Browser-based interactive interface |
| Async Runtime | `nest_asyncio`, `asyncio` | Non-blocking TTS synthesis |
| Quality Metrics | `jiwer` | WER scoring for ASR evaluation |
| Logging | `rich` | Colored terminal output |
| Memory Store | JSON (built-in) | Persistent agent memory across sessions |

---

## 🌍 Supported Languages

| Language | Input (ASR) | Output (TTS) | TTS Voice |
|---|---|---|---|
| Kannada | ✅ | ✅ | `kn-IN-SapnaNeural` |
| Hindi | ✅ | ✅ | `hi-IN-SwaraNeural` |
| English | ✅ | ✅ | `en-US-JennyNeural` |
| Tamil | ✅ | ✅ | `ta-IN-PallaviNeural` |
| Telugu | ✅ | ✅ | `te-IN-ShrutiNeural` |
| French | ✅ | ✅ | `fr-FR-DeniseNeural` |
| Spanish | ✅ | ✅ | `es-ES-ElviraNeural` |
| German | ✅ | ✅ | `de-DE-KatjaNeural` |
| Italian | ✅ | ✅ | `it-IT-ElsaNeural` |
| Portuguese | ✅ | ✅ | `pt-BR-FranciscaNeural` |
| Arabic | ✅ | ✅ | `ar-SA-ZariyahNeural` |
| Chinese | ✅ | ✅ | `zh-CN-XiaoxiaoNeural` |
| Japanese | ✅ | ✅ | `ja-JP-NanamiNeural` |
| Korean | ✅ | ✅ | `ko-KR-SunHiNeural` |
| + Auto-detect | ✅ | — | Whisper auto language detection |

> Any of the above can be the **source** or **target** language. Mix and match freely.

---

## ⚙️ Configuration Reference

Edit these values in **Cell 2** of the notebook:

```python
GROQ_API_KEY        = "gsk_..."     # Required — get free at console.groq.com
WHISPER_SIZE        = "small"       # tiny | base | small | medium
MAX_RETRIES         = 3             # Max retry attempts before keeping best result
ASR_CONF_THRESHOLD  = 0.55          # Min ASR confidence to accept (0.0–1.0)
TRANS_LEN_MIN_RATIO = 0.25          # Translation must be ≥ 25% length of source
TRANS_LEN_MAX_RATIO = 5.0           # Translation must be ≤ 5× length of source
AUDIO_MIN_DURATION  = 0.4           # Synthesized audio must be ≥ 0.4 seconds
AUDIO_MIN_RMS       = 0.002         # Silence detection threshold
OUTPUT_DIR          = "/content/dub_outputs"
MEMORY_FILE         = "/content/dub_outputs/agent_memory.json"
```

---

## 🖥️ Gradio UI

After running all cells, a Gradio interface launches with a public share link. It provides:

- **Microphone recording** or **file upload** for audio input
- **Language dropdowns** for source and target language
- **Dubbed audio player** with direct download button
- **Live agent status** showing strategy, per-agent scores, elapsed time
- **Transcribed text** from Whisper (source language)
- **Translated text** sent to TTS (target language)
- **Memory panel** showing last 5 job history with scores

---

## 📁 Project Structure

```
📦 agentic-ai-dubbing/
 ┣ 📓 BEST_AGENTIC.ipynb        ← Main notebook (7 cells)
 ┣ 📄 README.md                 ← This file
 ┣ 📄 How_This_Works.md         ← Plain-English explanation of agentic concepts
 ┗ 📁 dub_outputs/              ← Created at runtime
    ┣ 🔊 dubbed_final.wav       ← Latest dubbed audio output
    ┗ 🧠 agent_memory.json      ← Persistent memory store (grows over time)
```

---

## 🔐 API Keys & Secrets

This project requires one API key:

| Service | Key | Cost | Get It |
|---|---|---|---|
| **Groq** | `GROQ_API_KEY` | **Free** (no credit card) | [console.groq.com](https://console.groq.com) |

> ⚠️ **Never commit your API key to GitHub.** Use Colab Secrets or a `.env` file locally. The notebook includes a runtime check that refuses to proceed if the placeholder key is still set.

---

## 🧩 Notebook Cell Reference

| Cell | Name | What It Does |
|---|---|---|
| **Cell 1** | Install Dependencies | Pins numpy 1.26.4, installs all packages, runs import verification |
| **Cell 2** | Configuration | API key, Whisper size, agentic thresholds, output paths |
| **Cell 3** | Memory Store | `AgentMemory` class — JSON-backed persistent learning store |
| **Cell 4** | Load Models | Loads Whisper, initialises Groq client, sets up TTS voice cache |
| **Cell 5** | Specialist Agents | `ASRAgent`, `TranslationAgent`, `VoiceAgent` — each with `run()` + `evaluate()` |
| **Cell 6** | Orchestrator | `OrchestratorAgent` — coordinates agents, retry loop, final scoring |
| **Cell 7** | Gradio UI | Full browser interface — connects everything, shares a public URL |

---

## 💡 Example Output

```
✅ Job a3f2c1b0 | 18s
Strategy: literary
ASR:   0.81 | Trans: 0.90 | Voice: 0.88
Final: Excellent (0.87)
Memory: 7 jobs, 85.7% success
```

```
📜 Source (Whisper):
ನಮ್ಮ ಊರಿನ ಹಬ್ಬದ ದಿನ ಎಲ್ಲರೂ ಒಂದೆಡೆ ಸೇರಿ ಸಂತೋಷದಿಂದ ಆಚರಿಸುತ್ತಾರೆ.

🌐 Translated (LLaMA 3.3):
On the festival day of our village, everyone gathers together and celebrates with joy.

🔊 Dubbed audio: dubbed_final.wav (3.4 seconds, en-US-JennyNeural)
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `numpy` version conflict / Numba crash | Re-run Cell 1 — it force-reinstalls `numpy==1.26.4` |
| Groq `AuthenticationError` | Check your `GROQ_API_KEY` in Cell 2 |
| `ffmpeg not found` | Re-run Cell 1 — `apt-get install ffmpeg` is included |
| TTS produces silence | Audio RMS below threshold — TTS retries automatically |
| Whisper wrong language | Set source language explicitly instead of `Auto-detect` |
| Colab disconnects | Re-run Cells 1–6, then Cell 7. Memory is preserved in `dub_outputs/` |
| Low quality score | Increase `MAX_RETRIES` or reduce `ASR_CONF_THRESHOLD` in Cell 2 |

---

## 🗺️ Roadmap

- [ ] Voice cloning with Coqui XTTS v2 (speaker identity preservation)
- [ ] Multi-speaker diarization with Pyannote
- [ ] Subtitle/SRT file export alongside dubbed audio
- [ ] Streamlit version for easier local deployment
- [ ] Docker container for one-command local setup
- [ ] Batch processing mode for multiple audio files

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repo
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

Please make sure all cells in the notebook run cleanly end-to-end before submitting.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

| Project | What it contributes |
|---|---|
| [OpenAI Whisper](https://github.com/openai/whisper) | Multilingual speech recognition |
| [Groq](https://groq.com) | Ultra-fast free LLaMA 3.3 inference |
| [Microsoft Edge TTS](https://github.com/rany2/edge-tts) | Neural voice synthesis |
| [Gradio](https://gradio.app) | Web UI in pure Python |
| [PyDub](https://github.com/jiaaro/pydub) | Audio manipulation |

---

<div align="center">

**Built with ❤️ as a demonstration of Agentic AI concepts**

*If this project helped you, please ⭐ the repo!*

[![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/YOUR_REPO?style=social)](https://github.com/YOUR_USERNAME/YOUR_REPO)

</div>

