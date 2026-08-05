# Voice Presentation Analyzer (VPA)

> **Languages:** [English](./README.md) | [繁體中文](./README.zh-Hant.md) | [簡體中文](./README.zh-Hans.md) | [日本語](./README.ja.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![WebGPU Accelerated](https://img.shields.io/badge/WebGPU-Accelerated-blue.svg)](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)
[![Privacy First](https://img.shields.io/badge/Privacy-100%25%20Local-brightgreen.svg)](#)
[![GitHub Release](https://img.shields.io/github/v/release/shusei/vpa.svg?color=orange)](https://github.com/shusei/vpa/releases)

---

> A 100% privacy-first, in-browser voice impression and acoustic analysis tool. Evaluates feminine / masculine voice impression tendencies, provides quick test challenges, practice drawers, real-time acoustic pitch & formant tracking, dynamic voice card sharing, and advanced analytics.

- 🌐 **Live Demo (GitHub Pages)**: https://shusei.github.io/vpa
- 📋 **Changelog**: [CHANGELOG.md](./CHANGELOG.md)
- 🔬 **Algorithm Report**: [ALGORITHM_VERIFICATION.md](./ALGORITHM_VERIFICATION.md)
- 🔒 **Privacy Guarantee**: Audio processing and neural model inference are 100% local inside your browser. No voice data is ever uploaded to any server.
- 🌍 **Multilingual Support**: English, Traditional Chinese (繁體中文), Simplified Chinese (簡體中文), and Japanese (日本語) with automatic browser locale detection.

---

## Table of Contents

- [Key Features & Dual Modes](#key-features--dual-modes)
  - [1. Quick Test Experience](#1-quick-test-experience)
  - [2. Professional Experience](#2-professional-experience)
  - [3. Practice Drawer](#3-practice-drawer)
  - [4. Dynamic Social Card Sharing](#4-dynamic-social-card-sharing)
- [Core Capabilities](#core-capabilities)
- [Workflow & Usage](#workflow--usage)
- [Technical Architecture](#technical-architecture)
- [Development & Testing](#development--testing)
- [Privacy & Disclaimer](#privacy--disclaimer)
- [License & Credits](#license--credits)

---

## Key Features & Dual Modes

Voice Presentation Analyzer (VPA) offers dual interface modes tailored for both quick daily voice check-ins and deep acoustic analysis:

### 1. Quick Test Experience
- **Daily Test**: Read a short daily phrase to get a voice impression score in seconds.
- **Standard Challenge**: Read 3 consecutive prompts to measure pitch stability and consistency across sentences.
- **Score Card & Replay Controls**: Shows impression percentage, estimated voice age, and archetype. Replay buttons support **Play/Pause** toggle.

### 2. Professional Experience
- **Real-time Pitch Stream**: Live 50–450 Hz pitch contour, instantaneous loudness, and noise floor monitoring.
- **Formant & Resonance Panel**: Real-time estimation of F1–F3 formants, chest / mask / head resonance proportions, and breathiness ratio.
- **Intonation & Advanced Analytics**: Intonation curve visualization, speaking rate (syllables/sec), continuous voicing ratio, and personalized advice.

### 3. Practice Drawer
- **Core-36 Phrase Library**: Categorized phrase drills with quick recording, instant replay, score tracking, and history comparison.

### 4. Dynamic Social Card Sharing
- **Dynamic Voice Cards (PNG / Video)**：Generate 9:16 video cards or custom PNG score graphics completely inside the browser.
- **One-Tap Social Sharing**: Direct short-link and card sharing to X (Twitter), Threads, and LINE.

---

## Core Capabilities

### Privacy & Performance

- **Local Inference**: Built with [`@xenova/transformers`](https://github.com/xenova/transformers.js) and ONNX Runtime (WebGPU / WASM). Models run locally once cached in IndexedDB.
- **Offline Support**: Works 100% offline after the initial model download.
- **Format Support**: Supports MediaRecorder live recording and file uploads (`.mp3`, `.m4a`, `.mp4`, `.mov`, `.wav`).

---

## Development & Testing

```bash
npm test      # Run unit tests, syntax checks, and markup checks
npm run test:e2e # Run Playwright end-to-end browser test suite
```

---

## Privacy & Disclaimer

- **Non-Medical Tool**: VPA is an acoustic feedback tool for voice exploration and training. It does **not** determine gender identity, legal status, or medical diagnoses.
- **Vocal Safety**: Practice at a comfortable volume. If you experience vocal fatigue or hoarseness, pause and consult a licensed speech-language pathologist or ENT physician.

---

## License

- **Project**: MIT License
- **Model**: Apache-2.0 License (`prithivMLmods/Common-Voice-Gender-Detection-ONNX`)
