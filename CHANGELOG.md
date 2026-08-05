# Changelog (更新日誌)

All notable changes to this project will be documented in this file.
本專案的所有顯著變更都將記錄於此檔案中。

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.1] - 2026-08-05

### Added & Changed (新增與變更)
- **Bilingual International CHANGELOG (中英雙語國際更新日誌)**:
  - Formatted `CHANGELOG.md` into a clean bilingual layout (English & Traditional Chinese) to ensure global developers and GitHub Release readers understand all updates.
    將 `CHANGELOG.md` 重構為中英雙語專業對照排版，方便全球開發者與 GitHub Release 讀者查閱版本演進。

---

## [1.2.0] - 2026-08-05

### Changed & Refactored (變更與校正)
- **Multilingual Copy & Acoustic Safety Refactor (多語系文案與聲學安全重構)**:
  - Polished all 4 supported language dictionaries (English, Traditional Chinese, Simplified Chinese, Japanese) for authentic native fluency and clinical safety.
    全面優化英、繁中、簡中、日文自然母語表達，確保符合專業聲學倫理與各國語境。
  - Renamed resonance labels from organ locations to objective spectral energy bands (`Low Band`, `Mid Band`, `High Band` / `低頻能量`, `中頻能量`, `高頻能量`).
    將共鳴頻譜標籤由器官定位名稱（胸腔/面罩/頭腔）修訂為客觀聲學能量帶 (`低頻能量` / `中頻能量` / `高頻能量`)。
  - Re-framed "Next challenge" insights into relaxed acoustic observations rather than mechanical vocal training prescriptions.
    將「下一個挑戰」與自動建議調整為無壓力的客觀聲學觀察與放鬆探索提示，避免機械式開出發聲處方。
  - Renamed `falsettoContrast` to `metricMismatch` (Pitch × spectral mismatch) and enhanced vocal safety notices across all views.
    調整 `falsettoContrast` 為 `metricMismatch` (指標分歧)，並強化所有介面之實驗性聲學說明與發聲安全提醒。

---

## [1.1.2] - 2026-08-05

### Added (新增)
- **GitHub Shields Status Badges (GitHub Shields 質感徽章)**:
  - Added live status badges for MIT License, WebGPU Accelerated, Privacy 100% Local, and GitHub Release across all README files.
    於所有 README 頂端新增 MIT License、WebGPU Accelerated、Privacy 100% Local 與 GitHub Release 即時動態徽章。

---

## [1.1.1] - 2026-08-05

### Added (新增)
- **Multilingual README Documentation (多語系 README 文件)**:
  - Added dedicated README files for English (`README.md`), Traditional Chinese (`README.zh-Hant.md`), Simplified Chinese (`README.zh-Hans.md`), and Japanese (`README.ja.md`) with a top language selector bar.
    建立 `README.md` (English)、`README.zh-Hant.md` (繁體中文)、`README.zh-Hans.md` (簡體中文) 與 `README.ja.md` (日本語)，並於頂端新增跨語言快捷切換列。

---

## [1.1.0] - 2026-08-05

### Added (新增)
- **Quick Test Experience (快速測試模式)**: Concise phrase test and Daily Test challenge.
  提供精簡直觀的單句練習與每日挑戰 (Daily Test)，方便快速評估語音呈現。
- **Standard Voice Challenge (標準語音測試)**: 3-line consecutive test to measure pitch stability across sentences.
  支援連續 3 句語音測試，計算跨語句的聲音穩定度與綜合評分。
- **Dynamic Social Card & Video Sharing (社群卡片與影音分享)**:
  - On-device 9:16 video generation and custom PNG voice result cards.
    支援本機繪製專屬語音結果動態卡片 (PNG) 與 9:16 短影片。
  - One-tap sharing to X (Twitter), Threads, and LINE with short-link integration.
    整合 X (Twitter)、Threads、LINE 社群一鍵分享與短網址產出。
- **Play/Pause Playback Toggle (即時播放/暫停控制)**:
  - Interactive playback buttons allowing instant pause and resume during audio review.
    快速測試結果卡片上的播放按鈕支援「播放/暫停」切換，可隨時暫停與繼續播放。
- **Practice Drawer (句庫練習抽屜)**:
  - Built-in Core-36 phrase library supporting quick recording, replay, score logging, and history.
    內建 Core-36 經典句子庫，支援錄音、回聽、成績自動填回與練習歷程記錄。
- **Intonation & Acoustic Metrics (語調與聲學進階指標)**:
  - Added Intonation Curve, speech rate, continuous liaison ratio, and spectral brightness tags.
    統計卡新增語調曲線圖 (Intonation Curve)、連音比例 (Connected Speech)、語速與聲學亮度標籤。

### Changed (變更)
- **ESM Architecture Refactor (純 ESM 架構重構)**: Modularized codebase into ES Modules for better maintainability and performance.
  將過往單體腳本重構為模組化 `src/` 與 `assets/js/` 結構，提升代碼維護性與載入效率。
- **Dark Theme Contrast Optimization (深色主題對比度優化)**: Enhanced font and chart scale contrast for dark mode readability.
  修復深色主題下字體與圖表刻度之對比度問題，強化可讀性。

---

## [1.0.0] - 2026-08-01

### Initial Release (初始版本)
- **100% In-Browser Inference (100% 瀏覽器端推論)**: Privacy-first voice inference powered by Transformers.js & ONNX Runtime.
  基於 Transformers.js 與 ONNX Runtime，音訊檔案與錄音完全不離身，保護隱私。
- **Offline Support & Cache (離線運作與快取)**: IndexedDB model caching for offline usage.
  首次下載模型後快取於 IndexedDB，後續造訪支援完全離線分析。
- **WebGPU & WASM Dual Pipeline (雙模式硬體推論)**: Auto-selection between WebGPU and WASM acceleration.
  自動根據瀏覽器環境選擇 WebGPU 或 WASM 進行模型推論與視窗串流。
- **Multi-language UI (多語系支援)**: Support for English, Traditional Chinese, and Simplified Chinese.
  預設繁體中文，支援簡體中文與英文即時切換。
- **30+ Theme Engine (多主題系統)**: Built-in 30+ themes with auto light/dark mode tracking.
  內建 30+ 派別與 Lux 系列主題，支援自動跟隨系統深淺色模式。
