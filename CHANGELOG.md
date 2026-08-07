# Changelog (更新日誌)

All notable changes to this project will be documented in this file.
本專案的所有顯著變更都將記錄於此檔案中。

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.9] - 2026-08-08

### Fixed & Hardened (根源修復與多語系卡片防護)
- **Eliminated Cached Chinese Analysis Metrics Leak (`assets/js/advanced-metrics.js`) (徹底消滅動態分析卡片殘留中文 Root Cause)**:
  - Removed stale `analysisText()` top-level module closure references across all 10 analysis categorizers (`categorizeBrightness`, `describeResonanceFromEnergy`, `categorizeTilt`, `categorizeBreathiness`, `makeFormantHint`, `buildFormantTrendDisplay`, `analyzeVowelFocus`, `analyzeSpeechRate`, `analyzeConnectedSpeech`, `analyzeIntonation`).
    重構 `assets/js/advanced-metrics.js` 中所有 10 個音訊分析估算函式，移除原先在模組頂層快取的 `analysisText()` 閉包，改為 100% 直連 `t()` 動態多語系尋址。
  - Prevents initial module imports from capturing stale Traditional Chinese text closures, guaranteeing that dynamic Focus Cards and Advanced Analysis Metrics render 100% in English when switched to English mode.
    徹底解決模組初始化時誤扣繁體中文字典閉包的深層 bug，確保切換至英文模式後，動態產出的重點建議卡與進階指標卡片 100% 呈現為純英文。

- **Enhanced Dynamic Locale Cleanliness Regression Test (`npm run test:locale`) (升級自動化語系潔淨度模擬斷言)**:
  - Added Step 4 to `tests/verify-locale-cleanliness.mjs` to dynamically execute `computeAdvancedSummary` and `buildFocusInsights` in EN mode and assert 0 CJK Chinese characters in generated card labels, hints, displays, and severity titles.
    在 `tests/verify-locale-cleanliness.mjs` 補齊第四階段動態分析測試，於 EN 模式下直接執行音訊分析與重點卡片生成，強制斷言產出之所有動態內文、標題與建議 0% 含有中文字元。

---

## [1.3.8] - 2026-08-08

### Fixed & Automated (修復與自動化測試)
- **100% Complete i18n Tag Coverage (進階分析與全站 100% 標籤多語言化)**:
  - Added `data-i18n`, `data-i18n-html`, and `data-i18n-attrs` attributes to all 52 static HTML text elements in `index.html` (including Help Overlay ❓, Voice Guide 📖, chip badges, and model technical details).
    為 `index.html` 中所有 52 個原本寫死的繁體中文 HTML 標籤（包含使用指南 ❓、訓練手冊 📖、頂部 `.chip` 標籤、模型方法簡述等）全部補齊 `data-i18n` 屬性。
  - Fully translated Help & Guide overlays into `en`, `ja`, `zh-Hans`, and `zh-Hant` with 100% symmetric 602-key dictionaries across all supported locales, completely resolving Chinese text leakage when viewing Advanced Analysis in English mode.
    於 `en`、`ja`、`zh-Hans` 與 `zh-Hant` 四個字典中同步補齊使用指南、訓練手冊與標籤鍵值（達 602 個 Key 100% 對稱），徹底解決英文模式下查看進階分析與對話框時出現中文的 Bug。

- **Automated Locale Cleanliness & Contamination Regression Suite (`npm run test:locale`) (自動化語系防污染回歸測試腳本)**:
  - Introduced `tests/verify-locale-cleanliness.mjs` and integrated `test:locale` directly into the default `npm test` pipeline.
    建立 `tests/verify-locale-cleanliness.mjs` 並將 `npm run test:locale` 直接納入主測試管線中。
  - Enforces 3 strict guarantees on every build: 100% symmetric dictionary key coverage, 0 CJK Chinese characters in `en` dictionary values, and 100% `data-i18n` tag binding for all static text nodes in `index.html`.
    強制在每次建置測試中執行三項嚴格斷言：四語系字典 Key 100% 對稱、`en` 字典絕無中文字元殘留、`index.html` 靜態內文標籤 100% 繫結 `data-i18n`，從機制上 100% 杜絕未來再次發生語系混雜問題。
  - Verified 100% pass across all unit, locale cleanliness, and 42 Playwright E2E tests.
    全數通過單元測試、語系潔淨度測試與 42 項 Playwright E2E 瀏覽器測試。

---

## [1.3.7] - 2026-08-06

### Improved & Fixed (優化與修復)
- **i18n Fallback Alignment Across Locales (多語系 Key 全全面對齊與退回機制優化)**:
  - Optimized fallback key resolution in `assets/js/advanced-summary-render.js`. In advanced section view, fully aligned key references across `en`, `ja`, `zh-Hans`, and `zh-Hant` dictionaries to eliminate unformatted English fallback text leaks.
    優化 `assets/js/advanced-summary-render.js` 中的 fallback 讀取邏輯。在切換至「進階面板（Advanced Section）」時，徹底補強 `en`、`ja`、`zh-Hans` 與 `zh-Hant` 的欄位 key 尋址，消滅偶發出現英文備援硬編碼文字的問題。
- **Replay Loop Ergonomics & Player Positioning (重播播放器位置上移與極速練習迴圈)**:
  - Relocated the `.player` playback container (`▶︎ Play last take`) directly below the primary `.start-wrap` recording controls.
    將聲音重播區塊 (`.player` / `▶︎ 播放剛才的聲音`) 調整掛載於錄音控制區 (`.start-wrap`) 的正下方。
  - Practice loops ("Record -> Replay -> Re-record") are now fully accessible without scrolling the viewport on desktop or mobile devices.
    使用者進行「錄音 ➔ 重播回聽 ➔ 再次錄音」高頻率偽音練習時，無需再滑動畫面，大幅升級練習效率與手感。

---

## [1.3.6] - 2026-08-05

### Fixed (修復)
- **Quick Experience Mobile File Picker (快速測試手機原生檔案選單修復)**:
  - Removed `hidden` attribute from `<input type="file">` elements inside landing page labels — `hidden` equals `display:none` which iOS Safari ignores entirely when activating via a `<label>`.
    移除快速測試首頁 `<label>` 內的 `<input type="file">` 上的 `hidden` 屬性。`hidden` 等同 `display:none`，iOS Safari 對此類 input 完全不觸發，即使有 `<label for>` 關聯也無效。
  - Removed `pointer-events:none` from `.quick-file-input` CSS — this was blocking all touch events from reaching the input through the label on mobile devices.
    移除 `.quick-file-input` CSS 中的 `pointer-events:none`，該屬性阻擋了所有觸控事件穿透至 input，導致手機點擊 label 時 input 完全收不到信號。
  - Both fixes together allow iOS Safari and Android Chrome to correctly present the native "Photo Library / Video / Choose File" three-option picker.
    兩項修復合力確保 iOS Safari 與 Android Chrome 能正確彈出「照片圖庫 / 錄影 / 選擇檔案」三選項原生選單。
  - 42 Playwright E2E tests pass.

---

## [1.3.5] - 2026-08-05

### Fixed (修復)
- **Quick Experience Upload Analysis Flow & E2E Test (快速測試上傳解碼推論與結果跳轉修復)**:
  - Exported `recorderCtl.handleFileOrBlob` in `assets/app-core.js` and fixed `ReferenceError` when processing uploaded audio files in `assets/experiments/experience-shell.js`.
    在 `assets/app-core.js` 的 `recorderCtl` 補齊暴露 `handleFileOrBlob` 方法，徹底解決快速測試選擇語音檔案後拋出 `ReferenceError: handleFileOrBlob is not defined` 導致無法進行分析的嚴重大錯。
  - Added `isQuickUpload` state flag so uploading MP3/M4A/WAV files in Quick Experience mode immediately transitions to `"analyzing"` and presents the complete result card upon completion.
    新增 `isQuickUpload` 狀態，使快速測試選取語音檔案後會立即轉入「分析中」畫面，並於解碼推論完成後直接彈出最終女性/男性傾向、音高與角色結果卡。
  - Added dedicated E2E automated test in `tests/e2e/quick-experience.spec.js`, passing 100% across all 42 Playwright browser tests.
    在 `tests/e2e/quick-experience.spec.js` 補上針對「快速測試上傳音檔」的全新 Playwright E2E 自動化測試案例並全數通過。

---

## [1.3.4] - 2026-08-05

### Fixed (修復)
- **Mobile Touch File Upload Compatibility (行動端觸控原生檔案選擇器修復與相容性強化)**:
  - Replaced script-driven hidden input clicks (`document.getElementById("fileInput").click()`) with native `<label for="fileInput">` elements across all landing templates and FAB floating buttons (`#uploadFab`).
    將所有上傳按鈕與右下角 ⬆︎ 按鈕重構為原生 `<label for="fileInput">` 標籤，徹底解決 iOS Safari / Android Chrome / LINE 內建瀏覽器阻擋腳本觸發 `.click()` 導致點擊沒反應的問題。
  - Verified 100% PASS across 41 Playwright E2E browser tests.
    通過 41 項 Playwright 瀏覽器自動化 E2E 測試。

---

## [1.3.3] - 2026-08-05

### Added & Fixed (新增與修復)
- **PC No-Microphone User Experience & Fast Timeout (無麥克風桌機體驗優化與快速逾時處理)**:
  - Added a prominent **"📁 Upload Audio File (📁 上傳音訊檔案)"** action button in Quick Experience landing pages, allowing PC users without microphones to immediately select MP3/M4A/WAV audio files for analysis.
    在快速測試首頁新增「📁 上傳音訊檔案」按鈕，讓沒有麥克風的電腦使用者能直接選取 MP3/M4A/WAV 語音進行分析，免去設備限制。
  - Reduced microphone initialization wait timeout from 30 seconds down to 4 seconds, presenting immediate clear error prompts and alternative upload guidance instead of freezing.
    將麥克風請求逾時從 30 秒縮短至 4 秒，若無麥克風或權限拒絕能立即提示親切的引導說明，不再出現畫面假死僵住現象。

---

## [1.3.2] - 2026-08-05

### Added & Verified (新增與驗證)
- **E2E Pitch Stream Contrast Assertion Suite (Playwright E2E 即時圖表高對比度自動化檢驗清單)**:
  - Added `--stream-ink` and `--stream-axis` tokens to Playwright's automated contrast ratio test across all 30+ light and dark themes. Verified 100% PASS with 0 contrast violations.
    將即時圖表數據線 (`--stream-ink`) 與刻度文字 (`--stream-axis`) 正式加入 Playwright 瀏覽器自動化審查，全系統 30+ 款主題 100% 乾淨通過 41 項 E2E 測試。

---

## [1.3.1] - 2026-08-05

### Fixed (修復)
- **Dark Theme Pitch Stream & Canvas Contrast Fix (深色主題即時 Hz 圖表對比度修正與架構優化)**:
  - Fixed dark theme contrast bugs where canvas pitch stream lines and axis text appeared pitch-black (#000) on dark gray backgrounds.
    修復深色主題 (Dark Theme) 下即時 Hz 圖表背景色與數據線變黑不可見的對比度問題。
  - Refactored `base.css`, `intonation-visual.js`, and `realtime-pitch-stream.js` to dynamically query theme CSS variables (`--band-gray`, `--stream-ink`, `--stream-axis`, `--chart-bg`), ensuring 100% WCAG high-contrast readability across all 30+ light and dark themes.
    全面導入 CSS 變數動態讀取機制，徹底消滅硬編碼顏色的歷史遺留，確保所有 30+ 款主題在深色與淺色模式下都具有高對比清晰讀取度。

---

## [1.3.0] - 2026-08-05

### Changed & Unified (變更與統一)
- **Single Authoritative Score UI in Professional Mode (專業進階模式單一權威評分介面)**:
  - Hidden the redundant legacy basic meter (`#meter`) in Professional Mode to eliminate percentage confusion (no competing or duplicate % displays on screen).
    在進階專業模式中隱藏舊版基礎雙向儀表 (`#meter`)，解決畫面上同時出現 3 個相異百分比（Feminine % / Masculine % / 進階嚴格分數 %）的混淆問題。
  - Maintained 100% of underlying raw model inference, binary probabilities, telemetry, and JSON data exports untouched.
    100% 完整保留底層神經網絡推論、原始機率分布、Telemetry 數據與 JSON 匯出邏輯，資料零抹除。
  - Simplified the score label to clean **Feminine Tendency (女性化傾向)**, removing technical jargon ("Strict Score / 進階嚴格分析").
    將分數標籤簡化為客觀直觀的「女性化傾向」，去除「進階嚴格分析」等硬核用語。

---

## [1.2.2] - 2026-08-05

### Fixed & Refactored (修復與優化)
- **Advanced Hero UI Layout Optimization (進階模式英雄卡視覺優化)**:
  - Unified the duplicate 7rem giant percentage block into a sleek, balanced 3-column identity card alongside Voice Age, Character Archetype, and Pitch Median. Eliminates visual clutter while enhancing hierarchy.
    將進階模式過往重複的大數字百分比塊重構整合至 3 欄式聲音身份卡，去除視覺雜亂與重複壓迫感，整體介面更加精緻優雅。

---

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
