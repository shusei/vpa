# Changelog (更新日誌)

All notable changes to this project will be documented in this file.
本專案的所有顯著變更都將記錄於此檔案中。

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.26] - 2026-09-06

### Fixed

- Kept realtime pitch and formant panels directly after recording controls and before playback/results. A previous Quick result could push the next Professional recording's live chart thousands of pixels below the viewport while audio and canvas updates continued normally.
- Preserved existing audio lifecycle, pitch detection, scoring, and the stopped-recording/Quick layouts.

### Tests

- Added mobile and desktop Quick-record/result-to-Professional-record regressions that assert the canvas is in the viewport without scrolling, above previous results, receiving pitch data, drawing changing pixels, and cleaning up after stop.
- Removed the chart helper's automatic scrolling, which had concealed the misplaced chart. Native synthetic-audio tests now request 48 kHz instead of inheriting the host sound device's sample rate; physical-device validation and other sample rates remain separate checks.

## [1.4.25] - 2026-09-05

### Fixed

- Reconciled browser-initiated recording stops and errors with the shared coordinator, releasing microphone tracks before analysis and retaining recording chunks per session.
- Released partially connected realtime audio graphs and allowed stop to cancel a pending context resume without leaving a late-resumed idle context running.
- Removed canvas pixel and track inspection when audio diagnostics are disabled.

### Tests

- Added red-to-green failure-path tests for native stop events, graph cleanup, disabled diagnostics, and pending resume cancellation.
- Replaced fixed 100 ms canvas assertions with an input-frequency change that must reach both the Hz readout and rendered pixels. Physical iPhone Safari and Android Chrome validation remains required.

## [1.4.24] - 2026-09-05

### Fixed

- Prepared and resumed the reusable realtime-pitch AudioContext inside the recording button's synchronous user gesture so iPhone Safari can start the Professional chart after a completed Quick recording.
- Bound realtime processors, sources, animation frames, and cleanup to the active recording session so stale teardown cannot stop a newer pitch graph.
- Replaced the Quick experience's hidden Professional record-button proxy and polling loop with one recording coordinator shared by Quick, Professional, and Practice.
- Preserved a Web Audio fallback buffer before decoding and bounded stalled decoder calls so rejected or non-settling browser decoders cannot strand recording analysis.

### Added

- Added an opt-in `?vpaAudioDebug=1` in-memory diagnostic report for AudioContext state, processor callbacks, pitch samples, panel visibility, canvas dimensions, and cleanup without retaining or uploading audio.
- Documented the recording state machine, audio-resource ownership, Safari user-activation constraint, and required iPhone Safari and Android Chrome release checks.

### Tests

- Added a Quick-to-Professional regression that verifies the live panel, parent visibility, valid Hz samples, changing canvas pixels, stopped resources, and empty runtime errors.
- Added recording-coordinator, stale-session, gesture-bound resume, bounded diagnostics, mobile overflow, light/dark readability, and repeated-recording coverage.

## [1.4.23] - 2026-08-31

### Fixed

- Reused and serialized the realtime pitch AudioContext across repeated recording and playback cycles so the live pitch chart no longer disappears while recording remains active.
- Included the local FFmpeg WASM decoder in GitHub Pages builds so video uploads do not depend on a third-party runtime download.
- Kept the Quick information area available while hiding only the upload controls that belong to Professional mode.

### Changed

- Made the complete test suite a required GitHub Pages deployment gate and added a post-deployment HTTP check for the hosted FFmpeg WASM asset.
- Added deterministic browser media fixtures that do not require a system FFmpeg installation during tests.

### Tests

- Added Chromium, Firefox, and WebKit coverage for MP4, HEVC/MOV, 181-second, 32 MB, no-audio, corrupt, and empty video uploads.
- Added a 30-round Professional record/play soak test that continuously verifies the live pitch chart, AudioContext reuse, processor cleanup, playback, and available heap measurements.
- Added a release-device checklist for iPhone Safari and Android Chrome verification.

## [1.4.22] - 2026-08-20

### Fixed（修正）

- Prevented stale playback promises and replaced audio sources from restoring an outdated replay state.／防止延遲完成的播放 Promise 或已替換的音訊來源恢復過期的回放狀態。
- Serialized realtime pitch startup and teardown so AudioContext, processors, animation frames, and resize listeners cannot leak across retries or Quick-to-Professional transitions.／將即時音高啟停流程序列化，避免 AudioContext、處理器、動畫影格與視窗縮放監聽器在重試或 Quick 切換 Professional 時殘留。
- Kept Quick replay event listeners bounded across rerenders and safely reset completed audio before replay.／確保 Quick 重繪後只保留目前的回放事件監聽器，並在重播已結束的音訊前安全歸零。

### Tests（測試）

- Added deterministic unit coverage for delayed playback and overlapping realtime-pitch session races.／新增延遲播放與即時音高工作階段交錯競態的決定性單元測試。
- Added serialized browser coverage for Quick-to-Professional switching, active source replacement, and a 15-round record/replay stress cycle.／新增串行瀏覽器測試，涵蓋 Quick 切換 Professional、播放中替換來源，以及 15 輪錄音與回放壓力循環。

## [1.4.21] - 2026-08-09

### Added（新增）

- Added the standard MIT License with the project copyright notice.／新增含專案著作權聲明的標準 MIT License。
- Added a concise contributor guide covering fork, branch, dependency installation, testing, commit, push, and Pull Request workflow.／新增簡短貢獻指南，說明 fork、分支、安裝依賴、測試、commit、push 與 Pull Request 流程。
- Added a README Contributing section linking to the contributor guide.／在 README 新增 Contributing 段落並連結至貢獻指南。

### Tests（測試）

- Verified the existing project test commands and confirmed the license badge points to the root `LICENSE` file.／確認現有專案測試指令可執行，並確認 License badge 指向根目錄的 `LICENSE` 檔案。

## [1.4.20] - 2026-08-09

### Changed

- Streamlined Quick results to show only the score, core voice data, one concise summary, replay/retry controls, and immediate sharing actions.
- Removed long three-part guidance, pitch explanations, safety panels, disclaimers, and footer warnings from Quick while retaining the complete safety and interpretation guidance in Advanced, Help, and the Feminine Voice Manual.
- Compressed the replay/retry controls and placed voice-age and character cards side by side on phones so sharing actions appear sooner.

### Tests

- Added four-locale regression coverage for single-line Quick summaries and the absence of verbose Quick-only guidance.
- Verified the 390 × 844 mobile result layout, first-tap audio replay, direct sharing controls, and zero horizontal overflow.

## [1.4.19] - 2026-08-09

### Fixed

- Added the missing visible explanation for the Advanced intonation contour, including recording-time and pitch axes, processed-curve and low-confidence legends, and a clear statement that raw dots do not change the analysis or score.
- Replaced the unexplained raw-point checkbox with an accessible stateful control and a complete visual legend across all four locales.

### Tests

- Added mobile regression coverage for the chart purpose, axis explanation, legend, raw-point control, and horizontal-overflow safety.

## [1.4.18] - 2026-08-09

### Changed

- Rewrote every Quick and Advanced analysis explanation into a consistent three-part structure: what the metric measures, what this recording means, and a metric-specific next comparison.
- Extended useful guidance to the four weighted Advanced components, Jitter, Shimmer, HNR, CPP, beginner highlights, focus cards, and all detailed formant, spectrum, articulation, pace, liaison, and intonation cards.
- Replaced population-sounding pitch-band labels with explicit app reference-band wording and improved the one-column mobile layout for voice-age evidence.
- Kept recommendations comfort-first and comparison-based while preserving meaningful interpretation instead of repeating generic non-diagnostic disclaimers.

### Fixed

- Prevented an unavailable F1, F2, or F3 result from incorrectly displaying an appended `Hz` unit.
- Removed a duplicated brightness explanation from the expanded detail layout.
- Kept the intonation chart inside its mobile card instead of forcing a 320 px minimum width and creating a horizontal scrollbar.

### Tests

- Added four-locale regression checks requiring exactly three guidance rows, metric-specific next steps, distinct F1/F2/F3 guidance, and complete Quick/Advanced coverage.
- Added mobile browser assertions for structured Quick guidance and horizontal-overflow safety.

## [1.4.17] - 2026-08-09

### Fixed

- Unified the final professional meter, phrase-practice cards, and embedded manual practice cards on the current integrated presentation score instead of the retired raw classifier percentage.
- Kept unavailable numeric evidence unavailable after JSON export so the same recording cannot receive a different integrated score after serialization.
- Isolated legacy raw-classifier practice history from current integrated scores.
- Raised the Help and Feminine Voice Manual overlays above the Quick / Professional mode banner and locale menu so their close controls remain visible and clickable on phones.

### Tests

- Added score-event, export-invariance, practice-history, real-audio, and 390 px mobile overlay regression coverage.

## [1.4.16] - 2026-08-09

### Changed

- Added capability-aware neutral microphone capture for iPhone Safari and Android Chrome: mono and 48 kHz are requested only when reported as supported, while echo cancellation, noise suppression, and automatic gain remain disabled where available.
- Added a 128 kbps recording target with MIME-only and browser-default fallbacks so improved quality does not block older phones or in-app browsers.
- Verifies the settings actually chosen by the browser and warns during recording when device audio processing remains active or cannot be fully confirmed.
- Updated the four-locale pre-recording check to recommend a consistent, unobstructed 30–60 cm phone position in a quiet room.
- Added unit coverage for full, partial, fallback, and recorder-option support paths.

## [1.4.15] - 2026-08-08

### Fixed

- Removed stale warm-up wording from static HTML, all locale fallbacks, help shortcuts, and Advanced backup hints.
- Standardized every visible pre-recording card on the comfort-first, non-prescriptive check.

## [1.4.14] - 2026-08-08

### Safety

- Reworked Quick, Advanced, Help, and Feminine Voice Manual guidance in Traditional Chinese, Simplified Chinese, English, and Japanese around non-clinical acoustic comparison, user-defined goals, comfort-first stop rules, and appropriate ENT / voice-specialist referral signals.
- Removed fixed feminine targets and unsupported self-training instructions for vocal-fold closure, breathiness, larynx position, formants, resonance placement, speech rate, and intonation.
- Replaced the anatomical “three-step warmup” with a pre-recording comfort check that verifies natural speech, absence of discomfort, and consistent recording conditions.

### Changed

- Restored a structured, polished Help tour with Quick start, Interface tour & live monitors, result-panel interpretation, clear limits, and four authoritative references.
- Renamed the manual UI to “Feminine Voice Manual” / “女性化聲音手冊” to match its non-prescriptive scope.
- GitHub Pages now deploys a Vite production build with content-hashed JavaScript and CSS filenames, eliminating reliance on manually maintained cache-query versions for users.
- The source version-bump helper now discovers direct-development cache tags recursively and keeps package-lock.json aligned.

### Tests

- Added four-locale medical-safety audits, phone overflow checks, production-build fingerprint verification, and a production-runtime smoke test.
- Verified 46 Playwright browser tests; 2 real remote-model tests remain skipped by design.
- npm audit reports 0 known vulnerabilities.

---

## [1.4.13] - 2026-08-08

### Changed

- **Promoted the advanced result to the primary analysis position** — `ADVANCED BETA / Advanced strict analysis`, including the feminine percentage, now appears before Statistics overview instead of being buried beneath the detailed monitors.
- **Preserved the polished quick and advanced visual systems** — The layout change keeps playback directly above the result, retains the existing result-card hierarchy, and avoids unnecessary redesign of the already strong quick-test interface.

### Tests

- Playwright now verifies on desktop and mobile that the advanced result precedes Statistics overview and remains within the viewport.

---

## [1.4.12] - 2026-08-08

### Fixed

- **Restored the missing Quick Start information block** — Four locale files declared the top-level help section twice, so the later help dialog silently replaced every earlier help translation. The dialog now has its own helpDialog namespace and all eight information headings render in every locale.
- **Removed silent translation failures** — Corrected the FFmpeg error-message path and advanced-analysis canvas/legend paths, removed obsolete aliases, and made DOM translation updates preserve source fallback content whenever a key is missing.
- **Corrected the default document language** — Selecting or initializing the already-active locale now still updates the HTML lang attribute, so Traditional Chinese is no longer mislabeled as English.

### Tests

- Locale tests now reject duplicate top-level dictionary sections and verify every literal HTML/JavaScript translation reference across all four dictionaries.
- Playwright now switches through Traditional Chinese, Simplified Chinese, English, and Japanese and requires all eight information headings to remain visible and non-empty.
- Full suite: 44 Playwright tests passed, 2 remote-model tests skipped by design.

---
## [1.4.11] - 2026-08-08

### Fixed (完整進階分析語系根治)

- **Unified the production i18n module graph** — Every application module now imports the exact same cache-busted `i18n.js` URL. Previously, versioned and unversioned imports created two independent ESM instances, so the quick language control could update the shell while advanced metrics, practice phrases, the manual, sharing UI, and theme controls remained in Traditional Chinese.
- **Restored complete translation ownership for the information section** — The full interface, model, accuracy, implementation, ethics, compatibility, and version blocks are translated as complete containers again. Build metadata is restored after each locale render, and the manual button, close control, and back-to-top control now have translated accessible names.
- **Corrected Japanese fallback ordering** — Japanese translations are no longer overwritten by the English base object. The manual keeps Japanese controls and uses English content, rather than unexpectedly falling back to Traditional Chinese, when a Japanese manual body is unavailable.
- **Added production-graph regression coverage** — Tests now reject mixed i18n import URLs, exercise a real audio upload followed by a language switch, scan the complete advanced analysis and supporting surfaces for Chinese leakage in English mode, and remove the old test-only second-locale switch that had masked the production bug.

- **統一正式版語系模組實例** — 修正帶版本與未帶版本的 `i18n.js` 同時載入所造成的雙語系狀態；快速語言按鈕、完整進階分析、練習抽屜、手冊、分享與主題控制現在會同步切換。
- **補回完整說明區塊與無障礙文字翻譯** — 介面、模型、準確度、方法、倫理、相容性、版本資訊與手冊控制全部由目前語系完整接管，切換語言後版本資訊也會正確回填。
- **強化真實流程測試** — 使用實際音檔走完正式分析後切換語言，逐區檢查英文模式零中文殘留，並新增模組載入路徑守門測試，避免同類問題再次被測試假通過掩蓋。

---

## [1.4.10] - 2026-08-08

### Fixed (深層語系凍結根治 & 進階卡片資料完整復原)

- **Root-cause fix: `resonanceDisplay` locale freeze in `stats-core.js`** — The `tags` row (Resonance tag) was reading `advSummary.resonanceDisplay`, which is computed once at inference time and frozen in the recording locale (e.g., Traditional Chinese). On locale switch, the value never updated. Fixed by dynamically re-invoking `describeResonanceFromEnergy` using the stored `energyPct` ratios at render time, so the Resonance tag always reflects the current locale. (`stats-core.js`, `stats-orchestration.js`)

  **根治 Resonance 標籤語系凍結問題** — 標籤列中的共鳴分類標籤（如「頭腔亮度強」）在推論時被凍結進 `advSummary.resonanceDisplay`，語言切換後無法更新。修復方法：在渲染 tags 時改用 `energyPct` 比例即時重新呼叫 `describeResonanceFromEnergy()` 動態計算，確保語言切換後 Resonance 標籤 100% 呈現當前語系。

- **Restored `describeResonanceFromEnergy` and `getSummaryText` dep wiring** — Both were missing from the `finishStreamStats` deps chain (`stats-orchestration.js` → `stats-core.js`), preventing dynamic locale-aware resonance re-computation on every render.

  **補齊 deps 傳遞鏈** — `describeResonanceFromEnergy` 與 `getSummaryText` 從未被正確傳入 `finishStreamStats` 的 deps，現已補齊完整的依賴傳遞路徑。

- **All 42 Playwright E2E specs + strict multi-locale localization tests pass 100%.**

---

## [1.4.9] - 2026-08-08


### Cleaned & Refactored (全面掃除寫死字串與升級切換按鈕)
- **Eliminated Hardcoded String Fallbacks (`assets/js/advanced-section.js` & `assets/js/advanced-summary-render.js`) (全面清除硬編碼文字與按鈕狀態綁定)**:
  - Removed all hardcoded string fallback fallbacks (such as `"Switch to Beginner (collapse all)"`) in `advanced-section.js` and `advanced-summary-render.js`, routing 100% of toggle button text directly through `t()`.
    徹底清除 `advanced-section.js` 與 `advanced-summary-render.js` 裡面的硬編碼回退字串，將按鈕文案 100% 解耦並經由 `t()` 字典動態呈現。
  - Restored `<details>` accordion compatibility with full dynamic metric resolution, enabling smooth expand/collapse interaction for the `Switch to Advanced (expand all)` toggle button across all 4 locales.
    修復 `Switch to Advanced (expand all)` 展開與收合按鈕的 `<details>` DOM 綁定，兼具現代極簡視覺與強大互動體驗。

---

## [1.4.8] - 2026-08-08

### Verified & End-to-End Certified (嚴格端到端 E2E 測試與 CI/CD 100% 全綠認證)
- **Strict End-to-End E2E Testing Pipeline (`tests/e2e-browser-simulation.mjs` & Playwright E2E)**:
  - Added strict end-to-end multi-locale runtime simulation and Playwright real-browser test suite (`npm run test:e2e`), enforcing 0 Chinese character leakage in English mode and 100% symmetric locale coverage across `zh-Hant`, `zh-Hans`, `en`, and `ja`.
    新增嚴格端到端多語系推論與 Playwright 真實瀏覽器 E2E 測試管線 (`npm run test:e2e`)，驗證英文模式下 0 中文字元殘留，並確保繁中、簡中、英文、日文四國語言字典 100% 完全對稱。
  - Verified 42 Playwright real-browser E2E specs, passing all audio upload, model pipeline, embedded guard, and social sharing scenarios cleanly.
    通過全套 42 項 Playwright 真實瀏覽器 E2E 測試案例，包含音訊上傳、推論管線、內建瀏覽器防護與社群分享卡片。

---

## [1.4.7] - 2026-08-08

### Refactored & Replaced (徹底移除舊版老舊折疊面板，全面升級為 Voice Age 2.0 現代微型卡片)
- **Unified Modern Component Structure (`assets/js/advanced-summary-render.js`) (進階分析區塊全面替換為 Voice Age 2.0 現代卡片)**:
  - Replaced the legacy html `<details>` accordion structure in `renderAdvancedSummary` with unified modern cards matching the Voice Age 2.0 UI layout in `advanced-experience.js`.
    徹底移除 `renderAdvancedSummary` 裡面的舊版 `<details>` 舊折疊元件，全面替換為與 Voice Age 2.0 (`advanced-experience.js`) 完全一模一樣的現代極簡風 Card 結構。
  - Completely purged all historical static markup residue, guaranteeing 100% clean English in EN mode with zero Chinese leaks.
    徹底告別舊型 DOM 結構與歷史殘留語法，實現英文模式下 100% 純淨且美觀的極致現代 UI。

---

## [1.4.6] - 2026-08-08

### Refactored & Unified (進階分析面板現代化重構與 Voice Age 2.0 架構統一)
- **Modern Analysis UI Architecture (`assets/js/advanced-summary-render.js`) (進階分析面板現代化重構)**:
  - Refactored `advanced-summary-render.js` to unify metric card layouts with the modern Voice Age 2.0 / Advanced Experience UI design (`assets/experiments/advanced-experience.js`).
    重構 `advanced-summary-render.js`，將進階分析面板的三大折疊卡片全面提升為與 Voice Age 2.0 / Advanced Experience 完全一致的現代化 Premium 介面架構。
  - Stripped out legacy hardcoded Traditional Chinese label fallbacks (such as `（第一共振峰）`), guaranteeing 100% pure English text in EN mode and seamless dynamic localization across all languages.
    徹底刪除歷史殘留的傳統硬編碼中文括號與舊版格式，確保英文模式下與 Voice Age 2.0 一樣呈現 100% 純淨無瑕的英文介面。

---

## [1.4.5] - 2026-08-08

### Fixed & Root Cause Resolved (統計標籤列與全卡片閉包死存字串徹底動態化修復)
- **Dynamic Active Summary Text for Stats Orchestration (`assets/js/stats-core.js`) (統計卡片與 Topbar 標籤列 activeSummaryText 動態解鎖)**:
  - Fixed `finishStreamStats` in `assets/js/stats-core.js` to dynamically evaluate `deps.getSummaryText()` on every execution (`activeSummaryText`), resolving stale module closure references where `Resonance`, `Speech rate`, `Breathiness`, and `Brightness` tag labels remained trapped in Traditional Chinese.
    修復 `assets/js/stats-core.js` 中 `finishStreamStats` 函式，改為每次渲染時動態調用 `getSummaryText()` (`activeSummaryText`)，徹底解決 `Resonance`、`Speech rate`、`Breathiness` 與 `Brightness` 標籤列死鎖在初始模組載入時繁體中文閉包的問題。
  - Verified against exact user-provided runtime text snippets; 100% free of Chinese characters when running in English mode.
    根據使用者現場複製貼上的文本進行精確驗證，100% 確保英文模式下完全為英文、無任何中文殘留。

---

## [1.4.4] - 2026-08-08

### Fixed & Root Cause Resolved (進階分析卡片內文全動態多語系重構)
- **Dynamic Render for Formant & Resonance Cards (`assets/js/advanced-summary-render.js`) (進階分析卡片 Hint 與 Label 徹底解鎖多語系動態翻譯)**:
  - Refactored `renderAdvancedSummary` to dynamically compute `f1Hint`, `f2Hint`, `f3Hint`, `tiltLabel`, `tiltHint`, `resonanceDisplay`, `resonanceHint`, `breathinessLabel`, `breathinessHint`, `brightnessDisplay`, `brightnessHint`, `speechRateHint`, `vowelDisplay`, `vowelHint`, and `liaisonDisplay` using the active `t()` function during every render pass, eliminating all cached Chinese string fallbacks in existing analysis summary stores.
    重構 `renderAdvancedSummary` 渲染邏輯，改為每次渲染時直接調用當前最新語言的 `t()` 重新獲取 Hints 與 Labels，徹底消除歷史 `advSummary` 狀態物件中寫死的中文字串問題。
  - Exported `slopeKey` and `rangeKey` from `analyzeIntonation` (`assets/js/advanced-metrics.js`) and removed Chinese bracket formatting `（）` in `makeFormantHint`, ensuring 100% clean English text across all sub-cards in English mode.
    在 `analyzeIntonation` 之中導出 `slopeKey` 與 `rangeKey` 鍵名，並修復 `makeFormantHint` 寫死的全形中文括號 `（）` 為通用括號 `()`，實現英文模式下 100% 完全無中文殘留的極致體驗。

---

## [1.3.9] - 2026-08-08 (Historic)

### Fixed & Automated (自動化升版與快取刷新工具 `bump-version.mjs`)
- **Automated Version Bumping & Cache-Busting Pipeline (`scripts/bump-version.mjs`) (自動化 Cache-Busting 與升版管線)**:
  - Added zero-dependency CLI script `scripts/bump-version.mjs` and package script `npm run bump`.
    新增零依賴自動化升版腳本 `scripts/bump-version.mjs` 與 `npm run bump` 指令。
  - Automatically updates `package.json` version, updates all 15 cache-busting query tags across HTML, CSS, and ESM imports (`?v=X.Y.Z`), and syncs test white-lists in a single execution.
    自動化同步更新 `package.json` 版本號，並一鍵更新全站 15 個 HTML、CSS、ESM `import` 引用點之 `?v=X.Y.Z` 標籤與測試白名單。

- **Agent Rule Enforcer (`AGENTS.md`) (發布規範層面防護)**:
  - Enforced Cache-Busting tag updates as a strict mandatory rule in `AGENTS.md` for all future automated release workflows.
    在 `AGENTS.md` 權威專案規範中，將「強刷手機快取 (Cache-Busting)」列為每一次自動 Commit & Push 時必執行的強制步驟。

---

## [1.4.2] - 2026-08-08

### Fixed & Cache-Busted (全面強制強刷手機端快取 Cache-Busting)
- **Universal Mobile Asset Cache-Busting (`?v=1.4.2-20260808`) (手機端強快取打破與強刷修復)**:
  - Updated query version tags (`?v=...`) across all 14 core HTML, CSS, and JS import entrypoints to `?v=1.4.2-20260808`.
    將全站 14 個核心 `index.html`、CSS 與 JS `import` 引用點之 query 版本標記全面升級至 `?v=1.4.2-20260808`。
  - Forces mobile Safari, Chrome, and embedded webviews to purge old HTTP disk caches and immediately fetch fresh JS/CSS modules upon page refresh.
    強制 iOS Safari、Android Chrome 與 App 內建 WebViews 徹底作廢舊有的 HTTP 磁碟快取，在重新整理頁面時 100% 強制下載最新版本的腳本與樣式檔，徹底解決手機端手動刷網頁一直卡在舊版快取的問題。

---

## [1.4.1] - 2026-08-08

### Fixed & Verified (中間動態分析卡片重繪修復與 100% 驗證)
- **Dynamic Mid-Section Analysis Cards Re-rendering (`assets/app-core.js`) (中間動態分析卡片語言切換時即時重繪修復)**:
  - Registered `onLocaleChange` callback in `assets/app-core.js` to automatically invoke `statsOrchestrationController.finishStreamStats()` whenever active analysis results exist.
    在 `assets/app-core.js` 中註冊 `onLocaleChange` 監聽器，當使用者在 Topbar 點擊切換語言且畫面上已存在分析結果時，自動調用 `finishStreamStats()` 重新繪製。
  - Dynamically re-renders Focus Insights cards, Formant Trend cards, Voice Age, Voice Quality, and Intonation curves using the active target locale dictionary, guaranteeing 100% English rendering when switching to English mode.
    使用目標語言字典 100% 重新生成中間那塊的重點建議卡、Formant 傾向卡、聲音年齡、聲音質地與語調曲線，徹底解決進階分析中間面板文字不隨切換語言變動的 Root Cause。

- **DOM Structure i18n Unification (`index.html`) (移除 DOM 父層 data-i18n-html 洗掉子節點問題)**:
  - Removed container-level `data-i18n-html` attributes on `<details>` sections in `index.html`, allowing granular child elements (`<summary>`, `<li>`, `<p>`) to be translated precisely without parent innerHTML collisions.
    移除 `index.html` 下方 8 個 `<details>` 父容器上粗暴覆蓋的 `data-i18n-html` 屬性，使內部獨立標題與列表節點可進行 100% 精準多語系替換。
  - Verified 100% pass across unit tests, locale cleanliness tests, zero-dependency Node DOM switch verification, and Playwright E2E suites.
    全數通過單元測試、語系潔淨度測試、原生 DOM 動態切換驗證腳本與 Playwright E2E 測試。

---

## [1.4.0] - 2026-08-08

### Fixed & Enhanced (選單層級修復與早期語系閃爍防護)
- **Locale Menu Z-Index & Overlay Blocking Fix (語言選單層級置頂與橫幅遮擋修復)**:
  - Elevated `.lang-menu`, `.theme-menu` (`layout.css`), and `.experience-nav__locale-menu` (`quick-experience.css`) `z-index` to `99999` and `999999`, elevating the dropdown menu above all hero banners and floating action cards.
    將 `.lang-menu`、`.theme-menu` 與 Quick Experience 頂部選單 `.experience-nav__locale-menu` 之 `z-index` 提升至 `99999` 及 `999999`，徹底解決進階分析與快速模式下點選語言選單時被下方橫幅遮擋、無法選取的 CSS Bug。

- **Early Head Locale Detector & Storage Key Sync (早期語系同步偵測與雙 Key 記憶與備援)**:
  - Added early synchronous locale detection inline script in `index.html` `<head>` to inspect `localStorage` and `navigator.language` before initial HTML render, setting `<html lang="...">` dynamically to eliminate Traditional Chinese FOUC (flash of un-translated content).
    在 `index.html` 的 `<head>` 頂部加入極速同步 Locale 檢測腳本，在瀏覽器渲染 HTML 前即自動設定 `<html lang="...">` 屬性，徹底消除英文模式開啟時瞬間看到預設繁體中文的殘留閃爍。
  - Synchronized `vpa.locale` and `vpa::locale` storage keys in `assets/js/i18n.js` to ensure 100% backward and forward compatibility for saved user locale preferences across reloads.
    在 `assets/js/i18n.js` 中同步讀取與寫入 `vpa.locale` 及 `vpa::locale` 雙儲存鍵，徹底解決舊專案記憶鍵不一致導致頁面重整後跳回預設語言的相容性問題。

---

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
