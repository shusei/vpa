# Changelog

所有本專案的顯著變更都將記錄於此檔案中。

本專案遵循 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.0.0/) 格式，並實施 [語意化版本 (Semantic Versioning)](https://semver.org/lang/zh-TW/)。

## [1.2.0] - 2026-08-05

### 變更與校正 (Changed & Refactored)
- **科學誠實度與多語系文案安全重構**:
  - 全面優化英、繁中、簡中、日文（en / zh-Hant / zh-Hans / ja）自然母語表達，確保符合專業聲學倫理與各國語境。
  - 將共鳴頻譜標籤由器官定位名稱（胸腔/面罩/頭腔）修訂為客觀聲學能量帶 (`低頻能量` / `中頻能量` / `高頻能量`)。
  - 將「下一個挑戰」與自動建議調整為無壓力的客觀聲學觀察與放鬆探索提示，避免機械式開出發聲處方。
  - 調整 `falsettoContrast` 為 `metricMismatch` (指標分歧)，並強化所有介面之實驗性聲學說明與發聲安全提醒。

---

## [1.1.2] - 2026-08-05

### 新增 (Added)
- **GitHub Shields 質感徽章**: 於 README 頂端新增 MIT License、WebGPU Accelerated、Privacy 100% Local 與 GitHub Release 即時動態徽章。

---

## [1.1.1] - 2026-08-05

### 新增 (Added)
- **多語系 README 文件**: 建立 `README.md` (English)、`README.zh-Hant.md` (繁體中文)、`README.zh-Hans.md` (簡體中文) 與 `README.ja.md` (日本語)，並於頂端新增跨語言快捷切換列。

---

## [1.1.0] - 2026-08-05

### 新增 (Added)
- **快速測試模式 (Quick Test Experience)**：提供精簡直觀的單句練習與每日挑戰 (Daily Test)，方便快速評估語音呈現。
- **標準語音測試 (Standard Voice Challenge)**：支援連續 3 句語音測試，計算跨語句的聲音穩定度與綜合評分。
- **社群卡片與影音分享 (Dynamic Social Sharing)**：
  - 支援繪製專屬語音結果動態卡片 (PNG) 與短影片。
  - 整合 X (Twitter)、Threads、LINE 社群一鍵分享與短網址產出。
- **即時播放控制 (Play/Pause Toggle)**：
  - 快速測試結果卡片上的播放按鈕支援「播放/暫停」切換，音訊播放中再按一次可直接暫停，再次按壓從當前位置繼續播放。
- **句庫練習抽屜 (Practice Drawer)**：
  - 內建 Core-36 經典句子庫，支援錄音、回聽、成績自動填回與練習歷程記錄。
- **語調與聲學進階指標**：
  - 統計卡新增語調曲線圖 (Intonation Curve)、連音比例 (Connected Speech)、語速與聲學亮度標籤。

### 變更 (Changed)
- **純 ESM 架構重構**：將過往單體腳本重構為模組化 `src/` 與 `assets/js/` 結構，提升代碼維護性與載入效率。
- **深色主題對比度優化**：修復深色主題下字體與圖表刻度之對比度問題，強化可讀性。
- **說明文件更新**：更新 README.md 補齊快速測試模式、挑戰分享與句庫練習相關說明。

---

## [1.0.0] - 2026-08-01

### 初始版本 (Initial Release)
- **100% 瀏覽器端推論**：基於 Transformers.js 與 ONNX Runtime，音訊檔案與錄音完全不離身，保護隱私。
- **離線運作與快取**：首次下載模型後快取於 IndexedDB，後續造訪支援完全離線分析。
- **雙模式硬體推論**：自動根據瀏覽器環境選擇 WebGPU 或 WASM 進行模型推論與視窗串流。
- **多語系支援**：預設繁體中文，支援簡體中文與英文即時切換。
- **多主題系統**：內建 30+ 派別與 Lux 系列主題，支援自動跟隨系統深淺色模式。
