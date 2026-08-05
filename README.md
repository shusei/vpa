# Voice Presentation Analyzer (VPA)

> 100% 瀏覽器端推論的聲音呈現分析與練習工具，從錄音或上傳的語音片段中推估女性化 (Feminine) / 男性化 (Masculine) 呈現傾向，提供快速挑戰測試、句庫練習、即時聲學監控、動態卡片分享與進階統計摘要。

- 🌐 **Demo (GitHub Pages)**：https://shusei.github.io/vpa
- 📋 **更新日誌 (Changelog)**：[CHANGELOG.md](./CHANGELOG.md)
- 🔬 **演算法驗證報告**：[ALGORITHM_VERIFICATION.md](./ALGORITHM_VERIFICATION.md)
- 🔒 **隱私防護**：音訊與模型推論 100% 於本機瀏覽器完成，音檔絕不上傳任何伺服器。
- 🌍 **多語系支援**：繁體中文 / 簡體中文 / English，自動記憶預設偏好。

---

## 目錄

- [產品特色與雙模式體驗](#產品特色與雙模式體驗)
  - [1. 快速測試模式 (Quick Test Experience)](#1-快速測試模式-quick-test-experience)
  - [2. 進階專業模式 (Professional Experience)](#2-進階專業模式-professional-experience)
  - [3. 句庫練習抽屜 (Practice Drawer)](#3-句庫練習抽屜-practice-drawer)
  - [4. 社群分享與動態卡片 (Social Sharing)](#4-社群分享與動態卡片-social-sharing)
- [核心能力一覽](#核心能力一覽)
- [操作流程](#操作流程)
- [輸出解讀指南](#輸出解讀指南)
- [技術架構](#技術架構)
- [開發與測試](#開發與測試)
- [常見問題與疑難排解](#常見問題與疑難排解)
- [隱私、定位與免責聲明](#隱私定位與免責聲明)
- [版本資訊與授權](#版本資訊與授權)

---

## 產品特色與雙模式體驗

Voice Presentation Analyzer（VPA）支援雙重介面體驗，無論是想快速測量單句聲線，或是進行專業聲學指標分析，都能一鍵切換：

### 1. 快速測試模式 (Quick Test Experience)
- **每日一練 (Daily Test)**：精選經典測試句，幾秒鐘即可完成一次語音檢測。
- **標準測試 (Standard Test)**：連續 3 句測試挑戰，全面評估跨語句的聲音穩定度與一致性。
- **直觀評分卡與即時重播**：顯示聲音傾向百分比、估計聲齡與聲線原型；結果頁上的播放按鈕支援 **播放/暫停 (Play/Pause)** 切換與重放。

### 2. 進階專業模式 (Professional Experience)
- **即時 Pitch Stream 走勢**：50–450 Hz 聲線即時追蹤、瞬時音量與底噪監控。
- **Formant / Resonance 共鳴面板**：估計 F1–F3 共鳴峰、胸腔 / 前置 / 頭腔共鳴比例與氣聲比例。
- **語調與進階統計卡**：提供洋紅語調曲線圖、語速、連音比例、共鳴亮度評估與個人化建議簡評。

### 3. 句庫練習抽屜 (Practice Drawer)
- **Core-36 經典句庫**：內建分類豐富的訓練句型，支援單句快捷錄音、回聽、歷程成績紀錄與自動比對。

### 4. 社群分享與動態卡片 (Social Sharing)
- **動態成果卡片 (PNG / Video)**：一鍵繪製包含聲線傾向、音高 Hz 與評語的動態視覺卡片。
- **社群快捷傳送**：自動整合短網址與圖片影片，支援 X (Twitter)、Threads、LINE 免費短連結一鍵分享。

---

## 核心能力一覽

### 隱私與安全

- **本地推論**：採用 [`@xenova/transformers`](https://github.com/xenova/transformers.js) 的 ONNX Runtime，模型自 Hugging Face Hub 下載後完全於本機執行，音檔絕不離身。
- **離線支援**：首次下載後模型快取於 IndexedDB，日後造訪即可 100% 離線運作。
- **最小化前處理**：僅將輸入混成單聲道並重採樣至 16 kHz，完整保留語音品質與音量細節。

### 互動與主題

- **多主題系統**：內建 30+ 派別與 Lux 系列主題，支援系統深淺色模式自動跟隨與手動切換。
- **錄音與檔案上傳二合一**：支援 MediaRecorder 即時錄音，或直接上傳 `mp3 / m4a / mp4 / mov / wav` 格式檔案（可自動從影片抽取音軌），亦支援拖曳檔案上傳。
- **鍵盤快捷與無障礙 (ARIA)**：支援 <kbd>Space</kbd> 快捷錄音、全介面 `aria-*` 屬性標記與鍵盤友善操作。

---

## 操作流程

1. **開啟網頁**：造訪 <https://shusei.github.io/vpa>。首次造訪時請連網下載模型。
2. **選擇測試體驗**：
   - **快速測試**：點擊「開始測試」，跟隨畫面提示朗讀句子，完成後即可獲得傾向評分、聲齡與原型，並可隨時點按播放按鈕進行「播放 / 暫停」回聽。
   - **進階專業模式**：切換至專業模式，隨時進行 5-10 秒自由發聲錄音或上傳音訊檔，檢視 Pitch 走勢、共鳴面板與詳細統計數據。
3. **句庫練習**：點擊錄音鍵旁「句庫練習」開啟抽屜，挑選 Core-36 句子進行單句反覆訓練與成績追蹤。
4. **社群分享**：在測試結果頁點擊「分享結果」，生成專屬視覺卡片或複製短連結至 X、Threads 或 LINE。

---

## 輸出解讀指南

- **傾向儀表**：顯示模型推估的 feminine / masculine 百分比。40–60% 為灰色過渡帶，建議多錄幾段觀察聲音趨勢。
- **Pitch 音高卡**：包含平均音高 (Hz)、中位數與 5th / 95th 百分位數，標記常見聲域範圍 (50–600 Hz)。
- **Formant & Resonance 共鳴卡**：展示 F1–F3 中位數、胸腔 / 前置 / 頭腔共鳴比例與氣聲比例。
- **語調曲線與進階摘要**：洋紅線描繪語調走勢，灰色區段標示靜音或低信心區，並標註連音比例與說話語速。

---

## 技術架構

```
使用者行為 → MediaRecorder / 檔案上傳 → 音訊解碼 (WebAudio / FFmpeg)
  ↘ WebAudio 即時分析 (Pitch / Formant / Noise)
     ↘ IndexedDB 快取模型 → Transformers.js ONNX Runtime (WebGPU / WASM)
        ↘ 整段 / 串流分段 → Log-odds 聚合 → 傾向評估
           ↘ 統計彙整 (百分位、語速、共鳴、語調曲線)
              ↘ 快速測試 / 專業模式 UI → 社群分享卡片生成
```

- **推論引擎**：`@xenova/transformers` 搭配 `prithivMLmods/Common-Voice-Gender-Detection-ONNX` 模型。
- **分段策略**：自動根據裝置硬體（WebGPU / WASM）與時長選擇最佳串流視窗與 hop。

---

## 開發與測試

### 前置需求
- Node.js 18+
- npm 8+

### 執行測試
```bash
npm test      # 執行單元測試、語法檢查、HTML/CSS 標記檢查與社群分享驗證
```

---

## 常見問題與疑難排解

| 問題 | 可能原因 | 建議作法 |
| ---- | -------- | -------- |
| 無法錄音 | 瀏覽器未授予麥克風權限 | 請在瀏覽器網址列左側允許麥克風權限，或使用檔案上傳功能。 |
| 快速測試重聽 | 播放中再次點擊按鈕 | 播放按鈕具備「播放/暫停」切換功能，播放中按下可直接暫停。 |
| 模型下載緩慢 | 首次造訪時網路較慢 | 重整頁面或等待下載完成，完成後模型會自動快取於 IndexedDB 供離線使用。 |

---

## 隱私、定位與免責聲明

- **非醫療診斷工具**：本工具指標係結合聲學演算法與機器學習模型，僅供自我聲音探索、訓練與語音呈現回饋，**不等同於性別認同、身分判定或任何醫療診斷／法律結論**。
- **發聲安全提醒**：練習時請保持自然舒服的音量。若喉嚨感覺沙啞、疲勞或不適，請立即停止休息並諮詢專業語言治療師或耳鼻喉科醫師。
- **隱私承諾**：所有錄音與聲音分析 100% 於您的個人裝置內完成，資料絕不上傳任何伺服器。

---

## 版本資訊與授權

- **更新細節**：完整版本歷史請參閱 [CHANGELOG.md](./CHANGELOG.md)。
- **專案授權**：MIT License。
- **模型授權**：Apache-2.0 License (`prithivMLmods/Common-Voice-Gender-Detection-ONNX`)。
- **句庫資料**：Core-36 句庫以 CC0-1.0 釋出。

---

## 支援作者

[![Buy Me a Coffee](https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png)](https://buymeacoffee.com/shusei)
