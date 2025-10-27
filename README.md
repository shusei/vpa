# Voice Presentation Analyzer (Browser-only)

在 **瀏覽器端** 本地推論，快速查看聲音被模型感知為 **女性化／男性化** 的傾向。**不會上傳音檔**。

- Demo：https://shusei.github.io/vpa （支援子路徑部署）
- 可錄音或上傳 mp3 / m4a / mp4 / mov，完成後顯示傾向儀表、統計與簡評。
- 內建 30+ 顏色主題，支援自動／淺色／深色派記憶。

---

## 特色

- **隱私**：100% 本地推論，音檔不離開裝置。僅保留「最新一段」的回放 URL，換檔即釋放。
- **即時**：錄完立刻分析；支援錄音或上傳 mp3 / m4a / mp4 / mov。
- **主題系統**：可在 Auto／淺色／深色派中切換 30+ 主題，會記住上一個偏好。
- **即時音高 Stream**：錄音期間顯示 50–450 Hz 走勢、音高分區、瞬時音量；上傳檔也會離線抽樣以供統計。
- **統計＋簡評**：分析後輸出 Pitch / Volume 百分位數、環境底噪、SNR 與一行簡評，並提示指標分歧。
- **整段優先**：音檔 ≤ **150 秒** 走「整段一次、不分段」。
- **長檔穩定**：>150 秒自動 **串流分段**（預設 12s 窗 / 3s 位移），若記憶體不足會自動降載至 8 / 6 / 4 秒，避免 WASM OOM。
- **自適應 VAD**：長檔可偵測靜音區段，只「選段」讓有效語音優先推論（不更動原音）。
- **最小前處理**：僅混單聲道 + 16 kHz 重採樣（為符合模型）；**不**去靜音、**不**調音量、**不**改內容。
- **進度與 ETA**：長檔顯示片段進度、百分比、預估剩餘時間。
- **回放**：分析後可一鍵回放「剛剛那段」（只保留最新一段；舊音檔自動釋放）。
- **快取**：模型 ONNX 會存於 IndexedDB，下次更快。

---

## 模型與方法

- 模型：`prithivMLmods/Common-Voice-Gender-Detection-ONNX`（基於 Wav2Vec2 的二分類：女性／男性）  
  Model card：https://huggingface.co/prithivMLmods/Common-Voice-Gender-Detection （授權 Apache-2.0）
- 推論引擎：`@xenova/transformers`（Transformers.js，瀏覽器 ONNX Runtime；支援 WebGPU → WASM 回退）
- 前處理（最小化）：立體聲 → 單聲道混合；取樣率 → 16 kHz 重採樣。
- 推論策略：
  - ≤ 150s：**整段一次、不分段**。
  - > 150s：**串流分段**（12s 窗 / 3s 位移），若記憶體不足自動降載 8/6/4s。
- VAD：長檔時可啟用自適應 VAD，排除長靜音後再推論（語音內容不變）。
- 聚合：長檔以 **對數勝算**（log-odds）做時間加權聚合，盡量貼近整段一次結果。
- 解碼備援：優先 WebAudio；失敗自動落到 `ffmpeg.wasm`（ESM 優先、UMD 備援），轉完釋放記憶體。
- 透明度：顯示進度、ETA、匯報使用的 device（WebGPU / WASM），以及即時音高 / 音量圖表。

---

## 統計與簡評

分析完成後會在儀表下方輸出 Statistics 卡，內容包含：

- Pitch：平均、Median、5th / 95th 百分位數，並判斷常見音高區（男性常見區、重疊帶、女性常見區等）。
- Volume：平均、Median、標準差（σ）、5th / 95th 百分位數，幫助掌握音量穩定度。
- Environment：估算環境底噪（10th 百分位）與 SNR，提供錄音環境建議。
- 簡評：整合模型傾向（Female / Male %）、音高區、SNR、變化幅度，偵測「指標分歧」並給予提示。
- 取樣提醒：若錄音期間有聲樣本不足，會提醒延長語句以提升可信度。

---

## 用途定位與免責

- 這個分數是 **模型對語音表現的傾向**（feminine/masculine），不是性別認同，也不是醫療／法律判定。
- 請把它當作 **自我練習的回饋**；不要用來評價他人或從事任何歧視行為。
- **灰色帶**：分數介於 **40–60%** 屬於模糊區，建議多錄幾段觀察趨勢。

**已知侷限**
- 模型主要以 Mozilla Common Voice 的 **英語朗讀**資料訓練；中文／方言／唱歌／戲腔可能有落差。
- 噪音、回音、鼻音重、感冒、僅「硬拉高音高」等，都可能造成偏差。

---

## 使用者快速開始

1. 打開：https://shusei.github.io/vpa
2. 按「開始錄音」說話 5–10 秒（非唱歌），再按停止；或右下角「上傳」選擇 mp3/m4a/mp4/mov。
3. 查看儀表、即時音高圖與百分比；統計卡會顯示簡評、音高／音量分佈與環境建議。
4. 錄音建議：環境安靜、麥克風距離 10–15 cm、用日常對話音量與語速。
   - iOS Safari 上傳語音備忘錄：在 iPhone「語音備忘錄」→ 分享 → 存到檔案（Files），本頁上傳時選「瀏覽」。

---

## 部署（站長）

### A) GitHub Pages（預設、無後端）
1. 將 `index.html` 與 `assets/` 推到公開 repo，啟用 GitHub Pages。
2. 預設 `window.INFERENCE_MODE = 'browser'`（不需更動）。
3. 首次載入會下載 ONNX 模型（數十 MB），之後使用快取。

### B) Cloudflare Pages（可選，作為 HF API 代理）
若需要 serverless 代理 `/api/classify`，請在 Pages Functions 設定 `HUGGING_FACE_TOKEN`，前端改為：

```html
<script>
  window.INFERENCE_MODE = 'server';
  window.API_BASE_URL   = 'https://<project>.pages.dev/api/classify';
</script>
```

本專案預設 **純前端**；僅在必要時啟用後端。

---

## 相容性與表現

- 瀏覽器：Chrome / Edge / Firefox / Safari（近期版本）。
- 效能：短檔（幾秒到數十秒）幾乎即時；長檔會自動分段並顯示 ETA。  
  若裝置記憶體吃緊會自動縮短分段長度，以避免 WASM OOM。
- 支援格式：`audio/*`, `.m4a`, `.mp3`, `.wav`, `.mp4`, `.mov`, `video/mp4`, `video/quicktime`  
  （影片僅取音軌；WebAudio 解不動時自動落到 ffmpeg.wasm）
- 主題與偏好：主題設定會記錄於 `localStorage`；可清除瀏覽器資料重置。
- 隱私與快取：
  - 音檔不會上傳；推論在瀏覽器完成。
  - 模型快取存於 IndexedDB。要釋放，可用頁面上的「清除模型快取」或清除網站資料。

---

## 專案結構

```
.
├─ index.html            # UI / 主題設定 / 說明 / 免責 / 統計卡
├─ assets/
│  ├─ styles.css         # 造型（主題色盤、儀表、Pitch Stream、Statistics 卡）
│  └─ app.js             # 錄音、即時 Pitch Stream、解碼備援、
│                        #   自適應 VAD、整段推論（≤150s）、長檔串流分段、
│                        #   對數勝算聚合、統計與簡評、回放、GC、安全釋放、進度心跳
```

---

## 版本與授權

- 網站程式碼：依本 repo 授權（例如 MIT）。
- 模型：`prithivMLmods/Common-Voice-Gender-Detection`（Apache-2.0）
- 致謝：`@xenova/transformers`、`@ffmpeg/ffmpeg`（Transformers.js / ffmpeg.wasm）

**版本**：v2025-10-29

---

## 變更紀錄（摘要）

- **v2025-10-29**
  - 新增主題系統（Auto／淺色／深色派，30+ 主題，記憶偏好）。
  - 新增即時 Pitch Stream（音高／音量）與錄後統計抽樣。
  - 新增 Statistics 卡與一行簡評、SNR、指標分歧提示。
  - 加入自適應 VAD，長檔排除靜音後再推論。
  - 顯示使用 device（WebGPU / WASM）、環境噪音建議與取樣提醒。

- **v2025-10-22**
  - 新增：長檔 **串流分段** 模式（12s / 3s），遇記憶體不足自動降載 8/6/4s。
  - 維持短檔 **整段一次**；不去靜音、不調音量。
  - 新增進度與 ETA；上傳影片檔自動 ffmpeg.wasm 備援。
  - 強化 GC：釋放 ObjectURL、關閉 AudioContext、清除暫存。
  - 新增回放按鈕；README/說明更新。
