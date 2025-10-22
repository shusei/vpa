# Voice Presentation Analyzer (Browser-only)

在 **瀏覽器端** 本地推論，快速查看聲音被模型感知為 **女性化／男性化** 的傾向。**不會上傳音檔**。  
Demo：https://shusei.github.io/vpa （或子路徑部署）

---

## 特色

- **隱私**：100% 本地推論，音檔不離開裝置。
- **即時**：錄完立刻分析；支援上傳 mp3 / m4a / mp4 / mov。
- **整段優先**：音檔 ≤ **150 秒** 走「整段一次、不分段」。 
- **長檔穩定**：>150 秒自動 **串流分段**（預設 12s 窗 / 3s 位移），若記憶體不足會自動降載至 8 / 6 / 4 秒，避免 WASM OOM。
- **最小前處理**：僅混單聲道 + 16 kHz 重採樣（為符合模型）；**不**去靜音、**不**調音量、**不**改內容。
- **進度與 ETA**：長檔顯示片段進度、百分比、預估剩餘時間。
- **回放**：分析後可一鍵回放「剛剛那段」（只保留最新一段；舊音檔自動釋放）。
- **快取**：模型 ONNX 會存於 IndexedDB，下次更快。

---

## 模型與方法

- 模型：`prithivMLmods/Common-Voice-Gender-Detection-ONNX`（基於 Wav2Vec2 的二分類：女性／男性）  
  Model card：https://huggingface.co/prithivMLmods/Common-Voice-Gender-Detection （授權 Apache-2.0）
- 推論引擎：`@xenova/transformers`（Transformers.js，瀏覽器 ONNX / WASM / WebGPU）
- 前處理（最小化）：立體聲 → 單聲道混合；取樣率 → 16 kHz 重採樣。
- 推論策略：
  - ≤ 150s：**整段一次、不分段**。
  - > 150s：**串流分段**（12s 窗 / 3s 位移），若記憶體不足自動降載 8/6/4s。
- 聚合：長檔以 **對數勝算**（log-odds）做時間加權聚合，盡量貼近整段一次結果。
- 透明度：全程呈現進度與 ETA；影片檔解不動時自動使用 **ffmpeg.wasm** 轉 16k/mono WAV 後再推論（轉檔完成即釋放）。

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
3. 查看儀表與百分比；可按「播放剛才的聲音」重聽原音。
4. 錄音建議：環境安靜、麥克風距離 10–15 cm、用日常對話音量與語速。  
   iOS Safari 上傳語音備忘錄：在 iPhone「語音備忘錄」→ 分享 → 存到檔案（Files），本頁上傳時選「瀏覽」。

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
- 隱私與快取：
  - 音檔不會上傳；推論在瀏覽器完成。
  - **不囤錄音**：只保留最新一段的回放音檔；換檔會釋放舊 URL。
  - 模型快取存於 IndexedDB。要釋放，可用頁面上的「清除模型快取」或清除網站資料。

---

## 專案結構

```
.
├─ index.html            # UI / 說明 / 免責 / 隱私 / 版本
├─ assets/
│  ├─ styles.css         # 造型
│  └─ app.js             # 錄音、解碼（WebAudio→ffmpeg.wasm 備援）、
│                        #   整段推論（≤150s）、長檔串流分段（12→8→6→4s 自動降載）、
│                        #   對數勝算聚合、回放、GC、安全釋放、進度心跳
```

---

## 版本與授權

- 網站程式碼：依本 repo 授權（例如 MIT）。
- 模型：`prithivMLmods/Common-Voice-Gender-Detection`（Apache-2.0）
- 致謝：`@xenova/transformers`（Transformers.js）

**版本**：v2025-10-22

---

## 變更紀錄（摘要）

- **v2025-10-22**
  - 新增：長檔 **串流分段** 模式（12s / 3s），遇記憶體不足自動降載 8/6/4s。
  - 維持短檔 **整段一次**；不去靜音、不調音量。
  - 新增進度與 ETA；上傳影片檔自動 ffmpeg.wasm 備援。
  - 強化 GC：釋放 ObjectURL、關閉 AudioContext、清除暫存。
  - 新增回放按鈕；README/說明更新。
