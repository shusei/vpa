# Voice Presentation Analyzer (Browser-only)

> 在 **瀏覽器端** 直接推論，快速查看聲音被模型感知為 **女性化／男性化** 的傾向。**不會上傳音檔**。  
> Demo：<https://shusei.github.io/vpa>

- **隱私**：100% 本地推論，音檔不離開裝置  
- **即時**：錄完立刻分析；上傳 mp3 / m4a / mp4 / mov 皆可  
- **單次整段**：不分段、不去靜音、不調音量（僅混單聲道＋16k 重採樣以符合模型）  
- **快取**：模型 ONNX 檔會緩存至瀏覽器（IndexedDB）；下次更快  
- **回放**：分析後可一鍵回放「剛剛那段」（只保留最新一段，舊音檔會釋放）

---

## 模型與方法

- **模型**：`prithivMLmods/Common-Voice-Gender-Detection-ONNX`（基於 **Wav2Vec2** 的二分類：女性／男性）  
  Model card：<https://huggingface.co/prithivMLmods/Common-Voice-Gender-Detection>（授權 **Apache-2.0**）
- **推論引擎**：[`@xenova/transformers`](https://github.com/xenova/transformers.js)（Transformers.js，瀏覽器 ONNX 推論）
- **音訊前處理（最小化）**  
  立體聲 → **單聲道混合**；取樣率 → **16 kHz 重採樣**。  
  **不**去靜音、**不**正規化音量、**不**分段。

---

## 用途定位與免責

此分數為聲音**被模型感知**為「女性化／男性化」的**傾向**，**不是**性別認同、也**不是**醫療／法律判定。  
請把它當作**自我練習的回饋**，不要用來評價他人或進行任何歧視。

> **灰色帶**：分數介於 **40–60%** 視為模糊區，建議多錄幾段觀察趨勢。

**已知侷限**
- 模型主要以 **Mozilla Common Voice 英語朗讀**資料訓練；中文／方言／唱歌／戲腔可能有落差。
- 大噪音、回音、鼻音重、感冒、僅「硬拉高音高」等，都可能造成偏差。

---

## 使用者快速開始

1. 打開：<https://shusei.github.io/vpa>  
2. 按「**開始錄音**」說話 **5–10 秒**（非唱歌），再按停止；或右下角「**上傳**」選擇 mp3/m4a/mp4/mov。  
3. 查看儀表與百分比；可按「**播放剛才的聲音**」重聽原音。

**錄音建議**：環境安靜、麥克風距離 10–15 cm、用日常對話的音量與語速。  
**iOS Safari 上傳語音備忘錄**：在 iPhone「語音備忘錄」App → 分享 → **存到檔案**（Files），本頁上傳時選「瀏覽」。

---

## 部署（站長）

### A) GitHub Pages（預設、無後端）
1. 將 `index.html` 與 `assets/` 推到公開 repo。  
2. 啟用 GitHub Pages。  
3. 首次載入會下載 ONNX 模型（數十 MB），之後使用快取。

> 預設 `window.INFERENCE_MODE = 'browser'`，無須更動。

### B) Cloudflare Pages（可選，作為 HF API 代理）
若需要 serverless 代理 `/api/classify`，在 Pages Functions 設 `HUGGING_FACE_TOKEN`，前端改：

```html
<script>
  window.INFERENCE_MODE = 'server';
  window.API_BASE_URL   = 'https://<project>.pages.dev/api/classify';
</script>
```

> 本專案預設「純前端」；僅在必要時啟用後端。

---

## 相容性與表現

- **瀏覽器**：Chrome / Edge / Firefox / Safari（近期版本）。  
- **效能**：短檔（幾秒）幾乎即時；長檔取決於裝置性能。  
- **格式**：`audio/*, .m4a, .mp3, .wav, .mp4, .mov, video/mp4, video/quicktime`  
  （影片僅取音軌；WebAudio 解不動時會自動落到 ffmpeg.wasm）

---

## 隱私與快取

- **音檔不會上傳**；推論在瀏覽器完成。  
- **不囤錄音**：只保留最新一段的回放音檔；換檔即釋放舊 URL。  
- **模型快取**：存於 IndexedDB（數十 MB）。要釋放，可用頁面上的「清除模型快取」或清除網站資料。

---

## 專案結構

```
.
├─ index.html            # UI / 說明 / 免責 / 隱私 / 版本
├─ assets/
│  ├─ styles.css         # 造型
│  └─ app.js             # 錄音、解碼（WebAudio→ffmpeg.wasm 備援）、整段推論、回放、GC
```

`assets/app.js` 重點：整段一次、不切窗；WebAudio 先、ffmpeg.wasm 後（轉完 `exit()`）；  
GC 安全（關閉 `AudioContext`、清空 `chunks`、釋放舊 `ObjectURL`）；長檔有心跳式進度。

---

## 版本與授權

- **網站程式碼**：依本 repo 授權（例如 MIT）。  
- **模型**：`prithivMLmods/Common-Voice-Gender-Detection`（Apache-2.0）  
- **致謝**：`@xenova/transformers`（Transformers.js）

版本：`v2025-10-22`

---

## 變更紀錄（摘要）

- `v2025-10-22`  
  - 預設改為整段推論（不分段）；新增回放按鈕  
  - 上傳支援 mp4/mov（WebAudio → ffmpeg.wasm 備援）  
  - 新增隱私／免責／使用建議／侷限；提供清快取按鈕  
  - 強化 GC 與進度心跳
