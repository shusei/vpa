# Developer Notes

## Sweet Fixture 驗收
1. 直接以瀏覽器開啟 `index.html`（或透過本機伺服器提供）。
2. 打開 DevTools Console，輸入並執行：
   ```js
   await import('./tests/verify-sweet-fixture.js?' + Date.now());
   ```
3. Console 會逐條顯示 PASS/FAIL，確認所有檢查均為 PASS 即完成驗收。

## 音訊解碼備援（ffmpeg.wasm）
- `assets/app.js` 會先嘗試以 WebAudio 解碼檔案；失敗時才動態載入 `assets/js/ffmpeg-transcode.js`。
- 備援模組使用 `@ffmpeg/ffmpeg` 0.12.10 並鎖定 `@ffmpeg/core-mt` 0.12.10 的 wasm/worker，將音訊轉為單聲道 32-bit float，並回傳 `Float32Array` 與取樣率。
- UI 會依 `status.ffmpeg*` 字串顯示下載／轉檔進度，確保 m4a / mp4 等 Safari 不支援的格式仍可處理。
- `npm test` 會額外執行 `tests/verify-ffmpeg-download.mjs`，以 HEAD 請求確認 `ffmpeg-core.wasm` 可下載且回應具有 CORS 標頭；若環境無法連線會自動跳過。
