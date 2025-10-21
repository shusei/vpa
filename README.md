## README – Deploy notes (精簡)


### A) GitHub Pages（無後端，隱私最高）
1. 把 `index.html`, `assets/` 推到任一公開 repo，開啟 GitHub Pages。
2. 保持 `window.INFERENCE_MODE = 'browser'`（預設）。
3. 第一次載入會抓 ONNX 模型（約數十 MB），等待載入後即可離線重用快取。


### B) Cloudflare Pages（可選 serverless 代理）
1. 新建 Pages 專案，來源設為這個專案目錄。
2. 在 **Pages → Settings → Environment variables** 新增：`HUGGING_FACE_TOKEN=<你的HF存取權杖>`。
3. 推上去後，`/api/classify` 就會生效。若前端也部署在 Pages，同網域可直接使用。
4. 若你的前端放在 GH Pages，將 `window.INFERENCE_MODE = 'server'`，`window.API_BASE_URL` 改成你的 Pages 網域 `https://<project>.pages.dev/api/classify`。


### 其他
- 想改成「只顯示一個大圓百分比」或「條形圖」，改 `renderResults()` 即可。
- Safari 某些版本不支援 `audio/webm`，若遇到錄音問題，建議用「上傳」或改用現成錄音庫（如 opus-recorder）。
- 模型倉庫命名是 `Common-Voice-Gender-Detection`；本頁已改用其 **ONNX 版本** 供瀏覽器推論：`prithivMLmods/Common-Voice-Gender-Detection-ONNX`。
