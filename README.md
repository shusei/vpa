# Voice Presentation Analyzer

> 100% 瀏覽器端推論的聲音呈現分析工具，從錄音或上傳的語音片段中推估模型感知到的 feminine / masculine 傾向，並提供即時監控與統計摘要。

- Demo（GitHub Pages）：https://shusei.github.io/vpa
- 演算法驗證報告：[ALGORITHM_VERIFICATION.md](./ALGORITHM_VERIFICATION.md)
- 支援子路徑部署與離線使用（首次需連網下載模型，後續快取於 IndexedDB）
- 前端語系：繁體中文／简体中文／English，可記住上一個偏好

---

## 目錄

- [產品概覽](#產品概覽)
- [核心能力一覽](#核心能力一覽)
- [操作流程](#操作流程)
- [輸出解讀指南](#輸出解讀指南)
- [技術架構](#技術架構)
- [開發與測試](#開發與測試)
- [部署方式](#部署方式)
- [常見問題與疑難排解](#常見問題與疑難排解)
- [隱私、定位與免責聲明](#隱私定位與免責聲明)
- [專案結構](#專案結構)
- [版本資訊與授權](#版本資訊與授權)
- [句庫 A + 自訂 UI B](#句庫-a--自訂-ui-b)

---

## 產品概覽

Voice Presentation Analyzer（以下簡稱 VPA）是一款完全以 Web 技術打造、在本機瀏覽器完成推論的語音呈現分析器。使用者可以直接在瀏覽器內錄音，或上傳 `mp3 / m4a / mp4 / mov / wav` 等常見音訊格式，系統會即時顯示：

- 模型對「女性化 / 男性化」呈現的機率傾向
- 錄音期間的音高、音量、噪音走勢
- 分析完成後的統計卡（Pitch、Volume、Environment、Formant / Resonance、Speech Rate 等）與簡短評語

整個流程不會把音檔傳回伺服器，所有計算都在前端完成。首次造訪時需連網下載前端頁面與模型檔案；完成後，模型會快取在 IndexedDB，日後再次造訪可以離線使用。若清除網站資料、換裝置或模型更新，則需重新下載一次。

---

## 核心能力一覽

### 隱私與安全

- **本地推論**：採用 [`@xenova/transformers`](https://github.com/xenova/transformers.js) 的 ONNX Runtime，模型固定自 Hugging Face Hub 下載，音檔絕不離開裝置。【F:assets/app.js†L1-L110】
- **最小化前處理**：僅將輸入混成單聲道並重採樣至 16 kHz，保留原始語音內容與音量。【F:assets/js/constants.js†L1-L17】
- **快取可控**：IndexedDB 儲存模型，使用者可透過介面按鈕或清除網站資料移除快取。

### 互動體驗

- **錄音／上傳二合一流程**：支援 MediaRecorder 錄音，或從檔案系統挑選音訊／影片檔（自動抽出音軌）。【F:assets/app.js†L196-L320】【F:index.html†L311-L348】
- **即時儀表與提示**：分析期間的儀表會即時刷新，狀態列同步顯示目前片段、進度百分比與預估剩餘時間。【F:assets/app.js†L566-L628】【F:index.html†L351-L397】
- **句庫練習抽屜**：錄音鍵旁提供句子清單與練習歷程；卡片左側按鈕負責錄音／停止，右側播放鍵可快速回聽上一段，完成後回寫女性傾向／男性傾向百分比，主儀表維持乾淨。【F:index.html†L150-L189】【F:assets/js/practice.js†L1-L520】
- **暖身卡片與快捷操作**：主畫面保留「暖身三步驟」卡片，記住上次展開狀態並提供錄音前 1 分鐘的放鬆流程；支援拖曳檔案到主畫面觸發上傳，錄音鍵與播放鈕也會自動關閉導覽提示。【F:index.html†L190-L213】【F:assets/app.js†L616-L736】【F:assets/app.js†L1668-L1776】
- **多主題與派別**：內建 30+ 顏色主題，可在 Auto / Light / Dark / 彩色派別中切換並記憶選擇。【F:index.html†L61-L123】【F:assets/js/theme.js†L1-L220】
- **主題與偵測設定**：齒輪選單可在 Auto / Light / Dark 派別間切換並記住偏好，亦可展開「說話者設定」套用 Feminine / Soprano 等偵測預設或手動調整音高上下限。【F:index.html†L61-L161】【F:assets/js/theme.js†L222-L339】
- **新手引導與使用指南**：首次造訪會出現提示泡泡，右上角 ❓ 可開啟圖文說明覆蓋層，支援鍵盤 `Esc` 關閉。【F:index.html†L214-L308】【F:assets/js/theme.js†L222-L339】
- **Beginner / Advanced 統計視圖**：統計卡底部提供一鍵展開進階面板；按 Shift / Alt 點擊可同步記住展開狀態，初學者則可切回精簡卡片摘要。【F:assets/app.js†L60-L154】【F:assets/app.js†L3728-L3899】
- **匯出分析**：最新一次結果會整理成 JSON，透過齒輪選單的「匯出分析（JSON）」即可下載，方便留存或進一步分析。【F:index.html†L150-L161】【F:assets/app.js†L2038-L2078】

### 即時監測與統計

- **Pitch Stream**：以 50 ms 解析度顯示 50–450 Hz 音高走勢、瞬時音量、噪音估計並提供穩定度平滑處理。【F:assets/app.js†L180-L259】【F:index.html†L125-L213】
- **Formant / Resonance 面板**：估計 F1–F3、胸腔／前置／頭腔共鳴比例、氣聲比例、頻譜傾斜等資訊，錄音或上傳時即時更新。【F:assets/app.js†L202-L254】【F:index.html†L134-L204】
- **統計卡**：分析完成後輸出 Pitch、Volume、Environment、Formant & Resonance、Speech Rate 等指標的平均值、百分位數與建議範圍。【F:assets/app.js†L1407-L1642】【F:index.html†L400-L575】
- **自適應音高與精緻共鳴分析**：內建條件啟用的 YIN-lite 音高偵測與進階共鳴摘要，會根據裝置得分與即時計時決定是否啟用，並在預算不足或發生錯誤時自動回退至基線 ACF／FFT 演算法，過載解除後再度嘗試升級。【F:assets/app.js†L251-L517】【F:assets/app.js†L568-L650】【F:assets/app.js†L2048-L2364】
- **模型簡評**：整合模型輸出與統計指標，給予 1 行摘要並凸顯指標矛盾或建議。【F:assets/app.js†L1365-L1406】
- **語調與進階摘要**：統計區塊新增語調曲線、語速、連音與亮度標籤；圖例可切換原始偵測點，協助比對平滑後的洋紅曲線與灰色靜音區段。【F:assets/i18n/zh-Hant.js†L667-L703】【F:assets/app.js†L2898-L3040】

### 長檔案處理

- **整段 / 串流自動切換**：`≤ 150 秒` 採單次推論；長檔會依裝置（WebGPU / WASM）與時長選擇視窗長度與 hop，逐步回退避免 OOM。【F:assets/js/constants.js†L5-L12】【F:assets/app.js†L600-L704】
- **自適應 VAD**：當錄音超過 20 秒且靜音比例超過 15% 時，會自動偵測有效語音區段、優先推論語音內容。【F:assets/js/constants.js†L9-L17】【F:assets/app.js†L884-L1032】
- **進度心跳與 ETA**：長檔會顯示分段進度、已處理時長與預估剩餘時間，同時標示採用的串流策略。【F:assets/app.js†L566-L628】【F:assets/app.js†L643-L704】

### 多語系與可及性

- **三語介面**：支援繁中／簡中／英文，透過 `assets/js/i18n.js` 進行延遲載入，會記住使用者偏好並更新 `<title>` 與 `<meta>`。【F:assets/js/i18n.js†L1-L120】
- **ARIA 與鍵盤操作**：所有主要控制都有 `aria-*` 屬性與鍵盤事件，確保可及性。【F:index.html†L274-L308】【F:index.html†L311-L348】

---

## 操作流程

1. **開啟網頁**：
   造訪 <https://shusei.github.io/vpa>。
   首次使用請在有網路的環境下開啟，以便下載模型。
2. **選擇來源**：
   - 點擊「開始錄音」並說話 5–10 秒（建議自然口語，避免唱歌）。
   - 或點擊右下角的上傳按鈕挑選 `mp3 / m4a / mp4 / mov / wav` 等檔案，也支援把檔案直接拖曳到主畫面。
     按下浮動按鈕時頁面會自動捲回頂端，方便直接看到上傳區塊與狀態列。【F:assets/app.js†L700-L730】
   若音高偵測需要更貼合自身聲域，可在齒輪 →「說話者設定」選擇 Feminine / Masculine / Soprano 等預設，或手動輸入音高上下限。【F:index.html†L104-L151】
3. **需要靈感或暖身？** 錄音鍵旁的「句庫練習」提供多語句子：挑一句後按卡片按鈕（或按 <kbd>Space</kbd>）立即錄音，再用右側播放鍵重聽上一段。建議每句練習 3–7 秒，卡片會回寫女性傾向／男性傾向百分比。正式錄音前也可展開主畫面的「暖身三步驟」卡片，依序完成哼聲、前母音、呼氣放鬆。
4. **即時觀察**：錄音期間會顯示 Pitch Stream 與 Formant / Resonance 面板；上傳檔案則會離線抽樣並於統計卡展示。
5. **等待推論完成**：儀表盤會顯示模型即時傾向、狀態列同步告知進度與預估剩餘時間。
6. **檢視結果**：推論完成後，可在儀表下方閱讀統計卡與簡評，必要時點擊播放器回放剛才的音檔。
7. **重複練習**：新的錄音或上傳會覆蓋舊音檔，快取的模型仍保留以加快後續推論。
8. **下載分析**：需要備份或自訂分析時，可開啟齒輪選單並點擊「匯出分析（JSON）」取得最新完整結果。【F:index.html†L150-L161】【F:assets/app.js†L2038-L2078】

錄音建議：
- 環境盡量安靜，與麥克風保持 10–15 公分距離。
- 使用自然說話聲音與語速；句庫練習卡可搭配 <kbd>Space</kbd> 開始錄音、<kbd>J</kbd>/<kbd>K</kbd> 快速切換句子。
- 若要上傳 iOS 語音備忘錄，可先在「檔案」App 儲存，再於本頁上傳。

---

## 輸出解讀指南

- **傾向儀表**：顯示模型推估的 feminine / masculine 百分比。40–60% 為灰色帶，建議再錄幾段觀察趨勢。【F:index.html†L351-L397】
- **Pitch 卡**：包含平均、Median、5th / 95th 百分位數並標記常見頻帶（男性常見、重疊帶、女性常見、高混聲 310–450 Hz 與 Soprano / Falsetto 450–600 Hz）。【F:index.html†L416-L447】【F:assets/i18n/zh-Hant.js†L360-L376】
- **高音涵蓋提醒**：圖軸延伸至 600 Hz，新增淡紫帶標示 Soprano / Falsetto，提醒假聲與頭聲練習者注意。既有範例（如 `fixtures/analysis/sweet_feminine.json`）回歸檢查後仍落在原粉色帶，確保解讀一致。【F:index.html†L416-L447】【F:fixtures/analysis/sweet_feminine.json†L1-L40】
- **Volume 卡**：平均、Median、標準差與 5th / 95th 百分位，用來評估音量穩定度。【F:index.html†L449-L486】
- **Environment 卡**：估計環境底噪（10th 百分位）與 SNR，提供錄音環境建議。【F:index.html†L488-L520】
- **Formant & Resonance 卡**：展示 F1–F3 中位數、共鳴亮度與氣聲比例，並附上提示文字。【F:index.html†L522-L566】
- **Speech Rate 卡**：統計語速與語音佔比，提醒是否語音不足。【F:index.html†L568-L575】
- **語調與進階面板**：洋紅線描繪平滑後的語調曲線，灰色區塊代表低信心或無聲；可切換原始偵測點比對，也會標示語速、連音比例與亮度標籤。【F:assets/i18n/zh-Hant.js†L667-L703】【F:assets/app.js†L2898-L3040】
- **簡評**：整合以上指標，標示指標分歧或錄音品質提醒。【F:assets/app.js†L1365-L1406】
- **句庫練習卡**：沿用同一條推論管線，只在卡片顯示女性傾向／男性傾向兩個百分比；每句保存最近 20 筆歷程於本機，徽章會標出上次成績。【F:assets/js/practice.js†L1-L520】

---

## 技術架構

```
使用者行為 → MediaRecorder / 檔案上傳 → 音訊解碼
  ↘ WebAudio 即時分析（Pitch / Formant / Noise）
     ↘ IndexedDB 快取模型 → Transformers.js ONNX Runtime
        ↘ 整段 / 串流分段 → Log-odds 聚合 → 傾向儀表
           ↘ 統計彙整（百分位、語速、噪音、SNR、formant）
              ↘ I18n / 主題系統 → UI 呈現
```

- **推論引擎**：`@xenova/transformers` 搭配 `prithivMLmods/Common-Voice-Gender-Detection-ONNX` 二分類模型（Apache-2.0 授權）。【F:assets/app.js†L1-L110】
- **音訊處理**：WebAudio 進行即時頻譜、pitch、formant 與噪音估計；必要時退回 `ffmpeg.wasm` 解析影片音軌。【F:assets/app.js†L320-L565】
- **分段策略**：根據裝置（WebGPU / WASM）與長度決定視窗與 hop，並顯示策略描述（如「WebGPU：24 秒窗 / hop 6 秒」）。【F:assets/app.js†L600-L704】
- **聚合方法**：長檔使用對數勝算（log-odds）加權整合片段結果，以貼近整段一次推論的輸出。【F:assets/app.js†L566-L628】
- **句庫練習流程**：共用單一 MediaRecorder 錄音（卡片按鈕切換錄音／停止、右側播放鍵重播上一段）→ 回寫 pf / pm 並以 localStorage 保存最近 20 筆歷程。【F:assets/js/practice.js†L1-L520】
- **狀態管理**：透過 `analysisSeq` / `activeAnalysisToken` 確保並行操作時只保留最新一輪結果，避免 race condition。【F:assets/app.js†L70-L118】

---

## 開發與測試

### 前置需求

- Node.js 18+（僅用於靜態檢查，前端仍為純靜態頁面）
- npm 8+

### 安裝與檢查

```bash
npm install      # 本專案無額外依賴，但可保留安裝流程
npm test         # 依序執行 JS 語法檢查與 HTML/CSS 結構檢查
```

個別命令：

- `npm run test:syntax`：使用 `node --check` 檢查 `assets/app.js` 語法。【F:package.json†L6-L11】
- `npm run test:markup`：執行 `scripts/check-static.js`，掃描預設的 HTML / CSS 檔案是否存在未配對標籤、括號或引號。【F:scripts/check-static.js†L1-L207】
  - 可加上額外檔案：`npm run test:markup -- docs/landing.html assets/css/custom.css`
- `npm run test:phrases`：驗證 Core‑36 句庫的結構（筆數、分類、ID、版本）。【F:tests/verify-phrases-core36.js†L1-L24】

### 本地預覽

本專案為純靜態頁面，可使用任何靜態伺服器，例如：

```bash
npx serve .
```

---

## 句庫 A + 自訂 UI B

### 句庫核心資料（Core‑36）

- **檔案位置**：`assets/data/phrases/zh-Hant-core-36.json`
- **格式定義**：請參考 `assets/js/phrases/types.d.ts`；`PhrasePack` 內含 `categories` 與 `items`，每筆 `PhraseItem` 可附 `alts`、`tags`、`difficulty` 與 `notes`。
- **授權**：`license` 欄位標示為 `CC0-1.0`，示例句可自由再利用。
- **驗收**：執行 `npm run test:phrases` 會檢查 Core‑36 結構（36 句、分類存在、ID 唯一、版本符合 `x.y.z`）。

### Phrase Lab（使用者自訂介面）

- **入口**：開啟 `assets/ui/phrases-lab.html`（靜態頁面），即可載入 Core‑36 與使用者在 `localStorage` (`phrases:userEdits`) 留存的客製資料。
- **瀏覽與篩選**：
  - 頂部提供搜尋（主句、變體、標籤、備註皆會匹配）。
  - 可依分類、難度、收藏切換列表；收藏標記存於 `favorites`。
- **分類管理**：
  - 左側清單支援新增、改名、刪除、拖曳重排（拖動同時更新 `categories.order`）。
  - 刪除分類時會同步移除底下句子與收藏紀錄。
- **句子管理**：
  - 右側卡片顯示主句、變體、標籤與備註；支援星標、編輯、刪除。
  - 拖曳卡片可重新排序，或拖到其他分類；變更即寫入 `items.order` 與 `items.updated`。
- **練習模式**：底部控制列提供依序／隨機、30/60/90 秒計時、隱藏提示、下一句等功能；進度顯示會跟著篩選結果更新。
- **匯入/匯出**：
  - 匯出會下載一份 `phrases-user-edits-YYYYMMDD-HHmmss.json`，包含 `schema`、基底版本與所有使用者修改。
  - 匯入支援拖檔或點選檔案；若 `baseId` 不符 `zh-Hant-core-36` 會跳錯，避免混入其他語系。
  - `重設` 會清除 `localStorage`，恢復為純 Core‑36。

### JSON 格式與相容性

- `categories.added / updated / removed / order`：記錄分類的增刪改與排序。
- `items.added / updated / removed / order`：記錄句子的增刪改與分類內排序。
- `favorites`：收藏句子的 ID 陣列，匯入時會自動過濾不存在的 ID。
- 版本升級時可參考 `USER_EDITS_SCHEMA`（目前為 `1`），若未來 schema 變動請記得提供轉換邏輯。

### 常見錯誤與排查

- **`JSON 解析失敗`**：匯入的檔案非有效 JSON，重新輸出或檢查是否被編輯器插入註解。
- **`基底句庫不符`**：匯入檔案的 `baseId` 與 Core‑36 不一致，請確認是否匯入了其他語言或舊專案的備份。
- **LocalStorage 滿**：瀏覽器儲存空間不足時 `saveUserEdits` 會丟出例外，建議匯出備份後清除快取再重新匯入。
- **排序失真**：若外部編輯 JSON 手動刪除 `items.order`，重載後會依現有排序重新建立，不影響資料內容。

啟動後於瀏覽器開啟 `http://localhost:3000` 即可操作。

---

## 部署方式

### GitHub Pages（預設建議）

1. 將 `index.html` 與整個 `assets/` 資料夾推到公開 repository。
2. 於 GitHub 設定 Pages（Branch 或 GitHub Actions 均可）。
3. 預設 `window.INFERENCE_MODE = 'browser'`，無須額外設定伺服器。首次載入會下載模型，後續使用快取。

### Cloudflare Pages（提供 Hugging Face 代理）

若需透過 serverless 代理呼叫 Hugging Face API，可在 Pages Functions 設定 `HUGGING_FACE_TOKEN`，並於 `index.html`（或額外 script）覆寫以下設定：

```html
<script>
  window.INFERENCE_MODE = 'server';
  window.API_BASE_URL = 'https://<project>.pages.dev/api/classify';
</script>
```

預設仍推薦純前端模式，僅在企業環境或需要封鎖外網時才設定代理。

### 子路徑部署

VPA 遵循相對路徑，可直接放在 `https://domain/app/vpa/` 等子路徑。若遇路徑問題，檢查 `<base>` 或 CDN 來源是否正確。

---

## 常見問題與疑難排解

| 問題 | 可能原因 | 建議作法 |
| ---- | -------- | -------- |
| 無法錄音 | 瀏覽器拒絕麥克風權限 / MediaRecorder 不支援 | 允許麥克風、改用 Chrome／Edge；或直接上傳檔案。【F:assets/app.js†L216-L320】 |
| 長檔推論耗時 | 裝置採用 WASM 且檔案 > 4 分鐘 | 進度列會顯示實際策略（例如 hop 4 秒），可等待或手動裁切檔案。【F:assets/app.js†L643-L704】 |
| 統計卡顯示「資料不足」 | 語音時長不足 5 秒、語音不連續 | 延長錄音時間並保持連續語句。【F:assets/app.js†L1603-L1676】 |
| 模型載入失敗 | 首次載入網路不穩、Hugging Face CDN 無法存取 | 重整頁面，或在離線前確保模型已載入一次（IndexedDB 快取）。【F:assets/app.js†L520-L565】 |
| 推論速度與桌機不同 | 系統會依 `navigator.hardwareConcurrency` 自動挑選 1–4 個 WASM 執行緒；Safari 與行動裝置為穩定性固定 1 執行緒 | 桌機若清除快取會重新偵測，可於 DevTools console 查看實際執行緒數；行動裝置回退單執行緒以避免排程過載。【F:assets/app.js†L1-L120】 |
| 要清除舊音檔 | 重新錄音或上傳新檔案 | 系統僅保留「最新一段」的回放 URL，換檔即釋放。【F:assets/app.js†L710-L756】 |

---

## 隱私、定位與免責聲明

- **非醫療診斷工具**：本工具指標係結合數值聲學演算法與機器學習模型，僅供自我聲音探索、練習與語音呈現回饋，**不等同於性別認同、身分判定或任何醫療診斷／法律結論**。
- **安全提醒**：練習時請保持舒適音量與自然發聲。若發聲過程中感覺喉嚨疼痛、刺痛、沙啞或疲勞，請立即停止休息；若不適感持續，請尋求耳鼻喉科醫師或語言治療師之診治與建議。
- **隱私安全**：所有音訊與模型推論皆 100% 於使用者本機瀏覽器內完成，聲音檔案絕不上傳。
- **灰色帶**：若結果介於 **40–60%**，代表模型不確定，請多錄幾段觀察趨勢。
- **已知侷限**：模型以 Mozilla Common Voice 英語朗讀資料訓練，面對中文、方言、唱歌、戲腔等聲音時可能有偏差；噪音、回音、鼻音或刻意拉高音高亦可能影響結果。【F:assets/app.js†L566-L628】【F:index.html†L351-L397】
- **練習數據**：句庫練習結果僅存於瀏覽器 localStorage，可隨時清除；卡片按鈕可快速切換錄音／停止，完成後回寫女性／男性傾向百分比。【F:assets/js/practice.js†L1-L460】

---

## 專案結構

```
.
├─ index.html            # 單頁應用程式：UI、主題、引導、儀表與統計卡
├─ assets/
│  ├─ app.js             # 入口腳本：錄音 / 上傳、解碼、推論、即時監控、統計彙整
│  ├─ css/
│  │  ├─ base.css        # Reset、CSS 變數、共用樣式
│  │  ├─ layout.css      # 網格與排版
│  │  ├─ components.css  # 儀表、統計卡、控制元件樣式
│  │  └─ overlays.css    # 引導泡泡、說明覆蓋層、模態
│  ├─ js/
│  │  ├─ constants.js    # 模型 ID、採樣率、串流參數、VAD 閾值
│  │  ├─ dom.js          # 常用 DOM 快取與查找工具
│  │  ├─ theme.js        # 主題切換、派別控制、引導狀態
│  │  ├─ ui.js           # 狀態列、儀表更新、格式化工具
│  │  └─ i18n.js         # 多語系載入、DOM 套用、偏好儲存
│  └─ i18n/              # 各語系文案（繁中為預設，簡中 / 英文延遲載入）
├─ scripts/
│  └─ check-static.js    # HTML / CSS 結構檢查腳本
├─ package.json          # npm 指令與測試設定
└─ package-lock.json
```

---

## 版本資訊與授權

- **網站程式碼**：依 repository 所附授權（預設 MIT，請依實際授權檔確認）。
- **模型**：`prithivMLmods/Common-Voice-Gender-Detection-ONNX`（Apache-2.0）。
- **句庫資料**：Core‑36 句庫（`assets/data/phrases/zh-Hant-core-36.json`）以 CC0-1.0 釋出；練習抽屜示例句（`assets/data/practice/*.json`）以 CC BY-NC 4.0 提供，可依需求替換或擴充。【F:assets/data/phrases/zh-Hant-core-36.json†L1-L70】【F:assets/data/practice/zh-Hant.json†L1-L25】
- **第三方致謝**：`@xenova/transformers`、`@ffmpeg/ffmpeg`（ffmpeg.wasm）、`@ffmpeg/util`（vendored ESM `fetchFile`）等開源專案。
- **版本標記**：載入頁面時會自動填入 `build-YYYYMMDD-HHMM`，可於頁面右下角查看。【F:assets/app.js†L214-L244】【F:index.html†L577-L607】

若本 README 與程式碼有出入，請以程式碼邏輯為準並歡迎提出 Issue 或 PR。

---

## 支援作者

[![Buy Me a Coffee](https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png)](https://buymeacoffee.com/shusei)
