# Voice Presentation Analyzer – Contribution Guide

## Scope
這份指引涵蓋整個倉庫。若未來在子資料夾中加入新的 `AGENTS.md`，請以最接近的指引為準。

## 專案定位
- 前端為 100% 靜態頁面，所有推論都在瀏覽器端完成，禁止引入後端服務或建置流程。
- 程式碼採用原生 ES Modules，直接以 `<script type="module">` 從 CDN 載入依賴，**不要**加入打包器或額外編譯步驟。
- 任何對使用者可見的文案都必須透過 `assets/js/i18n.js` 管理；若新增字串，請同步更新 `assets/i18n/zh-Hant.js`、`zh-Hans.js`、`en.js` 等字典。

## 目錄速查
- `index.html`：唯一的頁面，含 UI 標記、ARIA 屬性與 `data-i18n` 標記。
- `assets/app.js`：核心流程（載入模型、錄音／上傳處理、推論管線、統計與 UI 更新）。
- `assets/js/`：以功能拆分的輔助模組（音訊工具、pitch 計算、主題 / DOM 操作、句庫練習 UI、進階面板折疊控制等）。
- `assets/css/`：基礎樣式、佈局、元件與覆蓋層；維持既有的壓縮式格式（`property:value;`）。
- `assets/data/`、`fixtures/`：示範資料與測試樣本。
- `scripts/`、`tests/`：Node.js 驗證腳本與回歸檢查。
- `assets/vendor/`：第三方 WASM 與支援檔案，請保留原始檔名與版號註記。

## JavaScript 編碼規範（瀏覽器環境）
- 使用雙引號 (`""`)、每層縮排 2 個空格，保留分號與尾逗號（`trailing comma`）。
- 優先採用 `const`；僅在確需重新指定時使用 `let`，避免 `var`。
- 模組開頭依序排列：
  1. 外部 CDN / vendor 引入
  2. 本地模組（依檔案路徑字典排序）
  3. 常數宣告
- 函式應維持單一職責，邏輯分支長於 ~80 行時請考慮拆模組；複雜流程請加入區塊註解（`// ----- Section -----`）。
- 使用 `async/await` 管理非同步流程，並妥善處理錯誤：
  - 針對使用者可恢復的情境，呼叫 `setStatus()` 或其他 UI 告知函式。
  - 無法復原的錯誤仍需記錄在主控台（`console.error(...)`）。
- 操作 DOM 時統一透過 `assets/js/dom.js` 暴露的快取節點或輔助函式；新增節點時，先在該模組定義並保持命名一致。
- UI 更新應透過既有的 `setStatus`／`setRealtimePanelsActive`／`reset*` 等工具函式，避免直接操作樣式屬性造成狀態失衡。新增進階統計卡或 Beginner / Advanced 切換時，請沿用 `data-adv` 設定與 `setupAdvancedSection()` 所期望的屬性。
- 調整或新增文字時務必使用 `t('key.path')`；若無對應條目，請在字典中新增多語翻譯並與現有結構一致。
- 保持記錄與快取策略：
  - 若引入新的快取項目，沿用 `localStorage` 鍵名命名慣例（`vpa::` 前綴）。
  - 推論參數改動需同步更新 `assets/js/constants.js` 並評估回歸測試。

## HTML 樣板規範
- 縮排 2 空格，採用自閉合語法（`<input ... />`）時請保留結尾斜線。
- 屬性順序維持現有慣例：`id` → `class` → 互動屬性（`type`、`role`、`tabindex` 等）→ 可訪問性（`aria-*`）→ `data-*`。
- 為可本地化的元素加上 `data-i18n` 或 `data-i18n-attrs`；靜態文字若無翻譯需求需明確加註註解說明原因。
- 修改互動元件時必須檢查對應的 `aria-*`、`role` 與鍵盤操作是否仍然成立。

## CSS 風格
- 維持檔案現有排版：屬性與值之間不留空格（`color:#fff;`），選擇器與大括號之間保留空格。
- 主題變數使用 `--token-name` 慣例；新增主題時需同步更新 `assets/js/theme.js` 的清單與預設值。
- 僅在確定新屬性不會破壞 `color-mix()`／`oklab` 等現有計算時才調整變數。
- 每次更新 CSS 後務必檢查深色主題的文字可讀性；不可再出現文字對背景對比不足的問題，必要時請同步調整 `--color-*` 變數與對應元件狀態樣式。

## Node.js 腳本與測試
- 本專案已切換至 ESM；檔案起始請使用 `import`，必要時透過 `node:` 前綴引用核心模組。
- 腳本中的字串風格依檔案既有格式（多為單引號），編輯時請就地遵循。
- CLI 腳本應保持零依賴（僅使用 Node.js 內建模組），並提供清楚的錯誤訊息。

## 測試與驗證
- 一律執行 `npm test`（或分項執行 `npm run test:*`）確保：
  - `assets/app.js` 可通過語法檢查。
  - 靜態標記與 CSS 結構無錯誤。
  - FFmpeg 下載、解碼流程與語音句庫驗證都能通過。
- 若修改音訊處理或推論流程，請更新 `fixtures/` 內的對照檔並執行 `npm run regression` 確認行為未退步。

## 其他約定
- 請勿提交格式化器自動產生的大量噪音變更；只調整必要區塊。
- 變更主流程時在 PR 描述中清楚標示影響面向與回歸測試狀況。
- Commit 訊息採用一般英文祈使句（例如：`Add pitch post-processing guard`）。
