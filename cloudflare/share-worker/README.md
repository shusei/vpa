# VPA Cloudflare 免費分享服務

架構為 GitHub Pages + Cloudflare Workers Free + D1 Free。這個 Worker 只保存個人化結果圖片與顯示文字，不會接收錄音或影片。

## 為什麼使用 D1 Free

- 免費方案超過每日或儲存額度時會停止查詢，不會自動產生超額費用。
- D1 新資料可立即讀取，適合 X／Threads 在分享後立刻抓取預覽圖。
- KV Free 跨地區同步最慢可能約 60 秒，不適合這個即時分享用途。
- 個人化預覽使用壓縮 JPEG，單張最多 400 KB。
- 結果預設保存 365 天；過期資料由每日排程分批刪除。

免費方案用完時，前端會退回普通分享連結，不會阻止使用者繼續使用分析功能。

## 路由

- `POST /api/shares`：接收 1200×630 JPEG 與結果頁 metadata。
- `GET /r/:id`：輸出含 Open Graph／Twitter Card 的分享結果頁。
- `GET /i/:id.jpg`：輸出個人化結果圖片。

## 第一次部署

1. 登入 Cloudflare，但不要訂閱 R2 或 Workers Paid。
2. 在本目錄執行 `npm install`。
3. 執行 `npx wrangler login`，在瀏覽器授權 Wrangler。
4. 建立免費 D1 資料庫：`npx wrangler d1 create vpa-share`。
5. 將指令回傳的 `database_id` 填入 `wrangler.toml`。
6. 建立資料表：`npm run db:apply`。
7. 執行 `npm test`，再執行 `npm run deploy`。
8. 將部署後的固定 Worker origin 填入主站的 `VPA_SHARE_SERVICE_ORIGIN`。

## 免費額度保護

- 保持 Cloudflare 帳戶在 Workers Free，不升級 Workers Paid。
- D1 Free 達到讀取、寫入或儲存上限時會回傳錯誤；前端會自動改用普通分享。
- Worker Free 達到每日請求上限時會回傳 Cloudflare 1027，不會按超額流量扣款。
- D1 排程每天清理最多 5,000 筆過期結果，避免儲存空間無限累積。

若日後更換主站網址，只需修改 `PUBLIC_APP_URL` 與 `SITE_ORIGINS` 後重新部署；已存在且未過期的 `/r/:id` 短網址保持不變。
