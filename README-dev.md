# Developer Notes

## Sweet Fixture 驗收
1. 直接以瀏覽器開啟 `index.html`（或透過本機伺服器提供）。
2. 打開 DevTools Console，輸入並執行：
   ```js
   await import('./tests/verify-sweet-fixture.js?' + Date.now());
   ```
3. Console 會逐條顯示 PASS/FAIL，確認所有檢查均為 PASS 即完成驗收。
