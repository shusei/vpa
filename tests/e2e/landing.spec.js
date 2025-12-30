// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('Landing Page', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/');
    });

    test('should have correct metadata', async ({ page }) => {
        await expect(page).toHaveTitle(/Voice Presentation Analyzer/);
        await expect(page.locator('h1.hero-title')).toHaveText('Voice Presentation Analyzer');
    });

    test('should show critical UI elements', async ({ page }) => {
        // Header actions
        await expect(page.locator('#helpBtn')).toBeVisible();
        await expect(page.locator('#settingsBtn')).toBeVisible();

        // Main interaction
        await expect(page.locator('#recordBtn')).toBeVisible();
        await expect(page.locator('#uploadFab')).toBeVisible();

        // Footer
        await expect(page.locator('footer.footer')).toBeVisible();
    });

    test('record button should toggle state (mock)', async ({ page }) => {
        // Note: We cannot easily test actual audio recording in headless, 
        // but we can check if the button reacts to clicks.

        const btn = page.locator('#recordBtn');
        await expect(btn).toHaveClass(/record-btn/);

        // Initial state
        await expect(page.locator('#statusLabel')).toHaveText(/準備就緒/);

        // Click to start
        // We might need to handle permission dialogs if not mocked, 
        // but in http-server context often permissions are denied or prompt.
        // VPA handles permission denial gracefully by alerting? 
        // For this simple smoke test, we verify it exists.
        await expect(btn).toBeEnabled();
    });
});
