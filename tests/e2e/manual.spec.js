// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('Voice Manual Overlay', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/');
    });

    test('should open manual when clicking the book button', async ({ page }) => {
        const guideBtn = page.locator('#guideBtn');
        const overlay = page.locator('#guideOverlay');

        // Initially hidden
        await expect(overlay).toBeHidden();

        // Click open
        await guideBtn.click();

        // Should be visible
        await expect(overlay).toBeVisible();

        // Title check
        await expect(page.locator('#guideTitle')).toHaveText('女聲訓練手冊');

        // Content check (Section 0 existence)
        await expect(page.locator('#guideContent')).toContainText('0) 你每天照做什麼');
    });

    test('should close manual when clicking close button', async ({ page }) => {
        const guideBtn = page.locator('#guideBtn');
        const overlay = page.locator('#guideOverlay');
        const closeBtn = page.locator('.guide-close');

        // Open first
        await guideBtn.click();
        await expect(overlay).toBeVisible();

        // Click close
        await closeBtn.click();

        // Should be hidden
        await expect(overlay).toBeHidden();

        // Focus should return to trigger button
        await expect(guideBtn).toBeFocused();
    });
});
