import { MANUAL_DATA } from './manual-data.js';
import { getCurrentLocale, onLocaleChange } from './i18n.js';
import { mountManualWidgets } from './practice.js';

/**
 * Initializes the Guide (Manual) Overlay with I18n support.
 */
export function initManualUI() {
    const btn = document.getElementById('guideBtn');
    const overlay = document.getElementById('guideOverlay');
    const guideBody = document.getElementById('guideContent');
    const guideTitle = document.getElementById('guideTitle');

    if (!btn || !overlay) {
        console.warn('VPA: Guide elements not found.');
        return;
    }

    /**
     * Renders the manual content based on the provided locale.
     * Falls back to 'en' if the locale is not found in MANUAL_DATA.
     */
    function renderManual(locale) {
        // Fallback chain: requested -> zh-Hant -> en
        let data = MANUAL_DATA[locale];
        if (!data) {
            data = MANUAL_DATA['zh-Hant'] || MANUAL_DATA['en'];
        }

        if (data) {
            if (guideTitle) guideTitle.textContent = data.title;

            // Inject HTML content
            if (guideBody) {
                guideBody.innerHTML = data.html;
                // Re-scroll to top when content changes? Optional, but good UX.
                guideBody.parentElement.scrollTop = 0;
                mountManualWidgets(guideBody);
            }
        }
    }

    // Initial Render
    renderManual(getCurrentLocale());

    // Listen for locale changes
    onLocaleChange((newLocale) => {
        renderManual(newLocale);
    });

    // --- Interaction Logic ---

    function openGuide() {
        overlay.hidden = false;
        btn.setAttribute('aria-expanded', 'true');
        const closeBtn = overlay.querySelector('.guide-close');
        closeBtn?.focus();
        document.body.style.overflow = 'hidden';
    }

    function closeGuide() {
        overlay.hidden = true;
        btn.setAttribute('aria-expanded', 'false');
        btn.focus();
        document.body.style.overflow = '';
    }

    btn.addEventListener('click', openGuide);

    const closeBtn = overlay.querySelector('.guide-close');
    if (closeBtn) {
        closeBtn.addEventListener('click', closeGuide);
    }

    const topBtn = overlay.querySelector('.guide-top');
    if (topBtn && guideBody) {
        topBtn.addEventListener('click', () => {
            if (guideBody.parentElement) {
                guideBody.parentElement.scrollTop = 0;
            }
        });
    }

    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) {
            closeGuide();
        }
    });

    overlay.addEventListener('keydown', (e) => {
        if (!overlay.hidden && e.key === 'Escape') {
            e.stopPropagation();
            closeGuide();
        }
    });
}
