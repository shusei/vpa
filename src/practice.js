// Facade for Practice Module
export { setupPracticeUI, openPracticeCategory, mountManualWidgets } from "./practice/practice-ui.js";
export { createCard } from "./practice/practice-ui.js"; // Exported just in case, or for manual-data usage if needed
// Re-export specific core functions if tests or other modules need them?
// Generally main.js only needs setupPracticeUI.
