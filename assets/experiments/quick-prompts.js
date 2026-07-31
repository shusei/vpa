export const DAILY_PROMPT_IDS = [
  "arrival",
  "commute",
  "directions",
  "dinner",
  "repeat",
  "shopping",
  "timing",
  "weather",
  "weekend",
  "workload",
];

export const STANDARD_PROMPT_IDS = [
  "arrival",
  "dinner",
  "shopping",
];

export const STANDARD_TEST_ID = "standard-v1";

export function getDailyPromptId(date = new Date()) {
  const value = date instanceof Date ? date : new Date(date);
  if (Number.isNaN(value.getTime())) throw new TypeError("invalid prompt date");
  const localDay = Date.UTC(
    value.getFullYear(),
    value.getMonth(),
    value.getDate(),
  );
  const dayNumber = Math.floor(localDay / 86400000);
  return DAILY_PROMPT_IDS[((dayNumber % DAILY_PROMPT_IDS.length)
    + DAILY_PROMPT_IDS.length) % DAILY_PROMPT_IDS.length];
}

export function getStandardPromptId(step) {
  return STANDARD_PROMPT_IDS[Number(step)] || null;
}

export function isQuickPromptId(value) {
  return DAILY_PROMPT_IDS.includes(String(value || ""));
}

export function promptTranslationKey(promptId) {
  if (!isQuickPromptId(promptId)) return null;
  return `experiment.quick.prompts.${promptId}`;
}
