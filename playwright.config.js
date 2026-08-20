import { defineConfig } from "@playwright/test";
import { resolve } from "node:path";

const fakeAudioPath = resolve("tests/.generated-media/tone.wav");

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 90_000,
  // Chromium exposes one deterministic fake microphone file to every test.
  // Keep media lifecycle tests exclusive so parallel workers cannot exhaust it.
  workers: 1,
  expect: {
    timeout: 10_000,
  },
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: "http://127.0.0.1:8080",
    permissions: ["microphone"],
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
    video: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: {
        browserName: "chromium",
        launchOptions: {
          args: [
            "--use-fake-device-for-media-stream",
            "--use-fake-ui-for-media-stream",
            `--use-file-for-fake-audio-capture=${fakeAudioPath}`,
          ],
        },
      },
    },
  ],
  webServer: {
    // Spawn the server directly so Playwright can terminate it cleanly on Windows.
    command: "node node_modules/http-server/bin/http-server -p 8080 -c-1",
    url: "http://127.0.0.1:8080",
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
});
