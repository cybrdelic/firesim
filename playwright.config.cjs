const { defineConfig } = require('playwright/test');

module.exports = defineConfig({
  testDir: './tests/stability',
  timeout: 90_000,
  expect: {
    timeout: 30_000,
  },
  workers: 1,
  retries: process.env.CI ? 2 : 0,
  use: {
    baseURL: 'http://127.0.0.1:4173',
    headless: true,
    viewport: { width: 1280, height: 720 },
    deviceScaleFactor: 1,
    screenshot: 'off',
    video: 'off',
  },
  webServer: {
    command: 'npm run dev -- --host 127.0.0.1 --port 4173',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: true,
    timeout: 120_000,
  },
});
