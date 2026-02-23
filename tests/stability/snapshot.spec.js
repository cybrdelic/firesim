import { test, expect } from 'playwright/test';
import { isWebGPUSupported } from './support.js';

const snapshotCases = [
  { name: 'campfire-64', scene: 0, grid: 64, smoke: 1, maxFrames: 100 },
  { name: 'wood-128', scene: 4, grid: 128, smoke: 1, maxFrames: 120 },
  { name: 'candle-64', scene: 1, grid: 64, smoke: 1, maxFrames: 100 },
];

for (const scenario of snapshotCases) {
  test(`snapshot ${scenario.name}`, async ({ page }) => {
    const webgpuSupported = await isWebGPUSupported(page);
    test.skip(!webgpuSupported, 'WebGPU adapter unavailable in this runtime.');

    const url = `/?deterministic=1&sweep=1&scene=${scenario.scene}&grid=${scenario.grid}&smoke=${scenario.smoke}&maxFrames=${scenario.maxFrames}`;
    await page.goto(url);

    await page.waitForFunction(
      () => window.__FIRE_SIM_STATUS__?.ready === true,
      { timeout: 45_000 }
    );

    await expect(page.locator('canvas')).toHaveScreenshot(`${scenario.name}.png`, {
      animations: 'disabled',
      maxDiffPixelRatio: 0.03,
    });
  });
}
