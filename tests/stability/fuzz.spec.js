import { test, expect } from 'playwright/test';
import { isWebGPUSupported } from './support.js';

test('parameter fuzzing remains bounded and stable', async ({ page }) => {
  const webgpuSupported = await isWebGPUSupported(page);
  test.skip(!webgpuSupported, 'WebGPU adapter unavailable in this runtime.');

  await page.goto('/?deterministic=1&sweep=1&scene=0&grid=64&smoke=1&maxFrames=800');
  await page.waitForFunction(() => Boolean(window.__FIRE_SIM_CONTROL__), { timeout: 30_000 });

  const result = await page.evaluate(async () => {
    return window.__FIRE_SIM_CONTROL__.runFuzz({ iterations: 220, seed: 424242 });
  });

  expect(result.runtimeError).toBeNull();

  const params = result.finalParams;
  expect(params.timeStep).toBeGreaterThanOrEqual(0.001);
  expect(params.timeStep).toBeLessThanOrEqual(0.05);
  expect(params.stepQuality).toBeGreaterThanOrEqual(0.25);
  expect(params.stepQuality).toBeLessThanOrEqual(4);
  expect(params.drag).toBeGreaterThanOrEqual(0);
  expect(params.drag).toBeLessThanOrEqual(0.2);
  expect(params.absorption).toBeGreaterThanOrEqual(0);
  expect(params.absorption).toBeLessThanOrEqual(100);
  expect(params.gamma).toBeGreaterThanOrEqual(0.1);
  expect(params.gamma).toBeLessThanOrEqual(4);

  const status = await page.evaluate(() => window.__FIRE_SIM_STATUS__);
  expect(status).toBeTruthy();
  expect(status.error).toBeNull();
});
