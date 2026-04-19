import fs from 'node:fs';
import path from 'node:path';
import { expect, test } from 'playwright/test';

// Nudge Chromium toward exposing a WebGPU adapter in more environments.
test.use({
  launchOptions: {
    args: ['--enable-unsafe-webgpu', '--use-angle=d3d11'],
    ...(process.env.PLAYWRIGHT_CHANNEL ? { channel: process.env.PLAYWRIGHT_CHANNEL } : {}),
  },
});

async function tryCdp(page, fn) {
  try {
    const client = await page.context().newCDPSession(page);
    return await fn(client);
  } catch {
    return null;
  }
}

test('gpu health report', async ({ page, baseURL }) => {
  expect(baseURL).toBeTruthy();

  const origin = String(baseURL).replace(/\/$/, '');

  // Load the app in its lightest harness mode.
  const url = `${origin}/?deterministic=1&sweep=1&maxFrames=0&grid=64&smoke=0&cacheBust=${Date.now()}`;
  await page.goto(url, { waitUntil: 'domcontentloaded' });

  let cdpError = null;
  let sysInfo = null;
  let gpuInfo = null;

  // CDP + cache clearing (Chromium only)
  try {
    const client = await page.context().newCDPSession(page);
    if (process.env.CLEAR_SITE_DATA === '1') {
      await client.send('Network.clearBrowserCache');
      await client.send('Network.clearBrowserCookies');
      await client.send('Storage.clearDataForOrigin', {
        origin,
        storageTypes: 'all',
      });
    }
    sysInfo = await client.send('SystemInfo.getInfo');
    gpuInfo = await client.send('GPU.getInfo');
  } catch (error) {
    cdpError = error instanceof Error ? error.message : String(error);
  }

  const webgpu = await page.evaluate(async () => {
    const out = {
      supported: Boolean(navigator.gpu),
      userAgent: navigator.userAgent,
      platform: navigator.platform,
      devicePixelRatio: window.devicePixelRatio,
      deviceLost: null,
      adapter: null,
      device: null,
      status: null,
      memory: null,
      errors: [],
    };

    // Optional JS heap info (Chromium only)
    // eslint-disable-next-line no-undef
    const pm = performance && performance.memory ? performance.memory : null;
    if (pm) {
      out.memory = {
        jsHeapSizeLimit: pm.jsHeapSizeLimit,
        totalJSHeapSize: pm.totalJSHeapSize,
        usedJSHeapSize: pm.usedJSHeapSize,
      };
    }

    const status = (window).__FIRE_SIM_STATUS__;
    if (status) {
      out.status = {
        ready: Boolean(status.ready),
        frame: status.frame ?? null,
        error: status.error ?? null,
        warning: status.warning ?? null,
      };
    }

    if (!navigator.gpu) return out;

    let adapter = null;
    try {
      adapter =
        (await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })) ||
        (await navigator.gpu.requestAdapter({ powerPreference: 'low-power' })) ||
        (await navigator.gpu.requestAdapter());
    } catch (err) {
      out.errors.push(`requestAdapter:${err instanceof Error ? err.message : String(err)}`);
    }
    if (!adapter) {
      out.adapter = { ok: false };
      return out;
    }

    const adapterInfo = adapter.info ? adapter.info : null;
    out.adapter = {
      ok: true,
      isFallbackAdapter: adapter.isFallbackAdapter ?? null,
      features: Array.from(adapter.features ?? []).map(String).sort(),
      limits: adapter.limits ? Object.fromEntries(Object.entries(adapter.limits)) : null,
      info: adapterInfo ? {
        vendor: adapterInfo.vendor ?? null,
        architecture: adapterInfo.architecture ?? null,
        device: adapterInfo.device ?? null,
        description: adapterInfo.description ?? null,
      } : null,
    };

    let device = null;
    try {
      device = await adapter.requestDevice();
      device.lost.then((info) => {
        out.deviceLost = {
          reason: info.reason ?? null,
          message: info.message ?? null,
        };
      });

      out.device = {
        ok: true,
        features: Array.from(device.features ?? []).map(String).sort(),
        limits: device.limits ? Object.fromEntries(Object.entries(device.limits)) : null,
      };

      // Submit a trivial empty command buffer to ensure the queue is alive.
      const encoder = device.createCommandEncoder();
      const cmd = encoder.finish();
      device.queue.submit([cmd]);
      await device.queue.onSubmittedWorkDone();
    } catch (err) {
      out.device = {
        ok: false,
        error: err instanceof Error ? err.message : String(err),
      };
      out.errors.push(`requestDevice:${err instanceof Error ? err.message : String(err)}`);
    }

    return out;
  });

  const report = {
    ts: new Date().toISOString(),
    baseURL: origin,
    clearedSiteData: process.env.CLEAR_SITE_DATA === '1',
    cdp: {
      error: cdpError,
      systemInfo: sysInfo,
      gpuInfo,
    },
    webgpu,
  };

  const outDir = path.join(process.cwd(), 'test-results');
  fs.mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, 'gpu-health.json');
  fs.writeFileSync(outPath, JSON.stringify(report, null, 2));

  // High-signal console summary in test output.
  // Keep this short: it’s meant to be scanned.
  // eslint-disable-next-line no-console
  console.log('[gpu-health] wrote', outPath);
  // eslint-disable-next-line no-console
  console.log('[gpu-health] webgpu.supported=', report.webgpu.supported);
  // eslint-disable-next-line no-console
  console.log('[gpu-health] adapter.ok=', report.webgpu.adapter?.ok);
  // eslint-disable-next-line no-console
  console.log('[gpu-health] device.ok=', report.webgpu.device?.ok);
  // eslint-disable-next-line no-console
  if (report.webgpu.device?.ok === false && report.webgpu.device?.error) {
    const first = String(report.webgpu.device.error).split(/\r?\n/)[0].trim();
    console.log('[gpu-health] device.error=', first);
  }
});
