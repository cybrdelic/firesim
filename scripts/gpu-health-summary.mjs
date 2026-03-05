import fs from 'node:fs';
import path from 'node:path';

const reportPath = path.join(process.cwd(), 'test-results', 'gpu-health.json');

const readJson = (p) => {
  try {
    return JSON.parse(fs.readFileSync(p, 'utf8'));
  } catch {
    return null;
  }
};

const firstLine = (s) => String(s ?? '').split(/\r?\n/)[0].trim();

const print = (line = '') => process.stdout.write(`${line}\n`);

const report = readJson(reportPath);
if (!report) {
  print(`[gpu-health:summary] No report found at ${reportPath}`);
  print('[gpu-health:summary] Run: pnpm -s gpu:health');
  process.exit(0);
}

const webgpu = report.webgpu ?? {};
const adapter = webgpu.adapter ?? null;
const device = webgpu.device ?? null;

print(`[gpu-health:summary] ${reportPath}`);
print(`- WebGPU API present: ${Boolean(webgpu.supported)}`);
print(`- Adapter: ${adapter?.ok === true ? 'OK' : 'NOT AVAILABLE'}`);

if (adapter?.ok) {
  const info = adapter.info ?? {};
  const vendor = info.vendor ?? 'unknown';
  const arch = info.architecture ?? 'unknown';
  const desc = info.description ?? '';
  print(`- Adapter info: vendor=${vendor} arch=${arch}${desc ? ` desc=${desc}` : ''}`);
}

if (device?.ok === true) {
  print('- Device: OK');
  process.exit(0);
}

if (device?.ok === false) {
  print('- Device: FAILED');
  const errLine = firstLine(device.error);
  if (errLine) print(`- Error: ${errLine}`);

  const fullErr = String(device.error ?? '');
  if (/dxil\.dll/i.test(fullErr) || /EnsureDXCLibraries/i.test(fullErr) || /DXC/i.test(fullErr)) {
    print('');
    print('[diagnosis] Chromium WebGPU (D3D12) could not load DXIL/DXC (dxil.dll).');
    print('[meaning] This is an environment/driver/runtime dependency issue, not your app logic.');
    print('[fixes]');
    print('- Try headed run: pnpm -s gpu:health:headed');
    print('- Update NVIDIA/AMD/Intel GPU driver (clean install if needed)');
    print('- Run Windows Update (DXIL components ship via OS updates)');
    print('- Verify dxil.dll exists in C:/Windows/System32 and isn’t blocked/corrupt');
    print('- If you’re using Canary/Dev Chrome, try Stable');
    process.exit(0);
  }

  if (/out[- ]of[- ]memory/i.test(fullErr)) {
    print('');
    print('[diagnosis] WebGPU device creation failed due to memory pressure.');
    print('[fixes] Reduce grid/texture sizes; reuse GPU resources; close other GPU-heavy apps.');
    process.exit(0);
  }

  print('');
  print('[diagnosis] WebGPU device creation failed (unknown category).');
  print('[next] Try: pnpm -s gpu:health:headed, then re-run after updating GPU driver/Windows.');
  process.exit(0);
}

print('- Device: not created (no details)');
if (Array.isArray(webgpu.errors) && webgpu.errors.length) {
  print(`- Errors: ${firstLine(webgpu.errors[0])}`);
}
