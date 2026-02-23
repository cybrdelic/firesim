# Stability Report

## Hardening Changes

- Added strict parameter sanitization and clamping before uniforms are written.
- Added GPU limit gating for grid sizes based on adapter limits.
- Added device-loss and uncaptured-error handling.
- Added shader compilation diagnostics at startup.
- Added deterministic sweep mode and runtime status/control hooks for automation.
- Added Playwright stability harness (fuzz + snapshot specs with WebGPU capability gating).

## Stability Limits

### Grid Memory Footprint (simulation buffers only)

- `64^3`: ~10 MiB total (double-buffered density + velocity)
- `128^3`: ~80 MiB total
- `192^3`: ~270 MiB total
- `256^3`: ~640 MiB total

### Practical Hardware Boundaries

- A single `256^3` velocity buffer needs ~256 MiB.
- GPUs with `maxStorageBufferBindingSize < 256 MiB` cannot run `256^3`.
- Runtime now auto-downgrades unsupported grid sizes and surfaces a warning.

## Known Runtime Edge Cases

- No WebGPU adapter available: app reports error, automation tests skip.
- Device loss during run: app reports the loss reason/message.
- Out-of-range control values or fuzzed params: values are clamped to safe ranges.
- Invalid gamma/step quality ranges: shader-side guards avoid divide-by-zero/invalid marching.

## Automation Entry Points

- `window.__FIRE_SIM_STATUS__`: readiness, frame, grid, scene, error/warning.
- `window.__FIRE_SIM_CONTROL__`: scene/grid/params setters plus `runFuzz()`.

## Commands

- `npm run typecheck`
- `npm run build`
- `npm run test:stability`
- `npm run test:stability:update`
