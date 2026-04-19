import type { MutableRefObject } from 'react';
import {
    FIRE_OCCLUSION_MODE_TO_UNIFORM,
    FIRE_OVERLAY_MODE_TO_UNIFORM,
    QUALITY_BUDGETS,
    type CompositionDebugMode,
    type FireOcclusionMode,
    type FireOverlayMode,
    type QualityMode,
    type RuntimePerfCounters,
} from './debugConfig';

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const GRID_CANDIDATES = [256, 192, 128, 64] as const;

export interface FireVolumeCameraState {
  pos: [number, number, number];
  target: [number, number, number];
}

export interface FireVolumeBaseParams {
  stepQuality: number;
  scattering: number;
  absorption: number;
}

export interface FireVolumeRefs<TParams extends FireVolumeBaseParams> {
  paramsRef: MutableRefObject<TParams>;
  cameraRef: MutableRefObject<FireVolumeCameraState>;
  playingRef: MutableRefObject<boolean>;
  sceneRef: MutableRefObject<number>;
  smokeEnabledRef: MutableRefObject<boolean>;
  qualityModeRef: MutableRefObject<QualityMode>;
  compositionModeRef: MutableRefObject<CompositionDebugMode>;
  fireOcclusionModeRef: MutableRefObject<FireOcclusionMode>;
  fireOverlayModeRef: MutableRefObject<FireOverlayMode>;
  adaptiveStepScaleRef: MutableRefObject<number>;
  stepFramesRef: MutableRefObject<number>;
  worldNeedsRenderRef: MutableRefObject<boolean>;
  renderWorldRef: MutableRefObject<(() => number) | null>;
}

export interface FireVolumeTransport {
  dim: number;
  physicsGroups: GPUBindGroup[];
  projectionDivGroups: GPUBindGroup[];
  projectionJacobiGroups: GPUBindGroup[];
  projectionGradGroups: GPUBindGroup[][];
  renderGroups: GPUBindGroup[];
  occupancyGroups: GPUBindGroup[];
  macrocellsPerAxis: number;
  rayStepCounter: GPUBuffer;
  velocityA: GPUBuffer;
  velocityB: GPUBuffer;
  velocityScratch: GPUBuffer;
  velocityBufferSize: number;
}

export interface FireVolumeResources<TTransport extends FireVolumeTransport> {
  transport: TTransport;
  computePipeline: GPUComputePipeline;
  projectionDivPipeline: GPUComputePipeline;
  projectionJacobiPipeline: GPUComputePipeline;
  projectionGradPipeline: GPUComputePipeline;
  occupancyPipeline: GPUComputePipeline;
  renderPipeline: GPURenderPipeline;
  upsamplePipeline: GPURenderPipeline;
  upsampleBindGroupLayout: GPUBindGroupLayout;
  upsampleSampler: GPUSampler;
  temporalBlendPipeline: GPURenderPipeline;
  temporalBlendBGL: GPUBindGroupLayout;
}

export interface UniformUpdateInput {
  stepQuality: number;
  debugOverlayMode: number;
  occlusionMode: number;
  rayStepBudget: number;
  occlusionStepBudget: number;
  renderWidth: number;
  renderHeight: number;
}

interface StartFireVolumeSystemOptions<TTransport extends FireVolumeTransport, TParams extends FireVolumeBaseParams> {
  canvas: HTMLCanvasElement;
  gridSize: number;
  defaultTimeStep: number;
  refs: FireVolumeRefs<TParams>;
  onStats: (stats: RuntimePerfCounters) => void;
  onFrame?: (frame: number) => void;
  onLog?: (line: string) => void;
  onRuntimeWarning: (warning: string | null) => void;
  onError: (message: string) => void;
  onGridUnsupported: (nextGrid: number, maxStorageBinding: number) => void;
  pushTimeline: (message: string) => void;
  createResources: (
    device: GPUDevice,
    format: GPUTextureFormat,
    gridSize: number
  ) => FireVolumeResources<TTransport> | Promise<FireVolumeResources<TTransport>>;
  updateUniforms: (transport: TTransport, now: number, input: UniformUpdateInput) => void;
}

export const startFireVolumeSystem = <TTransport extends FireVolumeTransport, TParams extends FireVolumeBaseParams>(
  options: StartFireVolumeSystemOptions<TTransport, TParams>
) => {
  let animationFrameId = 0;
  let device: GPUDevice | null = null;
  let context: GPUCanvasContext | null = null;
  let isDestroyed = false;
  let halfResTexture: GPUTexture | null = null;
  let historyTexA: GPUTexture | null = null;
  let historyTexB: GPUTexture | null = null;

  const {
    canvas,
    gridSize,
    defaultTimeStep,
    refs,
    onStats,
    onFrame,
    onLog,
    onRuntimeWarning,
    onError,
    onGridUnsupported,
    pushTimeline,
    createResources,
    updateUniforms,
  } = options;

  const log = (message: string) => {
    try {
      onLog?.(message);
    } catch {
      // Never allow logging failures to impact rendering.
    }
  };

  const init = async () => {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter || isDestroyed) throw new Error('No adapter');

      log(`webgpu: adapter ok; requested grid=${gridSize}`);

      const requiredStorageForGrid = gridSize * gridSize * gridSize * 16;
      const requestedStorageLimit = Math.min(adapter.limits.maxStorageBufferBindingSize, requiredStorageForGrid);
      const requestedBufferLimit = Math.min(adapter.limits.maxBufferSize, requiredStorageForGrid);
      const hasTimestamp = adapter.features.has('timestamp-query');
      try {
        const requiredFeatures: GPUFeatureName[] = hasTimestamp ? ['timestamp-query' as GPUFeatureName] : [];
        device = await (adapter as any).requestDevice({
          requiredFeatures,
          requiredLimits: {
            maxStorageBufferBindingSize: requestedStorageLimit,
            maxBufferSize: requestedBufferLimit,
          },
        });
      } catch {
        try {
          device = await (adapter as any).requestDevice({
            requiredLimits: {
              maxStorageBufferBindingSize: requestedStorageLimit,
              maxBufferSize: requestedBufferLimit,
            },
          });
        } catch {
          device = await adapter.requestDevice();
        }
      }
      if (!device || isDestroyed) {
        device?.destroy();
        return;
      }

      // Handle device loss explicitly to avoid silent failures.
      device.lost.then((info) => {
        if (isDestroyed) return;
        isDestroyed = true;
        cancelAnimationFrame(animationFrameId);
        halfResTexture?.destroy();
        historyTexA?.destroy();
        historyTexB?.destroy();
        context?.unconfigure();
        device?.destroy();
        onError(`WebGPU device lost${info?.reason ? ` (${String(info.reason)})` : ''}: ${info?.message ?? 'unknown reason'}`);
      }).catch(() => {
        // Ignore: device loss handler should never throw.
      });

      const deviceLimits = ((device as any).limits ?? adapter.limits) as GPUSupportedLimits;
      const maxStorageBinding = deviceLimits.maxStorageBufferBindingSize ?? adapter.limits.maxStorageBufferBindingSize;

      log(`webgpu: device acquired; timestamp-query=${String(device.features.has('timestamp-query' as GPUFeatureName))}; maxStorageBufferBindingSize=${String(maxStorageBinding)}`);
      const maxDimByStorageBinding = Math.floor(Math.cbrt(maxStorageBinding / 16));

      // IMPORTANT: a smaller-than-max grid is still valid.
      // Only fall back when the requested grid is too large for device limits.
      if (gridSize > maxDimByStorageBinding) {
        const supportedGrid = GRID_CANDIDATES.find((candidate) => candidate <= maxDimByStorageBinding) ?? 64;
        onGridUnsupported(supportedGrid, maxStorageBinding);
        device.destroy();
        return;
      }

      context = canvas.getContext('webgpu') as GPUCanvasContext;
      const format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device, format, alphaMode: 'premultiplied' });
      if (isDestroyed) {
        device.destroy();
        return;
      }

      let resources: FireVolumeResources<TTransport> | null = null;
      try {
        log('webgpu: creating resources (pipelines, buffers, bind groups)...');
        resources = await Promise.resolve(createResources(device, format, gridSize));
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        const nextGrid = GRID_CANDIDATES.find((candidate) => candidate < gridSize) ?? gridSize;
        log(`webgpu: createResources failed for grid=${gridSize}: ${message}`);
        if (nextGrid !== gridSize) {
          pushTimeline(`Grid ${gridSize} init failed; falling back to ${nextGrid}.`);
          onGridUnsupported(nextGrid, maxStorageBinding);
        } else {
          onError(`Failed to initialize WebGPU resources: ${message}`);
        }

        // Clean up before returning.
        context?.unconfigure();
        device.destroy();
        return;
      }

      const {
        transport,
        computePipeline,
        projectionDivPipeline,
        projectionJacobiPipeline,
        projectionGradPipeline,
        occupancyPipeline,
        renderPipeline,
        upsamplePipeline,
        upsampleBindGroupLayout,
        upsampleSampler,
        temporalBlendPipeline,
        temporalBlendBGL,
      } = resources;

      // Task A: GPU timestamp queries
      // 10 timestamps: [0]advect-start [1]advect-end [2]div-start [3]div-end
      //                [4]jacobi-start [5]jacobi-end [6]grad-start [7]grad-end
      //                [8]raymarch-start [9]raymarch-end
      const TIMESTAMP_COUNT = 10;
      const supportsTimestamp = device.features.has('timestamp-query' as GPUFeatureName);
      let querySet: GPUQuerySet | null = null;
      let queryResolveBuffer: GPUBuffer | null = null;
      let queryReadBuffer: GPUBuffer | null = null;
      let gpuTimingPending = false;
      let lastGpuTiming: { advect: number; divergence: number; jacobi: number; gradient: number; raymarch: number; total: number } | null = null;

      if (supportsTimestamp) {
        querySet = device.createQuerySet({ type: 'timestamp', count: TIMESTAMP_COUNT });
        queryResolveBuffer = device.createBuffer({
          size: TIMESTAMP_COUNT * 8,
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        queryReadBuffer = device.createBuffer({
          size: TIMESTAMP_COUNT * 8,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
      }

      // Task D: Half-res volume render state
      let halfResView: GPUTextureView | null = null;
      let upsampleBindGroupA: GPUBindGroup | null = null;
      let upsampleBindGroupB: GPUBindGroup | null = null;
      let halfResToHistoryBindGroup: GPUBindGroup | null = null;
      let halfResW = 0;
      let halfResH = 0;

      // Task E: Temporal accumulation - history ping-pong
      let historyViewA: GPUTextureView | null = null;
      let historyViewB: GPUTextureView | null = null;
      let temporalBlendGroupA: GPUBindGroup | null = null; // reads halfRes + historyA → writes historyB
      let temporalBlendGroupB: GPUBindGroup | null = null; // reads halfRes + historyB → writes historyA
      let historyPing = 0; // 0: write to B (read A), 1: write to A (read B)
      let historyValid = false;

      // Task B: Ray step counter readback
      const stepReadBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      let stepReadPending = false;
      let lastAvgSteps: number | null = null;
      let lastRenderPixelCount = 0;
      const zeroU32 = new Uint32Array([0]);

      let simFrame = 0;
      let activeRenderGroup = 0;
      let simAccumulatorSeconds = 0;
      let lastTime = performance.now();
      let statsTimer = 0;
      let frameAccumMs = 0;
      let rafCount = 0;
      let simAccumMs = 0;
      let fireAccumMs = 0;
      let worldAccumMs = 0;
      let compositeAccumMs = 0;
      let substepsAccum = 0;
      let lastBudgetWarning = '';
      let lastWorldRenderAtMs = Number.NEGATIVE_INFINITY;

      // Logging can be surprisingly expensive (string churn -> GC spikes).
      // Keep it informative but throttled so it doesn't become the bottleneck.
      const FRAME_LOG_INTERVAL_MS = 250;
      let lastFrameLogMs = -1;
      const SIM_DETAIL_LOG_INTERVAL_MS = 500;
      let lastSimDetailLogMs = -1;
      const BUDGET_LOG_MIN_INTERVAL_MS = 1000;
      let lastBudgetLogMs = -1;

      // Throttle GPU readbacks to reduce jitter from frequent map/unmap + readback copies.
      // (Readbacks are best-effort diagnostics, not required for rendering correctness.)
      const READBACK_INTERVAL_MS = 250;
      let lastReadbackMs = 0;

      let loggedOcclusionNote = false;

      const render = () => {
        if (isDestroyed || !device || !context) return;
        const frameStartMs = performance.now();
        const now = frameStartMs;
        const dt = now - lastTime;
        lastTime = now;
        const frameSeconds = Math.min(dt, 100) / 1000;
        rafCount += 1;
        frameAccumMs += dt;
        statsTimer += dt;
        const shouldAdvance = refs.playingRef.current || refs.stepFramesRef.current > 0;
        const activeQuality = refs.qualityModeRef.current;
        const useHalfResFirePass = activeQuality === 'realtime';
        const budget = QUALITY_BUDGETS[activeQuality];
        const qualityBoost = activeQuality === 'accurate' ? 1.15 : 1.0;
        const adaptiveScale = activeQuality === 'accurate' ? 1.0 : refs.adaptiveStepScaleRef.current;
        const stepQuality = clamp(refs.paramsRef.current.stepQuality * qualityBoost * adaptiveScale, 0.25, 4.0);
        const rayStepBudget = Math.max(24, Math.round(budget.rayStepBudget * (activeQuality === 'accurate' ? 1.0 : adaptiveScale)));
        const occlusionStepBudget =
          activeQuality === 'realtime'
            ? clamp(Math.round(budget.occlusionStepBudget * adaptiveScale), 12, budget.occlusionStepBudget)
            : budget.occlusionStepBudget;
        const debugOverlayMode = FIRE_OVERLAY_MODE_TO_UNIFORM[refs.fireOverlayModeRef.current];
        const occlusionMode = FIRE_OCCLUSION_MODE_TO_UNIFORM[refs.fireOcclusionModeRef.current];

        // Projection cost (Jacobi iterations) is often the dominant GPU budget item.
        // In realtime mode, scale iterations down with the adaptive scale so we can
        // actually hit the frame budget on slower GPUs / larger grids.
        const jacobiIterations =
          activeQuality === 'accurate'
            ? (transport.dim >= 192 ? 8 : 10)
            : activeQuality === 'realtime'
              ? clamp(Math.round(6 * adaptiveScale), 2, 6)
              : 6;

        if (!loggedOcclusionNote) {
          loggedOcclusionNote = true;
          if (refs.fireOcclusionModeRef.current === 'depth_coupled') {
            log('boundary: fireOcclusionMode=depth_coupled currently uses analytic SDF occlusion (world depth is not shared with the Three.js WebGPU renderer)');
          }
        }

        const shouldLogFrame =
          lastFrameLogMs < 0 ||
          now - lastFrameLogMs >= FRAME_LOG_INTERVAL_MS ||
          dt >= 50;
        if (shouldLogFrame) {
          lastFrameLogMs = now;
          log(
            `frame: now=${now.toFixed(2)}ms simFrame=${simFrame} dt=${dt.toFixed(2)}ms quality=${activeQuality} stepQuality=${stepQuality.toFixed(3)} adapt=${adaptiveScale.toFixed(3)} jacobi=${jacobiIterations} budgets(ray=${rayStepBudget},occ=${occlusionStepBudget}) comp=${refs.compositionModeRef.current} occ=${refs.fireOcclusionModeRef.current} overlay=${refs.fireOverlayModeRef.current}`
          );
        }

        updateUniforms(transport, now, {
          stepQuality,
          debugOverlayMode,
          occlusionMode,
          rayStepBudget,
          occlusionStepBudget,
          renderWidth: canvas.width,
          renderHeight: canvas.height,
        });

        const enc = device.createCommandEncoder();
        const simStartMs = performance.now();
        let substepsThisFrame = 0;
        if (shouldAdvance) {
          if (refs.playingRef.current) {
            // In realtime mode, avoid the "catch-up spiral" after a stutter:
            // one slow frame should not force 4 heavy sim substeps, which makes
            // the next frame even slower.
            const maxAccum = activeQuality === 'realtime' ? defaultTimeStep * 2 : defaultTimeStep * 6;
            simAccumulatorSeconds = Math.min(simAccumulatorSeconds + frameSeconds, maxAccum);
          } else if (refs.stepFramesRef.current > 0) {
            simAccumulatorSeconds += defaultTimeStep;
          }

          const maxSubsteps = activeQuality === 'realtime' ? 2 : 4;
          let substeps = 0;
          while (
            simAccumulatorSeconds >= defaultTimeStep &&
            substeps < maxSubsteps &&
            (refs.playingRef.current || refs.stepFramesRef.current > 0)
          ) {
            const stepIndex = simFrame % 2;

            const shouldLogSimDetail =
              (dt >= 50 || maxSubsteps > 2) &&
              (lastSimDetailLogMs < 0 || now - lastSimDetailLogMs >= SIM_DETAIL_LOG_INTERVAL_MS);
            if (shouldLogSimDetail) {
              lastSimDetailLogMs = now;
              log(`simStep: simFrame=${simFrame} substep=${substeps} stepIndex=${stepIndex}`);
            }
            // Determine if this is the last substep (for GPU timestamp)
            const isLastSubstep = supportsTimestamp && querySet && (
              simAccumulatorSeconds < defaultTimeStep * 2 || substeps === maxSubsteps - 1
            );
            const cp = enc.beginComputePass(isLastSubstep ? {
              timestampWrites: { querySet: querySet!, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
            } : undefined);
            cp.setPipeline(computePipeline);
            cp.setBindGroup(0, transport.physicsGroups[stepIndex]);
            const wc = Math.ceil(transport.dim / 4);
            cp.dispatchWorkgroups(wc, wc, wc);
            cp.end();

            const velocityTarget = stepIndex === 0 ? transport.velocityB : transport.velocityA;
            (enc as any).copyBufferToBuffer(velocityTarget, 0, transport.velocityScratch, 0, transport.velocityBufferSize);

            const divPass = enc.beginComputePass(isLastSubstep ? {
              timestampWrites: { querySet: querySet!, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 },
            } : undefined);
            divPass.setPipeline(projectionDivPipeline);
            divPass.setBindGroup(0, transport.projectionDivGroups[stepIndex]);
            divPass.dispatchWorkgroups(wc, wc, wc);
            divPass.end();

            if (dt >= 50 && (lastSimDetailLogMs < 0 || now - lastSimDetailLogMs >= SIM_DETAIL_LOG_INTERVAL_MS)) {
              lastSimDetailLogMs = now;
              log(`projection: jacobiIterations=${jacobiIterations}`);
            }
            let pressurePing = 0;
            for (let iter = 0; iter < jacobiIterations; iter++) {
              // Timestamp the full Jacobi block: start of first, end of last
              const jacobiTsStart = isLastSubstep && iter === 0;
              const jacobiTsEnd = isLastSubstep && iter === jacobiIterations - 1;
              const jacobiTsWrites = (jacobiTsStart || jacobiTsEnd) ? {
                querySet: querySet!,
                ...(jacobiTsStart ? { beginningOfPassWriteIndex: 4 } : {}),
                ...(jacobiTsEnd ? { endOfPassWriteIndex: 5 } : {}),
              } : undefined;
              const jacobiPass = enc.beginComputePass(jacobiTsWrites ? { timestampWrites: jacobiTsWrites } : undefined);
              jacobiPass.setPipeline(projectionJacobiPipeline);
              jacobiPass.setBindGroup(0, transport.projectionJacobiGroups[pressurePing]);
              jacobiPass.dispatchWorkgroups(wc, wc, wc);
              jacobiPass.end();
              pressurePing = 1 - pressurePing;
            }

            const pressureIndex = pressurePing === 0 ? 0 : 1;
            const gradPass = enc.beginComputePass(isLastSubstep ? {
              timestampWrites: { querySet: querySet!, beginningOfPassWriteIndex: 6, endOfPassWriteIndex: 7 },
            } : undefined);
            gradPass.setPipeline(projectionGradPipeline);
            gradPass.setBindGroup(0, transport.projectionGradGroups[stepIndex][pressureIndex]);
            gradPass.dispatchWorkgroups(wc, wc, wc);
            gradPass.end();

            activeRenderGroup = stepIndex;
            simFrame += 1;
            substeps += 1;
            substepsThisFrame += 1;
            simAccumulatorSeconds -= defaultTimeStep;

            if (!refs.playingRef.current && refs.stepFramesRef.current > 0) {
              refs.stepFramesRef.current -= 1;
            }
          }

          // If we hit our substep cap, drop remaining accumulator rather than
          // trying to catch up over many future frames (keeps realtime stable).
          if (substeps === maxSubsteps && simAccumulatorSeconds >= defaultTimeStep) {
            if (activeQuality === 'realtime') {
              const dropped = simAccumulatorSeconds;
              simAccumulatorSeconds = 0;
              if (dt >= 50) {
                log(`sim: dropped ${dropped.toFixed(4)}s accumulator to stay realtime`);
              }
            } else {
              simAccumulatorSeconds = Math.min(simAccumulatorSeconds, defaultTimeStep * 2);
            }
          }
          if (substeps > 0) {
            refs.worldNeedsRenderRef.current = true;
          }
        }
        const simMsThisFrame = performance.now() - simStartMs;

        // Task C: Build macrocell occupancy grid before render pass
        {
          const occPass = enc.beginComputePass();
          occPass.setPipeline(occupancyPipeline);
          occPass.setBindGroup(0, transport.occupancyGroups[activeRenderGroup]);
          const macroWc = Math.ceil(transport.macrocellsPerAxis / 4);
          occPass.dispatchWorkgroups(macroWc, macroWc, macroWc);
          occPass.end();
        }

        const fireStartMs = performance.now();

        // Task B: Reset ray step counter before render
        device.queue.writeBuffer(transport.rayStepCounter, 0, zeroU32);

        const canvasTex = context.getCurrentTexture();
        const cw = canvasTex.width;
        const ch = canvasTex.height;
        if (useHalfResFirePass) {
          const needW = Math.max(1, Math.ceil(cw / 2));
          const needH = Math.max(1, Math.ceil(ch / 2));
          if (needW !== halfResW || needH !== halfResH) {
            log(`resize: halfRes ${halfResW}x${halfResH} -> ${needW}x${needH}`);
            halfResTexture?.destroy();
            historyTexA?.destroy();
            historyTexB?.destroy();
            halfResW = needW;
            halfResH = needH;
            const texDesc = {
              size: [halfResW, halfResH, 1] as [number, number, number],
              format,
              usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            };
            halfResTexture = device.createTexture({ ...texDesc, label: 'HalfResVolume' });
            historyTexA = device.createTexture({ ...texDesc, label: 'HistoryA' });
            historyTexB = device.createTexture({ ...texDesc, label: 'HistoryB' });
            halfResView = halfResTexture.createView();
            historyViewA = historyTexA.createView();
            historyViewB = historyTexB.createView();

            upsampleBindGroupA = device.createBindGroup({
              layout: upsampleBindGroupLayout,
              entries: [
                { binding: 0, resource: historyViewA },
                { binding: 1, resource: upsampleSampler },
              ],
              label: 'Upsample_BG_A',
            });
            upsampleBindGroupB = device.createBindGroup({
              layout: upsampleBindGroupLayout,
              entries: [
                { binding: 0, resource: historyViewB },
                { binding: 1, resource: upsampleSampler },
              ],
              label: 'Upsample_BG_B',
            });

            halfResToHistoryBindGroup = device.createBindGroup({
              layout: upsampleBindGroupLayout,
              entries: [
                { binding: 0, resource: halfResView },
                { binding: 1, resource: upsampleSampler },
              ],
              label: 'HalfResToHistory_BG',
            });

            temporalBlendGroupA = device.createBindGroup({
              layout: temporalBlendBGL,
              entries: [
                { binding: 0, resource: halfResView },
                { binding: 1, resource: historyViewA },
                { binding: 2, resource: upsampleSampler },
              ],
              label: 'TemporalBlend_A',
            });
            temporalBlendGroupB = device.createBindGroup({
              layout: temporalBlendBGL,
              entries: [
                { binding: 0, resource: halfResView },
                { binding: 1, resource: historyViewB },
                { binding: 2, resource: upsampleSampler },
              ],
              label: 'TemporalBlend_B',
            });
            historyPing = 0;
            historyValid = false;
          }

          lastRenderPixelCount = needW * needH;

          const rp = enc.beginRenderPass({
            colorAttachments: [{
              view: halfResView!,
              clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
              loadOp: 'clear',
              storeOp: 'store',
            }],
            ...(supportsTimestamp && querySet ? {
              timestampWrites: { querySet, beginningOfPassWriteIndex: 8, endOfPassWriteIndex: 9 },
            } : {}),
          });
          if (refs.compositionModeRef.current !== 'world_only') {
            rp.setPipeline(renderPipeline);
            rp.setBindGroup(0, transport.renderGroups[activeRenderGroup]);
            rp.draw(6);
          }
          rp.end();

          if (!historyValid && refs.compositionModeRef.current !== 'world_only') {
            const initHistoryPass = enc.beginRenderPass({
              colorAttachments: [{
                view: historyViewA!,
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
                loadOp: 'clear',
                storeOp: 'store',
              }],
            });
            initHistoryPass.setPipeline(upsamplePipeline);
            initHistoryPass.setBindGroup(0, halfResToHistoryBindGroup!);
            initHistoryPass.draw(6);
            initHistoryPass.end();
            historyValid = true;
          }

          const writeToB = historyPing === 0;
          const blendTarget = writeToB ? historyViewB! : historyViewA!;
          const blendGroup = writeToB ? temporalBlendGroupA : temporalBlendGroupB;
          if (refs.compositionModeRef.current !== 'world_only' && blendGroup) {
            const tbPass = enc.beginRenderPass({
              colorAttachments: [{
                view: blendTarget,
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
                loadOp: 'clear',
                storeOp: 'store',
              }],
            });
            tbPass.setPipeline(temporalBlendPipeline);
            tbPass.setBindGroup(0, blendGroup);
            tbPass.draw(6);
            tbPass.end();
          }

          const upsampleBindGroup = writeToB ? upsampleBindGroupB : upsampleBindGroupA;
          historyPing = 1 - historyPing;

          const upPass = enc.beginRenderPass({
            colorAttachments: [{
              view: canvasTex.createView(),
              clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
              loadOp: 'clear',
              storeOp: 'store',
            }],
          });
          if (refs.compositionModeRef.current !== 'world_only' && upsampleBindGroup) {
            upPass.setPipeline(upsamplePipeline);
            upPass.setBindGroup(0, upsampleBindGroup);
            upPass.draw(6);
          }
          upPass.end();
        } else {
          lastRenderPixelCount = cw * ch;
          historyValid = false;

          const rp = enc.beginRenderPass({
            colorAttachments: [{
              view: canvasTex.createView(),
              clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
              loadOp: 'clear',
              storeOp: 'store',
            }],
            ...(supportsTimestamp && querySet ? {
              timestampWrites: { querySet, beginningOfPassWriteIndex: 8, endOfPassWriteIndex: 9 },
            } : {}),
          });
          if (refs.compositionModeRef.current !== 'world_only') {
            rp.setPipeline(renderPipeline);
            rp.setBindGroup(0, transport.renderGroups[activeRenderGroup]);
            rp.draw(6);
          }
          rp.end();
        }

        // Readback copies: only copy into a readback buffer when it's not mapped / pending map,
        // and at a throttled cadence to reduce driver stalls / jitter.
        const allowReadback = now - lastReadbackMs >= READBACK_INTERVAL_MS;
        const needTimestampCopy =
          allowReadback &&
          supportsTimestamp &&
          querySet &&
          queryResolveBuffer &&
          queryReadBuffer &&
          !gpuTimingPending;
        const needStepCopy = allowReadback && !stepReadPending && refs.compositionModeRef.current !== 'world_only';
        if (needTimestampCopy || needStepCopy) lastReadbackMs = now;

        // Keep resolve/copy in the main encoder (single submit).
        if (needTimestampCopy) {
          enc.resolveQuerySet(querySet!, 0, TIMESTAMP_COUNT, queryResolveBuffer!, 0);
          enc.copyBufferToBuffer(queryResolveBuffer!, 0, queryReadBuffer!, 0, TIMESTAMP_COUNT * 8);
        }
        if (needStepCopy) {
          enc.copyBufferToBuffer(transport.rayStepCounter, 0, stepReadBuffer, 0, 4);
        }

        device.queue.submit([enc.finish()]);
        const fireMsThisFrame = performance.now() - fireStartMs;

        // Async readback of GPU timestamps (non-blocking)
        if (needTimestampCopy) {
          gpuTimingPending = true;
          queryReadBuffer!.mapAsync(GPUMapMode.READ).then(() => {
            // Timestamp queries are u64. Compute deltas in BigInt first to avoid
            // signed interpretation and precision loss from large absolute values.
            const times = new BigUint64Array(queryReadBuffer!.getMappedRange());
            const toMs = (start: number, end: number) => {
              const deltaNs = times[end] - times[start];
              return Number(deltaNs) / 1e6;
            };
            lastGpuTiming = {
              advect: toMs(0, 1),
              divergence: toMs(2, 3),
              jacobi: toMs(4, 5),
              gradient: toMs(6, 7),
              raymarch: toMs(8, 9),
              total: toMs(0, 1) + toMs(2, 3) + toMs(4, 5) + toMs(6, 7) + toMs(8, 9),
            };
            queryReadBuffer!.unmap();
            gpuTimingPending = false;
          }).catch(() => {
            gpuTimingPending = false;
          });
        }

        // Task B: Async readback of ray step counter
        if (needStepCopy) {
          stepReadPending = true;
          stepReadBuffer.mapAsync(GPUMapMode.READ).then(() => {
            const data = new Uint32Array(stepReadBuffer.getMappedRange());
            const totalSteps = data[0];
            const pixelCount = Math.max(1, lastRenderPixelCount);
            lastAvgSteps = pixelCount > 0 ? totalSteps / pixelCount : 0;
            stepReadBuffer.unmap();
            stepReadPending = false;
          }).catch(() => {
            stepReadPending = false;
          });
        }

        let worldMsThisFrame = 0;
        if (refs.worldNeedsRenderRef.current && refs.renderWorldRef.current) {
          const worldRenderIntervalMs = activeQuality === 'accurate' ? 33 : 50;
          const worldRenderDue = now - lastWorldRenderAtMs >= worldRenderIntervalMs;
          if (refs.compositionModeRef.current !== 'fire_only' && worldRenderDue) {
            worldMsThisFrame = refs.renderWorldRef.current();
            lastWorldRenderAtMs = now;
            refs.worldNeedsRenderRef.current = false;
          }
        }

        const frameElapsedMs = performance.now() - frameStartMs;
        const compositeMsThisFrame = Math.max(0, frameElapsedMs - simMsThisFrame - fireMsThisFrame - worldMsThisFrame);
        simAccumMs += simMsThisFrame;
        fireAccumMs += fireMsThisFrame;
        worldAccumMs += worldMsThisFrame;
        compositeAccumMs += compositeMsThisFrame;
        substepsAccum += substepsThisFrame;

        if (statsTimer >= 1000) {
          const safeCount = Math.max(1, rafCount);
          const frameTimeMs = frameAccumMs / safeCount;
          const avgSimMs = simAccumMs / safeCount;
          const avgFireMs = fireAccumMs / safeCount;
          const avgWorldMs = worldAccumMs / safeCount;
          const avgCompositeMs = compositeAccumMs / safeCount;
          const avgSubsteps = substepsAccum / safeCount;

          onStats({
            frame: simFrame,
            fps: safeCount,
            frameTimeMs,
            simMs: avgSimMs,
            fireMs: avgFireMs,
            worldMs: avgWorldMs,
            compositeMs: avgCompositeMs,
            substeps: avgSubsteps,
            gpuAdvectMs: lastGpuTiming?.advect ?? null,
            gpuDivergenceMs: lastGpuTiming?.divergence ?? null,
            gpuJacobiMs: lastGpuTiming?.jacobi ?? null,
            gpuGradientMs: lastGpuTiming?.gradient ?? null,
            gpuRaymarchMs: lastGpuTiming?.raymarch ?? null,
            gpuTotalMs: lastGpuTiming?.total ?? null,
            avgRaySteps: lastAvgSteps,
            fieldSampledVoxels: null,
            fieldAvgTemp: null,
            fieldPeakTemp: null,
            fieldAvgSoot: null,
            fieldDivL1: null,
            fieldReactionFraction: null,
          });

          if (activeQuality === 'realtime') {
            let nextScale = refs.adaptiveStepScaleRef.current;
            const gpuTotalMs = lastGpuTiming?.total ?? null;
            const overCpuBudget = frameTimeMs > budget.frameBudgetMs * 1.08;
            const overGpuBudget = gpuTotalMs !== null && gpuTotalMs > budget.frameBudgetMs * 1.05;

            // Prefer GPU-based control when we have timestamps.
            if (overGpuBudget) {
              if (gpuTotalMs! > budget.frameBudgetMs * 1.35) nextScale *= 0.80;
              else if (gpuTotalMs! > budget.frameBudgetMs * 1.20) nextScale *= 0.88;
              else nextScale *= 0.94;
            } else if (overCpuBudget) {
              if (frameTimeMs > budget.frameBudgetMs * 1.35) nextScale *= 0.84;
              else if (frameTimeMs > 18) nextScale *= 0.92;
              else nextScale *= 0.96;
            } else if (frameTimeMs < 7 && (gpuTotalMs === null || gpuTotalMs < budget.frameBudgetMs * 0.80)) {
              nextScale *= 1.04;
            }

            // Allow deeper scaling in realtime so we can actually meet the budget.
            // (This also reduces Jacobi iterations via the formula above.)
            refs.adaptiveStepScaleRef.current = clamp(nextScale, 0.50, 1.0);
          } else {
            refs.adaptiveStepScaleRef.current = 1.0;
          }

          const overBudget = frameTimeMs > budget.frameBudgetMs * 1.08;
          const gpuTotalMs = lastGpuTiming?.total ?? null;
          const gpuAdvectMs = lastGpuTiming?.advect ?? null;
          const gpuJacobiMs = lastGpuTiming?.jacobi ?? null;
          const gpuRaymarchMs = lastGpuTiming?.raymarch ?? null;
          const budgetWarning = overBudget
            ? `Frame budget exceeded (${frameTimeMs.toFixed(1)}ms > ${budget.frameBudgetMs.toFixed(1)}ms). sim=${avgSimMs.toFixed(1)} fire=${avgFireMs.toFixed(1)} world=${avgWorldMs.toFixed(1)} substeps=${avgSubsteps.toFixed(2)} ray=${rayStepBudget} occ=${occlusionStepBudget}`
              + (gpuTotalMs !== null
                ? ` gpu=${gpuTotalMs.toFixed(1)} adv=${(gpuAdvectMs ?? 0).toFixed(1)} jac=${(gpuJacobiMs ?? 0).toFixed(1)} raym=${(gpuRaymarchMs ?? 0).toFixed(1)}`
                : '')
            : '';
          if (budgetWarning !== lastBudgetWarning) {
            lastBudgetWarning = budgetWarning;
            onRuntimeWarning(budgetWarning || null);
            if (budgetWarning) {
              pushTimeline(budgetWarning);
            }
          }

          if (budgetWarning && (lastBudgetLogMs < 0 || now - lastBudgetLogMs >= BUDGET_LOG_MIN_INTERVAL_MS)) {
            lastBudgetLogMs = now;
            log(`budget: ${budgetWarning}`);
          }

          rafCount = 0;
          frameAccumMs = 0;
          simAccumMs = 0;
          fireAccumMs = 0;
          worldAccumMs = 0;
          compositeAccumMs = 0;
          substepsAccum = 0;
          statsTimer = 0;
        }

        onFrame?.(simFrame);

        animationFrameId = requestAnimationFrame(render);
      };

      render();
    } catch (error) {
      if (!isDestroyed) {
        onError(error instanceof Error ? error.message : String(error));
      }
    }
  };

  init();

  return () => {
    isDestroyed = true;
    cancelAnimationFrame(animationFrameId);
    halfResTexture?.destroy();
    historyTexA?.destroy();
    historyTexB?.destroy();
    context?.unconfigure();
    device?.destroy();
  };
};
