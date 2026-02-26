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
  renderPipeline: GPURenderPipeline;
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
  onRuntimeWarning: (warning: string | null) => void;
  onError: (message: string) => void;
  onGridUnsupported: (nextGrid: number, maxStorageBinding: number) => void;
  pushTimeline: (message: string) => void;
  createResources: (device: GPUDevice, format: GPUTextureFormat, gridSize: number) => FireVolumeResources<TTransport>;
  updateUniforms: (transport: TTransport, now: number, input: UniformUpdateInput) => void;
}

export const startFireVolumeSystem = <TTransport extends FireVolumeTransport, TParams extends FireVolumeBaseParams>(
  options: StartFireVolumeSystemOptions<TTransport, TParams>
) => {
  let animationFrameId = 0;
  let device: GPUDevice | null = null;
  let context: GPUCanvasContext | null = null;
  let isDestroyed = false;

  const {
    canvas,
    gridSize,
    defaultTimeStep,
    refs,
    onStats,
    onRuntimeWarning,
    onError,
    onGridUnsupported,
    pushTimeline,
    createResources,
    updateUniforms,
  } = options;

  const init = async () => {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter || isDestroyed) throw new Error('No adapter');

      const requiredStorageForGrid = gridSize * gridSize * gridSize * 16;
      const requestedStorageLimit = Math.min(adapter.limits.maxStorageBufferBindingSize, requiredStorageForGrid);
      const requestedBufferLimit = Math.min(adapter.limits.maxBufferSize, requiredStorageForGrid);
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
      if (!device || isDestroyed) {
        device?.destroy();
        return;
      }

      const deviceLimits = ((device as any).limits ?? adapter.limits) as GPUSupportedLimits;
      const maxStorageBinding = deviceLimits.maxStorageBufferBindingSize ?? adapter.limits.maxStorageBufferBindingSize;
      const maxDimByStorageBinding = Math.floor(Math.cbrt(maxStorageBinding / 16));
      const supportedGrid = GRID_CANDIDATES.find((candidate) => candidate <= maxDimByStorageBinding) ?? 64;
      if (supportedGrid !== gridSize) {
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

      const {
        transport,
        computePipeline,
        projectionDivPipeline,
        projectionJacobiPipeline,
        projectionGradPipeline,
        renderPipeline,
      } = createResources(device, format, gridSize);

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
        const budget = QUALITY_BUDGETS[activeQuality];
        const qualityBoost = activeQuality === 'accurate' ? 1.5 : 1.0;
        const adaptiveScale = activeQuality === 'accurate' ? 1.0 : refs.adaptiveStepScaleRef.current;
        const stepQuality = clamp(refs.paramsRef.current.stepQuality * qualityBoost * adaptiveScale, 0.25, 4.0);
        const rayStepBudget = Math.max(24, Math.round(budget.rayStepBudget * (activeQuality === 'accurate' ? 1.0 : adaptiveScale)));
        const occlusionStepBudget = budget.occlusionStepBudget;
        const debugOverlayMode = FIRE_OVERLAY_MODE_TO_UNIFORM[refs.fireOverlayModeRef.current];
        const occlusionMode = FIRE_OCCLUSION_MODE_TO_UNIFORM[refs.fireOcclusionModeRef.current];

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
            simAccumulatorSeconds += frameSeconds;
          } else if (refs.stepFramesRef.current > 0) {
            simAccumulatorSeconds += defaultTimeStep;
          }

          const maxSubsteps = 4;
          let substeps = 0;
          while (
            simAccumulatorSeconds >= defaultTimeStep &&
            substeps < maxSubsteps &&
            (refs.playingRef.current || refs.stepFramesRef.current > 0)
          ) {
            const stepIndex = simFrame % 2;
            const cp = enc.beginComputePass();
            cp.setPipeline(computePipeline);
            cp.setBindGroup(0, transport.physicsGroups[stepIndex]);
            const wc = Math.ceil(transport.dim / 4);
            cp.dispatchWorkgroups(wc, wc, wc);
            cp.end();

            const velocityTarget = stepIndex === 0 ? transport.velocityB : transport.velocityA;
            (enc as any).copyBufferToBuffer(velocityTarget, 0, transport.velocityScratch, 0, transport.velocityBufferSize);

            const divPass = enc.beginComputePass();
            divPass.setPipeline(projectionDivPipeline);
            divPass.setBindGroup(0, transport.projectionDivGroups[stepIndex]);
            divPass.dispatchWorkgroups(wc, wc, wc);
            divPass.end();

            const jacobiIterations = activeQuality === 'accurate' ? 12 : 6;
            let pressurePing = 0;
            for (let iter = 0; iter < jacobiIterations; iter++) {
              const jacobiPass = enc.beginComputePass();
              jacobiPass.setPipeline(projectionJacobiPipeline);
              jacobiPass.setBindGroup(0, transport.projectionJacobiGroups[pressurePing]);
              jacobiPass.dispatchWorkgroups(wc, wc, wc);
              jacobiPass.end();
              pressurePing = 1 - pressurePing;
            }

            const pressureIndex = pressurePing === 0 ? 0 : 1;
            const gradPass = enc.beginComputePass();
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

          if (substeps === maxSubsteps && simAccumulatorSeconds > defaultTimeStep * 2) {
            simAccumulatorSeconds = defaultTimeStep * 2;
          }
          if (substeps > 0) {
            refs.worldNeedsRenderRef.current = true;
          }
        }
        const simMsThisFrame = performance.now() - simStartMs;

        const fireStartMs = performance.now();
        const rp = enc.beginRenderPass({
          colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
            loadOp: 'clear',
            storeOp: 'store',
          }],
        });
        if (refs.compositionModeRef.current !== 'world_only') {
          rp.setPipeline(renderPipeline);
          rp.setBindGroup(0, transport.renderGroups[activeRenderGroup]);
          rp.draw(6);
        }
        rp.end();
        device.queue.submit([enc.finish()]);
        const fireMsThisFrame = performance.now() - fireStartMs;

        let worldMsThisFrame = 0;
        if (refs.worldNeedsRenderRef.current && refs.renderWorldRef.current) {
          if (refs.compositionModeRef.current !== 'fire_only') {
            worldMsThisFrame = refs.renderWorldRef.current();
          }
          refs.worldNeedsRenderRef.current = false;
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
          });

          if (activeQuality === 'realtime') {
            let nextScale = refs.adaptiveStepScaleRef.current;
            if (frameTimeMs > 18) nextScale *= 0.92;
            else if (frameTimeMs > 10) nextScale *= 0.96;
            else if (frameTimeMs < 7) nextScale *= 1.04;
            refs.adaptiveStepScaleRef.current = clamp(nextScale, 0.85, 1.0);
          } else {
            refs.adaptiveStepScaleRef.current = 1.0;
          }

          const overBudget = frameTimeMs > budget.frameBudgetMs * 1.08;
          const budgetWarning = overBudget
            ? `Frame budget exceeded (${frameTimeMs.toFixed(1)}ms > ${budget.frameBudgetMs.toFixed(1)}ms).`
            : '';
          if (budgetWarning !== lastBudgetWarning) {
            lastBudgetWarning = budgetWarning;
            onRuntimeWarning(budgetWarning || null);
            if (budgetWarning) {
              pushTimeline(budgetWarning);
            }
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
    context?.unconfigure();
    device?.destroy();
  };
};
