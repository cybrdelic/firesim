export type QualityMode = 'realtime' | 'accurate';
export type CompositionDebugMode = 'composited' | 'world_only' | 'fire_only';
export type FireOcclusionMode = 'depth_coupled' | 'analytic_sdf' | 'none';
export type FireOverlayMode = 'final' | 'alpha' | 'occlusion' | 'wood_sdf' | 'combustion';

export interface RuntimePerfCounters {
  frame: number;
  fps: number;
  frameTimeMs: number;
  simMs: number;
  fireMs: number;
  worldMs: number;
  compositeMs: number;
  substeps: number;
  // GPU timestamp-based timing (null if timestamp-query not supported)
  gpuAdvectMs: number | null;
  gpuDivergenceMs: number | null;
  gpuJacobiMs: number | null;
  gpuGradientMs: number | null;
  gpuRaymarchMs: number | null;
  gpuTotalMs: number | null;
  // Task B: Average ray steps per pixel (null if not yet available)
  avgRaySteps: number | null;
  // Field-sampled diagnostics from simulation buffers (null until first sample completes)
  fieldSampledVoxels: number | null;
  fieldAvgTemp: number | null;
  fieldPeakTemp: number | null;
  fieldAvgSoot: number | null;
  fieldDivL1: number | null;
  fieldReactionFraction: number | null;
}

export interface QualityBudget {
  frameBudgetMs: number;
  rayStepBudget: number;
  occlusionStepBudget: number;
}

export const QUALITY_BUDGETS: Record<QualityMode, QualityBudget> = {
  realtime: {
    frameBudgetMs: 16.67,
    rayStepBudget: 104,
    occlusionStepBudget: 32,
  },
  accurate: {
    frameBudgetMs: 33.3,
    rayStepBudget: 136,
    occlusionStepBudget: 40,
  },
};

export const FIRE_OVERLAY_MODE_TO_UNIFORM: Record<FireOverlayMode, number> = {
  final: 0,
  alpha: 1,
  occlusion: 2,
  wood_sdf: 3,
  combustion: 4,
};

export const FIRE_OCCLUSION_MODE_TO_UNIFORM: Record<FireOcclusionMode, number> = {
  none: 0,
  analytic_sdf: 1,
  depth_coupled: 2,
};

export const COMPOSITION_MODE_LABELS: Record<CompositionDebugMode, string> = {
  composited: 'World + Fire',
  world_only: 'World Only',
  fire_only: 'Fire Only',
};

export const FIRE_OCCLUSION_LABELS: Record<FireOcclusionMode, string> = {
  depth_coupled: 'Approx Depth',
  analytic_sdf: 'World SDF',
  none: 'No Occlusion',
};

export const FIRE_OVERLAY_LABELS: Record<FireOverlayMode, string> = {
  final: 'Final',
  alpha: 'Alpha',
  occlusion: 'Occlusion',
  wood_sdf: 'Wood SDF',
  combustion: 'Combustion',
};
