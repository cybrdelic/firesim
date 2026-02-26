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
}

export interface QualityBudget {
  frameBudgetMs: number;
  rayStepBudget: number;
  occlusionStepBudget: number;
}

export const QUALITY_BUDGETS: Record<QualityMode, QualityBudget> = {
  realtime: {
    frameBudgetMs: 16.67,
    rayStepBudget: 130,
    occlusionStepBudget: 42,
  },
  accurate: {
    frameBudgetMs: 33.3,
    rayStepBudget: 220,
    occlusionStepBudget: 90,
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
  depth_coupled: 'Depth Coupled',
  analytic_sdf: 'Analytic SDF',
  none: 'No Occlusion',
};

export const FIRE_OVERLAY_LABELS: Record<FireOverlayMode, string> = {
  final: 'Final',
  alpha: 'Alpha',
  occlusion: 'Occlusion',
  wood_sdf: 'Wood SDF',
  combustion: 'Combustion',
};
