import * as THREE from 'three/webgpu';

export type Vec3Tuple = [number, number, number];
export type EulerTuple = [number, number, number];

export interface WoodLogDescriptor {
  id: string;
  length: number;
  radius: number;
  seed: number;
  position: Vec3Tuple;
  rotation: EulerTuple;
}

export interface WoodPileDescriptor {
  logs: WoodLogDescriptor[];
}

export interface WoodLogSegment {
  id: string;
  start: Vec3Tuple;
  end: Vec3Tuple;
  radius: number;
}

export interface WoodLogTransform {
  id: string;
  seed: number;
  length: number;
  radius: number;
  position: THREE.Vector3;
  rotation: THREE.Euler;
}

export interface WoodCombustionLogState {
  burnProgress: number;
  sootProgress: number;
}

export interface WoodCombustionParams {
  emission: number;
  fuelEfficiency: number;
  buoyancy: number;
  burnRate: number;
  smokeDissipation: number;
  absorption: number;
  smokeWeight: number;
  turbSpeed: number;
}

export interface WoodCombustionRuntime {
  logSideMaterial?: THREE.MeshPhysicalMaterial;
  logCapMaterial?: THREE.MeshPhysicalMaterial;
  logBaseSideColor?: THREE.Color;
  logBaseCapColor?: THREE.Color;
  logBurnStateById: Record<string, WoodCombustionLogState>;
  logLastUpdateMs: number;
}

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

export const WOOD_PILE_DESCRIPTOR: WoodPileDescriptor = {
  logs: [
    {
      id: 'log-a',
      length: 0.5086,
      radius: 0.0527,
      seed: 313,
      position: [0.5, 0.1, 0.5],
      rotation: [0, 0, -Math.PI * 0.5 - 0.15],
    },
    {
      id: 'log-b',
      length: 0.5271,
      radius: 0.0551,
      seed: 911,
      position: [0.5, 0.13, 0.5],
      rotation: [Math.PI * 0.5 - 0.15, 0, 0],
    },
    {
      id: 'log-c',
      length: 0.558,
      radius: 0.0485,
      seed: 1907,
      position: [0.5, 0.16, 0.5],
      rotation: [0, Math.PI * 0.25, -Math.PI * 0.5 + 0.1],
    },
  ],
};

const toVec3Tuple = (value: THREE.Vector3): Vec3Tuple => [value.x, value.y, value.z];

export const getWoodLogTransforms = (descriptor: WoodPileDescriptor): WoodLogTransform[] => (
  descriptor.logs.map((log) => ({
    id: log.id,
    seed: log.seed,
    length: log.length,
    radius: log.radius,
    position: new THREE.Vector3(...log.position),
    rotation: new THREE.Euler(...log.rotation, 'XYZ'),
  }))
);

export const getWoodSegmentsFromDescriptor = (descriptor: WoodPileDescriptor): WoodLogSegment[] => (
  descriptor.logs.map((log) => {
    const center = new THREE.Vector3(...log.position);
    const rotation = new THREE.Euler(...log.rotation, 'XYZ');
    const axisHalf = new THREE.Vector3(0, log.length * 0.5, 0).applyEuler(rotation);
    const start = center.clone().sub(axisHalf);
    const end = center.clone().add(axisHalf);
    return {
      id: log.id,
      start: toVec3Tuple(start),
      end: toVec3Tuple(end),
      radius: log.radius,
    };
  })
);

const wgslFloat = (value: number) => Number(value).toFixed(4);

export const buildWoodSdfWgsl = (descriptor: WoodPileDescriptor) => {
  const segments = getWoodSegmentsFromDescriptor(descriptor);
  const distanceNames = segments.map((_, index) => `d${index + 1}`);
  const segmentLines = segments
    .map((segment, index) => {
      const name = distanceNames[index];
      return `  let ${name} = sd_capsule_segment(p, vec3f(${wgslFloat(segment.start[0])}, ${wgslFloat(segment.start[1])}, ${wgslFloat(segment.start[2])}), vec3f(${wgslFloat(segment.end[0])}, ${wgslFloat(segment.end[1])}, ${wgslFloat(segment.end[2])}), ${wgslFloat(segment.radius)});`;
    })
    .join('\n');
  const minExpr = distanceNames.slice(1).reduce((expression, name) => `min(${expression}, ${name})`, distanceNames[0] ?? '1e6');
  return `
fn sd_capsule_segment(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

fn get_wood_sdf(p: vec3f) -> f32 {
${segmentLines}
  return ${minExpr};
}

fn get_wood_normal(p: vec3f) -> vec3f {
    let eps = 0.003;
    return normalize(vec3f(
        get_wood_sdf(p + vec3f(eps, 0.0, 0.0)) - get_wood_sdf(p - vec3f(eps, 0.0, 0.0)),
        get_wood_sdf(p + vec3f(0.0, eps, 0.0)) - get_wood_sdf(p - vec3f(0.0, eps, 0.0)),
        get_wood_sdf(p + vec3f(0.0, 0.0, eps)) - get_wood_sdf(p - vec3f(0.0, 0.0, eps))
    ));
}

// Wall SDF - a vertical wall behind the fire (proper box SDF)
fn get_wall_sdf(p: vec3f) -> f32 {
    // Wall center and half-extents
    let center = vec3f(0.5, 0.65, 0.08);
    let halfSize = vec3f(4.0, 1.2, 0.04);

    // Standard box SDF
    let d = abs(p - center) - halfSize;
    return length(max(d, vec3f(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
}

fn get_wall_normal(p: vec3f) -> vec3f {
    let eps = 0.003;
    return normalize(vec3f(
        get_wall_sdf(p + vec3f(eps, 0.0, 0.0)) - get_wall_sdf(p - vec3f(eps, 0.0, 0.0)),
        get_wall_sdf(p + vec3f(0.0, eps, 0.0)) - get_wall_sdf(p - vec3f(0.0, eps, 0.0)),
        get_wall_sdf(p + vec3f(0.0, 0.0, eps)) - get_wall_sdf(p - vec3f(0.0, 0.0, eps))
    ));
}
`;
};

export const createWoodBurnStateById = (descriptor: WoodPileDescriptor): Record<string, WoodCombustionLogState> => (
  Object.fromEntries(descriptor.logs.map((log) => [log.id, { burnProgress: 0, sootProgress: 0 }]))
);

export const updateWoodCombustionSystem = (
  runtime: WoodCombustionRuntime,
  params: WoodCombustionParams,
  selectedSceneId: number,
  logCharSideColor: THREE.Color,
  logCharCapColor: THREE.Color
) => {
  if (!runtime.logSideMaterial || !runtime.logCapMaterial || !runtime.logBaseSideColor || !runtime.logBaseCapColor) {
    return;
  }

  const nowMs = performance.now();
  const dtSec = Math.min(0.1, Math.max(0.0, (nowMs - runtime.logLastUpdateMs) * 0.001));
  runtime.logLastUpdateMs = nowMs;

  const woodSceneActive = selectedSceneId === 0 || selectedSceneId === 4;
  const thermalDrive = clamp(
    params.emission * 0.072 + params.fuelEfficiency * 0.22 + params.buoyancy * 0.03 + params.burnRate * 0.012,
    0.0,
    2.4
  );
  const sootDrive = clamp(
    (1.0 - params.smokeDissipation) * 52.0 + params.absorption * 0.009 + Math.max(0.0, params.smokeWeight) * 0.03,
    0.0,
    2.0
  );
  const activeGain = woodSceneActive ? 1.0 : 0.16;

  const allStates = Object.entries(runtime.logBurnStateById);
  if (allStates.length === 0) return;

  let burnSum = 0;
  let sootSum = 0;
  for (const [logId, state] of allStates) {
    const variation = 0.92 + (logId.charCodeAt(logId.length - 1) % 5) * 0.03;
    state.burnProgress = clamp(state.burnProgress + dtSec * thermalDrive * 0.18 * activeGain * variation, 0.0, 1.0);
    state.sootProgress = clamp(state.sootProgress + dtSec * sootDrive * 0.15 * activeGain * (2.0 - variation), 0.0, 1.0);
    if (!woodSceneActive) {
      state.burnProgress = Math.max(0.0, state.burnProgress - dtSec * 0.06);
    }
    burnSum += state.burnProgress;
    sootSum += state.sootProgress;
  }

  const averageBurn = burnSum / allStates.length;
  const averageSoot = sootSum / allStates.length;
  const charLevel = clamp(averageBurn * 0.74 + averageSoot * 0.70, 0.0, 1.0);
  const crackLevel = clamp(averageBurn * 1.12 - averageSoot * 0.26, 0.0, 1.0);
  const crackFlicker = 0.86 + 0.14 * Math.sin(nowMs * 0.0065 + params.turbSpeed * 0.8);

  runtime.logSideMaterial.color.copy(runtime.logBaseSideColor).lerp(logCharSideColor, charLevel * 0.9);
  runtime.logSideMaterial.roughness = clamp(0.84 + charLevel * 0.15, 0.45, 1.0);
  runtime.logSideMaterial.emissiveIntensity = clamp(0.02 + crackLevel * 0.34 * crackFlicker, 0.0, 0.62);

  runtime.logCapMaterial.color.copy(runtime.logBaseCapColor).lerp(logCharCapColor, charLevel * 0.82);
  runtime.logCapMaterial.roughness = clamp(0.78 + charLevel * 0.18, 0.35, 1.0);
  runtime.logCapMaterial.emissiveIntensity = clamp(0.015 + crackLevel * 0.22 * crackFlicker, 0.0, 0.4);
};
