// Engine module extracted from FluidSimulation for maintainability and testing.

export interface SimulationParams {
  timeStep: number;
  vorticity: number;
  dissipation: number;
  buoyancy: number;
  drag: number;
  emission: number;
  exposure: number;
  gamma: number;
  scattering: number;
  absorption: number;
  smokeWeight: number;
  plumeTurbulence: number;
  smokeDissipation: number;
  windX: number;
  windZ: number;
  turbFreq: number;
  turbSpeed: number;
  fuelEfficiency: number;
  heatDiffusion: number;
  stepQuality: number;

  // Optional extended combustion + smoke taxonomy + rendering controls
  T_ignite?: number;
  T_burn?: number;
  burnRate?: number;
  fuelInject?: number;
  heatYield?: number;
  sootYieldFlame?: number;
  sootYieldSmolder?: number;
  hazeConvertRate?: number;
  T_hazeStart?: number;
  T_hazeFull?: number;
  anisotropyG?: number;
  smokeThickness?: number;
  smokeDarkness?: number;
  flameSharpness?: number;
  sootDissipation?: number;
}

export interface ScenePreset {
  id: number;
  name: string;
  params: Omit<SimulationParams, 'timeStep' | 'exposure' | 'gamma'> & Partial<Pick<SimulationParams, 'exposure' | 'gamma'>>;
}

export const DEFAULT_TIME_STEP = 0.016;
export const GRID_OPTIONS = [64, 128, 192, 256] as const;

const clampNumber = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const finiteOrDefault = (value: number, fallback: number) => (Number.isFinite(value) ? value : fallback);

export const sanitizeSimParams = (params: SimulationParams): SimulationParams => {
  const safe = {
    ...params,
    timeStep: finiteOrDefault(params.timeStep, DEFAULT_TIME_STEP),
    vorticity: finiteOrDefault(params.vorticity, 0),
    dissipation: finiteOrDefault(params.dissipation, 0.95),
    buoyancy: finiteOrDefault(params.buoyancy, 1),
    drag: finiteOrDefault(params.drag, 0),
    emission: finiteOrDefault(params.emission, 1),
    exposure: finiteOrDefault(params.exposure, 1),
    gamma: finiteOrDefault(params.gamma, 2.2),
    scattering: finiteOrDefault(params.scattering, 1),
    absorption: finiteOrDefault(params.absorption, 1),
    smokeWeight: finiteOrDefault(params.smokeWeight, 0),
    plumeTurbulence: finiteOrDefault(params.plumeTurbulence, 1),
    smokeDissipation: finiteOrDefault(params.smokeDissipation, 0.95),
    windX: finiteOrDefault(params.windX, 0),
    windZ: finiteOrDefault(params.windZ, 0),
    turbFreq: finiteOrDefault(params.turbFreq, 28),
    turbSpeed: finiteOrDefault(params.turbSpeed, 1),
    fuelEfficiency: finiteOrDefault(params.fuelEfficiency, 1),
    heatDiffusion: finiteOrDefault(params.heatDiffusion, 0),
    stepQuality: finiteOrDefault(params.stepQuality, 1),
  };

  return {
    ...safe,
    timeStep: clampNumber(safe.timeStep, 0.001, 0.05),
    vorticity: clampNumber(safe.vorticity, 0, 60),
    dissipation: clampNumber(safe.dissipation, 0, 1),
    buoyancy: clampNumber(safe.buoyancy, 0, 40),
    drag: clampNumber(safe.drag, 0, 0.2),
    emission: clampNumber(safe.emission, 0, 25),
    exposure: clampNumber(safe.exposure, 0.1, 10),
    gamma: clampNumber(safe.gamma, 0.1, 4),
    scattering: clampNumber(safe.scattering, 0, 25),
    absorption: clampNumber(safe.absorption, 0, 100),
    smokeWeight: clampNumber(safe.smokeWeight, -5, 15),
    plumeTurbulence: clampNumber(safe.plumeTurbulence, 0, 20),
    smokeDissipation: clampNumber(safe.smokeDissipation, 0, 0.999),
    windX: clampNumber(safe.windX, -0.5, 0.5),
    windZ: clampNumber(safe.windZ, -0.5, 0.5),
    turbFreq: clampNumber(safe.turbFreq, 1, 100),
    turbSpeed: clampNumber(safe.turbSpeed, 0, 10),
    fuelEfficiency: clampNumber(safe.fuelEfficiency, 0.1, 10),
    heatDiffusion: clampNumber(safe.heatDiffusion, 0, 1),
    stepQuality: clampNumber(safe.stepQuality, 0.25, 4),
  };
};

export const getBufferFootprintBytes = (dim: number) => {
  const voxelCount = dim * dim * dim;
  return {
    densityBytes: voxelCount * 4,
    fuelBytes: voxelCount * 4,
    velocityBytes: voxelCount * 16,
  };
};

export const getSupportedGridSizes = (adapter: GPUAdapter): number[] => {
  const maxStorage = Number(adapter?.limits?.maxStorageBufferBindingSize ?? 0);
  const maxBuffer = Number(adapter?.limits?.maxBufferSize ?? maxStorage);
  const maxWorkgroups = Number(adapter?.limits?.maxComputeWorkgroupsPerDimension ?? Number.MAX_SAFE_INTEGER);

  return GRID_OPTIONS.filter((dim) => {
    const { densityBytes, fuelBytes, velocityBytes } = getBufferFootprintBytes(dim);
    const dispatchSize = Math.ceil(dim / 4);
    return (
      densityBytes <= maxStorage &&
      fuelBytes <= maxStorage &&
      velocityBytes <= maxStorage &&
      densityBytes <= maxBuffer &&
      fuelBytes <= maxBuffer &&
      velocityBytes <= maxBuffer &&
      dispatchSize <= maxWorkgroups
    );
  });
};

/**
 * SECTION 1: TRANSPORT LAYER (3D)
 */

class ShaderContract {
  public layout: GPUPipelineLayout;
  public bindGroupLayout: GPUBindGroupLayout;

  constructor(device: GPUDevice, label: string, entries: GPUBindGroupLayoutEntry[]) {
    this.bindGroupLayout = device.createBindGroupLayout({
      entries,
      label: `${label}_BGL`
    });

    this.layout = device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout],
      label: `${label}_PL`
    });
  }

  createBindGroup(device: GPUDevice, label: string, entries: GPUBindGroupEntry[]): GPUBindGroup {
    return device.createBindGroup({
      layout: this.bindGroupLayout,
      entries,
      label: `${label}_BG`
    });
  }
}

export class FluidTransport {
  public uniformBuffer: GPUBuffer;

  public densityA: GPUBuffer;
  public densityB: GPUBuffer;
  public fuelA: GPUBuffer;
  public fuelB: GPUBuffer;
  public velocityA: GPUBuffer;
  public velocityB: GPUBuffer;

  public physicsContract!: ShaderContract;
  public renderContract!: ShaderContract;

  public physicsGroups: GPUBindGroup[] = [];
  public renderGroups: GPUBindGroup[] = [];

  constructor(private device: GPUDevice, public dim: number) {
    const VOXEL_COUNT = dim * dim * dim;

    this.uniformBuffer = device.createBuffer({
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    const storageUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

    this.densityA = device.createBuffer({ size: VOXEL_COUNT * 4, usage: storageUsage });
    this.densityB = device.createBuffer({ size: VOXEL_COUNT * 4, usage: storageUsage });

    this.fuelA = device.createBuffer({ size: VOXEL_COUNT * 4, usage: storageUsage });
    this.fuelB = device.createBuffer({ size: VOXEL_COUNT * 4, usage: storageUsage });

    this.velocityA = device.createBuffer({ size: VOXEL_COUNT * 16, usage: storageUsage });
    this.velocityB = device.createBuffer({ size: VOXEL_COUNT * 16, usage: storageUsage });

    const zeroF32 = new Float32Array(VOXEL_COUNT);
    const zeroVec4 = new Float32Array(VOXEL_COUNT * 4);

    device.queue.writeBuffer(this.densityA, 0, zeroF32);
    device.queue.writeBuffer(this.densityB, 0, zeroF32);
    device.queue.writeBuffer(this.fuelA, 0, zeroF32);
    device.queue.writeBuffer(this.fuelB, 0, zeroF32);
    device.queue.writeBuffer(this.velocityA, 0, zeroVec4);
    device.queue.writeBuffer(this.velocityB, 0, zeroVec4);

    this.initContracts();
    this.initBindGroups();
  }

  private initContracts() {
    this.physicsContract = new ShaderContract(this.device, 'Physics', [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ]);

    this.renderContract = new ShaderContract(this.device, 'Render', [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    ]);
  }

  private initBindGroups() {
    this.physicsGroups[0] = this.physicsContract.createBindGroup(this.device, 'Phys0', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.densityA } },
      { binding: 2, resource: { buffer: this.densityB } },
      { binding: 3, resource: { buffer: this.velocityA } },
      { binding: 4, resource: { buffer: this.velocityB } },
      { binding: 5, resource: { buffer: this.fuelA } },
      { binding: 6, resource: { buffer: this.fuelB } },
    ]);

    this.renderGroups[0] = this.renderContract.createBindGroup(this.device, 'Render0', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.densityB } },
      { binding: 2, resource: { buffer: this.velocityB } },
      { binding: 3, resource: { buffer: this.fuelB } },
    ]);

    this.physicsGroups[1] = this.physicsContract.createBindGroup(this.device, 'Phys1', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.densityB } },
      { binding: 2, resource: { buffer: this.densityA } },
      { binding: 3, resource: { buffer: this.velocityB } },
      { binding: 4, resource: { buffer: this.velocityA } },
      { binding: 5, resource: { buffer: this.fuelB } },
      { binding: 6, resource: { buffer: this.fuelA } },
    ]);

    this.renderGroups[1] = this.renderContract.createBindGroup(this.device, 'Render1', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.densityA } },
      { binding: 2, resource: { buffer: this.velocityA } },
      { binding: 3, resource: { buffer: this.fuelA } },
    ]);
  }

  public updateUniforms(
    now: number,
    params: SimulationParams,
    camera: { pos: [number, number, number], target: [number, number, number] },
    sceneType: number
  ) {
    const safeParams = sanitizeSimParams(params);
    const uniformData = new ArrayBuffer(256);
    const view = new DataView(uniformData);

    view.setFloat32(0, this.dim, true);
    view.setFloat32(4, now / 1000.0, true);
    view.setFloat32(8, safeParams.timeStep, true);
    view.setFloat32(12, safeParams.vorticity, true);

    view.setFloat32(16, safeParams.dissipation, true);
    view.setFloat32(20, safeParams.buoyancy, true);
    view.setFloat32(24, safeParams.drag, true);
    view.setFloat32(28, safeParams.emission, true);

    view.setFloat32(32, safeParams.exposure, true);
    view.setFloat32(36, safeParams.gamma, true);
    view.setFloat32(40, sceneType, true);
    view.setFloat32(44, safeParams.scattering, true);

    view.setFloat32(48, safeParams.absorption, true);
    view.setFloat32(52, safeParams.smokeWeight, true);
    view.setFloat32(56, safeParams.plumeTurbulence, true);
    view.setFloat32(60, safeParams.smokeDissipation, true);

    view.setFloat32(64, camera.pos[0], true);
    view.setFloat32(68, camera.pos[1], true);
    view.setFloat32(72, camera.pos[2], true);
    view.setFloat32(76, 0, true); // pad2

    view.setFloat32(80, camera.target[0], true);
    view.setFloat32(84, camera.target[1], true);
    view.setFloat32(88, camera.target[2], true);
    view.setFloat32(92, 0, true); // pad3

    view.setFloat32(96, safeParams.windX, true);
    view.setFloat32(100, safeParams.windZ, true);
    view.setFloat32(104, safeParams.turbFreq, true);
    view.setFloat32(108, safeParams.turbSpeed, true);
    view.setFloat32(112, safeParams.fuelEfficiency, true);
    view.setFloat32(116, safeParams.heatDiffusion, true);
    view.setFloat32(120, safeParams.stepQuality, true);
    view.setFloat32(124, 0.0, true); // pad4

    const heightFactor = clampNumber(safeParams.buoyancy / 8.0, 0.5, 5.0);
    const baseBurnRate = (safeParams as any).burnRate ?? safeParams.fuelEfficiency * 6.0;
    const baseFuelInject = (safeParams as any).fuelInject ?? (0.4 + safeParams.fuelEfficiency * 0.6);
    const derivedBurnRate = baseBurnRate / heightFactor;
    const derivedFuelInject = baseFuelInject * heightFactor;
    const derivedVolumeHeight = (safeParams as any).volumeHeight ?? 1.0;
    view.setFloat32(128, (safeParams as any).T_ignite ?? 0.18, true);
    view.setFloat32(132, (safeParams as any).T_burn ?? 0.55, true);
    view.setFloat32(136, derivedBurnRate, true);
    view.setFloat32(140, derivedFuelInject, true);

    view.setFloat32(144, (safeParams as any).heatYield ?? 3.4, true);
    view.setFloat32(148, (safeParams as any).sootYieldFlame ?? 0.22, true);
    view.setFloat32(152, (safeParams as any).sootYieldSmolder ?? 0.55, true);
    view.setFloat32(156, (safeParams as any).hazeConvertRate ?? 0.0, true);

    view.setFloat32(160, (safeParams as any).T_hazeStart ?? 0.35, true);
    view.setFloat32(164, (safeParams as any).T_hazeFull ?? 0.75, true);
    view.setFloat32(168, (safeParams as any).anisotropyG ?? 0.82, true);
    view.setFloat32(172, (safeParams as any).smokeThickness ?? 1.0, true);

    view.setFloat32(176, (safeParams as any).smokeDarkness ?? 0.65, true);
    view.setFloat32(180, (safeParams as any).flameSharpness ?? 4.0, true);
    view.setFloat32(184, (safeParams as any).sootDissipation ?? safeParams.smokeDissipation ?? 0.985, true);
    view.setFloat32(188, derivedVolumeHeight, true);

    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);
  }
}

/**
 * SECTION 2: THE WGSL SHADERS
 */

const STRUCT_DEF = `
alias vec3i = vec3<i32>;

struct SimParams {
  dim: f32,
  time: f32,
  dt: f32,
  vorticity: f32,

  dissipation: f32,
  buoyancy: f32,
  drag: f32,
  emission: f32,

  exposure: f32,
  gamma: f32,
  sceneType: f32,
  scattering: f32,

  absorption: f32,
  smokeWeight: f32,
  plumeTurbulence: f32,
  smokeDissipation: f32,

  cameraPos: vec3f,
  pad2: f32,

  targetPos: vec3f,
  pad3: f32,

  windX: f32,
  windZ: f32,
  turbFreq: f32,
  turbSpeed: f32,
  fuelEfficiency: f32,
  heatDiffusion: f32,
  stepQuality: f32,
  pad4: f32,

  // Extended combustion + smoke taxonomy + rendering controls
  T_ignite: f32,
  T_burn: f32,
  burnRate: f32,
  fuelInject: f32,

  heatYield: f32,
  sootYieldFlame: f32,
  sootYieldSmolder: f32,
  hazeConvertRate: f32,

  T_hazeStart: f32,
  T_hazeFull: f32,
  anisotropyG: f32,
  smokeThickness: f32,

  smokeDarkness: f32,
  flameSharpness: f32,
  sootDissipation: f32,
  volumeHeight: f32,
};
`;

const WOOD_SDF_FN = `
fn get_wood_sdf(p: vec3f) -> f32 {
     let c = vec3f(0.5, 0.08, 0.5);

     // Log 1: Base X-axis
     let p1 = p - c;
     let ang1 = 0.15;
     let r1x = p1.x * cos(ang1) - p1.y * sin(ang1);
     let r1y = p1.x * sin(ang1) + p1.y * cos(ang1);
     let d1 = max(length(vec2f(r1y, p1.z)) - 0.035, abs(r1x) - 0.18);

     // Log 2: Base Z-axis
     let p2 = p - (c + vec3f(0.0, 0.03, 0.0));
     let ang2 = -0.15;
     let r2z = p2.z * cos(ang2) - p2.y * sin(ang2);
     let r2y = p2.z * sin(ang2) + p2.y * cos(ang2);
     let d2 = max(length(vec2f(p2.x, r2y)) - 0.035, abs(r2z) - 0.18);

     // Log 3: Diagonal across the top
     let p3 = p - (c + vec3f(0.0, 0.06, 0.0));
     let ang3 = 0.785;
     let c3 = cos(ang3); let s3 = sin(ang3);
     let r3x = p3.x * c3 - p3.z * s3;
     let r3z = p3.x * s3 + p3.z * c3;
     let r3y = r3z * sin(0.1) + p3.y * cos(0.1);
     let d3 = max(length(vec2f(r3x, r3y)) - 0.032, abs(r3z) - 0.2);

     return min(d1, min(d2, d3));
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
    let center = vec3f(0.5, 0.4, 0.1);
    let halfSize = vec3f(0.8, 0.4, 0.02);

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

export const COMPUTE_SHADER = `
${STRUCT_DEF}
${WOOD_SDF_FN}

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> densityIn: array<f32>;
@group(0) @binding(2) var<storage, read_write> densityOut: array<f32>;
@group(0) @binding(3) var<storage, read> velocityIn: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> velocityOut: array<vec4f>;
@group(0) @binding(5) var<storage, read> fuelIn: array<f32>;
@group(0) @binding(6) var<storage, read_write> fuelOut: array<f32>;

fn get_idx(p: vec3i) -> u32 {
  let d = i32(params.dim);
  let cp = clamp(p, vec3i(0), vec3i(d - 1));
  return u32(cp.z * d * d + cp.y * d + cp.x);
}

fn inside_volume_world(p: vec3f) -> bool {
  return p.x >= 0.0 && p.x <= 1.0 && p.z >= 0.0 && p.z <= 1.0 && p.y >= 0.0 && p.y <= params.volumeHeight;
}

fn to_volume_uv(p: vec3f) -> vec3f {
  let h = max(0.0001, params.volumeHeight);
  return vec3f(p.x, clamp(p.y / h, 0.0, 1.0), p.z);
}

fn safe_normalize(v: vec3f) -> vec3f {
    let len = length(v);
    if (len < 1e-5) { return vec3f(0.0); }
    return v / len;
}

fn hash(p: vec3f) -> vec3f {
    var p3 = fract(p * vec3f(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yxx) * p3.zyx) * 2.0 - 1.0;
}

fn noise_fbm(p: vec3f) -> f32 {
    var res = 0.0;
    var amp = 0.5;
    var f = p;
    for(var i=0; i<4; i++) {
        res += dot(hash(f), vec3f(1.0)) * amp;
        f *= 2.15;
        amp *= 0.5;
    }
    return res;
}

fn voronoi_cracks(p: vec3f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    var min_dist = 1.0;
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let neighbor = vec3f(f32(x), f32(y), f32(z));
                let pt = hash(i + neighbor) * 0.5 + 0.5;
                let diff = neighbor + pt - f;
                let dist = length(diff);
                min_dist = min(min_dist, dist);
            }
        }
    }
    return min_dist;
}

struct State {
  vel: vec3f,
  soot: f32,
  temp: f32,
  fuel: f32,
};

fn sample_state(pos: vec3f) -> State {
  let d = params.dim;
  let st = pos * d - 0.5;
  let i = floor(st);
  let f = fract(st);
  let u = f * f * (3.0 - 2.0 * f);
  let i0 = vec3i(i);
  let i1 = i0 + vec3i(1);
  let v000 = velocityIn[get_idx(vec3i(i0.x, i0.y, i0.z))];
  let v100 = velocityIn[get_idx(vec3i(i1.x, i0.y, i0.z))];
  let v010 = velocityIn[get_idx(vec3i(i0.x, i1.y, i0.z))];
  let v110 = velocityIn[get_idx(vec3i(i1.x, i1.y, i0.z))];
  let v001 = velocityIn[get_idx(vec3i(i0.x, i0.y, i1.z))];
  let v101 = velocityIn[get_idx(vec3i(i1.x, i0.y, i1.z))];
  let v011 = velocityIn[get_idx(vec3i(i0.x, i1.y, i1.z))];
  let v111 = velocityIn[get_idx(vec3i(i1.x, i1.y, i1.z))];
  let vm = mix(mix(mix(v000, v100, u.x), mix(v010, v110, u.x), u.y), mix(mix(v001, v101, u.x), mix(v011, v111, u.x), u.y), u.z);
  let f000 = densityIn[get_idx(vec3i(i0.x, i0.y, i0.z))];
  let f100 = densityIn[get_idx(vec3i(i1.x, i0.y, i0.z))];
  let f010 = densityIn[get_idx(vec3i(i0.x, i1.y, i0.z))];
  let f110 = densityIn[get_idx(vec3i(i1.x, i1.y, i0.z))];
  let f001 = densityIn[get_idx(vec3i(i0.x, i0.y, i1.z))];
  let f101 = densityIn[get_idx(vec3i(i1.x, i0.y, i1.z))];
  let f011 = densityIn[get_idx(vec3i(i0.x, i1.y, i1.z))];
  let f111 = densityIn[get_idx(vec3i(i1.x, i1.y, i1.z))];
  let fm = mix(mix(mix(f000, f100, u.x), mix(f010, f110, u.x), u.y), mix(mix(f001, f101, u.x), mix(f011, f111, u.x), u.y), u.z);

  let fu000 = fuelIn[get_idx(vec3i(i0.x, i0.y, i0.z))];
  let fu100 = fuelIn[get_idx(vec3i(i1.x, i0.y, i0.z))];
  let fu010 = fuelIn[get_idx(vec3i(i0.x, i1.y, i0.z))];
  let fu110 = fuelIn[get_idx(vec3i(i1.x, i1.y, i0.z))];
  let fu001 = fuelIn[get_idx(vec3i(i0.x, i0.y, i1.z))];
  let fu101 = fuelIn[get_idx(vec3i(i1.x, i0.y, i1.z))];
  let fu011 = fuelIn[get_idx(vec3i(i0.x, i1.y, i1.z))];
  let fu111 = fuelIn[get_idx(vec3i(i1.x, i1.y, i1.z))];
  let fum = mix(mix(mix(fu000, fu100, u.x), mix(fu010, fu110, u.x), u.y), mix(mix(fu001, fu101, u.x), mix(fu011, fu111, u.x), u.y), u.z);

  return State(vm.xyz, vm.w, fm, fum);
}

fn curl(p: vec3i) -> vec3f {
  let L = velocityIn[get_idx(p + vec3i(-1,0,0))];
  let R = velocityIn[get_idx(p + vec3i(1,0,0))];
  let D = velocityIn[get_idx(p + vec3i(0,-1,0))];
  let U = velocityIn[get_idx(p + vec3i(0,1,0))];
  let B = velocityIn[get_idx(p + vec3i(0,0,-1))];
  let F = velocityIn[get_idx(p + vec3i(0,0,1))];
  let dvz_dy = (U.z - D.z);
  let dvy_dz = (F.y - B.y);
  let dvx_dz = (F.x - B.x);
  let dvz_dx = (R.z - L.z);
  let dvy_dx = (R.y - L.y);
  let dvx_dy = (U.x - D.x);
  return 0.5 * vec3f(dvz_dy - dvy_dz, dvx_dz - dvz_dx, dvy_dx - dvx_dy);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dim = u32(params.dim);
  if (id.x >= dim || id.y >= dim || id.z >= dim) { return; }
  let idx = get_idx(vec3i(id));
  let dt = params.dt;
  let uvw = (vec3f(id) + 0.5) / params.dim;
  let scene = params.sceneType;

  var woodDist = 1.0;
  let usesWood = (scene == 0.0 || scene == 4.0);
  if (usesWood) { woodDist = get_wood_sdf(uvw); }

  let vel = velocityIn[idx].xyz;
  // High buoyancy can produce very large velocities, which makes the semi-Lagrangian
  // backtrace jump out of bounds and collapse the flame (samples clamp to borders).
  // Reduce the backtrace distance as buoyancy rises to keep advection stable.
  let advectGain = 12.0 * clamp(8.0 / (8.0 + params.buoyancy), 0.25, 1.0);
  let backPos = uvw - vel * dt * (1.3 / params.dim) * advectGain;
  var state = sample_state(backPos);

  var newVel = state.vel;
  var temp = state.temp;
  var soot = state.soot;
  var fuel = state.fuel;

  // External forces: Wind
  newVel += vec3f(params.windX, 0.0, params.windZ) * dt * 40.0;

  // Heat Diffusion (Simple kernel approximation)
  if (params.heatDiffusion > 0.0) {
      let T_avg = (densityIn[get_idx(vec3i(id) + vec3i(1,0,0))] + densityIn[get_idx(vec3i(id) - vec3i(1,0,0))] +
                   densityIn[get_idx(vec3i(id) + vec3i(0,1,0))] + densityIn[get_idx(vec3i(id) - vec3i(0,1,0))] +
                   densityIn[get_idx(vec3i(id) + vec3i(0,0,1))] + densityIn[get_idx(vec3i(id) - vec3i(0,0,1))]) / 6.0;
      temp = mix(temp, T_avg, params.heatDiffusion * dt * 10.0);
  }

  let localOxygen = smoothstep(1.2, 0.2, soot);
  let insulation_factor = smoothstep(0.12, 0.0, woodDist);
  let insulation = mix(params.dissipation, 1.0, insulation_factor * 0.88);
  temp *= insulation;

  soot *= params.sootDissipation;

  let buoyancyDir = vec3f(0.0, 1.0, 0.0);
  let thermalLift = max(0.0, (pow(temp, 1.25) * 0.4) - (soot * params.smokeWeight * 0.0018));
  newVel += buoyancyDir * thermalLift * params.buoyancy * dt * 20.0;
  newVel *= max(0.0, 1.0 - params.drag);

  if (temp > 0.01) {
      let noise_scale = params.turbFreq + sin(params.time * 1.1) * 6.0;
      let turb = hash(uvw * noise_scale + params.time * 0.35 * params.turbSpeed) * temp * params.plumeTurbulence * 0.7;
      newVel += turb;
  }

  let omega = curl(vec3i(id));
  let oR = length(curl(vec3i(id) + vec3i(1,0,0)));
  let oL = length(curl(vec3i(id) + vec3i(-1,0,0)));
  let oU = length(curl(vec3i(id) + vec3i(0,1,0)));
  let oD = length(curl(vec3i(id) + vec3i(0,-1,0)));
  let oF = length(curl(vec3i(id) + vec3i(0,0,1)));
  let oB = length(curl(vec3i(id) + vec3i(0,0,-1)));
  let eta = vec3f(oR - oL, oU - oD, oF - oB);
  if (length(eta) > 1e-6) {
    newVel += cross(safe_normalize(eta), omega) * params.vorticity * dt * 12.0;
  }

  if (usesWood) {
     let logSurfaceZone = smoothstep(0.05, -0.02, woodDist);
     if (logSurfaceZone > 0.0) {
         let crack_field = voronoi_cracks(uvw * 22.0 + params.time * 0.1);
         let crack_mask = smoothstep(0.0, 0.2, crack_field);
         let n_val = noise_fbm(uvw * 15.0 + params.time * 0.5);
         let inject = params.fuelInject * logSurfaceZone * (0.55 + 0.45 * crack_mask) * dt;
         fuel = clamp(fuel + inject, 0.0, 1.0);

         let fuelAvailability = logSurfaceZone * fuel;
         let combustionShape = smoothstep(-0.2, 0.8, n_val);

         // Pilot heat: lets ignition bootstrap from a cold start.
         temp += params.emission * combustionShape * logSurfaceZone * dt * 0.22;
         let ignite = smoothstep(params.T_ignite, params.T_burn, temp);
         let combustionIntensity = combustionShape * fuelAvailability * localOxygen * ignite;

         let R = combustionIntensity * params.burnRate;
         fuel = max(0.0, fuel - R * dt);
         temp += R * params.heatYield * dt;

         let hot = smoothstep(params.T_ignite, params.T_burn, temp);
         let smolder = 1.0 - localOxygen;
         let sootYield = mix(params.sootYieldSmolder, params.sootYieldFlame, hot) * (0.4 + 0.6 * smolder);
         soot += R * sootYield * dt;

         let intensity = R * (0.55 + 0.45 * crack_mask);
         let logNorm = get_wood_normal(uvw);
         newVel += normalize(logNorm * 0.35 + vec3f(0.0, 2.5, 0.0)) * intensity * 160.0 * dt;
     }
  } else {
     let d_emit = length(uvw - vec3f(0.5, 0.2, 0.5));
     if (d_emit < 0.05) {
        let inject = params.fuelInject * dt * 2.0;
        fuel = clamp(fuel + inject, 0.0, 1.0);

        // Pilot heat for point-source scenes.
        temp += params.emission * dt * 2.0;
        let ignite = smoothstep(params.T_ignite, params.T_burn, temp);
        let R = ignite * localOxygen * fuel * params.burnRate * 0.45;
        fuel = max(0.0, fuel - R * dt);
        temp += R * params.heatYield * dt;
        soot += R * params.sootYieldFlame * dt;
        newVel += vec3f(0.0, 0.8, 0.0) * R * dt;
     }
  }

  if (usesWood && woodDist < 0.0) {
      let friction = smoothstep(0.0, 0.015, -woodDist);
      newVel *= (1.0 - friction * 0.2);
      temp *= (1.0 - friction * 0.015);
  }

  // Boundary damping: keep floor and side walls stable, but avoid a hard “ceiling”
  // that artificially chops the plume height.
  let b_dist_xz = min(min(uvw.x, 1.0 - uvw.x), min(uvw.z, 1.0 - uvw.z));
  let b_dist = min(b_dist_xz, uvw.y);
  let edge_damp = smoothstep(0.0, 0.02, b_dist);
  temp *= edge_damp; soot *= edge_damp; fuel *= edge_damp; newVel *= edge_damp;
  temp = max(temp, 0.0);
  soot = max(soot, 0.0);
  fuel = max(fuel, 0.0);

  densityOut[idx] = temp;
  velocityOut[idx] = vec4f(clamp(newVel, vec3f(-120.0), vec3f(120.0)), soot);
  fuelOut[idx] = fuel;
}
`;

export const RENDER_SHADER = `
${STRUCT_DEF}
${WOOD_SDF_FN}
@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> densityIn: array<f32>;
@group(0) @binding(2) var<storage, read> velocityIn: array<vec4f>;
@group(0) @binding(3) var<storage, read> fuelIn: array<f32>;

struct VertexOutput { @builtin(position) Position : vec4f, @location(0) uv : vec2f };

@vertex fn vert_main(@builtin(vertex_index) VertexIndex : u32) -> VertexOutput {
  var output : VertexOutput;
  var pos = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  output.Position = vec4f(pos[VertexIndex], 0.0, 1.0); output.uv = pos[VertexIndex] * 0.5 + 0.5; return output;
}

fn get_idx(p: vec3i) -> u32 {
  let d = i32(params.dim);
  let cp = clamp(p, vec3i(0), vec3i(d - 1));
  return u32(cp.z * d * d + cp.y * d + cp.x);
}

fn inside_volume_world(p: vec3f) -> bool {
  return p.x >= 0.0 && p.x <= 1.0 && p.z >= 0.0 && p.z <= 1.0 && p.y >= 0.0 && p.y <= params.volumeHeight;
}

fn to_volume_uv(p: vec3f) -> vec3f {
  let h = max(0.0001, params.volumeHeight);
  return vec3f(p.x, clamp(p.y / h, 0.0, 1.0), p.z);
}

fn sample_volume(pos: vec3f) -> vec3f {
  let d = params.dim; let p = pos * d - 0.5; let i = vec3i(floor(p)); let f = fract(p);
  let f_res = mix(mix(mix(densityIn[get_idx(i)], densityIn[get_idx(i + vec3i(1,0,0))], f.x), mix(densityIn[get_idx(i + vec3i(0,1,0))], densityIn[get_idx(i + vec3i(1,1,0))], f.x), f.y), mix(mix(densityIn[get_idx(i + vec3i(0,0,1))], densityIn[get_idx(i + vec3i(1,0,1))], f.x), mix(densityIn[get_idx(i + vec3i(0,1,1))], densityIn[get_idx(i + vec3i(1,1,1))], f.x), f.y), f.z);
  let s_res = mix(mix(mix(velocityIn[get_idx(i)].w, velocityIn[get_idx(i + vec3i(1,0,0))].w, f.x), mix(velocityIn[get_idx(i + vec3i(0,1,0))].w, velocityIn[get_idx(i + vec3i(1,1,0))].w, f.x), f.y), mix(mix(velocityIn[get_idx(i + vec3i(0,0,1))].w, velocityIn[get_idx(i + vec3i(1,0,1))].w, f.x), mix(velocityIn[get_idx(i + vec3i(0,1,1))].w, velocityIn[get_idx(i + vec3i(1,1,1))].w, f.x), f.y), f.z);
  let fu_res = mix(mix(mix(fuelIn[get_idx(i)], fuelIn[get_idx(i + vec3i(1,0,0))], f.x), mix(fuelIn[get_idx(i + vec3i(0,1,0))], fuelIn[get_idx(i + vec3i(1,1,0))], f.x), f.y), mix(mix(fuelIn[get_idx(i + vec3i(0,0,1))], fuelIn[get_idx(i + vec3i(1,0,1))], f.x), mix(fuelIn[get_idx(i + vec3i(0,1,1))], fuelIn[get_idx(i + vec3i(1,1,1))], f.x), f.y), f.z);
  return vec3f(f_res, s_res, fu_res);
}

fn compute_reaction(pos: vec3f) -> f32 {
  // Flame lives on thin temperature edges, not in the hot volume.
  // reaction = smoothstep(T_ignite, T_hot, T) * smoothstep(g0, g1, |∇T|)
  let temp = sample_volume(pos).x;
  let eps = 1.0 / params.dim;
  let txp = sample_volume(pos + vec3f(eps, 0.0, 0.0)).x;
  let txn = sample_volume(pos - vec3f(eps, 0.0, 0.0)).x;
  let typ = sample_volume(pos + vec3f(0.0, eps, 0.0)).x;
  let tyn = sample_volume(pos - vec3f(0.0, eps, 0.0)).x;
  let tzp = sample_volume(pos + vec3f(0.0, 0.0, eps)).x;
  let tzn = sample_volume(pos - vec3f(0.0, 0.0, eps)).x;
  let grad = vec3f(txp - txn, typ - tyn, tzp - tzn);
  let gradMag = length(grad) * 0.5;

  let tempGate = smoothstep(params.T_ignite, params.T_burn, temp);

  // Sharpen the reaction front so emission collapses into sheets/tongues.
  let front = smoothstep(0.035, 0.055, gradMag);
  let r = tempGate * front;
  let sharp = max(1.0, params.flameSharpness);
  return pow(clamp(r, 0.0, 1.0), sharp);
}

fn tonemap_aces(color: vec3f) -> vec3f {
  // ACES approximation (Narkowicz 2015). Keeps HDR highlights structured.
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3f(0.0), vec3f(1.0));
}

fn phase_function(costheta: f32, g: f32) -> f32 {
    let g2 = g * g; return (1.0 - g2) / (4.0 * 3.14159 * pow(1.0 + g2 - 2.0 * g * costheta, 1.5));
}

fn getBlackbodyColor(temp: f32) -> vec3f {
   let t = max(0.0, temp - 0.12);
   if (t < 0.3) { return mix(vec3f(0.0), vec3f(2.2, 0.05, 0.002), t / 0.3); }
   else if (t < 0.75) { return mix(vec3f(2.2, 0.05, 0.002), vec3f(5.5, 1.8, 0.1), (t - 0.3) / 0.45); }
   else if (t < 1.4) { return mix(vec3f(5.5, 1.8, 0.1), vec3f(12.0, 8.0, 1.2), (t - 0.75) / 0.65); }
   else { return mix(vec3f(12.0, 8.0, 1.2), vec3f(35.0, 35.0, 35.0), clamp((t - 1.4) * 0.4, 0.0, 1.0)); }
}

fn get_light_transmittance(pos: vec3f, lightDir: vec3f) -> f32 {
  var p = pos;
  let step = 0.045;
  var tau = 0.0;
    for(var i=0; i<12; i++) {
        p += lightDir * step;
        if (!inside_volume_world(p)) { break; }
    let uv = to_volume_uv(p);
    let val = sample_volume(uv);
    let soot = val.y;
    let temp = val.x;
    // Smoke taxonomy (minimum viable realism):
    // - soot: dark, absorption-dominant, reduced near hottest flame but not removed
    // - haze: scattering-dominant, appears as soot cools (mid/far plume)
    let hot = smoothstep(0.35, 0.7, temp);
    let cool = smoothstep(params.T_hazeStart, params.T_hazeFull, 1.0 - temp);
    let height = smoothstep(0.22, 0.92, uv.y);

    let sootVis = mix(1.0, 0.25, hot);
    let sootRaw = soot * sootVis;
    let hazeRaw = soot * cool * height;

    let sootOpt = 1.0 - exp(-sootRaw * 0.10);
    let hazeOpt = 1.0 - exp(-hazeRaw * 0.04);

    let thickness = max(0.0, params.smokeThickness);
    let darkness = clamp(params.smokeDarkness, 0.0, 1.0);
    let absorption = params.absorption * thickness * (0.65 + 0.7 * darkness);
    let scattering = params.scattering * thickness * (0.85 - 0.55 * darkness);

    let sigmaA = sootOpt * absorption * 1.10 + hazeOpt * absorption * 0.12;
    let sigmaS = sootOpt * scattering * 0.12 + hazeOpt * scattering * 0.55;
    tau += (sigmaA + sigmaS) * step;
    }
  return exp(-tau);
}

// HEMISPHERE GI - Sample upward in multiple directions to find actual fire
fn get_volume_lighting(pos: vec3f) -> vec3f {
    var totalLight = vec3f(0.0);

    // Sample directions in a hemisphere above the floor point
    // This finds light from wherever the fire actually IS
    let dirs = array<vec3f, 6>(
        vec3f(0.0, 1.0, 0.0),     // Straight up
        vec3f(0.4, 0.9, 0.0),     // Forward-up
        vec3f(-0.4, 0.9, 0.0),    // Back-up
        vec3f(0.0, 0.9, 0.4),     // Right-up
        vec3f(0.0, 0.9, -0.4),    // Left-up
        vec3f(0.3, 0.95, 0.3)     // Diagonal
    );

    for (var s = 0; s < 6; s++) {
        let dir = normalize(dirs[s]);

        // March upward from floor, find any fire along this direction
        var t = 0.02;
        var transmittance = 1.0;

        for (var i = 0; i < 12; i++) {
            let p = pos + dir * t;

            // Stop if outside volume
          if (!inside_volume_world(p)) { break; }
            if (transmittance < 0.02) { break; }

          let uv = to_volume_uv(p);
          let val = sample_volume(uv);

            let fuel = val.z;
            var reaction = compute_reaction(uv);
            let fuelGate = 0.25 + 0.75 * smoothstep(0.0, 0.10, fuel);
            reaction *= fuelGate;

            // Fire emits light downward to floor
            if (reaction > 0.01) {
              let emission = getBlackbodyColor(val.x) * (params.emission * 0.12) * reaction;
                let atten = 1.0 / (1.0 + t * t * 5.0);
              totalLight += emission * atten * transmittance * 0.12;
            }

            // Smoke blocks light
            let soot = val.y;
            let temp = val.x;
            let hot = smoothstep(0.35, 0.7, temp);
            let cool = smoothstep(params.T_hazeStart, params.T_hazeFull, 1.0 - temp);
            let height = smoothstep(0.22, 0.92, uv.y);
            let sootVis = mix(1.0, 0.25, hot);
            let sootRaw = soot * sootVis;
            let hazeRaw = soot * cool * height;
            let sootOpt = 1.0 - exp(-sootRaw * 0.10);
            let hazeOpt = 1.0 - exp(-hazeRaw * 0.04);
            let thickness = max(0.0, params.smokeThickness);
            let darkness = clamp(params.smokeDarkness, 0.0, 1.0);
            let absorption = params.absorption * thickness * (0.65 + 0.7 * darkness);
            let scattering = params.scattering * thickness * (0.85 - 0.55 * darkness);
            let sigmaA = sootOpt * absorption * 1.10 + hazeOpt * absorption * 0.12;
            let sigmaS = sootOpt * scattering * 0.12 + hazeOpt * scattering * 0.55;
            transmittance *= exp(-(sigmaA + sigmaS) * 0.06 * 0.3);

            t += 0.06;
        }
    }

    return totalLight * 0.25;
}

fn intersectAABB(ro: vec3f, rd: vec3f, bmin: vec3f, bmax: vec3f) -> vec2f {
    let tMin = (bmin - ro) / rd; let tMax = (bmax - ro) / rd;
    let t1 = min(tMin, tMax); let t2 = max(tMin, tMax);
    return vec2f(max(max(t1.x, t1.y), t1.z), min(min(t2.x, t2.y), t2.z));
}

fn get_floor_material(p: vec3f) -> vec3f {
    // Grid pattern
    let grid_size = 0.15;
    let grid_line = 0.003;
    let check = fract(p.xz / grid_size);
    let line = smoothstep(grid_line, 0.0, check.x) + smoothstep(1.0-grid_line, 1.0, check.x) +
               smoothstep(grid_line, 0.0, check.y) + smoothstep(1.0-grid_line, 1.0, check.y);

    let dist = length(p.xz - 0.5);
    let fade = smoothstep(5.0, 0.8, dist);

    // Floor albedo (how reflective it is) - light gray with grid
    let albedo = mix(vec3f(0.85), vec3f(0.6), clamp(line, 0.0, 1.0) * fade);

    // FIRE IS THE ONLY LIGHT SOURCE
    // Sample fire illumination from the volume
    let gi = get_volume_lighting(p);

    // Soft ambient so floor is slightly visible even without fire
    let ambient = vec3f(0.04);

    // Subtle fire illumination, not overpowering
    return albedo * gi * 1.2 + ambient * albedo;
}

@fragment fn frag_main(in: VertexOutput) -> @location(0) vec4f {
  let ro = params.cameraPos; let fwd = normalize(params.targetPos - ro);
  let right = normalize(cross(vec3f(0.0, 1.0, 0.0), fwd)); let up = cross(fwd, right);
  let rd = normalize(fwd + right * (in.uv.x - 0.5) * 2.0 + up * (in.uv.y - 0.5) * 2.0);
  let jitter = fract(sin(dot(in.uv + fract(params.time * 0.05), vec2f(12.9898, 78.233))) * 43758.5453);
  let t = intersectAABB(ro, rd, vec3f(0.0), vec3f(1.0, params.volumeHeight, 1.0));

  let lightDir = normalize(vec3f(0.3, 1.0, 0.4));
  var bgCol = vec3f(0.22, 0.22, 0.24); // Medium gray background

  // Calculate floor color once, used by both paths
  var floor_color = bgCol;
  if (rd.y < 0.0) {
      let t_floor = -ro.y / rd.y;
      if (t_floor > 0.0) {
          floor_color = get_floor_material(ro + rd * t_floor);
      }
  }

  // Early exit if ray misses volume - just show floor/background
  if (t.x > t.y || t.y < 0.0) {
    let mapped = tonemap_aces(floor_color * params.exposure);
    let safeGamma = max(params.gamma, 0.1);
    return vec4f(pow(mapped, vec3f(1.0 / safeGamma)), 1.0);
  }

  let tNear = max(0.0, t.x); var tVolumeFar = t.y; var solidColor = vec3f(0.0); var hasSolid = false;

  // LOGS / WOOD RENDERING - lit only by fire
  if (params.sceneType == 0.0 || params.sceneType == 4.0) {
     var tS = tNear;
     for(var i=0; i<90; i++) {
        let p = ro + rd * tS; let d = get_wood_sdf(p);
        if (d < 0.0004) {
            hasSolid = true; tVolumeFar = min(t.y, tS);
            let n = get_wood_normal(p);
            let woodCol = mix(vec3f(0.12, 0.07, 0.03), vec3f(0.25, 0.15, 0.08), sin(length(p.xz-0.5)*120.0 + p.y*35.0)*0.5+0.5);

            // Fire is the only light source for wood
            let gi = get_volume_lighting(p);
            let ambient = vec3f(0.01);
            solidColor = woodCol * gi * 3.0 + woodCol * ambient;
            break;
        }
        tS += d * 0.92; if (tS > t.y) { break; }
     }
  }

  // WALL RENDERING - lit by fire (independent of volume bounds)
  if (!hasSolid) {
     var tW = 0.01;
     for(var i=0; i<80; i++) {
        let p = ro + rd * tW;
        let d = get_wall_sdf(p);
        if (d < 0.001) {
            hasSolid = true;
            if (tW < tVolumeFar) { tVolumeFar = tW; }
            let n = get_wall_normal(p);

            // Rough plaster/concrete wall color
            let wallCol = vec3f(0.75, 0.72, 0.68);

            // Fire illumination on wall
            let gi = get_volume_lighting(p);
            let ambient = vec3f(0.015);

            // Simple lambertian - wall faces toward fire get more light
            let toFire = normalize(vec3f(0.5, 0.3, 0.5) - p);
            let facing = max(0.0, dot(n, toFire));

            solidColor = wallCol * gi * 2.5 * (0.3 + facing * 0.7) + wallCol * ambient;
            break;
        }
        tW += max(d * 0.9, 0.005);
        if (tW > 5.0) { break; }
     }
  }

  let baseSteps = 240;
  let steps = max(1, i32(round(f32(baseSteps) * params.stepQuality)));
  let stepSize = max(1e-5, (tVolumeFar - tNear) / f32(steps)); var pos = ro + rd * (tNear + stepSize * jitter);
  var accumCol = vec3f(0.0); var transmittance = 1.0; let phaseSun = phase_function(dot(rd, lightDir), params.anisotropyG);

  for (var i = 0; i < steps; i++) {
      if (transmittance < 0.005) { break; }
     if (inside_volume_world(pos)) {
       let uv = to_volume_uv(pos);
       let val = sample_volume(uv);
           let soot = val.y;
           let temp = val.x;
           let fuel = val.z;
       var reaction = compute_reaction(uv);
           let fuelGate = 0.25 + 0.75 * smoothstep(0.0, 0.10, fuel);
           reaction *= fuelGate;

           // Smoke taxonomy (minimum viable realism):
           // - soot: dark/absorbing, reduced near hottest flame but not removed
           // - haze: scattering-dominant, appears as soot cools (mid/far plume)
           let hot = smoothstep(0.35, 0.7, temp);
           let cool = smoothstep(params.T_hazeStart, params.T_hazeFull, 1.0 - temp);
            let height = smoothstep(0.22, 0.92, uv.y);
           let sootVis = mix(1.0, 0.25, hot);
           let sootRaw = soot * sootVis;
           let hazeRaw = soot * cool * height;
           let sootOpt = 1.0 - exp(-sootRaw * 0.10);
           let hazeOpt = 1.0 - exp(-hazeRaw * 0.04);

           if ((sootOpt + hazeOpt) > 0.0001 || reaction > 0.0001) {
             // Absorption/scattering split: soot absorbs, haze scatters.
             let thickness = max(0.0, params.smokeThickness);
             let darkness = clamp(params.smokeDarkness, 0.0, 1.0);
             let absorption = params.absorption * thickness * (0.65 + 0.7 * darkness);
             let scattering = params.scattering * thickness * (0.85 - 0.55 * darkness);

             let sigmaA = sootOpt * absorption * 1.10 + hazeOpt * absorption * 0.12;
             let sigmaS = sootOpt * scattering * 0.12 + hazeOpt * scattering * 0.55;
             let sigmaT = sigmaA + sigmaS;
             let stepTrans = exp(-sigmaT * stepSize);

             let sunTrans = get_light_transmittance(pos, lightDir);
             let sootTint = vec3f(0.10, 0.09, 0.085);
             let hazeTint = vec3f(0.18, 0.18, 0.19);
             let sootFrac = clamp(sootOpt / max(1e-6, sootOpt + hazeOpt), 0.0, 1.0);
             let smokeTint = mix(hazeTint, sootTint, sootFrac);
             let scatterRadiance = vec3f(9.0) * sunTrans * phaseSun * smokeTint;
             let scatterWeight = (sigmaS / max(1e-6, sigmaT)) * (1.0 - stepTrans);
             accumCol += scatterRadiance * scatterWeight * transmittance;

             // Flame-only emission: reaction emits, soot only attenuates.
             // Apply a *soft* self-shadow (don't zero out emission inside smoke).
             let shadowTr = max(0.35, mix(1.0, sunTrans, 0.35));
             let emission = getBlackbodyColor(temp) * params.emission * reaction * shadowTr;
             accumCol += emission * transmittance * stepSize;

             transmittance *= stepTrans;
           }
      }
      pos += rd * stepSize;
  }

  // Use same floor_color calculated at start
  let surfaceCol = select(floor_color, solidColor, hasSolid);
  let mapped = tonemap_aces((accumCol + transmittance * surfaceCol) * params.exposure);
  let safeGamma = max(params.gamma, 0.1);
  return vec4f(pow(mapped, vec3f(1.0 / safeGamma)), 1.0);
}
`;

export const SCENES: ScenePreset[] = [
  { id: 0, name: 'Campfire', params: { vorticity: 3.4, dissipation: 0.936, buoyancy: 1.5, drag: 0.0, emission: 8.4, scattering: 2.9, absorption: 26.5, smokeWeight: -2.0, plumeTurbulence: 10.0, smokeDissipation: 0.92, exposure: 0.9, gamma: 2.2, windX: 0.0, windZ: 0.0, turbFreq: 28.0, turbSpeed: 1.0, fuelEfficiency: 1.0, heatDiffusion: 0.0, stepQuality: 1.0 } },
  { id: 4, name: 'Wood Combustion', params: { vorticity: 2.2, dissipation: 0.903, buoyancy: 1.8, drag: 0.037, emission: 1.9, scattering: 6.5, absorption: 12.0, smokeWeight: 0.5, plumeTurbulence: 2.81, smokeDissipation: 0.85, windX: -0.05, windZ: 0.05, turbFreq: 15.0, turbSpeed: 2.5, fuelEfficiency: 1.5, heatDiffusion: 0.1, stepQuality: 1.0 } },
  { id: 1, name: 'Candle', params: { vorticity: 3.5, dissipation: 0.92, buoyancy: 4.5, drag: 0.08, emission: 1.0, scattering: 2.5, absorption: 2.0, smokeWeight: 0.3, plumeTurbulence: 0.05, smokeDissipation: 0.985, windX: 0.0, windZ: 0.0, turbFreq: 45.0, turbSpeed: 0.2, fuelEfficiency: 0.5, heatDiffusion: 0.0, stepQuality: 1.5 } },
  { id: 2, name: 'Dual Source', params: { vorticity: 15.0, dissipation: 0.985, buoyancy: 8.0, drag: 0.02, emission: 1.8, scattering: 4.5, absorption: 4.0, smokeWeight: 1.5, plumeTurbulence: 0.3, smokeDissipation: 0.992, windX: 0.2, windZ: 0.2, turbFreq: 20.0, turbSpeed: 1.5, fuelEfficiency: 1.0, heatDiffusion: 0.05, stepQuality: 1.0 } },
  { id: 3, name: 'Firebending', params: { vorticity: 12.0, dissipation: 0.965, buoyancy: 5.0, drag: 0.002, emission: 3.0, scattering: 3.5, absorption: 1.5, smokeWeight: 0.5, plumeTurbulence: 0.8, smokeDissipation: 0.98, windX: 0.0, windZ: 0.0, turbFreq: 32.0, turbSpeed: 5.0, fuelEfficiency: 1.2, heatDiffusion: 0.0, stepQuality: 0.8 } },
  { id: 5, name: 'Gas Explosion', params: { vorticity: 35.0, dissipation: 0.94, buoyancy: 16.0, drag: 0.01, emission: 6.0, scattering: 4.0, absorption: 1.0, smokeWeight: -0.5, plumeTurbulence: 1.5, smokeDissipation: 0.92, windX: 0.0, windZ: 0.0, turbFreq: 12.0, turbSpeed: 0.5, fuelEfficiency: 3.0, heatDiffusion: 0.2, stepQuality: 1.0 } },
  { id: 6, name: 'Nuke', params: { vorticity: 50.0, dissipation: 0.998, buoyancy: 3.0, drag: 0.05, emission: 6.5, scattering: 8.0, absorption: 7.0, smokeWeight: 3.0, plumeTurbulence: 0.4, smokeDissipation: 0.999, windX: 0.0, windZ: 0.0, turbFreq: 8.0, turbSpeed: 0.1, fuelEfficiency: 5.0, heatDiffusion: 0.5, stepQuality: 1.2 } }
];
