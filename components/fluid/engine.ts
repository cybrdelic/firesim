import { WOOD_PILE_DESCRIPTOR, buildWoodSdfWgsl } from './woodSystem';

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

class ShaderContract {
  public layout: GPUPipelineLayout;
  public bindGroupLayout: GPUBindGroupLayout;

  constructor(device: GPUDevice, label: string, entries: any[]) {
    this.bindGroupLayout = device.createBindGroupLayout({
      entries,
      label: `${label}_BGL`
    });

    this.layout = device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout],
      label: `${label}_PL`
    });
  }

  createBindGroup(device: GPUDevice, label: string, entries: any[]): GPUBindGroup {
    return device.createBindGroup({
      layout: this.bindGroupLayout,
      entries,
      label: `${label}_BG`
    });
  }
}

export class FluidTransport {
  public uniformBuffer: GPUBuffer;

  private uniformStaging: ArrayBuffer;
  private uniformView: DataView;

  public densityA: GPUBuffer;
  public densityB: GPUBuffer;
  public fuelA: GPUBuffer;
  public fuelB: GPUBuffer;
  public sootA: GPUBuffer;
  public sootB: GPUBuffer;
  public velocityA: GPUBuffer;
  public velocityB: GPUBuffer;
  public velocityScratch: GPUBuffer;
  public divergence: GPUBuffer;
  public pressureA: GPUBuffer;
  public pressureB: GPUBuffer;
  public occupancy: GPUBuffer;
  public rayStepCounter: GPUBuffer;
  public velocityBufferSize: number;
  public macrocellsPerAxis: number;

  public physicsContract!: ShaderContract;
  public renderContract!: ShaderContract;
  public projectionDivContract!: ShaderContract;
  public projectionJacobiContract!: ShaderContract;
  public projectionGradContract!: ShaderContract;
  public occupancyContract!: ShaderContract;

  public physicsGroups: GPUBindGroup[] = [];
  public renderGroups: GPUBindGroup[] = [];
  public projectionDivGroups: GPUBindGroup[] = [];
  public projectionJacobiGroups: GPUBindGroup[] = [];
  public projectionGradGroups: GPUBindGroup[][] = [[], []];
  public occupancyGroups: GPUBindGroup[] = [];

  constructor(
    private device: GPUDevice,
    public dim: number
  ) {
    const VOXEL_COUNT = dim * dim * dim;

    this.uniformBuffer = device.createBuffer({
      size: 288,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    this.uniformStaging = new ArrayBuffer(288);
    this.uniformView = new DataView(this.uniformStaging);

    const bufferUsage = (window as any).GPUBufferUsage;
    const storageUsage = bufferUsage.STORAGE | bufferUsage.COPY_DST | bufferUsage.COPY_SRC;

    this.densityA = device.createBuffer({ size: VOXEL_COUNT * 4, usage: storageUsage });
    this.densityB = device.createBuffer({ size: VOXEL_COUNT * 4, usage: storageUsage });

    this.fuelA = device.createBuffer({ size: VOXEL_COUNT * 4, usage: storageUsage });
    this.fuelB = device.createBuffer({ size: VOXEL_COUNT * 4, usage: storageUsage });

    this.sootA = device.createBuffer({ size: VOXEL_COUNT * 4, usage: storageUsage });
    this.sootB = device.createBuffer({ size: VOXEL_COUNT * 4, usage: storageUsage });

    this.velocityBufferSize = VOXEL_COUNT * 16;
    this.velocityA = device.createBuffer({ size: this.velocityBufferSize, usage: storageUsage });
    this.velocityB = device.createBuffer({ size: this.velocityBufferSize, usage: storageUsage });
    this.velocityScratch = device.createBuffer({ size: this.velocityBufferSize, usage: storageUsage });
    this.divergence = device.createBuffer({ size: VOXEL_COUNT * 4, usage: storageUsage });
    this.pressureA = device.createBuffer({ size: VOXEL_COUNT * 4, usage: storageUsage });
    this.pressureB = device.createBuffer({ size: VOXEL_COUNT * 4, usage: storageUsage });

    // Task C: Macrocell occupancy grid (8 voxels per macrocell per axis)
    const MACROCELL_SIZE = 8;
    this.macrocellsPerAxis = Math.ceil(dim / MACROCELL_SIZE);
    const macrocellCount = this.macrocellsPerAxis ** 3;
    this.occupancy = device.createBuffer({ size: macrocellCount * 4, usage: storageUsage });
    device.queue.writeBuffer(this.occupancy, 0, new Uint32Array(macrocellCount));

    // Task B: Ray step counter (single atomic u32)
    this.rayStepCounter = device.createBuffer({ size: 4, usage: storageUsage });

    const zeroF32 = new Float32Array(VOXEL_COUNT);
    // Initialize velocity.w = oxygen = 1.0 (fully oxygenated atmosphere)
    const initVec4 = new Float32Array(VOXEL_COUNT * 4);
    for (let i = 0; i < VOXEL_COUNT; i++) {
      initVec4[i * 4 + 3] = 1.0; // .w = oxygen
    }

    device.queue.writeBuffer(this.densityA, 0, zeroF32);
    device.queue.writeBuffer(this.densityB, 0, zeroF32);
    device.queue.writeBuffer(this.fuelA, 0, zeroF32);
    device.queue.writeBuffer(this.fuelB, 0, zeroF32);
    device.queue.writeBuffer(this.sootA, 0, zeroF32);
    device.queue.writeBuffer(this.sootB, 0, zeroF32);
    device.queue.writeBuffer(this.velocityA, 0, initVec4);
    device.queue.writeBuffer(this.velocityB, 0, initVec4);
    device.queue.writeBuffer(this.velocityScratch, 0, initVec4);
    device.queue.writeBuffer(this.divergence, 0, zeroF32);
    device.queue.writeBuffer(this.pressureA, 0, zeroF32);
    device.queue.writeBuffer(this.pressureB, 0, zeroF32);
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
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ]);

    this.renderContract = new ShaderContract(this.device, 'Render', [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 6, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'storage' } },
    ]);

    // Task C: Occupancy compute contract
    this.occupancyContract = new ShaderContract(this.device, 'Occupancy', [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ]);

    this.projectionDivContract = new ShaderContract(this.device, 'ProjectionDiv', [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ]);

    this.projectionJacobiContract = new ShaderContract(this.device, 'ProjectionJacobi', [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ]);

    this.projectionGradContract = new ShaderContract(this.device, 'ProjectionGrad', [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
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
      { binding: 7, resource: { buffer: this.sootA } },
      { binding: 8, resource: { buffer: this.sootB } },
    ]);

    this.renderGroups[0] = this.renderContract.createBindGroup(this.device, 'Render0', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.densityB } },
      { binding: 2, resource: { buffer: this.velocityB } },
      { binding: 3, resource: { buffer: this.fuelB } },
      { binding: 4, resource: { buffer: this.sootB } },
      { binding: 5, resource: { buffer: this.occupancy } },
      { binding: 6, resource: { buffer: this.rayStepCounter } },
    ]);

    this.physicsGroups[1] = this.physicsContract.createBindGroup(this.device, 'Phys1', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.densityB } },
      { binding: 2, resource: { buffer: this.densityA } },
      { binding: 3, resource: { buffer: this.velocityB } },
      { binding: 4, resource: { buffer: this.velocityA } },
      { binding: 5, resource: { buffer: this.fuelB } },
      { binding: 6, resource: { buffer: this.fuelA } },
      { binding: 7, resource: { buffer: this.sootB } },
      { binding: 8, resource: { buffer: this.sootA } },
    ]);

    this.renderGroups[1] = this.renderContract.createBindGroup(this.device, 'Render1', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.densityA } },
      { binding: 2, resource: { buffer: this.velocityA } },
      { binding: 3, resource: { buffer: this.fuelA } },
      { binding: 4, resource: { buffer: this.sootA } },
      { binding: 5, resource: { buffer: this.occupancy } },
      { binding: 6, resource: { buffer: this.rayStepCounter } },
    ]);

    // Task C: Occupancy bind groups (match render group buffer selection)
    this.occupancyGroups[0] = this.occupancyContract.createBindGroup(this.device, 'Occupancy0', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.densityB } },
      { binding: 2, resource: { buffer: this.fuelB } },
      { binding: 3, resource: { buffer: this.sootB } },
      { binding: 4, resource: { buffer: this.occupancy } },
    ]);
    this.occupancyGroups[1] = this.occupancyContract.createBindGroup(this.device, 'Occupancy1', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.densityA } },
      { binding: 2, resource: { buffer: this.fuelA } },
      { binding: 3, resource: { buffer: this.sootA } },
      { binding: 4, resource: { buffer: this.occupancy } },
    ]);

    this.projectionDivGroups[0] = this.projectionDivContract.createBindGroup(this.device, 'ProjectionDiv0', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.velocityB } },
      { binding: 2, resource: { buffer: this.divergence } },
    ]);

    this.projectionDivGroups[1] = this.projectionDivContract.createBindGroup(this.device, 'ProjectionDiv1', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.velocityA } },
      { binding: 2, resource: { buffer: this.divergence } },
    ]);

    this.projectionJacobiGroups[0] = this.projectionJacobiContract.createBindGroup(this.device, 'ProjectionJacobiA2B', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.divergence } },
      { binding: 2, resource: { buffer: this.pressureA } },
      { binding: 3, resource: { buffer: this.pressureB } },
    ]);

    this.projectionJacobiGroups[1] = this.projectionJacobiContract.createBindGroup(this.device, 'ProjectionJacobiB2A', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.divergence } },
      { binding: 2, resource: { buffer: this.pressureB } },
      { binding: 3, resource: { buffer: this.pressureA } },
    ]);

    this.projectionGradGroups[0][0] = this.projectionGradContract.createBindGroup(this.device, 'ProjectionGrad0PressureA', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.velocityScratch } },
      { binding: 2, resource: { buffer: this.pressureA } },
      { binding: 3, resource: { buffer: this.velocityB } },
    ]);

    this.projectionGradGroups[0][1] = this.projectionGradContract.createBindGroup(this.device, 'ProjectionGrad0PressureB', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.velocityScratch } },
      { binding: 2, resource: { buffer: this.pressureB } },
      { binding: 3, resource: { buffer: this.velocityB } },
    ]);

    this.projectionGradGroups[1][0] = this.projectionGradContract.createBindGroup(this.device, 'ProjectionGrad1PressureA', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.velocityScratch } },
      { binding: 2, resource: { buffer: this.pressureA } },
      { binding: 3, resource: { buffer: this.velocityA } },
    ]);

    this.projectionGradGroups[1][1] = this.projectionGradContract.createBindGroup(this.device, 'ProjectionGrad1PressureB', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.velocityScratch } },
      { binding: 2, resource: { buffer: this.pressureB } },
      { binding: 3, resource: { buffer: this.velocityA } },
    ]);
  }

  public updateUniforms(
    now: number,
    params: any,
    camera: { pos: number[], target: number[] },
    sceneType: number
  ) {
    const view = this.uniformView;

    view.setFloat32(0, this.dim, true);
    view.setFloat32(4, now / 1000.0, true);
    view.setFloat32(8, params.timeStep, true);
    view.setFloat32(12, params.vorticity, true);

    view.setFloat32(16, params.dissipation, true);
    view.setFloat32(20, params.buoyancy, true);
    view.setFloat32(24, params.drag, true);
    view.setFloat32(28, params.emission, true);

    view.setFloat32(32, params.exposure, true);
    view.setFloat32(36, params.gamma, true);
    view.setFloat32(40, sceneType, true);
    view.setFloat32(44, params.scattering, true);

    view.setFloat32(48, params.absorption, true);
    view.setFloat32(52, params.smokeWeight, true);
    view.setFloat32(56, params.plumeTurbulence, true);
    view.setFloat32(60, params.smokeDissipation, true);

    view.setFloat32(64, camera.pos[0], true);
    view.setFloat32(68, camera.pos[1], true);
    view.setFloat32(72, camera.pos[2], true);
    view.setFloat32(76, 0, true); // cameraPos.w

    view.setFloat32(80, camera.target[0], true);
    view.setFloat32(84, camera.target[1], true);
    view.setFloat32(88, camera.target[2], true);
    view.setFloat32(92, 0, true); // targetPos.w

    view.setFloat32(96, params.windX || 0, true);
    view.setFloat32(100, params.windZ || 0, true);
    view.setFloat32(104, params.turbFreq || 28.0, true);
    view.setFloat32(108, params.turbSpeed || 1.0, true);
    view.setFloat32(112, params.fuelEfficiency || 1.0, true);
    view.setFloat32(116, params.heatDiffusion || 0.0, true);
    view.setFloat32(120, params.stepQuality || 1.0, true);
    view.setFloat32(124, 0.0, true); // pad4

    // Extended combustion + smoke taxonomy + rendering controls.
    const fuelEff = params.fuelEfficiency || 1.0;
    const heightFactor = clamp(params.buoyancy / 8.0, 0.5, 5.0);
    const baseBurnRate = params.burnRate ?? fuelEff * 6.0;
    const baseFuelInject = params.fuelInject ?? (0.4 + fuelEff * 0.6);
    const derivedBurnRate = baseBurnRate / heightFactor;
    const derivedFuelInject = baseFuelInject * heightFactor;
    const derivedVolumeHeight = params.volumeHeight ?? 1.0;
    view.setFloat32(128, params.T_ignite ?? 0.18, true);
    view.setFloat32(132, params.T_burn ?? 0.55, true);
    view.setFloat32(136, derivedBurnRate, true);
    view.setFloat32(140, derivedFuelInject, true);

    view.setFloat32(144, params.heatYield ?? 3.4, true);
    view.setFloat32(148, params.sootYieldFlame ?? 0.55, true);
    view.setFloat32(152, params.sootYieldSmolder ?? 1.1, true);
    view.setFloat32(156, params.hazeConvertRate ?? 0.0, true);

    view.setFloat32(160, params.T_hazeStart ?? 0.35, true);
    view.setFloat32(164, params.T_hazeFull ?? 0.75, true);
    view.setFloat32(168, params.anisotropyG ?? 0.82, true);
    view.setFloat32(172, params.smokeThickness ?? 1.0, true);

    view.setFloat32(176, params.smokeDarkness ?? 0.65, true);
    view.setFloat32(180, params.flameSharpness ?? 4.0, true);
    view.setFloat32(184, params.sootDissipation ?? params.smokeDissipation ?? 0.985, true);
    view.setFloat32(188, derivedVolumeHeight, true);

    // Floor + scene lighting controls.
    view.setFloat32(192, params.floorUvScale ?? 1.35, true);
    view.setFloat32(196, params.floorUvWarp ?? 1.2, true);
    view.setFloat32(200, params.floorBlendStrength ?? 0.85, true);
    view.setFloat32(204, params.floorNormalStrength ?? 1.05, true);

    view.setFloat32(208, params.floorMicroStrength ?? 0.55, true);
    view.setFloat32(212, params.floorSootDarkening ?? 1.25, true);
    view.setFloat32(216, params.floorSootRoughness ?? 1.1, true);
    view.setFloat32(220, params.floorCharStrength ?? 1.2, true);

    view.setFloat32(224, params.floorContactShadow ?? 1.0, true);
    view.setFloat32(228, params.floorSpecular ?? 1.1, true);
    view.setFloat32(232, params.floorFireBounce ?? 1.7, true);
    view.setFloat32(236, params.floorAmbient ?? 0.55, true);

    view.setFloat32(240, params.lightingFireIntensity ?? 1.75, true);
    view.setFloat32(244, params.lightingFireFalloff ?? 1.1, true);
    view.setFloat32(248, params.lightingFlicker ?? 0.2, true);
    view.setFloat32(252, params.lightingGlow ?? 1.35, true);

    const viewportWidth = Math.max(1, Number(params.renderWidth ?? window.innerWidth));
    const viewportHeight = Math.max(1, Number(params.renderHeight ?? window.innerHeight));
    const cameraAspect = viewportWidth / viewportHeight;
    const cameraTanHalfFov = Math.tan((90.0 * Math.PI / 180.0) * 0.5);
    view.setFloat32(256, viewportWidth, true);
    view.setFloat32(260, viewportHeight, true);
    view.setFloat32(264, cameraAspect, true);
    view.setFloat32(268, cameraTanHalfFov, true);
    view.setFloat32(272, params.debugOverlayMode ?? 0.0, true);
    view.setFloat32(276, params.occlusionMode ?? 1.0, true);
    view.setFloat32(280, params.rayStepBudget ?? 160.0, true);
    view.setFloat32(284, params.occlusionStepBudget ?? 80.0, true);

    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformStaging);
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

  // Use vec4 slots to avoid implicit std140-style padding traps for vec3.
  // xyz used; w unused.
  cameraPos: vec4f,
  targetPos: vec4f,

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

  // Floor + scene lighting controls
  floorUvScale: f32,
  floorUvWarp: f32,
  floorBlendStrength: f32,
  floorNormalStrength: f32,

  floorMicroStrength: f32,
  floorSootDarkening: f32,
  floorSootRoughness: f32,
  floorCharStrength: f32,

  floorContactShadow: f32,
  floorSpecular: f32,
  floorFireBounce: f32,
  floorAmbient: f32,

  lightingFireIntensity: f32,
  lightingFireFalloff: f32,
  lightingFlicker: f32,
  lightingGlow: f32,

  viewportWidth: f32,
  viewportHeight: f32,
  cameraAspect: f32,
  cameraTanHalfFov: f32,

  // Debug + budget controls
  debugOverlayMode: f32,
  occlusionMode: f32,
  rayStepBudget: f32,
  occlusionStepBudget: f32,
};
`;

const WOOD_SDF_FN = buildWoodSdfWgsl(WOOD_PILE_DESCRIPTOR);

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
@group(0) @binding(7) var<storage, read> sootIn: array<f32>;
@group(0) @binding(8) var<storage, read_write> sootOut: array<f32>;

// --- Index helpers ---
fn get_idx(p: vec3i) -> u32 {
  let d = i32(params.dim);
  let cp = clamp(p, vec3i(0), vec3i(d - 1));
  return u32(cp.z * d * d + cp.y * d + cp.x);
}

fn in_bounds(p: vec3i) -> bool {
  let d = i32(params.dim);
  return p.x >= 0 && p.x < d && p.y >= 0 && p.y < d && p.z >= 0 && p.z < d;
}

// --- Boundary classification ---
// Floor (y<=0): solid wall
// Top (y>=dim-1): open outflow
// Sides (x,z at 0 or dim-1): open outflow
// Wood SDF < 0: solid obstacle
fn is_solid_voxel(p: vec3i) -> bool {
  if (p.y <= 0) { return true; }
  // Wood is NOT a hard solid here - combustion must be able to occur
  // at and slightly inside the wood surface. Wood friction is applied
  // later; projection shaders handle velocity zeroing inside wood.
  return false;
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

// --- State struct now includes oxygen from velocity.w ---
struct State {
  vel: vec3f,
  oxygen: f32,
  temp: f32,
  fuel: f32,
  soot: f32,
};

// Outflow-aware trilinear sampling (Task 2):
// Returns zero/ambient for scalars outside domain instead of clamping.
fn sample_state(pos: vec3f) -> State {
  let d = params.dim;
  let st = pos * d - 0.5;
  let i = floor(st);
  let f = fract(st);
  let u = f * f * (3.0 - 2.0 * f);
  let i0 = vec3i(i);
  let i1 = i0 + vec3i(1);

  // Outflow BC: if backtrace center is outside the domain, return ambient.
  // (Stencil overlap at interior boundaries is handled by get_idx clamping.)
  if (pos.x < 0.0 || pos.x > 1.0 || pos.y < 0.0 || pos.y > 1.0 || pos.z < 0.0 || pos.z > 1.0) {
    return State(vec3f(0.0), 1.0, 0.0, 0.0, 0.0);
  }

  // Velocity+oxygen (packed as vec4: xyz=vel, w=oxygen)
  let v000 = velocityIn[get_idx(vec3i(i0.x, i0.y, i0.z))];
  let v100 = velocityIn[get_idx(vec3i(i1.x, i0.y, i0.z))];
  let v010 = velocityIn[get_idx(vec3i(i0.x, i1.y, i0.z))];
  let v110 = velocityIn[get_idx(vec3i(i1.x, i1.y, i0.z))];
  let v001 = velocityIn[get_idx(vec3i(i0.x, i0.y, i1.z))];
  let v101 = velocityIn[get_idx(vec3i(i1.x, i0.y, i1.z))];
  let v011 = velocityIn[get_idx(vec3i(i0.x, i1.y, i1.z))];
  let v111 = velocityIn[get_idx(vec3i(i1.x, i1.y, i1.z))];
  let vm = mix(mix(mix(v000, v100, u.x), mix(v010, v110, u.x), u.y), mix(mix(v001, v101, u.x), mix(v011, v111, u.x), u.y), u.z);

  // Temperature (density buffer)
  let t000 = densityIn[get_idx(vec3i(i0.x, i0.y, i0.z))];
  let t100 = densityIn[get_idx(vec3i(i1.x, i0.y, i0.z))];
  let t010 = densityIn[get_idx(vec3i(i0.x, i1.y, i0.z))];
  let t110 = densityIn[get_idx(vec3i(i1.x, i1.y, i0.z))];
  let t001 = densityIn[get_idx(vec3i(i0.x, i0.y, i1.z))];
  let t101 = densityIn[get_idx(vec3i(i1.x, i0.y, i1.z))];
  let t011 = densityIn[get_idx(vec3i(i0.x, i1.y, i1.z))];
  let t111 = densityIn[get_idx(vec3i(i1.x, i1.y, i1.z))];
  let tm = mix(mix(mix(t000, t100, u.x), mix(t010, t110, u.x), u.y), mix(mix(t001, t101, u.x), mix(t011, t111, u.x), u.y), u.z);

  // Fuel
  let fu000 = fuelIn[get_idx(vec3i(i0.x, i0.y, i0.z))];
  let fu100 = fuelIn[get_idx(vec3i(i1.x, i0.y, i0.z))];
  let fu010 = fuelIn[get_idx(vec3i(i0.x, i1.y, i0.z))];
  let fu110 = fuelIn[get_idx(vec3i(i1.x, i1.y, i0.z))];
  let fu001 = fuelIn[get_idx(vec3i(i0.x, i0.y, i1.z))];
  let fu101 = fuelIn[get_idx(vec3i(i1.x, i0.y, i1.z))];
  let fu011 = fuelIn[get_idx(vec3i(i0.x, i1.y, i1.z))];
  let fu111 = fuelIn[get_idx(vec3i(i1.x, i1.y, i1.z))];
  let fum = mix(mix(mix(fu000, fu100, u.x), mix(fu010, fu110, u.x), u.y), mix(mix(fu001, fu101, u.x), mix(fu011, fu111, u.x), u.y), u.z);

  // Soot (separate buffer - Task 4)
  let s000 = sootIn[get_idx(vec3i(i0.x, i0.y, i0.z))];
  let s100 = sootIn[get_idx(vec3i(i1.x, i0.y, i0.z))];
  let s010 = sootIn[get_idx(vec3i(i0.x, i1.y, i0.z))];
  let s110 = sootIn[get_idx(vec3i(i1.x, i1.y, i0.z))];
  let s001 = sootIn[get_idx(vec3i(i0.x, i0.y, i1.z))];
  let s101 = sootIn[get_idx(vec3i(i1.x, i0.y, i1.z))];
  let s011 = sootIn[get_idx(vec3i(i0.x, i1.y, i1.z))];
  let s111 = sootIn[get_idx(vec3i(i1.x, i1.y, i1.z))];
  let sm = mix(mix(mix(s000, s100, u.x), mix(s010, s110, u.x), u.y), mix(mix(s001, s101, u.x), mix(s011, s111, u.x), u.y), u.z);

  return State(vm.xyz, vm.w, tm, fum, sm);
}

fn curl(p: vec3i) -> vec3f {
  let L = velocityIn[get_idx(p + vec3i(-1,0,0))];
  let R = velocityIn[get_idx(p + vec3i(1,0,0))];
  let D = velocityIn[get_idx(p + vec3i(0,-1,0))];
  let U = velocityIn[get_idx(p + vec3i(0,1,0))];
  let B = velocityIn[get_idx(p + vec3i(0,0,-1))];
  let F = velocityIn[get_idx(p + vec3i(0,0,1))];
  return 0.5 * vec3f(
    (U.z - D.z) - (F.y - B.y),
    (F.x - B.x) - (R.z - L.z),
    (R.y - L.y) - (U.x - D.x)
  );
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dim = u32(params.dim);
  if (id.x >= dim || id.y >= dim || id.z >= dim) { return; }
  let idx = get_idx(vec3i(id));
  let dt = params.dt;
  let uvw = (vec3f(id) + 0.5) / params.dim;
  let dx = 1.0 / params.dim;
  let scene = params.sceneType;

  var woodDist = 1.0;
  let usesWood = (scene == 0.0 || scene == 4.0);
  if (usesWood) { woodDist = get_wood_sdf(uvw); }

  // Task 7: Solid mask - zero out everything inside solids
  let isSolid = is_solid_voxel(vec3i(id));
  if (isSolid) {
    densityOut[idx] = 0.0;
    velocityOut[idx] = vec4f(0.0, 0.0, 0.0, 1.0); // oxygen=1 inside solids
    fuelOut[idx] = 0.0;
    sootOut[idx] = 0.0;
    return;
  }

  // --- Task 3: CFL-safe semi-Lagrangian backtrace ---
  // Velocity-to-UVW conversion: historical force tuning assumes ~10x dx scaling
  // (original: vel * dt * (1.3/dim) * advectGain where advectGain ≈ 3..12)
  let vel = velocityIn[idx].xyz;
  let displacement = vel * dt * dx * 10.0;
  let dispMag = length(displacement);
  let maxDisp = dx * 2.0; // CFL limit: max 2 cells of backtrace
  var safeDisp = displacement;
  if (dispMag > maxDisp) {
    safeDisp = displacement * (maxDisp / dispMag);
  }
  let backPos = uvw - safeDisp;

  // Task 2: sample_state returns outflow (zero) for out-of-bounds
  var state = sample_state(backPos);

  var newVel = state.vel;
  var temp = state.temp;
  var oxygen = state.oxygen;
  var fuel = state.fuel;
  var soot = state.soot;

  // External forces: Wind
  newVel += vec3f(params.windX, 0.0, params.windZ) * dt * 40.0;

  // Heat Diffusion
  if (params.heatDiffusion > 0.0) {
    let T_avg = (densityIn[get_idx(vec3i(id) + vec3i(1,0,0))] + densityIn[get_idx(vec3i(id) - vec3i(1,0,0))] +
                 densityIn[get_idx(vec3i(id) + vec3i(0,1,0))] + densityIn[get_idx(vec3i(id) - vec3i(0,1,0))] +
                 densityIn[get_idx(vec3i(id) + vec3i(0,0,1))] + densityIn[get_idx(vec3i(id) - vec3i(0,0,1))]) / 6.0;
    temp = mix(temp, T_avg, params.heatDiffusion * dt * 10.0);
  }

  // Oxygen diffusion (slowly replenishes from surroundings)
  {
    let o_avg = (velocityIn[get_idx(vec3i(id) + vec3i(1,0,0))].w + velocityIn[get_idx(vec3i(id) - vec3i(1,0,0))].w +
                 velocityIn[get_idx(vec3i(id) + vec3i(0,1,0))].w + velocityIn[get_idx(vec3i(id) - vec3i(0,1,0))].w +
                 velocityIn[get_idx(vec3i(id) + vec3i(0,0,1))].w + velocityIn[get_idx(vec3i(id) - vec3i(0,0,1))].w) / 6.0;
    oxygen = mix(oxygen, o_avg, dt * 2.5);
  }

  // Temperature dissipation (near wood: insulation)
  let insulation_factor = smoothstep(0.12, 0.0, woodDist);
  let insulation = mix(params.dissipation, 1.0, insulation_factor * 0.88);
  temp *= insulation;

  // Soot dissipation (independent of velocity - Task 4)
  soot *= clamp(params.smokeDissipation, 0.0, 1.0);

  // Buoyancy (Task 7: not applied inside solids - handled by early return above)
  let thermalLift = max(0.0, (pow(max(0.0, temp), 1.25) * 0.4) - (soot * params.smokeWeight * 0.0018));
  newVel += vec3f(0.0, 1.0, 0.0) * thermalLift * params.buoyancy * dt * 20.0;
  newVel *= max(0.0, 1.0 - params.drag);

  // Turbulence
  if (temp > 0.01) {
    let noise_scale = params.turbFreq + sin(params.time * 1.1) * 6.0;
    let turb = hash(uvw * noise_scale + params.time * 0.35 * params.turbSpeed) * temp * params.plumeTurbulence * 0.7;
    newVel += turb;
  }

  // Vorticity confinement
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

  // --- Task 10: Physics-based reaction (fuel * oxygen * temp gate) ---
  if (usesWood) {
    let logSurfaceZone = smoothstep(0.07, -0.03, woodDist);
    if (logSurfaceZone > 0.0) {
      let crack_field = voronoi_cracks(uvw * 22.0 + params.time * 0.1);
      let crack_mask = smoothstep(0.0, 0.2, crack_field);
      let n_val = noise_fbm(uvw * 15.0 + params.time * 0.5);

      // Fuel injection at wood surface
      let inject = params.fuelInject * logSurfaceZone * (0.55 + 0.45 * crack_mask) * dt;
      fuel = clamp(fuel + inject, 0.0, 1.0);

      let combustionShape = smoothstep(-0.2, 0.8, n_val);

      // Pilot heat for ignition bootstrap
      temp += params.emission * combustionShape * logSurfaceZone * dt * 0.22;

      // Task 10: reaction rate = fuel * oxygen * ignition_gate * burnRate
      let ignite = smoothstep(params.T_ignite, params.T_burn, temp);
      let R = combustionShape * logSurfaceZone * fuel * oxygen * ignite * params.burnRate;
      fuel = max(0.0, fuel - R * dt);
      temp += R * params.heatYield * dt;

      // Task 5: Oxygen consumption (stoichiometric)
      oxygen = max(0.0, oxygen - R * 3.0 * dt);

      // Soot yield: higher under low oxygen (smoldering)
      let hot = smoothstep(params.T_ignite, params.T_burn, temp);
      let smolder = 1.0 - oxygen;
      let sootYield = mix(params.sootYieldFlame, params.sootYieldSmolder, smolder) * (0.4 + 0.6 * (1.0 - hot));
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
      temp += params.emission * dt * 2.0;
      let ignite = smoothstep(params.T_ignite, params.T_burn, temp);
      let R = ignite * oxygen * fuel * params.burnRate * 0.45;
      fuel = max(0.0, fuel - R * dt);
      temp += R * params.heatYield * dt;
      oxygen = max(0.0, oxygen - R * 3.0 * dt);
      soot += R * params.sootYieldFlame * dt;
      newVel += vec3f(0.0, 0.8, 0.0) * R * dt;
    }
  }

  // Wood interior friction (Task 7: obstacle coupling)
  if (usesWood && woodDist < 0.0) {
    let friction = smoothstep(0.0, 0.022, -woodDist);
    newVel *= (1.0 - friction * 0.95);
    temp *= (1.0 - friction * 0.05);
  }

  // --- Task 6: Consistent boundary conditions ---
  let c = vec3i(id);
  let d = i32(params.dim);

  // Floor (y=0): no-slip solid wall
  if (c.y <= 1) {
    let floorDamp = smoothstep(0.0, 2.0, f32(c.y));
    newVel *= floorDamp;
    // Don't erase fuel/soot at floor (combustion happens near y≈0)
  }

  // Open boundaries: no velocity clamping at sides/top.
  // Outflow is handled by: (1) sample_state returns ambient for out-of-bounds backtrace,
  // (2) Dirichlet p=0 in the pressure solver prevents artificial confinement.

  // Clamp outputs
  temp = max(temp, 0.0);
  soot = max(soot, 0.0);
  fuel = max(fuel, 0.0);
  oxygen = clamp(oxygen, 0.0, 1.0);

  densityOut[idx] = temp;
  velocityOut[idx] = vec4f(clamp(newVel, vec3f(-120.0), vec3f(120.0)), oxygen);
  fuelOut[idx] = fuel;
  sootOut[idx] = soot;
}
`;

export const PROJECTION_DIVERGENCE_SHADER = `
${STRUCT_DEF}
${WOOD_SDF_FN}
@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> velocityIn: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> divergenceOut: array<f32>;

fn get_idx(p: vec3i) -> u32 {
  let d = i32(params.dim);
  let cp = clamp(p, vec3i(0), vec3i(d - 1));
  return u32(cp.z * d * d + cp.y * d + cp.x);
}

// Task 6: Consistent velocity BC for divergence computation
// Floor (y=0): solid no-slip → reflect normal component
// Top (y=dim-1): open outflow → extrapolate (use interior value)
// Sides: open outflow → extrapolate
// Wood SDF < 0: solid → zero velocity
fn sample_velocity_div_bc(p: vec3i, center: vec3i) -> vec3f {
  let d = i32(params.dim);
  var v = velocityIn[get_idx(p)].xyz;

  // Solid: wood obstacle
  let uvw = (vec3f(p) + 0.5) / params.dim;
  let scene = params.sceneType;
  if ((scene == 0.0 || scene == 4.0) && get_wood_sdf(uvw) < 0.0) {
    return vec3f(0.0);
  }

  // Floor (solid wall): reflect normal component (v.y=0 at y=0)
  if (p.y < 0) {
    let cv = velocityIn[get_idx(center)].xyz;
    return vec3f(cv.x, 0.0, cv.z);
  }

  // Open boundaries (sides, top): extrapolate from interior
  if (p.x < 0 || p.x >= d) {
    return velocityIn[get_idx(center)].xyz;
  }
  if (p.y >= d) {
    return velocityIn[get_idx(center)].xyz;
  }
  if (p.z < 0 || p.z >= d) {
    return velocityIn[get_idx(center)].xyz;
  }

  return v;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dim = u32(params.dim);
  if (gid.x >= dim || gid.y >= dim || gid.z >= dim) { return; }
  let c = vec3i(gid);

  let vL = sample_velocity_div_bc(c + vec3i(-1, 0, 0), c);
  let vR = sample_velocity_div_bc(c + vec3i(1, 0, 0), c);
  let vD = sample_velocity_div_bc(c + vec3i(0, -1, 0), c);
  let vU = sample_velocity_div_bc(c + vec3i(0, 1, 0), c);
  let vB = sample_velocity_div_bc(c + vec3i(0, 0, -1), c);
  let vF = sample_velocity_div_bc(c + vec3i(0, 0, 1), c);

  let h = 1.0 / max(1.0, params.dim);
  let div = (vR.x - vL.x + vU.y - vD.y + vF.z - vB.z) / (2.0 * h);

  // Task 7: Zero divergence inside solids
  let uvw = (vec3f(c) + 0.5) / params.dim;
  var solid = false;
  if (c.y <= 0) { solid = true; }
  let scene = params.sceneType;
  if ((scene == 0.0 || scene == 4.0) && get_wood_sdf(uvw) < 0.0) { solid = true; }

  divergenceOut[get_idx(c)] = select(div, 0.0, solid);
}
`;

export const PROJECTION_JACOBI_SHADER = `
${STRUCT_DEF}
${WOOD_SDF_FN}
@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> divergenceIn: array<f32>;
@group(0) @binding(2) var<storage, read> pressureIn: array<f32>;
@group(0) @binding(3) var<storage, read_write> pressureOut: array<f32>;

fn get_idx(p: vec3i) -> u32 {
  let d = i32(params.dim);
  let cp = clamp(p, vec3i(0), vec3i(d - 1));
  return u32(cp.z * d * d + cp.y * d + cp.x);
}

// Task 6+7: Explicit Neumann BC for pressure
// At solid boundaries (floor, wood): dp/dn = 0 → use center pressure
// At open boundaries (sides, top): p = 0 (Dirichlet open)
fn sample_pressure_bc(p: vec3i, center: vec3i) -> f32 {
  let d = i32(params.dim);
  let pCenter = pressureIn[get_idx(center)];

  // Solid: wood
  let uvw = (vec3f(p) + 0.5) / params.dim;
  let scene = params.sceneType;
  if ((scene == 0.0 || scene == 4.0) && get_wood_sdf(uvw) < 0.0) {
    return pCenter; // Neumann: dp/dn = 0
  }

  // Floor solid: Neumann
  if (p.y < 0) { return pCenter; }

  // Open boundaries: Dirichlet p=0
  if (p.x < 0 || p.x >= d) { return 0.0; }
  if (p.y >= d) { return 0.0; }
  if (p.z < 0 || p.z >= d) { return 0.0; }

  return pressureIn[get_idx(p)];
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dim = u32(params.dim);
  if (gid.x >= dim || gid.y >= dim || gid.z >= dim) { return; }
  let c = vec3i(gid);

  // Task 7: Skip solid cells (pressure = 0 inside solids)
  let uvw = (vec3f(c) + 0.5) / params.dim;
  var solid = false;
  if (c.y <= 0) { solid = true; }
  let scene = params.sceneType;
  if ((scene == 0.0 || scene == 4.0) && get_wood_sdf(uvw) < 0.0) { solid = true; }
  if (solid) {
    pressureOut[get_idx(c)] = 0.0;
    return;
  }

  let pL = sample_pressure_bc(c + vec3i(-1, 0, 0), c);
  let pR = sample_pressure_bc(c + vec3i(1, 0, 0), c);
  let pD = sample_pressure_bc(c + vec3i(0, -1, 0), c);
  let pU = sample_pressure_bc(c + vec3i(0, 1, 0), c);
  let pB = sample_pressure_bc(c + vec3i(0, 0, -1), c);
  let pF = sample_pressure_bc(c + vec3i(0, 0, 1), c);

  let h = 1.0 / max(1.0, params.dim);
  let h2 = h * h;
  let div = divergenceIn[get_idx(c)];

  // Standard Jacobi iteration (SOR with omega>1 diverges on ping-pong buffers)
  let jacobi = (pL + pR + pD + pU + pB + pF - div * h2) / 6.0;
  pressureOut[get_idx(c)] = jacobi;
}
`;

export const PROJECTION_GRADIENT_SHADER = `
${STRUCT_DEF}
${WOOD_SDF_FN}
@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> velocityIn: array<vec4f>;
@group(0) @binding(2) var<storage, read> pressureIn: array<f32>;
@group(0) @binding(3) var<storage, read_write> velocityOut: array<vec4f>;

fn get_idx(p: vec3i) -> u32 {
  let d = i32(params.dim);
  let cp = clamp(p, vec3i(0), vec3i(d - 1));
  return u32(cp.z * d * d + cp.y * d + cp.x);
}

// Consistent pressure BC for gradient (same as Jacobi)
fn sample_pressure_bc(p: vec3i, center: vec3i) -> f32 {
  let d = i32(params.dim);
  let pCenter = pressureIn[get_idx(center)];

  let uvw = (vec3f(p) + 0.5) / params.dim;
  let scene = params.sceneType;
  if ((scene == 0.0 || scene == 4.0) && get_wood_sdf(uvw) < 0.0) {
    return pCenter;
  }
  if (p.y < 0) { return pCenter; }
  if (p.x < 0 || p.x >= d) { return 0.0; }
  if (p.y >= d) { return 0.0; }
  if (p.z < 0 || p.z >= d) { return 0.0; }
  return pressureIn[get_idx(p)];
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dim = u32(params.dim);
  if (gid.x >= dim || gid.y >= dim || gid.z >= dim) { return; }
  let c = vec3i(gid);
  let d = i32(params.dim);
  let h = 1.0 / max(1.0, params.dim);

  let current = velocityIn[get_idx(c)];

  // Task 7: Zero velocity inside solids
  let uvw = (vec3f(c) + 0.5) / params.dim;
  var solid = false;
  if (c.y <= 0) { solid = true; }
  let scene = params.sceneType;
  if ((scene == 0.0 || scene == 4.0) && get_wood_sdf(uvw) < 0.0) { solid = true; }
  if (solid) {
    velocityOut[get_idx(c)] = vec4f(0.0, 0.0, 0.0, current.w);
    return;
  }

  let pL = sample_pressure_bc(c + vec3i(-1, 0, 0), c);
  let pR = sample_pressure_bc(c + vec3i(1, 0, 0), c);
  let pD = sample_pressure_bc(c + vec3i(0, -1, 0), c);
  let pU = sample_pressure_bc(c + vec3i(0, 1, 0), c);
  let pB = sample_pressure_bc(c + vec3i(0, 0, -1), c);
  let pF = sample_pressure_bc(c + vec3i(0, 0, 1), c);

  let gradP = vec3f(
    (pR - pL) / (2.0 * h),
    (pU - pD) / (2.0 * h),
    (pF - pB) / (2.0 * h)
  );

  var projected = current.xyz - gradP;

  // Floor: no-slip (prevent downward penetration)
  if (c.y <= 1) { projected.y = max(0.0, projected.y); }
  // Open boundaries: Dirichlet p=0 already handles outflow via pressure BC.
  // No velocity clamping at sides/top - that creates artificial walls.

  velocityOut[get_idx(c)] = vec4f(projected, current.w);
}
`;

// Task C: Macrocell occupancy compute shader
// Builds a coarse 3D grid marking which macrocells contain any visible content.
// Each thread handles one macrocell (8³ voxels), sampling every 2nd voxel (4³=64 reads).
export const OCCUPANCY_SHADER = `
${STRUCT_DEF}
@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> densityIn: array<f32>;
@group(0) @binding(2) var<storage, read> fuelIn: array<f32>;
@group(0) @binding(3) var<storage, read> sootIn: array<f32>;
@group(0) @binding(4) var<storage, read_write> occupancy: array<u32>;

const MACROCELL_SIZE: u32 = 8u;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dim = u32(params.dim);
  let macroPerAxis = dim / MACROCELL_SIZE;
  if (gid.x >= macroPerAxis || gid.y >= macroPerAxis || gid.z >= macroPerAxis) { return; }

  let base = gid * MACROCELL_SIZE;
  var occupied = 0u;
  let isWoodScene = params.sceneType == 0.0 || params.sceneType == 4.0;
  let occupancyThreshold = select(0.0009, 0.00055, isWoodScene);
  // Sample both parity lattices (even + odd) to avoid missing thin boundary features.
  for (var z = 0u; z < MACROCELL_SIZE && occupied == 0u; z += 2u) {
    for (var y = 0u; y < MACROCELL_SIZE && occupied == 0u; y += 2u) {
      for (var x = 0u; x < MACROCELL_SIZE && occupied == 0u; x += 2u) {
        let p = base + vec3u(x, y, z);
        if (p.x < dim && p.y < dim && p.z < dim) {
          let idx = p.z * dim * dim + p.y * dim + p.x;
          let d = abs(densityIn[idx]);
          let f = abs(fuelIn[idx]);
          let s = abs(sootIn[idx]);
          let hot = smoothstep(params.T_ignite * 0.75, params.T_burn, d);
          let activity = d + f + s + hot * f * 0.6;
          if (activity > occupancyThreshold) {
            occupied = 1u;
          }
        }
      }
    }
  }
  for (var z = 1u; z < MACROCELL_SIZE && occupied == 0u; z += 2u) {
    for (var y = 1u; y < MACROCELL_SIZE && occupied == 0u; y += 2u) {
      for (var x = 1u; x < MACROCELL_SIZE && occupied == 0u; x += 2u) {
        let p = base + vec3u(x, y, z);
        if (p.x < dim && p.y < dim && p.z < dim) {
          let idx = p.z * dim * dim + p.y * dim + p.x;
          let d = abs(densityIn[idx]);
          let f = abs(fuelIn[idx]);
          let s = abs(sootIn[idx]);
          let hot = smoothstep(params.T_ignite * 0.75, params.T_burn, d);
          let activity = d + f + s + hot * f * 0.6;
          if (activity > occupancyThreshold) {
            occupied = 1u;
          }
        }
      }
    }
  }
  let macroIdx = gid.z * macroPerAxis * macroPerAxis + gid.y * macroPerAxis + gid.x;
  occupancy[macroIdx] = occupied;
}
`;

// Task E: Temporal accumulation — blend current frame with history + neighborhood clamping
export const TEMPORAL_BLEND_SHADER = `
@group(0) @binding(0) var currentTex: texture_2d<f32>;
@group(0) @binding(1) var historyTex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VertexOutput { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex fn vert_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
  var positions = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  var out: VertexOutput;
  out.pos = vec4f(positions[vi], 0.0, 1.0);
  out.uv = positions[vi] * 0.5 + 0.5;
  return out;
}

@fragment fn frag_main(in: VertexOutput) -> @location(0) vec4f {
  let texSize = vec2f(textureDimensions(currentTex));
  let texelSize = 1.0 / texSize;
  let current = textureSample(currentTex, samp, in.uv);
  let history = textureSample(historyTex, samp, in.uv);

  // 3x3 neighborhood clamping: prevent ghosting by clamping history to current's local range
  var minC = current;
  var maxC = current;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      if (dx == 0 && dy == 0) { continue; }
      let neighbor = textureSample(currentTex, samp, in.uv + vec2f(f32(dx), f32(dy)) * texelSize);
      minC = min(minC, neighbor);
      maxC = max(maxC, neighbor);
    }
  }
  // Expand clamp range slightly to allow subtle sub-pixel detail through
  let margin = (maxC - minC) * 0.15;
  let clampedHistory = clamp(history, minC - margin, maxC + margin);

  // Exponential blend: keep history for stability but recover from sampling holes faster.
  let blendAlpha = 0.30;
  return mix(clampedHistory, current, blendAlpha);
}
`;

// Task D: Half-res volume render — bilateral upsample shader
export const UPSAMPLE_SHADER = `
@group(0) @binding(0) var halfResTex: texture_2d<f32>;
@group(0) @binding(1) var halfResSamp: sampler;

struct VertexOutput { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex fn vert_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
  var positions = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  var out: VertexOutput;
  out.pos = vec4f(positions[vi], 0.0, 1.0);
  out.uv = positions[vi] * 0.5 + 0.5;
  return out;
}

@fragment fn frag_main(in: VertexOutput) -> @location(0) vec4f {
  let texSize = vec2f(textureDimensions(halfResTex));
  let texelSize = 1.0 / texSize;
  // 4-tap bilateral: weight by color similarity to preserve flame edges
  let center = textureSample(halfResTex, halfResSamp, in.uv);
  let offsets = array<vec2f, 4>(
    vec2f(-0.5, -0.5) * texelSize,
    vec2f( 0.5, -0.5) * texelSize,
    vec2f(-0.5,  0.5) * texelSize,
    vec2f( 0.5,  0.5) * texelSize
  );
  var weightSum = 1.0;
  var colorSum = center;
  let sigma = 0.15;
  for (var i = 0; i < 4; i++) {
    let s = textureSample(halfResTex, halfResSamp, in.uv + offsets[i]);
    let diff = length(s.rgb - center.rgb);
    let w = exp(-diff * diff / (2.0 * sigma * sigma));
    colorSum += s * w;
    weightSum += w;
  }
  return colorSum / weightSum;
}
`;

export const RENDER_SHADER = `
${STRUCT_DEF}
${WOOD_SDF_FN}
@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> densityIn: array<f32>;
@group(0) @binding(2) var<storage, read> velocityIn: array<vec4f>;
@group(0) @binding(3) var<storage, read> fuelIn: array<f32>;
@group(0) @binding(4) var<storage, read> sootIn: array<f32>;
@group(0) @binding(5) var<storage, read> occupancy: array<u32>;
@group(0) @binding(6) var<storage, read_write> rayStepCounter: atomic<u32>;

// Task C: Macrocell empty-space skip
const MACROCELL_SIZE: u32 = 8u;
fn is_macrocell_occupied(uv: vec3f) -> bool {
  let dim = u32(params.dim);
  let macroPerAxis = dim / MACROCELL_SIZE;
  let fMpa = f32(macroPerAxis);
  let mc = vec3u(clamp(uv * fMpa, vec3f(0.0), vec3f(fMpa - 1.0)));
  let idx = mc.z * macroPerAxis * macroPerAxis + mc.y * macroPerAxis + mc.x;
  return occupancy[idx] != 0u;
}

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

fn safe_norm(v: vec3f, fallback: vec3f) -> vec3f {
  let l2 = dot(v, v);
  if (!(l2 > 1e-12)) { return fallback; }
  return v * inverseSqrt(l2);
}

fn inside_volume_world(p: vec3f) -> bool {
  return p.x >= 0.0 && p.x <= 1.0 && p.z >= 0.0 && p.z <= 1.0 && p.y >= 0.0 && p.y <= params.volumeHeight;
}

fn to_volume_uv(p: vec3f) -> vec3f {
  let h = max(0.0001, params.volumeHeight);
  return vec3f(p.x, clamp(p.y / h, 0.0, 1.0), p.z);
}

// Task 1: REMOVED volume_edge_falloff_uv - no more artificial box edges

// Task 4: sample_volume now reads soot from dedicated sootIn buffer
// Returns vec4f: (temp, soot, fuel, oxygen)
fn sample_volume(pos: vec3f) -> vec4f {
  let d = params.dim; let p = pos * d - 0.5; let i = vec3i(floor(p)); let f = fract(p);
  // Temperature
  let t_res = mix(mix(mix(densityIn[get_idx(i)], densityIn[get_idx(i + vec3i(1,0,0))], f.x), mix(densityIn[get_idx(i + vec3i(0,1,0))], densityIn[get_idx(i + vec3i(1,1,0))], f.x), f.y), mix(mix(densityIn[get_idx(i + vec3i(0,0,1))], densityIn[get_idx(i + vec3i(1,0,1))], f.x), mix(densityIn[get_idx(i + vec3i(0,1,1))], densityIn[get_idx(i + vec3i(1,1,1))], f.x), f.y), f.z);
  // Soot from dedicated buffer
  let s_res = mix(mix(mix(sootIn[get_idx(i)], sootIn[get_idx(i + vec3i(1,0,0))], f.x), mix(sootIn[get_idx(i + vec3i(0,1,0))], sootIn[get_idx(i + vec3i(1,1,0))], f.x), f.y), mix(mix(sootIn[get_idx(i + vec3i(0,0,1))], sootIn[get_idx(i + vec3i(1,0,1))], f.x), mix(sootIn[get_idx(i + vec3i(0,1,1))], sootIn[get_idx(i + vec3i(1,1,1))], f.x), f.y), f.z);
  // Fuel
  let fu_res = mix(mix(mix(fuelIn[get_idx(i)], fuelIn[get_idx(i + vec3i(1,0,0))], f.x), mix(fuelIn[get_idx(i + vec3i(0,1,0))], fuelIn[get_idx(i + vec3i(1,1,0))], f.x), f.y), mix(mix(fuelIn[get_idx(i + vec3i(0,0,1))], fuelIn[get_idx(i + vec3i(1,0,1))], f.x), mix(fuelIn[get_idx(i + vec3i(0,1,1))], fuelIn[get_idx(i + vec3i(1,1,1))], f.x), f.y), f.z);
  // Oxygen from velocity.w
  let o_res = mix(mix(mix(velocityIn[get_idx(i)].w, velocityIn[get_idx(i + vec3i(1,0,0))].w, f.x), mix(velocityIn[get_idx(i + vec3i(0,1,0))].w, velocityIn[get_idx(i + vec3i(1,1,0))].w, f.x), f.y), mix(mix(velocityIn[get_idx(i + vec3i(0,0,1))].w, velocityIn[get_idx(i + vec3i(1,0,1))].w, f.x), mix(velocityIn[get_idx(i + vec3i(0,1,1))].w, velocityIn[get_idx(i + vec3i(1,1,1))].w, f.x), f.y), f.z);
  return vec4f(t_res, s_res, fu_res, o_res);
}

fn sample_volume_safe(pos: vec3f) -> vec4f {
  if (pos.x < 0.0 || pos.x > 1.0 || pos.y < 0.0 || pos.y > 1.0 || pos.z < 0.0 || pos.z > 1.0) {
    return vec4f(0.0);
  }
  return sample_volume(pos);
}

// Task 1: No falloff multiplication - just check bounds and return raw values
fn sample_medium_world(p: vec3f) -> vec4f {
  if (!inside_volume_world(p)) { return vec4f(0.0); }
  let uv = to_volume_uv(p);
  return sample_volume(uv);
}

fn sample_velocity_nearest(pos: vec3f) -> vec3f {
  let d = params.dim;
  let p = pos * d - 0.5;
  let i = vec3i(floor(p));
  return velocityIn[get_idx(i)].xyz;
}

fn hash_vec3(p: vec3f) -> vec3f {
  var p3 = fract(p * vec3f(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yxz + 33.33);
  return fract((p3.xxy + p3.yxx) * p3.zyx) * 2.0 - 1.0;
}

fn cheap_noise(p: vec3f) -> f32 {
  return dot(hash_vec3(p), vec3f(0.3333333));
}

// Task 10: Physics-based reaction using fuel * oxygen * temperature gating
fn compute_reaction(pos: vec3f, temp: f32) -> f32 {
  let tempGate = smoothstep(params.T_ignite, params.T_burn, temp);
  if (tempGate < 0.001) { return 0.0; }

  // Sample fuel and oxygen at this point
  let vol = sample_volume_safe(pos);
  let fuel = vol.z;
  let oxygen = vol.w;
  let fuelGate = smoothstep(0.0, 0.08, fuel);
  let oxygenGate = smoothstep(0.05, 0.3, oxygen);

  // Task G: Softened gradient gating - wider band, less harsh sheets
  let eps = 1.5 / params.dim;
  let txp = sample_volume_safe(pos + vec3f(eps, 0.0, 0.0)).x;
  let txn = sample_volume_safe(pos - vec3f(eps, 0.0, 0.0)).x;
  let typ = sample_volume_safe(pos + vec3f(0.0, eps, 0.0)).x;
  let tyn = sample_volume_safe(pos - vec3f(0.0, eps, 0.0)).x;
  let tzp = sample_volume_safe(pos + vec3f(0.0, 0.0, eps)).x;
  let tzn = sample_volume_safe(pos - vec3f(0.0, 0.0, eps)).x;
  let grad = vec3f(txp - txn, typ - tyn, tzp - tzn);
  let gradMag = length(grad) * 0.5;

  // Widened gradient gate: volumetric glow instead of torn-paper sheets
  let front = smoothstep(0.015, 0.08, gradMag);
  // Blend: 60% temp-gated volumetric + 40% gradient-sharpened front
  let r = tempGate * (0.6 + 0.4 * front) * fuelGate * oxygenGate;
  // Cap sharpness to prevent crunchy artifacts
  let sharp = clamp(params.flameSharpness, 1.0, 3.0);
  return pow(clamp(r, 0.0, 1.0), sharp);
}

fn tonemap_aces(color: vec3f) -> vec3f {
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

// Task 14: Calibrated blackbody ramp - reduced peak to prevent blowout
fn getBlackbodyColor(temp: f32) -> vec3f {
   let t = max(0.0, temp - 0.12);
   if (t < 0.3) { return mix(vec3f(0.0), vec3f(2.0, 0.04, 0.001), t / 0.3); }
   else if (t < 0.75) { return mix(vec3f(2.0, 0.04, 0.001), vec3f(5.0, 1.6, 0.08), (t - 0.3) / 0.45); }
   else if (t < 1.4) { return mix(vec3f(5.0, 1.6, 0.08), vec3f(10.0, 7.0, 1.0), (t - 0.75) / 0.65); }
   else { return mix(vec3f(10.0, 7.0, 1.0), vec3f(18.0, 16.0, 12.0), clamp((t - 1.4) * 0.4, 0.0, 1.0)); }
}

fn get_light_transmittance(pos: vec3f, lightDir: vec3f) -> f32 {
  var p = pos;
  let step = 0.11;
  var tau = 0.0;
  for(var i=0; i<4; i++) {
    p += lightDir * step;
    let m = sample_medium_world(p);
    let soot = m.y;
    let temp = m.x;
    if ((soot + temp) < 1e-5) { continue; }
    let uv = to_volume_uv(p);
    // Smoke taxonomy (minimum viable realism):
    // - soot: dark, absorption-dominant, reduced near hottest flame but not removed
    // - haze: scattering-dominant, appears as soot cools (mid/far plume)
    let hot = smoothstep(0.35, 0.7, temp);
    let cool = smoothstep(params.T_hazeStart, params.T_hazeFull, 1.0 - temp);
    let height = smoothstep(0.22, 0.92, uv.y);

    let sootVis = mix(1.0, 0.25, hot);
    let sootRaw = soot * sootVis;
    let hazeRaw = soot * cool * height;

    let sootOpt = 1.0 - exp(-sootRaw * 0.35);
    let hazeOpt = 1.0 - exp(-hazeRaw * 0.12);

    let thickness = max(0.25, params.smokeThickness);
    let darkness = clamp(params.smokeDarkness, 0.0, 1.0);
    let absorption = params.absorption * thickness * (0.65 + 0.7 * darkness);
    let scattering = params.scattering * thickness * (0.85 - 0.55 * darkness);

    let sigmaA = sootOpt * absorption * 1.10 + hazeOpt * absorption * 0.12;
    let sigmaS = sootOpt * scattering * 0.12 + hazeOpt * scattering * 0.55;
    tau += (sigmaA + sigmaS) * step;
  }
  return exp(-tau);
}

fn intersect_aabb(ro: vec3f, rd: vec3f, bmin: vec3f, bmax: vec3f) -> vec2f {
  let s = select(vec3f(-1.0), vec3f(1.0), rd >= vec3f(0.0));
  let invRd = s / max(abs(rd), vec3f(1e-6));
  let t0 = (bmin - ro) * invRd;
  let t1 = (bmax - ro) * invRd;
  let tMin = min(t0, t1);
  let tMax = max(t0, t1);
  let tNear = max(max(tMin.x, tMin.y), tMin.z);
  let tFar = min(min(tMax.x, tMax.y), tMax.z);
  return vec2f(tNear, tFar);
}

fn sd_box_local(p: vec3f, b: vec3f) -> f32 {
  let q = abs(p) - b;
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sd_sphere_local(p: vec3f, r: f32) -> f32 {
  return length(p) - r;
}

fn sd_capsule_local(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
  return length(pa - ba * h) - r;
}

fn world_occluder_sdf(p: vec3f) -> f32 {
  var d = 1e6;

  d = min(d, get_wood_sdf(p));
  d = min(d, sd_sphere_local(p - vec3f(1.08, 0.16, 0.26), 0.16));
  d = min(d, sd_box_local(p - vec3f(-0.06, 0.13, 0.34), vec3f(0.13, 0.13, 0.13)));
  d = min(d, sd_capsule_local(p, vec3f(0.34, 0.10, -0.1), vec3f(0.34, 0.34, -0.1), 0.1));
  d = min(d, sd_box_local(p - vec3f(0.5, 3.25, -3.4), vec3f(12.0, 4.0, 0.03)));
  return d;
}

fn intersect_occluders(ro: vec3f, rd: vec3f, minT: f32, maxT: f32, useSdfOccluders: bool) -> f32 {
  var hitT = maxT;

  // Analytic floor hit (y=0) is exact and much cheaper than SDF marching.
  if (rd.y < -1e-5) {
    let tFloor = (0.0 - ro.y) / rd.y;
    if (tFloor > minT && tFloor < hitT) {
      hitT = tFloor;
    }
  }

  if (!useSdfOccluders) {
    return hitT;
  }

  let stepBudget = clamp(i32(round(params.occlusionStepBudget)), 0, 128);
  if (stepBudget == 0) {
    return hitT;
  }

  var t = minT + 0.006;
  for (var i = 0; i < 128; i++) {
    if (i >= stepBudget) { break; }
    if (t > hitT) { break; }
    let p = ro + rd * t;
    let d = world_occluder_sdf(p);
    if (d < 0.0017) { return t; }
    t += clamp(d * 0.95, 0.012, 0.24);
  }
  return hitT;
}

@fragment fn frag_main(in: VertexOutput) -> @location(0) vec4f {
  let ro = params.cameraPos.xyz;
  let fwd = safe_norm(params.targetPos.xyz - ro, vec3f(0.0, 0.0, -1.0));
  // Use a right-handed camera basis matching the Three.js world render.
  let rightRaw = cross(fwd, vec3f(0.0, 1.0, 0.0));
  let rightFallback = cross(fwd, vec3f(1.0, 0.0, 0.0));
  let right = safe_norm(select(rightRaw, rightFallback, dot(rightRaw, rightRaw) < 1e-6), vec3f(1.0, 0.0, 0.0));
  let up = safe_norm(cross(right, fwd), vec3f(0.0, 1.0, 0.0));
  let aspect = max(0.001, params.cameraAspect);
  let tanHalfFov = max(0.05, params.cameraTanHalfFov);
  let ndc = (in.uv - 0.5) * 2.0;
  let rd = safe_norm(fwd + right * (ndc.x * aspect * tanHalfFov) + up * (ndc.y * tanHalfFov), fwd);
  // Task F: Blue-noise jitter via R2 quasi-random sequence + frame cycling
  // R2 sequence: alpha1 = 1/phi2, alpha2 = 1/phi2^2 where phi2 = 1.32471795724...
  let frameIdx = u32(params.time * 60.0) % 64u;
  let r2_alpha = vec2f(0.7548776662, 0.5698402910);
  let r2_base = fract(in.uv * vec2f(params.viewportWidth, params.viewportHeight) * r2_alpha);
  let jitter = fract(r2_base.x + f32(frameIdx) * 0.7548776662);

  let lightDir = normalize(vec3f(0.3, 1.0, 0.4));
  let boundsPad = 0.12;
  let volumeHit = intersect_aabb(
    ro,
    rd,
    vec3f(-boundsPad, 0.0, -boundsPad),
    vec3f(1.0 + boundsPad, params.volumeHeight, 1.0 + boundsPad)
  );
  if (volumeHit.y <= max(volumeHit.x, 0.0)) {
    return vec4f(0.0);
  }

  let marchStart = max(0.0, volumeHit.x);
  let volumeExit = volumeHit.y;
  let occlusionMode = i32(round(params.occlusionMode));
  let useSdfOccluders = occlusionMode != 0;
  let worldHitT = intersect_occluders(ro, rd, marchStart, volumeExit, useSdfOccluders);
  if (worldHitT <= marchStart) {
    return vec4f(0.0);
  }

  let maxTraceDist = min(volumeExit, worldHitT);
  let baseSteps = max(1, i32(round(params.rayStepBudget)));
  let steps = clamp(i32(round(f32(baseSteps) * params.stepQuality)), 1, max(baseSteps, 1));
  let baseStep = max(1e-4, maxTraceDist / f32(steps));
  var t = marchStart + baseStep * (0.5 + jitter);
  var accumCol = vec3f(0.0); var transmittance = 1.0; let phaseSun = phase_function(dot(rd, lightDir), params.anisotropyG);
  var cachedSunTrans = 1.0;
  var shadowRefreshCountdown = 0;
  var maxReactionSeen = 0.0;
  var minWoodDist = 1e6;
  var stepsTaken = 0u;
  let isWoodScene = params.sceneType == 0.0 || params.sceneType == 4.0;
  let transmittanceCutoff = select(0.005, 0.0018, isWoodScene);

  for (var i = 0; i < steps; i++) {
    stepsTaken += 1u;
    if (transmittance < transmittanceCutoff || t > maxTraceDist) { break; }
    let pos = ro + rd * t;
    let uv = to_volume_uv(pos);

    // Task C: Macrocell empty-space skip - jump through empty macrocells
    if (!isWoodScene && !is_macrocell_occupied(uv)) {
      // Probe local medium before skipping to avoid false-negative occupancy holes.
      let probe = sample_medium_world(pos);
      let probeActivity = abs(probe.x) + abs(probe.y) + abs(probe.z);
      let probeHot = smoothstep(params.T_ignite * 0.8, params.T_burn, probe.x) * probe.z;
      if (probeActivity + probeHot * 0.4 < 8e-5) {
        // Near wood surfaces, keep marching conservatively to avoid culling thin flames.
        let nearWood = isWoodScene && abs(get_wood_sdf(pos)) < 0.12;
        if (nearWood) {
          t += baseStep * 0.9;
        } else {
          let macroStep = 1.0 / f32(u32(params.dim) / MACROCELL_SIZE);
          t += max(baseStep, macroStep * 0.22);
        }
        continue;
      }
    }

    let m = sample_medium_world(pos);
    let temp = m.x;
    let soot = m.y;
    let fuel = m.z;
    let oxygen = m.w;
    // Task 1: No mediumMask/falloff check - just skip empty voxels by content
    let anyContent = abs(temp) + abs(soot) + abs(fuel);
    if (anyContent < 3e-6) {
      t += baseStep * 1.9;
      continue;
    }
    {
      let baseReaction = smoothstep(params.T_ignite, params.T_burn, temp);
      var reaction = baseReaction;
      if (baseReaction > 0.0005) {
        if (((i & 1) == 0) || baseReaction > 0.72) {
          reaction = compute_reaction(uv, temp);
        } else {
          let approxSharp = max(1.0, params.flameSharpness * 0.4);
          reaction = pow(baseReaction, approxSharp);
        }
      }
      maxReactionSeen = max(maxReactionSeen, reaction);
      if (params.sceneType == 0.0 || params.sceneType == 4.0) {
        minWoodDist = min(minWoodDist, abs(get_wood_sdf(pos)));
      }

      // Smoke taxonomy (minimum viable realism):
      // - soot: dark/absorbing, reduced near hottest flame but not removed
      // - haze: scattering-dominant, appears as soot cools (mid/far plume)
      let hot = smoothstep(0.35, 0.7, temp);
      let cool = smoothstep(params.T_hazeStart, params.T_hazeFull, 1.0 - temp);
      let height = smoothstep(0.22, 0.92, uv.y);
      let sootVis = mix(1.0, 0.25, hot);
      let sootRaw = soot * sootVis;
      let hazeRaw = soot * cool * height;
      let sootOpt = 1.0 - exp(-sootRaw * 0.35);
      let hazeOpt = 1.0 - exp(-hazeRaw * 0.12);
      let activity = sootOpt + hazeOpt + reaction;
      let emptyThreshold = 0.0011 / max(0.5, params.stepQuality);

      if (activity < emptyThreshold) {
        t += baseStep * 1.7;
        continue;
      }

      // Task 11: Adaptive step size based on local extinction + emissive focus
      let emissiveFocus = clamp(max(reaction, baseReaction), 0.0, 1.0);
      // Estimate rough sigmaT for step adaptation before full calculation
      let roughSigmaT = (sootOpt + hazeOpt) * 0.5 + reaction * 0.1;
      let adaptFactor = clamp(1.0 / max(0.1, roughSigmaT * 2.0), 0.3, 1.5);
      let localStep = clamp(baseStep * mix(adaptFactor, 0.42, emissiveFocus), baseStep * 0.35, baseStep * 1.2);

      if ((sootOpt + hazeOpt) > 0.0001 || reaction > 0.0001) {
        // Flow-locked micro detail (optimized): nearest velocity fetch + cheap hash noise.
        let vel = sample_velocity_nearest(uv);
        let velClamped = clamp(vel, vec3f(-60.0), vec3f(60.0));
        let flowUv = uv - velClamped * 0.0006;
        let detailFreq = 10.0 + params.turbFreq * 0.15;
        let detail = clamp(cheap_noise(flowUv * detailFreq + vec3f(0.0, params.time * params.turbSpeed * 0.25, 0.0)), -1.0, 1.0);

        // Absorption/scattering split: soot absorbs, haze scatters.
        let thickness = max(0.25, params.smokeThickness);
        let darkness = clamp(params.smokeDarkness, 0.0, 1.0);
        let absorption = params.absorption * thickness * (0.65 + 0.7 * darkness);
        let scattering = params.scattering * thickness * (0.85 - 0.55 * darkness);

        let microSmoke = clamp(1.0 + detail * 0.06, 0.85, 1.15);
        let sigmaA = (sootOpt * absorption * 1.10 + hazeOpt * absorption * 0.12) * microSmoke;
        let sigmaS = (sootOpt * scattering * 0.12 + hazeOpt * scattering * 0.55) * microSmoke;
        // Keep a small in-flame extinction term so emissive sheets integrate smoothly.
        let sigmaT = max(1e-5, sigmaA + sigmaS + reaction * 0.045);
        let stepTrans = exp(-sigmaT * localStep);

        if (shadowRefreshCountdown <= 0) {
          cachedSunTrans = get_light_transmittance(pos, lightDir);
          let denseMedium = reaction > 0.08 || sigmaT > 0.55;
          shadowRefreshCountdown = select(8, 2, denseMedium);
        } else {
          shadowRefreshCountdown -= 1;
        }

        let sunTrans = cachedSunTrans;
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
        let microFlame = clamp(1.0 + detail * 0.18, 0.7, 1.3);
        let reactionEm = reaction * reaction;
        let emission = getBlackbodyColor(temp) * params.emission * reactionEm * shadowTr * microFlame;
        let emissionIntegral = emission * ((1.0 - stepTrans) / max(1e-4, sigmaT));
        accumCol += emissionIntegral * transmittance;

        transmittance *= stepTrans;
        t += localStep;
      } else {
        t += baseStep;
      }
    }
  }

  // Task B: Accumulate total ray steps for readout
  atomicAdd(&rayStepCounter, stepsTaken);

  let mapped = tonemap_aces(accumCol * params.exposure);
  let outColor = pow(mapped, vec3f(1.0 / params.gamma));
  let luma = dot(outColor, vec3f(0.2126, 0.7152, 0.0722));
  let alpha = clamp(max(1.0 - transmittance, smoothstep(0.01, 0.09, luma) * 0.82), 0.0, 0.98);
  let overlayMode = i32(round(params.debugOverlayMode));
  if (overlayMode == 1) {
    let alphaViz = clamp(alpha, 0.0, 1.0);
    return vec4f(vec3f(alphaViz), 1.0);
  }
  if (overlayMode == 2) {
    let occlusionMask = select(0.0, 1.0, worldHitT < (volumeExit - 1e-4));
    return vec4f(vec3f(occlusionMask), 1.0);
  }
  if (overlayMode == 3) {
    let woodViz = select(0.0, 1.0 - smoothstep(0.0, 0.08, minWoodDist), minWoodDist < 1e5);
    return vec4f(vec3f(woodViz, woodViz * 0.45, 0.08), 1.0);
  }
  if (overlayMode == 4) {
    let reactionViz = clamp(maxReactionSeen, 0.0, 1.0);
    return vec4f(vec3f(reactionViz, reactionViz * reactionViz * 0.72, 0.08), 1.0);
  }
  // Canvas is configured as premultiplied alpha, so output premultiplied color.
  return vec4f(outColor * alpha, alpha);
}
`;

export interface SceneParams {
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

  // Extended combustion + smoke taxonomy + rendering controls
  T_ignite: number;
  T_burn: number;
  burnRate: number;
  fuelInject: number;
  heatYield: number;
  sootYieldFlame: number;
  sootYieldSmolder: number;
  hazeConvertRate: number;
  T_hazeStart: number;
  T_hazeFull: number;
  anisotropyG: number;
  smokeThickness: number;
  smokeDarkness: number;
  flameSharpness: number;
  volumeHeight: number;
}

export interface FloorLightingParams {
  floorUvScale: number;
  floorUvWarp: number;
  floorBlendStrength: number;
  floorNormalStrength: number;
  floorMicroStrength: number;
  floorSootDarkening: number;
  floorSootRoughness: number;
  floorCharStrength: number;
  floorContactShadow: number;
  floorSpecular: number;
  floorFireBounce: number;
  floorAmbient: number;
  lightingFireIntensity: number;
  lightingFireFalloff: number;
  lightingFlicker: number;
  lightingGlow: number;
}

export const SCENES: Array<{ id: number; name: string; params: SceneParams }> = [
  {
    id: 0,
    name: 'Campfire',
    params: {
      vorticity: 4.2,
      dissipation: 0.94,
      buoyancy: 1.8,
      drag: 0.01,
      emission: 7.6,
      scattering: 3.2,
      absorption: 20.0,
      smokeWeight: -1.2,
      plumeTurbulence: 4.8,
      smokeDissipation: 0.987,
      exposure: 0.9,
      gamma: 2.2,
      windX: 0.0,
      windZ: 0.0,
      turbFreq: 24.0,
      turbSpeed: 0.8,
      fuelEfficiency: 1.1,
      heatDiffusion: 0.02,
      stepQuality: 1.0,
      T_ignite: 0.18,
      T_burn: 0.55,
      burnRate: 6.5,
      fuelInject: 0.95,
      heatYield: 3.4,
      sootYieldFlame: 0.55,
      sootYieldSmolder: 1.1,
      hazeConvertRate: 0.0,
      T_hazeStart: 0.35,
      T_hazeFull: 0.75,
      anisotropyG: 0.82,
      smokeThickness: 1.0,
      smokeDarkness: 0.25,
      flameSharpness: 4.0,
      volumeHeight: 1.0,
    },
  },
  {
    id: 4,
    name: 'Wood Combustion',
    params: {
      vorticity: 3.0,
      dissipation: 0.928,
      buoyancy: 2.3,
      drag: 0.03,
      emission: 2.2,
      scattering: 5.6,
      absorption: 13.5,
      smokeWeight: 1.1,
      plumeTurbulence: 1.8,
      smokeDissipation: 0.989,
      windX: -0.03,
      windZ: 0.04,
      turbFreq: 14.0,
      turbSpeed: 1.3,
      fuelEfficiency: 1.35,
      heatDiffusion: 0.08,
      stepQuality: 1.0,
      exposure: 1.0,
      gamma: 2.2,
      T_ignite: 0.18,
      T_burn: 0.55,
      burnRate: 7.5,
      fuelInject: 1.15,
      heatYield: 3.4,
      sootYieldFlame: 0.55,
      sootYieldSmolder: 1.1,
      hazeConvertRate: 0.0,
      T_hazeStart: 0.35,
      T_hazeFull: 0.75,
      anisotropyG: 0.82,
      smokeThickness: 1.0,
      smokeDarkness: 0.25,
      flameSharpness: 4.0,
      volumeHeight: 1.0,
    },
  },
  {
    id: 1,
    name: 'Candle',
    params: {
      vorticity: 2.6,
      dissipation: 0.952,
      buoyancy: 3.7,
      drag: 0.09,
      emission: 0.9,
      scattering: 1.8,
      absorption: 1.6,
      smokeWeight: 0.1,
      plumeTurbulence: 0.12,
      smokeDissipation: 0.992,
      windX: 0.0,
      windZ: 0.0,
      turbFreq: 38.0,
      turbSpeed: 0.15,
      fuelEfficiency: 0.55,
      heatDiffusion: 0.0,
      stepQuality: 1.35,
      exposure: 1.0,
      gamma: 2.2,
      T_ignite: 0.18,
      T_burn: 0.55,
      burnRate: 2.4,
      fuelInject: 0.55,
      heatYield: 3.4,
      sootYieldFlame: 0.55,
      sootYieldSmolder: 1.1,
      hazeConvertRate: 0.0,
      T_hazeStart: 0.35,
      T_hazeFull: 0.75,
      anisotropyG: 0.82,
      smokeThickness: 1.0,
      smokeDarkness: 0.25,
      flameSharpness: 4.0,
      volumeHeight: 1.0,
    },
  },
  {
    id: 2,
    name: 'Dual Source',
    params: {
      vorticity: 9.0,
      dissipation: 0.972,
      buoyancy: 5.2,
      drag: 0.028,
      emission: 2.1,
      scattering: 4.2,
      absorption: 3.2,
      smokeWeight: 1.1,
      plumeTurbulence: 0.65,
      smokeDissipation: 0.992,
      windX: 0.08,
      windZ: 0.08,
      turbFreq: 18.0,
      turbSpeed: 1.1,
      fuelEfficiency: 1.0,
      heatDiffusion: 0.04,
      stepQuality: 0.95,
      exposure: 1.0,
      gamma: 2.2,
      T_ignite: 0.18,
      T_burn: 0.55,
      burnRate: 5.6,
      fuelInject: 1.0,
      heatYield: 3.4,
      sootYieldFlame: 0.55,
      sootYieldSmolder: 1.1,
      hazeConvertRate: 0.0,
      T_hazeStart: 0.35,
      T_hazeFull: 0.75,
      anisotropyG: 0.82,
      smokeThickness: 1.0,
      smokeDarkness: 0.25,
      flameSharpness: 4.0,
      volumeHeight: 1.0,
    },
  },
  {
    id: 3,
    name: 'Firebending',
    params: {
      vorticity: 11.5,
      dissipation: 0.958,
      buoyancy: 4.6,
      drag: 0.012,
      emission: 3.1,
      scattering: 3.0,
      absorption: 1.3,
      smokeWeight: 0.35,
      plumeTurbulence: 1.2,
      smokeDissipation: 0.986,
      windX: 0.0,
      windZ: 0.0,
      turbFreq: 30.0,
      turbSpeed: 1.8,
      fuelEfficiency: 1.25,
      heatDiffusion: 0.01,
      stepQuality: 0.9,
      exposure: 1.0,
      gamma: 2.2,
      T_ignite: 0.18,
      T_burn: 0.55,
      burnRate: 6.6,
      fuelInject: 1.08,
      heatYield: 3.4,
      sootYieldFlame: 0.55,
      sootYieldSmolder: 1.1,
      hazeConvertRate: 0.0,
      T_hazeStart: 0.35,
      T_hazeFull: 0.75,
      anisotropyG: 0.82,
      smokeThickness: 1.0,
      smokeDarkness: 0.25,
      flameSharpness: 4.0,
      volumeHeight: 1.0,
    },
  },
  {
    id: 5,
    name: 'Gas Explosion',
    params: {
      vorticity: 18.0,
      dissipation: 0.948,
      buoyancy: 10.0,
      drag: 0.022,
      emission: 5.0,
      scattering: 3.8,
      absorption: 1.2,
      smokeWeight: -0.2,
      plumeTurbulence: 1.1,
      smokeDissipation: 0.988,
      windX: 0.0,
      windZ: 0.0,
      turbFreq: 10.0,
      turbSpeed: 0.9,
      fuelEfficiency: 2.2,
      heatDiffusion: 0.12,
      stepQuality: 0.9,
      exposure: 1.0,
      gamma: 2.2,
      T_ignite: 0.18,
      T_burn: 0.55,
      burnRate: 12.0,
      fuelInject: 1.9,
      heatYield: 3.4,
      sootYieldFlame: 0.55,
      sootYieldSmolder: 1.1,
      hazeConvertRate: 0.0,
      T_hazeStart: 0.35,
      T_hazeFull: 0.75,
      anisotropyG: 0.82,
      smokeThickness: 1.0,
      smokeDarkness: 0.25,
      flameSharpness: 4.0,
      volumeHeight: 1.0,
    },
  },
  {
    id: 6,
    name: 'Nuke',
    params: {
      vorticity: 22.0,
      dissipation: 0.986,
      buoyancy: 6.2,
      drag: 0.045,
      emission: 4.8,
      scattering: 6.5,
      absorption: 5.5,
      smokeWeight: 2.4,
      plumeTurbulence: 0.6,
      smokeDissipation: 0.995,
      windX: 0.02,
      windZ: -0.02,
      turbFreq: 9.0,
      turbSpeed: 0.6,
      fuelEfficiency: 2.8,
      heatDiffusion: 0.28,
      stepQuality: 1.0,
      exposure: 1.0,
      gamma: 2.2,
      T_ignite: 0.18,
      T_burn: 0.55,
      burnRate: 16.0,
      fuelInject: 2.4,
      heatYield: 3.4,
      sootYieldFlame: 0.55,
      sootYieldSmolder: 1.1,
      hazeConvertRate: 0.0,
      T_hazeStart: 0.35,
      T_hazeFull: 0.75,
      anisotropyG: 0.82,
      smokeThickness: 1.0,
      smokeDarkness: 0.25,
      flameSharpness: 4.0,
      volumeHeight: 1.0,
    },
  },
];

export const DEFAULT_TIME_STEP = 0.016;

export const FLOOR_LIGHTING_DEFAULTS: FloorLightingParams = {
  floorUvScale: 1.35,
  floorUvWarp: 1.2,
  floorBlendStrength: 0.85,
  floorNormalStrength: 1.05,
  floorMicroStrength: 0.55,
  floorSootDarkening: 1.25,
  floorSootRoughness: 1.1,
  floorCharStrength: 1.2,
  floorContactShadow: 1.0,
  floorSpecular: 1.1,
  floorFireBounce: 1.7,
  floorAmbient: 0.55,
  lightingFireIntensity: 1.75,
  lightingFireFalloff: 1.1,
  lightingFlicker: 0.2,
  lightingGlow: 1.35,
};

