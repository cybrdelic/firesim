export class ShaderContract {
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

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

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
    const voxelCount = dim * dim * dim;

    this.uniformBuffer = device.createBuffer({
      size: 288,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    this.uniformStaging = new ArrayBuffer(288);
    this.uniformView = new DataView(this.uniformStaging);

    const bufferUsage = (window as typeof window & { GPUBufferUsage: typeof GPUBufferUsage }).GPUBufferUsage;
    const storageUsage = bufferUsage.STORAGE | bufferUsage.COPY_DST | bufferUsage.COPY_SRC;

    this.densityA = device.createBuffer({ size: voxelCount * 4, usage: storageUsage });
    this.densityB = device.createBuffer({ size: voxelCount * 4, usage: storageUsage });
    this.fuelA = device.createBuffer({ size: voxelCount * 4, usage: storageUsage });
    this.fuelB = device.createBuffer({ size: voxelCount * 4, usage: storageUsage });
    this.sootA = device.createBuffer({ size: voxelCount * 4, usage: storageUsage });
    this.sootB = device.createBuffer({ size: voxelCount * 4, usage: storageUsage });

    this.velocityBufferSize = voxelCount * 16;
    this.velocityA = device.createBuffer({ size: this.velocityBufferSize, usage: storageUsage });
    this.velocityB = device.createBuffer({ size: this.velocityBufferSize, usage: storageUsage });
    this.velocityScratch = device.createBuffer({ size: this.velocityBufferSize, usage: storageUsage });
    this.divergence = device.createBuffer({ size: voxelCount * 4, usage: storageUsage });
    this.pressureA = device.createBuffer({ size: voxelCount * 4, usage: storageUsage });
    this.pressureB = device.createBuffer({ size: voxelCount * 4, usage: storageUsage });

    const macrocellSize = 8;
    this.macrocellsPerAxis = Math.ceil(dim / macrocellSize);
    const macrocellCount = this.macrocellsPerAxis ** 3;
    this.occupancy = device.createBuffer({ size: macrocellCount * 4, usage: storageUsage });
    device.queue.writeBuffer(this.occupancy, 0, new Uint32Array(macrocellCount));

    this.rayStepCounter = device.createBuffer({ size: 4, usage: storageUsage });

    const zeroF32 = new Float32Array(voxelCount);
    const initVec4 = new Float32Array(voxelCount * 4);
    for (let i = 0; i < voxelCount; i++) {
      initVec4[i * 4 + 3] = 1.0;
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
    params: Record<string, number>,
    camera: { pos: number[]; target: number[] },
    sceneType: number
  ) {
    const view = this.uniformView;
    view.setFloat32(0, now, true);
    view.setFloat32(4, this.dim, true);
    view.setFloat32(8, params.vorticity ?? 2, true);
    view.setFloat32(12, params.dissipation ?? 0.992, true);
    view.setFloat32(16, params.buoyancy ?? 1.0, true);
    view.setFloat32(20, params.drag ?? 0.02, true);
    view.setFloat32(24, params.emission ?? 4.0, true);
    view.setFloat32(28, params.exposure ?? 1.0, true);
    view.setFloat32(32, params.gamma ?? 2.2, true);
    view.setFloat32(36, params.scattering ?? 1.0, true);
    view.setFloat32(40, params.absorption ?? 1.0, true);
    view.setFloat32(44, params.smokeWeight ?? 1.0, true);
    view.setFloat32(48, camera.pos[0] ?? 0.5, true);
    view.setFloat32(52, camera.pos[1] ?? 0.5, true);
    view.setFloat32(56, camera.pos[2] ?? 2.0, true);
    view.setFloat32(60, camera.target[0] ?? 0.5, true);
    view.setFloat32(64, camera.target[1] ?? 0.5, true);
    view.setFloat32(68, camera.target[2] ?? 0.5, true);
    view.setFloat32(72, sceneType, true);
    view.setFloat32(76, params.plumeTurbulence ?? 0.0, true);
    view.setFloat32(80, params.smokeDissipation ?? 0.0, true);
    view.setFloat32(84, params.windX ?? 0.0, true);
    view.setFloat32(88, params.windZ ?? 0.0, true);
    view.setFloat32(92, params.turbFreq ?? 0.0, true);
    view.setFloat32(96, params.turbSpeed ?? 0.0, true);
    view.setFloat32(100, params.fuelEfficiency ?? 1.0, true);
    view.setFloat32(104, params.heatDiffusion ?? 0.0, true);
    view.setFloat32(108, params.stepQuality ?? 1.0, true);
    view.setFloat32(112, params.T_ignite ?? 0.18, true);
    view.setFloat32(116, params.T_burn ?? 0.55, true);
    view.setFloat32(120, params.burnRate ?? 6.5, true);
    view.setFloat32(124, params.fuelInject ?? 0.95, true);
    view.setFloat32(128, params.heatYield ?? 3.4, true);
    view.setFloat32(132, params.sootYieldFlame ?? 0.55, true);
    view.setFloat32(136, params.sootYieldSmolder ?? 1.1, true);
    view.setFloat32(140, params.hazeConvertRate ?? 0.0, true);
    view.setFloat32(144, params.T_hazeStart ?? 0.35, true);
    view.setFloat32(148, params.T_hazeFull ?? 0.75, true);
    view.setFloat32(152, params.anisotropyG ?? 0.82, true);
    view.setFloat32(156, params.smokeThickness ?? 1.0, true);
    view.setFloat32(160, params.smokeDarkness ?? 0.25, true);
    view.setFloat32(164, params.flameSharpness ?? 4.0, true);
    view.setFloat32(168, params.volumeHeight ?? 1.0, true);
    view.setFloat32(172, params.floorUvScale ?? 1.35, true);
    view.setFloat32(176, params.floorUvWarp ?? 1.2, true);
    view.setFloat32(180, params.floorBlendStrength ?? 0.85, true);
    view.setFloat32(184, params.floorNormalStrength ?? 1.05, true);
    view.setFloat32(188, params.floorMicroStrength ?? 0.55, true);
    view.setFloat32(192, params.floorSootDarkening ?? 1.25, true);
    view.setFloat32(196, params.floorSootRoughness ?? 1.1, true);
    view.setFloat32(200, params.floorCharStrength ?? 1.2, true);
    view.setFloat32(204, params.floorContactShadow ?? 1.0, true);
    view.setFloat32(208, params.floorSpecular ?? 1.1, true);
    view.setFloat32(212, params.floorFireBounce ?? 1.7, true);
    view.setFloat32(216, params.floorAmbient ?? 0.55, true);
    view.setFloat32(220, params.lightingFireIntensity ?? 1.75, true);
    view.setFloat32(224, params.lightingFireFalloff ?? 1.1, true);
    view.setFloat32(228, params.lightingFlicker ?? 0.2, true);
    view.setFloat32(232, params.lightingGlow ?? 1.35, true);

    const occlusionModeMap: Record<string, number> = {
      analytic_sdf: 0,
      depth_coupled: 1,
      none: 2,
    };
    const overlayModeMap: Record<string, number> = {
      final: 0,
      alpha: 1,
      occlusion: 2,
      wood_sdf: 3,
      combustion: 4,
    };

    view.setFloat32(236, occlusionModeMap[String(params.fireOcclusionMode ?? 'analytic_sdf')] ?? 0, true);
    view.setFloat32(240, overlayModeMap[String(params.debugOverlayMode ?? 'final')] ?? 0, true);

    const renderWidth = clamp(Number(params.renderWidth ?? 0), 0, 16384);
    const renderHeight = clamp(Number(params.renderHeight ?? 0), 0, 16384);
    view.setFloat32(244, renderWidth, true);
    view.setFloat32(248, renderHeight, true);
    view.setFloat32(252, clamp(Number(params.rayStepBudget ?? 0), 0, 4096), true);
    view.setFloat32(256, clamp(Number(params.occlusionStepBudget ?? 0), 0, 1024), true);
    view.setFloat32(260, clamp(Number(params.worldViewProjReady ?? 0), 0, 1), true);

    const worldViewProj = Array.isArray(params.worldViewProj) ? params.worldViewProj : [];
    for (let i = 0; i < 4; i++) {
      view.setFloat32(264 + i * 4, Number(worldViewProj[i] ?? (i === 0 ? 1 : 0)), true);
    }
    for (let i = 0; i < 2; i++) {
      view.setFloat32(280 + i * 4, Number(params[`worldViewProjRow1_${i}`] ?? (i === 1 ? 1 : 0)), true);
    }

    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformStaging);
  }
}
