
import {
    AlertTriangle,
    Camera,
    ChevronDown,
    Copy,
    Droplet,
    Eye,
    Flame,
    Gauge,
    Keyboard,
    Lock,
    LockOpen,
    Pause,
    Play,
    RefreshCw,
    Settings,
    Shuffle,
    SkipForward,
    Thermometer,
    Users,
    Wind,
    Zap
} from 'lucide-react';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

/**
 * SECTION 1: TRANSPORT LAYER (3D)
 */

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

interface FloorMaterialTextures {
  albedoView: GPUTextureView;
  roughnessView: GPUTextureView;
  normalView: GPUTextureView;
  sampler: GPUSampler;
}

class FluidTransport {
  public uniformBuffer: GPUBuffer;

  public densityA: GPUBuffer;
  public densityB: GPUBuffer;
  public fuelA: GPUBuffer;
  public fuelB: GPUBuffer;
  public velocityA: GPUBuffer;
  public velocityB: GPUBuffer;

  public sootFloorSize = 256;
  public sootFloorTextures: GPUTexture[] = [];
  public sootFloorViews: GPUTextureView[] = [];

  public radianceDim = 40;
  public radianceA: GPUBuffer;
  public radianceB: GPUBuffer;

  public physicsContract: ShaderContract;
  public renderContract: ShaderContract;
  public floorContract: ShaderContract;
  public radianceInjectContract: ShaderContract;
  public radiancePropContract: ShaderContract;

  public physicsGroups: GPUBindGroup[] = [];
  public renderGroups: GPUBindGroup[][] = [[], []];
  public floorGroups: GPUBindGroup[][] = [[], []];
  public radianceInjectGroups: GPUBindGroup[] = [];
  public radiancePropGroup: GPUBindGroup;

  constructor(private device: GPUDevice, public dim: number, private floorMaterial: FloorMaterialTextures) {
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

    const radianceVoxelCount = this.radianceDim * this.radianceDim * this.radianceDim;
    const radianceBufferSize = radianceVoxelCount * 16;
    this.radianceA = device.createBuffer({ size: radianceBufferSize, usage: storageUsage });
    this.radianceB = device.createBuffer({ size: radianceBufferSize, usage: storageUsage });

    const zeroF32 = new Float32Array(VOXEL_COUNT);
    const zeroVec4 = new Float32Array(VOXEL_COUNT * 4);
    const zeroRadiance = new Float32Array(radianceVoxelCount * 4);

    device.queue.writeBuffer(this.densityA, 0, zeroF32);
    device.queue.writeBuffer(this.densityB, 0, zeroF32);
    device.queue.writeBuffer(this.fuelA, 0, zeroF32);
    device.queue.writeBuffer(this.fuelB, 0, zeroF32);
    device.queue.writeBuffer(this.velocityA, 0, zeroVec4);
    device.queue.writeBuffer(this.velocityB, 0, zeroVec4);
    device.queue.writeBuffer(this.radianceA, 0, zeroRadiance);
    device.queue.writeBuffer(this.radianceB, 0, zeroRadiance);

    const textureUsage = (window as any).GPUTextureUsage;
    const floorTextureUsage = textureUsage.TEXTURE_BINDING | textureUsage.STORAGE_BINDING | textureUsage.COPY_DST;
    for (let i = 0; i < 2; i++) {
      const texture = (device as any).createTexture({
        size: [this.sootFloorSize, this.sootFloorSize],
        format: 'r32float',
        usage: floorTextureUsage,
        label: `SootFloor_${i}`
      });
      this.sootFloorTextures.push(texture);
      this.sootFloorViews.push(texture.createView());
    }

    const zeroFloor = new Float32Array(this.sootFloorSize * this.sootFloorSize);
    for (const texture of this.sootFloorTextures) {
      (device.queue as any).writeTexture(
        { texture },
        zeroFloor,
        { bytesPerRow: this.sootFloorSize * 4, rowsPerImage: this.sootFloorSize },
        { width: this.sootFloorSize, height: this.sootFloorSize }
      );
    }

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
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
      { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
      { binding: 7, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
      { binding: 8, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      { binding: 9, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    ]);

    this.floorContract = new ShaderContract(this.device, 'FloorSoot', [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '2d' } },
    ]);

    this.radianceInjectContract = new ShaderContract(this.device, 'RadianceInject', [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ]);

    this.radiancePropContract = new ShaderContract(this.device, 'RadianceProp', [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
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

    this.renderGroups[0][0] = this.renderContract.createBindGroup(this.device, 'Render0Soot0', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.densityB } },
      { binding: 2, resource: { buffer: this.velocityB } },
      { binding: 3, resource: { buffer: this.fuelB } },
      { binding: 4, resource: this.sootFloorViews[0] },
      { binding: 5, resource: this.floorMaterial.albedoView },
      { binding: 6, resource: this.floorMaterial.roughnessView },
      { binding: 7, resource: this.floorMaterial.normalView },
      { binding: 8, resource: this.floorMaterial.sampler },
      { binding: 9, resource: { buffer: this.radianceA } },
    ]);

    this.renderGroups[0][1] = this.renderContract.createBindGroup(this.device, 'Render0Soot1', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.densityB } },
      { binding: 2, resource: { buffer: this.velocityB } },
      { binding: 3, resource: { buffer: this.fuelB } },
      { binding: 4, resource: this.sootFloorViews[1] },
      { binding: 5, resource: this.floorMaterial.albedoView },
      { binding: 6, resource: this.floorMaterial.roughnessView },
      { binding: 7, resource: this.floorMaterial.normalView },
      { binding: 8, resource: this.floorMaterial.sampler },
      { binding: 9, resource: { buffer: this.radianceA } },
    ]);

    this.floorGroups[0][0] = this.floorContract.createBindGroup(this.device, 'Floor0Soot0', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.densityB } },
      { binding: 2, resource: { buffer: this.velocityB } },
      { binding: 3, resource: this.sootFloorViews[0] },
      { binding: 4, resource: this.sootFloorViews[1] },
    ]);

    this.floorGroups[0][1] = this.floorContract.createBindGroup(this.device, 'Floor0Soot1', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.densityB } },
      { binding: 2, resource: { buffer: this.velocityB } },
      { binding: 3, resource: this.sootFloorViews[1] },
      { binding: 4, resource: this.sootFloorViews[0] },
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

    this.renderGroups[1][0] = this.renderContract.createBindGroup(this.device, 'Render1Soot0', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.densityA } },
      { binding: 2, resource: { buffer: this.velocityA } },
      { binding: 3, resource: { buffer: this.fuelA } },
      { binding: 4, resource: this.sootFloorViews[0] },
      { binding: 5, resource: this.floorMaterial.albedoView },
      { binding: 6, resource: this.floorMaterial.roughnessView },
      { binding: 7, resource: this.floorMaterial.normalView },
      { binding: 8, resource: this.floorMaterial.sampler },
      { binding: 9, resource: { buffer: this.radianceA } },
    ]);

    this.renderGroups[1][1] = this.renderContract.createBindGroup(this.device, 'Render1Soot1', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.densityA } },
      { binding: 2, resource: { buffer: this.velocityA } },
      { binding: 3, resource: { buffer: this.fuelA } },
      { binding: 4, resource: this.sootFloorViews[1] },
      { binding: 5, resource: this.floorMaterial.albedoView },
      { binding: 6, resource: this.floorMaterial.roughnessView },
      { binding: 7, resource: this.floorMaterial.normalView },
      { binding: 8, resource: this.floorMaterial.sampler },
      { binding: 9, resource: { buffer: this.radianceA } },
    ]);

    this.floorGroups[1][0] = this.floorContract.createBindGroup(this.device, 'Floor1Soot0', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.densityA } },
      { binding: 2, resource: { buffer: this.velocityA } },
      { binding: 3, resource: this.sootFloorViews[0] },
      { binding: 4, resource: this.sootFloorViews[1] },
    ]);

    this.floorGroups[1][1] = this.floorContract.createBindGroup(this.device, 'Floor1Soot1', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.densityA } },
      { binding: 2, resource: { buffer: this.velocityA } },
      { binding: 3, resource: this.sootFloorViews[1] },
      { binding: 4, resource: this.sootFloorViews[0] },
    ]);

    this.radianceInjectGroups[0] = this.radianceInjectContract.createBindGroup(this.device, 'RadianceInject0', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.densityB } },
      { binding: 2, resource: { buffer: this.velocityB } },
      { binding: 3, resource: { buffer: this.fuelB } },
      { binding: 4, resource: { buffer: this.radianceA } },
      { binding: 5, resource: { buffer: this.radianceB } },
    ]);

    this.radianceInjectGroups[1] = this.radianceInjectContract.createBindGroup(this.device, 'RadianceInject1', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.densityA } },
      { binding: 2, resource: { buffer: this.velocityA } },
      { binding: 3, resource: { buffer: this.fuelA } },
      { binding: 4, resource: { buffer: this.radianceA } },
      { binding: 5, resource: { buffer: this.radianceB } },
    ]);

    this.radiancePropGroup = this.radiancePropContract.createBindGroup(this.device, 'RadianceProp', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.radianceB } },
      { binding: 2, resource: { buffer: this.radianceA } },
    ]);
  }

  public updateUniforms(
    now: number,
    params: any,
    camera: { pos: number[], target: number[] },
    sceneType: number
  ) {
    const uniformData = new ArrayBuffer(256);
    const view = new DataView(uniformData);

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
    view.setFloat32(248, params.lightingFlicker ?? 0.7, true);
    view.setFloat32(252, params.lightingGlow ?? 1.35, true);

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

const COMPUTE_SHADER = `
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

  // Cheap oxygen proxy (no separate oxygen field yet): soot self-shades combustion.
  // Once soot is generated from reaction, this hack becomes self-consistent enough.
  let localOxygen = smoothstep(1.2, 0.2, soot);
  let insulation_factor = smoothstep(0.12, 0.0, woodDist);
  let insulation = mix(params.dissipation, 1.0, insulation_factor * 0.88);
  temp *= insulation;

  // Species persistence (soot lingers; fuel only decreases via reaction)
  soot *= clamp(params.smokeDissipation, 0.0, 1.0);

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
        // Fuel injection at the wood surface (replaces "always on" heat sources)
        let inject = params.fuelInject * logSurfaceZone * (0.55 + 0.45 * crack_mask) * dt;
        fuel = clamp(fuel + inject, 0.0, 1.0);

        let fuelAvailability = logSurfaceZone * fuel;
        let combustionShape = smoothstep(-0.2, 0.8, n_val);

      // Pilot heat: lets ignition bootstrap from a cold start.
      temp += params.emission * combustionShape * logSurfaceZone * dt * 0.22;
        let ignite = smoothstep(params.T_ignite, params.T_burn, temp);
        let combustionIntensity = combustionShape * fuelAvailability * localOxygen * ignite;

        // Reaction proxy R: drives heat, soot generation, and fuel consumption
        let R = combustionIntensity * params.burnRate;
        fuel = max(0.0, fuel - R * dt);

        temp += R * params.heatYield * dt;

        // Soot yield: higher when oxygen is low and temperature is moderate.
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

  // Boundary damping: keep velocity stable near floor + side walls,
  // but do NOT erase soot/fuel at the floor (it is born near y≈0).
  let b_dist_xz = min(min(uvw.x, 1.0 - uvw.x), min(uvw.z, 1.0 - uvw.z));
  let floor_dist = uvw.y;

  // Velocity: damp near all boundaries.
  let damp_vel = smoothstep(0.0, 0.02, min(b_dist_xz, floor_dist));
  newVel *= damp_vel;

  // Temperature: mild damping near floor + walls (avoid sticky heat at boundaries).
  let damp_temp = smoothstep(0.0, 0.01, floor_dist) * smoothstep(0.0, 0.02, b_dist_xz);
  temp *= damp_temp;

  // Species: only damp near side walls.
  let damp_species = smoothstep(0.0, 0.02, b_dist_xz);
  soot *= damp_species;
  fuel *= damp_species;

  densityOut[idx] = temp;
  velocityOut[idx] = vec4f(clamp(newVel, vec3f(-120.0), vec3f(120.0)), soot);
  fuelOut[idx] = fuel;
}
`;

const SOOT_FLOOR_UPDATE_SHADER = `
${STRUCT_DEF}
@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> densityIn: array<f32>;
@group(0) @binding(2) var<storage, read> velocityIn: array<vec4f>;
@group(0) @binding(3) var sootFloorIn: texture_2d<f32>;
@group(0) @binding(4) var sootFloorOut: texture_storage_2d<r32float, write>;

fn get_idx(p: vec3i) -> u32 {
  let d = i32(params.dim);
  let cp = clamp(p, vec3i(0), vec3i(d - 1));
  return u32(cp.z * d * d + cp.y * d + cp.x);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let texSize = textureDimensions(sootFloorOut);
  if (gid.x >= texSize.x || gid.y >= texSize.y) { return; }

  let uv = (vec2f(gid.xy) + 0.5) / vec2f(texSize);
  let x = i32(clamp(uv.x * params.dim, 0.0, params.dim - 1.0));
  let z = i32(clamp(uv.y * params.dim, 0.0, params.dim - 1.0));

  let yMax = max(0.03, min(0.12, params.volumeHeight * 0.12));
  let ySamples = 10;
  var sootSum = 0.0;
  var wSum = 0.0;

  for (var i = 0; i < ySamples; i++) {
    let fy = (f32(i) + 0.5) / f32(ySamples);
    let yUv = fy * yMax;
    let y = i32(clamp(yUv * params.dim, 0.0, params.dim - 1.0));
    let idx = get_idx(vec3i(x, y, z));

    let temp = max(0.0, densityIn[idx]);
    let soot = max(0.0, velocityIn[idx].w);
    let vy = velocityIn[idx].y;

    let coolGate = smoothstep(0.8, 0.12, temp);
    let settleGate = smoothstep(1.0, -0.15, vy);
    let nearFloor = 1.0 - fy;
    let w = coolGate * settleGate * nearFloor;
    let thermalResidue = max(0.0, temp - params.T_ignite) * 0.35;
    let source = soot + thermalResidue;

    sootSum += source * w;
    wSum += w;
  }

  let sootSlab = sootSum / max(1e-5, wSum);
  let coord = vec2i(gid.xy);
  let prev = textureLoad(sootFloorIn, coord, 0).x;

  let depositRate = 2.4;
  let decayRate = 0.0005;
  let add = sootSlab * depositRate * params.dt;
  let next = clamp(prev * (1.0 - decayRate) + add, 0.0, 1.0);
  textureStore(sootFloorOut, coord, vec4f(next, 0.0, 0.0, 0.0));
}
`;

const RADIANCE_CACHE_INJECT_SHADER = `
${STRUCT_DEF}
@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> densityIn: array<f32>;
@group(0) @binding(2) var<storage, read> velocityIn: array<vec4f>;
@group(0) @binding(3) var<storage, read> fuelIn: array<f32>;
@group(0) @binding(4) var<storage, read> radiancePrev: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> radianceOut: array<vec4f>;

const RAD_DIM: i32 = 40;
const RAD_DIM_U: u32 = 40u;

fn sim_idx(p: vec3i) -> u32 {
  let d = i32(params.dim);
  let cp = clamp(p, vec3i(0), vec3i(d - 1));
  return u32(cp.z * d * d + cp.y * d + cp.x);
}

fn sample_volume(uv: vec3f) -> vec3f {
  let d = params.dim;
  let p = uv * d - 0.5;
  let i = vec3i(floor(p));
  let f = fract(p);
  let temp = mix(
    mix(mix(densityIn[sim_idx(i)], densityIn[sim_idx(i + vec3i(1, 0, 0))], f.x), mix(densityIn[sim_idx(i + vec3i(0, 1, 0))], densityIn[sim_idx(i + vec3i(1, 1, 0))], f.x), f.y),
    mix(mix(densityIn[sim_idx(i + vec3i(0, 0, 1))], densityIn[sim_idx(i + vec3i(1, 0, 1))], f.x), mix(densityIn[sim_idx(i + vec3i(0, 1, 1))], densityIn[sim_idx(i + vec3i(1, 1, 1))], f.x), f.y),
    f.z
  );
  let soot = mix(
    mix(mix(velocityIn[sim_idx(i)].w, velocityIn[sim_idx(i + vec3i(1, 0, 0))].w, f.x), mix(velocityIn[sim_idx(i + vec3i(0, 1, 0))].w, velocityIn[sim_idx(i + vec3i(1, 1, 0))].w, f.x), f.y),
    mix(mix(velocityIn[sim_idx(i + vec3i(0, 0, 1))].w, velocityIn[sim_idx(i + vec3i(1, 0, 1))].w, f.x), mix(velocityIn[sim_idx(i + vec3i(0, 1, 1))].w, velocityIn[sim_idx(i + vec3i(1, 1, 1))].w, f.x), f.y),
    f.z
  );
  let fuel = mix(
    mix(mix(fuelIn[sim_idx(i)], fuelIn[sim_idx(i + vec3i(1, 0, 0))], f.x), mix(fuelIn[sim_idx(i + vec3i(0, 1, 0))], fuelIn[sim_idx(i + vec3i(1, 1, 0))], f.x), f.y),
    mix(mix(fuelIn[sim_idx(i + vec3i(0, 0, 1))], fuelIn[sim_idx(i + vec3i(1, 0, 1))], f.x), mix(fuelIn[sim_idx(i + vec3i(0, 1, 1))], fuelIn[sim_idx(i + vec3i(1, 1, 1))], f.x), f.y),
    f.z
  );
  return vec3f(temp, soot, fuel);
}

fn blackbody_fast(temp: f32) -> vec3f {
  let t = max(0.0, temp - 0.12);
  if (t < 0.3) { return mix(vec3f(0.0), vec3f(2.0, 0.06, 0.004), t / 0.3); }
  if (t < 0.75) { return mix(vec3f(2.0, 0.06, 0.004), vec3f(5.0, 1.7, 0.2), (t - 0.3) / 0.45); }
  if (t < 1.4) { return mix(vec3f(5.0, 1.7, 0.2), vec3f(10.0, 7.0, 1.6), (t - 0.75) / 0.65); }
  return mix(vec3f(10.0, 7.0, 1.6), vec3f(22.0, 20.0, 16.0), clamp((t - 1.4) * 0.4, 0.0, 1.0));
}

fn rad_idx(p: vec3i) -> u32 {
  let cp = clamp(p, vec3i(0), vec3i(RAD_DIM - 1));
  return u32(cp.z * RAD_DIM * RAD_DIM + cp.y * RAD_DIM + cp.x);
}

fn rad_bounds_min() -> vec3f {
  return vec3f(-0.7, 0.0, -0.7);
}

fn rad_bounds_max() -> vec3f {
  return vec3f(1.7, max(1.25, params.volumeHeight * 1.95), 1.7);
}

fn inside_sim_world(p: vec3f) -> bool {
  return p.x >= 0.0 && p.x <= 1.0 && p.z >= 0.0 && p.z <= 1.0 && p.y >= 0.0 && p.y <= params.volumeHeight;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= RAD_DIM_U || gid.y >= RAD_DIM_U || gid.z >= RAD_DIM_U) { return; }

  let cell = vec3i(gid.xyz);
  let idx = rad_idx(cell);
  let uv = (vec3f(gid.xyz) + 0.5) / f32(RAD_DIM);
  let bmin = rad_bounds_min();
  let bmax = rad_bounds_max();
  let worldPos = mix(bmin, bmax, uv);

  var emit = vec3f(0.0);
  var absorb = 0.0;

  if (inside_sim_world(worldPos)) {
    let simUv = vec3f(worldPos.x, clamp(worldPos.y / max(1e-4, params.volumeHeight), 0.0, 1.0), worldPos.z);
    let s = sample_volume(simUv);
    let temp = max(0.0, s.x);
    let soot = max(0.0, s.y);
    let fuel = max(0.0, s.z);
    let reaction = smoothstep(params.T_ignite, params.T_burn, temp) * (0.2 + 0.8 * smoothstep(0.0, 0.08, fuel));
    emit = blackbody_fast(temp) * params.emission * reaction * 0.045;
    absorb = soot * 0.04;
  }

  let prev = radiancePrev[idx].xyz;
  let history = prev * max(0.0, 0.962 - absorb * 0.15);
  let next = max(vec3f(0.0), history + emit * params.dt * 60.0);
  radianceOut[idx] = vec4f(next, 1.0);
}
`;

const RADIANCE_CACHE_PROPAGATE_SHADER = `
${STRUCT_DEF}
@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> radianceIn: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> radianceOut: array<vec4f>;

const RAD_DIM: i32 = 40;
const RAD_DIM_U: u32 = 40u;

fn rad_idx(p: vec3i) -> u32 {
  let cp = clamp(p, vec3i(0), vec3i(RAD_DIM - 1));
  return u32(cp.z * RAD_DIM * RAD_DIM + cp.y * RAD_DIM + cp.x);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= RAD_DIM_U || gid.y >= RAD_DIM_U || gid.z >= RAD_DIM_U) { return; }

  let c = vec3i(gid.xyz);
  let center = radianceIn[rad_idx(c)].xyz;
  var accum = center * 0.38;
  accum += radianceIn[rad_idx(c + vec3i(1, 0, 0))].xyz * 0.103333;
  accum += radianceIn[rad_idx(c + vec3i(-1, 0, 0))].xyz * 0.103333;
  accum += radianceIn[rad_idx(c + vec3i(0, 1, 0))].xyz * 0.103333;
  accum += radianceIn[rad_idx(c + vec3i(0, -1, 0))].xyz * 0.103333;
  accum += radianceIn[rad_idx(c + vec3i(0, 0, 1))].xyz * 0.103333;
  accum += radianceIn[rad_idx(c + vec3i(0, 0, -1))].xyz * 0.103333;
  let next = max(vec3f(0.0), accum * 0.988);
  radianceOut[rad_idx(c)] = vec4f(next, 1.0);
}
`;

const RENDER_SHADER = `
${STRUCT_DEF}
${WOOD_SDF_FN}
@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> densityIn: array<f32>;
@group(0) @binding(2) var<storage, read> velocityIn: array<vec4f>;
@group(0) @binding(3) var<storage, read> fuelIn: array<f32>;
@group(0) @binding(4) var sootFloorTex: texture_2d<f32>;
@group(0) @binding(5) var floorAlbedoTex: texture_2d<f32>;
@group(0) @binding(6) var floorRoughnessTex: texture_2d<f32>;
@group(0) @binding(7) var floorNormalTex: texture_2d<f32>;
@group(0) @binding(8) var floorSampler: sampler;
@group(0) @binding(9) var<storage, read> radianceCache: array<vec4f>;

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

fn volume_edge_falloff_uv(uv: vec3f) -> f32 {
  let ex = min(uv.x, 1.0 - uv.x);
  let ey = min(uv.y, 1.0 - uv.y);
  let ez = min(uv.z, 1.0 - uv.z);
  let edge = min(min(ex, ey), ez);
  return smoothstep(0.0, 0.065, edge);
}

fn sample_volume(pos: vec3f) -> vec3f {
  let d = params.dim; let p = pos * d - 0.5; let i = vec3i(floor(p)); let f = fract(p);
  let f_res = mix(mix(mix(densityIn[get_idx(i)], densityIn[get_idx(i + vec3i(1,0,0))], f.x), mix(densityIn[get_idx(i + vec3i(0,1,0))], densityIn[get_idx(i + vec3i(1,1,0))], f.x), f.y), mix(mix(densityIn[get_idx(i + vec3i(0,0,1))], densityIn[get_idx(i + vec3i(1,0,1))], f.x), mix(densityIn[get_idx(i + vec3i(0,1,1))], densityIn[get_idx(i + vec3i(1,1,1))], f.x), f.y), f.z);
  let s_res = mix(mix(mix(velocityIn[get_idx(i)].w, velocityIn[get_idx(i + vec3i(1,0,0))].w, f.x), mix(velocityIn[get_idx(i + vec3i(0,1,0))].w, velocityIn[get_idx(i + vec3i(1,1,0))].w, f.x), f.y), mix(mix(velocityIn[get_idx(i + vec3i(0,0,1))].w, velocityIn[get_idx(i + vec3i(1,0,1))].w, f.x), mix(velocityIn[get_idx(i + vec3i(0,1,1))].w, velocityIn[get_idx(i + vec3i(1,1,1))].w, f.x), f.y), f.z);
  let fu_res = mix(mix(mix(fuelIn[get_idx(i)], fuelIn[get_idx(i + vec3i(1,0,0))], f.x), mix(fuelIn[get_idx(i + vec3i(0,1,0))], fuelIn[get_idx(i + vec3i(1,1,0))], f.x), f.y), mix(mix(fuelIn[get_idx(i + vec3i(0,0,1))], fuelIn[get_idx(i + vec3i(1,0,1))], f.x), mix(fuelIn[get_idx(i + vec3i(0,1,1))], fuelIn[get_idx(i + vec3i(1,1,1))], f.x), f.y), f.z);
  return vec3f(f_res, s_res, fu_res);
}

fn sample_medium_world(p: vec3f) -> vec4f {
  if (!inside_volume_world(p)) { return vec4f(0.0); }
  let uv = to_volume_uv(p);
  let falloff = volume_edge_falloff_uv(uv);
  let v = sample_volume(uv) * falloff;
  return vec4f(v, falloff);
}

fn sample_velocity_nearest(pos: vec3f) -> vec3f {
  // Nearest-voxel velocity fetch (much cheaper than tri-linear).
  // Good enough for flow-locked micro detail.
  let d = params.dim;
  let p = pos * d - 0.5;
  let i = vec3i(floor(p));
  return velocityIn[get_idx(i)].xyz;
}

const RAD_DIM: i32 = 40;

fn rad_idx(p: vec3i) -> u32 {
  let cp = clamp(p, vec3i(0), vec3i(RAD_DIM - 1));
  return u32(cp.z * RAD_DIM * RAD_DIM + cp.y * RAD_DIM + cp.x);
}

fn rad_bounds_min() -> vec3f {
  return vec3f(-0.7, 0.0, -0.7);
}

fn rad_bounds_max() -> vec3f {
  return vec3f(1.7, max(1.25, params.volumeHeight * 1.95), 1.7);
}

fn sample_radiance_cache(worldPos: vec3f) -> vec3f {
  let bmin = rad_bounds_min();
  let bmax = rad_bounds_max();
  let ext = max(vec3f(1e-4), bmax - bmin);
  let uv = (worldPos - bmin) / ext;
  if (any(uv < vec3f(0.0)) || any(uv > vec3f(1.0))) { return vec3f(0.0); }

  let p = uv * f32(RAD_DIM) - 0.5;
  let i = vec3i(floor(p));
  let f = fract(p);

  let c000 = radianceCache[rad_idx(i)].xyz;
  let c100 = radianceCache[rad_idx(i + vec3i(1, 0, 0))].xyz;
  let c010 = radianceCache[rad_idx(i + vec3i(0, 1, 0))].xyz;
  let c110 = radianceCache[rad_idx(i + vec3i(1, 1, 0))].xyz;
  let c001 = radianceCache[rad_idx(i + vec3i(0, 0, 1))].xyz;
  let c101 = radianceCache[rad_idx(i + vec3i(1, 0, 1))].xyz;
  let c011 = radianceCache[rad_idx(i + vec3i(0, 1, 1))].xyz;
  let c111 = radianceCache[rad_idx(i + vec3i(1, 1, 1))].xyz;

  return mix(
    mix(mix(c000, c100, f.x), mix(c010, c110, f.x), f.y),
    mix(mix(c001, c101, f.x), mix(c011, c111, f.x), f.y),
    f.z
  );
}

fn sample_soot_floor(uv: vec2f) -> f32 {
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    return 0.0;
  }
  let dimsI = vec2i(textureDimensions(sootFloorTex));
  let dims = vec2f(dimsI);
  let p = clamp(uv, vec2f(0.0), vec2f(0.9999)) * dims - 0.5;
  let i = vec2i(floor(p));
  let f = fract(p);
  let maxI = dimsI - vec2i(1, 1);
  let i00 = clamp(i, vec2i(0, 0), maxI);
  let i10 = clamp(i + vec2i(1, 0), vec2i(0, 0), maxI);
  let i01 = clamp(i + vec2i(0, 1), vec2i(0, 0), maxI);
  let i11 = clamp(i + vec2i(1, 1), vec2i(0, 0), maxI);
  let s00 = textureLoad(sootFloorTex, i00, 0).x;
  let s10 = textureLoad(sootFloorTex, i10, 0).x;
  let s01 = textureLoad(sootFloorTex, i01, 0).x;
  let s11 = textureLoad(sootFloorTex, i11, 0).x;
  return mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
}

fn sample_tex2d(tex: texture_2d<f32>, uv: vec2f) -> vec4f {
  return textureSampleLevel(tex, floorSampler, uv, 0.0);
}

fn sample_floor_albedo(uv: vec2f) -> vec3f {
  return sample_tex2d(floorAlbedoTex, uv).rgb;
}

fn sample_floor_roughness(uv: vec2f) -> f32 {
  return sample_tex2d(floorRoughnessTex, uv).r;
}

fn sample_floor_normal(uv: vec2f) -> vec3f {
  let t = sample_tex2d(floorNormalTex, uv).xyz * 2.0 - 1.0;
  // Floor tangent frame: +x tangent, +z bitangent, +y normal.
  return normalize(vec3f(t.x, t.z, max(0.05, t.y)));
}

fn floor_micro_height(uv: vec2f, sootDep: f32) -> f32 {
  let n0 = cheap_noise(vec3f(uv * 22.0, 1.7));
  let n1 = cheap_noise(vec3f(uv * 63.0 + vec2f(13.0, 7.0), 5.1));
  let n2 = cheap_noise(vec3f(uv * 141.0 + vec2f(4.0, 27.0), 9.7));
  let amp = mix(0.0013, 0.0022, clamp(sootDep, 0.0, 1.0));
  return (n0 * 0.45 + n1 * 0.35 + n2 * 0.2) * amp;
}

fn floor_micro_normal(uv: vec2f, sootDep: f32) -> vec3f {
  let e = 0.0018;
  let hL = floor_micro_height(uv - vec2f(e, 0.0), sootDep);
  let hR = floor_micro_height(uv + vec2f(e, 0.0), sootDep);
  let hD = floor_micro_height(uv - vec2f(0.0, e), sootDep);
  let hU = floor_micro_height(uv + vec2f(0.0, e), sootDep);
  let dx = (hR - hL) / (2.0 * e);
  let dz = (hU - hD) / (2.0 * e);
  return normalize(vec3f(-dx, 1.0, -dz));
}

fn fresnel_schlick(cosTheta: f32, f0: vec3f) -> vec3f {
  let m = clamp(1.0 - cosTheta, 0.0, 1.0);
  let m2 = m * m;
  let m5 = m2 * m2 * m;
  return f0 + (vec3f(1.0) - f0) * m5;
}

fn hash_vec3(p: vec3f) -> vec3f {
  // Cheap hash (no trig) in [-1, 1]
  var p3 = fract(p * vec3f(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yxz + 33.33);
  return fract((p3.xxy + p3.yxx) * p3.zyx) * 2.0 - 1.0;
}

fn cheap_noise(p: vec3f) -> f32 {
  return dot(hash_vec3(p), vec3f(0.3333333));
}

fn rotate_uv(uv: vec2f, angle: f32) -> vec2f {
  let c = cos(angle);
  let s = sin(angle);
  return vec2f(c * uv.x - s * uv.y, s * uv.x + c * uv.y);
}

fn rotate_floor_tangent_normal(n: vec3f, angle: f32) -> vec3f {
  let c = cos(angle);
  let s = sin(angle);
  return normalize(vec3f(n.x * c - n.z * s, n.y, n.x * s + n.z * c));
}

fn floor_uv_warp(uv: vec2f) -> vec2f {
  let w0 = cheap_noise(vec3f(uv * 0.43 + vec2f(1.2, -0.7), 2.3));
  let w1 = cheap_noise(vec3f(uv * 0.36 + vec2f(-2.7, 3.9), 5.1));
  let warpStrength = clamp(params.floorUvWarp, 0.0, 3.0);
  let warp = vec2f(w0, w1) * (0.09 * warpStrength);
  return uv + warp;
}

fn sample_floor_albedo_layered(uv: vec2f) -> vec3f {
  let uvw = floor_uv_warp(uv);
  let floorScale = max(0.2, params.floorUvScale);
  let uv0 = rotate_uv(uvw * (1.42 * floorScale) + vec2f(0.17, -0.13), 0.31);
  let uv1 = rotate_uv(uvw * (2.31 * floorScale) + vec2f(0.37, 0.11), 0.97);
  let blendNoiseA = cheap_noise(vec3f(uvw * 0.77 + vec2f(3.2, -1.7), 2.4)) * 0.5 + 0.5;
  let blendNoiseB = cheap_noise(vec3f(uvw * 1.13 + vec2f(-4.1, 2.6), 4.2)) * 0.5 + 0.5;
  let blendStrength = clamp(params.floorBlendStrength, 0.0, 1.5);
  let blend = clamp(0.5 + (blendNoiseA - 0.5) * 0.46 * blendStrength + (blendNoiseB - 0.5) * 0.34 * blendStrength, 0.2, 0.8);
  let a0 = sample_floor_albedo(uv0);
  let a1 = sample_floor_albedo(uv1);
  return mix(a0, a1, blend);
}

fn sample_floor_roughness_layered(uv: vec2f) -> f32 {
  let uvw = floor_uv_warp(uv);
  let floorScale = max(0.2, params.floorUvScale);
  let uv0 = rotate_uv(uvw * (1.42 * floorScale) + vec2f(0.17, -0.13), 0.31);
  let uv1 = rotate_uv(uvw * (2.31 * floorScale) + vec2f(0.37, 0.11), 0.97);
  let blendNoiseA = cheap_noise(vec3f(uvw * 0.71 + vec2f(-2.1, 4.8), 5.6)) * 0.5 + 0.5;
  let blendNoiseB = cheap_noise(vec3f(uvw * 1.21 + vec2f(1.5, -3.4), 6.4)) * 0.5 + 0.5;
  let blendStrength = clamp(params.floorBlendStrength, 0.0, 1.5);
  let blend = clamp(0.5 + (blendNoiseA - 0.5) * 0.5 * blendStrength + (blendNoiseB - 0.5) * 0.32 * blendStrength, 0.2, 0.8);
  let r0 = sample_floor_roughness(uv0);
  let r1 = sample_floor_roughness(uv1);
  return mix(r0, r1, blend);
}

fn sample_floor_normal_layered(uv: vec2f) -> vec3f {
  let uvw = floor_uv_warp(uv);
  let angle0 = 0.31;
  let angle1 = 0.97;
  let floorScale = max(0.2, params.floorUvScale);
  let uv0 = rotate_uv(uvw * (1.42 * floorScale) + vec2f(0.17, -0.13), angle0);
  let uv1 = rotate_uv(uvw * (2.31 * floorScale) + vec2f(0.37, 0.11), angle1);
  let blendNoiseA = cheap_noise(vec3f(uvw * 0.83 + vec2f(1.7, 0.6), 7.1)) * 0.5 + 0.5;
  let blendNoiseB = cheap_noise(vec3f(uvw * 1.17 + vec2f(-0.9, -2.8), 8.6)) * 0.5 + 0.5;
  let blendStrength = clamp(params.floorBlendStrength, 0.0, 1.5);
  let blend = clamp(0.5 + (blendNoiseA - 0.5) * 0.48 * blendStrength + (blendNoiseB - 0.5) * 0.28 * blendStrength, 0.2, 0.8);
  let n0 = sample_floor_normal(uv0);
  let n1 = rotate_floor_tangent_normal(sample_floor_normal(uv1), angle1 - angle0);
  return normalize(mix(n0, n1, blend));
}

fn floor_contact_mask(p: vec3f) -> f32 {
  if (params.sceneType != 0.0 && params.sceneType != 4.0) {
    return 0.0;
  }
  let d = get_wood_sdf(vec3f(p.x, 0.028, p.z));
  let near = 1.0 - smoothstep(0.01, 0.13, d);
  return clamp(near, 0.0, 1.0);
}

fn compute_reaction(pos: vec3f, temp: f32) -> f32 {
  // Flame lives on thin temperature edges, not in the hot volume.
  // reaction = smoothstep(T_ignite, T_hot, T) * smoothstep(g0, g1, |∇T|)
  let tempGate = smoothstep(params.T_ignite, params.T_burn, temp);
  if (tempGate < 0.001) { return 0.0; }

  let eps = 1.0 / params.dim;
  let txp = sample_volume(pos + vec3f(eps, 0.0, 0.0)).x;
  let txn = sample_volume(pos - vec3f(eps, 0.0, 0.0)).x;
  let typ = sample_volume(pos + vec3f(0.0, eps, 0.0)).x;
  let tyn = sample_volume(pos - vec3f(0.0, eps, 0.0)).x;
  let tzp = sample_volume(pos + vec3f(0.0, 0.0, eps)).x;
  let tzn = sample_volume(pos - vec3f(0.0, 0.0, eps)).x;
  let grad = vec3f(txp - txn, typ - tyn, tzp - tzn);
  let gradMag = length(grad) * 0.5;

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

fn fire_anchor() -> vec3f {
  let h = clamp(params.volumeHeight * 0.22, 0.1, 0.24);
  return vec3f(0.5, h, 0.5);
}

fn fire_light_approx(p: vec3f) -> vec3f {
  let firePos = fire_anchor();
  let toF = firePos - p;
  let r2 = max(1e-5, dot(toF, toF));
  let r = sqrt(r2);
  let dir = toF / r;
  let falloffScale = max(0.15, params.lightingFireFalloff);
  let falloff = 1.0 / (1.0 + r2 * (13.0 * falloffScale));
  let flickAmp = clamp(params.lightingFlicker, 0.0, 1.0);
  let flickA = (1.0 - 0.22 * flickAmp) + (0.22 * flickAmp) * sin(params.time * 9.3 + p.x * 4.1 + p.z * 3.7);
  let flickB = (1.0 - 0.14 * flickAmp) + (0.14 * flickAmp) * sin(params.time * 14.2);
  let flicker = max(0.55, flickA * flickB);
  let upBias = 0.45 + 0.55 * clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
  let warm = vec3f(1.0, 0.40, 0.15);
  let hot = vec3f(1.0, 0.74, 0.33);
  let spectrum = mix(warm, hot, 0.42);
  let intensity = max(0.0, params.emission) * 0.095 * max(0.0, params.lightingFireIntensity);
  return spectrum * intensity * falloff * flicker * upBias;
}

fn fire_glow_along_ray(ro: vec3f, rd: vec3f) -> vec3f {
  let firePos = fire_anchor();
  let t = clamp(dot(firePos - ro, rd), 0.0, 4.0);
  let closest = ro + rd * t;
  let d = length(closest - firePos);
  let haze = exp(-d * d * 6.5) * max(0.0, params.emission) * 0.026 * max(0.0, params.lightingGlow);
  let flickAmp = clamp(params.lightingFlicker, 0.0, 1.0);
  let flick = (1.0 - 0.18 * flickAmp) + (0.18 * flickAmp) * sin(params.time * 7.1 + d * 11.0);
  return vec3f(1.0, 0.44, 0.18) * haze * flick;
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

// HEMISPHERE GI - Sample upward in multiple directions to find actual fire
fn get_volume_lighting(pos: vec3f) -> vec3f {
    var totalLight = vec3f(0.0);

    // Sample directions in a hemisphere above the floor point
    // This finds light from wherever the fire actually IS
    let dirs = array<vec3f, 3>(
        vec3f(0.0, 1.0, 0.0),     // Straight up
        vec3f(0.5, 0.866, 0.0),   // Forward-up
        vec3f(-0.5, 0.866, 0.0)   // Back-up
    );

    for (var s = 0; s < 3; s++) {
      let dir = normalize(dirs[s]);

      // March upward from floor, find any fire along this direction
      var t = 0.05;
      var transmittance = 1.0;

      for (var i = 0; i < 8; i++) {
        if (transmittance < 0.02) { break; }
        let p = pos + dir * t;
        let uv = to_volume_uv(p);
        let m = sample_medium_world(p);
        let fuel = m.z;
        let temp = m.x;
        let soot = m.y;

        var reaction = smoothstep(params.T_ignite, params.T_burn, temp);
        let fuelGate = 0.25 + 0.75 * smoothstep(0.0, 0.10, fuel);
        reaction *= fuelGate;

        // Fire emits light downward to floor
        if (reaction > 0.01) {
          let emission = getBlackbodyColor(temp) * (params.emission * 0.12) * reaction;
          let atten = 1.0 / (1.0 + t * t * 5.0);
          totalLight += emission * atten * transmittance * 0.2;
        }

        // Smoke blocks light
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
        transmittance *= exp(-(sigmaA + sigmaS) * 0.1 * 0.3);

        t += 0.1;
      }
    }

    return totalLight * 0.5;
}

fn intersectAABB(ro: vec3f, rd: vec3f, bmin: vec3f, bmax: vec3f) -> vec2f {
    let tMin = (bmin - ro) / rd; let tMax = (bmax - ro) / rd;
    let t1 = min(tMin, tMax); let t2 = max(tMin, tMax);
    return vec2f(max(max(t1.x, t1.y), t1.z), min(min(t2.x, t2.y), t2.z));
}

fn get_floor_material(p: vec3f, ro: vec3f) -> vec3f {
    let uv = p.xz;
    let sootDep = clamp(sample_soot_floor(uv), 0.0, 1.0);
    let sootVis = pow(clamp(sootDep * 2.4, 0.0, 1.0), 0.72);
    let contact = floor_contact_mask(p);
    let baseAlbedoTex = sample_floor_albedo_layered(uv);
    let baseRoughTex = clamp(sample_floor_roughness_layered(uv), 0.0, 1.0);
    let baseNormalTex = sample_floor_normal_layered(uv);

    let macroTone = cheap_noise(vec3f(uv * 2.6 + vec2f(0.7, -0.4), 4.2)) * 0.5 + 0.5;
    let microTone = cheap_noise(vec3f(uv * 18.0 + vec2f(5.0, 3.0), 6.8)) * 0.5 + 0.5;
    var albedo = baseAlbedoTex * mix(0.83, 1.12, macroTone * 0.72 + microTone * 0.28);

    let sootTint = vec3f(0.08, 0.075, 0.07);
    let charTint = vec3f(0.095, 0.086, 0.08);
    let charStrength = clamp(params.floorCharStrength, 0.0, 2.0);
    albedo = mix(albedo, albedo * 0.72 + charTint * 0.28, contact * 0.4 * charStrength);
    let sootDarkening = clamp((sootVis * 0.72 + contact * 0.34) * max(0.0, params.floorSootDarkening), 0.0, 0.95);
    let stainedAlbedo = mix(albedo, albedo * (1.0 - sootDarkening) + sootTint * sootDarkening, sootDarkening);
    let nMicro = floor_micro_normal(uv, sootDep);
    let normalStrength = clamp(params.floorNormalStrength, 0.0, 2.0);
    let microStrength = clamp(params.floorMicroStrength, 0.0, 2.0);
    var n = normalize(mix(vec3f(0.0, 1.0, 0.0), baseNormalTex, normalStrength * 0.75));
    n = normalize(mix(n, nMicro, (0.18 + sootVis * 0.24) * microStrength));
    let v = normalize(ro - p);
    let l = normalize(vec3f(0.5, 0.24, 0.5) - p);
    let h = normalize(v + l);

    let fireBounceGain = max(0.0, params.floorFireBounce);
    let cacheLight = sample_radiance_cache(p);
    let gi = get_volume_lighting(p) + fire_light_approx(p) * fireBounceGain + cacheLight * (0.35 + 0.65 * fireBounceGain);
    let giStrength = dot(gi, vec3f(0.2126, 0.7152, 0.0722));
    let baseRough = clamp(baseRoughTex * 0.88 + 0.08, 0.08, 0.97);
    let sootRoughness = clamp(params.floorSootRoughness, 0.0, 2.0);
    let roughness = clamp(baseRough + (sootVis * 0.28 + contact * 0.24) * sootRoughness, 0.08, 0.99);
    let alpha = max(0.04, roughness * roughness);
    let NdotL = max(dot(n, l), 0.0);
    let NdotV = max(dot(n, v), 0.0);
    let NdotH = max(dot(n, h), 0.0);
    let VdotH = max(dot(v, h), 0.0);
    let f0 = mix(vec3f(0.024), vec3f(0.045), smoothstep(0.08, 0.5, sootDep));
    let F = fresnel_schlick(VdotH, f0);
    let a2 = alpha * alpha;
    let denom = max(1e-4, NdotH * NdotH * (a2 - 1.0) + 1.0);
    let D = a2 / (3.14159 * denom * denom);
    let k = roughness + 1.0;
    let k2 = (k * k) * 0.125;
    let Gv = NdotV / max(1e-4, NdotV * (1.0 - k2) + k2);
    let Gl = NdotL / max(1e-4, NdotL * (1.0 - k2) + k2);
    let G = Gv * Gl;
    let specTerm = (D * G) / max(1e-4, 4.0 * NdotV * NdotL);
    let specular = F * specTerm * (0.03 + giStrength * 0.6) * NdotL * mix(1.0, 0.42, sootVis) * max(0.0, params.floorSpecular);

    let contactOcclusion = 1.0 - contact * 0.45 * clamp(params.floorContactShadow, 0.0, 1.5);
    let ambient = vec3f(0.028) * max(0.0, params.floorAmbient);
    let diffuse = stainedAlbedo * gi * (0.82 + 0.18 * NdotL) * contactOcclusion;
    return diffuse + ambient * stainedAlbedo * (0.9 - sootVis * 0.25) + specular;
}

fn get_floor_material_fast(p: vec3f, ro: vec3f) -> vec3f {
    let uv = p.xz;
    let sootDep = clamp(sample_soot_floor(uv), 0.0, 1.0);
    let sootVis = pow(clamp(sootDep * 2.4, 0.0, 1.0), 0.72);
    let contact = floor_contact_mask(p);
    let baseAlbedoTex = sample_floor_albedo_layered(uv);
    let baseRoughTex = clamp(sample_floor_roughness_layered(uv), 0.0, 1.0);
    let tone = cheap_noise(vec3f(uv * 3.0 + vec2f(0.6, -0.9), 5.2)) * 0.5 + 0.5;
    var baseAlbedo = baseAlbedoTex * mix(0.86, 1.1, tone);
    let charTint = vec3f(0.095, 0.086, 0.08);
    let charStrength = clamp(params.floorCharStrength, 0.0, 2.0);
    baseAlbedo = mix(baseAlbedo, baseAlbedo * 0.74 + charTint * 0.26, contact * 0.35 * charStrength);
    let sootDarkening = clamp((sootVis * 0.72 + contact * 0.4) * max(0.0, params.floorSootDarkening), 0.0, 0.92);
    let sootTint = vec3f(0.08, 0.075, 0.07);
    let albedo = mix(baseAlbedo, baseAlbedo * (1.0 - sootDarkening) + sootTint * sootDarkening, sootDarkening);
    let nMap = sample_floor_normal_layered(uv);
    let n = normalize(mix(vec3f(0.0, 1.0, 0.0), nMap, 0.55 * clamp(params.floorNormalStrength, 0.0, 2.0)));
    let v = normalize(ro - p);
    let l = normalize(vec3f(0.5, 0.18, 0.5) - p);
    let h = normalize(v + l);
    let sootRoughness = clamp(params.floorSootRoughness, 0.0, 2.0);
    let roughness = clamp(baseRoughTex * 0.88 + 0.1 + (sootVis * 0.2 + contact * 0.12) * sootRoughness, 0.16, 0.99);
    let shininess = mix(110.0, 9.0, roughness);
    let NdotL = max(dot(n, l), 0.0);
    let specular = pow(max(dot(n, h), 0.0), shininess) * NdotL * mix(0.03, 0.008, sootVis) * max(0.0, params.floorSpecular);
    let heatMask = clamp(sootVis * 0.85 + contact * 0.65, 0.0, 1.0);
    let cacheLight = sample_radiance_cache(p);
    let fireBounce = (fire_light_approx(p) + cacheLight * 0.9) * (0.48 + 0.52 * heatMask) * max(0.0, params.floorFireBounce);
    let ambient = vec3f(0.023) * max(0.0, params.floorAmbient);
    let contactOcclusion = 1.0 - contact * 0.4 * clamp(params.floorContactShadow, 0.0, 1.5);
    return albedo * (ambient + fireBounce * contactOcclusion * (0.35 + 0.65 * NdotL)) + vec3f(specular);
}

@fragment fn frag_main(in: VertexOutput) -> @location(0) vec4f {
  let ro = params.cameraPos.xyz; let fwd = normalize(params.targetPos.xyz - ro);
  let right = normalize(cross(vec3f(0.0, 1.0, 0.0), fwd)); let up = cross(fwd, right);
  let rd = normalize(fwd + right * (in.uv.x - 0.5) * 2.0 + up * (in.uv.y - 0.5) * 2.0);
  let jitter = fract(sin(dot(in.uv + fract(params.time * 0.05), vec2f(12.9898, 78.233))) * 43758.5453);

  let lightDir = normalize(vec3f(0.3, 1.0, 0.4));
  var bgCol = vec3f(0.22, 0.22, 0.24); // Medium gray background

  let maxTraceDist = 5.5;
  var t_floor = -1.0;
  if (rd.y < -0.0001) {
    let tf = -ro.y / rd.y;
    if (tf > 0.0 && tf < maxTraceDist) {
      t_floor = tf;
    }
  }

  // Surface hits are solved in world space, independent of the sim AABB.
  var tWood = maxTraceDist + 1.0;
  var woodColor = vec3f(0.0);
  if (params.sceneType == 0.0 || params.sceneType == 4.0) {
    var tS = 0.02;
    for(var i=0; i<96; i++) {
      if (tS > maxTraceDist) { break; }
      let p = ro + rd * tS;
      let d = get_wood_sdf(p);
      if (d < 0.0004) {
        tWood = tS;
        let woodCol = mix(vec3f(0.12, 0.07, 0.03), vec3f(0.25, 0.15, 0.08), sin(length(p.xz-0.5)*120.0 + p.y*35.0)*0.5+0.5);
        let gi = get_volume_lighting(p) + fire_light_approx(p) * 0.75 + sample_radiance_cache(p) * 0.9;
        let ambient = vec3f(0.01);
        woodColor = woodCol * gi * 3.0 + woodCol * ambient;
        break;
      }
      tS += max(d * 0.92, 0.003);
    }
  }

  var tWall = maxTraceDist + 1.0;
  var wallColor = vec3f(0.0);
  var tW = 0.02;
  for(var i=0; i<96; i++) {
    if (tW > maxTraceDist) { break; }
    let p = ro + rd * tW;
    let d = get_wall_sdf(p);
    if (d < 0.001) {
      tWall = tW;
      let n = get_wall_normal(p);
      let wallCol = vec3f(0.75, 0.72, 0.68);
      let gi = get_volume_lighting(p) + fire_light_approx(p) * 0.7 + sample_radiance_cache(p) * 0.85;
      let ambient = vec3f(0.015);
      let toFire = normalize(vec3f(0.5, 0.3, 0.5) - p);
      let facing = max(0.0, dot(n, toFire));
      wallColor = wallCol * gi * 2.5 * (0.3 + facing * 0.7) + wallCol * ambient;
      break;
    }
    tW += max(d * 0.9, 0.005);
  }

  var tScene = maxTraceDist;
  var surfaceCol = bgCol;
  if (t_floor > 0.0 && t_floor < tScene) {
    tScene = t_floor;
    surfaceCol = get_floor_material(ro + rd * t_floor, ro);
  }
  if (tWall < tScene) {
    tScene = tWall;
    surfaceCol = wallColor;
  }
  if (tWood < tScene) {
    tScene = tWood;
    surfaceCol = woodColor;
  }

  let baseSteps = 160;
  let rayFactor = clamp(tScene / 1.6, 0.65, 1.75);
  let steps = clamp(i32(round(f32(baseSteps) * params.stepQuality * rayFactor)), 1, 420);
  let stepSize = max(1e-4, tScene / f32(steps));
  var pos = ro + rd * (stepSize * (0.5 + jitter));
  var accumCol = vec3f(0.0); var transmittance = 1.0; let phaseSun = phase_function(dot(rd, lightDir), params.anisotropyG);
  var cachedSunTrans = 1.0;
  var shadowRefreshCountdown = 0;

  for (var i = 0; i < steps; i++) {
    if (transmittance < 0.005) { break; }
    let uv = to_volume_uv(pos);
    let m = sample_medium_world(pos);
    let soot = m.y;
    let temp = m.x;
    let fuel = m.z;
    let mediumMask = m.w;
    if (mediumMask <= 0.00001) {
      pos += rd * stepSize * 2.2;
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
      let sootOpt = 1.0 - exp(-sootRaw * 0.35);
      let hazeOpt = 1.0 - exp(-hazeRaw * 0.12);
      let activity = sootOpt + hazeOpt + reaction;
      let emptyThreshold = 0.0018 / max(0.5, params.stepQuality);

      if (activity < emptyThreshold) {
        pos += rd * stepSize * 2.2;
        continue;
      }

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
        let sigmaT = sigmaA + sigmaS;
        let stepTrans = exp(-sigmaT * stepSize);

        if (shadowRefreshCountdown <= 0) {
          cachedSunTrans = get_light_transmittance(pos, lightDir);
          let denseMedium = reaction > 0.08 || sigmaT > 0.55;
          shadowRefreshCountdown = select(4, 0, denseMedium);
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
        accumCol += emission * transmittance * stepSize;

        transmittance *= stepTrans;
      }
    }
    pos += rd * stepSize;
  }

  let glowProbe = sample_radiance_cache(ro + rd * 0.9) * 0.25;
  let outsideGlow = (fire_glow_along_ray(ro, rd) + glowProbe) * transmittance;
  let mapped = tonemap_aces((accumCol + transmittance * surfaceCol + outsideGlow) * params.exposure);
  return vec4f(pow(mapped, vec3f(1.0 / params.gamma)), 1.0);
}
`;

interface SceneParams {
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

interface FloorLightingParams {
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

const SCENES: Array<{ id: number; name: string; params: SceneParams }> = [
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

const DEFAULT_TIME_STEP = 0.016;

const FLOOR_LIGHTING_DEFAULTS: FloorLightingParams = {
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
  lightingFlicker: 0.7,
  lightingGlow: 1.35,
};

type SimParamState = SceneParams & FloorLightingParams & { timeStep: number };
type EditableParamKey = Exclude<keyof SimParamState, 'timeStep'>;
type ParamGroup = 'fluid' | 'environment' | 'matter' | 'optics';
type ParamScale = 'linear' | 'log';
type MacroId = 'flameStyle' | 'smokeDensity' | 'convectionScale' | 'turbulenceCharacter';
type ControlSection = 'macros' | ParamGroup;
type DiagnosticsTab = 'runtime' | 'performance' | 'overlays' | 'logs';
type QualityMode = 'realtime' | 'accurate';
type PresetSlot = 'A' | 'B';

type DeckRailSection = 'home' | 'flame' | 'smoke' | 'convection' | 'turbulence' | 'floor' | 'lighting' | 'library';

interface ParameterSpec {
  key: EditableParamKey;
  label: string;
  group: ParamGroup;
  min: number;
  max: number;
  step: number;
  unit: string;
  hint: string;
  scale?: ParamScale;
}

interface MacroSpec {
  id: MacroId;
  label: string;
  low: string;
  high: string;
  hint: string;
}

interface TimelineEvent {
  id: number;
  at: number;
  message: string;
}

interface StabilitySnapshot {
  stability: number;
  cflProxy: number;
  vorticityEnergy: number;
  smokeIntegral: number;
  thermalDrive: number;
}

const PARAM_GROUP_LABELS: Record<ParamGroup, string> = {
  fluid: 'Fluid Field',
  environment: 'Environment',
  matter: 'Fuel & Matter',
  optics: 'Optics',
};

const PARAM_SPECS: ParameterSpec[] = [
  { key: 'buoyancy', label: 'Buoyancy', group: 'fluid', min: 0, max: 40, step: 0.1, unit: 'm/s^2*', hint: 'Upward thermal lift coefficient.' },
  { key: 'dissipation', label: 'Heat Decay', group: 'fluid', min: 0.9, max: 0.999, step: 0.001, unit: 'ratio/frame', hint: 'Temperature retention per frame.' },
  { key: 'vorticity', label: 'Vorticity', group: 'fluid', min: 0, max: 60, step: 0.1, unit: '1/s*', hint: 'Rotational energy injection.' },
  { key: 'plumeTurbulence', label: 'Plume Turbulence', group: 'fluid', min: 0, max: 20, step: 0.01, unit: 'coef', hint: 'Small-scale breakup intensity.' },
  { key: 'drag', label: 'Drag', group: 'fluid', min: 0, max: 0.2, step: 0.001, unit: 'coef', hint: 'Velocity damping.' },
  { key: 'windX', label: 'Wind X', group: 'environment', min: -0.5, max: 0.5, step: 0.01, unit: 'm/s*', hint: 'Cross-axis ambient flow.' },
  { key: 'windZ', label: 'Wind Z', group: 'environment', min: -0.5, max: 0.5, step: 0.01, unit: 'm/s*', hint: 'Depth-axis ambient flow.' },
  { key: 'turbFreq', label: 'Turbulence Frequency', group: 'environment', min: 1, max: 100, step: 1, unit: 'Hz*', hint: 'Noise field frequency.', scale: 'log' },
  { key: 'turbSpeed', label: 'Turbulence Speed', group: 'environment', min: 0, max: 10, step: 0.1, unit: 'm/s*', hint: 'Noise transport speed.' },
  { key: 'smokeDissipation', label: 'Matter Decay', group: 'matter', min: 0, max: 0.999, step: 0.001, unit: 'ratio/frame', hint: 'Matter density fade rate.' },
  { key: 'smokeWeight', label: 'Mass Coefficient', group: 'matter', min: -5, max: 15, step: 0.1, unit: 'kg/m^3*', hint: 'Smoke weight contribution.' },
  { key: 'emission', label: 'Heat Emission', group: 'matter', min: 0, max: 25, step: 0.1, unit: 'kW*', hint: 'Source heat per frame.' },
  { key: 'fuelEfficiency', label: 'Fuel Efficiency', group: 'matter', min: 0.1, max: 10, step: 0.1, unit: 'x', hint: 'Combustion conversion multiplier.', scale: 'log' },
  { key: 'T_ignite', label: 'Ignition Threshold', group: 'matter', min: 0.0, max: 1.0, step: 0.01, unit: 'T', hint: 'Temperature where reaction begins.' },
  { key: 'T_burn', label: 'Full Burn Threshold', group: 'matter', min: 0.0, max: 1.0, step: 0.01, unit: 'T', hint: 'Temperature where reaction fully engages.' },
  { key: 'burnRate', label: 'Burn Rate', group: 'matter', min: 0.0, max: 40.0, step: 0.1, unit: '1/s*', hint: 'Base reaction rate multiplier.' },
  { key: 'fuelInject', label: 'Fuel Injection', group: 'matter', min: 0.0, max: 5.0, step: 0.01, unit: 'amount/frame*', hint: 'Fuel source injection strength.' },
  { key: 'heatYield', label: 'Heat Yield', group: 'matter', min: 0.0, max: 10.0, step: 0.05, unit: 'T per R*', hint: 'Temperature added per unit reaction.' },
  { key: 'sootYieldFlame', label: 'Soot Yield (Flame)', group: 'matter', min: 0.0, max: 5.0, step: 0.01, unit: 'soot per R*', hint: 'Soot created in hot flame.' },
  { key: 'sootYieldSmolder', label: 'Soot Yield (Smolder)', group: 'matter', min: 0.0, max: 10.0, step: 0.01, unit: 'soot per R*', hint: 'Soot created in cooler reaction.' },
  { key: 'hazeConvertRate', label: 'Haze Conversion', group: 'matter', min: 0.0, max: 2.0, step: 0.01, unit: 'rate*', hint: 'Optional soot-to-haze conversion factor.' },
  { key: 'heatDiffusion', label: 'Heat Diffusion', group: 'matter', min: 0, max: 1, step: 0.01, unit: 'm^2/s*', hint: 'Thermal spread factor.' },
  { key: 'scattering', label: 'Scattering', group: 'optics', min: 0, max: 25, step: 0.1, unit: '1/m*', hint: 'Forward light scatter.' },
  { key: 'absorption', label: 'Absorption', group: 'optics', min: 0, max: 100, step: 0.1, unit: '1/m*', hint: 'Light energy removal.' },
  { key: 'smokeThickness', label: 'Smoke Thickness', group: 'optics', min: 0.0, max: 5.0, step: 0.05, unit: 'x', hint: 'Multiplies sigmaA and sigmaS. Set > 0 to see smoke.' },
  { key: 'smokeDarkness', label: 'Smoke Darkness', group: 'optics', min: 0.0, max: 1.0, step: 0.01, unit: 'mix', hint: 'Biases absorption vs scattering.' },
  { key: 'anisotropyG', label: 'Phase Anisotropy', group: 'optics', min: -0.2, max: 0.95, step: 0.01, unit: 'g', hint: 'Henyey–Greenstein phase parameter.' },
  { key: 'T_hazeStart', label: 'Haze Start', group: 'optics', min: 0.0, max: 1.0, step: 0.01, unit: 'T', hint: 'Cooling threshold where haze begins.' },
  { key: 'T_hazeFull', label: 'Haze Full', group: 'optics', min: 0.0, max: 1.0, step: 0.01, unit: 'T', hint: 'Cooling threshold where haze is maximal.' },
  { key: 'flameSharpness', label: 'Flame Sharpness', group: 'optics', min: 0.5, max: 10.0, step: 0.1, unit: 'x', hint: 'Reaction-front sharpening in rendering.' },
  { key: 'volumeHeight', label: 'Volume Height', group: 'optics', min: 0.5, max: 6.0, step: 0.05, unit: 'x', hint: 'Render-time volume bounds scale (Y axis).' },
  { key: 'floorUvScale', label: 'Floor UV Scale', group: 'optics', min: 0.2, max: 4.0, step: 0.01, unit: 'x', hint: 'Base floor texture tiling scale.' },
  { key: 'floorUvWarp', label: 'Floor UV Warp', group: 'optics', min: 0.0, max: 3.0, step: 0.01, unit: 'x', hint: 'Breaks repetition in floor texture mapping.' },
  { key: 'floorBlendStrength', label: 'Floor Blend Strength', group: 'optics', min: 0.0, max: 1.5, step: 0.01, unit: 'x', hint: 'Strength of layered anti-tiling blend.' },
  { key: 'floorNormalStrength', label: 'Floor Normal Strength', group: 'optics', min: 0.0, max: 2.0, step: 0.01, unit: 'x', hint: 'Normal-map influence on floor shading.' },
  { key: 'floorMicroStrength', label: 'Floor Micro Detail', group: 'optics', min: 0.0, max: 2.0, step: 0.01, unit: 'x', hint: 'Micro-normal detail strength for floor roughness breakup.' },
  { key: 'floorSootDarkening', label: 'Floor Soot Darkening', group: 'optics', min: 0.0, max: 2.0, step: 0.01, unit: 'x', hint: 'How much soot darkens stained regions.' },
  { key: 'floorSootRoughness', label: 'Floor Soot Roughness', group: 'optics', min: 0.0, max: 2.0, step: 0.01, unit: 'x', hint: 'How much soot increases floor roughness.' },
  { key: 'floorCharStrength', label: 'Floor Char Strength', group: 'optics', min: 0.0, max: 2.0, step: 0.01, unit: 'x', hint: 'Burn/char tint intensity near contact zones.' },
  { key: 'floorContactShadow', label: 'Floor Contact Shadow', group: 'optics', min: 0.0, max: 1.5, step: 0.01, unit: 'x', hint: 'Contact shadowing strength around wood intersections.' },
  { key: 'floorSpecular', label: 'Floor Specular Gain', group: 'optics', min: 0.0, max: 2.0, step: 0.01, unit: 'x', hint: 'Overall floor specular response.' },
  { key: 'floorFireBounce', label: 'Floor Fire Bounce', group: 'optics', min: 0.0, max: 3.0, step: 0.01, unit: 'x', hint: 'How strongly fire light contributes to floor bounce.' },
  { key: 'floorAmbient', label: 'Floor Ambient', group: 'optics', min: 0.0, max: 2.0, step: 0.01, unit: 'x', hint: 'Base ambient visibility for the floor.' },
  { key: 'lightingFireIntensity', label: 'Fire Light Intensity', group: 'optics', min: 0.0, max: 4.0, step: 0.01, unit: 'x', hint: 'Global analytic fire-light multiplier outside volume bounds.' },
  { key: 'lightingFireFalloff', label: 'Fire Light Falloff', group: 'optics', min: 0.15, max: 3.0, step: 0.01, unit: 'x', hint: 'Distance falloff scale for analytic fire light.' },
  { key: 'lightingFlicker', label: 'Fire Light Flicker', group: 'optics', min: 0.0, max: 1.0, step: 0.01, unit: 'mix', hint: 'Temporal flicker amplitude for fire light/glow.' },
  { key: 'lightingGlow', label: 'Fire Glow Strength', group: 'optics', min: 0.0, max: 3.0, step: 0.01, unit: 'x', hint: 'Out-of-volume atmospheric glow around fire.' },
  { key: 'exposure', label: 'Exposure', group: 'optics', min: 0.1, max: 10.0, step: 0.05, unit: 'EV', hint: 'Post-tonemap gain.' },
  { key: 'gamma', label: 'Gamma', group: 'optics', min: 0.1, max: 4, step: 0.05, unit: 'gamma', hint: 'Output transfer curve.' },
  { key: 'stepQuality', label: 'Step Quality', group: 'optics', min: 0.25, max: 4, step: 0.25, unit: 'samples', hint: 'Raymarch sample density.', scale: 'log' },
];

const PARAM_SPEC_BY_KEY = Object.fromEntries(PARAM_SPECS.map((spec) => [spec.key, spec])) as Record<EditableParamKey, ParameterSpec>;

const SECTION_GROUP_MAP: Record<ControlSection, ParamGroup | null> = {
  macros: null,
  fluid: 'fluid',
  environment: 'environment',
  matter: 'matter',
  optics: 'optics',
};

const MACRO_SPECS: MacroSpec[] = [
  { id: 'flameStyle', label: 'Flame Style', low: 'Calm', high: 'Aggressive', hint: 'Couples vorticity, drag, and emission.' },
  { id: 'smokeDensity', label: 'Smoke Density', low: 'Clean', high: 'Sooty', hint: 'Couples soot weight and optical attenuation.' },
  { id: 'convectionScale', label: 'Convection Scale', low: 'Tight', high: 'Broad', hint: 'Shapes plume height and spread.' },
  { id: 'turbulenceCharacter', label: 'Turbulence Character', low: 'Laminar', high: 'Chaotic', hint: 'Controls flow break-up.' },
];

const INITIAL_PARAMS: SimParamState = {
  ...SCENES[0].params,
  ...FLOOR_LIGHTING_DEFAULTS,
  exposure: SCENES[0].params.exposure ?? 1,
  gamma: SCENES[0].params.gamma ?? 2.2,
  timeStep: DEFAULT_TIME_STEP,
};

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const lerp = (start: number, end: number, t: number) => start + (end - start) * t;
const inverseLerp = (start: number, end: number, value: number) => (value - start) / Math.max(1e-9, end - start);
const roundToStep = (value: number, step: number, min = 0) => {
  const safeStep = Math.max(step, 1e-6);
  const next = Math.round((value - min) / safeStep) * safeStep + min;
  return Number(next.toFixed(6));
};

const formatWithStep = (value: number, step: number) => {
  if (!Number.isFinite(value)) return '-';
  let decimals = 2;
  if (step >= 1) decimals = 0;
  else if (step >= 0.1) decimals = 1;
  else if (step >= 0.01) decimals = 2;
  else decimals = 3;
  return value.toFixed(decimals);
};

const MAX_INTERNAL_RENDER_PIXELS = 1600 * 900;

const getRenderScale = (width: number, height: number) => {
  const safeWidth = Math.max(1, width);
  const safeHeight = Math.max(1, height);
  const pixelCount = safeWidth * safeHeight;
  if (pixelCount <= MAX_INTERNAL_RENDER_PIXELS) return 1;
  return Math.sqrt(MAX_INTERNAL_RENDER_PIXELS / pixelCount);
};

const createSolidRgbaTexture = (
  device: GPUDevice,
  rgba: [number, number, number, number],
  format: string = 'rgba8unorm'
) => {
  const textureUsage = (window as any).GPUTextureUsage;
  const texture = (device as any).createTexture({
    size: [1, 1],
    format,
    usage: textureUsage.TEXTURE_BINDING | textureUsage.COPY_DST | textureUsage.RENDER_ATTACHMENT,
  });
  const pixels = new Uint8Array(rgba);
  (device.queue as any).writeTexture(
    { texture },
    pixels,
    { bytesPerRow: 4, rowsPerImage: 1 },
    { width: 1, height: 1 }
  );
  return texture as GPUTexture;
};

const createTextureFromBitmap = (device: GPUDevice, bitmap: ImageBitmap, format: string) => {
  const textureUsage = (window as any).GPUTextureUsage;
  const texture = (device as any).createTexture({
    size: [bitmap.width, bitmap.height],
    format,
    usage: textureUsage.TEXTURE_BINDING | textureUsage.COPY_DST | textureUsage.RENDER_ATTACHMENT,
  });
  (device.queue as any).copyExternalImageToTexture(
    { source: bitmap },
    { texture },
    { width: bitmap.width, height: bitmap.height }
  );
  return texture as GPUTexture;
};

const loadFloorMaterialTextures = async (device: GPUDevice) => {
  const textures: GPUTexture[] = [];
  let fallbackUsed = false;
  const baseUrl = (import.meta as any).env?.BASE_URL ?? '/';
  const normalizedBase = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
  const assetUrl = (path: string) => new URL(path, `${window.location.origin}${normalizedBase}`).toString();
  const sampler = (device as any).createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    addressModeU: 'repeat',
    addressModeV: 'repeat',
    maxAnisotropy: 16,
  }) as GPUSampler;
  try {
    const albedoUrl = assetUrl('textures/concrete016/Concrete016_1K-JPG_Color.jpg');
    const roughnessUrl = assetUrl('textures/concrete016/Concrete016_1K-JPG_Roughness.jpg');
    const normalUrl = assetUrl('textures/concrete016/Concrete016_1K-JPG_NormalGL.jpg');

    const [albedoResp, roughnessResp, normalResp] = await Promise.all([
      fetch(albedoUrl),
      fetch(roughnessUrl),
      fetch(normalUrl),
    ]);

    if (!albedoResp.ok || !roughnessResp.ok || !normalResp.ok) {
      throw new Error('Failed to fetch floor PBR textures');
    }

    const isImage = (resp: Response) => (resp.headers.get('content-type') ?? '').startsWith('image/');
    if (!isImage(albedoResp) || !isImage(roughnessResp) || !isImage(normalResp)) {
      throw new Error('Floor PBR fetch returned non-image content');
    }

    const [albedoBlob, roughnessBlob, normalBlob] = await Promise.all([
      albedoResp.blob(),
      roughnessResp.blob(),
      normalResp.blob(),
    ]);

    const [albedoBitmap, roughnessBitmap, normalBitmap] = await Promise.all([
      createImageBitmap(albedoBlob),
      createImageBitmap(roughnessBlob),
      createImageBitmap(normalBlob),
    ]);

    const albedo = createTextureFromBitmap(device, albedoBitmap, 'rgba8unorm-srgb');
    const roughness = createTextureFromBitmap(device, roughnessBitmap, 'rgba8unorm');
    const normal = createTextureFromBitmap(device, normalBitmap, 'rgba8unorm');
    textures.push(albedo, roughness, normal);

    albedoBitmap.close();
    roughnessBitmap.close();
    normalBitmap.close();
    console.info('[FloorPBR] Loaded floor material textures.', { albedoUrl, roughnessUrl, normalUrl });
  } catch (error) {
    fallbackUsed = true;
    console.warn('[FloorPBR] Falling back to solid floor material.', error);
    // Fallback keeps rendering functional when static assets are unavailable.
    textures.push(
      createSolidRgbaTexture(device, [112, 110, 108, 255], 'rgba8unorm-srgb'),
      createSolidRgbaTexture(device, [200, 200, 200, 255]),
      createSolidRgbaTexture(device, [128, 255, 128, 255])
    );
  }

  const [albedo, roughness, normal] = textures;
  return {
    material: {
      albedoView: albedo.createView(),
      roughnessView: roughness.createView(),
      normalView: normal.createView(),
      sampler,
    } as FloorMaterialTextures,
    fallbackUsed,
    dispose: () => {
      for (const texture of textures) (texture as any).destroy();
      (sampler as any).destroy?.();
    },
  };
};

const createSeededRandom = (seed: number) => {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x100000000;
  };
};

const toLogUnit = (value: number, min: number, max: number) => {
  const safeMin = Math.max(1e-6, min);
  const safeValue = clamp(value, safeMin, max);
  return inverseLerp(Math.log(safeMin), Math.log(max), Math.log(safeValue));
};

const fromLogUnit = (unit: number, min: number, max: number) => {
  const safeMin = Math.max(1e-6, min);
  const t = clamp(unit, 0, 1);
  return Math.exp(lerp(Math.log(safeMin), Math.log(max), t));
};

const isTextInputTarget = (target: EventTarget | null): boolean => {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable;
};

const downloadJson = (filename: string, payload: unknown) => {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
};

const FluidSimulation: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [controlsVisible, setControlsVisible] = useState(true);
  const [isPlaying, setIsPlaying] = useState(true);
  const [isSmokeEnabled, setIsSmokeEnabled] = useState(true);
  const [qualityMode, setQualityMode] = useState<QualityMode>('realtime');
  const [diagnosticsTab, setDiagnosticsTab] = useState<DiagnosticsTab>('runtime');
  const [runtimeWarning, setRuntimeWarning] = useState<string | null>(null);
  const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
  const [stats, setStats] = useState({ fps: 0, frameTimeMs: 0, frame: 0 });
  const [selectedSceneId, setSelectedSceneId] = useState(0);
  const [activeSlot, setActiveSlot] = useState<PresetSlot>('A');
  const [presetSlots, setPresetSlots] = useState<Record<PresetSlot, SimParamState>>({
    A: { ...INITIAL_PARAMS },
    B: { ...INITIAL_PARAMS },
  });
  const [activeSection, setActiveSection] = useState<ControlSection>('macros');
  const [paramSearch, setParamSearch] = useState('');
  const [deckRailSection, setDeckRailSection] = useState<DeckRailSection>('home');
  const [expandedGroups, setExpandedGroups] = useState<Record<ParamGroup, boolean>>({
    fluid: true,
    environment: false,
    matter: false,
    optics: false,
  });
  const [dimensions, setDimensions] = useState({ width: window.innerWidth, height: window.innerHeight });
  const [gridSize, setGridSize] = useState(128);
  const [runtimeResolutionScale, setRuntimeResolutionScale] = useState(1);
  const [simParams, setSimParams] = useState<SimParamState>({ ...INITIAL_PARAMS });
  const [lockedParams, setLockedParams] = useState<Record<EditableParamKey, boolean>>(() => (
    Object.fromEntries(PARAM_SPECS.map((spec) => [spec.key, false])) as Record<EditableParamKey, boolean>
  ));

  const deriveSmokeAmountT = (params: Pick<SimParamState, 'smokeDissipation'>) => clamp(inverseLerp(0.85, 0.995, params.smokeDissipation), 0, 1);
  const deriveTurbulenceCharacterT = (params: Pick<SimParamState, 'plumeTurbulence'>) => clamp(inverseLerp(0.1, 14, params.plumeTurbulence), 0, 1);

  const [tuningMacroKnobs, setTuningMacroKnobs] = useState(() => ({
    smokeAmount: deriveSmokeAmountT(INITIAL_PARAMS),
    turbulenceCharacter: deriveTurbulenceCharacterT(INITIAL_PARAMS),
  }));
  const renderScale = useMemo(
    () => clamp(getRenderScale(dimensions.width, dimensions.height) * runtimeResolutionScale, 0.35, 1.0),
    [dimensions.height, dimensions.width, runtimeResolutionScale]
  );
  const internalCanvasSize = useMemo(
    () => ({
      width: Math.max(1, Math.round(dimensions.width * renderScale)),
      height: Math.max(1, Math.round(dimensions.height * renderScale)),
    }),
    [dimensions.height, dimensions.width, renderScale]
  );
  const cameraRef = useRef({ theta: 1.625, phi: 1.35, radius: 1.25, target: [0.5, 0.4, 0.5] as [number, number, number], pos: [0.45, 0.38, 1.3] as [number, number, number] });
  const interactionRef = useRef({ isDragging: false, lastX: 0, lastY: 0, button: 0 });
  const paramsRef = useRef(simParams);
  const playingRef = useRef(isPlaying);
  const sceneRef = useRef(selectedSceneId);
  const smokeEnabledRef = useRef(isSmokeEnabled);
  const qualityModeRef = useRef(qualityMode);
  const adaptiveStepScaleRef = useRef(1.0);
  const runtimeResolutionScaleRef = useRef(1.0);
  const stepFramesRef = useRef(0);
  const randomSeedRef = useRef(1337);
  const timelineIdRef = useRef(1);

  useEffect(() => { paramsRef.current = simParams; }, [simParams]);
  useEffect(() => { playingRef.current = isPlaying; }, [isPlaying]);
  useEffect(() => { sceneRef.current = selectedSceneId; }, [selectedSceneId]);
  useEffect(() => { smokeEnabledRef.current = isSmokeEnabled; }, [isSmokeEnabled]);
  useEffect(() => { qualityModeRef.current = qualityMode; }, [qualityMode]);
  useEffect(() => { runtimeResolutionScaleRef.current = runtimeResolutionScale; }, [runtimeResolutionScale]);

  const pushTimeline = useCallback((message: string) => {
    const item: TimelineEvent = { id: timelineIdRef.current, at: Date.now(), message };
    timelineIdRef.current += 1;
    setTimeline((prev) => [item, ...prev].slice(0, 60));
  }, []);

  const copyText = useCallback(async (text: string, successMessage: string) => {
    try {
      await navigator.clipboard.writeText(text);
      pushTimeline(successMessage);
    } catch {
      setRuntimeWarning('Clipboard copy failed (permission denied).');
    }
  }, [pushTimeline]);

  useEffect(() => {
    const handleResize = () => setDimensions({ width: window.innerWidth, height: window.innerHeight });
    window.addEventListener('resize', handleResize); return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.repeat || isTextInputTarget(event.target)) return;
      const key = event.key.toLowerCase();

      if (key === ' ') {
        event.preventDefault();
        setIsPlaying((prev) => !prev);
      } else if (key === 'r') {
        handleSceneChange(selectedSceneId);
      } else if (key === 'c') {
        setControlsVisible((prev) => !prev);
      } else if (key === 'm') {
        setIsSmokeEnabled((prev) => !prev);
      } else if (key === '.') {
        setIsPlaying(false);
        stepFramesRef.current = Math.min(8, stepFramesRef.current + 1);
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [selectedSceneId]);

  const handleSceneChange = (id: number) => {
    setSelectedSceneId(id);
    const scene = SCENES.find(s => s.id === id);
    if (scene) {
      setTuningMacroKnobs((prev) => ({
        ...prev,
        smokeAmount: deriveSmokeAmountT(scene.params),
        turbulenceCharacter: deriveTurbulenceCharacterT(scene.params),
      }));
      setSimParams((prev) => ({
        ...prev,
        ...FLOOR_LIGHTING_DEFAULTS,
        ...scene.params,
        exposure: scene.params.exposure ?? prev.exposure ?? 1,
        gamma: scene.params.gamma ?? prev.gamma ?? 2.2,
        timeStep: DEFAULT_TIME_STEP,
      }));
      pushTimeline(`Scenario changed to ${scene.name}`);
    }
  };

  const resetPreset = useCallback(() => {
    const scene = SCENES.find((entry) => entry.id === selectedSceneId) ?? SCENES[0];
    setTuningMacroKnobs((prev) => ({
      ...prev,
      smokeAmount: deriveSmokeAmountT(scene.params),
      turbulenceCharacter: deriveTurbulenceCharacterT(scene.params),
    }));
    setSimParams((prev) => ({
      ...prev,
      ...FLOOR_LIGHTING_DEFAULTS,
      ...scene.params,
      exposure: scene.params.exposure ?? prev.exposure ?? 1,
      gamma: scene.params.gamma ?? prev.gamma ?? 2.2,
      timeStep: DEFAULT_TIME_STEP,
    }));
    setRuntimeWarning(null);
    pushTimeline('Reset preset to scenario defaults');
  }, [pushTimeline, selectedSceneId]);

  const updateParam = useCallback((key: EditableParamKey, value: number) => {
    if (lockedParams[key]) return;
    setSimParams((prev) => ({
      ...prev,
      [key]: clamp(value, PARAM_SPEC_BY_KEY[key].min, PARAM_SPEC_BY_KEY[key].max),
    }));
  }, [lockedParams]);

  const toggleParamLock = useCallback((key: EditableParamKey) => {
    setLockedParams((prev) => {
      const nextLocked = !prev[key];
      pushTimeline(`${nextLocked ? 'Locked' : 'Unlocked'} ${PARAM_SPEC_BY_KEY[key].label}`);
      return { ...prev, [key]: nextLocked };
    });
  }, [pushTimeline]);

  const resetParamToSceneDefault = useCallback((key: EditableParamKey) => {
    const scene = SCENES.find((entry) => entry.id === selectedSceneId) ?? SCENES[0];
    const sceneValue = scene.params[key];
    const fallback = INITIAL_PARAMS[key];
    const next = Number.isFinite(sceneValue as number) ? Number(sceneValue) : Number(fallback);
    updateParam(key, next);
  }, [selectedSceneId, updateParam]);

  const setExpandedGroup = useCallback((group: ParamGroup) => {
    setExpandedGroups((prev) => ({ ...prev, [group]: !prev[group] }));
  }, []);

  const applyMacro = useCallback((id: MacroId, macroValue: number) => {
    const t = clamp(macroValue, 0, 1);
    setSimParams((prev) => {
      const next = { ...prev };
      const assign = (key: EditableParamKey, value: number) => {
        if (lockedParams[key]) return;
        next[key] = clamp(value, PARAM_SPEC_BY_KEY[key].min, PARAM_SPEC_BY_KEY[key].max) as SimParamState[EditableParamKey];
      };

      if (id === 'flameStyle') {
        assign('vorticity', lerp(2, 55, t));
        assign('drag', lerp(0.12, 0.01, t));
        assign('emission', lerp(1.2, 8.5, t));
        assign('fuelEfficiency', lerp(0.7, 3.0, t));
      } else if (id === 'smokeDensity') {
        assign('smokeWeight', lerp(-3, 12, t));
        // Higher values keep matter around longer (more visible smoke).
        assign('smokeDissipation', lerp(0.85, 0.995, t));
        assign('absorption', lerp(4, 65, t));
        assign('scattering', lerp(2, 20, t));
      } else if (id === 'convectionScale') {
        assign('buoyancy', lerp(2, 18, t));
        assign('heatDiffusion', lerp(0.02, 0.55, t));
        assign('drag', lerp(0.1, 0.02, t));
      } else {
        assign('plumeTurbulence', lerp(0.1, 14, t));
        assign('turbFreq', lerp(6, 80, t));
        assign('turbSpeed', lerp(0.5, 8, t));
        assign('vorticity', lerp(3, 45, t));
      }
      return next;
    });
    pushTimeline(`Macro ${id} set to ${Math.round(t * 100)}%`);
  }, [lockedParams, pushTimeline]);

  const randomizeUnlockedParams = useCallback(() => {
    randomSeedRef.current = (randomSeedRef.current + 2654435761) >>> 0;
    const seed = randomSeedRef.current || 1337;
    const random = createSeededRandom(seed);

    setSimParams((prev) => {
      const next = { ...prev };
      PARAM_SPECS.forEach((spec) => {
        if (lockedParams[spec.key]) return;
        const raw = spec.scale === 'log'
          ? fromLogUnit(random(), spec.min, spec.max)
          : lerp(spec.min, spec.max, random());
        next[spec.key] = clamp(roundToStep(raw, spec.step, spec.min), spec.min, spec.max) as SimParamState[EditableParamKey];
      });
      return next;
    });

    pushTimeline(`Randomized unlocked parameters (seed ${seed})`);
  }, [lockedParams, pushTimeline]);

  const storeSlot = useCallback((slot: PresetSlot) => {
    setPresetSlots((prev) => ({ ...prev, [slot]: { ...simParams, timeStep: DEFAULT_TIME_STEP } }));
    setActiveSlot(slot);
    pushTimeline(`Stored active params to slot ${slot}`);
  }, [pushTimeline, simParams]);

  const loadSlot = useCallback((slot: PresetSlot) => {
    setActiveSlot(slot);
    setSimParams((prev) => ({ ...prev, ...presetSlots[slot], timeStep: DEFAULT_TIME_STEP }));
    pushTimeline(`Loaded slot ${slot}`);
  }, [presetSlots, pushTimeline]);

  const updateCameraVectors = () => {
    const { theta, phi, radius, target } = cameraRef.current;
    const sP = Math.max(0.01, Math.min(Math.PI - 0.01, phi));
    cameraRef.current.pos = [target[0] + radius * Math.sin(sP) * Math.cos(theta), target[1] + radius * Math.cos(sP), target[2] + radius * Math.sin(sP) * Math.sin(theta)];
  };

  useEffect(() => {
    updateCameraVectors();
    if (!canvasRef.current || !navigator.gpu) { setError(navigator.gpu ? null : "WebGPU not supported"); return; }
    let animationFrameId: number;
    let device: GPUDevice;
    let context: GPUCanvasContext;
    let disposeFloorMaterial: (() => void) | null = null;
    let isDestroyed = false;

    const init = async () => {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter || isDestroyed) throw new Error("No adapter");
        device = await adapter.requestDevice();
        if (isDestroyed) { device.destroy(); return; }

        context = canvasRef.current!.getContext('webgpu') as any;
        const format = navigator.gpu.getPreferredCanvasFormat();
        context.configure({ device, format, alphaMode: 'premultiplied' });

        const loadedFloorMaterial = await loadFloorMaterialTextures(device);
        if (loadedFloorMaterial.fallbackUsed) {
          pushTimeline('Floor PBR load failed; using fallback floor material.');
        } else {
          pushTimeline('Floor PBR textures loaded.');
        }
        disposeFloorMaterial = loadedFloorMaterial.dispose;
        if (isDestroyed) {
          disposeFloorMaterial?.();
          device.destroy();
          return;
        }

        const transport = new FluidTransport(device, gridSize, loadedFloorMaterial.material);
        const computePipeline = device.createComputePipeline({ layout: transport.physicsContract.layout, compute: { module: device.createShaderModule({ code: COMPUTE_SHADER }), entryPoint: 'main' } });
        const sootFloorPipeline = device.createComputePipeline({ layout: transport.floorContract.layout, compute: { module: device.createShaderModule({ code: SOOT_FLOOR_UPDATE_SHADER }), entryPoint: 'main' } });
        const radianceInjectPipeline = device.createComputePipeline({ layout: transport.radianceInjectContract.layout, compute: { module: device.createShaderModule({ code: RADIANCE_CACHE_INJECT_SHADER }), entryPoint: 'main' } });
        const radiancePropPipeline = device.createComputePipeline({ layout: transport.radiancePropContract.layout, compute: { module: device.createShaderModule({ code: RADIANCE_CACHE_PROPAGATE_SHADER }), entryPoint: 'main' } });
        const renderPipeline = device.createRenderPipeline({ layout: transport.renderContract.layout, vertex: { module: device.createShaderModule({ code: RENDER_SHADER }), entryPoint: 'vert_main' }, fragment: { module: device.createShaderModule({ code: RENDER_SHADER }), entryPoint: 'frag_main', targets: [{ format }] }, primitive: { topology: 'triangle-list' } });

        let simFrame = 0;
        let activeRenderGroup = 0;
        let activeSootFloor = 0;
        let simAccumulatorSeconds = 0;
        let lastTime = performance.now();
        let statsTimer = 0;
        let dtAccum = 0;
        let rafCount = 0;
        const render = () => {
          if (isDestroyed) return;
          const now = performance.now(); const dt = now - lastTime; lastTime = now;
          const frameSeconds = Math.min(dt, 100) / 1000;
          rafCount += 1;
          dtAccum += dt;
          statsTimer += dt;
          if (statsTimer >= 1000) {
            const safeCount = Math.max(1, rafCount);
            const frameTimeMs = dtAccum / safeCount;
            setStats({ fps: safeCount, frameTimeMs, frame: simFrame });
            if (qualityModeRef.current === 'realtime') {
              let nextScale = adaptiveStepScaleRef.current;
              if (frameTimeMs > 18) nextScale *= 0.92;
              else if (frameTimeMs > 10) nextScale *= 0.96;
              else if (frameTimeMs < 7) nextScale *= 1.04;
              adaptiveStepScaleRef.current = clamp(nextScale, 0.85, 1.0);

              let nextResolutionScale = runtimeResolutionScaleRef.current;
              if (frameTimeMs > 8.5) nextResolutionScale *= 0.88;
              else if (frameTimeMs > 6.8) nextResolutionScale *= 0.94;
              else if (frameTimeMs < 5.4) nextResolutionScale *= 1.03;
              nextResolutionScale = clamp(nextResolutionScale, 0.35, 1.0);
              if (Math.abs(nextResolutionScale - runtimeResolutionScaleRef.current) > 0.015) {
                runtimeResolutionScaleRef.current = nextResolutionScale;
                setRuntimeResolutionScale(nextResolutionScale);
              }
            } else {
              adaptiveStepScaleRef.current = 1.0;
              if (runtimeResolutionScaleRef.current !== 1.0) {
                runtimeResolutionScaleRef.current = 1.0;
                setRuntimeResolutionScale(1.0);
              }
            }
            rafCount = 0;
            dtAccum = 0;
            statsTimer = 0;
          }
          const shouldAdvance = playingRef.current || stepFramesRef.current > 0;
          const qualityBoost = qualityModeRef.current === 'accurate' ? 1.35 : 1.0;
          const adaptiveScale = qualityModeRef.current === 'accurate' ? 1.0 : adaptiveStepScaleRef.current;
          const stepQuality = clamp(paramsRef.current.stepQuality * qualityBoost * adaptiveScale, 0.25, 4.0);

          transport.updateUniforms(now, {
            ...paramsRef.current,
            timeStep: DEFAULT_TIME_STEP,
            stepQuality,
            scattering: smokeEnabledRef.current ? paramsRef.current.scattering : 0.0,
            absorption: smokeEnabledRef.current ? paramsRef.current.absorption : 0.0
          }, { pos: cameraRef.current.pos, target: cameraRef.current.target }, sceneRef.current);

          const enc = device.createCommandEncoder();
          if (shouldAdvance) {
            if (playingRef.current) {
              simAccumulatorSeconds += frameSeconds;
            } else if (stepFramesRef.current > 0) {
              simAccumulatorSeconds += DEFAULT_TIME_STEP;
            }

            const maxSubsteps = 4;
            let substeps = 0;
            while (
              simAccumulatorSeconds >= DEFAULT_TIME_STEP &&
              substeps < maxSubsteps &&
              (playingRef.current || stepFramesRef.current > 0)
            ) {
              const stepIndex = simFrame % 2;
              const cp = enc.beginComputePass();
              cp.setPipeline(computePipeline);
              cp.setBindGroup(0, transport.physicsGroups[stepIndex]);
              const wc = Math.ceil(transport.dim / 4);
              cp.dispatchWorkgroups(wc, wc, wc);
              cp.end();

              activeRenderGroup = stepIndex;
              simFrame += 1;
              substeps += 1;
              simAccumulatorSeconds -= DEFAULT_TIME_STEP;

              if (!playingRef.current && stepFramesRef.current > 0) {
                stepFramesRef.current -= 1;
              }
            }

            if (substeps === maxSubsteps && simAccumulatorSeconds > DEFAULT_TIME_STEP * 2) {
              simAccumulatorSeconds = DEFAULT_TIME_STEP * 2;
            }

            if (substeps > 0) {
              const nextSootFloor = 1 - activeSootFloor;
              const fp = enc.beginComputePass();
              fp.setPipeline(sootFloorPipeline);
              fp.setBindGroup(0, transport.floorGroups[activeRenderGroup][activeSootFloor]);
              const twc = Math.ceil(transport.sootFloorSize / 8);
              fp.dispatchWorkgroups(twc, twc, 1);
              fp.end();
              activeSootFloor = nextSootFloor;

              const rpInject = enc.beginComputePass();
              rpInject.setPipeline(radianceInjectPipeline);
              rpInject.setBindGroup(0, transport.radianceInjectGroups[activeRenderGroup]);
              const rwc = Math.ceil(transport.radianceDim / 4);
              rpInject.dispatchWorkgroups(rwc, rwc, rwc);
              rpInject.end();

              const rpProp = enc.beginComputePass();
              rpProp.setPipeline(radiancePropPipeline);
              rpProp.setBindGroup(0, transport.radiancePropGroup);
              rpProp.dispatchWorkgroups(rwc, rwc, rwc);
              rpProp.end();
            }
          }

          const rp = enc.beginRenderPass({ colorAttachments: [{ view: context.getCurrentTexture().createView(), clearValue: { r: 0.22, g: 0.22, b: 0.24, a: 1 }, loadOp: 'clear', storeOp: 'store' }] });
          rp.setPipeline(renderPipeline);
          rp.setBindGroup(0, transport.renderGroups[activeRenderGroup][activeSootFloor]);
          rp.draw(6);
          rp.end();
          device.queue.submit([enc.finish()]);
          animationFrameId = requestAnimationFrame(render);
        };
        render();
      } catch (e: any) { if (!isDestroyed) setError(e.message); }
    };
    init();
    return () => {
      isDestroyed = true;
      cancelAnimationFrame(animationFrameId);
      disposeFloorMaterial?.();
      if (context) context.unconfigure();
      if (device) device.destroy();
    };
  }, [gridSize]);

  const activeScene = SCENES.find((scene) => scene.id === selectedSceneId) ?? SCENES[0];
  const stepOneFrame = () => {
    setIsPlaying(false);
    stepFramesRef.current = Math.min(8, stepFramesRef.current + 1);
    pushTimeline('Stepped one frame');
  };

  const captureFrame = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const link = document.createElement('a');
    link.href = canvas.toDataURL('image/png');
    link.download = `firesim-frame-${Date.now()}.png`;
    link.click();
  };

const copyParamsToClipboard = useCallback(() => {
    const cpuUniformWriter = `// CPU-side SimParams uniform packing (DataView, little-endian)
// NOTE: WGSL uses vec4f for cameraPos/targetPos to avoid vec3 padding traps.
// Buffer size in this app: 256 bytes (fields used up through byte 252).
const uniformData = new ArrayBuffer(256);
const view = new DataView(uniformData);
const f32 = (byteOff: number, v: number) => view.setFloat32(byteOff, v, true);

f32(0, dim);
f32(4, timeSeconds);
f32(8, dt);
f32(12, vorticity);

f32(16, dissipation);
f32(20, buoyancy);
f32(24, drag);
f32(28, emission);

f32(32, exposure);
f32(36, gamma);
f32(40, sceneType);
f32(44, scattering);

f32(48, absorption);
f32(52, smokeWeight);
f32(56, plumeTurbulence);
f32(60, smokeDissipation);

// cameraPos: vec4f @ 64 (xyz + w)
f32(64, cameraPos.x); f32(68, cameraPos.y); f32(72, cameraPos.z); f32(76, 0);
// targetPos: vec4f @ 80 (xyz + w)
f32(80, targetPos.x); f32(84, targetPos.y); f32(88, targetPos.z); f32(92, 0);

f32(96, windX);
f32(100, windZ);
f32(104, turbFreq);
f32(108, turbSpeed);
f32(112, fuelEfficiency);
f32(116, heatDiffusion);
f32(120, stepQuality);
f32(124, 0);

// Extended block @ 128
f32(128, T_ignite);
f32(132, T_burn);
f32(136, burnRate);
f32(140, fuelInject);

f32(144, heatYield);
f32(148, sootYieldFlame);
f32(152, sootYieldSmolder);
f32(156, hazeConvertRate);

f32(160, T_hazeStart);
f32(164, T_hazeFull);
f32(168, anisotropyG);
f32(172, smokeThickness);

f32(176, smokeDarkness);
f32(180, flameSharpness);
f32(184, sootDissipation);
f32(188, volumeHeight);

f32(192, floorUvScale);
f32(196, floorUvWarp);
f32(200, floorBlendStrength);
f32(204, floorNormalStrength);

f32(208, floorMicroStrength);
f32(212, floorSootDarkening);
f32(216, floorSootRoughness);
f32(220, floorCharStrength);

f32(224, floorContactShadow);
f32(228, floorSpecular);
f32(232, floorFireBounce);
f32(236, floorAmbient);

f32(240, lightingFireIntensity);
f32(244, lightingFireFalloff);
f32(248, lightingFlicker);
f32(252, lightingGlow);
`;

    const payload = {
      schema: 'firesim-params.v1',
      copiedAt: new Date().toISOString(),
      packingDebugPrompt: 'If you paste your CPU-side uniform-buffer write code (the part that creates the ArrayBuffer / Float32Array), I’ll tell you exactly which line is wrong and give you a corrected writer that matches the layout.',
      cpuUniformWriter,
      sceneId: selectedSceneId,
      gridSize,
      smokeEnabled: isSmokeEnabled,
      qualityMode,
      params: { ...simParams, timeStep: DEFAULT_TIME_STEP },
    };
    void copyText(JSON.stringify(payload, null, 2), 'Copied params JSON');
  }, [copyText, gridSize, isSmokeEnabled, qualityMode, selectedSceneId, simParams]);

  const copyAlgorithmToClipboard = useCallback(() => {
    const payload = `// firesim algorithm (WGSL)\n// copiedAt: ${new Date().toISOString()}\n\n// === COMPUTE_SHADER ===\n${COMPUTE_SHADER}\n\n// === SOOT_FLOOR_UPDATE_SHADER ===\n${SOOT_FLOOR_UPDATE_SHADER}\n\n// === RADIANCE_CACHE_INJECT_SHADER ===\n${RADIANCE_CACHE_INJECT_SHADER}\n\n// === RADIANCE_CACHE_PROPAGATE_SHADER ===\n${RADIANCE_CACHE_PROPAGATE_SHADER}\n\n// === RENDER_SHADER ===\n${RENDER_SHADER}\n`;
    void copyText(payload, 'Copied algorithm (WGSL)');
  }, [copyText]);

  const savePresetToDisk = () => {
    downloadJson(`firesim-preset-${Date.now()}.json`, {
      schema: 'firesim-preset.v1',
      savedAt: new Date().toISOString(),
      sceneId: selectedSceneId,
      gridSize,
      smokeEnabled: isSmokeEnabled,
      qualityMode,
      params: { ...simParams, timeStep: DEFAULT_TIME_STEP },
    });
    pushTimeline('Saved preset JSON');
  };

  const loadPresetFromDisk = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;

    try {
      const text = await file.text();
      const parsed = JSON.parse(text) as Record<string, unknown>;
      const nextParams = (parsed.params && typeof parsed.params === 'object'
        ? parsed.params
        : parsed) as Partial<SimParamState>;

      const mergedPreview = { ...simParams, ...nextParams } as SimParamState;
      setTuningMacroKnobs((prev) => ({
        ...prev,
        smokeAmount: deriveSmokeAmountT(mergedPreview),
        turbulenceCharacter: deriveTurbulenceCharacterT(mergedPreview),
      }));

      setSimParams((prev) => ({
        ...prev,
        ...nextParams,
        timeStep: DEFAULT_TIME_STEP,
      }));

      if (typeof parsed.sceneId === 'number') {
        const exists = SCENES.some((scene) => scene.id === parsed.sceneId);
        if (exists) setSelectedSceneId(parsed.sceneId);
      }
      if (typeof parsed.gridSize === 'number') {
        const clampedGrid = [64, 128, 192, 256].includes(parsed.gridSize) ? parsed.gridSize : gridSize;
        setGridSize(clampedGrid);
      }
      if (typeof parsed.smokeEnabled === 'boolean') {
        setIsSmokeEnabled(parsed.smokeEnabled);
      }
      if (parsed.qualityMode === 'realtime' || parsed.qualityMode === 'accurate') {
        setQualityMode(parsed.qualityMode);
      }
      setRuntimeWarning(null);
      pushTimeline(`Loaded preset file ${file.name}`);
    } catch (loadError: unknown) {
      const message = loadError instanceof Error ? loadError.message : String(loadError);
      setRuntimeWarning(`Failed to load preset: ${message}`);
    }
  };

  const exportSession = () => {
    const cflProxy = DEFAULT_TIME_STEP * (Math.abs(simParams.windX) + Math.abs(simParams.windZ) + simParams.vorticity * 0.02);
    const payload = {
      schema: 'firesim-session.v1',
      exportedAt: new Date().toISOString(),
      scene: activeScene.name,
      stats,
      gridSize,
      qualityMode,
      smokeEnabled: isSmokeEnabled,
      cflProxy,
      params: { ...simParams, timeStep: DEFAULT_TIME_STEP },
    };
    downloadJson(`firesim-session-${Date.now()}.json`, payload);
    pushTimeline('Exported session report');
  };

  const macroValues = useMemo(() => ({
    flameStyle: clamp(inverseLerp(2, 55, simParams.vorticity), 0, 1),
    smokeDensity: clamp(inverseLerp(-3, 12, simParams.smokeWeight), 0, 1),
    convectionScale: clamp(inverseLerp(2, 18, simParams.buoyancy), 0, 1),
    turbulenceCharacter: clamp(inverseLerp(0.1, 14, simParams.plumeTurbulence), 0, 1),
  }), [simParams]);

  const filteredSpecs = useMemo(() => {
    const targetGroup = SECTION_GROUP_MAP[activeSection];
    const query = paramSearch.trim().toLowerCase();
    return PARAM_SPECS.filter((spec) => {
      if (targetGroup && spec.group !== targetGroup) return false;
      if (!query) return true;
      return (
        spec.label.toLowerCase().includes(query) ||
        spec.key.toLowerCase().includes(query) ||
        spec.hint.toLowerCase().includes(query)
      );
    });
  }, [activeSection, paramSearch]);

  const groupedSpecs = useMemo(() => {
    const grouped: Record<ParamGroup, ParameterSpec[]> = {
      fluid: [],
      environment: [],
      matter: [],
      optics: [],
    };
    filteredSpecs.forEach((spec) => grouped[spec.group].push(spec));
    return grouped;
  }, [filteredSpecs]);

  const sceneDefaults = useMemo(() => {
    const scene = SCENES.find((entry) => entry.id === selectedSceneId) ?? SCENES[0];
    const defaults = {} as Record<EditableParamKey, number>;
    PARAM_SPECS.forEach((spec) => {
      const sceneValue = scene.params[spec.key];
      defaults[spec.key] = Number.isFinite(sceneValue as number) ? Number(sceneValue) : Number(INITIAL_PARAMS[spec.key]);
    });
    return defaults;
  }, [selectedSceneId]);

  const stabilitySnapshot = useMemo<StabilitySnapshot>(() => {
    const cflProxy = DEFAULT_TIME_STEP * (Math.abs(simParams.windX) + Math.abs(simParams.windZ) + simParams.vorticity * 0.02 + simParams.turbSpeed * 0.1);
    const vorticityEnergy = simParams.vorticity * simParams.plumeTurbulence;
    const smokeIntegral = (1 - simParams.smokeDissipation) * (simParams.smokeWeight + 6) * 10;
    const thermalDrive = simParams.emission * simParams.fuelEfficiency * simParams.buoyancy;
    const risk = clamp(
      inverseLerp(0.04, 0.55, cflProxy) * 0.45 +
      inverseLerp(20, 780, vorticityEnergy) * 0.35 +
      inverseLerp(0, 85, smokeIntegral) * 0.2,
      0,
      1
    );

    return {
      stability: 1 - risk,
      cflProxy,
      vorticityEnergy,
      smokeIntegral,
      thermalDrive,
    };
  }, [simParams]);

  const cameraAngleDeg = ((cameraRef.current.theta * 180) / Math.PI + 360) % 360;
  const displayAngle = Math.round(cameraAngleDeg);
  const frameTime = stats.frameTimeMs || 0;
  const simMs = Math.max(0, frameTime * 0.76);
  const renderMs = Math.max(0, frameTime * 0.19);
  const tempC = Math.round(simParams.emission * 120 + simParams.buoyancy * 60 + simParams.vorticity * 10);
  const fuelPct = Math.round(clamp(100 - simParams.emission * 2.6, 0, 100));
  const simTimeSeconds = stats.frame * DEFAULT_TIME_STEP;
  const smokeAmount = Math.round(clamp(inverseLerp(0.85, 0.995, simParams.smokeDissipation), 0, 1) * 1000 + simParams.scattering * 3.8);
  const smokeDarkness = clamp(simParams.absorption / 150, 0, 1);
  const turbulenceCharacter = clamp(inverseLerp(0.1, 14, simParams.plumeTurbulence) * 1.05, 0, 1);
  const windGusts = Math.round(simParams.turbSpeed * 6);
  const burnRate = simParams.fuelEfficiency * 0.9;
  const sootMassKg = Math.round(smokeAmount * 1.68);
  const heatRelease = sootMassKg;
  const totalMw = (simParams.emission * simParams.fuelEfficiency * simParams.buoyancy) / 5.25;
  const divergence = stabilitySnapshot.cflProxy;
  const isStable = stabilitySnapshot.stability > 0.55 && divergence < 0.01;

  const tuningSliders = useMemo(() => {
    const isAll = deckRailSection === 'home';
    const isFlame = deckRailSection === 'flame';
    const isSmoke = deckRailSection === 'smoke';
    const isConvection = deckRailSection === 'convection';
    const isTurb = deckRailSection === 'turbulence';
    const isFloor = deckRailSection === 'floor';
    const isLighting = deckRailSection === 'lighting';

    return [
      (isAll || isConvection) && {
        id: 'flameHeight',
        label: 'Flame Height',
        low: 'Low',
        high: 'High',
        tone: 'warm' as const,
        value: clamp(inverseLerp(0.5, 40.0, simParams.buoyancy), 0, 1),
        valueText: `${simParams.buoyancy.toFixed(2)} m/s²`,
        onChange: (t: number) => {
          updateParam('buoyancy', lerp(0.5, 40.0, t));
          // Sustain lift over height: keep heat from collapsing and reduce drag.
          updateParam('dissipation', lerp(0.88, 0.995, t));
          // Keep a small amount of drag at high lift to avoid runaway velocities
          // that can destabilize advection and "blow out" the flame.
          updateParam('drag', lerp(0.08, 0.015, t));
          updateParam('heatDiffusion', lerp(0.12, 0.0, t));
        },
      },
      (isAll || isFlame) && {
        id: 'flameIntensity',
        label: 'Flame Intensity',
        low: 'Cool',
        high: 'Hot',
        tone: 'warmCool' as const,
        value: clamp(inverseLerp(0, 25, simParams.emission), 0, 1),
        valueText: `${(simParams.emission / 5.6).toFixed(2)} MW/m³`,
        onChange: (t: number) => updateParam('emission', lerp(0, 25, t)),
      },
      (isAll || isFlame) && {
        id: 'fireDetail',
        label: 'Fire Detail',
        low: 'Soft',
        high: 'Fine',
        tone: 'warm' as const,
        value: clamp(simParams.stepQuality * 0.8, 0, 1),
        valueText: `${clamp(simParams.stepQuality * 0.8, 0, 1).toFixed(2)}`,
        onChange: (t: number) => updateParam('stepQuality', clamp(t / 0.8, 0.25, 4)),
      },
      (isAll || isSmoke) && {
        id: 'smokeAmount',
        label: 'Smoke Amount',
        low: 'Sparse',
        high: 'Heavy',
        tone: 'mono' as const,
        value: tuningMacroKnobs.smokeAmount,
        valueText: `${smokeAmount} kg/m³`,
        onChange: (t: number) => {
          setTuningMacroKnobs((prev) => ({ ...prev, smokeAmount: t }));
          applyMacro('smokeDensity', t);
        },
      },
      (isAll || isSmoke) && {
        id: 'smokeDarkness',
        label: 'Smoke Darkness',
        low: 'Clean',
        high: 'Sooty',
        tone: 'mono' as const,
        value: smokeDarkness,
        valueText: `${smokeDarkness.toFixed(2)}`,
        onChange: (t: number) => updateParam('absorption', clamp(t * 150, 0, 100)),
      },
      (isAll || isTurb) && {
        id: 'turbulenceCharacter',
        label: 'Turbulence Character',
        low: 'Laminar',
        high: 'Chaotic',
        tone: 'warm' as const,
        value: tuningMacroKnobs.turbulenceCharacter,
        valueText: `${turbulenceCharacter.toFixed(2)}`,
        onChange: (t: number) => {
          setTuningMacroKnobs((prev) => ({ ...prev, turbulenceCharacter: t }));
          applyMacro('turbulenceCharacter', t);
        },
      },
      (isAll || isConvection) && {
        id: 'windGusts',
        label: 'Wind / Gusts',
        low: 'Calm',
        high: 'Breezy',
        tone: 'cool' as const,
        value: clamp(inverseLerp(0, 2, simParams.turbSpeed), 0, 1),
        valueText: `${windGusts} m/s`,
        onChange: (t: number) => updateParam('turbSpeed', lerp(0, 2, t)),
      },
      (isAll || isFlame) && {
        id: 'burnRate',
        label: 'Burn Rate',
        low: 'Slow',
        high: 'Critical',
        tone: 'warm' as const,
        value: clamp(inverseLerp(0.2, 2.0, simParams.fuelEfficiency), 0, 1),
        valueText: `${burnRate.toFixed(2)} kg/s`,
        onChange: (t: number) => {
          const nextEff = lerp(0.2, 2.0, t);
          updateParam('fuelEfficiency', nextEff);
          // Keep the extended combustion knobs in sync with this macro control
          // unless the user explicitly overrides/locks them.
          updateParam('burnRate', nextEff * 6.0);
          updateParam('fuelInject', 0.4 + nextEff * 0.6);
        },
      },
      (isAll || isFloor) && {
        id: 'floorUvScale',
        label: 'Floor Tile Scale',
        low: 'Large',
        high: 'Fine',
        tone: 'mono' as const,
        value: clamp(inverseLerp(0.2, 4.0, simParams.floorUvScale), 0, 1),
        valueText: `${simParams.floorUvScale.toFixed(2)}x`,
        onChange: (t: number) => updateParam('floorUvScale', lerp(0.2, 4.0, t)),
      },
      (isAll || isFloor) && {
        id: 'floorUvWarp',
        label: 'Floor UV Warp',
        low: 'Flat',
        high: 'Broken',
        tone: 'mono' as const,
        value: clamp(inverseLerp(0.0, 3.0, simParams.floorUvWarp), 0, 1),
        valueText: `${simParams.floorUvWarp.toFixed(2)}`,
        onChange: (t: number) => updateParam('floorUvWarp', lerp(0.0, 3.0, t)),
      },
      (isAll || isFloor) && {
        id: 'floorBlendStrength',
        label: 'Floor Anti-Tile',
        low: 'Low',
        high: 'Strong',
        tone: 'mono' as const,
        value: clamp(inverseLerp(0.0, 1.5, simParams.floorBlendStrength), 0, 1),
        valueText: `${simParams.floorBlendStrength.toFixed(2)}`,
        onChange: (t: number) => updateParam('floorBlendStrength', lerp(0.0, 1.5, t)),
      },
      (isAll || isFloor) && {
        id: 'floorNormalStrength',
        label: 'Floor Normal',
        low: 'Flat',
        high: 'Crisp',
        tone: 'mono' as const,
        value: clamp(inverseLerp(0.0, 2.0, simParams.floorNormalStrength), 0, 1),
        valueText: `${simParams.floorNormalStrength.toFixed(2)}`,
        onChange: (t: number) => updateParam('floorNormalStrength', lerp(0.0, 2.0, t)),
      },
      (isAll || isFloor) && {
        id: 'floorMicroStrength',
        label: 'Floor Micro Detail',
        low: 'Smooth',
        high: 'Grainy',
        tone: 'mono' as const,
        value: clamp(inverseLerp(0.0, 2.0, simParams.floorMicroStrength), 0, 1),
        valueText: `${simParams.floorMicroStrength.toFixed(2)}`,
        onChange: (t: number) => updateParam('floorMicroStrength', lerp(0.0, 2.0, t)),
      },
      (isAll || isFloor) && {
        id: 'floorSootDarkening',
        label: 'Soot Darkening',
        low: 'Light',
        high: 'Heavy',
        tone: 'mono' as const,
        value: clamp(inverseLerp(0.0, 2.0, simParams.floorSootDarkening), 0, 1),
        valueText: `${simParams.floorSootDarkening.toFixed(2)}`,
        onChange: (t: number) => updateParam('floorSootDarkening', lerp(0.0, 2.0, t)),
      },
      (isAll || isFloor) && {
        id: 'floorSootRoughness',
        label: 'Soot Roughness',
        low: 'Smooth',
        high: 'Dry',
        tone: 'mono' as const,
        value: clamp(inverseLerp(0.0, 2.0, simParams.floorSootRoughness), 0, 1),
        valueText: `${simParams.floorSootRoughness.toFixed(2)}`,
        onChange: (t: number) => updateParam('floorSootRoughness', lerp(0.0, 2.0, t)),
      },
      (isAll || isFloor) && {
        id: 'floorCharStrength',
        label: 'Char Strength',
        low: 'Clean',
        high: 'Charred',
        tone: 'warm' as const,
        value: clamp(inverseLerp(0.0, 2.0, simParams.floorCharStrength), 0, 1),
        valueText: `${simParams.floorCharStrength.toFixed(2)}`,
        onChange: (t: number) => updateParam('floorCharStrength', lerp(0.0, 2.0, t)),
      },
      (isAll || isFloor) && {
        id: 'floorContactShadow',
        label: 'Contact Shadow',
        low: 'Soft',
        high: 'Deep',
        tone: 'mono' as const,
        value: clamp(inverseLerp(0.0, 1.5, simParams.floorContactShadow), 0, 1),
        valueText: `${simParams.floorContactShadow.toFixed(2)}`,
        onChange: (t: number) => updateParam('floorContactShadow', lerp(0.0, 1.5, t)),
      },
      (isAll || isFloor) && {
        id: 'floorSpecular',
        label: 'Floor Specular',
        low: 'Matte',
        high: 'Glossy',
        tone: 'mono' as const,
        value: clamp(inverseLerp(0.0, 2.0, simParams.floorSpecular), 0, 1),
        valueText: `${simParams.floorSpecular.toFixed(2)}`,
        onChange: (t: number) => updateParam('floorSpecular', lerp(0.0, 2.0, t)),
      },
      (isAll || isFloor) && {
        id: 'floorFireBounce',
        label: 'Floor Fire Bounce',
        low: 'Low',
        high: 'Strong',
        tone: 'warmCool' as const,
        value: clamp(inverseLerp(0.0, 3.0, simParams.floorFireBounce), 0, 1),
        valueText: `${simParams.floorFireBounce.toFixed(2)}x`,
        onChange: (t: number) => updateParam('floorFireBounce', lerp(0.0, 3.0, t)),
      },
      (isAll || isFloor) && {
        id: 'floorAmbient',
        label: 'Floor Ambient',
        low: 'Dark',
        high: 'Lifted',
        tone: 'mono' as const,
        value: clamp(inverseLerp(0.0, 2.0, simParams.floorAmbient), 0, 1),
        valueText: `${simParams.floorAmbient.toFixed(2)}x`,
        onChange: (t: number) => updateParam('floorAmbient', lerp(0.0, 2.0, t)),
      },
      (isAll || isLighting) && {
        id: 'lightingFireIntensity',
        label: 'Fire Light Intensity',
        low: 'Dim',
        high: 'Bright',
        tone: 'warmCool' as const,
        value: clamp(inverseLerp(0.0, 4.0, simParams.lightingFireIntensity), 0, 1),
        valueText: `${simParams.lightingFireIntensity.toFixed(2)}x`,
        onChange: (t: number) => updateParam('lightingFireIntensity', lerp(0.0, 4.0, t)),
      },
      (isAll || isLighting) && {
        id: 'lightingFireFalloff',
        label: 'Fire Light Falloff',
        low: 'Wide',
        high: 'Tight',
        tone: 'warm' as const,
        value: clamp(inverseLerp(0.15, 3.0, simParams.lightingFireFalloff), 0, 1),
        valueText: `${simParams.lightingFireFalloff.toFixed(2)}`,
        onChange: (t: number) => updateParam('lightingFireFalloff', lerp(0.15, 3.0, t)),
      },
      (isAll || isLighting) && {
        id: 'lightingFlicker',
        label: 'Fire Flicker',
        low: 'Stable',
        high: 'Lively',
        tone: 'warm' as const,
        value: clamp(simParams.lightingFlicker, 0, 1),
        valueText: `${simParams.lightingFlicker.toFixed(2)}`,
        onChange: (t: number) => updateParam('lightingFlicker', clamp(t, 0, 1)),
      },
      (isAll || isLighting) && {
        id: 'lightingGlow',
        label: 'Atmospheric Glow',
        low: 'None',
        high: 'Bloomy',
        tone: 'warmCool' as const,
        value: clamp(inverseLerp(0.0, 3.0, simParams.lightingGlow), 0, 1),
        valueText: `${simParams.lightingGlow.toFixed(2)}x`,
        onChange: (t: number) => updateParam('lightingGlow', lerp(0.0, 3.0, t)),
      },
    ].filter(Boolean) as Array<{
      id: string;
      label: string;
      low: string;
      high: string;
      tone: 'warm' | 'cool' | 'mono' | 'warmCool';
      value: number;
      valueText?: string;
      onChange: (t: number) => void;
    }>;
  }, [applyMacro, deckRailSection, simParams, smokeAmount, smokeDarkness, turbulenceCharacter, tuningMacroKnobs, updateParam, windGusts, burnRate]);

  if (error) return <div className="deck-error" role="alert">{error}</div>;

  return (
    <div className="deck-root">
      <canvas ref={canvasRef} width={internalCanvasSize.width} height={internalCanvasSize.height} className="deck-canvas"
        onPointerDown={e => { (e.target as HTMLElement).setPointerCapture(e.pointerId); interactionRef.current = { isDragging: true, lastX: e.clientX, lastY: e.clientY, button: e.button }; }}
        onPointerUp={e => { (e.target as HTMLElement).releasePointerCapture(e.pointerId); interactionRef.current.isDragging = false; }}
        onPointerMove={e => {
          if (!interactionRef.current.isDragging) return;
          const dx = e.clientX - interactionRef.current.lastX; const dy = e.clientY - interactionRef.current.lastY;
          interactionRef.current.lastX = e.clientX; interactionRef.current.lastY = e.clientY;
          if (interactionRef.current.button <= 1 && !e.shiftKey) { cameraRef.current.theta -= dx * 0.005; cameraRef.current.phi -= dy * 0.005; }
          else { const s = 0.002 * cameraRef.current.radius; cameraRef.current.target[0] += -dx * s; cameraRef.current.target[1] += dy * s; }
          updateCameraVectors();
        }}
        onWheel={e => { cameraRef.current.radius = Math.max(0.1, Math.min(20, cameraRef.current.radius * (1 + e.deltaY * 0.001))); updateCameraVectors(); }}
      />

      <input
        ref={fileInputRef}
        type="file"
        accept="application/json,.json"
        className="sim-visually-hidden"
        onChange={loadPresetFromDisk}
      />

      <header className="deck-topbar">
        <div className="deck-topbar-left">
          <Flame size={16} className="deck-brand-icon" />
          <div className="deck-brand">Firesim Control Deck</div>
        </div>

        <div className="deck-topbar-center">
          <div className="deck-pill deck-pill--select">
            <select
              value={selectedSceneId}
              onChange={(event) => handleSceneChange(Number(event.target.value))}
              aria-label="Scenario"
            >
              {SCENES.map((scene) => <option key={scene.id} value={scene.id}>{scene.name}</option>)}
            </select>
            <ChevronDown size={14} className="deck-pill-caret" />
          </div>

          <div className="deck-toolbar">
            <button type="button" className={`deck-tool ${isPlaying ? 'is-active' : ''}`} aria-label="Play" onClick={() => setIsPlaying(true)}><Play size={16} /></button>
            <button type="button" className={`deck-tool ${!isPlaying ? 'is-active' : ''}`} aria-label="Pause" onClick={() => setIsPlaying(false)}><Pause size={16} /></button>
            <button type="button" className="deck-tool" aria-label="Step" onClick={stepOneFrame}><SkipForward size={16} /></button>
            <button type="button" className="deck-tool" aria-label="Reset" onClick={() => handleSceneChange(selectedSceneId)}><RefreshCw size={16} /></button>
          </div>

          <div className="deck-pill deck-pill--button deck-pill--number" aria-label="Frametime">
            {frameTime.toFixed(1)}
          </div>

          <button type="button" className="deck-pill deck-pill--button deck-pill--action" onClick={resetPreset} aria-label="Reset preset">
            Reset
          </button>

          <div className="deck-pill deck-pill--button deck-pill--number" aria-label="Simulation time">
            {simTimeSeconds.toFixed(1)} <span className="deck-unit">s</span>
          </div>
        </div>

        <div className="deck-topbar-right">
          <button type="button" className="deck-tool" aria-label="Diagnostics"><AlertTriangle size={16} /></button>
          <button
            type="button"
            className={`deck-tool ${isSmokeEnabled ? 'is-active' : ''}`}
            aria-label={isSmokeEnabled ? 'Disable smoke' : 'Enable smoke'}
            aria-pressed={isSmokeEnabled}
            onClick={() => setIsSmokeEnabled((prev) => !prev)}
          >
            <Droplet size={16} />
          </button>
          <button type="button" className="deck-tool" aria-label="Copy params" onClick={copyParamsToClipboard}>
            <Copy size={16} />
          </button>
          <button type="button" className="deck-tool" aria-label="Copy algorithm" onClick={copyAlgorithmToClipboard}>
            <Zap size={16} />
          </button>
          <button type="button" className="deck-tool" aria-label="Capture" onClick={captureFrame}><Camera size={16} /></button>

          <div className="deck-metrics" aria-label="Performance summary">
            <span>FPS {stats.fps}</span>
            <span className="deck-metrics-sep">|</span>
            <span>Frametime {frameTime.toFixed(1)} ms</span>
            <span className="deck-metrics-sep">|</span>
            <span>Frame {stats.frame}</span>
          </div>
        </div>
      </header>

      <aside className={`deck-left deck-panel ${controlsVisible ? '' : 'is-hidden'}`} aria-label="Control deck">
        <nav className="deck-left-rail" aria-label="Sections">
          <button type="button" className={`deck-rail-btn ${deckRailSection === 'home' ? 'is-active' : ''}`} aria-label="Tuning" onClick={() => setDeckRailSection('home')}><Flame size={18} /></button>
          <button type="button" className={`deck-rail-btn ${deckRailSection === 'smoke' ? 'is-active' : ''}`} aria-label="Smoke" onClick={() => setDeckRailSection('smoke')}><Droplet size={18} /></button>
          <button type="button" className={`deck-rail-btn ${deckRailSection === 'flame' ? 'is-active' : ''}`} aria-label="Heat" onClick={() => setDeckRailSection('flame')}><Thermometer size={18} /></button>
          <button type="button" className={`deck-rail-btn ${deckRailSection === 'convection' ? 'is-active' : ''}`} aria-label="Convection" onClick={() => setDeckRailSection('convection')}><Wind size={18} /></button>
          <button type="button" className={`deck-rail-btn ${deckRailSection === 'turbulence' ? 'is-active' : ''}`} aria-label="Turbulence" onClick={() => setDeckRailSection('turbulence')}><Shuffle size={18} /></button>
          <button type="button" className={`deck-rail-btn ${deckRailSection === 'floor' ? 'is-active' : ''}`} aria-label="Floor" onClick={() => setDeckRailSection('floor')}><Eye size={18} /></button>
          <button type="button" className={`deck-rail-btn ${deckRailSection === 'lighting' ? 'is-active' : ''}`} aria-label="Lighting" onClick={() => setDeckRailSection('lighting')}><Gauge size={18} /></button>
          <button type="button" className={`deck-rail-btn ${deckRailSection === 'library' ? 'is-active' : ''}`} aria-label="Presets" onClick={() => { setDeckRailSection('library'); fileInputRef.current?.click(); }}><Users size={18} /></button>
        </nav>

        <div className="deck-left-body" aria-label="Controls">
          <div className="deck-tuning-head">
            <div className="deck-tuning-title">Tuning</div>
            <button type="button" className="deck-link" onClick={resetPreset}>Reset Preset</button>
          </div>
          <div className="deck-controls-inner" role="group" aria-label="Tuning sliders">
            {tuningSliders.map((slider) => (
              <DeckTuningSlider
                key={slider.id}
                label={slider.label}
                low={slider.low}
                high={slider.high}
                tone={slider.tone}
                value={slider.value}
                valueText={slider.valueText}
                onChange={slider.onChange}
              />
            ))}
          </div>
        </div>
      </aside>

      <aside className="deck-diagnostics deck-panel" aria-label="Diagnostics">
        <div className="deck-diag-head">
          <span>Diagnostics</span>
          <button type="button" className="deck-gear" aria-label="Diagnostics settings"><Settings size={14} /></button>
        </div>
        <div className="deck-tabs" role="tablist" aria-label="Diagnostics tabs">
          <button type="button" role="tab" aria-selected={diagnosticsTab === 'runtime'} className={`deck-tab ${diagnosticsTab === 'runtime' ? 'is-active' : ''}`} onClick={() => setDiagnosticsTab('runtime')}>Runtime</button>
          <button type="button" role="tab" aria-selected={diagnosticsTab === 'performance'} className={`deck-tab ${diagnosticsTab === 'performance' ? 'is-active' : ''}`} onClick={() => setDiagnosticsTab('performance')}>Performance</button>
          <button type="button" role="tab" aria-selected={diagnosticsTab === 'overlays'} className={`deck-tab ${diagnosticsTab === 'overlays' ? 'is-active' : ''}`} onClick={() => setDiagnosticsTab('overlays')}>Overlays</button>
          <button type="button" role="tab" aria-selected={diagnosticsTab === 'logs'} className={`deck-tab ${diagnosticsTab === 'logs' ? 'is-active' : ''}`} onClick={() => setDiagnosticsTab('logs')}>Logs</button>
        </div>

        <div className="deck-diag-body" role="tabpanel">
          {diagnosticsTab === 'runtime' && (
            <div className="deck-runtime">
              <div className="deck-stability" aria-label="Stability">
                <div className={`deck-badge ${isStable ? 'is-stable' : 'is-warn'}`}>{isStable ? 'Stable' : 'Unstable'}</div>
                <div className="deck-badge is-muted">Stable</div>
              </div>

              <dl className="deck-kv deck-kv--tight">
                <div><dt>Peak Temp</dt><dd>{tempC} °C</dd></div>
                <div><dt>Soot Mass</dt><dd>{sootMassKg} kg</dd></div>
                <div><dt>Heat Release</dt><dd>{heatRelease} MW m³</dd></div>
                <div><dt>Total MW</dt><dd>{totalMw.toFixed(1)} MW</dd></div>
              </dl>

              <div className="deck-divergence" aria-label="Divergence">
                <div className="deck-div-left">
                  <span className={`deck-dot ${isStable ? 'is-ok' : 'is-bad'}`} aria-hidden="true" />
                  <span>Divergence</span>
                  <span className={`deck-state ${isStable ? 'is-ok' : 'is-bad'}`}>{isStable ? 'Ok' : 'High'}</span>
                </div>
                <div className="deck-div-val">{divergence.toFixed(3)}</div>
              </div>

              <div className="deck-overlay-grid" aria-label="Overlay shortcuts">
                <button type="button" className="deck-overlay-tile" aria-pressed="false"><Thermometer size={14} /><span>Temperature</span><span className="deck-spark" aria-hidden="true" /></button>
                <button type="button" className="deck-overlay-tile" aria-pressed="false"><Eye size={14} /><span>Soot</span><span className="deck-spark" aria-hidden="true" /></button>
                <button type="button" className="deck-overlay-tile" aria-pressed="false"><Wind size={14} /><span>Velocity</span><span className="deck-spark" aria-hidden="true" /></button>
                <button type="button" className="deck-overlay-tile" aria-pressed="false"><Zap size={14} /><span>Fuel</span><span className="deck-spark" aria-hidden="true" /></button>
                <button type="button" className="deck-overlay-tile" aria-pressed="false"><Shuffle size={14} /><span>Vorticity</span><span className="deck-spark" aria-hidden="true" /></button>
                <button type="button" className="deck-overlay-tile" aria-pressed="false"><Flame size={14} /><span>Reaction</span><span className="deck-spark" aria-hidden="true" /></button>
              </div>
            </div>
          )}

          {diagnosticsTab === 'performance' && (
            <div className="deck-perf">
              <dl className="deck-kv">
                <div><dt>Stability</dt><dd>{Math.round(stabilitySnapshot.stability * 100)}%</dd></div>
                <div><dt>CFL Proxy</dt><dd>{stabilitySnapshot.cflProxy.toFixed(3)}</dd></div>
                <div><dt>Thermal Drive</dt><dd>{stabilitySnapshot.thermalDrive.toFixed(1)}</dd></div>
              </dl>

              <div className="deck-meters">
                <div className="deck-meter">
                  <div className="deck-meter-row"><span>Frame Budget</span><span>{frameTime.toFixed(1)} ms</span></div>
                  <div className="deck-meter-track"><div className="deck-meter-fill" style={{ width: `${clamp(frameTime / 16.67, 0, 1) * 100}%` }} /></div>
                </div>
                <div className="deck-meter">
                  <div className="deck-meter-row"><span>GPU Load Proxy</span><span>{stabilitySnapshot.vorticityEnergy.toFixed(1)}</span></div>
                  <div className="deck-meter-track"><div className="deck-meter-fill is-secondary" style={{ width: `${clamp(inverseLerp(0, 780, stabilitySnapshot.vorticityEnergy), 0, 1) * 100}%` }} /></div>
                </div>
              </div>
            </div>
          )}

          {diagnosticsTab === 'overlays' && (
            <div className="deck-overlays">
              <button
                type="button"
                className={`deck-toggle ${isSmokeEnabled ? 'is-on' : ''}`}
                onClick={() => setIsSmokeEnabled((prev) => !prev)}
                aria-pressed={isSmokeEnabled}
              >
                <span>Smoke</span>
                <span className="deck-toggle-meta">{isSmokeEnabled ? 'Enabled' : 'Disabled'}</span>
              </button>
              <button
                type="button"
                className={`deck-toggle ${controlsVisible ? 'is-on' : ''}`}
                onClick={() => setControlsVisible((prev) => !prev)}
                aria-pressed={controlsVisible}
              >
                <span>Controls</span>
                <span className="deck-toggle-meta">{controlsVisible ? 'Shown' : 'Hidden'}</span>
              </button>
              <button
                type="button"
                className="deck-toggle"
                onClick={exportSession}
              >
                <span>Export Session</span>
                <span className="deck-toggle-meta">JSON</span>
              </button>
            </div>
          )}

          {diagnosticsTab === 'logs' && (
            <div className="deck-logs">
              <ul className="deck-log-list">
                {timeline.length === 0 && <li className="deck-log-empty">No events yet.</li>}
                {timeline.map((event) => (
                  <li key={event.id} className="deck-log-item">
                    <time>{new Date(event.at).toLocaleTimeString()}</time>
                    <span>{event.message}</span>
                  </li>
                ))}
              </ul>
              <div className="deck-log-hint"><Keyboard size={12} />Space play/pause, . step, R reset, C controls, M smoke</div>
            </div>
          )}

          <div className="deck-meters">
            <div className="deck-meter">
              <div className="deck-meter-row"><span>Physics Step</span><span>{simMs.toFixed(1)} ms</span></div>
              <div className="deck-meter-track"><div className="deck-meter-fill" style={{ width: `${frameTime > 0 ? clamp(simMs / frameTime, 0, 1) * 100 : 0}%` }} /></div>
            </div>
            <div className="deck-meter">
              <div className="deck-meter-row"><span>Render Time</span><span>{renderMs.toFixed(1)} ms</span></div>
              <div className="deck-meter-track"><div className="deck-meter-fill is-secondary" style={{ width: `${frameTime > 0 ? clamp(renderMs / frameTime, 0, 1) * 100 : 0}%` }} /></div>
            </div>
          </div>
        </div>
      </aside>

      <footer className="deck-statusbar" aria-label="Status">
        <div className="deck-status-inner">
          <div className="deck-status-item"><Thermometer size={14} /><span>Peak Temp:</span><strong>{tempC} °C</strong></div>
          <div className="deck-status-item"><Eye size={14} /><span>Total Soot:</span><strong>{sootMassKg} kg</strong></div>
          <div className="deck-status-item"><Flame size={14} /><span>Heat Release:</span><strong>{totalMw.toFixed(1)} MW</strong></div>
          <div className="deck-status-item"><Shuffle size={14} /><span>Vorticity:</span><strong>{simParams.vorticity.toFixed(1)}</strong></div>
          <div className="deck-status-item"><Gauge size={14} /><span>Sim Time:</span><strong>{simTimeSeconds.toFixed(1)} s</strong></div>
        </div>
      </footer>

      {runtimeWarning && (
        <div className="deck-toast deck-panel" role="alert">
          <AlertTriangle size={14} />
          <span>{runtimeWarning}</span>
        </div>
      )}
    </div>
  );
};

const DeckMacroSlider: React.FC<{
  spec: MacroSpec;
  value: number;
  onChange: (value: number) => void;
}> = ({ spec, value, onChange }) => (
  <div className="deck-macro">
    <div className="deck-macro-label">{spec.label}</div>
    <div className="deck-macro-ends" aria-hidden="true">
      <span>{spec.low}</span>
      <span>{spec.high}</span>
    </div>
    <div className="deck-range">
      <div className="deck-range-track" aria-hidden="true" />
      <div className="deck-range-ticks" aria-hidden="true" />
      <input
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={value}
        onChange={(event) => onChange(clamp(Number.parseFloat(event.target.value), 0, 1))}
        aria-label={spec.label}
      />
    </div>
  </div>
);

const DeckTuningSlider: React.FC<{
  label: string;
  low: string;
  high: string;
  value: number;
  valueText?: string;
  tone?: 'warm' | 'cool' | 'mono' | 'warmCool';
  onChange: (t: number) => void;
}> = ({ label, low, high, value, valueText, tone = 'warm', onChange }) => (
  <div className={`deck-slider deck-slider--${tone}`} style={{ ['--t' as any]: clamp(value, 0, 1) }}>
    <div className="deck-slider-row">
      <div className="deck-slider-label">{label}</div>
      {valueText && <div className="deck-slider-value">{valueText}</div>}
    </div>
    <div className="deck-slider-track">
      <div className="deck-slider-rail" aria-hidden="true" />
      <div className="deck-slider-fill" aria-hidden="true" />
      <input
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={clamp(value, 0, 1)}
        onChange={(event) => onChange(clamp(Number.parseFloat(event.target.value), 0, 1))}
        aria-label={label}
      />
    </div>
    <div className="deck-slider-ends" aria-hidden="true">
      <span>{low}</span>
      <span>{high}</span>
    </div>
  </div>
);

const ControlGroup: React.FC<{ title: string; icon?: React.ReactNode; children: React.ReactNode }> = ({ title, icon, children }) => (
  <section className="sim-group">
    <header>
      {icon}
      <h4>{title}</h4>
    </header>
    <div className="sim-group-body">{children}</div>
  </section>
);

const MacroSlider: React.FC<{
  spec: MacroSpec;
  value: number;
  onChange: (value: number) => void;
}> = ({ spec, value, onChange }) => (
  <div className="sim-macro-row">
    <div className="sim-macro-head">
      <span>{spec.label}</span>
      <span>{Math.round(value * 100)}%</span>
    </div>
    <input
      type="range"
      min={0}
      max={1}
      step={0.01}
      value={value}
      className="sim-macro-slider"
      onChange={(event) => onChange(clamp(Number.parseFloat(event.target.value), 0, 1))}
      aria-label={spec.label}
    />
    <div className="sim-macro-scale">
      <span>{spec.low}</span>
      <span>{spec.high}</span>
    </div>
    <p>{spec.hint}</p>
  </div>
);

const ParameterRow: React.FC<{
  spec: ParameterSpec;
  value: number;
  defaultValue: number;
  locked: boolean;
  onChange: (value: number) => void;
  onReset: () => void;
  onToggleLock: () => void;
}> = ({ spec, value, defaultValue, locked, onChange, onReset, onToggleLock }) => {
  const dragModeRef = useRef<'coarse' | 'fine' | 'ultra'>('coarse');
  const safeValue = Number.isFinite(value) ? value : spec.min;
  const safeRange = Math.max(1e-6, spec.max - spec.min);
  const isLog = spec.scale === 'log';
  const sliderValue = isLog ? toLogUnit(safeValue, spec.min, spec.max) : safeValue;
  const sliderMin = isLog ? 0 : spec.min;
  const sliderMax = isLog ? 1 : spec.max;
  const sliderStep = isLog ? 0.001 : Math.max(spec.step * 0.05, 0.0001);
  const progress = isLog
    ? clamp(sliderValue, 0, 1) * 100
    : clamp((safeValue - spec.min) / safeRange, 0, 1) * 100;

  const applySliderValue = (rawSliderValue: number) => {
    const raw = isLog ? fromLogUnit(rawSliderValue, spec.min, spec.max) : rawSliderValue;

    let localStep = spec.step;
    if (dragModeRef.current === 'fine') localStep *= 0.2;
    if (dragModeRef.current === 'ultra') localStep *= 0.05;

    const snapped = clamp(roundToStep(raw, localStep, spec.min), spec.min, spec.max);
    onChange(snapped);
  };

  return (
    <article className={`sim-param-row ${locked ? 'is-locked' : ''}`}>
      <div className="sim-param-head">
        <div>
          <strong>{spec.label}</strong>
          <span>{spec.hint}</span>
        </div>
        <div className="sim-param-meta">
          <span>{spec.unit}</span>
          <span>{spec.min} to {spec.max}</span>
          <span>{isLog ? 'log' : 'linear'}</span>
        </div>
      </div>
      <div className="sim-param-controls">
        <input
          type="range"
          min={sliderMin}
          max={sliderMax}
          step={sliderStep}
          value={sliderValue}
          className="sim-param-slider"
          style={{ '--sim-param-progress': `${progress}%` } as React.CSSProperties}
          disabled={locked}
          onPointerDown={(event) => {
            dragModeRef.current = event.altKey ? 'ultra' : event.shiftKey ? 'fine' : 'coarse';
          }}
          onPointerUp={() => {
            dragModeRef.current = 'coarse';
          }}
          onChange={(event) => {
            const next = Number.parseFloat(event.target.value);
            if (!Number.isFinite(next)) return;
            applySliderValue(next);
          }}
          aria-label={spec.label}
        />
        <input
          type="number"
          min={spec.min}
          max={spec.max}
          step={spec.step}
          value={formatWithStep(safeValue, spec.step)}
          className="sim-param-number"
          disabled={locked}
          onChange={(event) => {
            const next = Number.parseFloat(event.target.value);
            if (!Number.isFinite(next)) return;
            onChange(clamp(next, spec.min, spec.max));
          }}
        />
        <button type="button" className="sim-param-button" onClick={onReset} aria-label={`Reset ${spec.label}`} disabled={locked}><RefreshCw size={12} /></button>
        <button
          type="button"
          className={`sim-param-button ${locked ? 'is-active' : ''}`}
          onClick={onToggleLock}
          aria-label={locked ? `Unlock ${spec.label}` : `Lock ${spec.label}`}
          data-tooltip={`${locked ? 'Unlock' : 'Lock'} ${spec.label}\nDefault ${formatWithStep(defaultValue, spec.step)}`}
        >
          {locked ? <Lock size={12} /> : <LockOpen size={12} />}
        </button>
      </div>
    </article>
  );
};

const ProfileMeter: React.FC<{
  label: string;
  value: number;
  leftText: string;
  rightText: string;
}> = ({ label, value, leftText, rightText }) => (
  <div className="sim-prof-row">
    <div className="sim-prof-head">
      <span>{label}</span>
      <span>{Math.round(clamp(value, 0, 1) * 100)}%</span>
    </div>
    <div className="sim-prof-bar">
      <span style={{ width: `${Math.round(clamp(value, 0, 1) * 100)}%` }} />
    </div>
    <div className="sim-prof-meta">
      <span>{leftText}</span>
      <span>{rightText}</span>
    </div>
  </div>
);

export default FluidSimulation;
