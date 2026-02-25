
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
    Monitor,
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
import * as THREE from 'three/webgpu';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

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

class FluidTransport {
  public uniformBuffer: GPUBuffer;

  public densityA: GPUBuffer;
  public densityB: GPUBuffer;
  public fuelA: GPUBuffer;
  public fuelB: GPUBuffer;
  public velocityA: GPUBuffer;
  public velocityB: GPUBuffer;
  public velocityScratch: GPUBuffer;
  public divergence: GPUBuffer;
  public pressureA: GPUBuffer;
  public pressureB: GPUBuffer;
  public velocityBufferSize: number;

  public physicsContract: ShaderContract;
  public renderContract: ShaderContract;
  public projectionDivContract: ShaderContract;
  public projectionJacobiContract: ShaderContract;
  public projectionGradContract: ShaderContract;

  public physicsGroups: GPUBindGroup[] = [];
  public renderGroups: GPUBindGroup[] = [];
  public projectionDivGroups: GPUBindGroup[] = [];
  public projectionJacobiGroups: GPUBindGroup[] = [];
  public projectionGradGroups: GPUBindGroup[][] = [[], []];

  constructor(
    private device: GPUDevice,
    public dim: number
  ) {
    const VOXEL_COUNT = dim * dim * dim;

    this.uniformBuffer = device.createBuffer({
      size: 288,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    const bufferUsage = (window as any).GPUBufferUsage;
    const storageUsage = bufferUsage.STORAGE | bufferUsage.COPY_DST | bufferUsage.COPY_SRC;

    this.densityA = device.createBuffer({ size: VOXEL_COUNT * 4, usage: storageUsage });
    this.densityB = device.createBuffer({ size: VOXEL_COUNT * 4, usage: storageUsage });

    this.fuelA = device.createBuffer({ size: VOXEL_COUNT * 4, usage: storageUsage });
    this.fuelB = device.createBuffer({ size: VOXEL_COUNT * 4, usage: storageUsage });

    this.velocityBufferSize = VOXEL_COUNT * 16;
    this.velocityA = device.createBuffer({ size: this.velocityBufferSize, usage: storageUsage });
    this.velocityB = device.createBuffer({ size: this.velocityBufferSize, usage: storageUsage });
    this.velocityScratch = device.createBuffer({ size: this.velocityBufferSize, usage: storageUsage });
    this.divergence = device.createBuffer({ size: VOXEL_COUNT * 4, usage: storageUsage });
    this.pressureA = device.createBuffer({ size: VOXEL_COUNT * 4, usage: storageUsage });
    this.pressureB = device.createBuffer({ size: VOXEL_COUNT * 4, usage: storageUsage });

    const zeroF32 = new Float32Array(VOXEL_COUNT);
    const zeroVec4 = new Float32Array(VOXEL_COUNT * 4);

    device.queue.writeBuffer(this.densityA, 0, zeroF32);
    device.queue.writeBuffer(this.densityB, 0, zeroF32);
    device.queue.writeBuffer(this.fuelA, 0, zeroF32);
    device.queue.writeBuffer(this.fuelB, 0, zeroF32);
    device.queue.writeBuffer(this.velocityA, 0, zeroVec4);
    device.queue.writeBuffer(this.velocityB, 0, zeroVec4);
    device.queue.writeBuffer(this.velocityScratch, 0, zeroVec4);
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
    ]);

    this.renderContract = new ShaderContract(this.device, 'Render', [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
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
    const uniformData = new ArrayBuffer(288);
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

  viewportWidth: f32,
  viewportHeight: f32,
  cameraAspect: f32,
  cameraTanHalfFov: f32,
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

const PROJECTION_DIVERGENCE_SHADER = `
${STRUCT_DEF}
@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> velocityIn: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> divergenceOut: array<f32>;

fn get_idx(p: vec3i) -> u32 {
  let d = i32(params.dim);
  let cp = clamp(p, vec3i(0), vec3i(d - 1));
  return u32(cp.z * d * d + cp.y * d + cp.x);
}

fn sample_velocity_bc(p: vec3i) -> vec3f {
  let d = i32(params.dim);
  var v = velocityIn[get_idx(p)].xyz;
  if (p.x < 0 || p.x >= d) { v.x = 0.0; }
  if (p.y < 0 || p.y >= d) { v.y = 0.0; }
  if (p.z < 0 || p.z >= d) { v.z = 0.0; }
  return v;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dim = u32(params.dim);
  if (gid.x >= dim || gid.y >= dim || gid.z >= dim) { return; }
  let c = vec3i(gid);
  let vL = sample_velocity_bc(c + vec3i(-1, 0, 0));
  let vR = sample_velocity_bc(c + vec3i(1, 0, 0));
  let vD = sample_velocity_bc(c + vec3i(0, -1, 0));
  let vU = sample_velocity_bc(c + vec3i(0, 1, 0));
  let vB = sample_velocity_bc(c + vec3i(0, 0, -1));
  let vF = sample_velocity_bc(c + vec3i(0, 0, 1));

  let h = 1.0 / max(1.0, params.dim);
  let div = (vR.x - vL.x + vU.y - vD.y + vF.z - vB.z) / (2.0 * h);
  divergenceOut[get_idx(c)] = div;
}
`;

const PROJECTION_JACOBI_SHADER = `
${STRUCT_DEF}
@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> divergenceIn: array<f32>;
@group(0) @binding(2) var<storage, read> pressureIn: array<f32>;
@group(0) @binding(3) var<storage, read_write> pressureOut: array<f32>;

fn get_idx(p: vec3i) -> u32 {
  let d = i32(params.dim);
  let cp = clamp(p, vec3i(0), vec3i(d - 1));
  return u32(cp.z * d * d + cp.y * d + cp.x);
}

fn sample_pressure_neumann(p: vec3i) -> f32 {
  return pressureIn[get_idx(p)];
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dim = u32(params.dim);
  if (gid.x >= dim || gid.y >= dim || gid.z >= dim) { return; }
  let c = vec3i(gid);

  let pL = sample_pressure_neumann(c + vec3i(-1, 0, 0));
  let pR = sample_pressure_neumann(c + vec3i(1, 0, 0));
  let pD = sample_pressure_neumann(c + vec3i(0, -1, 0));
  let pU = sample_pressure_neumann(c + vec3i(0, 1, 0));
  let pB = sample_pressure_neumann(c + vec3i(0, 0, -1));
  let pF = sample_pressure_neumann(c + vec3i(0, 0, 1));

  let h = 1.0 / max(1.0, params.dim);
  let h2 = h * h;
  let div = divergenceIn[get_idx(c)];
  let relaxed = (pL + pR + pD + pU + pB + pF - div * h2) / 6.0;
  pressureOut[get_idx(c)] = relaxed;
}
`;

const PROJECTION_GRADIENT_SHADER = `
${STRUCT_DEF}
@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> velocityIn: array<vec4f>;
@group(0) @binding(2) var<storage, read> pressureIn: array<f32>;
@group(0) @binding(3) var<storage, read_write> velocityOut: array<vec4f>;

fn get_idx(p: vec3i) -> u32 {
  let d = i32(params.dim);
  let cp = clamp(p, vec3i(0), vec3i(d - 1));
  return u32(cp.z * d * d + cp.y * d + cp.x);
}

fn sample_pressure_neumann(p: vec3i) -> f32 {
  return pressureIn[get_idx(p)];
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dim = u32(params.dim);
  if (gid.x >= dim || gid.y >= dim || gid.z >= dim) { return; }
  let c = vec3i(gid);
  let h = 1.0 / max(1.0, params.dim);

  let pL = sample_pressure_neumann(c + vec3i(-1, 0, 0));
  let pR = sample_pressure_neumann(c + vec3i(1, 0, 0));
  let pD = sample_pressure_neumann(c + vec3i(0, -1, 0));
  let pU = sample_pressure_neumann(c + vec3i(0, 1, 0));
  let pB = sample_pressure_neumann(c + vec3i(0, 0, -1));
  let pF = sample_pressure_neumann(c + vec3i(0, 0, 1));

  let gradP = vec3f(
    (pR - pL) / (2.0 * h),
    (pU - pD) / (2.0 * h),
    (pF - pB) / (2.0 * h)
  );

  let current = velocityIn[get_idx(c)];
  var projected = current.xyz - gradP;
  let d = i32(params.dim);
  if (c.x == 0 || c.x == d - 1) { projected.x = 0.0; }
  if (c.y == 0) { projected.y = max(0.0, projected.y); }
  if (c.z == 0 || c.z == d - 1) { projected.z = 0.0; }

  velocityOut[get_idx(c)] = vec4f(projected, current.w);
}
`;

const RENDER_SHADER = `
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

fn volume_edge_falloff_uv(uv: vec3f) -> f32 {
  let ex = min(uv.x, 1.0 - uv.x);
  let ey = min(uv.y, 1.0 - uv.y);
  let ez = min(uv.z, 1.0 - uv.z);
  let edge = min(min(ex, ey), ez);
  return smoothstep(0.0, 0.11, edge);
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

fn compute_reaction(pos: vec3f, temp: f32) -> f32 {
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
  d = min(d, sd_box_local(p - vec3f(0.5, 0.04, 0.5), vec3f(0.8, 0.04, 0.8)));
  d = min(d, sd_sphere_local(p - vec3f(1.08, 0.18, 0.26), 0.16));
  d = min(d, sd_box_local(p - vec3f(-0.06, 0.13, 0.34), vec3f(0.13, 0.13, 0.13)));
  d = min(d, sd_capsule_local(p, vec3f(0.34, 0.05, -0.1), vec3f(0.34, 0.29, -0.1), 0.1));
  d = min(d, sd_box_local(p - vec3f(0.5, 3.25, -3.4), vec3f(12.0, 4.0, 0.03)));
  return d;
}

fn intersect_occluders(ro: vec3f, rd: vec3f, minT: f32, maxT: f32) -> f32 {
  var hitT = maxT;

  // Analytic floor hit (y=0) is exact and much cheaper than SDF marching.
  if (rd.y < -1e-5) {
    let tFloor = (0.0 - ro.y) / rd.y;
    if (tFloor > minT && tFloor < hitT) {
      hitT = tFloor;
    }
  }

  var t = minT + 0.006;
  for (var i = 0; i < 52; i++) {
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
  let jitter = fract(sin(dot(in.uv, vec2f(12.9898, 78.233))) * 43758.5453);

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
  let volumeExit = min(5.5, volumeHit.y);
  let worldHitT = intersect_occluders(ro, rd, marchStart, volumeExit);
  if (worldHitT <= marchStart) {
    return vec4f(0.0);
  }

  let maxTraceDist = min(volumeExit, worldHitT);
  let baseSteps = 220;
  let steps = clamp(i32(round(f32(baseSteps) * params.stepQuality)), 1, 720);
  let baseStep = max(1e-4, maxTraceDist / f32(steps));
  var t = marchStart + baseStep * (0.5 + jitter);
  var accumCol = vec3f(0.0); var transmittance = 1.0; let phaseSun = phase_function(dot(rd, lightDir), params.anisotropyG);
  var cachedSunTrans = 1.0;
  var shadowRefreshCountdown = 0;

  for (var i = 0; i < steps * 3; i++) {
    if (transmittance < 0.005 || t > maxTraceDist) { break; }
    let pos = ro + rd * t;
    let uv = to_volume_uv(pos);
    let m = sample_medium_world(pos);
    let soot = m.y;
    let temp = m.x;
    let fuel = m.z;
    let mediumMask = m.w;
    if (mediumMask <= 0.00001) {
      t += baseStep * 2.6;
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
      let emissiveFocus = clamp(max(reaction, baseReaction), 0.0, 1.0);
      let localStep = clamp(baseStep * mix(1.0, 0.24, emissiveFocus), baseStep * 0.2, baseStep * 1.2);

      if (activity < emptyThreshold) {
        t += max(baseStep * 1.4, localStep * 1.9);
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

  let mapped = tonemap_aces(accumCol * params.exposure);
  let outColor = pow(mapped, vec3f(1.0 / params.gamma));
  let luma = dot(outColor, vec3f(0.2126, 0.7152, 0.0722));
  let alpha = clamp(max(1.0 - transmittance, smoothstep(0.01, 0.09, luma) * 0.82), 0.0, 0.98);
  // Canvas is configured as premultiplied alpha, so output premultiplied color.
  return vec4f(outColor * alpha, alpha);
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
  lightingFlicker: 0.2,
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

type DeckRailSection = 'home' | 'flame' | 'smoke' | 'convection' | 'turbulence' | 'floor' | 'lighting' | 'resolution' | 'library';

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
  { key: 'lightingFlicker', label: 'Sky Day/Night Blend', group: 'optics', min: 0.0, max: 1.0, step: 0.01, unit: 'mix', hint: '0 = daytime sky, 1 = nighttime sky.' },
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

const MAX_INTERNAL_RENDER_PIXELS = 3840 * 2160;

const getRenderScale = (width: number, height: number, pixelRatio: number) => {
  const safeWidth = Math.max(1, width);
  const safeHeight = Math.max(1, height);
  const safePixelRatio = Math.max(1, pixelRatio);
  const pixelCount = safeWidth * safeHeight * safePixelRatio * safePixelRatio;
  if (pixelCount <= MAX_INTERNAL_RENDER_PIXELS) return 1;
  return Math.sqrt(MAX_INTERNAL_RENDER_PIXELS / pixelCount);
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

interface WorldRuntime {
  renderer: THREE.WebGPURenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  floorMaterial: THREE.MeshPhysicalMaterial;
  wallMaterial: THREE.MeshPhysicalMaterial;
  keyLight: THREE.DirectionalLight;
  fillLight: THREE.HemisphereLight;
  fireLight: THREE.PointLight;
  textures: THREE.Texture[];
}

const SCANNED_LOG_CANDIDATES = [
  '/models/scanned-log.glb',
  '/models/scanned-log.gltf',
  '/models/scanned-log.obj',
  '/models/scanned-log.fbx',
  '/models/log-scan/scanned-log.glb',
  '/models/log-scan/scanned-log.gltf',
  '/models/log-scan/scanned-log.obj',
  '/models/log-scan/scanned-log.fbx',
];

const addWorldProps = (scene: THREE.Scene) => {
  const baseMaterial = new THREE.MeshPhysicalMaterial({
    color: 0x6e747c,
    roughness: 0.5,
    metalness: 0.08,
  });

  const stoneMaterial = new THREE.MeshPhysicalMaterial({
    color: 0x5b6066,
    roughness: 0.88,
    metalness: 0.02,
  });

  const reflectiveMaterial = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    roughness: 0.06,
    metalness: 1.0,
    clearcoat: 0.22,
    clearcoatRoughness: 0.12,
  });

  const pad = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.08, 1.6), baseMaterial);
  pad.position.set(0.5, 0.04, 0.5);
  pad.castShadow = false;
  pad.receiveShadow = false;
  scene.add(pad);

  const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.16, 48, 48), reflectiveMaterial);
  sphere.position.set(1.08, 0.18, 0.26);
  sphere.castShadow = false;
  sphere.receiveShadow = false;
  scene.add(sphere);

  const box = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.26, 0.26), stoneMaterial);
  box.position.set(-0.06, 0.13, 0.34);
  box.castShadow = false;
  box.receiveShadow = false;
  scene.add(box);

  const capsule = new THREE.Mesh(new THREE.CapsuleGeometry(0.1, 0.24, 14, 24), baseMaterial);
  capsule.position.set(0.34, 0.17, -0.1);
  capsule.castShadow = false;
  capsule.receiveShadow = false;
  scene.add(capsule);
};

const getScannedLogTransforms = () => ([
  { position: new THREE.Vector3(0.5, 0.08, 0.5), rotation: new THREE.Euler(0, 0, -Math.PI * 0.5 - 0.15) },
  { position: new THREE.Vector3(0.5, 0.11, 0.5), rotation: new THREE.Euler(Math.PI * 0.5 - 0.15, 0, 0) },
  { position: new THREE.Vector3(0.5, 0.14, 0.5), rotation: new THREE.Euler(0, Math.PI * 0.25, -Math.PI * 0.5 + 0.1) },
]);

const makeFallbackLogMaterial = () => new THREE.MeshPhysicalMaterial({
  color: 0x3c2a1f,
  roughness: 0.94,
  metalness: 0.01,
  clearcoat: 0.0,
});

const fract = (value: number) => value - Math.floor(value);
const smooth01 = (value: number) => {
  const t = clamp(value, 0, 1);
  return t * t * (3 - 2 * t);
};

const hash2 = (x: number, y: number, seed: number) => {
  const dot = x * 127.1 + y * 311.7 + seed * 74.7;
  return fract(Math.sin(dot) * 43758.5453123);
};

const valueNoise2 = (x: number, y: number, seed: number) => {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
  const a = hash2(ix, iy, seed);
  const b = hash2(ix + 1, iy, seed);
  const c = hash2(ix, iy + 1, seed);
  const d = hash2(ix + 1, iy + 1, seed);
  const ux = smooth01(fx);
  const uy = smooth01(fy);
  const x1 = lerp(a, b, ux);
  const x2 = lerp(c, d, ux);
  return lerp(x1, x2, uy);
};

const fbm2 = (x: number, y: number, seed: number, octaves = 5) => {
  let sum = 0;
  let amp = 0.5;
  let freq = 1;
  let ampSum = 0;
  for (let i = 0; i < octaves; i++) {
    sum += valueNoise2(x * freq, y * freq, seed + i * 31) * amp;
    ampSum += amp;
    amp *= 0.5;
    freq *= 2;
  }
  return sum / Math.max(1e-6, ampSum);
};

interface ProceduralLogMaterialSet {
  side: THREE.MeshPhysicalMaterial;
  cap: THREE.MeshPhysicalMaterial;
  textures: THREE.Texture[];
}

const makeProceduralLogMaterials = (seed = 9127, size = 512): ProceduralLogMaterialSet => {
  const pixelCount = size * size;
  const colorData = new Uint8Array(pixelCount * 4);
  const normalData = new Uint8Array(pixelCount * 4);
  const roughnessData = new Uint8Array(pixelCount * 4);
  const aoData = new Uint8Array(pixelCount * 4);
  const heightData = new Float32Array(pixelCount);

  const rowStride = size;
  for (let y = 0; y < size; y++) {
    const v = y / Math.max(1, size - 1);
    for (let x = 0; x < size; x++) {
      const u = x / Math.max(1, size - 1);
      const index = y * rowStride + x;
      const p = index * 4;

      const warp = fbm2(u * 4.3, v * 3.2, seed + 19, 3);
      const ridges = 1.0 - Math.abs(Math.sin((u + (warp - 0.5) * 0.17) * Math.PI * 36.0));
      const ridgeMask = Math.pow(clamp(ridges, 0.0, 1.0), 4.2);
      const coarse = fbm2(u * 12.0 + 1.2, v * 2.1 - 0.8, seed + 41, 4);
      const fine = fbm2(u * 40.0, v * 8.0 + 0.5, seed + 77, 2);
      const knotField = fbm2(u * 7.0 - 2.7, v * 12.0 + 1.9, seed + 131, 4);
      const knot = Math.pow(clamp((knotField - 0.72) / 0.28, 0.0, 1.0), 2.1);
      const charField = fbm2(u * 9.0 + 2.3, v * 11.0 - 1.1, seed + 211, 3);
      const char = clamp((charField - 0.81) * 3.0, 0.0, 1.0);

      const height = clamp(0.50 * ridgeMask + 0.34 * coarse + 0.16 * fine + 0.2 * knot, 0.0, 1.0);
      heightData[index] = height;
      const cavity = Math.pow(1.0 - height, 1.6);

      const baseR = 78 + height * 64 - cavity * 20 - char * 18;
      const baseG = 54 + height * 42 - cavity * 15 - char * 12;
      const baseB = 36 + height * 28 - cavity * 12 - char * 9;

      colorData[p] = clamp(Math.round(baseR), 8, 255);
      colorData[p + 1] = clamp(Math.round(baseG), 5, 255);
      colorData[p + 2] = clamp(Math.round(baseB), 3, 255);
      colorData[p + 3] = 255;

      const roughness = 186 + cavity * 48 + char * 25 - height * 32;
      roughnessData[p] = clamp(Math.round(roughness), 0, 255);
      roughnessData[p + 1] = roughnessData[p];
      roughnessData[p + 2] = roughnessData[p];
      roughnessData[p + 3] = 255;

      const ao = 138 + cavity * 84 - char * 18;
      aoData[p] = clamp(Math.round(ao), 0, 255);
      aoData[p + 1] = aoData[p];
      aoData[p + 2] = aoData[p];
      aoData[p + 3] = 255;
    }
  }

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const index = y * rowStride + x;
      const p = index * 4;
      const xPrev = (x - 1 + size) % size;
      const xNext = (x + 1) % size;
      const yPrev = Math.max(0, y - 1);
      const yNext = Math.min(size - 1, y + 1);
      const hL = heightData[y * rowStride + xPrev];
      const hR = heightData[y * rowStride + xNext];
      const hD = heightData[yPrev * rowStride + x];
      const hU = heightData[yNext * rowStride + x];

      const nx = (hL - hR) * 2.8;
      const ny = (hD - hU) * 2.3;
      const nz = 1.0;
      const invLen = 1.0 / Math.sqrt(nx * nx + ny * ny + nz * nz);

      normalData[p] = clamp(Math.round((nx * invLen * 0.5 + 0.5) * 255), 0, 255);
      normalData[p + 1] = clamp(Math.round((ny * invLen * 0.5 + 0.5) * 255), 0, 255);
      normalData[p + 2] = clamp(Math.round((nz * invLen * 0.5 + 0.5) * 255), 0, 255);
      normalData[p + 3] = 255;
    }
  }

  const colorMap = new THREE.DataTexture(colorData, size, size, THREE.RGBAFormat, THREE.UnsignedByteType);
  colorMap.wrapS = THREE.RepeatWrapping;
  colorMap.wrapT = THREE.RepeatWrapping;
  colorMap.anisotropy = 8;
  colorMap.colorSpace = THREE.SRGBColorSpace;
  colorMap.needsUpdate = true;

  const normalMap = new THREE.DataTexture(normalData, size, size, THREE.RGBAFormat, THREE.UnsignedByteType);
  normalMap.wrapS = THREE.RepeatWrapping;
  normalMap.wrapT = THREE.RepeatWrapping;
  normalMap.anisotropy = 8;
  normalMap.colorSpace = THREE.NoColorSpace;
  normalMap.needsUpdate = true;

  const roughnessMap = new THREE.DataTexture(roughnessData, size, size, THREE.RGBAFormat, THREE.UnsignedByteType);
  roughnessMap.wrapS = THREE.RepeatWrapping;
  roughnessMap.wrapT = THREE.RepeatWrapping;
  roughnessMap.anisotropy = 8;
  roughnessMap.colorSpace = THREE.NoColorSpace;
  roughnessMap.needsUpdate = true;

  const aoMap = new THREE.DataTexture(aoData, size, size, THREE.RGBAFormat, THREE.UnsignedByteType);
  aoMap.wrapS = THREE.RepeatWrapping;
  aoMap.wrapT = THREE.RepeatWrapping;
  aoMap.anisotropy = 8;
  aoMap.colorSpace = THREE.NoColorSpace;
  aoMap.needsUpdate = true;

  const side = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    map: colorMap,
    normalMap,
    roughnessMap,
    aoMap,
    roughness: 1.0,
    metalness: 0.0,
    clearcoat: 0.0,
    emissive: 0x120804,
    emissiveIntensity: 0.1,
  });
  side.normalScale.set(0.85, 0.85);

  const cap = new THREE.MeshPhysicalMaterial({
    color: 0xb18b67,
    map: colorMap,
    roughness: 0.9,
    metalness: 0.0,
    emissive: 0x100603,
    emissiveIntensity: 0.07,
  });

  return { side, cap, textures: [colorMap, normalMap, roughnessMap, aoMap] };
};

const makeProceduralLogGeometry = (length: number, radius: number, seed: number) => {
  const geometry = new THREE.CylinderGeometry(radius * 0.95, radius * 1.03, length, 72, 40, false);
  const position = geometry.getAttribute('position') as THREE.BufferAttribute;
  const point = new THREE.Vector3();
  const random = createSeededRandom(seed);
  const bendX = (random() - 0.5) * length * 0.05;
  const bendZ = (random() - 0.5) * length * 0.04;

  for (let i = 0; i < position.count; i++) {
    point.fromBufferAttribute(position, i);
    const y01 = point.y / Math.max(1e-6, length) + 0.5;
    const radial = Math.hypot(point.x, point.z);
    if (radial > 1e-6) {
      const angle = Math.atan2(point.z, point.x);
      const u = angle / (Math.PI * 2) + 0.5;
      const ridge = fbm2(u * 10.5 + 0.3, y01 * 3.8 - 0.1, seed + 17, 4);
      const groove = Math.pow(Math.abs(Math.sin((u + fbm2(u * 3.7, y01 * 6.2, seed + 31, 2) * 0.12) * Math.PI * 26.0)), 2.6);
      const knot = Math.pow(clamp((fbm2(u * 6.0 - 1.7, y01 * 10.0 + 0.9, seed + 79, 3) - 0.82) / 0.18, 0.0, 1.0), 2.0);
      const end = Math.pow(Math.abs(y01 * 2.0 - 1.0), 2.8);
      const taper = 1.0 - end * 0.22;
      const scale = clamp(taper + (ridge - 0.5) * 0.22 + (0.5 - groove) * 0.08 + knot * 0.12, 0.72, 1.42);
      const target = radial * scale;
      const inv = target / radial;
      point.x *= inv;
      point.z *= inv;
    }

    const bendT = Math.sin(y01 * Math.PI);
    point.x += bendX * bendT;
    point.z += bendZ * bendT;
    point.y += (fbm2(y01 * 12.0 + 0.3, seed * 0.013 + 2.1, seed + 101, 2) - 0.5) * radius * 0.18;
    position.setXYZ(i, point.x, point.y, point.z);
  }

  position.needsUpdate = true;
  geometry.computeVertexNormals();

  const uv = geometry.getAttribute('uv') as THREE.BufferAttribute | undefined;
  if (uv) {
    geometry.setAttribute('uv2', new THREE.BufferAttribute(new Float32Array(uv.array as ArrayLike<number>), 2));
  }
  return geometry;
};

const applyFallbackLogMaterial = (root: THREE.Object3D) => {
  const fallbackMaterial = makeFallbackLogMaterial();
  root.traverse((child) => {
    const mesh = child as THREE.Mesh;
    if (!mesh.isMesh) return;
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    const material = mesh.material as THREE.Material | THREE.Material[] | undefined;
    if (!material) {
      mesh.material = fallbackMaterial;
      return;
    }
    if (Array.isArray(material)) {
      mesh.material = material.map((entry) => {
        const anyEntry = entry as any;
        if (anyEntry?.map || anyEntry?.normalMap || anyEntry?.roughnessMap) return entry;
        return fallbackMaterial;
      });
      return;
    }
    const anyMaterial = material as any;
    if (!anyMaterial.map && !anyMaterial.normalMap && !anyMaterial.roughnessMap) {
      mesh.material = fallbackMaterial;
    }
  });
};

const normalizeLogSource = (source: THREE.Object3D) => {
  source.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(source);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  source.position.sub(center);
  const maxExtent = Math.max(size.x, size.y, size.z, 1e-4);
  const scale = 0.36 / maxExtent;
  source.scale.setScalar(scale);
  source.updateMatrixWorld(true);
};

const probeModelUrl = async (url: string): Promise<boolean> => {
  const isHtmlResponse = (contentType: string | null) =>
    (contentType ?? '').toLowerCase().includes('text/html');

  try {
    const head = await fetch(url, { method: 'HEAD' });
    if (head.ok) {
      if (isHtmlResponse(head.headers.get('content-type'))) return false;
      return true;
    }

    // Some dev/proxy setups reject HEAD; fall back to a small GET probe.
    if (head.status !== 405 && head.status !== 501) return false;
  } catch {
    // Fall through to GET probe.
  }

  try {
    const getProbe = await fetch(url, {
      method: 'GET',
      headers: { Range: 'bytes=0-256' },
      cache: 'no-store',
    });
    if (!getProbe.ok) return false;
    if (isHtmlResponse(getProbe.headers.get('content-type'))) return false;
    return true;
  } catch {
    return false;
  }
};

const loadScannedLogModel = async (): Promise<{ root: THREE.Object3D; sourceUrl: string } | null> => {
  const gltfLoader = new GLTFLoader();
  const objLoader = new OBJLoader();
  const fbxLoader = new FBXLoader();

  for (const url of SCANNED_LOG_CANDIDATES) {
    try {
      const exists = await probeModelUrl(url);
      if (!exists) continue;

      if (url.endsWith('.glb') || url.endsWith('.gltf')) {
        const gltf = await gltfLoader.loadAsync(url);
        return { root: gltf.scene, sourceUrl: url };
      }
      if (url.endsWith('.obj')) {
        const obj = await objLoader.loadAsync(url);
        return { root: obj, sourceUrl: url };
      }
      if (url.endsWith('.fbx')) {
        const fbx = await fbxLoader.loadAsync(url);
        return { root: fbx, sourceUrl: url };
      }
    } catch {
      continue;
    }
  }
  return null;
};

const addFallbackLogPile = (scene: THREE.Scene): THREE.Texture[] => {
  const materialSet = makeProceduralLogMaterials();
  const transforms = getScannedLogTransforms();
  const seeds = [313, 911, 1907];

  for (let i = 0; i < transforms.length; i++) {
    const transform = transforms[i];
    const random = createSeededRandom(seeds[i]);
    const length = 0.48 + random() * 0.08;
    const radius = 0.046 + random() * 0.01;
    const geometry = makeProceduralLogGeometry(length, radius, seeds[i] + 17);
    const mesh = new THREE.Mesh(geometry, [materialSet.side, materialSet.cap, materialSet.cap]);
    mesh.position.copy(transform.position);
    mesh.position.y += 0.02;
    mesh.rotation.copy(transform.rotation);
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    scene.add(mesh);
  }

  return materialSet.textures;
};

const addCampfireLogPile = async (
  scene: THREE.Scene
): Promise<{ loaded: boolean; source?: string; textures?: THREE.Texture[] }> => {
  const scanned = await loadScannedLogModel();
  if (!scanned) {
    const textures = addFallbackLogPile(scene);
    return { loaded: false, textures };
  }

  normalizeLogSource(scanned.root);
  applyFallbackLogMaterial(scanned.root);
  const transforms = getScannedLogTransforms();
  for (const transform of transforms) {
    const instance = scanned.root.clone(true);
    instance.position.copy(transform.position);
    instance.rotation.copy(transform.rotation);
    scene.add(instance);
  }
  return { loaded: true, source: scanned.sourceUrl };
};

const loadTextureSafe = async (
  loader: THREE.TextureLoader,
  url: string,
  colorSpace: THREE.ColorSpace,
): Promise<THREE.Texture | null> => {
  try {
    const texture = await loader.loadAsync(url);
    texture.colorSpace = colorSpace;
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.anisotropy = 8;
    return texture;
  } catch {
    return null;
  }
};

const FluidSimulation: React.FC = () => {
  const worldCanvasRef = useRef<HTMLCanvasElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const worldRuntimeRef = useRef<WorldRuntime | null>(null);
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
  const readPixelRatio = () => Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
  const [dimensions, setDimensions] = useState({ width: window.innerWidth, height: window.innerHeight, pixelRatio: readPixelRatio() });
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
    () => clamp(getRenderScale(dimensions.width, dimensions.height, dimensions.pixelRatio) * runtimeResolutionScale, 0.35, 1.0),
    [dimensions.height, dimensions.pixelRatio, dimensions.width, runtimeResolutionScale]
  );
  const internalCanvasSize = useMemo(
    () => ({
      width: Math.max(1, Math.round(dimensions.width * dimensions.pixelRatio * renderScale)),
      height: Math.max(1, Math.round(dimensions.height * dimensions.pixelRatio * renderScale)),
    }),
    [dimensions.height, dimensions.pixelRatio, dimensions.width, renderScale]
  );
  const cameraRef = useRef({ theta: 1.625, phi: 1.35, radius: 1.25, target: [0.5, 0.4, 0.5] as [number, number, number], pos: [0.45, 0.38, 1.3] as [number, number, number] });
  const interactionRef = useRef({ isDragging: false, lastX: 0, lastY: 0, button: 0 });
  const paramsRef = useRef(simParams);
  const playingRef = useRef(isPlaying);
  const sceneRef = useRef(selectedSceneId);
  const smokeEnabledRef = useRef(isSmokeEnabled);
  const qualityModeRef = useRef(qualityMode);
  const adaptiveStepScaleRef = useRef(1.0);
  const stepFramesRef = useRef(0);
  const randomSeedRef = useRef(1337);
  const timelineIdRef = useRef(1);
  const worldNeedsRenderRef = useRef(true);
  const renderWorldRef = useRef<(() => void) | null>(null);

  const markWorldDirty = useCallback(() => {
    worldNeedsRenderRef.current = true;
  }, []);

  useEffect(() => { paramsRef.current = simParams; }, [simParams]);
  useEffect(() => { playingRef.current = isPlaying; }, [isPlaying]);
  useEffect(() => { sceneRef.current = selectedSceneId; }, [selectedSceneId]);
  useEffect(() => { smokeEnabledRef.current = isSmokeEnabled; }, [isSmokeEnabled]);
  useEffect(() => { qualityModeRef.current = qualityMode; }, [qualityMode]);
  useEffect(() => { markWorldDirty(); }, [markWorldDirty, simParams, dimensions.width, dimensions.height, dimensions.pixelRatio]);

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
    const handleResize = () => setDimensions({ width: window.innerWidth, height: window.innerHeight, pixelRatio: readPixelRatio() });
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
    worldNeedsRenderRef.current = true;
  };

  useEffect(() => {
    const canvas = worldCanvasRef.current;
    if (!canvas || !navigator.gpu) return;

    let destroyed = false;
    let removeResize = () => {};

    const initWorld = async () => {
      const renderer = new THREE.WebGPURenderer({
        canvas,
        antialias: true,
        alpha: true,
      });
      await renderer.init();
      if (destroyed) {
        renderer.dispose();
        return;
      }

      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setSize(window.innerWidth, window.innerHeight, false);
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.0;
      renderer.shadowMap.enabled = false;

      const scene = new THREE.Scene();
      const skyDay = new THREE.Color(0x9eb6cc);
      const skyNight = new THREE.Color(0x111a25);
      const skyMix = new THREE.Color();
      scene.background = skyDay.clone();
      scene.fog = new THREE.Fog(0x9eb6cc, 3.5, 26.0);

      const camera = new THREE.PerspectiveCamera(
        90,
        window.innerWidth / Math.max(1, window.innerHeight),
        0.02,
        120
      );

      const keyLight = new THREE.DirectionalLight(0xfff2dd, 2.2);
      keyLight.position.set(-2.6, 4.8, -1.2);
      keyLight.castShadow = false;
      scene.add(keyLight);

      const fillLight = new THREE.HemisphereLight(0x9fbfe0, 0x2a3038, 0.48);
      scene.add(fillLight);

      const fireLight = new THREE.PointLight(0xff8a3c, 3.0, 5.0, 1.8);
      fireLight.position.set(0.5, 0.22, 0.5);
      fireLight.castShadow = false;
      scene.add(fireLight);

      const floorGeometry = new THREE.PlaneGeometry(34, 34);
      floorGeometry.rotateX(-Math.PI / 2);
      const floorMaterial = new THREE.MeshPhysicalMaterial({
        color: 0x868b90,
        roughness: 0.76,
        metalness: 0.06,
        clearcoat: 0.12,
        clearcoatRoughness: 0.44,
      });
      const floorMesh = new THREE.Mesh(floorGeometry, floorMaterial);
      floorMesh.position.set(0.5, 0.0, 0.5);
      floorMesh.receiveShadow = false;
      floorMesh.castShadow = false;
      scene.add(floorMesh);

      const wallMaterial = new THREE.MeshPhysicalMaterial({
        color: 0x434950,
        roughness: 0.78,
        metalness: 0.03,
      });
      const wall = new THREE.Mesh(new THREE.PlaneGeometry(24, 8), wallMaterial);
      wall.position.set(0.5, 3.25, -3.4);
      wall.receiveShadow = false;
      wall.castShadow = false;
      scene.add(wall);

      const textures: THREE.Texture[] = [];
      addWorldProps(scene);
      const logAsset = await addCampfireLogPile(scene);
      if (logAsset.textures?.length) textures.push(...logAsset.textures);
      if (logAsset.loaded) {
        pushTimeline(`Loaded scanned log asset (${logAsset.source})`);
      } else {
        pushTimeline('No scanned log asset found in /public/models; using procedural scanned-style logs.');
      }

      const loader = new THREE.TextureLoader();
      const [albedo, normal, roughness] = await Promise.all([
        loadTextureSafe(loader, '/textures/concrete016/Concrete016_1K-JPG_Color.jpg', THREE.SRGBColorSpace),
        loadTextureSafe(loader, '/textures/concrete016/Concrete016_1K-JPG_NormalGL.jpg', THREE.NoColorSpace),
        loadTextureSafe(loader, '/textures/concrete016/Concrete016_1K-JPG_Roughness.jpg', THREE.NoColorSpace),
      ]);
      if (!destroyed) {
        const repeatX = 18;
        const repeatY = 18;
        if (albedo) {
          albedo.repeat.set(repeatX, repeatY);
          floorMaterial.map = albedo;
          floorMaterial.color.setHex(0xffffff);
          textures.push(albedo);
        }
        if (normal) {
          normal.repeat.set(repeatX, repeatY);
          floorMaterial.normalMap = normal;
          floorMaterial.normalScale.set(0.58, 0.58);
          textures.push(normal);
        }
        if (roughness) {
          roughness.repeat.set(repeatX, repeatY);
          floorMaterial.roughnessMap = roughness;
          textures.push(roughness);
        }
        floorMaterial.needsUpdate = true;
      } else {
        albedo?.dispose();
        normal?.dispose();
        roughness?.dispose();
      }

      worldRuntimeRef.current = {
        renderer,
        scene,
        camera,
        floorMaterial,
        wallMaterial,
        keyLight,
        fillLight,
        fireLight,
        textures,
      };

      const handleResize = () => {
        const runtime = worldRuntimeRef.current;
        if (!runtime) return;
        const aspect = window.innerWidth / Math.max(1, window.innerHeight);
        runtime.camera.aspect = aspect;
        runtime.camera.updateProjectionMatrix();
        runtime.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        runtime.renderer.setSize(window.innerWidth, window.innerHeight, false);
        worldNeedsRenderRef.current = true;
      };
      window.addEventListener('resize', handleResize);
      removeResize = () => window.removeEventListener('resize', handleResize);

      const renderWorldNow = () => {
        const runtime = worldRuntimeRef.current;
        if (destroyed || !runtime) return;

        const cam = cameraRef.current;
        runtime.camera.position.set(cam.pos[0], cam.pos[1], cam.pos[2]);
        runtime.camera.lookAt(cam.target[0], cam.target[1], cam.target[2]);

        const p = paramsRef.current;
        const dayBlend = clamp(1.0 - p.lightingFlicker, 0.0, 1.0);
        skyMix.copy(skyNight).lerp(skyDay, dayBlend);
        runtime.scene.background = skyMix;
        (runtime.scene.fog as THREE.Fog).color.copy(skyMix);
        runtime.renderer.toneMappingExposure = clamp(p.exposure * 0.95, 0.45, 1.8);

        runtime.keyLight.intensity = clamp(0.9 + p.floorAmbient * 1.8, 0.55, 3.4);
        runtime.fillLight.intensity = clamp(0.22 + p.floorAmbient * 1.0, 0.12, 1.5);

        runtime.fireLight.position.set(0.5, 0.22 + p.volumeHeight * 0.12, 0.5);
        runtime.fireLight.intensity = clamp(0.65 + p.lightingFireIntensity * 0.30 + p.emission * 0.08, 0.4, 8.0);
        runtime.fireLight.distance = clamp(1.6 + p.lightingFireFalloff * 1.8, 1.2, 8.8);

        runtime.floorMaterial.roughness = clamp(
          0.22 + (1.0 - p.floorSpecular) * 0.58 + p.floorSootRoughness * 0.08,
          0.12,
          0.98
        );
        runtime.floorMaterial.metalness = clamp(0.01 + p.floorSpecular * 0.24, 0.0, 0.4);
        runtime.floorMaterial.clearcoat = clamp(0.05 + p.floorSpecular * 0.36, 0.0, 0.82);
        runtime.floorMaterial.clearcoatRoughness = clamp(0.82 - p.floorSpecular * 0.68, 0.1, 0.95);
        runtime.floorMaterial.envMapIntensity = clamp(0.6 + p.floorSpecular * 1.2, 0.2, 2.8);

        runtime.wallMaterial.roughness = clamp(0.62 + p.smokeDarkness * 0.35, 0.45, 0.98);
        runtime.wallMaterial.metalness = clamp(0.02 + p.floorSpecular * 0.08, 0.0, 0.18);

        runtime.renderer.render(runtime.scene, runtime.camera);
      };
      renderWorldRef.current = renderWorldNow;
      worldNeedsRenderRef.current = true;
      renderWorldNow();
    };

    initWorld().catch((err) => {
      if (!destroyed) {
        setRuntimeWarning(`World pipeline init failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    });

    return () => {
      destroyed = true;
      removeResize();
      renderWorldRef.current = null;

      const runtime = worldRuntimeRef.current;
      if (runtime) {
        runtime.textures.forEach((texture) => texture.dispose());
        runtime.scene.traverse((node) => {
          const mesh = node as THREE.Mesh;
          const geometry = mesh.geometry as THREE.BufferGeometry | undefined;
          if (geometry?.dispose) geometry.dispose();
          const material = mesh.material as THREE.Material | THREE.Material[] | undefined;
          if (Array.isArray(material)) material.forEach((entry) => entry?.dispose?.());
          else material?.dispose?.();
        });
        runtime.renderer.dispose();
        worldRuntimeRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    updateCameraVectors();
    if (!canvasRef.current || !navigator.gpu) { setError(navigator.gpu ? null : "WebGPU not supported"); return; }
    let animationFrameId: number;
    let device: GPUDevice;
    let context: GPUCanvasContext;
    let isDestroyed = false;

    const init = async () => {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter || isDestroyed) throw new Error("No adapter");
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
          // Fallback to default limits for older/quirky implementations.
          device = await adapter.requestDevice();
        }
        if (isDestroyed) { device.destroy(); return; }

        const deviceLimits = ((device as any).limits ?? adapter.limits) as GPUSupportedLimits;
        const maxStorageBinding = deviceLimits.maxStorageBufferBindingSize ?? adapter.limits.maxStorageBufferBindingSize;
        const maxDimByStorageBinding = Math.floor(Math.cbrt(maxStorageBinding / 16));
        const supportedGrid = [256, 192, 128, 64].find((candidate) => candidate <= maxDimByStorageBinding) ?? 64;
        if (supportedGrid !== gridSize) {
          pushTimeline(`Grid ${gridSize} exceeded max storage binding; falling back to ${supportedGrid}.`);
          setGridSize(supportedGrid);
          device.destroy();
          return;
        }

        context = canvasRef.current!.getContext('webgpu') as any;
        const format = navigator.gpu.getPreferredCanvasFormat();
        context.configure({ device, format, alphaMode: 'premultiplied' });
        if (isDestroyed) {
          device.destroy();
          return;
        }

        const transport = new FluidTransport(device, gridSize);
        const computePipeline = device.createComputePipeline({ layout: transport.physicsContract.layout, compute: { module: device.createShaderModule({ code: COMPUTE_SHADER }), entryPoint: 'main' } });
        const projectionDivPipeline = device.createComputePipeline({ layout: transport.projectionDivContract.layout, compute: { module: device.createShaderModule({ code: PROJECTION_DIVERGENCE_SHADER }), entryPoint: 'main' } });
        const projectionJacobiPipeline = device.createComputePipeline({ layout: transport.projectionJacobiContract.layout, compute: { module: device.createShaderModule({ code: PROJECTION_JACOBI_SHADER }), entryPoint: 'main' } });
        const projectionGradPipeline = device.createComputePipeline({ layout: transport.projectionGradContract.layout, compute: { module: device.createShaderModule({ code: PROJECTION_GRADIENT_SHADER }), entryPoint: 'main' } });
        const renderPipeline = device.createRenderPipeline({ layout: transport.renderContract.layout, vertex: { module: device.createShaderModule({ code: RENDER_SHADER }), entryPoint: 'vert_main' }, fragment: { module: device.createShaderModule({ code: RENDER_SHADER }), entryPoint: 'frag_main', targets: [{ format }] }, primitive: { topology: 'triangle-list' } });

        let simFrame = 0;
        let activeRenderGroup = 0;
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
            } else {
              adaptiveStepScaleRef.current = 1.0;
            }
            rafCount = 0;
            dtAccum = 0;
            statsTimer = 0;
          }
          const shouldAdvance = playingRef.current || stepFramesRef.current > 0;
          const qualityBoost = qualityModeRef.current === 'accurate' ? 1.5 : 1.0;
          const adaptiveScale = qualityModeRef.current === 'accurate' ? 1.0 : adaptiveStepScaleRef.current;
          const stepQuality = clamp(paramsRef.current.stepQuality * qualityBoost * adaptiveScale, 0.25, 4.0);

          transport.updateUniforms(now, {
            ...paramsRef.current,
            timeStep: DEFAULT_TIME_STEP,
            stepQuality,
            scattering: smokeEnabledRef.current ? paramsRef.current.scattering : 0.0,
            absorption: smokeEnabledRef.current ? paramsRef.current.absorption : 0.0,
            renderWidth: canvasRef.current?.width ?? window.innerWidth,
            renderHeight: canvasRef.current?.height ?? window.innerHeight,
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

              const velocityTarget = stepIndex === 0 ? transport.velocityB : transport.velocityA;
              (enc as any).copyBufferToBuffer(velocityTarget, 0, transport.velocityScratch, 0, transport.velocityBufferSize);

              const divPass = enc.beginComputePass();
              divPass.setPipeline(projectionDivPipeline);
              divPass.setBindGroup(0, transport.projectionDivGroups[stepIndex]);
              divPass.dispatchWorkgroups(wc, wc, wc);
              divPass.end();

              const jacobiIterations = qualityModeRef.current === 'accurate' ? 10 : 5;
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
              simAccumulatorSeconds -= DEFAULT_TIME_STEP;

              if (!playingRef.current && stepFramesRef.current > 0) {
                stepFramesRef.current -= 1;
              }
            }

            if (substeps === maxSubsteps && simAccumulatorSeconds > DEFAULT_TIME_STEP * 2) {
              simAccumulatorSeconds = DEFAULT_TIME_STEP * 2;
            }

          }

          const rp = enc.beginRenderPass({ colorAttachments: [{ view: context.getCurrentTexture().createView(), clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }, loadOp: 'clear', storeOp: 'store' }] });
          rp.setPipeline(renderPipeline);
          rp.setBindGroup(0, transport.renderGroups[activeRenderGroup]);
          rp.draw(6);
          rp.end();
          device.queue.submit([enc.finish()]);
          if (worldNeedsRenderRef.current && renderWorldRef.current) {
            renderWorldRef.current();
            worldNeedsRenderRef.current = false;
          }
          animationFrameId = requestAnimationFrame(render);
        };
        render();
      } catch (e: any) { if (!isDestroyed) setError(e.message); }
    };
    init();
    return () => {
      isDestroyed = true;
      cancelAnimationFrame(animationFrameId);
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
    const worldCanvas = worldCanvasRef.current;
    const fireCanvas = canvasRef.current;
    if (!worldCanvas && !fireCanvas) return;

    const baseCanvas = worldCanvas ?? fireCanvas!;
    const composite = document.createElement('canvas');
    composite.width = baseCanvas.width;
    composite.height = baseCanvas.height;
    const ctx = composite.getContext('2d');
    if (!ctx) return;

    if (worldCanvas) {
      ctx.drawImage(worldCanvas, 0, 0, composite.width, composite.height);
    }
    if (fireCanvas) {
      ctx.drawImage(fireCanvas, 0, 0, composite.width, composite.height);
    }

    const link = document.createElement('a');
    link.href = composite.toDataURL('image/png');
    link.download = `firesim-frame-${Date.now()}.png`;
    link.click();
  };

const copyParamsToClipboard = useCallback(() => {
    const cpuUniformWriter = `// CPU-side SimParams uniform packing (DataView, little-endian)
// NOTE: WGSL uses vec4f for cameraPos/targetPos to avoid vec3 padding traps.
// Buffer size in this app: 288 bytes (fields used up through byte 268).
const uniformData = new ArrayBuffer(288);
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
f32(256, viewportWidth);
f32(260, viewportHeight);
f32(264, cameraAspect);
f32(268, cameraTanHalfFov);
`;

    const payload = {
      schema: 'firesim-params.v1',
      copiedAt: new Date().toISOString(),
      packingDebugPrompt: 'If you paste your CPU-side uniform-buffer write code (the part that creates the ArrayBuffer / Float32Array), I’ll tell you exactly which line is wrong and give you a corrected writer that matches the layout.',
      cpuUniformWriter,
      sceneId: selectedSceneId,
      gridSize,
      runtimeResolutionScale,
      smokeEnabled: isSmokeEnabled,
      qualityMode,
      params: { ...simParams, timeStep: DEFAULT_TIME_STEP },
    };
    void copyText(JSON.stringify(payload, null, 2), 'Copied params JSON');
  }, [copyText, gridSize, isSmokeEnabled, qualityMode, runtimeResolutionScale, selectedSceneId, simParams]);

  const copyAlgorithmToClipboard = useCallback(() => {
    const payload = `// firesim algorithm (WGSL)\n// copiedAt: ${new Date().toISOString()}\n\n// === COMPUTE_SHADER ===\n${COMPUTE_SHADER}\n\n// === PROJECTION_DIVERGENCE_SHADER ===\n${PROJECTION_DIVERGENCE_SHADER}\n\n// === PROJECTION_JACOBI_SHADER ===\n${PROJECTION_JACOBI_SHADER}\n\n// === PROJECTION_GRADIENT_SHADER ===\n${PROJECTION_GRADIENT_SHADER}\n\n// === RENDER_SHADER ===\n${RENDER_SHADER}\n`;
    void copyText(payload, 'Copied algorithm (WGSL)');
  }, [copyText]);

  const savePresetToDisk = () => {
    downloadJson(`firesim-preset-${Date.now()}.json`, {
      schema: 'firesim-preset.v1',
      savedAt: new Date().toISOString(),
      sceneId: selectedSceneId,
      gridSize,
      runtimeResolutionScale,
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
      if (typeof parsed.runtimeResolutionScale === 'number') {
        const nextScale = clamp(parsed.runtimeResolutionScale, 0.35, 1.0);
        setRuntimeResolutionScale(nextScale);
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
      runtimeResolutionScale,
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
    const isResolution = deckRailSection === 'resolution';
    const gridOptions = [64, 128, 192, 256] as const;

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
      (isAll || isResolution) && {
        id: 'renderResolution',
        label: 'Render Resolution',
        low: '35%',
        high: '100%',
        tone: 'cool' as const,
        value: clamp(inverseLerp(0.35, 1.0, runtimeResolutionScale), 0, 1),
        valueText: `${Math.round(runtimeResolutionScale * 100)}% (${internalCanvasSize.width}x${internalCanvasSize.height})`,
        onChange: (t: number) => {
          const next = clamp(lerp(0.35, 1.0, t), 0.35, 1.0);
          setRuntimeResolutionScale(next);
        },
      },
      (isAll || isResolution) && {
        id: 'simGrid',
        label: 'Simulation Grid',
        low: '64',
        high: '256',
        tone: 'cool' as const,
        value: clamp(inverseLerp(64, 256, gridSize), 0, 1),
        valueText: `${gridSize}³`,
        onChange: (t: number) => {
          const target = lerp(64, 256, t);
          let nextGrid: number = gridOptions[0];
          let bestDist = Math.abs(target - nextGrid);
          for (const opt of gridOptions) {
            const dist = Math.abs(target - opt);
            if (dist < bestDist) {
              bestDist = dist;
              nextGrid = opt;
            }
          }
          setGridSize(nextGrid);
        },
      },
      (isAll || isResolution) && {
        id: 'qualityMode',
        label: 'Quality Mode',
        low: 'Realtime',
        high: 'Accurate',
        tone: 'mono' as const,
        value: qualityMode === 'accurate' ? 1 : 0,
        valueText: qualityMode === 'accurate' ? 'Accurate' : 'Realtime',
        onChange: (t: number) => setQualityMode(t >= 0.5 ? 'accurate' : 'realtime'),
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
        label: 'Sky Time',
        low: 'Day',
        high: 'Night',
        tone: 'cool' as const,
        value: clamp(simParams.lightingFlicker, 0, 1),
        valueText: `${simParams.lightingFlicker.toFixed(2)} ${simParams.lightingFlicker < 0.5 ? 'Day' : 'Night'}`,
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
  }, [
    applyMacro,
    burnRate,
    deckRailSection,
    gridSize,
    internalCanvasSize.height,
    internalCanvasSize.width,
    qualityMode,
    runtimeResolutionScale,
    simParams,
    smokeAmount,
    smokeDarkness,
    turbulenceCharacter,
    tuningMacroKnobs,
    updateParam,
    windGusts
  ]);

  if (error) return <div className="deck-error" role="alert">{error}</div>;

  return (
    <div className="deck-root">
      <canvas
        ref={worldCanvasRef}
        width={Math.max(1, Math.round(dimensions.width * dimensions.pixelRatio))}
        height={Math.max(1, Math.round(dimensions.height * dimensions.pixelRatio))}
        className="deck-canvas deck-world-canvas"
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
      <canvas
        ref={canvasRef}
        width={internalCanvasSize.width}
        height={internalCanvasSize.height}
        className="deck-canvas deck-fire-overlay"
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
          <button type="button" className={`deck-rail-btn ${deckRailSection === 'resolution' ? 'is-active' : ''}`} aria-label="Resolution" onClick={() => setDeckRailSection('resolution')}><Monitor size={18} /></button>
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
