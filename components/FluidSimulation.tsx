
import { Settings, Play, Pause, RefreshCw, Maximize2, RotateCcw, ChevronDown, Copy, Check, Wind, Zap, Thermometer, Box } from 'lucide-react';
import React, { useEffect, useRef, useState } from 'react';

// --- WebGPU Type Declarations ---
declare global {
  interface Navigator {
    gpu: any;
  }
  var GPUBufferUsage: any;
  var GPUShaderStage: any;
}
type GPUBuffer = any;
type GPUDevice = any;
type GPUPipelineLayout = any;
type GPUBindGroupLayout = any;
type GPUBindGroupLayoutEntry = any;
type GPUBindGroup = any;
type GPUBindGroupEntry = any;
type GPUCanvasContext = any;
type GPUAdapter = any;
type GPUCommandEncoder = any;
type GPUComputePassEncoder = any;
type GPURenderPassEncoder = any;
type GPUTexture = any;
type GPUTextureView = any;

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

class FluidTransport {
  public uniformBuffer: GPUBuffer;
  
  public densityA: GPUBuffer;
  public densityB: GPUBuffer;
  public velocityA: GPUBuffer;
  public velocityB: GPUBuffer;

  public physicsContract: ShaderContract;
  public renderContract: ShaderContract;

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
    
    this.velocityA = device.createBuffer({ size: VOXEL_COUNT * 16, usage: storageUsage });
    this.velocityB = device.createBuffer({ size: VOXEL_COUNT * 16, usage: storageUsage });

    const zeroF32 = new Float32Array(VOXEL_COUNT);
    const zeroVec4 = new Float32Array(VOXEL_COUNT * 4);
    
    device.queue.writeBuffer(this.densityA, 0, zeroF32);
    device.queue.writeBuffer(this.densityB, 0, zeroF32);
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
    ]);

    this.renderContract = new ShaderContract(this.device, 'Render', [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    ]);
  }

  private initBindGroups() {
    this.physicsGroups[0] = this.physicsContract.createBindGroup(this.device, 'Phys0', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.densityA } },
      { binding: 2, resource: { buffer: this.densityB } },
      { binding: 3, resource: { buffer: this.velocityA } },
      { binding: 4, resource: { buffer: this.velocityB } },
    ]);

    this.renderGroups[0] = this.renderContract.createBindGroup(this.device, 'Render0', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.densityB } },
      { binding: 2, resource: { buffer: this.velocityB } },
    ]);

    this.physicsGroups[1] = this.physicsContract.createBindGroup(this.device, 'Phys1', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.densityB } },
      { binding: 2, resource: { buffer: this.densityA } },
      { binding: 3, resource: { buffer: this.velocityB } },
      { binding: 4, resource: { buffer: this.velocityA } },
    ]);

    this.renderGroups[1] = this.renderContract.createBindGroup(this.device, 'Render1', [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: this.densityA } },
      { binding: 2, resource: { buffer: this.velocityA } },
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
    view.setFloat32(76, 0, true); // pad2

    view.setFloat32(80, camera.target[0], true);
    view.setFloat32(84, camera.target[1], true);
    view.setFloat32(88, camera.target[2], true);
    view.setFloat32(92, 0, true); // pad3

    view.setFloat32(96, params.windX || 0, true);
    view.setFloat32(100, params.windZ || 0, true);
    view.setFloat32(104, params.turbFreq || 28.0, true);
    view.setFloat32(108, params.turbSpeed || 1.0, true);
    view.setFloat32(112, params.fuelEfficiency || 1.0, true);
    view.setFloat32(116, params.heatDiffusion || 0.0, true);
    view.setFloat32(120, params.stepQuality || 1.0, true);

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

fn get_idx(p: vec3i) -> u32 {
  let d = i32(params.dim);
  let cp = clamp(p, vec3i(0), vec3i(d - 1));
  return u32(cp.z * d * d + cp.y * d + cp.x);
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
  matter: f32, 
  temp: f32,   
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
  return State(vm.xyz, vm.w, fm);
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
  let backPos = uvw - vel * dt * (1.3 / params.dim) * 12.0;
  var state = sample_state(backPos);
  
  var newVel = state.vel;
  var temp = state.temp;
  var matter = state.matter;

  // External forces: Wind
  newVel += vec3f(params.windX, 0.0, params.windZ) * dt * 40.0;
  
  // Heat Diffusion (Simple kernel approximation)
  if (params.heatDiffusion > 0.0) {
      let T_avg = (densityIn[get_idx(vec3i(id) + vec3i(1,0,0))] + densityIn[get_idx(vec3i(id) - vec3i(1,0,0))] +
                   densityIn[get_idx(vec3i(id) + vec3i(0,1,0))] + densityIn[get_idx(vec3i(id) - vec3i(0,1,0))] +
                   densityIn[get_idx(vec3i(id) + vec3i(0,0,1))] + densityIn[get_idx(vec3i(id) - vec3i(0,0,1))]) / 6.0;
      temp = mix(temp, T_avg, params.heatDiffusion * dt * 10.0);
  }
  
  let localOxygen = smoothstep(1.2, 0.2, matter);
  let insulation_factor = smoothstep(0.12, 0.0, woodDist);
  let insulation = mix(params.dissipation, 1.0, insulation_factor * 0.88);
  temp *= insulation; 
  let burnt = max(0.0, state.temp - temp);
  matter = matter * params.smokeDissipation + burnt * 2.8; 
  
  let buoyancyDir = vec3f(0.0, 1.0, 0.0);
  let thermalLift = (pow(temp, 1.25) * 0.4) - (matter * params.smokeWeight * 0.0018);
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
         let fuelAvailability = logSurfaceZone;
         let combustionIntensity = smoothstep(-0.2, 0.8, n_val) * fuelAvailability * localOxygen;
         let emissionRate = params.emission * (1.0 + crack_mask * 2.5) * params.fuelEfficiency;
         let intensity = combustionIntensity * emissionRate * 0.3; 
         let damping = exp(-temp * 2.0); 
         temp += intensity * 1.2 * damping; 
         matter += intensity * 1.5 * damping;
         let logNorm = get_wood_normal(uvw);
         newVel += normalize(logNorm * 0.35 + vec3f(0.0, 2.5, 0.0)) * intensity * 160.0 * dt;
     }
  } else {
     let d_emit = length(uvw - vec3f(0.5, 0.2, 0.5));
     if (d_emit < 0.05) {
        temp += params.emission * dt * 8.0;
        newVel += vec3f(0.0, 0.8, 0.0) * params.emission * dt;
     }
  }
  
  if (usesWood && woodDist < 0.0) {
      let friction = smoothstep(0.0, 0.015, -woodDist); 
      newVel *= (1.0 - friction * 0.2);
      temp *= (1.0 - friction * 0.015);
  }
  
  let b_dist = min(min(uvw.x, 1.0 - uvw.x), min(min(uvw.y, 1.0 - uvw.y), min(uvw.z, 1.0 - uvw.z)));
  let edge_damp = smoothstep(0.0, 0.06, b_dist);
  temp *= edge_damp; matter *= edge_damp; newVel *= edge_damp;
  
  densityOut[idx] = temp;
  velocityOut[idx] = vec4f(clamp(newVel, vec3f(-120.0), vec3f(120.0)), matter);
}
`;

const RENDER_SHADER = `
${STRUCT_DEF}
${WOOD_SDF_FN}
@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> densityIn: array<f32>; 
@group(0) @binding(2) var<storage, read> velocityIn: array<vec4f>; 

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

fn sample_volume(pos: vec3f) -> vec2f {
  let d = params.dim; let p = pos * d - 0.5; let i = vec3i(floor(p)); let f = fract(p);
  let f_res = mix(mix(mix(densityIn[get_idx(i)], densityIn[get_idx(i + vec3i(1,0,0))], f.x), mix(densityIn[get_idx(i + vec3i(0,1,0))], densityIn[get_idx(i + vec3i(1,1,0))], f.x), f.y), mix(mix(densityIn[get_idx(i + vec3i(0,0,1))], densityIn[get_idx(i + vec3i(1,0,1))], f.x), mix(densityIn[get_idx(i + vec3i(0,1,1))], densityIn[get_idx(i + vec3i(1,1,1))], f.x), f.y), f.z);
  let s_res = mix(mix(mix(velocityIn[get_idx(i)].w, velocityIn[get_idx(i + vec3i(1,0,0))].w, f.x), mix(velocityIn[get_idx(i + vec3i(0,1,0))].w, velocityIn[get_idx(i + vec3i(1,1,0))].w, f.x), f.y), mix(mix(velocityIn[get_idx(i + vec3i(0,0,1))].w, velocityIn[get_idx(i + vec3i(1,0,1))].w, f.x), mix(velocityIn[get_idx(i + vec3i(0,1,1))].w, velocityIn[get_idx(i + vec3i(1,1,1))].w, f.x), f.y), f.z);
  return vec2f(f_res, s_res);
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
    var p = pos; let step = 0.045; var densitySum = 0.0;
    for(var i=0; i<12; i++) {
        p += lightDir * step; if (p.x < 0.0 || p.x > 1.0 || p.y > 1.0 || p.z < 0.0 || p.z > 1.0 || p.y < 0.0) { break; }
        let val = sample_volume(p); densitySum += val.y * params.absorption + val.x * 0.15; 
    }
    return exp(-densitySum * step * 18.0);
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
            if (any(p < vec3f(0.0)) || any(p > vec3f(1.0))) { break; }
            if (transmittance < 0.02) { break; }

            let val = sample_volume(p);

            // Fire emits light downward to floor
            if (val.x > 0.02) {
                let emission = getBlackbodyColor(val.x * params.emission * 0.3);
                let atten = 1.0 / (1.0 + t * t * 5.0);
                totalLight += emission * val.x * atten * transmittance * 0.12;
            }

            // Smoke blocks light
            let density = val.y * 1.5 + val.x * 0.15;
            transmittance *= exp(-density * 0.06 * params.absorption * 0.3);

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
  let t = intersectAABB(ro, rd, vec3f(0.0), vec3f(1.0));
  
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
    let mapped = clamp(floor_color * params.exposure, vec3f(0.0), vec3f(1.0));
    return vec4f(pow(mapped, vec3f(1.0 / params.gamma)), 1.0);
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
  let steps = i32(f32(baseSteps) * params.stepQuality);
  let stepSize = (tVolumeFar - tNear) / f32(steps); var pos = ro + rd * (tNear + stepSize * jitter); 
  var accumCol = vec3f(0.0); var transmittance = 1.0; let phaseSun = phase_function(dot(rd, lightDir), 0.85);
  
  for (var i = 0; i < steps; i++) {
      if (transmittance < 0.005) { break; }
      if (all(pos >= vec3f(0.0)) && all(pos <= vec3f(1.0))) {
           let val = sample_volume(pos);
           // No boundary fade - fire extends naturally
           let total_d = val.y * 1.8 + val.x * 0.15;
           if (total_d > 0.0001) {
               let sunTrans = get_light_transmittance(pos, lightDir);
               let radiance = vec3f(12.0) * sunTrans * phaseSun * mix(vec3f(0.6, 0.65, 0.7), vec3f(0.2, 0.18, 0.15), clamp(val.y * 1.8 * 1.2, 0.0, 1.0)) * params.scattering + getBlackbodyColor(val.x * params.emission);
               let stepTrans = exp(-((val.y * 1.8 * params.absorption * 22.0) + (val.x * 0.4)) * stepSize);
               accumCol += radiance * (1.0 - stepTrans) * transmittance; transmittance *= stepTrans;
           }
      }
      pos += rd * stepSize;
  }
  
  // Use same floor_color calculated at start
  let surfaceCol = select(floor_color, solidColor, hasSolid);
  let mapped = clamp((accumCol + transmittance * surfaceCol) * params.exposure, vec3f(0.0), vec3f(1.0));
  return vec4f(pow(mapped, vec3f(1.0 / params.gamma)), 1.0);
}
`;

const SCENES = [
  { id: 0, name: 'Campfire', params: { vorticity: 3.4, dissipation: 0.936, buoyancy: 1.5, drag: 0.0, emission: 8.4, scattering: 2.9, absorption: 26.5, smokeWeight: -2.0, plumeTurbulence: 10.0, smokeDissipation: 0.92, exposure: 0.9, gamma: 2.2, windX: 0.0, windZ: 0.0, turbFreq: 28.0, turbSpeed: 1.0, fuelEfficiency: 1.0, heatDiffusion: 0.0, stepQuality: 1.0 } },
  { id: 4, name: 'Wood Combustion', params: { vorticity: 2.2, dissipation: 0.903, buoyancy: 1.8, drag: 0.037, emission: 1.9, scattering: 6.5, absorption: 12.0, smokeWeight: 0.5, plumeTurbulence: 2.81, smokeDissipation: 0.85, windX: -0.05, windZ: 0.05, turbFreq: 15.0, turbSpeed: 2.5, fuelEfficiency: 1.5, heatDiffusion: 0.1, stepQuality: 1.0 } },
  { id: 1, name: 'Candle', params: { vorticity: 3.5, dissipation: 0.92, buoyancy: 4.5, drag: 0.08, emission: 1.0, scattering: 2.5, absorption: 2.0, smokeWeight: 0.3, plumeTurbulence: 0.05, smokeDissipation: 0.985, windX: 0.0, windZ: 0.0, turbFreq: 45.0, turbSpeed: 0.2, fuelEfficiency: 0.5, heatDiffusion: 0.0, stepQuality: 1.5 } },
  { id: 2, name: 'Dual Source', params: { vorticity: 15.0, dissipation: 0.985, buoyancy: 8.0, drag: 0.02, emission: 1.8, scattering: 4.5, absorption: 4.0, smokeWeight: 1.5, plumeTurbulence: 0.3, smokeDissipation: 0.992, windX: 0.2, windZ: 0.2, turbFreq: 20.0, turbSpeed: 1.5, fuelEfficiency: 1.0, heatDiffusion: 0.05, stepQuality: 1.0 } },
  { id: 3, name: 'Firebending', params: { vorticity: 12.0, dissipation: 0.965, buoyancy: 5.0, drag: 0.002, emission: 3.0, scattering: 3.5, absorption: 1.5, smokeWeight: 0.5, plumeTurbulence: 0.8, smokeDissipation: 0.98, windX: 0.0, windZ: 0.0, turbFreq: 32.0, turbSpeed: 5.0, fuelEfficiency: 1.2, heatDiffusion: 0.0, stepQuality: 0.8 } },
  { id: 5, name: 'Gas Explosion', params: { vorticity: 35.0, dissipation: 0.94, buoyancy: 16.0, drag: 0.01, emission: 6.0, scattering: 4.0, absorption: 1.0, smokeWeight: -0.5, plumeTurbulence: 1.5, smokeDissipation: 0.92, windX: 0.0, windZ: 0.0, turbFreq: 12.0, turbSpeed: 0.5, fuelEfficiency: 3.0, heatDiffusion: 0.2, stepQuality: 1.0 } },
  { id: 6, name: 'Nuke', params: { vorticity: 50.0, dissipation: 0.998, buoyancy: 3.0, drag: 0.05, emission: 6.5, scattering: 8.0, absorption: 7.0, smokeWeight: 3.0, plumeTurbulence: 0.4, smokeDissipation: 0.999, windX: 0.0, windZ: 0.0, turbFreq: 8.0, turbSpeed: 0.1, fuelEfficiency: 5.0, heatDiffusion: 0.5, stepQuality: 1.2 } }
];

const FluidSimulation: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [controlsVisible, setControlsVisible] = useState(true);
  const [isPlaying, setIsPlaying] = useState(true);
  const [isSmokeEnabled, setIsSmokeEnabled] = useState(true);
  const [stats, setStats] = useState({ fps: 0, ms: 0 });
  const [selectedSceneId, setSelectedSceneId] = useState(0);
  const [copied, setCopied] = useState(false);
  const [dimensions, setDimensions] = useState({ width: window.innerWidth, height: window.innerHeight });
  const [gridSize, setGridSize] = useState(128);
  const [simParams, setSimParams] = useState({ ...SCENES[0].params, timeStep: 0.016 });
  const cameraRef = useRef({ theta: 1.625, phi: 1.35, radius: 1.25, target: [0.5, 0.4, 0.5] as [number, number, number], pos: [0.45, 0.38, 1.3] as [number, number, number] });
  const interactionRef = useRef({ isDragging: false, lastX: 0, lastY: 0, button: 0 });
  const paramsRef = useRef(simParams);
  const playingRef = useRef(isPlaying);
  const sceneRef = useRef(selectedSceneId);
  const smokeEnabledRef = useRef(isSmokeEnabled);

  useEffect(() => { paramsRef.current = simParams; }, [simParams]);
  useEffect(() => { playingRef.current = isPlaying; }, [isPlaying]);
  useEffect(() => { sceneRef.current = selectedSceneId; }, [selectedSceneId]);
  useEffect(() => { smokeEnabledRef.current = isSmokeEnabled; }, [isSmokeEnabled]);

  useEffect(() => {
    const handleResize = () => setDimensions({ width: window.innerWidth, height: window.innerHeight });
    window.addEventListener('resize', handleResize); return () => window.removeEventListener('resize', handleResize);
  }, []);

  const handleSceneChange = (id: number) => {
    setSelectedSceneId(id);
    const scene = SCENES.find(s => s.id === id);
    if (scene) setSimParams(prev => ({ ...prev, ...scene.params, timeStep: 0.016 }));
  };

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

        const transport = new FluidTransport(device, gridSize);
        const computePipeline = device.createComputePipeline({ layout: transport.physicsContract.layout, compute: { module: device.createShaderModule({ code: COMPUTE_SHADER }), entryPoint: 'main' } });
        const renderPipeline = device.createRenderPipeline({ layout: transport.renderContract.layout, vertex: { module: device.createShaderModule({ code: RENDER_SHADER }), entryPoint: 'vert_main' }, fragment: { module: device.createShaderModule({ code: RENDER_SHADER }), entryPoint: 'frag_main', targets: [{ format }] }, primitive: { topology: 'triangle-list' } });

        let frame = 0; let lastTime = performance.now(); let fpsTimer = 0; let frameCount = 0;
        const render = () => {
          if (isDestroyed) return;
          const now = performance.now(); const dt = now - lastTime; lastTime = now;
          frameCount++; fpsTimer += dt; if (fpsTimer >= 1000) { setStats({ fps: frameCount, ms: 1000 / frameCount }); frameCount = 0; fpsTimer = 0; }
          if (playingRef.current) {
            transport.updateUniforms(now, {
              ...paramsRef.current,
              scattering: smokeEnabledRef.current ? paramsRef.current.scattering : 0.0,
              absorption: smokeEnabledRef.current ? paramsRef.current.absorption : 0.0
            }, { pos: cameraRef.current.pos, target: cameraRef.current.target }, sceneRef.current);
            const enc = device.createCommandEncoder();
            const cp = enc.beginComputePass(); cp.setPipeline(computePipeline); cp.setBindGroup(0, transport.physicsGroups[frame % 2]);
            const wc = Math.ceil(transport.dim / 4); cp.dispatchWorkgroups(wc, wc, wc); cp.end();
            const rp = enc.beginRenderPass({ colorAttachments: [{ view: context.getCurrentTexture().createView(), clearValue: { r: 0.22, g: 0.22, b: 0.24, a: 1 }, loadOp: 'clear', storeOp: 'store' }] });
            rp.setPipeline(renderPipeline); rp.setBindGroup(0, transport.renderGroups[frame % 2]); rp.draw(6); rp.end();
            device.queue.submit([enc.finish()]); frame++;
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

  if (error) return <div className="p-8 text-red-500 font-mono">{error}</div>;

  return (
    <div className="relative w-full h-full bg-[#38383d] overflow-hidden font-sans selection:bg-orange-500/30 selection:text-orange-100">
      <canvas ref={canvasRef} width={dimensions.width} height={dimensions.height} className="block w-full h-full cursor-crosshair touch-none" 
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
      
      <div className="absolute top-3 right-3 pointer-events-none">
          <div className="bg-black/50 backdrop-blur-sm rounded px-3 py-2 text-[10px] text-white/60 font-mono space-y-1">
              <div className="flex justify-between gap-4"><span className="text-white/40">RESOLUTION</span><span>{gridSize}³</span></div>
              <div className="flex justify-between gap-4"><span className="text-white/40">FPS</span><span className="text-orange-400">{stats.fps}</span></div>
          </div>
      </div>

      <div className="absolute top-3 left-3">
        <div className="bg-black/50 backdrop-blur-sm rounded px-3 py-2 min-w-[180px]">
             <div className="flex items-center justify-between mb-2">
               <span className="text-[10px] font-mono text-white/40 uppercase tracking-wider">Fire Sandbox</span>
               <div className="flex gap-1">
                   <button onClick={() => handleSceneChange(selectedSceneId)} className="p-1 hover:bg-white/10 rounded text-white/40 transition-colors"><RefreshCw size={10}/></button>
                   <button onClick={() => setIsPlaying(!isPlaying)} className={`p-1 rounded transition-colors ${isPlaying ? 'text-orange-400' : 'text-white/40'}`}>{isPlaying ? <Pause size={10}/> : <Play size={10}/>}</button>
                   <button onClick={() => setControlsVisible(!controlsVisible)} className={`p-1 rounded transition-colors ${controlsVisible ? 'text-orange-400' : 'text-white/40'}`}><Settings size={10}/></button>
               </div>
             </div>

             <select value={selectedSceneId} onChange={e => handleSceneChange(Number(e.target.value))} className="w-full bg-white/5 border border-white/10 text-white/70 text-[10px] font-mono rounded px-2 py-1 outline-none mb-2">
                {SCENES.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
             </select>
             <select value={gridSize} onChange={e => setGridSize(Number(e.target.value))} className="w-full bg-white/5 border border-white/10 text-white/70 text-[10px] font-mono rounded px-2 py-1 outline-none">
                <option value="64">64³</option><option value="128">128³</option><option value="192">192³</option><option value="256">256³</option>
             </select>
        </div>
      </div>

      <div className={`absolute top-24 left-3 transition-all duration-300 ${controlsVisible ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
        <div className="bg-black/50 backdrop-blur-sm rounded px-3 py-2 min-w-[180px] max-h-[70vh] overflow-y-auto text-[10px]">
             <div className="space-y-3">
                 
                 <ControlGroup title="Fluid Mechanics" icon={<Wind size={12}/>}>
                    <Slider label="Buoyancy" value={simParams.buoyancy} min={0} max={25} onChange={v => setSimParams(p => ({...p, buoyancy: v}))} />
                    <Slider label="Heat Decay" value={simParams.dissipation} min={0.9} max={0.999} step={0.001} onChange={v => setSimParams(p => ({...p, dissipation: v}))} />
                    <Slider label="Vorticity" value={simParams.vorticity} min={0} max={60} onChange={v => setSimParams(p => ({...p, vorticity: v}))} />
                    <Slider label="Plume Turbulence" value={simParams.plumeTurbulence} min={0} max={20} step={0.01} onChange={v => setSimParams(p => ({...p, plumeTurbulence: v}))} />
                    <Slider label="Drag / Friction" value={simParams.drag} min={0} max={0.2} step={0.001} onChange={v => setSimParams(p => ({...p, drag: v}))} />
                 </ControlGroup>

                 <ControlGroup title="Environmental forces" icon={<Box size={12}/>}>
                    <Slider label="Wind X-Force" value={simParams.windX} min={-0.5} max={0.5} step={0.01} onChange={v => setSimParams(p => ({...p, windX: v}))} />
                    <Slider label="Wind Z-Force" value={simParams.windZ} min={-0.5} max={0.5} step={0.01} onChange={v => setSimParams(p => ({...p, windZ: v}))} />
                    <Slider label="Turbulence Frequency" value={simParams.turbFreq} min={1} max={100} step={1} onChange={v => setSimParams(p => ({...p, turbFreq: v}))} />
                    <Slider label="Turbulence Speed" value={simParams.turbSpeed} min={0} max={10} step={0.1} onChange={v => setSimParams(p => ({...p, turbSpeed: v}))} />
                 </ControlGroup>
                 
                 <ControlGroup title="Matter Properties" icon={<Zap size={12}/>}>
                    <Toggle label="Visible Smoke (Matter)" checked={isSmokeEnabled} onChange={setIsSmokeEnabled} />
                    <Slider label="Matter Decay" value={simParams.smokeDissipation} min={0.0} max={0.999} step={0.001} onChange={v => setSimParams(p => ({...p, smokeDissipation: v}))} />
                    <Slider label="Mass Coefficient" value={simParams.smokeWeight} min={-5} max={15} onChange={v => setSimParams(p => ({...p, smokeWeight: v}))} />
                    <Slider label="Heat Emission" value={simParams.emission} min={0} max={25} onChange={v => setSimParams(p => ({...p, emission: v}))} />
                    <Slider label="Fuel Efficiency" value={simParams.fuelEfficiency} min={0.1} max={10} step={0.1} onChange={v => setSimParams(p => ({...p, fuelEfficiency: v}))} />
                    <Slider label="Heat Diffusion" value={simParams.heatDiffusion} min={0} max={1.0} step={0.01} onChange={v => setSimParams(p => ({...p, heatDiffusion: v}))} />
                 </ControlGroup>
                 
                 <ControlGroup title="Optical Pipeline" icon={<Thermometer size={12}/>}>
                    <Slider label="Scattering (Albedo)" value={simParams.scattering} min={0} max={25} onChange={v => setSimParams(p => ({...p, scattering: v}))} />
                    <Slider label="Absorption (Density)" value={simParams.absorption} min={0} max={100} onChange={v => setSimParams(p => ({...p, absorption: v}))} />
                    <Slider label="Tone Exposure" value={simParams.exposure} min={0.1} max={10.0} onChange={v => setSimParams(p => ({...p, exposure: v}))} />
                    <Slider label="Ray Step Quality" value={simParams.stepQuality} min={0.25} max={4.0} step={0.25} onChange={v => setSimParams(p => ({...p, stepQuality: v}))} />
                 </ControlGroup>
             </div>
        </div>
      </div>

      <div className="absolute bottom-3 right-3 text-[9px] text-white/20 font-mono pointer-events-none">
        ◆
      </div>
    </div>
  );
};

const ControlGroup: React.FC<{title: string, icon?: React.ReactNode, children: React.ReactNode}> = ({ title, icon, children }) => (
  <div className="space-y-2">
    <div className="flex items-center gap-1.5 text-[9px] font-mono text-white/30 uppercase tracking-wider">
        {icon}
        {title}
    </div>
    <div className="space-y-2">{children}</div>
  </div>
);

const Toggle: React.FC<{label: string, checked: boolean, onChange: (v: boolean) => void}> = ({ label, checked, onChange }) => (
  <div className="flex items-center justify-between text-[9px] text-white/50 cursor-pointer select-none" onClick={() => onChange(!checked)}>
    <span>{label}</span>
    <div className={`relative w-6 h-3 rounded-full transition-colors ${checked ? 'bg-orange-500/80' : 'bg-white/20'}`}>
      <div className={`absolute top-0.5 h-2 w-2 rounded-full bg-white transition-all ${checked ? 'left-[13px]' : 'left-0.5'}`}></div>
    </div>
  </div>
);

const Slider: React.FC<{label: string, value: number, min: number, max: number, step?: number, onChange: (v: number) => void}> = ({ label, value, min, max, step=0.1, onChange }) => (
  <div className="group relative">
    <div className="flex justify-between text-[9px] mb-1 text-white/40">
      <span>{label}</span>
      <span className="font-mono text-white/60">{value.toFixed(step < 0.01 ? 3 : 2)}</span>
    </div>
    <div className="relative h-0.5 w-full bg-white/10 rounded-full overflow-hidden">
      <div className="absolute top-0 left-0 h-full bg-orange-500/60" style={{width: `${((value - min) / (max - min)) * 100}%`}}></div>
    </div>
    <input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(parseFloat(e.target.value))} className="absolute top-0 left-0 w-full h-full opacity-0 cursor-ew-resize" />
  </div>
);

export default FluidSimulation;
