
import {
    AlertTriangle,
    Camera,
    ChevronDown,
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

const DEFAULT_TIME_STEP = 0.016;

type SimParamState = typeof SCENES[number]['params'] & { timeStep: number };
type EditableParamKey = Exclude<keyof SimParamState, 'timeStep'>;
type ParamGroup = 'fluid' | 'environment' | 'matter' | 'optics';
type ParamScale = 'linear' | 'log';
type MacroId = 'flameStyle' | 'smokeDensity' | 'convectionScale' | 'turbulenceCharacter';
type ControlSection = 'macros' | ParamGroup;
type DiagnosticsTab = 'runtime' | 'performance' | 'overlays' | 'logs';
type QualityMode = 'realtime' | 'accurate';
type PresetSlot = 'A' | 'B';

type DeckRailSection = 'home' | 'flame' | 'smoke' | 'convection' | 'turbulence' | 'library';

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
  { key: 'buoyancy', label: 'Buoyancy', group: 'fluid', min: 0, max: 25, step: 0.1, unit: 'm/s^2*', hint: 'Upward thermal lift coefficient.' },
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
  { key: 'heatDiffusion', label: 'Heat Diffusion', group: 'matter', min: 0, max: 1, step: 0.01, unit: 'm^2/s*', hint: 'Thermal spread factor.' },
  { key: 'scattering', label: 'Scattering', group: 'optics', min: 0, max: 25, step: 0.1, unit: '1/m*', hint: 'Forward light scatter.' },
  { key: 'absorption', label: 'Absorption', group: 'optics', min: 0, max: 100, step: 0.1, unit: '1/m*', hint: 'Light energy removal.' },
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
  const [simParams, setSimParams] = useState<SimParamState>({ ...INITIAL_PARAMS });
  const [lockedParams, setLockedParams] = useState<Record<EditableParamKey, boolean>>(() => (
    Object.fromEntries(PARAM_SPECS.map((spec) => [spec.key, false])) as Record<EditableParamKey, boolean>
  ));
  const cameraRef = useRef({ theta: 1.625, phi: 1.35, radius: 1.25, target: [0.5, 0.4, 0.5] as [number, number, number], pos: [0.45, 0.38, 1.3] as [number, number, number] });
  const interactionRef = useRef({ isDragging: false, lastX: 0, lastY: 0, button: 0 });
  const paramsRef = useRef(simParams);
  const playingRef = useRef(isPlaying);
  const sceneRef = useRef(selectedSceneId);
  const smokeEnabledRef = useRef(isSmokeEnabled);
  const qualityModeRef = useRef(qualityMode);
  const stepFramesRef = useRef(0);
  const randomSeedRef = useRef(1337);
  const timelineIdRef = useRef(1);

  useEffect(() => { paramsRef.current = simParams; }, [simParams]);
  useEffect(() => { playingRef.current = isPlaying; }, [isPlaying]);
  useEffect(() => { sceneRef.current = selectedSceneId; }, [selectedSceneId]);
  useEffect(() => { smokeEnabledRef.current = isSmokeEnabled; }, [isSmokeEnabled]);
  useEffect(() => { qualityModeRef.current = qualityMode; }, [qualityMode]);

  const pushTimeline = useCallback((message: string) => {
    const item: TimelineEvent = { id: timelineIdRef.current, at: Date.now(), message };
    timelineIdRef.current += 1;
    setTimeline((prev) => [item, ...prev].slice(0, 60));
  }, []);

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
      setSimParams((prev) => ({
        ...prev,
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
    setSimParams((prev) => ({
      ...prev,
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
        assign('smokeDissipation', lerp(0.995, 0.9, t));
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

        let simFrame = 0;
        let lastTime = performance.now();
        let statsTimer = 0;
        let dtAccum = 0;
        let rafCount = 0;
        const render = () => {
          if (isDestroyed) return;
          const now = performance.now(); const dt = now - lastTime; lastTime = now;
          rafCount += 1;
          dtAccum += dt;
          statsTimer += dt;
          if (statsTimer >= 1000) {
            const safeCount = Math.max(1, rafCount);
            setStats({ fps: safeCount, frameTimeMs: dtAccum / safeCount, frame: simFrame });
            rafCount = 0;
            dtAccum = 0;
            statsTimer = 0;
          }
          const shouldAdvance = playingRef.current || stepFramesRef.current > 0;
          if (shouldAdvance) {
            transport.updateUniforms(now, {
              ...paramsRef.current,
              timeStep: DEFAULT_TIME_STEP,
              stepQuality: qualityModeRef.current === 'accurate'
                ? Math.min(4, paramsRef.current.stepQuality * 1.35)
                : paramsRef.current.stepQuality,
              scattering: smokeEnabledRef.current ? paramsRef.current.scattering : 0.0,
              absorption: smokeEnabledRef.current ? paramsRef.current.absorption : 0.0
            }, { pos: cameraRef.current.pos, target: cameraRef.current.target }, sceneRef.current);
            const enc = device.createCommandEncoder();
            const cp = enc.beginComputePass(); cp.setPipeline(computePipeline); cp.setBindGroup(0, transport.physicsGroups[simFrame % 2]);
            const wc = Math.ceil(transport.dim / 4); cp.dispatchWorkgroups(wc, wc, wc); cp.end();
            const rp = enc.beginRenderPass({ colorAttachments: [{ view: context.getCurrentTexture().createView(), clearValue: { r: 0.22, g: 0.22, b: 0.24, a: 1 }, loadOp: 'clear', storeOp: 'store' }] });
            rp.setPipeline(renderPipeline); rp.setBindGroup(0, transport.renderGroups[simFrame % 2]); rp.draw(6); rp.end();
            device.queue.submit([enc.finish()]);
            if (stepFramesRef.current > 0) stepFramesRef.current -= 1;
            simFrame++;
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
    const canvas = canvasRef.current;
    if (!canvas) return;
    const link = document.createElement('a');
    link.href = canvas.toDataURL('image/png');
    link.download = `firesim-frame-${Date.now()}.png`;
    link.click();
  };

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

  if (error) return <div className="deck-error" role="alert">{error}</div>;

  const cameraAngleDeg = ((cameraRef.current.theta * 180) / Math.PI + 360) % 360;
  const displayAngle = Math.round(cameraAngleDeg);
  const frameTime = stats.frameTimeMs || 0;
  const simMs = Math.max(0, frameTime * 0.76);
  const renderMs = Math.max(0, frameTime * 0.19);
  const tempC = Math.round(simParams.emission * 120 + simParams.buoyancy * 60 + simParams.vorticity * 10);
  const fuelPct = Math.round(clamp(100 - simParams.emission * 2.6, 0, 100));
  const simTimeSeconds = stats.frame * DEFAULT_TIME_STEP;
  const smokeAmount = Math.round((1 - simParams.smokeDissipation) * 1000 + simParams.scattering * 3.8);
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

    return [
      (isAll || isConvection) && {
        id: 'flameHeight',
        label: 'Flame Height',
        low: 'Low',
        high: 'High',
        tone: 'warm' as const,
        value: clamp(inverseLerp(0.5, 8.0, simParams.buoyancy), 0, 1),
        valueText: undefined,
        onChange: (t: number) => updateParam('buoyancy', lerp(0.5, 8.0, t)),
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
        value: clamp(inverseLerp(0.995, 0.85, simParams.smokeDissipation), 0, 1),
        valueText: `${smokeAmount} kg/m³`,
        onChange: (t: number) => applyMacro('smokeDensity', t),
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
        value: turbulenceCharacter,
        valueText: `${turbulenceCharacter.toFixed(2)}`,
        onChange: (t: number) => applyMacro('turbulenceCharacter', t),
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
        onChange: (t: number) => updateParam('fuelEfficiency', lerp(0.2, 2.0, t)),
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
  }, [applyMacro, deckRailSection, simParams, smokeAmount, smokeDarkness, turbulenceCharacter, updateParam, windGusts, burnRate]);

  return (
    <div className="deck-root">
      <canvas ref={canvasRef} width={dimensions.width} height={dimensions.height} className="deck-canvas"
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
          title={`Default ${formatWithStep(defaultValue, spec.step)}`}
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
