type GPUTextureFormat = string;
type GPUBufferUsageFlags = number;
type GPUShaderStageFlags = number;
type GPUDeviceLostReason = 'destroyed' | 'unknown' | string;
type GPULoadOp = 'load' | 'clear';
type GPUStoreOp = 'store' | 'discard';

interface GPUBufferUsageNamespace {
  COPY_DST: GPUBufferUsageFlags;
  STORAGE: GPUBufferUsageFlags;
  UNIFORM: GPUBufferUsageFlags;
}

interface GPUShaderStageNamespace {
  COMPUTE: GPUShaderStageFlags;
  FRAGMENT: GPUShaderStageFlags;
  VERTEX: GPUShaderStageFlags;
}

declare const GPUBufferUsage: GPUBufferUsageNamespace;
declare const GPUShaderStage: GPUShaderStageNamespace;

interface Navigator {
  gpu: GPU;
}

interface GPU {
  requestAdapter(): Promise<GPUAdapter | null>;
  getPreferredCanvasFormat(): GPUTextureFormat;
}

interface GPUSupportedLimits {
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
  maxComputeWorkgroupsPerDimension: number;
}

interface GPUAdapter {
  limits: GPUSupportedLimits;
  requestDevice(): Promise<GPUDevice>;
}

interface GPUDeviceLostInfo {
  reason: GPUDeviceLostReason;
  message: string;
}

interface GPUError {
  message: string;
}

interface GPUUncapturedErrorEvent extends Event {
  error: GPUError;
}

interface GPUQueue {
  submit(commandBuffers: GPUCommandBuffer[]): void;
  writeBuffer(
    buffer: GPUBuffer,
    bufferOffset: number,
    data: BufferSource | ArrayBuffer,
    dataOffset?: number,
    size?: number
  ): void;
}

interface GPUDevice extends EventTarget {
  queue: GPUQueue;
  lost: Promise<GPUDeviceLostInfo>;

  createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout;
  createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor): GPUPipelineLayout;
  createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup;
  createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
  createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;
  createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline;
  createRenderPipeline(descriptor: GPURenderPipelineDescriptor): GPURenderPipeline;
  createCommandEncoder(): GPUCommandEncoder;
  destroy(): void;

  addEventListener(
    type: 'uncapturederror',
    listener: (event: GPUUncapturedErrorEvent) => void,
    options?: boolean | AddEventListenerOptions
  ): void;
  removeEventListener(
    type: 'uncapturederror',
    listener: (event: GPUUncapturedErrorEvent) => void,
    options?: boolean | EventListenerOptions
  ): void;
}

interface GPUObjectDescriptorBase {
  label?: string;
}

interface GPUBufferDescriptor extends GPUObjectDescriptorBase {
  size: number;
  usage: GPUBufferUsageFlags;
}

interface GPUBuffer {}

type GPUBufferBindingType = 'uniform' | 'storage' | 'read-only-storage';

interface GPUBufferBindingLayout {
  type?: GPUBufferBindingType;
}

interface GPUBindGroupLayoutEntry {
  binding: number;
  visibility: GPUShaderStageFlags;
  buffer?: GPUBufferBindingLayout;
}

interface GPUBindGroupLayoutDescriptor extends GPUObjectDescriptorBase {
  entries: GPUBindGroupLayoutEntry[];
}

interface GPUBindGroupLayout {}

interface GPUPipelineLayoutDescriptor extends GPUObjectDescriptorBase {
  bindGroupLayouts: GPUBindGroupLayout[];
}

interface GPUPipelineLayout {}

interface GPUBufferBinding {
  buffer: GPUBuffer;
  offset?: number;
  size?: number;
}

interface GPUTextureView {}

interface GPUSampler {}

type GPUBindingResource = GPUBufferBinding | GPUTextureView | GPUSampler;

interface GPUBindGroupEntry {
  binding: number;
  resource: GPUBindingResource;
}

interface GPUBindGroupDescriptor extends GPUObjectDescriptorBase {
  layout: GPUBindGroupLayout;
  entries: GPUBindGroupEntry[];
}

interface GPUBindGroup {}

interface GPUShaderModuleDescriptor extends GPUObjectDescriptorBase {
  code: string;
}

type GPUCompilationMessageType = 'error' | 'warning' | 'info';

interface GPUCompilationMessage {
  type: GPUCompilationMessageType;
  message: string;
}

interface GPUCompilationInfo {
  messages: GPUCompilationMessage[];
}

interface GPUShaderModule {
  getCompilationInfo(): Promise<GPUCompilationInfo>;
}

interface GPUProgrammableStage {
  module: GPUShaderModule;
  entryPoint: string;
}

interface GPUComputePipelineDescriptor extends GPUObjectDescriptorBase {
  layout: GPUPipelineLayout;
  compute: GPUProgrammableStage;
}

interface GPURenderPipelineDescriptor extends GPUObjectDescriptorBase {
  layout: GPUPipelineLayout;
  vertex: GPUProgrammableStage;
  fragment: GPUFragmentState;
  primitive?: GPUPrimitiveState;
}

interface GPUFragmentState extends GPUProgrammableStage {
  targets: GPUColorTargetState[];
}

interface GPUColorTargetState {
  format: GPUTextureFormat;
}

interface GPUPrimitiveState {
  topology: 'triangle-list' | string;
}

interface GPUComputePipeline {}

interface GPURenderPipeline {}

interface GPUCommandEncoder {
  beginComputePass(): GPUComputePassEncoder;
  beginRenderPass(descriptor: GPURenderPassDescriptor): GPURenderPassEncoder;
  finish(): GPUCommandBuffer;
}

interface GPUComputePassEncoder {
  setPipeline(pipeline: GPUComputePipeline): void;
  setBindGroup(index: number, bindGroup: GPUBindGroup): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}

interface GPUColor {
  r: number;
  g: number;
  b: number;
  a: number;
}

interface GPURenderPassColorAttachment {
  view: GPUTextureView;
  clearValue?: GPUColor;
  loadOp: GPULoadOp;
  storeOp: GPUStoreOp;
}

interface GPURenderPassDescriptor {
  colorAttachments: GPURenderPassColorAttachment[];
}

interface GPURenderPassEncoder {
  setPipeline(pipeline: GPURenderPipeline): void;
  setBindGroup(index: number, bindGroup: GPUBindGroup): void;
  draw(vertexCount: number): void;
  end(): void;
}

interface GPUCommandBuffer {}

interface GPUTexture {
  createView(): GPUTextureView;
}

interface GPUCanvasConfiguration {
  device: GPUDevice;
  format: GPUTextureFormat;
  alphaMode?: 'premultiplied' | 'opaque';
}

interface GPUCanvasContext {
  configure(configuration: GPUCanvasConfiguration): void;
  unconfigure(): void;
  getCurrentTexture(): GPUTexture;
}

interface HTMLCanvasElement {
  getContext(contextId: 'webgpu'): GPUCanvasContext | null;
}
