type GPUTextureFormat = string;
type GPUBufferUsageFlags = number;
type GPUShaderStageFlags = number;
type GPUDeviceLostReason = 'destroyed' | 'unknown' | string;
type GPULoadOp = 'load' | 'clear';
type GPUStoreOp = 'store' | 'discard';
type GPUFeatureName = 'timestamp-query' | string;
type GPUTextureUsageFlags = number;
type GPUMapModeFlags = number;

interface GPUBufferUsageNamespace {
  COPY_DST: GPUBufferUsageFlags;
  COPY_SRC: GPUBufferUsageFlags;
  STORAGE: GPUBufferUsageFlags;
  UNIFORM: GPUBufferUsageFlags;
  MAP_READ: GPUBufferUsageFlags;
  QUERY_RESOLVE: GPUBufferUsageFlags;
}

interface GPUTextureUsageNamespace {
  COPY_SRC: GPUTextureUsageFlags;
  COPY_DST: GPUTextureUsageFlags;
  TEXTURE_BINDING: GPUTextureUsageFlags;
  STORAGE_BINDING: GPUTextureUsageFlags;
  RENDER_ATTACHMENT: GPUTextureUsageFlags;
}

interface GPUMapModeNamespace {
  READ: GPUMapModeFlags;
  WRITE: GPUMapModeFlags;
}

interface GPUShaderStageNamespace {
  COMPUTE: GPUShaderStageFlags;
  FRAGMENT: GPUShaderStageFlags;
  VERTEX: GPUShaderStageFlags;
}

declare const GPUBufferUsage: GPUBufferUsageNamespace;
declare const GPUShaderStage: GPUShaderStageNamespace;
declare const GPUTextureUsage: GPUTextureUsageNamespace;
declare const GPUMapMode: GPUMapModeNamespace;

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
  features: GPUSupportedFeatures;
  requestDevice(descriptor?: GPUDeviceDescriptor): Promise<GPUDevice>;
}

interface GPUSupportedFeatures {
  has(feature: GPUFeatureName): boolean;
}

interface GPUDeviceDescriptor extends GPUObjectDescriptorBase {
  requiredFeatures?: GPUFeatureName[];
  requiredLimits?: Record<string, number>;
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
  features: GPUSupportedFeatures;

  createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout;
  createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor): GPUPipelineLayout;
  createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup;
  createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
  createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;
  createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline;
  createRenderPipeline(descriptor: GPURenderPipelineDescriptor): GPURenderPipeline;
  createCommandEncoder(): GPUCommandEncoder;
  createQuerySet(descriptor: GPUQuerySetDescriptor): GPUQuerySet;
  createTexture(descriptor: GPUTextureDescriptor): GPUTexture;
  createSampler(descriptor?: GPUSamplerDescriptor): GPUSampler;
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

interface GPUBuffer {
  mapAsync(mode: GPUMapModeFlags, offset?: number, size?: number): Promise<void>;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  unmap(): void;
}

type GPUBufferBindingType = 'uniform' | 'storage' | 'read-only-storage';

interface GPUBufferBindingLayout {
  type?: GPUBufferBindingType;
}

interface GPUBindGroupLayoutEntry {
  binding: number;
  visibility: GPUShaderStageFlags;
  buffer?: GPUBufferBindingLayout;
  texture?: GPUTextureBindingLayout;
  sampler?: GPUSamplerBindingLayout;
}

interface GPUTextureBindingLayout {
  sampleType?: 'float' | 'unfilterable-float' | 'depth' | 'sint' | 'uint';
  viewDimension?: string;
  multisampled?: boolean;
}

interface GPUSamplerBindingLayout {
  type?: 'filtering' | 'non-filtering' | 'comparison';
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
  blend?: GPUBlendState;
}

interface GPUBlendState {
  color: GPUBlendComponent;
  alpha: GPUBlendComponent;
}

interface GPUBlendComponent {
  srcFactor?: string;
  dstFactor?: string;
  operation?: string;
}

interface GPUPrimitiveState {
  topology: 'triangle-list' | string;
}

interface GPUComputePipeline {}

interface GPURenderPipeline {}

interface GPUCommandEncoder {
  beginComputePass(descriptor?: GPUComputePassDescriptor): GPUComputePassEncoder;
  beginRenderPass(descriptor: GPURenderPassDescriptor): GPURenderPassEncoder;
  copyBufferToBuffer(src: GPUBuffer, srcOffset: number, dst: GPUBuffer, dstOffset: number, size: number): void;
  resolveQuerySet(querySet: GPUQuerySet, firstQuery: number, queryCount: number, destination: GPUBuffer, destinationOffset: number): void;
  finish(): GPUCommandBuffer;
}

interface GPUComputePassDescriptor {
  timestampWrites?: GPUComputePassTimestampWrites;
}

interface GPUComputePassTimestampWrites {
  querySet: GPUQuerySet;
  beginningOfPassWriteIndex?: number;
  endOfPassWriteIndex?: number;
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
  timestampWrites?: GPURenderPassTimestampWrites;
}

interface GPURenderPassTimestampWrites {
  querySet: GPUQuerySet;
  beginningOfPassWriteIndex?: number;
  endOfPassWriteIndex?: number;
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
  destroy(): void;
  width: number;
  height: number;
}

interface GPUTextureDescriptor extends GPUObjectDescriptorBase {
  size: number[] | [number, number, number];
  format: GPUTextureFormat;
  usage: GPUTextureUsageFlags;
}

interface GPUQuerySet {}

interface GPUQuerySetDescriptor extends GPUObjectDescriptorBase {
  type: 'timestamp' | 'occlusion';
  count: number;
}

interface GPUSamplerDescriptor extends GPUObjectDescriptorBase {
  magFilter?: 'nearest' | 'linear';
  minFilter?: 'nearest' | 'linear';
  mipmapFilter?: 'nearest' | 'linear';
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
