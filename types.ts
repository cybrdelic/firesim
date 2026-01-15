export type WgslScalarType = 'f32' | 'i32' | 'u32';
export type WgslVecType = 'vec2f' | 'vec3f' | 'vec4f';
export type WgslMatType = 'mat4x4f';
export type WgslType = WgslScalarType | WgslVecType | WgslMatType;

export interface StructField {
  name: string;
  type: WgslType;
}

export interface ResourceEntry {
  name: string;
  kind: 'uniform' | 'storage' | 'texture' | 'sampler';
  group?: number;
  // For 'uniform', we define fields to calculate padding
  structFields?: StructField[]; 
  // For 'storage', we usually just reference a type name or array
  typeName?: string; 
  access?: 'read' | 'read_write'; 
}

export interface Schema {
  passName: string;
  entries: ResourceEntry[];
}

export interface GeneratedOutput {
  wgsl: string;
  typescript: string;
  stats: {
    uniformsPadded: number;
    bindingsGenerated: number;
  };
}