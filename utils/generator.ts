import { Schema, ResourceEntry, StructField, GeneratedOutput } from '../types';

// WebGPU std140 alignment rules
const TYPE_INFO: Record<string, { size: number, align: number }> = {
  'f32': { size: 4, align: 4 },
  'i32': { size: 4, align: 4 },
  'u32': { size: 4, align: 4 },
  'vec2f': { size: 8, align: 8 },
  'vec3f': { size: 12, align: 16 }, // vec3 is 12 bytes but aligns as 16 in uniforms
  'vec4f': { size: 16, align: 16 },
  'mat4x4f': { size: 64, align: 16 },
};

function getPaddingNeeded(offset: number, align: number): number {
  return (align - (offset % align)) % align;
}

// Generate the WGSL Struct with explicit padding
function generateStructWGSL(name: string, fields: StructField[]): { code: string, size: number } {
  let code = `struct ${name} {\n`;
  let currentOffset = 0;
  let padCount = 0;
  
  // Helper to add explicit padding fields
  const addPadding = (bytes: number) => {
    // We use f32 (4 bytes) for padding.
    // Uniform buffers don't support array<f32, N> unless stride is 16, 
    // so we must generate individual fields for standard 4-byte padding.
    const floats = Math.ceil(bytes / 4);
    for (let i = 0; i < floats; i++) {
      code += `  _pad_${padCount++}: f32, // explicit pad\n`;
      currentOffset += 4;
    }
  };

  fields.forEach(field => {
    const info = TYPE_INFO[field.type] || { size: 0, align: 1 }; // fallback
    
    // Calculate padding for this field's alignment
    const padding = getPaddingNeeded(currentOffset, info.align);
    if (padding > 0) {
      addPadding(padding);
    }

    code += `  ${field.name}: ${field.type},\n`;
    currentOffset += info.size;
  });

  // Tail padding to 16 bytes (uniform buffer requirement)
  const tailPadding = getPaddingNeeded(currentOffset, 16);
  if (tailPadding > 0) {
     addPadding(tailPadding);
  }

  code += `};\n`;
  return { code, size: currentOffset };
}

export function generateContract(schema: Schema): GeneratedOutput {
  let wgsl = `// --------------------------------------------------------\n`;
  wgsl += `// AUTO-GENERATED CONTRACT: ${schema.passName}\n`;
  wgsl += `// Do not edit manually. Changes will be overwritten.\n`;
  wgsl += `// --------------------------------------------------------\n\n`;

  let ts = `import { GPUDevice, GPUBindGroupLayout, GPUPipelineLayout } from 'webgpu';\n\n`;
  ts += `// --------------------------------------------------------\n`;
  ts += `// AUTO-GENERATED TYPESCRIPT CONTRACT: ${schema.passName}\n`;
  ts += `// --------------------------------------------------------\n\n`;

  const groups: Record<number, ResourceEntry[]> = {};
  
  // 1. Organize by group
  schema.entries.forEach(entry => {
    const g = entry.group ?? 0;
    if (!groups[g]) groups[g] = [];
    groups[g].push(entry);
  });

  // 2. Generate WGSL Structs & Bindings
  let wgslBindings = '';
  let structDefs = '';
  let tsLayoutEntries = '';
  let tsInterface = `export interface ${schema.passName}Resources {\n`;

  // Sort groups to be deterministic
  const sortedGroups = Object.keys(groups).map(Number).sort((a, b) => a - b);

  let bindingsCount = 0;

  sortedGroups.forEach(groupIndex => {
    const entries = groups[groupIndex];
    tsLayoutEntries += `// Group ${groupIndex} Layout Entries\nconst group${groupIndex}Entries: GPUBindGroupLayoutEntry[] = [\n`;

    entries.forEach((entry, idx) => {
      const bindingIndex = idx; // Auto-assigned sequential index
      const constName = `B_${entry.name.toUpperCase()}`;
      
      wgsl += `const ${constName} = ${bindingIndex};\n`;
      
      // Handle Uniform Structs
      let wgslType = entry.typeName || 'unknown';
      if (entry.kind === 'uniform' && entry.structFields) {
        const structName = `${schema.passName}_${entry.name}_T`;
        const { code, size } = generateStructWGSL(structName, entry.structFields);
        structDefs += code + '\n';
        wgslType = structName;
        
        // Validation check in TS (comment)
        tsLayoutEntries += `  // Uniform: ${entry.name}, Size: ${size} bytes (padded to 16)\n`;
      }

      // WGSL Binding
      let varDecl = `@group(${groupIndex}) @binding(${constName}) var`;
      if (entry.kind === 'uniform') {
        varDecl += `<uniform> ${entry.name}: ${wgslType};`;
      } else if (entry.kind === 'storage') {
        varDecl += `<storage, ${entry.access || 'read'}> ${entry.name}: ${wgslType};`;
      } else if (entry.kind === 'texture') {
        varDecl += ` ${entry.name}: texture_2d<f32>;`;
      } else if (entry.kind === 'sampler') {
        varDecl += ` ${entry.name}: sampler;`;
      }
      wgslBindings += varDecl + '\n';

      // TS Layout Entry
      tsLayoutEntries += `  {\n    binding: ${bindingIndex},\n    visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,\n`;
      if (entry.kind === 'uniform') {
        tsLayoutEntries += `    buffer: { type: 'uniform' }\n`;
        tsInterface += `  ${entry.name}: GPUBuffer;\n`;
      } else if (entry.kind === 'storage') {
        tsLayoutEntries += `    buffer: { type: '${entry.access === 'read_write' ? 'storage' : 'read-only-storage'}' }\n`;
        tsInterface += `  ${entry.name}: GPUBuffer;\n`;
      } else if (entry.kind === 'texture') {
        tsLayoutEntries += `    texture: {}\n`;
        tsInterface += `  ${entry.name}: GPUTextureView;\n`;
      } else if (entry.kind === 'sampler') {
        tsLayoutEntries += `    sampler: {}\n`;
        tsInterface += `  ${entry.name}: GPUSampler;\n`;
      }
      tsLayoutEntries += `  },\n`;
      
      bindingsCount++;
    });

    tsLayoutEntries += `];\n\n`;
  });

  tsInterface += `}\n\n`;

  // Assemble WGSL
  wgsl += `\n${structDefs}\n${wgslBindings}`;

  // Assemble TS Class
  ts += tsInterface;
  ts += tsLayoutEntries;

  ts += `export class ${schema.passName}Contract {
  private layout: GPUPipelineLayout;
  private bindGroupLayouts: GPUBindGroupLayout[];

  constructor(device: GPUDevice) {
    this.bindGroupLayouts = [
      ${sortedGroups.map(g => `device.createBindGroupLayout({ entries: group${g}Entries, label: '${schema.passName}_G${g}' })`).join(',\n      ')}
    ];
    
    this.layout = device.createPipelineLayout({
      bindGroupLayouts: this.bindGroupLayouts,
      label: '${schema.passName}_Layout'
    });
  }

  getPipelineLayout(): GPUPipelineLayout {
    return this.layout;
  }

  createBindGroup(device: GPUDevice, resources: ${schema.passName}Resources): GPUBindGroup {
    // In a real multi-group scenario, this would accept a group index or return multiple groups
    // For simplicity, assuming Group 0 matches the input resources interface
    return device.createBindGroup({
      layout: this.bindGroupLayouts[0],
      entries: [
${groups[0]?.map((e, idx) => `        { binding: ${idx}, resource: ${e.kind === 'sampler' || e.kind === 'texture' ? `resources.${e.name}` : `{ buffer: resources.${e.name} }`} },`).join('\n') || ''}
      ]
    });
  }
}
`;

  return {
    wgsl,
    typescript: ts,
    stats: {
      uniformsPadded: 1,
      bindingsGenerated: bindingsCount
    }
  };
}