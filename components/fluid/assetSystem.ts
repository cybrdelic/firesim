import * as THREE from 'three/webgpu';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

export type WoodAssetState = 'scanned_asset_ready' | 'procedural_fallback' | 'no_asset';
type ScannedLogExtension = 'glb' | 'gltf' | 'obj' | 'fbx';

interface ScannedLogCandidate {
  url: string;
  extension: ScannedLogExtension;
}

interface ScannedLogLoadResult {
  root?: THREE.Object3D;
  sourceUrl?: string;
  reason?: string;
}

interface FallbackLogPileResult {
  textures: THREE.Texture[];
  sideMaterial: THREE.MeshPhysicalMaterial;
  capMaterial: THREE.MeshPhysicalMaterial;
}

export interface CampfireLogPileLoadResult {
  assetState: WoodAssetState;
  source?: string;
  reason?: string;
  textures?: THREE.Texture[];
  sideMaterial?: THREE.MeshPhysicalMaterial;
  capMaterial?: THREE.MeshPhysicalMaterial;
}

export interface AddCampfireLogPileOptions {
  scene: THREE.Scene;
  transforms: Array<{ position: THREE.Vector3; rotation: THREE.Euler }>;
  addFallbackLogPile: (scene: THREE.Scene) => FallbackLogPileResult;
  normalizeLogSource: (source: THREE.Object3D) => void;
  applyFallbackLogMaterial: (root: THREE.Object3D) => void;
  candidates?: readonly string[];
}

const DEFAULT_SCANNED_LOG_CANDIDATES = [
  '/models/scanned-log.glb',
  '/models/scanned-log.gltf',
  '/models/scanned-log.obj',
  '/models/scanned-log.fbx',
  '/models/log-scan/scanned-log.glb',
  '/models/log-scan/scanned-log.gltf',
  '/models/log-scan/scanned-log.obj',
  '/models/log-scan/scanned-log.fbx',
] as const;

let cachedScannedLogSelection: ScannedLogCandidate | null | undefined;

export const WOOD_ASSET_STATE_LABELS: Record<WoodAssetState, string> = {
  scanned_asset_ready: 'scanned_asset_ready',
  procedural_fallback: 'procedural_fallback',
  no_asset: 'no_asset',
};

const resolveScannedLogExtension = (url: string): ScannedLogExtension | null => {
  if (url.endsWith('.glb')) return 'glb';
  if (url.endsWith('.gltf')) return 'gltf';
  if (url.endsWith('.obj')) return 'obj';
  if (url.endsWith('.fbx')) return 'fbx';
  return null;
};

const looksLikeHtmlFallback = (sample: string, contentType: string) => {
  if (contentType.includes('text/html')) return true;
  const trimmed = sample.trimStart().toLowerCase();
  return trimmed.startsWith('<!doctype html') || trimmed.startsWith('<html');
};

const isValidModelSignature = (extension: ScannedLogExtension, bytes: Uint8Array, sample: string) => {
  if (extension === 'glb') {
    if (bytes.length < 4) return false;
    return String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]) === 'glTF';
  }
  if (extension === 'gltf') {
    const trimmed = sample.trimStart();
    return trimmed.startsWith('{') && trimmed.includes('"asset"');
  }
  if (extension === 'obj') {
    return /(^|\n)\s*(#|v |vn |vt |f |o |g |mtllib |usemtl )/m.test(sample);
  }
  return sample.includes('Kaydara FBX Binary') || /(^|\n)\s*;\s*FBX/m.test(sample);
};

const preflightModelCandidate = async (url: string): Promise<ScannedLogCandidate | null> => {
  const extension = resolveScannedLogExtension(url);
  if (!extension) return null;
  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: { Range: 'bytes=0-2047' },
      cache: 'no-store',
    });
    if (!response.ok) return null;
    const contentType = (response.headers.get('content-type') ?? '').toLowerCase();
    const bytes = new Uint8Array(await response.arrayBuffer());
    const sample = new TextDecoder().decode(bytes);
    if (looksLikeHtmlFallback(sample, contentType)) return null;
    if (!isValidModelSignature(extension, bytes, sample)) return null;
    return { url, extension };
  } catch {
    return null;
  }
};

const resolveScannedLogCandidate = async (candidates: readonly string[]): Promise<ScannedLogCandidate | null> => {
  if (cachedScannedLogSelection !== undefined) return cachedScannedLogSelection;
  for (const candidateUrl of candidates) {
    const candidate = await preflightModelCandidate(candidateUrl);
    if (candidate) {
      cachedScannedLogSelection = candidate;
      return candidate;
    }
  }
  cachedScannedLogSelection = null;
  return null;
};

const loadScannedLogModel = async (candidates: readonly string[]): Promise<ScannedLogLoadResult | null> => {
  const candidate = await resolveScannedLogCandidate(candidates);
  if (!candidate) return null;

  const gltfLoader = new GLTFLoader();
  const objLoader = new OBJLoader();
  const fbxLoader = new FBXLoader();

  try {
    if (candidate.extension === 'glb' || candidate.extension === 'gltf') {
      const gltf = await gltfLoader.loadAsync(candidate.url);
      return { root: gltf.scene, sourceUrl: candidate.url };
    }
    if (candidate.extension === 'obj') {
      const obj = await objLoader.loadAsync(candidate.url);
      return { root: obj, sourceUrl: candidate.url };
    }
    const fbx = await fbxLoader.loadAsync(candidate.url);
    return { root: fbx, sourceUrl: candidate.url };
  } catch {
    cachedScannedLogSelection = null;
    return { reason: `failed_to_load:${candidate.url}`, sourceUrl: candidate.url };
  }
};

export const addCampfireLogPile = async ({
  scene,
  transforms,
  addFallbackLogPile,
  normalizeLogSource,
  applyFallbackLogMaterial,
  candidates = DEFAULT_SCANNED_LOG_CANDIDATES,
}: AddCampfireLogPileOptions): Promise<CampfireLogPileLoadResult> => {
  const scanned = await loadScannedLogModel(candidates);
  if (!scanned?.root) {
    try {
      const fallback = addFallbackLogPile(scene);
      const reason = scanned?.reason ?? 'no_valid_scanned_asset_candidate';
      return {
        assetState: 'procedural_fallback',
        reason,
        textures: fallback.textures,
        sideMaterial: fallback.sideMaterial,
        capMaterial: fallback.capMaterial,
      };
    } catch (error) {
      return {
        assetState: 'no_asset',
        reason: `fallback_generation_failed:${error instanceof Error ? error.message : String(error)}`,
      };
    }
  }

  normalizeLogSource(scanned.root);
  applyFallbackLogMaterial(scanned.root);
  for (const transform of transforms) {
    const instance = scanned.root.clone(true);
    instance.position.copy(transform.position);
    instance.rotation.copy(transform.rotation);
    scene.add(instance);
  }
  return { assetState: 'scanned_asset_ready', source: scanned.sourceUrl };
};
