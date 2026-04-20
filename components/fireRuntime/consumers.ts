import type {
  CompositionDebugMode,
  QualityMode,
} from '../fluid/debugConfig';

export type FireRuntimeConsumerId =
  | 'control-deck'
  | 'scene-embed'
  | 'site-hero'
  | 'cursor-fx'
  | 'ambient-background';

export interface FireRuntimeConsumerDefinition {
  id: FireRuntimeConsumerId;
  label: string;
  tagline: string;
  description: string;
  useCase: string;
  initialSceneId: number;
  initialQualityMode: QualityMode;
  initialCompositionMode: CompositionDebugMode;
  initialSmokeEnabled: boolean;
  initialResolutionScale: number;
  showTopbar: boolean;
  showControlsPanel: boolean;
  showDiagnosticsPanel: boolean;
  showStatusBar: boolean;
}

export const FIRE_RUNTIME_CONSUMERS: FireRuntimeConsumerDefinition[] = [
  {
    id: 'control-deck',
    label: 'Control Deck',
    tagline: 'Full product surface for tuning, diagnostics, and capture.',
    description: 'The existing operator-facing app with all deck chrome enabled. This is the heavy consumer that exposes presets, telemetry, capture, and debugging controls.',
    useCase: 'Primary authoring and debugging surface for the runtime.',
    initialSceneId: 0,
    initialQualityMode: 'accurate',
    initialCompositionMode: 'composited',
    initialSmokeEnabled: true,
    initialResolutionScale: 0.9,
    showTopbar: true,
    showControlsPanel: true,
    showDiagnosticsPanel: true,
    showStatusBar: true,
  },
  {
    id: 'scene-embed',
    label: '3D Scene Embed',
    tagline: 'Runtime mounted into a cinematic scene shell with minimal chrome.',
    description: 'A product-facing 3D consumer where the fire is part of a scene, not the whole app. The runtime stays accurate, but control chrome is hidden so it behaves like an embeddable scene widget.',
    useCase: 'Three.js scenes, configurators, product worlds, and interactive art.',
    initialSceneId: 4,
    initialQualityMode: 'accurate',
    initialCompositionMode: 'composited',
    initialSmokeEnabled: true,
    initialResolutionScale: 0.85,
    showTopbar: false,
    showControlsPanel: false,
    showDiagnosticsPanel: false,
    showStatusBar: false,
  },
  {
    id: 'site-hero',
    label: '2D Site Hero',
    tagline: 'Fire as a branded hero treatment behind marketing copy.',
    description: 'A leaner consumer meant for landing pages and editorial sites. It emphasizes the flame layer and strips the app chrome so the runtime behaves like a motion background component.',
    useCase: 'Hero sections, editorial headers, and campaign pages.',
    initialSceneId: 1,
    initialQualityMode: 'accurate',
    initialCompositionMode: 'fire_only',
    initialSmokeEnabled: false,
    initialResolutionScale: 0.78,
    showTopbar: false,
    showControlsPanel: false,
    showDiagnosticsPanel: false,
    showStatusBar: false,
  },
  {
    id: 'cursor-fx',
    label: 'Cursor Effect',
    tagline: 'A compact, responsive flame treatment for pointer-led interactions.',
    description: 'A lightweight presentation of the same physics tuned as an effect primitive rather than a scene. It keeps only the fire layer and lower resolution defaults so the runtime can back interactive cursor treatments.',
    useCase: 'Cursor trails, interactive reveals, and motion accents.',
    initialSceneId: 3,
    initialQualityMode: 'realtime',
    initialCompositionMode: 'fire_only',
    initialSmokeEnabled: false,
    initialResolutionScale: 0.68,
    showTopbar: false,
    showControlsPanel: false,
    showDiagnosticsPanel: false,
    showStatusBar: false,
  },
  {
    id: 'ambient-background',
    label: 'Ambient Background',
    tagline: 'Low-UI atmospheric fire for lobbies, installs, and backdrop loops.',
    description: 'An always-on background consumer focused on stability and lower steady-state cost. It demonstrates the runtime as an environmental motion system, not a direct interaction surface.',
    useCase: 'Ambient loops, lounge backdrops, installation screens, and idle states.',
    initialSceneId: 6,
    initialQualityMode: 'realtime',
    initialCompositionMode: 'fire_only',
    initialSmokeEnabled: true,
    initialResolutionScale: 0.72,
    showTopbar: false,
    showControlsPanel: false,
    showDiagnosticsPanel: false,
    showStatusBar: false,
  },
];

export const FIRE_RUNTIME_CONSUMER_MAP = Object.fromEntries(
  FIRE_RUNTIME_CONSUMERS.map((consumer) => [consumer.id, consumer])
) as Record<FireRuntimeConsumerId, FireRuntimeConsumerDefinition>;
