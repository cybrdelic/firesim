export {};

declare global {
  interface Window {
    __FIRE_SIM_STATUS__?: {
      ready: boolean;
      frame: number;
      grid: number;
      scene: number;
      smoke: 0 | 1;
      error: string | null;
      warning: string | null;
    };

    __FIRE_SIM_CONTROL__?: {
      setScene: (id: number) => void;
      setGrid: (size: number) => void;
      setSmoke: (enabled: boolean) => void;
      setPlaying: (playing: boolean) => void;
      setParams: (patch: Partial<Record<string, number>>) => void;
      step: (frames?: number) => Promise<void>;
      runFuzz: (options: { iterations: number; seed: number }) => Promise<{
        runtimeError: string | null;
        finalParams: Record<string, unknown>;
      }>;
    };
  }
}
