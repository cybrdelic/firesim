# 🔥 FireSim

A **real-time 3D fire and fluid simulation** powered by WebGPU compute shaders. Watch volumetric fire physics unfold in your browser with GPU-accelerated Navier-Stokes fluid dynamics.

![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=flat&logo=typescript&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)
![WebGPU](https://img.shields.io/badge/WebGPU-FF6B6B?style=flat&logo=webgl&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=flat&logo=vite&logoColor=white)

## ✨ Features

- **Real-Time 3D Simulation** - Volumetric fire rendered at interactive framerates
- **GPU Compute Shaders** - Physics calculations run entirely on the GPU via WebGPU
- **Fluid Dynamics** - Navier-Stokes based advection, diffusion, and pressure solving
- **Interactive Controls** - Adjust simulation parameters in real-time

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- A browser with WebGPU support (Chrome 113+, Edge 113+, or Firefox Nightly)

### Installation

```bash
# Clone the repository
git clone https://github.com/alexf/firesim.git
cd firesim

# Install dependencies
npm install

# Start development server
npm run dev
```

### Build for Production

```bash
npm run build
npm run preview
```

### Stability Harness

Run deterministic stability checks (fuzz + visual regression):

```bash
npm run test:stability
```

Generate/update snapshot baselines:

```bash
npm run test:stability:update
```

The app supports deterministic sweep mode via URL params:

- `deterministic=1` - fixed simulation time step
- `sweep=1` - enables readiness/status reporting for automation
- `scene=<id>` - initial scene preset
- `grid=<64|128|192|256>` - initial grid size
- `smoke=<0|1>` - smoke toggle
- `maxFrames=<n>` - freeze after `n` deterministic frames

Automation status/control hooks are available on `window`:

- `window.__FIRE_SIM_STATUS__`
- `window.__FIRE_SIM_CONTROL__`

Detailed limits and edge-case notes are documented in [STABILITY.md](STABILITY.md).

## 🏗️ Project Structure

```
firesim/
├── components/
│   ├── FluidSimulation.tsx   # Main runtime host that can mount as different consumers
│   ├── fireRuntime/
│   │   ├── consumers.ts      # Consumer definitions for control deck, 3D scene, hero, cursor, ambient
│   │   └── transport.ts      # Reusable GPU transport/buffer contract extracted from the app shell
│   ├── fluid/
│   │   ├── assetSystem.ts    # Scanned/procedural log asset loading
│   │   ├── debugConfig.ts    # Quality, composition, and debug mode mappings
│   │   ├── fireVolumeSystem.ts # WebGPU render loop and adaptive quality logic
│   │   └── woodSystem.ts     # Analytic wood SDF and burn-state helpers
│   └── TooltipLayer.tsx      # Global tooltip surface for deck controls
├── tests/stability/         # Playwright fuzz + snapshot stability harness
├── webgpu.d.ts              # Local WebGPU DOM typing surface for this project
├── App.tsx                  # Consumer gallery that remounts the runtime in multiple host shells
├── types.ts                 # TypeScript type definitions
└── index.css                # Deck styling
```

## 🔌 Runtime Consumers

The app now includes a consumer gallery that exercises the same fire runtime in five host shapes:

- `Control Deck` - the full authoring/debug product
- `3D Scene Embed` - a minimal scene-mounted runtime surface
- `2D Site Hero` - a branded hero/background presentation
- `Cursor Effect` - a lighter fire-only interactive effect shell
- `Ambient Background` - a low-UI background/idle consumer

This is the first extraction pass toward making FireSim behave more like a reusable runtime than a single hardcoded app shell.

## 🎯 How It Works

1. **3D Voxel Grid** - The simulation runs on a 3D grid of voxels
2. **Compute Shaders** - WGSL shaders handle advection, combustion, divergence, and pressure projection
3. **World + Volume Composition** - A Three.js world canvas sits behind a transparent fire volume canvas
4. **Quality Modes** - `realtime` uses half-resolution temporal upsampling, while `accurate` renders the fire pass at full resolution

## 🛠️ Tech Stack

- **React 19** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Lucide React** - Icons
- **WebGPU** - GPU compute and rendering

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
