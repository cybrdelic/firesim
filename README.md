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

## 🏗️ Project Structure

```
firesim/
├── components/
│   ├── FluidSimulation.tsx  # WebGPU fluid/fire simulation engine
│   ├── CodeBlock.tsx        # UI component for code display
│   ├── ProgressHeader.tsx   # UI header component
│   └── SafetyPatternCard.tsx # Info card component
├── utils/
│   └── generator.ts         # Utility functions
├── App.tsx                  # Main application component
├── types.ts                 # TypeScript type definitions
└── constants.ts             # Application constants
```

## 🎯 How It Works

1. **3D Voxel Grid** - The simulation runs on a 3D grid of voxels
2. **Compute Shaders** - WGSL shaders handle advection, diffusion, and pressure projection
3. **Double Buffering** - Ping-pong buffers for density and velocity fields
4. **Volume Rendering** - Ray marching through the density field to render fire

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
