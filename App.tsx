import React from 'react';
import FluidSimulation from './components/FluidSimulation';

const App: React.FC = () => {
  return (
    <div className="h-screen w-screen bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-slate-900 via-[#050505] to-black overflow-hidden">
      <FluidSimulation />
    </div>
  );
};

export default App;