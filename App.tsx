import React from 'react';
import FluidSimulation from './components/FluidSimulation';
import TooltipLayer from './components/TooltipLayer';

const App: React.FC = () => {
  return (
    <div className="app-shell">
      <FluidSimulation />
      <TooltipLayer />
    </div>
  );
};

export default App;
