import React, { useMemo, useState } from 'react';
import FluidSimulation from './components/FluidSimulation';
import {
  FIRE_RUNTIME_CONSUMER_MAP,
  FIRE_RUNTIME_CONSUMERS,
  type FireRuntimeConsumerId,
} from './components/fireRuntime/consumers';
import TooltipLayer from './components/TooltipLayer';

const RuntimeHostOverlay: React.FC<{ consumerId: FireRuntimeConsumerId }> = ({ consumerId }) => {
  if (consumerId === 'control-deck') return null;

  if (consumerId === 'scene-embed') {
    return (
      <div className="runtime-host runtime-host--scene-embed">
        <div className="runtime-host-toolbar">
          <span>Scene Graph</span>
          <span>Lighting</span>
          <span>Materials</span>
          <span>FX</span>
        </div>
        <div className="runtime-host-caption">
          Embedded fire actor inside a 3D scene shell.
        </div>
      </div>
    );
  }

  if (consumerId === 'site-hero') {
    return (
      <div className="runtime-host runtime-host--site-hero">
        <div className="runtime-host-nav">
          <span>Studio</span>
          <span>Work</span>
          <span>Services</span>
          <span>Contact</span>
        </div>
        <div className="runtime-host-copy">
          <p className="runtime-kicker">Reusable Fire Runtime</p>
          <h2>Physics-backed fire for interfaces, not just one app.</h2>
          <p>
            The same runtime can drive a branded hero, a scene embed, or an ambient backdrop
            without dragging the full control deck into production surfaces.
          </p>
        </div>
      </div>
    );
  }

  if (consumerId === 'cursor-fx') {
    return (
      <div className="runtime-host runtime-host--cursor-fx">
        <div className="runtime-cursor-card">Hover target</div>
        <div className="runtime-cursor-card">Reveal panel</div>
        <div className="runtime-cursor-card">Pointer accent</div>
      </div>
    );
  }

  return (
    <div className="runtime-host runtime-host--ambient-background">
      <div className="runtime-ambient-card">
        Ambient motion surface
      </div>
      <div className="runtime-ambient-card is-wide">
        Runtime-backed background loop for lobbies, installs, and idle states.
      </div>
    </div>
  );
};

const App: React.FC = () => {
  const [activeConsumerId, setActiveConsumerId] = useState<FireRuntimeConsumerId>('control-deck');
  const activeConsumer = useMemo(
    () => FIRE_RUNTIME_CONSUMER_MAP[activeConsumerId],
    [activeConsumerId]
  );

  return (
    <div className="app-shell runtime-gallery-app">
      <aside className="runtime-gallery-sidebar">
        <div className="runtime-gallery-head">
          <p className="runtime-gallery-eyebrow">Issue #6</p>
          <h1>Fire Runtime Consumers</h1>
          <p>
            First extraction pass: the app remains the flagship consumer, but the runtime is now
            exercised through multiple host shells that map to the target reuse cases.
          </p>
        </div>

        <div className="runtime-gallery-list" role="tablist" aria-label="Runtime consumers">
          {FIRE_RUNTIME_CONSUMERS.map((consumer) => (
            <button
              key={consumer.id}
              type="button"
              role="tab"
              aria-selected={activeConsumerId === consumer.id}
              className={`runtime-gallery-tab ${activeConsumerId === consumer.id ? 'is-active' : ''}`}
              onClick={() => setActiveConsumerId(consumer.id)}
            >
              <span className="runtime-gallery-tab-label">{consumer.label}</span>
              <span className="runtime-gallery-tab-tagline">{consumer.tagline}</span>
            </button>
          ))}
        </div>

        <dl className="runtime-gallery-details">
          <div>
            <dt>Use Case</dt>
            <dd>{activeConsumer.useCase}</dd>
          </div>
          <div>
            <dt>Consumer Goal</dt>
            <dd>{activeConsumer.description}</dd>
          </div>
          <div>
            <dt>Runtime Mode</dt>
            <dd>
              {activeConsumer.initialQualityMode} / {activeConsumer.initialCompositionMode}
            </dd>
          </div>
        </dl>
      </aside>

      <main className="runtime-gallery-stage">
        <div className="runtime-gallery-stage-head">
          <div>
            <p className="runtime-gallery-eyebrow">Active Consumer</p>
            <h2>{activeConsumer.label}</h2>
          </div>
          <p>{activeConsumer.tagline}</p>
        </div>

        <div className={`runtime-gallery-frame runtime-gallery-frame--${activeConsumer.id}`}>
          <FluidSimulation
            key={activeConsumer.id}
            consumerId={activeConsumer.id}
            className="runtime-gallery-sim"
          />
          <RuntimeHostOverlay consumerId={activeConsumer.id} />
        </div>
      </main>

      <TooltipLayer />
    </div>
  );
};

export default App;
