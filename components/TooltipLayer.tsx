import React, { useEffect, useRef, useState } from 'react';

type TooltipState = {
  text: string;
  x: number;
  y: number;
  visible: boolean;
};

function getTooltipText(element: HTMLElement): string | null {
  const dataTooltip = element.getAttribute('data-tooltip')?.trim();
  if (dataTooltip) return dataTooltip;

  const ariaLabel = element.getAttribute('aria-label')?.trim();
  if (ariaLabel) return ariaLabel;

  const title = element.getAttribute('title')?.trim();
  if (title) return title;

  const textContent = element.textContent?.replace(/\s+/g, ' ').trim();
  if (textContent) return textContent;

  return null;
}

function isTooltipEligible(element: HTMLElement): boolean {
  const tagName = element.tagName.toLowerCase();
  if (tagName === 'button' || tagName === 'a' || tagName === 'input' || tagName === 'select' || tagName === 'textarea') {
    return true;
  }

  const role = element.getAttribute('role');
  if (role && ['button', 'tab', 'switch', 'checkbox', 'radio', 'slider', 'menuitem'].includes(role)) {
    return true;
  }

  const classList = element.classList;
  if (
    classList.contains('deck-tool') ||
    classList.contains('deck-rail-btn') ||
    classList.contains('deck-gear') ||
    classList.contains('deck-overlay-tile') ||
    classList.contains('deck-pill--button') ||
    classList.contains('sim-button') ||
    classList.contains('sim-param-button')
  ) {
    return true;
  }

  return false;
}

function findTooltipTarget(start: EventTarget | null): HTMLElement | null {
  let node: HTMLElement | null = start instanceof HTMLElement ? start : null;
  while (node) {
    if (isTooltipEligible(node)) {
      const text = getTooltipText(node);
      if (text) return node;
    }
    node = node.parentElement;
  }
  return null;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

const TooltipLayer: React.FC = () => {
  const [state, setState] = useState<TooltipState>({ text: '', x: 0, y: 0, visible: false });

  const lastTextRef = useRef<string>('');
  const lastVisibleRef = useRef<boolean>(false);
  const rafRef = useRef<number | null>(null);
  const pendingPointerRef = useRef<{ x: number; y: number; target: EventTarget | null } | null>(null);

  const hide = () => {
    if (!lastVisibleRef.current) return;
    lastVisibleRef.current = false;
    lastTextRef.current = '';
    setState((prev) => ({ ...prev, visible: false, text: '' }));
  };

  const showAtPointer = (x: number, y: number, target: EventTarget | null) => {
    const el = findTooltipTarget(target);
    if (!el) {
      hide();
      return;
    }

    const text = getTooltipText(el);
    if (!text) {
      hide();
      return;
    }

    const w = window.innerWidth;
    const h = window.innerHeight;

    const tooltipMaxWidth = 260;
    const tooltipApproxHeight = 80;
    const margin = 8;

    let left = x + 14;
    let top = y + 18;

    if (left + tooltipMaxWidth > w - margin) left = w - margin - tooltipMaxWidth;
    if (top + tooltipApproxHeight > h - margin) top = y - 18 - 28;

    left = clamp(left, margin, w - margin - tooltipMaxWidth);
    top = clamp(top, margin, h - margin - tooltipApproxHeight);

    lastVisibleRef.current = true;
    lastTextRef.current = text;

    setState({ text, x: left, y: top, visible: true });
  };

  const showNearElement = (element: HTMLElement) => {
    const text = getTooltipText(element);
    if (!text) {
      hide();
      return;
    }

    const rect = element.getBoundingClientRect();
    const w = window.innerWidth;
    const h = window.innerHeight;

    const tooltipMaxWidth = 260;
    const tooltipApproxHeight = 80;
    const margin = 8;

    const anchorX = rect.left + rect.width * 0.5;
    const anchorY = rect.top;

    let left = anchorX - tooltipMaxWidth * 0.5;
    let top = anchorY - 10 - 28;

    left = clamp(left, margin, w - margin - tooltipMaxWidth);
    top = clamp(top, margin, h - margin - tooltipApproxHeight);

    lastVisibleRef.current = true;
    lastTextRef.current = text;

    setState({ text, x: left, y: top, visible: true });
  };

  useEffect(() => {
    const onPointerMove = (event: PointerEvent) => {
      if (event.pointerType === 'touch') return;
      if (event.buttons !== 0) {
        hide();
        return;
      }

      pendingPointerRef.current = { x: event.clientX, y: event.clientY, target: event.target };
      if (rafRef.current != null) return;

      rafRef.current = window.requestAnimationFrame(() => {
        rafRef.current = null;
        const pending = pendingPointerRef.current;
        if (!pending) return;
        showAtPointer(pending.x, pending.y, pending.target);
      });
    };

    const onPointerLeaveWindow = () => {
      hide();
    };

    const onFocusIn = (event: FocusEvent) => {
      const el = findTooltipTarget(event.target);
      if (!el) {
        hide();
        return;
      }
      showNearElement(el);
    };

    const onFocusOut = () => {
      hide();
    };

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') hide();
    };

    window.addEventListener('pointermove', onPointerMove, { capture: true, passive: true });
    window.addEventListener('blur', onPointerLeaveWindow);
    window.addEventListener('focusin', onFocusIn, true);
    window.addEventListener('focusout', onFocusOut, true);
    window.addEventListener('keydown', onKeyDown, true);

    return () => {
      window.removeEventListener('pointermove', onPointerMove, true);
      window.removeEventListener('blur', onPointerLeaveWindow);
      window.removeEventListener('focusin', onFocusIn, true);
      window.removeEventListener('focusout', onFocusOut, true);
      window.removeEventListener('keydown', onKeyDown, true);
      if (rafRef.current != null) window.cancelAnimationFrame(rafRef.current);
    };
  }, []);

  if (!state.visible) return null;

  return (
    <div className="app-tooltip-layer" aria-hidden="true">
      <div className="app-tooltip" style={{ left: state.x, top: state.y }}>
        {state.text}
      </div>
    </div>
  );
};

export default TooltipLayer;
