/** Wire keyboard, action buttons, and pointer/wheel orbit controls. */
export function createControls({ canvas, resetPose, isManualMode = () => true }) {
  const state = {
    keys: {},
    btn: 0,
    pbtn: 0,
    waveT: 0,
    jy: 0,
    vy: 0,
    pr: 0,
    ja: 0,
    orb: { th: 0.35, p: 1.3, r: 15 },
    auto: 1,
  };

  const jump = () => {
    if (state.jy === 0 && state.pr < 0.3) state.vy = 10;
  };

  addEventListener('keydown', (e) => {
    if (!isManualMode()) return;
    state.keys[e.key.toLowerCase()] = 1;
    if (e.key.toLowerCase() === 'w') state.waveT = 3;
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      jump();
    }
    if (e.key === 'ArrowDown') e.preventDefault();
  });

  addEventListener('keyup', (e) => {
    if (!isManualMode()) return;
    state.keys[e.key.toLowerCase()] = 0;
  });

  const hold = (id, d) => {
    const b = document.getElementById(id);
    b.onpointerdown = () => {
      state.btn = d;
    };
    b.onpointerup = b.onpointerleave = b.onpointercancel = () => {
      if (state.btn === d) state.btn = 0;
    };
  };

  hold('bl', -1);
  hold('br', 1);
  document.getElementById('bj').onclick = jump;

  const pb = document.getElementById('bp');
  pb.onpointerdown = () => {
    state.pbtn = 1;
  };
  pb.onpointerup = pb.onpointerleave = pb.onpointercancel = () => {
    state.pbtn = 0;
  };

  document.getElementById('bw').onclick = () => {
    state.waveT = 3;
  };
  document.getElementById('bx').onclick = resetPose;

  const ptr = new Map();
  let pd = 0;
  canvas.style.touchAction = 'none';

  canvas.addEventListener('pointerdown', (e) => {
    state.auto = 0;
    canvas.setPointerCapture(e.pointerId);
    ptr.set(e.pointerId, [e.clientX, e.clientY]);
    pd = 0;
  });

  canvas.addEventListener('pointermove', (e) => {
    const p = ptr.get(e.pointerId);
    if (!p) return;
    if (ptr.size === 1) {
      state.orb.th -= (e.clientX - p[0]) * 0.008;
      state.orb.p = Math.max(0.15, Math.min(1.62, state.orb.p - (e.clientY - p[1]) * 0.006));
    }
    ptr.set(e.pointerId, [e.clientX, e.clientY]);
    if (ptr.size === 2) {
      const [a, b] = [...ptr.values()];
      const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      if (pd) state.orb.r = Math.max(5, Math.min(60, (state.orb.r * pd) / d));
      pd = d;
    }
  });

  const pup = (e) => {
    ptr.delete(e.pointerId);
    pd = 0;
  };
  canvas.addEventListener('pointerup', pup);
  canvas.addEventListener('pointercancel', pup);

  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      state.orb.r = Math.max(5, Math.min(60, state.orb.r * (1 + e.deltaY * 0.001)));
    },
    { passive: false },
  );

  return state;
}
