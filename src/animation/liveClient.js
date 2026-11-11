/** WebSocket + REST client for streaming live GPU training states with smooth 60fps interpolation. */
export function createLiveClient({ onTelemetry, onConnectionChange }) {
  let ws = null;
  let reconnectTimer = null;
  let destroyed = false;

  // Two-sample snapshot interpolation buffer for silky-smooth rendering at any sim speed
  let prevSnap = null;
  let currSnap = null;
  let lastRecvMs = performance.now();
  let estIntervalMs = 16.7;

  const lerp = (a, b, t) => a + (b - a) * t;
  const lerpAngle = (a, b, t) => {
    const d = ((b - a + Math.PI) % (2 * Math.PI)) - Math.PI;
    const shortD = d < -Math.PI ? d + 2 * Math.PI : d;
    return a + shortD * t;
  };

  function connect() {
    if (destroyed) return;
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/ws`;
    onConnectionChange?.('connecting');

    try {
      ws = new WebSocket(url);
    } catch {
      scheduleReconnect();
      return;
    }

    ws.onopen = () => {
      onConnectionChange?.('connected');
    };

    ws.onmessage = (event) => {
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }
      if (msg.type === 'frame') {
        const now = performance.now();
        if (currSnap) {
          const dt = now - lastRecvMs;
          if (dt > 2 && dt < 250) {
            estIntervalMs = estIntervalMs * 0.8 + dt * 0.2;
          }
        }
        prevSnap = currSnap || msg;
        currSnap = msg;
        lastRecvMs = now;
      } else if (msg.type === 'telemetry' || msg.type === 'init') {
        onTelemetry?.(msg);
      }
    };

    ws.onclose = () => {
      onConnectionChange?.('disconnected');
      scheduleReconnect();
    };

    ws.onerror = () => {
      ws?.close();
    };
  }

  function scheduleReconnect() {
    if (destroyed || reconnectTimer) return;
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connect();
    }, 1500);
  }

  /** Return interpolated robot pose { pos, eul, joints, contacts, speed } for the current render frame. */
  function samplePose(now = performance.now()) {
    if (!currSnap) return null;
    if (!prevSnap) return currSnap;
    const alpha = Math.min(1.0, Math.max(0.0, (now - lastRecvMs) / Math.max(8, estIntervalMs)));
    const pos = [
      lerp(prevSnap.pos[0], currSnap.pos[0], alpha),
      lerp(prevSnap.pos[1], currSnap.pos[1], alpha),
      lerp(prevSnap.pos[2], currSnap.pos[2], alpha),
    ];
    const eul = [
      lerp(prevSnap.eul[0], currSnap.eul[0], alpha),
      lerpAngle(prevSnap.eul[1], currSnap.eul[1], alpha),
      lerp(prevSnap.eul[2], currSnap.eul[2], alpha),
    ];
    const joints = currSnap.joints.map((v, i) => lerp(prevSnap.joints[i], v, alpha));
    return {
      pos,
      eul,
      joints,
      contacts: currSnap.contacts,
      speed: currSnap.speed,
      step: currSnap.step,
    };
  }

  async function postJson(path, body = {}) {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return res.json();
  }

  function sendWs(payload) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    }
  }

  connect();

  return {
    samplePose,
    setPaused: (paused) => sendWs({ action: 'control', paused }),
    setSpeed: (speed_mult) => sendWs({ action: 'control', speed_mult }),
    setStageMode: (curriculum_mode) => sendWs({ action: 'control', curriculum_mode }),
    setPlayMode: (play_mode) => sendWs({ action: 'control', play_mode }),
    nudgeRobot: () => postJson('/api/nudge'),
    saveModel: () => postJson('/api/models/save'),
    loadModel: () => postJson('/api/models/load'),
    resetModel: () => postJson('/api/models/reset'),
    destroy: () => {
      destroyed = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      ws?.close();
    },
  };
}

