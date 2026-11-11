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
    const tau = 2 * Math.PI;
    const d = ((((b - a + Math.PI) % tau) + tau) % tau) - Math.PI;
    return a + d * t;
  };
  const slerpQuat = (a, b, t) => {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    let bx = b[0], by = b[1], bz = b[2], bw = b[3];
    if (dot < 0) {
      dot = -dot;
      bx = -bx;
      by = -by;
      bz = -bz;
      bw = -bw;
    }
    let s0, s1;
    if (dot > 0.9995) {
      s0 = 1 - t;
      s1 = t;
    } else {
      const omega = Math.acos(Math.min(1, dot));
      const sinO = Math.sin(omega) || 1;
      s0 = Math.sin((1 - t) * omega) / sinO;
      s1 = Math.sin(t * omega) / sinO;
    }
    const x = s0 * a[0] + s1 * bx;
    const y = s0 * a[1] + s1 * by;
    const z = s0 * a[2] + s1 * bz;
    const w = s0 * a[3] + s1 * bw;
    const len = Math.hypot(x, y, z, w) || 1;
    return [x / len, y / len, z / len, w / len];
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
        const jumpedReset = !currSnap || msg.step <= currSnap.step || (currSnap.stage !== undefined && msg.stage !== currSnap.stage);
        if (currSnap && !jumpedReset) {
          const dt = now - lastRecvMs;
          if (dt > 2 && dt < 250) {
            estIntervalMs = estIntervalMs * 0.8 + dt * 0.2;
          }
        }
        prevSnap = jumpedReset ? msg : currSnap;
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

  /** Return interpolated robot pose { pos, eul, quat, joints, contacts, speed } for the current render frame. */
  function samplePose(now = performance.now()) {
    if (!currSnap) return null;
    if (!prevSnap) return currSnap;
    const alpha = Math.min(1.0, Math.max(0.0, (now - lastRecvMs) / Math.max(8, estIntervalMs)));
    const pos = [
      lerp(prevSnap.pos[0], currSnap.pos[0], alpha),
      lerp(prevSnap.pos[1], currSnap.pos[1], alpha),
      lerp(prevSnap.pos[2], currSnap.pos[2], alpha),
    ];
    const quat = (prevSnap.quat && currSnap.quat)
      ? slerpQuat(prevSnap.quat, currSnap.quat, alpha)
      : null;
    const eul = [
      lerpAngle(prevSnap.eul[0], currSnap.eul[0], alpha),
      lerpAngle(prevSnap.eul[1], currSnap.eul[1], alpha),
      lerpAngle(prevSnap.eul[2], currSnap.eul[2], alpha),
    ];
    const joints = currSnap.joints.map((v, i) => lerp(prevSnap.joints[i], v, alpha));
    return {
      pos,
      eul,
      quat,
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
    setRobotMode: (mode) => sendWs({ action: 'control', mode }),
    setStageMode: (mode) => sendWs({ action: 'control', mode }),
    setPlayMode: (play_mode) => sendWs({ action: 'control', play_mode }),
    nudgeRobot: (dir = 'random') => postJson('/api/nudge', { dir }),
    saveModel: (mode) => postJson('/api/models/save', mode ? { mode } : {}),
    loadModel: (mode) => postJson('/api/models/load', mode ? { mode } : {}),
    resetModel: (mode) => postJson('/api/models/reset', mode ? { mode } : {}),
    destroy: () => {
      destroyed = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      ws?.close();
    },
  };
}

