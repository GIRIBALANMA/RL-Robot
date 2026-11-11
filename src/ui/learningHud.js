/** Learning Lab HUD controller: isolated Stand/Walk modes, telemetry readouts, Canvas chart, and Model Vault. */
export function createLearningHud({ liveClient, onModeChange }) {
  let mode = 'learn';
  let activeRobotMode = 'stand';
  let bestDistance = 0;
  let lastMilestone = '';
  let currentPaused = false;

  const tabLearn = document.getElementById('tab-learn');
  const tabManual = document.getElementById('tab-manual');
  const panelLearn = document.getElementById('panel-learn');
  const panelManual = document.getElementById('panel-manual');

  const connDot = document.getElementById('conn-dot');
  const connText = document.getElementById('conn-text');
  const gpuText = document.getElementById('gpu-text');

  const stageStandEl = document.getElementById('step-stand');
  const stageWalkEl = document.getElementById('step-walk');
  const milestoneEl = document.getElementById('milestone-banner');

  const valEp = document.getElementById('val-ep');
  const valRew = document.getElementById('val-rew');
  const valUp = document.getElementById('val-up');
  const valDist = document.getElementById('val-dist');
  const valLoss = document.getElementById('val-loss');
  const valSteps = document.getElementById('val-steps');

  const btnPause = document.getElementById('btn-pause');
  const nudgeBtns = [...document.querySelectorAll('[data-nudge]')];
  const speedBtns = [...document.querySelectorAll('[data-speed]')];
  const stageBtns = [...document.querySelectorAll('[data-stage]')];
  const policyBtns = [...document.querySelectorAll('[data-policy]')];

  const btnSave = document.getElementById('btn-save');
  const btnLoad = document.getElementById('btn-load');
  const btnResetWeights = document.getElementById('btn-reset-weights');
  const vaultModeTag = document.getElementById('vault-mode-tag');
  const modelMetaEl = document.getElementById('model-meta');
  const dlPt = document.getElementById('dl-pt');
  const dlOnnx = document.getElementById('dl-onnx');

  const chartCanvas = document.getElementById('reward-chart');
  const chartCtx = chartCanvas.getContext('2d');

  function setMode(nextMode) {
    mode = nextMode;
    const isLearn = mode === 'learn';
    tabLearn.setAttribute('aria-selected', String(isLearn));
    tabManual.setAttribute('aria-selected', String(!isLearn));
    tabLearn.toggleAttribute('focusgroupstart', isLearn);
    tabManual.toggleAttribute('focusgroupstart', !isLearn);
    panelLearn.hidden = !isLearn;
    panelManual.hidden = isLearn;
    onModeChange?.(mode);
  }

  tabLearn.addEventListener('click', () => setMode('learn'));
  tabManual.addEventListener('click', () => setMode('manual'));
  [tabLearn, tabManual].forEach((tab) => {
    tab.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        const next = tab === tabLearn ? tabManual : tabLearn;
        next.focus();
        setMode(next === tabLearn ? 'learn' : 'manual');
      }
    });
  });

  btnPause.addEventListener('click', () => {
    currentPaused = !currentPaused;
    btnPause.setAttribute('aria-pressed', String(currentPaused));
    btnPause.textContent = currentPaused ? 'Resume' : 'Pause';
    liveClient.setPaused(currentPaused);
  });

  nudgeBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      liveClient.nudgeRobot(btn.dataset.nudge).catch(() => {});
    });
  });

  speedBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const sp = Number(btn.dataset.speed);
      speedBtns.forEach((b) => b.setAttribute('aria-pressed', String(b === btn)));
      liveClient.setSpeed(sp);
    });
  });

  stageBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const st = btn.dataset.stage;
      activeRobotMode = st === 'walk' ? 'walk' : 'stand';
      stageBtns.forEach((b) => b.setAttribute('aria-pressed', String(b === btn)));
      liveClient.setRobotMode(activeRobotMode);
    });
  });

  policyBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const pm = btn.dataset.policy;
      policyBtns.forEach((b) => b.setAttribute('aria-pressed', String(b === btn)));
      liveClient.setPlayMode(pm);
    });
  });

  btnSave.addEventListener('click', async () => {
    btnSave.disabled = true;
    try {
      const res = await liveClient.saveModel(activeRobotMode);
      if (res?.model) renderModelInfo(res.model, activeRobotMode);
    } finally {
      btnSave.disabled = false;
    }
  });

  btnLoad.addEventListener('click', async () => {
    btnLoad.disabled = true;
    try {
      const res = await liveClient.loadModel(activeRobotMode);
      if (res?.model) renderModelInfo(res.model, activeRobotMode);
    } finally {
      btnLoad.disabled = false;
    }
  });

  btnResetWeights.addEventListener('click', async () => {
    btnResetWeights.disabled = true;
    try {
      await liveClient.resetModel(activeRobotMode);
    } finally {
      btnResetWeights.disabled = false;
    }
  });

  function renderModelInfo(info, robotMode = activeRobotMode) {
    if (vaultModeTag) {
      vaultModeTag.textContent = `models/ · ${robotMode}`;
    }
    if (!info) {
      modelMetaEl.textContent = `No ${robotMode} checkpoint saved yet — click Save Now or wait for auto-save.`;
      dlPt.hidden = true;
      dlOnnx.hidden = true;
      return;
    }
    const ptKb = Math.round((info.pt_bytes || 0) / 1024);
    const onnxKb = Math.round((info.onnx_bytes || 0) / 1024);
    modelMetaEl.innerHTML =
      `<strong>${info.pt_file}</strong> (${ptKb} KB)` +
      (info.onnx_file ? ` · <strong>${info.onnx_file}</strong> (${onnxKb} KB)` : '') +
      `<br>Ep ${info.episode ?? 0} · Best dist ${info.best_distance ?? 0}u · ${info.saved_at ?? ''}`;
    dlPt.href = `/api/models/download/${encodeURIComponent(info.pt_file)}`;
    dlPt.hidden = !info.pt_file;
    if (info.onnx_file) {
      dlOnnx.href = `/api/models/download/${encodeURIComponent(info.onnx_file)}`;
      dlOnnx.hidden = false;
    } else {
      dlOnnx.hidden = true;
    }
  }

  function drawChart(history) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = chartCanvas.clientWidth || 260;
    const h = chartCanvas.clientHeight || 88;
    if (chartCanvas.width !== w * dpr || chartCanvas.height !== h * dpr) {
      chartCanvas.width = w * dpr;
      chartCanvas.height = h * dpr;
    }
    chartCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    chartCtx.clearRect(0, 0, w, h);

    // Grid lines
    chartCtx.strokeStyle = 'rgba(255,255,255,0.08)';
    chartCtx.lineWidth = 1;
    for (let y = 16; y < h; y += 24) {
      chartCtx.beginPath();
      chartCtx.moveTo(0, y);
      chartCtx.lineTo(w, y);
      chartCtx.stroke();
    }

    if (!history || history.length < 2) {
      chartCtx.fillStyle = '#8b95a8';
      chartCtx.font = '11px "JetBrains Mono", monospace';
      chartCtx.fillText('Collecting first rollout batch...', 8, h / 2 + 4);
      return;
    }

    const rewards = history.map((d) => d.r);
    const rMin = Math.min(...rewards);
    const rMax = Math.max(...rewards, rMin + 1);
    const padX = 4;
    const padY = 8;
    const spanW = w - padX * 2;
    const spanH = h - padY * 2;

    // Reward curve (amber)
    chartCtx.strokeStyle = '#ffb347';
    chartCtx.lineWidth = 2;
    chartCtx.beginPath();
    history.forEach((pt, idx) => {
      const x = padX + (idx / (history.length - 1)) * spanW;
      const y = padY + spanH * (1 - (pt.r - rMin) / (rMax - rMin));
      if (idx === 0) chartCtx.moveTo(x, y);
      else chartCtx.lineTo(x, y);
    });
    chartCtx.stroke();

    // Uprightness curve 0..1 (emerald)
    chartCtx.strokeStyle = '#2ee59d';
    chartCtx.lineWidth = 1.7;
    chartCtx.beginPath();
    history.forEach((pt, idx) => {
      const x = padX + (idx / (history.length - 1)) * spanW;
      const y = padY + spanH * (1 - Math.max(0, Math.min(1, pt.up)));
      if (idx === 0) chartCtx.moveTo(x, y);
      else chartCtx.lineTo(x, y);
    });
    chartCtx.stroke();
  }

  function updateTelemetry(msg) {
    bestDistance = Number(msg.best_distance || 0);
    currentPaused = Boolean(msg.paused);
    btnPause.setAttribute('aria-pressed', String(currentPaused));
    btnPause.textContent = currentPaused ? 'Resume' : 'Pause';

    gpuText.textContent = `${msg.device || 'GPU'} · ${msg.vram_mb ?? 0} MB VRAM`;
    valEp.textContent = String(msg.episode ?? 0);
    valRew.textContent = Number(msg.avg_reward ?? 0).toFixed(1);
    valUp.textContent = `${Math.round(Number(msg.avg_upright ?? 0) * 100)}%`;
    valDist.textContent = `${bestDistance.toFixed(1)} u`;
    valLoss.textContent = Number(msg.policy_loss ?? 0).toFixed(3);
    valSteps.textContent = `${((msg.total_steps || 0) / 1000).toFixed(1)}k`;

    const isWalk = (msg.active_mode || msg.curriculum_mode) === 'walk' || Number(msg.stage) === 1;
    activeRobotMode = isWalk ? 'walk' : 'stand';
    stageStandEl.dataset.state = isWalk ? 'queued' : 'active';
    stageWalkEl.dataset.state = isWalk ? 'active' : 'queued';

    speedBtns.forEach((b) =>
      b.setAttribute('aria-pressed', String(Number(b.dataset.speed) === Number(msg.speed_mult))),
    );
    stageBtns.forEach((b) =>
      b.setAttribute('aria-pressed', String(b.dataset.stage === activeRobotMode)),
    );
    policyBtns.forEach((b) =>
      b.setAttribute('aria-pressed', String(b.dataset.policy === msg.play_mode)),
    );

    if (msg.milestone && msg.milestone !== lastMilestone) {
      lastMilestone = msg.milestone;
      milestoneEl.textContent = msg.milestone;
    }

    renderModelInfo(msg.model_info, activeRobotMode);
    drawChart(msg.history);
  }

  function updateConnection(status) {
    connDot.dataset.status = status;
    if (status === 'connected') connText.textContent = 'Live GPU Stream';
    else if (status === 'connecting') connText.textContent = 'Connecting...';
    else connText.textContent = 'Offline (run python3 run.py)';
  }

  return {
    getMode: () => mode,
    getBestDistance: () => bestDistance,
    updateTelemetry,
    updateConnection,
  };
}

