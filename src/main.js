import { createSceneSetup } from './scene/setup.js';
import { buildScenery } from './scene/scenery.js';
import { createRobot, resetToManualStance } from './robot/model.js';
import { createJointsPanel } from './ui/jointsPanel.js';
import { createControls } from './input/controls.js';
import { startAnimationLoop } from './animation/loop.js';
import { createLiveClient } from './animation/liveClient.js';
import { createLearningHud } from './ui/learningHud.js';

const { ren, scene, cam, dome, sun } = createSceneSetup();
const { sunMesh, bestMarker, contactRings } = buildScenery(scene);
const robotRig = createRobot(scene);
const { bot, hips, joints, legs } = robotRig;
const { pose, resetPose } = createJointsPanel(joints);

let hudRef = null;
const liveClient = createLiveClient({
  onTelemetry: (msg) => hudRef?.updateTelemetry(msg),
  onConnectionChange: (status) => hudRef?.updateConnection(status),
});

hudRef = createLearningHud({
  liveClient,
  onModeChange: (nextMode) => {
    if (nextMode === 'manual') {
      resetToManualStance(robotRig);
    }
  },
});

const state = createControls({
  canvas: ren.domElement,
  resetPose,
  isManualMode: () => hudRef.getMode() === 'manual',
});

startAnimationLoop({
  ren,
  scene,
  cam,
  dome,
  sun,
  sunMesh,
  bot,
  hips,
  joints,
  legs,
  pose,
  state,
  bestMarker,
  contactRings,
  liveClient,
  getMode: () => hudRef.getMode(),
  getBestDistance: () => hudRef.getBestDistance(),
});
