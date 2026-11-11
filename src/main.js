import { createSceneSetup } from './scene/setup.js';
import { buildScenery } from './scene/scenery.js';
import { createRobot } from './robot/model.js';
import { createJointsPanel } from './ui/jointsPanel.js';
import { createControls } from './input/controls.js';
import { startAnimationLoop } from './animation/loop.js';

const { ren, scene, cam, dome, sun } = createSceneSetup();
const { sunMesh } = buildScenery(scene);
const { bot, hips, joints } = createRobot(scene);
const { pose, resetPose } = createJointsPanel(joints);
const state = createControls({ canvas: ren.domElement, resetPose });

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
  pose,
  state,
});
