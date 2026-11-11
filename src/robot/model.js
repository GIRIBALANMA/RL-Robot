import { T } from '../three.js';
import { M, add, grp, B, C, S, glow } from '../scene/helpers.js';

/** Build the jointed humanoid robot (faces +z at idle) and its joint descriptor table. */
export function createRobot(scene) {
  const wht = M(0xf1f3f7, 0.15, 0.35);
  const blu = M(0x4b6ca4, 0.25, 0.4);
  const blk = M(0x14171c, 0.5, 0.4);
  const red = M(0xd8344a, 0.2, 0.5);

  const bot = new T.Group();
  scene.add(bot);

  const hips = grp(bot, 0, 2.55, 0);
  add(hips, B(1.0, 0.4, 0.65), wht);
  add(hips, C(0.38, 0.38, 0.2, 16), blk, 0, 0.2, 0);

  const torso = grp(hips, 0, 0.25, 0);
  add(torso, B(0.8, 0.2, 0.55), blk, 0, 0.05, 0);
  add(torso, B(1.1, 0.9, 0.7), blu, 0, 0.65, 0);
  add(torso, B(1.35, 0.8, 0.8), wht, 0, 1.3, 0);
  add(torso, B(0.55, 0.7, 0.1), blu, 0, 1.3, 0.43);
  add(torso, B(0.3, 0.4, 0.1), wht, 0, 0.65, 0.36);

  const neck = grp(torso, 0, 1.7, 0);
  add(neck, C(0.13, 0.16, 0.25, 10), blu, 0, 0.05, 0);

  const head = grp(neck, 0, 0.15, 0);
  add(head, S(0.42), wht, 0, 0.4, 0).scale.set(1, 1.05, 1.1);
  add(head, S(0.3), blk, 0, 0.38, 0.3).scale.set(0.95, 0.7, 0.7);
  add(head, B(0.1, 0.03, 0.55), glow, 0, 0.84, 0);
  [-1, 1].forEach((s) => {
    add(head, C(0.15, 0.15, 0.12, 16), blu, s * 0.43, 0.4, 0).rotation.z = Math.PI / 2;
    add(head, C(0.09, 0.09, 0.14, 12), wht, s * 0.43, 0.4, 0).rotation.z = Math.PI / 2;
  });

  const mkArm = (s) => {
    const sh = grp(torso, s * 0.9, 1.5, 0);
    add(sh, S(0.3), wht);
    add(sh, B(0.5, 0.22, 0.5), blu, 0, 0.22, 0);
    add(sh, C(0.2, 0.17, 1.2, 12), wht, 0, -0.65, 0);
    add(sh, C(0.205, 0.205, 0.1, 12), red, 0, -0.45, 0);
    const el = grp(sh, 0, -1.3, 0);
    add(el, S(0.2), blk);
    add(el, C(0.17, 0.14, 1.05, 12), blu, 0, -0.6, 0);
    add(el, C(0.175, 0.16, 0.45, 12), wht, 0, -0.25, 0);
    const wr = grp(el, 0, -1.2, 0);
    add(wr, B(0.26, 0.3, 0.26), blk, 0, -0.1, 0);
    return { sh, el };
  };

  const mkLeg = (s) => {
    const hp = grp(hips, s * 0.3, -0.15, 0);
    add(hp, S(0.22), blk);
    add(hp, C(0.27, 0.2, 1.2, 14), wht, 0, -0.65, 0);
    for (let i = 0; i < 3; i++) {
      add(hp, B(0.06, 0.04, 0.04), red, s * 0.05, -0.4 - i * 0.1, 0.25).rotation.z = 0.4;
    }
    const kn = grp(hp, 0, -1.2, 0);
    add(kn, S(0.24), blk);
    add(kn, B(0.3, 0.3, 0.15), blu, 0, 0, 0.22);
    add(kn, C(0.2, 0.15, 1.1, 12), blu, 0, -0.6, 0);
    add(kn, B(0.28, 0.8, 0.1), wht, 0, -0.55, 0.2);
    const ft = grp(kn, 0, -1.15, 0);
    add(ft, B(0.42, 0.2, 0.9), wht, 0, 0.02, 0.2);
    add(ft, B(0.43, 0.05, 0.5), blu, 0, 0.05, 0.2);
    add(ft, B(0.44, 0.06, 0.92), blk, 0, -0.07, 0.2);
    return { hp, kn, ft };
  };

  const aR = mkArm(1);
  const aL = mkArm(-1);
  const lR = mkLeg(1);
  const lL = mkLeg(-1);
  bot.traverse((o) => {
    if (o.isMesh) o.castShadow = true;
  });

  // joints: [key, label, group, axis, min, max, sign]
  const joints = [
    ['hy', 'Head turn', head, 'y', -1, 1, 1],
    ['hx', 'Head tilt', head, 'x', -0.6, 0.6, 1],
    ['wt', 'Waist twist', torso, 'y', -0.8, 0.8, 1],
    ['rs', 'R shoulder swing', aR.sh, 'x', -2.5, 2.5, -1],
    ['rr', 'R arm raise', aR.sh, 'z', 0, 2.8, 1],
    ['re', 'R elbow', aR.el, 'x', 0, 2.4, -1],
    ['ls', 'L shoulder swing', aL.sh, 'x', -2.5, 2.5, -1],
    ['lr', 'L arm raise', aL.sh, 'z', 0, 2.8, -1],
    ['le', 'L elbow', aL.el, 'x', 0, 2.4, -1],
    ['rh', 'R hip', lR.hp, 'x', -1.2, 1.2, -1],
    ['rk', 'R knee', lR.kn, 'x', 0, 2.2, 1],
    ['lh', 'L hip', lL.hp, 'x', -1.2, 1.2, -1],
    ['lk', 'L knee', lL.kn, 'x', 0, 2.2, 1],
    ['ra', 'R ankle', lR.ft, 'x', -0.8, 0.8, 1],
    ['la', 'L ankle', lL.ft, 'x', -0.8, 0.8, 1],
  ];

  return { bot, hips, joints, legs: { lR, lL } };
}

/** Apply a streamed physics snapshot directly to the 3D robot rig around hips origin. */
export function applyLivePose({ bot, hips, joints }, snap) {
  bot.position.set(snap.pos[0], snap.pos[1] - 2.55, snap.pos[2]);
  bot.rotation.set(0, 0, 0);
  hips.position.y = 2.55;
  hips.rotation.order = 'YXZ';
  hips.rotation.set(snap.eul[0], snap.eul[1], snap.eul[2]);
  for (let i = 0; i < joints.length; i++) {
    const j = joints[i];
    j[2].rotation[j[3]] = snap.joints[i] * j[6];
  }
}

/** Restore root hierarchy transforms when switching back to Manual Sandbox mode. */
export function resetToManualStance({ bot, hips }) {
  hips.rotation.set(0, 0, 0);
  hips.position.y = 2.55;
  bot.position.set(0, 0, 0);
  bot.rotation.set(0, 0, 0);
}
