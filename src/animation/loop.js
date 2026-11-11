import { T } from '../three.js';
import { glow } from '../scene/helpers.js';
import { applyLivePose } from '../robot/model.js';

/** Run the per-frame loop for both live GPU Learning Mode and Manual Sandbox Mode. */
export function startAnimationLoop({
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
  getMode = () => 'learn',
  getBestDistance = () => 0,
}) {
  const tgt = new T.Vector3(0, 2.6, 0);
  const footWorld = new T.Vector3();
  let ph = 0;
  let wb = 0;
  let yaw = 0;
  let t = 0;
  let last = performance.now();

  function frame(now) {
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    t += dt;

    const mode = getMode();
    if (mode === 'learn') {
      const snap = liveClient?.samplePose(now);
      if (snap) {
        applyLivePose({ bot, hips, joints }, snap);
        const footNodes = [legs.lR.ft, legs.lL.ft];
        for (let i = 0; i < 2; i++) {
          const ring = contactRings[i];
          if (snap.contacts[i]) {
            footNodes[i].getWorldPosition(footWorld);
            ring.position.set(footWorld.x, 0.132, footWorld.z);
            ring.visible = true;
          } else {
            ring.visible = false;
          }
        }
      }
      const bestDist = getBestDistance();
      if (bestDist > 0.35) {
        bestMarker.position.x = bestDist;
        bestMarker.visible = true;
      } else {
        bestMarker.visible = false;
      }
    } else {
      contactRings[0].visible = false;
      contactRings[1].visible = false;
      bestMarker.visible = false;

      const { keys } = state;
      const dir =
        state.btn ||
        ((keys.d || keys.arrowright ? 1 : 0) - (keys.a || keys.arrowleft ? 1 : 0));
      const wantP = state.pbtn || keys.arrowdown ? 1 : 0;
      state.pr += ((state.jy > 0 ? 0 : wantP) - state.pr) * Math.min(1, dt * 7);
      const run = keys.shift ? 1.8 : 1;
      const spd = 5 * run * (1 - 0.65 * state.pr);

      if (state.jy > 0 || state.vy > 0) {
        state.vy -= 26 * dt;
        state.jy += state.vy * dt;
        if (state.jy <= 0) {
          state.jy = 0;
          state.vy = 0;
        }
      }
      state.ja += ((state.jy > 0 ? 1 : 0) - state.ja) * Math.min(1, dt * 10);

      bot.position.x = Math.max(-330, Math.min(330, bot.position.x + dir * spd * dt));
      wb += ((dir ? 1 : 0) - wb) * Math.min(1, dt * 8);
      ph += dir * dt * spd * 1.35;
      yaw += ((dir > 0 ? Math.PI / 2 : dir < 0 ? -Math.PI / 2 : 0) - yaw) * Math.min(1, dt * 6);
      bot.rotation.order = 'YXZ';
      bot.rotation.y = yaw;
      bot.rotation.x = state.pr * (Math.PI / 2);
      bot.position.y = state.jy + state.pr * 0.55;

      const o = {};
      joints.forEach((j) => {
        o[j[0]] = pose[j[0]];
      });
      const s = Math.sin(ph);
      const c = Math.cos(ph);
      const kn = (x) => Math.max(0, x) * 1.0;

      o.rh += s * 0.6 * wb;
      o.lh -= s * 0.6 * wb;
      o.rk += kn(c) * wb;
      o.lk += kn(-c) * wb;
      o.rs -= s * 0.5 * wb;
      o.ls += s * 0.5 * wb;
      o.re += 0.5 * wb;
      o.le += 0.5 * wb;
      o.wt += s * 0.12 * wb;
      o.ra += (o.rh - o.rk) * 0.7 * wb;
      o.la += (o.lh - o.lk) * 0.7 * wb;

      if (!dir) o.hy += Math.sin(t * 0.8) * 0.12;

      o.rk += 0.9 * state.ja;
      o.lk += 0.9 * state.ja;
      o.rs += 1.4 * state.ja;
      o.ls += 1.4 * state.ja;
      o.re += 0.6 * state.ja;
      o.le += 0.6 * state.ja;
      o.rh += 0.5 * state.ja;
      o.lh += 0.5 * state.ja;

      o.hx -= state.pr;
      o.rr += 0.5 * state.pr;
      o.lr += 0.5 * state.pr;
      o.re += state.pr;
      o.le += state.pr;
      o.ra += 0.5 * state.ja + 0.6 * state.pr;
      o.la += 0.5 * state.ja + 0.6 * state.pr;

      if (state.waveT > 0) {
        state.waveT -= dt;
        o.rr = 2.5;
        o.rs = 0;
        o.re = 1.1 + Math.sin(t * 10) * 0.5;
      }

      joints.forEach((j) => {
        j[2].rotation[j[3]] = o[j[0]] * j[6];
      });
      hips.position.y = 2.55 + Math.abs(c) * 0.07 * wb;
    }

    glow.emissiveIntensity = 1.6 + Math.sin(t * 3) * 0.5;

    // Camera + lights follow robot smoothly in both modes
    const { orb } = state;
    const bx = bot.position.x;
    tgt.x += (bx - tgt.x) * Math.min(1, dt * 4);
    if (state.auto && t < 6) orb.th += dt * 0.25;
    cam.position.set(
      tgt.x + orb.r * Math.sin(orb.p) * Math.sin(orb.th),
      tgt.y + orb.r * Math.cos(orb.p),
      orb.r * Math.sin(orb.p) * Math.cos(orb.th),
    );
    dome.position.copy(cam.position);
    cam.lookAt(tgt);

    sun.position.set(bx + 30, 28, 18);
    sun.target.position.set(bx, 0, 0);
    sunMesh.position.copy(cam.position).add(new T.Vector3(150, 32, -520));

    ren.render(scene, cam);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}
