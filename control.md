# Robot Control Reference

Everything about how the robot in `robot-road.html` is built, posed and moved. Stack: Three.js r128, plain JS, one HTML file.

> **Physics note:** there is no physics engine. Movement is kinematic: joint angles are computed from sine waves and blended. The only real "physics" is the jump (gravity + velocity). There is no collision, no ground contact detection and no IK.

---

## 1. Coordinate system

| Axis | Meaning |
|------|---------|
| `+x` | along the road, to the right (the road runs left to right) |
| `+y` | up |
| `+z` | toward the default camera; the robot's front faces `+z` at idle |

- Units are arbitrary world units. Robot height is about 5.2.
- The road lies along the x axis at `z = 0`, and the robot walks on it.
- The robot is clamped to `x ∈ [-330, 330]`. The road is 720 long.

---

## 2. Body dimensions (units)

| Part | Value |
|------|-------|
| Hip (root) height | `hips.position.y = 2.55` |
| Hip joint | 0.15 below `hips`, at x = ±0.3 |
| Thigh length (hip to knee) | 1.2 |
| Shin length (knee to ankle) | 1.15 |
| Foot height above ground | about 0.05 |
| Waist (torso origin) | 0.25 above `hips` |
| Shoulder | torso y = 1.5, x = ±0.9 |
| Upper arm (shoulder to elbow) | 1.3 |
| Forearm (elbow to wrist) | 1.2 |
| Neck / head | torso y = 1.7, head 0.15 above neck |

---

## 3. Joint hierarchy

```
bot                       world position + yaw (facing) + prone pitch
└─ hips                   root of the body (height 2.55, bobs while walking)
   ├─ torso               waist joint (twist)
   │  ├─ neck
   │  │  └─ head          head joint (turn, tilt)
   │  ├─ aR.sh             right shoulder
   │  │  └─ aR.el          right elbow
   │  │     └─ wrist/hand  (fixed, no joint)
   │  └─ aL.sh             left shoulder
   │     └─ aL.el          left elbow
   ├─ lR.hp                right hip
   │  └─ lR.kn             right knee
   │     └─ lR.ft          right ankle (foot)
   └─ lL.hp                left hip
      └─ lL.kn             left knee
         └─ lL.ft          left ankle (foot)
```

"Right" and "left" are by screen side when the robot faces the camera (`x = +` is R). Each child group's position is relative to its parent, so rotating a parent carries all children with it.

---

## 4. Joint table (the `J` array)

```js
// [key, label, group, axis, min, max, sign]
const J=[
 ['hy','Head turn',        head,  'y', -1,   1,   1],
 ['hx','Head tilt',        head,  'x', -.6,  .6,  1],
 ['wt','Waist twist',      torso, 'y', -.8,  .8,  1],
 ['rs','R shoulder swing', aR.sh, 'x', -2.5, 2.5,-1],
 ['rr','R arm raise',      aR.sh, 'z', 0,    2.8, 1],
 ['re','R elbow',          aR.el, 'x', 0,    2.4,-1],
 ['ls','L shoulder swing', aL.sh, 'x', -2.5, 2.5,-1],
 ['lr','L arm raise',      aL.sh, 'z', 0,    2.8,-1],
 ['le','L elbow',          aL.el, 'x', 0,    2.4,-1],
 ['rh','R hip',            lR.hp, 'x', -1.2, 1.2,-1],
 ['rk','R knee',           lR.kn, 'x', 0,    2.2, 1],
 ['lh','L hip',            lL.hp, 'x', -1.2, 1.2,-1],
 ['lk','L knee',           lL.kn, 'x', 0,    2.2, 1],
 ['ra','R ankle',          lR.ft, 'x', -.8,  .8,  1],
 ['la','L ankle',          lL.ft, 'x', -.8,  .8,  1]
];
```

The final write to the scene is:

```js
group.rotation[axis] = value * sign   // radians
```

### Sign conventions (what a positive slider value means)

| Joint | Positive value | Why the sign is what it is |
|-------|----------------|-----------------------------|
| Shoulder swing (`rs`, `ls`) | arm swings forward | `sign = -1` (rotation.x positive would swing back) |
| Arm raise (`rr`, `lr`) | arm lifts outward (sideways) | `sign` flips per side so both go outward |
| Elbow (`re`, `le`) | forearm bends forward and up | `sign = -1` |
| Hip (`rh`, `lh`) | leg swings forward | `sign = -1` |
| Knee (`rk`, `lk`) | shin bends backward (heel to bottom) | `sign = +1` |
| Ankle (`ra`, `la`) | toes point down | `sign = +1` |
| Head tilt (`hx`) | looks down | `sign = +1` |
| Head turn, waist twist | rotate about the vertical axis | `sign = +1` |

Elbows and knees only bend one way (min 0), like human joints.

---

## 5. Pose system (how the final angles are produced)

Each frame, every joint value is built in layers:

```
final = slider value (P[key])
      + walk animation  × wb        (walk blend, 0..1)
      + jump pose       × ja        (air blend, 0..1)
      + prone pose      × pr        (prone blend, 0..1)
      + idle head sway
then wave overrides the right arm while waveT > 0
```

- `P` holds the slider values (your manual pose). The "Reset" button zeroes them.
- `o` is the per-frame working copy: `const o={}; J.forEach(j=>o[j[0]]=P[j[0]]);`
- The blends `wb`, `ja`, `pr` smoothly move toward 0 or 1, so poses fade in and out instead of snapping.

---

## 6. Movement

### 6.1 Walk and run

```js
const dir = btn || (right?1:0) - (left?1:0);   // -1, 0, +1 along the road
const spd = 5 * run * (1 - .65*pr);             // run = 1.8 with Shift
bot.position.x += dir * spd * dt;               // clamped to ±330
ph += dir * dt * spd * 1.35;                    // gait phase (radians)
```

| Mode | Speed (units/s) |
|------|-----------------|
| Walk | 5 |
| Run (Shift) | 9 |
| Crawl (prone) | 1.75 |
| Run while prone | 3.15 |

Walk cycle (all multiplied by the walk blend `wb`):

```js
s = Math.sin(ph); c = Math.cos(ph);
o.rh += s*.6*wb;   o.lh -= s*.6*wb;          // hips swing opposite
o.rk += max(0, c)*wb; o.lk += max(0,-c)*wb;  // knee bends on the forward swing
o.rs -= s*.5*wb;   o.ls += s*.5*wb;          // arms counter-swing the legs
o.re += .5*wb;     o.le += .5*wb;            // elbows stay slightly bent
o.wt += s*.12*wb;                            // small waist twist
o.ra += (o.rh - o.rk)*.7*wb;                 // ankle keeps foot roughly level
o.la += (o.lh - o.lk)*.7*wb;
hips.position.y = 2.55 + Math.abs(c)*.07*wb; // body bob (0.07 units)
```

- Gait phase advances at `spd × 1.35` rad/s, so one full step cycle takes about 0.93 s when walking.
- `wb` eases toward 1 while moving and 0 when stopped: `wb += (target - wb) * min(1, dt*8)`.

### 6.2 Facing direction

```js
yaw += (target - yaw) * min(1, dt*6);   // smooth turn
target = dir>0 ? +π/2 : dir<0 ? -π/2 : 0
```

- Moving right turns the robot to face `+x`, moving left to face `-x`.
- When idle it turns back to face the camera (`yaw = 0`).
- `bot.rotation.order = 'YXZ'` so yaw is applied first and prone pitch stays in the robot's own frame.

### 6.3 Idle

When `dir == 0`, the head slowly turns: `o.hy += sin(t*.8) * .12`.

---

## 7. Jump (the only real physics)

```js
// trigger (Arrow Up or Jump button), only on the ground and not prone
if (jy === 0 && pr < .3) vy = 10;

// every frame while airborne
vy -= 26 * dt;       // gravity
jy += vy * dt;       // height
if (jy <= 0) { jy = 0; vy = 0; }   // land
bot.position.y = jy + pr*.55;
```

| Value | Result |
|-------|--------|
| Launch speed `vy` | 10 units/s |
| Gravity | 26 units/s² |
| Peak height | `v²/(2g)` ≈ 1.9 units |
| Air time | `2v/g` ≈ 0.77 s |

Air pose (blend `ja`, eased at `dt*10`):

```js
o.rk += .9*ja; o.lk += .9*ja;   // knees tuck
o.rh += .5*ja; o.lh += .5*ja;   // hips lift forward
o.rs += 1.4*ja; o.ls += 1.4*ja; // arms swing up
o.re += .6*ja; o.le += .6*ja;   // elbows bend
o.ra += .5*ja; o.la += .5*ja;   // toes point down
```

You can move left or right while airborne. You cannot jump while prone.

---

## 8. Prone (lie down and crawl)

Hold Arrow Down or the Prone button.

```js
pr += (target - pr) * min(1, dt*7);        // ease in and out (target 0 while airborne)
bot.rotation.x = pr * π/2;                 // pitch the whole body face-down
bot.position.y = jy + pr*.55;              // lift so the body lies on the road
o.hx -= pr;                                // head tilts up to look forward
o.rr += .5*pr; o.lr += .5*pr;              // arms out slightly
o.re += pr;    o.le += pr;                 // elbows bent
o.ra += .6*pr; o.la += .6*pr;              // toes pointed
```

- Movement is slowed by `(1 - .65*pr)`, and the walk cycle drives a crawl.
- Released, the robot eases back to standing.

---

## 9. Wave

W key or the Wave button. For 3 seconds (`waveT = 3`):

```js
o.rr = 2.5;                            // right arm raised
o.rs = 0;                              // no swing
o.re = 1.1 + Math.sin(t*10) * .5;      // forearm oscillates (about 1.6 Hz)
```

It overrides the right arm only. Everything else keeps working while waving.

---

## 10. Controls summary

| Input | Action |
|-------|--------|
| A / D or ← / → | walk along the road |
| Shift | run (×1.8) |
| ↑ or Jump button | jump |
| ↓ (hold) or Prone button | lie down, crawl |
| W or Wave button | wave right arm |
| Reset button | zero all joint sliders |
| Joint sliders | manual pose, one slider per `J` entry |
| ◀ / ▶ buttons (hold) | walk left / right on touch |

Movement keys are world-based: left/right always means along the road, whatever the camera angle.

---

## 11. Camera (orbit)

```js
const orb = { th:.35, p:1.3, r:15 };   // azimuth, polar angle, distance
cam.position.set(
  tgt.x + r * sin(p) * sin(th),
  tgt.y + r * cos(p),
  r * sin(p) * cos(th));
cam.lookAt(tgt);                        // tgt follows the robot's x, y = 2.6
```

| Input | Action |
|-------|--------|
| Drag | orbit (azimuth) and tilt |
| Wheel / pinch | zoom, distance clamped to 5..60 |

- Polar angle is limited to `0.15..1.62` so the camera stays above the ground.
- The camera target eases toward the robot: `tgt.x += (bx - tgt.x) * min(1, dt*4)`.
- For the first 6 seconds the camera auto-rotates (`orb.th += dt*.25`) until you touch it.
- The sky dome and sun mesh follow the camera so they stay at infinity.

---

## 12. State variables

| Variable | Meaning |
|----------|---------|
| `P{}` | slider values per joint key |
| `dir` | movement direction, -1 / 0 / +1 |
| `ph` | gait phase (radians) |
| `wb` | walk blend 0..1 |
| `ja` | air-pose blend 0..1 |
| `pr` | prone blend 0..1 (also used as the body pitch) |
| `jy`, `vy` | jump height and vertical velocity |
| `yaw` | current facing angle |
| `waveT` | wave timer (seconds left) |
| `t` | elapsed time (seconds) |
| `dt` | frame time, clamped to 0.05 s |

---

## 13. How to add a new joint

1. Create a group for it in the model and keep a reference (for example the wrist: `const wr=grp(el,0,-1.2,0)` and return it from `mkArm`).
2. Add one row to `J`, for example `['rw','R wrist',aR.wr,'x',-1,1,-1]`.
3. The slider is created automatically from the `J` row.
4. For animation, add to `o`, for example `o.rw += s*.2*wb;`.
5. Choose `sign` by testing: if the joint moves the wrong way, flip it.

---

## 14. Known limits

- No inverse kinematics: feet can slide slightly on the road while walking.
- No collisions with scenery.
- Ankle compensation is a fixed 0.7 factor, not a real ground-contact solve.
- Prone height (`.55`) is a fixed offset tuned by eye.
- Movement keys are tied to the road axis, not to the camera direction.
