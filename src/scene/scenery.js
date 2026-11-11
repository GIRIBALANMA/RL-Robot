import { T } from '../three.js';
import { M, add, grp, B, C } from './helpers.js';

/** Populate ground, road, mountains, pine trees, streetlamps, and sun disc. */
export function buildScenery(scene) {
  const ground = add(scene, new T.PlaneGeometry(3000, 3000), M(0x4a5440, 0, 1), 0, 0, 0);
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;

  add(scene, B(720, 0.1, 10.5), M(0x6b6459, 0, 1), 0, 0.04, 0).receiveShadow = true;
  add(scene, B(720, 0.12, 8), M(0x23252b, 0, 0.9), 0, 0.06, 0).receiveShadow = true;

  const lineM = new T.MeshBasicMaterial({ color: 0xf2e6b8 });
  const dashG = B(3.5, 0.02, 0.2);
  for (let x = -350; x <= 350; x += 8) {
    add(scene, dashG, lineM, x, 0.13, 0);
  }
  [-3.6, 3.6].forEach((z) =>
    add(scene, B(720, 0.02, 0.12), new T.MeshBasicMaterial({ color: 0xffffff }), 0, 0.13, z),
  );

  let sd = 7;
  const rnd = () => (sd = (sd * 16807) % 2147483647) / 2147483647;

  for (let i = 0; i < 92; i++) {
    const e = i >= 80;
    const mx = e ? (i % 2 ? 1 : -1) * (440 + rnd() * 60) : -500 + (i >> 1) * 25 + rnd() * 10;
    const mz = e ? (rnd() - 0.5) * 300 : (i % 2 ? 1 : -1) * (190 + rnd() * 140);
    const h = 40 + rnd() * 80;
    const r = 30 + rnd() * 45;
    const col = new T.Color().setHSL(0.62 + rnd() * 0.05, 0.25, 0.22 + rnd() * 0.12);
    add(
      scene,
      new T.ConeGeometry(r, h, 6),
      new T.MeshStandardMaterial({ color: col, roughness: 1, flatShading: true }),
      mx,
      h / 2,
      mz,
    );
  }

  const trunkG = C(0.25, 0.3, 1.6, 6);
  const coneG = new T.ConeGeometry(1.5, 3.2, 7);
  const treeM = M(0x1f3a2a, 0, 1);
  const trunkM = M(0x4a3426, 0, 1);
  for (let i = 0; i < 170; i++) {
    const t = grp(scene, -340 + rnd() * 680, 0, (rnd() < 0.5 ? -1 : 1) * (9 + rnd() * 30));
    const s = 0.8 + rnd() * 1.4;
    t.scale.setScalar(s);
    add(t, trunkG, trunkM, 0, 0.8, 0);
    add(t, coneG, treeM, 0, 2.8, 0);
    add(t, coneG, treeM, 0, 4, 0).scale.setScalar(0.75);
  }

  const poleG = C(0.08, 0.1, 6, 6);
  const poleM = M(0x3a3d44, 0.5, 0.6);
  const lampM = new T.MeshBasicMaterial({ color: 0xffe2a8 });
  for (let x = -340; x <= 340; x += 30) {
    [-1, 1].forEach((s) => {
      const p = grp(scene, x, 0, s * 5.6);
      add(p, poleG, poleM, 0, 3, 0);
      add(p, B(0.5, 0.15, 0.9), poleM, 0, 6, -s * 0.3);
      add(p, B(0.3, 0.06, 0.5), lampM, 0, 5.9, -s * 0.4);
    });
  }

  const sunMesh = add(
    scene,
    new T.SphereGeometry(20, 24, 16),
    new T.MeshBasicMaterial({ color: 0xffd8a0, fog: false }),
    150,
    32,
    -520,
  );

  // Best-distance road marker for Walk curriculum milestones
  const bestMarker = add(
    scene,
    B(0.18, 0.03, 7.2),
    new T.MeshBasicMaterial({ color: 0x2ee59d }),
    0,
    0.135,
    0,
  );
  bestMarker.visible = false;

  // Foot ground-contact halos (Right, Left)
  const ringGeo = new T.RingGeometry(0.24, 0.36, 24);
  const contactRings = [0, 1].map(() => {
    const ring = new T.Mesh(
      ringGeo,
      new T.MeshBasicMaterial({ color: 0x2ee59d, side: T.DoubleSide, transparent: true, opacity: 0.78 }),
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.132;
    ring.visible = false;
    scene.add(ring);
    return ring;
  });

  return { sunMesh, bestMarker, contactRings };
}
