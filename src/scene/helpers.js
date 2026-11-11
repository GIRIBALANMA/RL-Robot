import { T } from '../three.js';

/** Create a standard PBR material. */
export const M = (c, m, r) =>
  new T.MeshStandardMaterial({ color: c, metalness: m, roughness: r });

export const steel = M(0x9aa1ab, 0.95, 0.28);
export const dark = M(0x2a2d33, 0.85, 0.45);
export const plate = M(0xc3c9d1, 0.95, 0.2);
export const glow = new T.MeshStandardMaterial({
  color: 0x330000,
  emissive: 0xff2a1a,
  emissiveIntensity: 2,
});

/** Create a mesh, position it, and attach it to a parent object. */
export const add = (p, g, m, x = 0, y = 0, z = 0) => {
  const o = new T.Mesh(g, m);
  o.position.set(x, y, z);
  p.add(o);
  return o;
};

/** Create a group, position it, and attach it to a parent object. */
export const grp = (p, x = 0, y = 0, z = 0) => {
  const o = new T.Group();
  o.position.set(x, y, z);
  p.add(o);
  return o;
};

/** Box geometry shorthand. */
export const B = (w, h, d) => new T.BoxGeometry(w, h, d);

/** Cylinder geometry shorthand. */
export const C = (a, b, h, s = 10) => new T.CylinderGeometry(a, b, h, s);

/** Sphere geometry shorthand. */
export const S = (r) => new T.SphereGeometry(r, 16, 12);
