import { T } from '../three.js';

/** Initialize WebGL renderer, scene, sky dome, fog, camera, lights, and resize handler. */
export function createSceneSetup() {
  const ren = new T.WebGLRenderer({ antialias: true });
  ren.setPixelRatio(Math.min(devicePixelRatio, 2));
  ren.setSize(innerWidth, innerHeight);
  ren.shadowMap.enabled = true;
  document.body.appendChild(ren.domElement);

  const scene = new T.Scene();

  // Sky dome gradient texture
  const cv = document.createElement('canvas');
  cv.width = 2;
  cv.height = 256;
  const cx = cv.getContext('2d');
  const gr = cx.createLinearGradient(0, 0, 0, 256);
  [
    [0, '#1b2a52'],
    [0.3, '#7a4f8a'],
    [0.47, '#ff9a5a'],
    [0.5, '#ffd08a'],
    [0.52, '#e39a78'],
    [1, '#e39a78'],
  ].forEach((a) => gr.addColorStop(a[0], a[1]));
  cx.fillStyle = gr;
  cx.fillRect(0, 0, 2, 256);

  const dome = new T.Mesh(
    new T.SphereGeometry(900, 32, 16),
    new T.MeshBasicMaterial({
      map: new T.CanvasTexture(cv),
      side: T.BackSide,
      fog: false,
      depthWrite: false,
    }),
  );
  scene.add(dome);

  scene.fog = new T.Fog(0xe39a78, 80, 620);

  const cam = new T.PerspectiveCamera(40, innerWidth / innerHeight, 0.1, 1500);
  scene.add(new T.HemisphereLight(0xffc89a, 0x33382e, 0.9));

  const sun = new T.DirectionalLight(0xffb27a, 2.2);
  sun.castShadow = true;
  Object.assign(sun.shadow.camera, {
    left: -14,
    right: 14,
    top: 14,
    bottom: -14,
    near: 1,
    far: 120,
  });
  sun.shadow.mapSize.set(1024, 1024);
  scene.add(sun, sun.target);

  addEventListener('resize', () => {
    cam.aspect = innerWidth / innerHeight;
    cam.updateProjectionMatrix();
    ren.setSize(innerWidth, innerHeight);
  });

  return { ren, scene, cam, dome, sun };
}
