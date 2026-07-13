/* The Polyglot Score pipeline figure: seed dataset → teacher model →
   synthetic dataset → supervised finetuning (which also takes a base model)
   → student model, revealed as a staged story that plays once. Three cubes:
   the teacher wears the brand-blue iridescent finish, the base model is a
   smaller matte-ink cube, and the student (same size as the base) ends up a
   cross between the two — the base coated partway in the teacher's finish.
   three.js comes from a CDN as an ES module — no build step. Falls back to
   flat squares without WebGL, and renders the finished state statically
   under prefers-reduced-motion. */
import * as THREE from 'https://esm.sh/three@0.168.0';
import { RoundedBoxGeometry } from 'https://esm.sh/three@0.168.0/examples/jsm/geometries/RoundedBoxGeometry.js';

const MATTE  = { color: 0x1a1a1a, metalness: 0.0,  roughness: 0.92, iridescence: 0, clearcoat: 0 };
const COATED = { color: 0x4d70ff, metalness: 0.85, roughness: 0.18, iridescence: 1, clearcoat: 1 };
const STUDENT_K = 0.62;   // how far the student gets from matte toward the teacher
const COAT_MS = 1800;     // duration of the coating sweep

// Chia-tinted equirect environment: vertical gradient plus two bright
// streaks so the clearcoat has something to reflect.
function makeEnvCanvas() {
  const c = document.createElement('canvas');
  c.width = 512; c.height = 256;
  const g = c.getContext('2d');
  const grad = g.createLinearGradient(0, 0, 0, 256);
  grad.addColorStop(0.00, '#ffffff');
  grad.addColorStop(0.25, '#f8ecda');
  grad.addColorStop(0.50, '#f2c4e4');
  grad.addColorStop(0.75, '#96abff');
  grad.addColorStop(1.00, '#254eff');
  g.fillStyle = grad;
  g.fillRect(0, 0, 512, 256);
  g.fillStyle = 'rgba(255,255,255,0.95)';
  g.fillRect(0, 42, 512, 12);
  g.fillRect(0, 148, 512, 7);
  return c;
}

function applyParams(mat, p) {
  mat.color.setHex(p.color);
  mat.metalness = p.metalness;
  mat.roughness = p.roughness;
  mat.iridescence = p.iridescence;
  mat.clearcoat = p.clearcoat;
}

function lerpParams(mat, a, b, k) {
  mat.color.lerpColors(new THREE.Color(a.color), new THREE.Color(b.color), k);
  mat.metalness = a.metalness + (b.metalness - a.metalness) * k;
  mat.roughness = a.roughness + (b.roughness - a.roughness) * k;
  mat.iridescence = a.iridescence + (b.iridescence - a.iridescence) * k;
  mat.clearcoat = a.clearcoat + (b.clearcoat - a.clearcoat) * k;
}

const smooth = (x) => x * x * (3 - 2 * x);
const envCanvas = makeEnvCanvas();
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

let coatStart = Infinity;  // set by the timeline when finetuning begins

// role: 'teacher' (blue finish), 'base' (stays matte), 'student' (matte,
// then coats partway toward the teacher once coatStart is set), or
// 'studentFinal' (already coated partway; used outside the figure).
function createCube(mount, role) {
  if (!mount) return null;
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  } catch (e) {
    mount.classList.add('is-fallback');
    return null;
  }
  const size = mount.clientWidth;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(size, size);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  mount.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 20);
  camera.position.set(0, 1.1, 4.6);
  camera.lookAt(0, 0, 0);

  const envTex = new THREE.CanvasTexture(envCanvas);
  envTex.mapping = THREE.EquirectangularReflectionMapping;
  envTex.colorSpace = THREE.SRGBColorSpace;
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromEquirectangular(envTex).texture;
  envTex.dispose();

  const light = new THREE.DirectionalLight(0xffffff, 1.4);
  light.position.set(2.5, 4, 3);
  scene.add(light);

  const material = new THREE.MeshPhysicalMaterial({
    iridescenceIOR: 1.6,
    iridescenceThicknessRange: [120, 480],
    envMapIntensity: 1.25,
    clearcoatRoughness: 0.08,
  });
  applyParams(material, role === 'teacher' ? COATED : MATTE);
  if (role === 'studentFinal') lerpParams(material, MATTE, COATED, STUDENT_K);

  const cube = new THREE.Mesh(new RoundedBoxGeometry(1.7, 1.7, 1.7, 5, 0.09), material);
  scene.add(cube);

  if (reduceMotion) {
    if (role === 'student') lerpParams(material, MATTE, COATED, STUDENT_K);
    cube.rotation.set(0.42, 0.65, 0);
    renderer.render(scene, camera);
    return null;
  }

  const tick = (tMs) => {
    const t = tMs / 1000;
    cube.rotation.y = t * 0.45;
    cube.rotation.x = 0.42 + Math.sin(t * 0.6) * 0.05;
    if (role === 'student') {
      const k = Math.min(Math.max((performance.now() - coatStart) / COAT_MS, 0), 1);
      lerpParams(material, MATTE, COATED, smooth(k) * STUDENT_K);
    }
    renderer.render(scene, camera);
  };
  return {
    start() { renderer.setAnimationLoop(tick); },
    stop()  { renderer.setAnimationLoop(null); },
  };
}

const figure = document.getElementById('pgscore-figure');
if (figure) {
  const stepEl = {};
  figure.querySelectorAll('.pg-anim').forEach((el) => { stepEl[el.dataset.step] = el; });

  const cubes = [
    createCube(document.getElementById('pg-cube-teacher'), 'teacher'),
    createCube(document.getElementById('pg-cube-base'), 'base'),
    createCube(document.getElementById('pg-cube-student'), 'student'),
  ].filter(Boolean);

  // Standalone student cube next to the extrinsic metrics box; already
  // coated, spins on its own (not tied to the figure's observer or story).
  const metricCube = createCube(document.getElementById('metric-cube-student'), 'studentFinal');
  if (metricCube) metricCube.start();

  if (reduceMotion) {
    Object.values(stepEl).forEach((el) => el.classList.add('is-on'));
  } else {
    const on = (name, ms) => setTimeout(() => stepEl[name].classList.add('is-on'), ms);

    // Plays once, in four beats (ingredients → synthesis → training →
    // payoff), with a small stagger inside each beat. The finished
    // pipeline stays on screen; the coating sweep is the finale.
    function runStory() {
      // beat 1: ingredients in
      on('seed', 200);
      on('arrow1', 350);
      on('teacher', 500);
      // beat 2: synthesis
      on('arrow2', 1300);
      on('synth', 1400);
      on('arrow2b', 1550);
      on('chat', 1700);
      // beat 3: training
      on('arrow3', 2500);
      on('sft', 2600);
      on('base', 2750);
      on('arrowbase', 2900);
      // beat 4: payoff
      on('arrow4', 3700);
      on('student', 3850);
      setTimeout(() => { coatStart = performance.now(); }, 4400);
    }

    let played = false;
    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          cubes.forEach((c) => c.start());
          if (!played) {
            played = true;
            runStory();
          }
        } else {
          cubes.forEach((c) => c.stop());
        }
      });
    }, { threshold: 0.15 });
    io.observe(figure);
  }
}
