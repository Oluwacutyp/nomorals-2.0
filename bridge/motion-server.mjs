#!/usr/bin/env node
/**
 * motion-server.mjs — Devon's motion-studio host rung.
 *
 * A dependency-free Node HTTP server (stdlib only) that serves the
 * Phase 2 motion rungs on her own host: the Lottie player, the GSAP
 * demo, and the three.js scene viewer (three.quarks particles +
 * Tweakpane live tuning). Binds 127.0.0.1 only; port from
 * NM_MOTION_PORT (default 8788, next to the WhatsApp bridge's 8787).
 *
 * Routes:
 *   GET  /            index — the four rungs, versions, licences
 *   GET  /health      {"ok": true, ...}
 *   GET  /lottie      Lottie player. ?src=<url> fetches a Lottie JSON,
 *                     ?data=<url-encoded JSON> embeds one inline;
 *                     no param plays the built-in demo animation.
 *   GET  /gsap        GSAP demo page (staggered hero, Replay button)
 *   GET  /three       three.js viewer. ?scene=<url-encoded JSON scene
 *                     spec> renders it; no param renders the default
 *                     scene (spinning gold box + particle fountain).
 *   POST /scene       JSON scene-spec body -> the rendered three.js
 *                     page (400 naming the bad field on a bad spec)
 *
 * The CDN pins are the same as nomorals/media/motion_rungs.py's
 * MOTION_CDN — that module is the source of truth; the tests pin
 * the URLs on both sides. The three.js scene builder below mirrors
 * that module's THREE_BUILDER_JS (same markers, same behaviour).
 *
 * No React, no framework, no npm install: `node bridge/motion-server.mjs`.
 */

// -- pinned CDN pins (mirror of nomorals/media/motion_rungs.py) --------
const CDN = {
  lottie: 'https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.13.0/lottie.min.js',
  gsap: 'https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js',
  three: 'https://cdn.jsdelivr.net/npm/three@0.187.0/build/three.module.js',
  threeAddons: 'https://cdn.jsdelivr.net/npm/three@0.187.0/examples/jsm/',
  threeQuarks: 'https://cdn.jsdelivr.net/npm/three.quarks@0.17.1/dist/three.quarks.esm.js',
  quarksCore: 'https://cdn.jsdelivr.net/npm/quarks.core@0.17.1/dist/quarks.core.esm.js',
  tweakpane: 'https://cdn.jsdelivr.net/npm/tweakpane@4.0.5/dist/tweakpane.min.js',
};
const VERSIONS = {
  lottie: '5.13.0 (MIT)', gsap: '3.12.5 (GreenSock standard license — free incl. commercial, NOT OSI)',
  three: '0.187.0 (MIT)', threeQuarks: '0.17.1 (MIT)', tweakpane: '4.0.5 (MIT)',
};

import http from 'node:http';

const PORT = parseInt(process.env.NM_MOTION_PORT || '8788', 10) || 8788;
const HOST = '127.0.0.1';
const MAX_BODY = 1_000_000;

// -- security headers (MWG legacy-scan audit, 2026-11-14) ------------------
// Devon set no Content-Security-Policy anywhere before this. The pages
// load their pinned CDNs (cdnjs + jsDelivr) and run inline scripts, so
// the policy allows exactly those and nothing else: no framing, no
// object/embed, no form posts, no font loads, no data:/blob: scripts.
// The `?src=` fetch may go to any https URL, so connect-src allows it.
const CSP = "default-src 'none'; script-src 'unsafe-inline' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; style-src 'unsafe-inline'; img-src 'self' data: https:; connect-src https:; font-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";

function send(res, code, body, ctype = 'text/html; charset=utf-8') {
  try {
    const buf = Buffer.from(body, 'utf-8');
    res.writeHead(code, {
      'Content-Type': ctype,
      'Content-Length': buf.length,
      'Content-Security-Policy': CSP,
      'X-Content-Type-Options': 'nosniff',
      'Referrer-Policy': 'no-referrer',
      'X-Frame-Options': 'DENY',
    });
    res.end(buf);
  } catch { /* client went away */ }
}
function sendJson(res, code, obj) {
  send(res, code, JSON.stringify(obj), 'application/json');
}

// -- shared bits ---------------------------------------------------------
function importmap() {
  return '<script type="importmap">' + JSON.stringify({ imports: {
    three: CDN.three, 'three/addons/': CDN.threeAddons,
    'quarks.core': CDN.quarksCore, 'three.quarks': CDN.threeQuarks,
  } }) + '</script>';
}
function pageShell(title, head, body) {
  return '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<title>'
    + title + '</title>\n' + head + '\n</head>\n<body>\n' + body + '\n</body>\n</html>';
}

// -- Lottie ---------------------------------------------------------------
const DEMO_LOTTIE = {
  v: '5.7.4', fr: 60, ip: 0, op: 120, w: 200, h: 200, nm: 'devon-pulse',
  ddd: 0, assets: [],
  layers: [{ ddd: 0, ind: 1, ty: 4, nm: 'pulse', sr: 1,
    ks: { o: { a: 0, k: 100 }, a: { a: 0, k: [100, 100, 0] },
      p: { a: 0, k: [100, 100, 0] }, r: { a: 0, k: 0 },
      s: { a: 1, k: [
        { i: { x: [0.667, 0.667, 0.667], y: [1, 1, 1] }, o: { x: [0.333, 0.333, 0.333], y: [0, 0, 0] }, t: 0, s: [60, 60, 100] },
        { i: { x: [0.667, 0.667, 0.667], y: [1, 1, 1] }, o: { x: [0.333, 0.333, 0.333], y: [0, 0, 0] }, t: 60, s: [100, 100, 100] },
        { t: 120, s: [60, 60, 100] } ] } },
    ao: 0,
    shapes: [
      { ty: 'el', p: { a: 0, k: [0, 0] }, s: { a: 0, k: [120, 120] }, nm: 'circle' },
      { ty: 'fl', c: { a: 0, k: [1, 0.8, 0.2, 1] }, o: { a: 0, k: 100 }, r: 1, nm: 'fill' }],
    ip: 0, op: 120, st: 0, bm: 0 }],
};

function lottiePage(animation, srcUrl) {
  const dataScript = animation
    ? '<script type="application/json" id="lottie-data">'
      + JSON.stringify(animation).replace(/<\//g, '<\\/') + '</script>'
    : '';
  const loader = srcUrl
    ? "fetch(" + JSON.stringify(srcUrl) + ").then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); }).then(function (data) { start(data); }).catch(function (e) { document.getElementById('status').textContent = 'Could not load animation: ' + e.message; });"
    : 'start(JSON.parse(document.getElementById("lottie-data").textContent));';
  return pageShell('Lottie player — Devon',
    '<style>body { margin: 0; background: #14141c; color: #eee; font-family: system-ui, sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; }'
    + '#stage { width: 300px; height: 300px; } #controls { display: flex; gap: 8px; margin-top: 16px; align-items: center; }'
    + 'button { background: #2a2a38; color: #eee; border: 1px solid #555; border-radius: 6px; padding: 8px 14px; cursor: pointer; }'
    + 'button:hover { background: #3a3a4c; } select { background: #2a2a38; color: #eee; border: 1px solid #555; border-radius: 6px; padding: 8px; }'
    + '#status { margin-top: 10px; font-size: 13px; color: #999; }</style>\n'
    + '<script src="' + CDN.lottie + '"></script>',
    '<div id="stage"></div>\n<div id="controls">\n<button id="play">Pause</button>\n<button id="restart">Restart</button>\n'
    + '<label>Speed <select id="speed"><option>0.5</option><option selected>1</option><option>1.5</option><option>2</option></select></label>\n</div>\n'
    + '<div id="status">Powered by peacethefirst</div>\n' + dataScript + '\n<script>\n'
    + 'function start(animationData) {\n'
    + '  var anim = lottie.loadAnimation({ container: document.getElementById("stage"), renderer: "svg", loop: true, autoplay: true, animationData: animationData });\n'
    + '  var playBtn = document.getElementById("play");\n'
    + '  playBtn.addEventListener("click", function () { if (anim.isPaused) { anim.play(); playBtn.textContent = "Pause"; } else { anim.pause(); playBtn.textContent = "Play"; } });\n'
    + '  document.getElementById("restart").addEventListener("click", function () { anim.goToAndPlay(0); playBtn.textContent = "Pause"; });\n'
    + '  document.getElementById("speed").addEventListener("change", function (e) { anim.setSpeed(parseFloat(e.target.value)); });\n'
    + '  document.getElementById("status").textContent = "Playing — Powered by peacethefirst";\n'
    + '}\n' + loader + '\n</script>');
}

// -- GSAP -----------------------------------------------------------------
function gsapPage() {
  return pageShell('GSAP demo — Devon',
    '<style>body { margin: 0; background: #14141c; color: #eee; font-family: system-ui, sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; overflow: hidden; }'
    + 'h1 { font-size: 64px; margin: 0; letter-spacing: 2px; } h1 span { display: inline-block; } p { color: #999; font-size: 18px; }'
    + 'button { background: #2a2a38; color: #eee; border: 1px solid #555; border-radius: 6px; padding: 10px 18px; cursor: pointer; margin-top: 24px; }'
    + 'button:hover { background: #3a3a4c; } #cue { position: fixed; bottom: 24px; color: #666; font-size: 13px; }</style>\n'
    + '<script src="' + CDN.gsap + '"></script>',
    '<h1 id="hero"><span>D</span><span>e</span><span>v</span><span>o</span><span>n</span></h1>\n<p id="sub">motion, by Devon</p>\n'
    + '<button id="replay">Replay</button>\n<div id="cue">Powered by peacethefirst</div>\n<script>\n'
    + 'function playIntro() {\n'
    + "  gsap.fromTo('#hero span', { y: 80, opacity: 0, rotate: 12 }, { y: 0, opacity: 1, rotate: 0, duration: 0.7, stagger: 0.07, ease: 'back.out(1.7)' });\n"
    + "  gsap.fromTo('#sub', { opacity: 0 }, { opacity: 1, duration: 0.6, delay: 0.5 });\n"
    + "  gsap.fromTo('#replay', { opacity: 0 }, { opacity: 1, duration: 0.4, delay: 0.7 });\n"
    + '}\nplayIntro();\ndocument.getElementById("replay").addEventListener("click", playIntro);\n</script>');
}

// -- three.js -------------------------------------------------------------
const OBJECT_TYPES = ['box', 'sphere', 'plane', 'cylinder', 'torus', 'cone'];
const HEX = /^#[0-9a-fA-F]{6}$/;
function isNum(v) { return typeof v === 'number' && Number.isFinite(v); }
function isVec3(v) { return Array.isArray(v) && v.length === 3 && v.every(isNum); }

/** Light structural check mirroring nomorals/media/motion_rungs.py's
 *  validate_scene_spec — returns an error string naming the bad field,
 *  or null when the spec is structurally sound. */
function validateScene(spec) {
  if (!spec || typeof spec !== 'object' || Array.isArray(spec)) return 'scene spec: expected a dict';
  if (spec.camera !== undefined) {
    if (typeof spec.camera !== 'object') return "scene spec: 'camera' must be a dict";
    if (spec.camera.position !== undefined && !isVec3(spec.camera.position)) return 'scene spec: camera.position must be [x, y, z] numbers';
    if (spec.camera.fov !== undefined && (!isNum(spec.camera.fov) || spec.camera.fov < 10 || spec.camera.fov > 120)) return 'scene spec: camera.fov must be 10..120';
  }
  if (spec.background !== undefined && (typeof spec.background !== 'string' || !HEX.test(spec.background))) return 'scene spec: background must be a #rrggbb colour';
  if (spec.objects !== undefined) {
    if (!Array.isArray(spec.objects)) return "scene spec: 'objects' must be a list";
    for (let i = 0; i < spec.objects.length; i++) {
      const o = spec.objects[i];
      if (!o || typeof o !== 'object') return 'scene spec: objects[' + i + '] must be a dict';
      if (!OBJECT_TYPES.includes(o.type)) return 'scene spec: objects[' + i + '].type must be one of ' + OBJECT_TYPES.join(', ');
      if (o.color !== undefined && (typeof o.color !== 'string' || !HEX.test(o.color))) return 'scene spec: objects[' + i + '].color must be a #rrggbb colour';
      if (o.position !== undefined && !isVec3(o.position)) return 'scene spec: objects[' + i + '].position must be [x, y, z] numbers';
    }
  }
  if (spec.particles !== undefined && spec.particles !== null) {
    if (typeof spec.particles !== 'object') return "scene spec: 'particles' must be a dict";
    if (spec.particles.shape !== undefined && !['point', 'cone'].includes(spec.particles.shape)) return 'scene spec: particles.shape must be point or cone';
    if (spec.particles.color !== undefined && (typeof spec.particles.color !== 'string' || !HEX.test(spec.particles.color))) return 'scene spec: particles.color must be a #rrggbb colour';
  }
  return null;
}

const DEFAULT_SCENE = {
  camera: { position: [0, 2.5, 7], fov: 50 },
  background: '#101018',
  objects: [
    { type: 'box', size: [1.6, 1.6, 1.6], position: [0, 1.2, 0], color: '#ffcc33', roughness: 0.35, metalness: 0.4, spin: { x: 0.2, y: 0.6, z: 0 } },
    { type: 'plane', size: [12, 12], position: [0, 0, 0], rotation: [-1.5708, 0, 0], color: '#1c1c28', roughness: 0.9, metalness: 0 },
  ],
  lights: [
    { type: 'ambient', color: '#ffffff', intensity: 0.5 },
    { type: 'directional', color: '#fff2cc', intensity: 1.4, position: [5, 8, 4] },
    { type: 'point', color: '#ffcc33', intensity: 12, position: [0, 3, 2], distance: 12 },
  ],
  particles: { rate: 40, life: [0.8, 1.6], speed: 2.0, size: 0.25, color: '#ffcc33', position: [0, 0.2, 0], shape: 'cone', max: 500, looping: true, duration: 2.0 },
  tweak: { speed: 1.0, particleRate: 40 },
  controls: { orbit: true, autoRotate: false },
};

// The generic scene builder — mirrors nomorals/media/motion_rungs.py's
// THREE_BUILDER_JS. SPEC/PARAMS are injected per request.
const BUILDER = `
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  ParticleSystem, BatchedParticleRenderer, ConstantValue, IntervalValue,
  ConstantColor, PointEmitter, ConeEmitter,
} from 'three.quarks';

const SPEC = __SPEC_JSON__;
const PARAMS = __PARAMS_JSON__;

const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
const scene = new THREE.Scene();
scene.background = new THREE.Color(SPEC.background);
const camera = new THREE.PerspectiveCamera(
  SPEC.camera.fov, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(SPEC.camera.position[0], SPEC.camera.position[1], SPEC.camera.position[2]);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.autoRotate = !!SPEC.controls.autoRotate;

const spinners = [];
function makeGeometry(o) {
  const s = o.size;
  if (o.type === 'box') return new THREE.BoxGeometry(s[0], s[1], s[2]);
  if (o.type === 'sphere') return new THREE.SphereGeometry(s, 32, 24);
  if (o.type === 'plane') return new THREE.PlaneGeometry(s[0], s[1]);
  if (o.type === 'cylinder') return new THREE.CylinderGeometry(s[0], s[1], s[2], 32);
  if (o.type === 'torus') return new THREE.TorusGeometry(s[0], s[1], 16, 64);
  if (o.type === 'cone') return new THREE.ConeGeometry(s[0], s[1], 32);
  throw new Error('unknown object type ' + o.type);
}
for (const o of SPEC.objects) {
  const mesh = new THREE.Mesh(
    makeGeometry(o),
    new THREE.MeshStandardMaterial({ color: o.color, roughness: o.roughness, metalness: o.metalness }));
  mesh.position.set(o.position[0], o.position[1], o.position[2]);
  mesh.rotation.set(o.rotation[0], o.rotation[1], o.rotation[2]);
  if (o.spin) { mesh.userData.spin = o.spin; spinners.push(mesh); }
  scene.add(mesh);
}
for (const l of SPEC.lights) {
  let light;
  if (l.type === 'ambient') light = new THREE.AmbientLight(l.color, l.intensity);
  else if (l.type === 'directional') light = new THREE.DirectionalLight(l.color, l.intensity);
  else light = new THREE.PointLight(l.color, l.intensity, l.distance || 0);
  if (l.position) light.position.set(l.position[0], l.position[1], l.position[2]);
  scene.add(light);
}

let particleSystem = null;
let batchRenderer = null;
if (SPEC.particles) {
  const p = SPEC.particles;
  const c = new THREE.Color(p.color);
  batchRenderer = new BatchedParticleRenderer();
  scene.add(batchRenderer);
  particleSystem = new ParticleSystem({
    duration: p.duration, looping: p.looping,
    startLife: new IntervalValue(p.life[0], p.life[1]),
    startSpeed: new ConstantValue(p.speed),
    startSize: new ConstantValue(p.size),
    startColor: new ConstantColor(new THREE.Vector4(c.r, c.g, c.b, 1)),
    worldSpace: true,
    maxParticle: p.max,
    emissionOverTime: new ConstantValue(p.rate),
    shape: p.shape === 'cone' ? new ConeEmitter({ angle: 0.35, radius: 0.2 }) : new PointEmitter(),
    material: new THREE.MeshBasicMaterial({ color: p.color, transparent: true, opacity: 0.9 }),
  });
  batchRenderer.addSystem(particleSystem);
  particleSystem.emitter.position.set(p.position[0], p.position[1], p.position[2]);
  scene.add(particleSystem.emitter);
}

const pane = new window.Tweakpane.Pane({ title: 'Scene' });
for (const [key, value] of Object.entries(PARAMS)) {
  let binding;
  if (typeof value === 'number') {
    binding = pane.addBinding(PARAMS, key, { min: 0, max: Math.max(1, value * 3), step: 0.01 });
  } else if (typeof value === 'boolean') {
    binding = pane.addBinding(PARAMS, key);
  } else if (typeof value === 'string' && /^#[0-9a-fA-F]{6}$/.test(value)) {
    binding = pane.addBinding(PARAMS, key, { view: 'color' });
  } else {
    binding = pane.addBinding(PARAMS, key);
  }
  if (key === 'particleRate' && particleSystem) {
    binding.on('change', (ev) => {
      try { particleSystem.emissionOverTime.value = ev.value; } catch (e) {}
    });
  }
  if (key === 'background') {
    binding.on('change', (ev) => {
      try { scene.background = new THREE.Color(ev.value); } catch (e) {}
    });
  }
}

const clock = new THREE.Clock();
function tick() {
  const delta = clock.getDelta();
  const speed = typeof PARAMS.speed === 'number' ? PARAMS.speed : 1;
  for (const m of spinners) {
    const s = m.userData.spin;
    m.rotation.x += (s.x || 0) * delta * speed;
    m.rotation.y += (s.y || 0) * delta * speed;
    m.rotation.z += (s.z || 0) * delta * speed;
  }
  if (batchRenderer) batchRenderer.update(delta);
  controls.update();
  renderer.render(scene, camera);
}
renderer.setAnimationLoop(tick);
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
`;

function threePage(spec) {
  const params = spec.tweak || { speed: 1.0 };
  // The spec/tweak JSON is user-supplied (?scene= / POST /scene) and
  // tweak is NOT validated — embed it the way the Lottie page does:
  // escape `</` so a `</script>` inside a tweak string cannot break
  // out of the module script (MWG audit H2, 2026-11-14).
  const safeJson = function (v) {
    return JSON.stringify(v).replace(/<\//g, '<\\/');
  };
  const builder = BUILDER
    .replace('__SPEC_JSON__', safeJson(spec))
    .replace('__PARAMS_JSON__', safeJson(params));
  return pageShell('Devon 3D scene',
    '<style>body { margin: 0; background: #101018; color: #eee; font-family: system-ui, sans-serif; overflow: hidden; }'
    + '#scene { display: block; width: 100vw; height: 100vh; }'
    + '#hint { position: fixed; left: 16px; bottom: 14px; font-size: 13px; color: #999; } #hint b { color: #eee; }</style>\n'
    + importmap() + '\n<script src="' + CDN.tweakpane + '"></script>',
    '<canvas id="scene"></canvas>\n<div id="hint"><b>Devon 3D</b> — drag to orbit, scroll to zoom. Powered by peacethefirst</div>\n'
    + '<script type="module">\n' + builder + '\n</script>');
}

// -- index ----------------------------------------------------------------
function indexPage() {
  const rows = [
    ['Lottie player', '/lottie', 'lottie-web ' + VERSIONS.lottie, 'Plays the Lottie animations Devon generates (emoji_anim.py). ?src=<url> or ?data=<json>.'],
    ['GSAP demo', '/gsap', 'GSAP ' + VERSIONS.gsap, 'The web animation library for her pages — staggered hero, Replay button.'],
    ['3D viewer', '/three', 'three.js ' + VERSIONS.three, 'Scene-spec → three.js page. ?scene=<url-encoded JSON spec>. Orbit + auto-rotate.'],
    ['3D viewer (particles + tuning)', '/three', 'three.quarks ' + VERSIONS.threeQuarks + ' + Tweakpane ' + VERSIONS.tweakpane, 'Particle fountain + the live parameter pane on the same page.'],
  ];
  const items = rows.map(function (r) {
    return '<li><a href="' + r[1] + '">' + r[0] + '</a> — <b>' + r[2] + '</b><br><span>' + r[3] + '</span></li>';
  }).join('\n');
  return pageShell('Devon motion — rungs',
    '<style>body { background: #14141c; color: #eee; font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 20px; }'
    + 'a { color: #ffcc33; } li { margin: 16px 0; } span { color: #999; font-size: 14px; }'
    + 'code { background: #22222e; padding: 2px 6px; border-radius: 4px; }</style>',
    '<h1>Devon motion rungs</h1>\n<p>POST a scene spec to <code>/scene</code> to render a three.js page, or open one below.</p>\n<ul>\n' + items + '\n</ul>\n<p>Powered by peacethefirst</p>');
}

// -- server ---------------------------------------------------------------
function readBody(req) {
  return new Promise(function (resolve, reject) {
    const chunks = [];
    let size = 0;
    req.on('data', function (c) {
      size += c.length;
      if (size > MAX_BODY) { reject(new Error('body too large')); req.destroy(); return; }
      chunks.push(c);
    });
    req.on('end', function () { resolve(Buffer.concat(chunks).toString('utf-8')); });
    req.on('error', reject);
  });
}

const server = http.createServer(function (req, res) {
  const url = new URL(req.url || '/', 'http://127.0.0.1');
  const path = url.pathname;
  try {
    if (req.method === 'GET' && path === '/') { send(res, 200, indexPage()); return; }
    if (req.method === 'GET' && path === '/health') {
      sendJson(res, 200, { ok: true, service: 'devon-motion', port: PORT,
        rungs: ['lottie', 'gsap', 'three', 'three-quarks', 'tweakpane'] });
      return;
    }
    if (req.method === 'GET' && path === '/lottie') {
      const src = url.searchParams.get('src');
      const data = url.searchParams.get('data');
      if (data) {
        let parsed;
        try { parsed = JSON.parse(data); } catch { sendJson(res, 400, { ok: false, error: 'bad ?data= JSON' }); return; }
        send(res, 200, lottiePage(parsed, null)); return;
      }
      if (src) { send(res, 200, lottiePage(null, src)); return; }
      send(res, 200, lottiePage(DEMO_LOTTIE, null)); return;
    }
    if (req.method === 'GET' && path === '/gsap') { send(res, 200, gsapPage()); return; }
    if (req.method === 'GET' && path === '/three') {
      const raw = url.searchParams.get('scene');
      let spec = DEFAULT_SCENE;
      if (raw) {
        try { spec = JSON.parse(raw); } catch { sendJson(res, 400, { ok: false, error: 'bad ?scene= JSON' }); return; }
        const err = validateScene(spec);
        if (err) { sendJson(res, 400, { ok: false, error: err }); return; }
      }
      send(res, 200, threePage(spec)); return;
    }
    if (req.method === 'POST' && path === '/scene') {
      readBody(req).then(function (body) {
        let spec;
        try { spec = JSON.parse(body); } catch { sendJson(res, 400, { ok: false, error: 'bad JSON body' }); return; }
        const err = validateScene(spec);
        if (err) { sendJson(res, 400, { ok: false, error: err }); return; }
        send(res, 200, threePage(spec));
      }).catch(function (e) { sendJson(res, 400, { ok: false, error: String(e.message || e) }); });
      return;
    }
    sendJson(res, 404, { ok: false, error: 'unknown route' });
  } catch (e) {
    sendJson(res, 500, { ok: false, error: String(e.message || e) });
  }
});

server.listen(PORT, HOST, function () {
  console.log('Devon motion host on http://' + HOST + ':' + PORT + '/');
});
