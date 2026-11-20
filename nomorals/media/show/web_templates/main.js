/* The hosted demo player — a Devon show, played in the browser.
 *
 * The show's manifest is inlined below as SHOW (the exporter writes
 * it in): frames, fps, captions and camera shots. Frames load as
 * images from data/ — that works from file:// and from any static
 * host, no fetch, no server, no CDN, no dependencies.
 *
 * Shots are crops of the frame, the same fractions as Devon's
 * vtuber camera (nomorals/media/vtuber/camera.py SHOTS): wide is
 * the full frame, medium the upper body, close head and shoulders,
 * face the face.
 *
 * ?autoplay=1 starts without a click (for embeds and headless
 * proofs); otherwise the Start button is the gesture, like the
 * demo this pattern comes from.
 */
const SHOW = {{MANIFEST}};

const SHOTS = {
  wide: [0.0, 0.0, 1.0, 1.0],
  medium: [0.15, 0.05, 0.85, 1.0],
  close: [0.25, 0.0, 0.75, 0.55],
  face: [0.35, 0.0, 0.65, 0.35],
};

const canvas = document.getElementById("stage");
const ctx = canvas.getContext("2d");
const captionEl = document.getElementById("caption");
const statusEl = document.getElementById("status");
const barEl = document.getElementById("bar");
const startBtn = document.getElementById("start");
const pauseBtn = document.getElementById("pause");
const replayBtn = document.getElementById("replay");

const images = [];
let loaded = 0;
let failed = 0;
let playing = false;
let paused = false;
let startedAt = 0;
let pausedAt = 0;
let timer = 0;
let lastFrame = -1;

/* Playback runs on a plain interval at the show's own frame
 * rate — not requestAnimationFrame. The frames are
 * pre-rendered images at 15 fps, so an interval is exact,
 * and it is the clock a headless browser actually advances:
 * rAF only fires when the compositor issues a new frame,
 * which a still page in a headless shell does not do, so a
 * rAF-driven player stalls at frame 1 there while playing
 * fine in a real browser. The interval plays in both.
 */
const frameMs = 1000 / SHOW.fps;

function overlayAt(list, t) {
  for (let i = list.length - 1; i >= 0; i--) {
    /* The last entry runs to the end of the clip, so at the
     * exact end instant it still holds — otherwise the final
     * frame would show no caption and the wide shot. */
    if (t >= list[i].from_s &&
        (t < list[i].to_s || i === list.length - 1)) return list[i];
  }
  return null;
}

function draw(frameIndex, t) {
  const img = images[frameIndex];
  if (!img) return;
  const shot = overlayAt(SHOW.shots, t);
  const crop = SHOTS[(shot && shot.shot) || "wide"] || SHOTS.wide;
  const sx = Math.floor(crop[0] * img.naturalWidth);
  const sy = Math.floor(crop[1] * img.naturalHeight);
  const sw = Math.max(1, Math.floor(crop[2] * img.naturalWidth) - sx);
  const sh = Math.max(1, Math.floor(crop[3] * img.naturalHeight) - sy);
  ctx.fillStyle = "#0a0a0e";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
  const caption = overlayAt(SHOW.captions, t);
  captionEl.textContent = caption ? caption.text : "";
  statusEl.textContent =
    "Frame " + (frameIndex + 1) + " / " + SHOW.frames.length +
    " · " + t.toFixed(1) + "s / " + SHOW.duration_s.toFixed(1) + "s" +
    " · " + ((shot && shot.shot) || "wide");
  barEl.style.width =
    ((frameIndex + 1) / SHOW.frames.length * 100).toFixed(1) + "%";
}

function elapsedS() {
  return pausedAt + (playing ? (performance.now() - startedAt) / 1000 : 0);
}

function tick() {
  if (!playing) return;
  const t = elapsedS();
  if (t >= SHOW.duration_s) {
    draw(SHOW.frames.length - 1, SHOW.duration_s);
    stop("done");
    return;
  }
  const frameIndex = Math.min(
    SHOW.frames.length - 1, Math.floor(t * SHOW.fps));
  if (frameIndex !== lastFrame) {
    draw(frameIndex, t);
    lastFrame = frameIndex;
  }
}

function stop(reason) {
  playing = false;
  clearInterval(timer);
  timer = 0;
  pauseBtn.disabled = true;
  startBtn.disabled = true;
  if (reason === "done") {
    statusEl.textContent += " — done";
  }
}

function start() {
  if (playing) return;
  playing = true;
  paused = false;
  startedAt = performance.now();
  lastFrame = -1;
  startBtn.disabled = true;
  pauseBtn.disabled = false;
  pauseBtn.textContent = "Pause";
  replayBtn.disabled = false;
  timer = setInterval(tick, frameMs);
  tick();
}

function pause() {
  if (!playing) return;
  pausedAt = elapsedS();
  playing = false;
  paused = true;
  clearInterval(timer);
  timer = 0;
  startBtn.disabled = false;
  startBtn.textContent = "Resume";
  pauseBtn.disabled = true;
}

function replay() {
  clearInterval(timer);
  timer = 0;
  playing = false;
  paused = false;
  pausedAt = 0;
  lastFrame = -1;
  startBtn.textContent = "Start the show";
  start();
}

startBtn.addEventListener("click", start);
pauseBtn.addEventListener("click", pause);
replayBtn.addEventListener("click", replay);

canvas.width = SHOW.width;
canvas.height = SHOW.height;

SHOW.frames.forEach(function (name, i) {
  const img = new Image();
  img.onload = function () {
    loaded++;
    if (loaded + failed === SHOW.frames.length) ready();
  };
  img.onerror = function () {
    failed++;
    if (loaded + failed === SHOW.frames.length) ready();
  };
  img.src = "data/" + name;
  images[i] = img;
});

function ready() {
  if (failed > 0) {
    statusEl.textContent =
      failed + " of " + SHOW.frames.length +
      " frames failed to load — the show cannot play";
    return;
  }
  draw(0, 0);
  lastFrame = 0;
  statusEl.textContent =
    "Ready — " + SHOW.frames.length + " frames at " + SHOW.fps +
    " fps, " + SHOW.duration_s.toFixed(1) + "s";
  startBtn.disabled = false;
  if (new URLSearchParams(window.location.search).get("autoplay") === "1") {
    start();
  }
}
