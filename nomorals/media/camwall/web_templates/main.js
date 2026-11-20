/* The cameras wall player — a Devon wall, played in the browser.
 *
 * The wall's manifest is inlined below as WALL (the exporter writes
 * it in): cameras with their snapshot URLs, rows, stats, and the
 * exported frames. Each tile refreshes from its camera's own
 * snapshot URL with a cache-busting timestamp — the clone's exact
 * mechanism — on one 800 ms interval, the clone's loop. A tile
 * whose live URL cannot be reached keeps its last exported frame
 * and is marked stale, like the clone's "unavailable" image.
 *
 * Click a tile for singleview (the rest pause, like the clone);
 * right-click toggles pause on that camera; the "wall" button (or
 * clicking the single view) goes back to the wall. The titlebar
 * shows location, last loaded time, queue % and fps, the clone's
 * figures. Works from file:// and from any static host, no fetch,
 * no server, no CDN, no dependencies.
 */
const WALL = {{MANIFEST}};

const wallEl = document.getElementById("wall");
const singleEl = document.getElementById("single");
const singleImgEl = document.getElementById("singleimg");
const singleStatsEl = document.getElementById("singlestats");
const statusEl = document.getElementById("status");
const wallBtn = document.getElementById("wallbtn");

const LOOP_MS = 800;
const tiles = new Map(); // id -> {cam, img, title, paused, loadedAt, loads, fails}
let singleId = null;

function stampUrl(url) {
  const t = Date.now().toString();
  return url + (url.indexOf("?") === -1 ? "?t=" : "&t=") + t;
}

function fmtTime(ms) {
  if (!ms) return "—";
  const d = new Date(ms * 1000);
  const p = (n) => (n < 10 ? "0" + n : "" + n);
  return d.getHours() + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
}

function titleText(tile) {
  const cam = tile.cam;
  const when = tile.loadedAt ? fmtTime(tile.loadedAt) : "never";
  return cam.location + " · loaded " + when + " · " + tile.loads +
    " frames · " + tile.fails + " fails" +
    (tile.paused ? " · PAUSED" : "") +
    (tile.stale ? " · STALE" : "");
}

function makeTile(cam) {
  const tile = document.createElement("div");
  tile.className = "cam";
  tile.dataset.id = cam.id;
  const img = document.createElement("img");
  img.alt = cam.name;
  if (cam.frame_file) img.src = cam.frame_file;
  const title = document.createElement("div");
  title.className = "camtitle";
  tile.appendChild(img);
  tile.appendChild(title);
  const state = {cam, img, title, paused: !!cam.paused,
                 loadedAt: cam.frame_at, loads: cam.counts.load,
                 fails: cam.counts.fail, stale: false};
  img.addEventListener("load", () => {
    if (!img.src) return;
    state.loadedAt = Date.now() / 1000;
    state.loads += 1;
    state.stale = false;
    title.textContent = titleText(state);
  });
  img.addEventListener("error", () => {
    state.fails += 1;
    state.stale = true;
    if (cam.frame_file && img.src.indexOf(cam.frame_file) === -1) {
      img.src = cam.frame_file; // fall back to the exported frame
    }
    title.textContent = titleText(state);
  });
  tile.addEventListener("click", () => singleview(cam.id));
  tile.addEventListener("contextmenu", (ev) => {
    ev.preventDefault();
    state.paused = !state.paused;
    title.textContent = titleText(state);
  });
  title.textContent = titleText(state);
  tiles.set(cam.id, state);
  return tile;
}

function refresh() {
  if (singleId !== null) return; // singleview refreshes its own image
  for (const state of tiles.values()) {
    if (state.paused) continue;
    state.img.src = stampUrl(state.cam.url);
  }
}

function singleview(id) {
  singleId = id;
  const state = tiles.get(id);
  wallEl.hidden = true;
  singleEl.hidden = false;
  singleImgEl.innerHTML = "";
  const img = document.createElement("img");
  img.alt = state.cam.name;
  img.src = state.cam.frame_file || stampUrl(state.cam.url);
  singleImgEl.appendChild(img);
  singleStatsEl.textContent = state.cam.name + " — " +
    state.cam.location + " · " + titleText(state);
  for (const [otherId, other] of tiles) {
    if (otherId !== id) other.paused = true;
  }
  statusEl.textContent = "single: " + state.cam.name +
    " (click to return to the wall)";
}

function multiview() {
  singleId = null;
  singleEl.hidden = true;
  wallEl.hidden = false;
  for (const state of tiles.values()) state.paused = false;
  statusEl.textContent = WALL.cameras.length + " cameras · refreshing";
}

function singleRefresh() {
  if (singleId === null) return;
  const state = tiles.get(singleId);
  if (!state || state.paused) return;
  const img = singleImgEl.querySelector("img");
  if (img) img.src = stampUrl(state.cam.url);
  singleStatsEl.textContent = state.cam.name + " — " +
    state.cam.location + " · " + titleText(state);
}

for (const cam of WALL.cameras) wallEl.appendChild(makeTile(cam));
wallEl.style.setProperty("--rows", WALL.rows);
statusEl.textContent = WALL.cameras.length + " cameras · refreshing";
wallBtn.addEventListener("click", multiview);
singleEl.addEventListener("click", multiview);
setInterval(() => { refresh(); singleRefresh(); }, LOOP_MS);
