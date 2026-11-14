---
id: anatomy
title: Anatomy — explain any idea as a working isometric machine
summary: Build crafted, highly detailed, interactive isometric figures that explain a technical idea as a real physical machine — many small precise solid parts, four-tone shading, correct painter's order, a framed card with a live readout, calm eased motion. Ported from the Anatomy skill by wheresryan22 (MIT); the zero-dependency kit is vendored under this skill's directory.
topics: anatomy, isometric, figures, explainer, diagram, svg, machine, illustration, technical drawing
triggers: isometric figure, explain as a machine, explanatory diagram, technical illustration, isometric drawing, explainer figure, how it works diagram, anatomy figure
layer: media
version: 1
---

# Anatomy

This skill makes figures like a small instrument on a bench: a heavy
base on feet, rails with tick rulers, a carriage on bushings, knurled
knobs, coil springs, a dial gauge whose needle moves, screws as tiny
dots. The reader understands the idea because they recognise the
machine. Every number the figure shows comes from the same model
that moves its parts.

Use it whenever someone wants an isometric illustration, an
explanatory diagram or figure for docs, a blog post, a landing page
or an article, a premium technical drawing, an interactive explainer,
or an SVG that shows how a system, algorithm or component works —
even if they never say "isometric".

## Where it lands in Devon

The output is a standalone HTML page (or a lone `.svg`), which fits
Devon's artifact delivery exactly: build it, hand over the file, and
it opens from disk with nothing to install. The kit is vendored
under `nomorals/skills/library/anatomy/` — `kit/` (zero
dependencies, Node built-ins only), `references/` (craft, kit API,
motion, verification, WebGL, 3D, print, React, walkthrough,
publish), and `scripts/` (headless-Chrome capture, drive, line
checks, print, publish). It needs Node 20.10+ and Chrome or
Chromium for the checks. The eight complete upstream examples
(dial-indicator, desk-computer, arcade-cabinet, raptor-engine,
ripple-tank, test-rig, spot-plate, turning-dial) are NOT vendored —
they live upstream at https://github.com/wheresryan22/anatomy — open
the closest one there before starting; `dial-indicator` is the
smallest (a dial gauge on a stand, ~300 lines).

## The one idea that matters

**The object is the explanation.** Do not draw boxes, arrows and
labels. Invent a real, well-made physical thing whose working *is*
the concept. How earlier figures mapped:

| Concept | Object |
|---|---|
| One shared GPU context serving many canvases | A gantry plotter: one print head visiting 16 wells |
| A cache key that skips work when nothing changed | A pin-tumbler lock: when every pin matches, the plug turns and nothing is redrawn |
| Seven render targets with different bytes per texel | A chest of seven drawers, each as deep as its bytes |
| A separable blur (49 × 49 taps vs 49 + 49) | A bed-of-nails stamp beside two combs on rails |
| A spring that presses and leans | A test rig with springs, a probe and a dial gauge |
| A height profile | A profile (contour) gauge pressed against the edge |
| Light that depends on facing | A small photo studio: key and fill lamps around a turntable |

Two ways this fails, both seen in practice:

- **Too abstract.** Stacked plates, floating slabs or labelled boxes
  do not read as anything.
- **Too flat and bland.** One big slab with a few lines on it has no
  volume, no parts and nothing to discover.

The fix for both is the same: a recognisable machine, built from
many small, precise, purposeful parts.

**Every creation is highly detailed.** This holds for every figure,
however small the idea or the request: a quick explainer is still a
dense, real machine, never a few slabs. What scales with the size
(step 1) is how far you push it: a Figure gets the 4× sweep over its
subject and mechanism, a Hero or an Epic over the whole drawing.
- **Read the real thing.** Build it the way the real object is
  built: every screw, seam, bevel, bushing, label plate, cable with
  its connector, scale with its ticks, and the parts that hold the
  parts.
- **Two looks.** At 1× it reads as one clear machine. At 4× every
  region still has something true to find; a bare face or an empty
  corner is unfinished work, not calm.
- **Purposeful, never noise.** Detail comes from parts that belong
  to the mechanism. Each one sits on another part, ends on an edge,
  and fades rather than stops.
- **Check it last.** Before finishing, zoom to 4× (over the subject
  and mechanism for a Figure, the whole drawing for a Hero or an
  Epic) and add the real parts that are missing wherever a region
  is plain.

## Workflow

Work in this order. Write the concept and the parts list before any
geometry. Most of the quality is decided there.

1. **Size it before you start.** Name the size before any other
   work, and tell the user roughly how long it will take when the
   tier below calls for an estimate.
   - **Figure**: one mechanism, 40–80 parts, SVG with at most one
     WebGL layer. Name the size and build it without asking. Do not
     quote a time to the user, since a Figure often finishes in well
     under an hour. For the progress checks below, budget up to an
     hour.
   - **Hero**: a dense object, several WebGL layers or one 3D
     subject, 80–150 parts. About 2–4 hours. Say so, then build it
     unless the user asked for something quick.
   - **Epic**: a whole vehicle, building or plant with many moving
     systems, or the 3D mode on more than one subject. 6 hours or
     more. Do not start it. Give the estimate, propose a Hero
     version that keeps the core of the idea, and list what the full
     version would add. Build the Epic only when the user chooses
     it.
   - Whatever the size, have a working page that opens in the
     browser within the first hour, and deepen it from there. When a
     step runs past twice its share of the estimate, tell the user
     where it stands and what is left.
2. **Truth first.** Collect the real numbers and behaviour of the
   thing being explained: sizes, counts, thresholds, curves. The
   readout, motion and caption must all be true. If you can, write
   the model as a small pure module and drive everything from it.
3. **Invent the object.** Pick a machine, tool, instrument or piece
   of furniture whose mechanism maps one-to-one onto the concept.
   Say in one sentence what moves and what that motion means. Never
   copy a reference image: take its finish, not its object.
4. **Parts list.** Write at least 40 parts (40–80 for a Figure,
   80–150 for a Hero, more for an Epic), grouped: base (plate, feet,
   groove, corner screws, label plate, ruler strip), structure
   (posts, rails, frame bars, end stops), mechanism (carriage,
   bushings, springs, plunger, needle), controls (knurled knob,
   thumb screw, crown) and the subject itself, which is the one lit
   part. Each part sits on another part. `references/craft.md` has a
   vocabulary of parts with recipes.
5. **Plan the world.** Use world units of about 1 px of the real
   thing: x and y on the floor, z up. Lay out plans as rounded
   rectangles `{x, y, w, d, r}` and circles. Choose the camera:
   azimuth 45° by default, 35–60° to show the long side, elevation
   30° (higher, like 44°, to look into something open). Then fit
   with `fitProjection(points, width, height, { pad, azimuth })`.
6. **Build geometry once.** Precompute every static path in a
   geometry module with the kit: `slabOf`, `cylinder`, `extrude`,
   `planOutline`, `topTicks`, `knurl`, `coil`, `radialTicks` and so
   on. Round parts on any axis (bells, flanges, valves, actuators)
   come from `kit/lathe.mjs`, and pipes and cables from
   `kit/tube.mjs`. Moving parts are built once at a rest position
   and moved with `translateAlong`. Only parts whose shape changes
   are rebuilt per frame.
7. **Paint back to front, bottom to top.** Sort repeated parts with
   `depthOf`. Split anything that wraps another part into a back
   half and a front half: the springs around a rod, the bars of a
   frame around a carriage. A moving part is drawn in the slot where
   it physically sits. See "Painter's order" in
   `references/craft.md`. Once a figure has pipes, cables and round
   parts, stop tuning keys by eye: record each item's shape as you
   put it into `A.recorder(P, { order, spring })`, let
   `A.settle(R, P)` measure true depth wherever two items overlap
   and set the keys, and end the build with
   `A.auditOrExit(R, P)`. Route pipes before you draw them: real
   ports at both ends, legs along the structure, crossings on
   straights, collars gapping the pipe (`references/craft.md`,
   section 8). Record the true shape of what you draw: a stand-in
   shape that silences the audit is a bug, and the audit fails one.
   A rod that slides out of a gland is one rod from `G.rodOut`,
   recorded with `A.slide`.
8. **Frame it.** Put the figure in a card: `Fig n` at top left, a
   short title at top right, the interaction hint at bottom left and
   the live readout at bottom right. Legend keys and a true caption
   go under the card, and words stay out of the drawing apart from
   at most one tiny engraved label.
9. **Make it live.** Use calm time-based motion: exponential follow
   or real springs. An idle tour loops when nobody touches it. The
   figure responds to pointer and keyboard, stops when offscreen,
   and goes still under reduced motion. Write attributes on refs and
   do not re-render per frame. See `references/motion.md`.
10. **Verify like a critic.** The owner finds faults by zooming in,
    so you zoom in first. Read every PNG you make.
    - Capture desktop, phone (390 wide), hover and several moments
      of motion with `scripts/capture.mjs`; use `scripts/drive.mjs`
      for sequences, the apart pose and the `--no-webgl` fallback.
    - **Run the audit to zero.** With pipes, cables or round parts,
      `node build.mjs --audit` must end `audit passed`, every failing
      line at 0: clearance at rest, apart and in mid-flight,
      terminals, loose ends, folds, self-overlap, crowding (against
      slender solids too), stand-in shapes, support, wobble, saddles
      and depth order. Fix an interlock in the geometry, never with
      a key bias.
    - **Run the line check to zero.** The audit never sees line
      markup. `node scripts/lines.mjs page.html --out shots/lines.png`
      (and on the `-light` page, and in every resting pose) must
      print `line ends: 0`: every seam, rule, rib, tick and pointer
      ends on a stroke, a dot or an outline. Read the ringed PNG.
    - **4× close-ups of every junction in both themes** (build
      `--light` too), assembled and apart: every flange, tee, elbow,
      clamp, valve, bellows, band, saddle, ball joint and crossing,
      every curved body (no tone staircase), every place a texture
      ends (on an edge or fading), flat box faces with their seams,
      every textured hose at a chunk joint, every sliding rod at rest
      and at full stroke, and every window with text or shader
      content in it.
    - **Shaders:** a contact sheet slowed with `["slow", 0.05]`
      across every transition; nothing pops, no colour switch, edges
      read 0. In light, also log ink per frame through each
      transition (it must follow the light monotonically) and crop
      where the ink meets strokes at 8–16× (`references/verify.md`,
      "Light-theme shaders").
    - Put your PNG next to the examples and ask: is it instantly a
      real object, rich in small parts with volume, with every line
      ending on something? Is it highly detailed at 1× and still
      rewarding at 4×, with no bare face or empty corner? Iterate
      until yes. Checklist: `references/verify.md`.

## The kit

Vendored under `nomorals/skills/library/anatomy/kit/` (zero
dependencies, Node built-ins only):

| File | Use it for |
|---|---|
| `kit/iso-kit.ts` | The whole core in one file: camera, solids, a parts library, an SVG string renderer, card and page builders, and the CSS (`ISO_CSS`). Copy it into a TypeScript project. |
| `kit/iso-kit.mjs` | The same core compiled to plain ESM, for Node scripts. Import it directly. |
| `kit/iso.css` | `ISO_CSS` written out. Import it once and put `class="iso"` (plus `data-theme="light"` for light) on a wrapper or `<body>`. |
| `kit/react/draw.tsx` | Server-safe React components: `Solid`, `Line`, `Wire`, `Dots`, `FaceText`, `IsoFigure`, `Plate`. |
| `kit/react/live.tsx` | `"use client"` hooks: `useReadout`, `useLoop`, `useEased`, `useInView`, `useStillness`, plus `spring` and `follow`. |
| `kit/lathe.mjs` | Round parts along any axis: `frameAlong`, `lathe`, `solidOf`, `disc`, `ringBand`, `sphereOf`, `saddleOf`, `bandsOf`, visible arcs, ribs and bolt circles (`ribsOf`, `dotsOf`), `rodOut` for a rod sliding out of a gland, `fillet` for pipe routes. |
| `kit/tube.mjs` | `tubePieces` splits a 3D pipe or cable into depth-sortable chunks (with `breaks`, `gaps` under collars and clamps, elliptical end caps, `rings` for braid/corrugation/wrap); `tubeSvg` draws one; `crossingsOf`, `runsOf`. |
| `kit/audit.mjs` | Proves the geometry: `recorder`, `settle`, `auditOrExit` — clearance, terminals, loose ends, supports, folds, self-overlaps, crowding, stand-in shapes, wobble, seats, depth order. `A.slide` records a sliding rod. `references/kit.md`, section 12. |
| `kit/gl.mjs` | The WebGL layer for shaders inside the drawing: `GLSL_ISO`, `glCamera(P)`, `glLayer(...)`, envelopes `smooth`, `burst`, `settle`. For 3D figures, `GLSL_TURN` and `glslPose(name)`. |
| `kit/turn.mjs`, `kit/turn-build.mjs`, `kit/turn-audit.mjs` | The optional 3D mode: `turning(P, { recorder })` groups (turn, slide, hinge, free), live layers, convex parts with materials, `TURN.mount`, controller, `TURN.cover`, `TURN.poseUniforms`, `TURN_CSS`; `orbitOrExit`, the orbit audit. `references/3d.md`. |

Scripts (vendored under `library/anatomy/scripts/`): `capture.mjs`
(headless-Chrome screenshots), `drive.mjs` (scripted Chrome:
move, click, key, slow motion, contact sheets), `lines.mjs` (the
line-end check), `inline-kit.mjs` (`kitScript()`, `glScript()`,
`turnScript()` — kit source to inline into a standalone page),
`turn-check.mjs` / `turn-bench.mjs` (3D fidelity, order, pops,
lines, perf), `print.mjs` / `print-sheet.mjs` (printable sheets),
`publish.mjs` (community gallery, only when asked).

Pick the path that fits the project:

- **Standalone HTML, a static SVG, a blog post or any non-React
  site.** Write a Node build script like the upstream
  `examples/dial-indicator/build.mjs`. It imports `kit/iso-kit.mjs`,
  assembles strings with `solidSvg`, `lineSvg`, `dotsSvg`, `wireSvg`
  and `faceTextSvg`, wraps them with `figureSvg` and `plateHtml`,
  and writes the page with `pageHtml({ script: kitScript() + live })`,
  or `kitScript() + glScript() + live` when it has a shader. For a
  lone `.svg` file use `figureSvg({ standalone: "dark" | "light" })`,
  which embeds the styles.
- **React or Next.js.** Copy `kit/` into the project (for example
  `components/iso/`). Keep geometry in plain `.ts` modules, build
  the static art in a server component with the `draw.tsx`
  components, and pass it as element props to one `"use client"`
  live component that moves things. `references/react.md` explains
  the split and a trap with large pages.

## Craft rules (short form)

These are the rules that separate a crafted figure from a diagram.
`references/craft.md` gives the full reasoning and recipes.

- **Solids, not wireframes.** Every part is filled in paper colour
  and painted in order, so things in front hide things behind. No
  see-through edges and no dashed hidden lines.
- **Let the kit shade.** `extrude` returns four side tones by
  facing plus a lighter top. Keep it for anything with height, and
  use `flat` only for floors and inlays thinner than 1 unit.
- **Tone hierarchy.** Base and feet are `lo`, structure is `mid`
  and the subject is `hi`. Exactly one accent is `lit` (white in
  dark, black in light): the part the reader should watch. Grooves
  are `faint`, minor ticks `lo`, major ticks `mid` and the zero mark
  `hi`. Use colour only when colour is the subject, such as red,
  green and blue channels.
- **Small precise detail, never noise.** Bevels of 0.4–1.6, grooves
  inset 3–5 from an edge, screws as 0.4–0.55 dots with a ring at the
  corners of plates, knurls on knobs, ruler ticks every 4–5 units
  with a major every 5th, a label plate with three rule lines. Each
  detail belongs to a part.
- **Lines resolve.** Every line ends on something: a face, another
  line or a terminal dot. Connectors are `Wire`s whose terminal dots
  land on objects. No leader lines, balloons, dimension arrows or
  labels floating in the air.
- **No pipe or cable passes through anything.** Not through another
  pipe, cable or solid, at rest, apart or in any frame of the
  explode. Keep about 1.5 radii of clearance to everything but a
  line's own connection, run lines beside the structure rather than
  across the subject's face, and gap a pipe wherever a flange, band,
  bellows, clamp or valve wraps it. Parts meet flush, tangent or
  rim-on-surface, never sunk in.
- **Every crossing is drawn in true depth order.** Where two items
  overlap on screen, the nearer is drawn on top, proved by
  `A.settle` and the audit, not tuned by eye. Lines cross on
  straights, well apart in depth. A pair that cannot be ordered
  interlocks, and the fix is a cut or a tangent contact, never a
  bias.
- **Curved surfaces shade smoothly, never in a staircase.** Mark
  curved knots smooth and let `lathe` shade them from one monotone
  curve that also draws the outline; give any curved body larger
  than a fitting an analytic profile, cut long bodies with
  `G.bandsOf`, and use `G.sphereOf` for balls. Tone bands taper to
  nothing instead of ending on a straight cut, and seams go under
  the outline.
- **A texture ends on a real edge or fades.** Ribs, bolt circles,
  rule lines, seams and grooves never stop in the middle of a
  surface: they end on seams, fade out, or wrap corners. A texture
  on a pipe (braid, corrugation, wrap) follows its cross-section
  and runs through every chunk joint. Only rule lines that stand for
  printed text, a needle's tail and a pointer's tip may stop on a
  bare face, marked `free`.
- **Shaders never pop or cut off.** Every uniform rides an envelope
  with an attack and a decay, phase changes are crossfades driven by
  the physical quantity, every term reaches zero before any bound or
  canvas edge, and the effect lives in world space in the figure's
  own camera.
- **Things on faces fit them.** Text a machine displays fits its
  window at its widest string and is clipped to it; content seen
  through an opening is clipped by the opening as well as by its own
  plane, with the recess walls drawn.
- **Volume.** Give things real thickness, make them stand on other
  things, and put a soft halo under the base and under anything
  lifted.
- **Composition.** Centre the figure and let it fill about 55–65%
  of the card. Keep a consistent scale across a set of figures, and
  make sure it still reads at 390 px wide.

## WebGL inside the drawing

Use a shader only for what lines cannot draw: fire and exhaust,
water and caustics, real lamp light, glow, steam. The parts stay
SVG. Read `references/webgl.md` before writing any shader.

- **Three layers.** A back SVG holds what the effect covers, the
  canvas holds the effect (`pointer-events: none`), and a front SVG
  holds what covers it. Split a part, or the effect, when it is on
  both sides.
- **World space, same camera.** Build every shader point from
  `onFloor`, `onWallX` and `onWallY`, or march the pixel's ray.
  Size everything in world units times `uK`. Anything lying on a
  plane registers exactly with that plane's SVG lines.
- **Nothing pops.** Every uniform follows an envelope with an attack
  and a decay. Phase changes are crossfades where colour shifts
  continuously with the physical quantity, and a fade-out only gets
  darker.
- **No hard cut-offs.** Every term reaches zero before any `if`,
  bound or canvas edge, and before any cell edge of a cell noise.
- **Seen through an opening, masked twice.** Content behind a window
  or a recess is masked in its own plane and by the opening in the
  face plane, multiplied.
- **Blending.** Use `tonemap()` and dither. In dark, a canvas that
  only adds light uses premultiplied output with
  `mix-blend-mode: screen`; one that must also darken stays
  normal-blended.
- **Light theme ships, and its shaders are designed for paper.**
  Use normal premultiplied alpha, never screen. A glow is a
  saturated body with a pale core, a coloured rim and an alpha
  floor, never a darkened dark-theme colour. Ink only grows with
  light: fade the finished ink once, never one term at a time. Ink
  ends at each stroke's inner edge. Measure ink per frame.
- **Restraint.** Cap the device ratio at 1.5 and render only while
  the effect is active and onscreen. Reduced motion gets one still
  frame, and the SVG must still read without WebGL.

## 3D: figures that turn (beta)

3D is an option like WebGL. Use it when a part moving in depth is
the idea: a dial or turntable that turns, fingers that flex, a piece
that breaks off and falls. The rest of the figure stays static kit
art. Read `references/3d.md` first.

It is in beta: the engine is proved on its example and one test
figure, not yet on a production figure. Turn one subject at a time
and keep the clutter around it static. A small figure runs at 60
fps; a dense 380-part flexing hand measured about 20 fps at 1440
and 13–15 fps on a phone profile on a heavily loaded machine.

- **Live parts are the kit's own picture.** Each frame, convex kit
  primitives are re-projected and re-shaded by the kit's rules. A
  turn by θ is the kit drawing at azimuth `a − θ`.
- **Order comes from separating planes.** Same-group pairs get a
  plane fixed in the group, pairs on different links get one fixed
  in their common frame, and the rest are solved each frame. The
  audit proves there are no cycles at any angle or in any clip
  frame.
- **Contacts are flush, tangent or rim-on-flat.** Joints are coaxial
  knuckle discs on the child group that touch the parent's link, or
  the audit calls the whole child group loose. No live line spans two
  moving groups: a cable crosses a joint through a clip on each side,
  or reaches the turntable through a slip ring at the hub.
- **Live tubes bend in near-horizontal planes or inside fittings,
  and never point along the view.** A bend needs R ≥ 2r / sin²(e −
  θmax): 8r is the bound when flat, 21r on a ±12° hinge. Each route
  runs from mount surface to mount surface, with its ports in `gaps`.
- **Materials remap the ramp.** Gold, chrome, steel, gunmetal,
  rubber, brass and copper each have a light and a dark set; tones
  still come from facing.
- **Free groups break off without a jump.** Coaxial parts are fixed
  pictures with turning textures.
- **Shaders follow the motion.** Anchor them to a group with
  `controller.anchor` or `toGroup`, occlude a front canvas with
  `coverOf`, and keep world effects (smoke, falling sparks, oil on
  the bench) out of group space.
- **Verify with `--audit`** (`turn audit passed`) and
  `scripts/turn-check.mjs` (`--fidelity`, `--order`, `--pops`,
  `--lines`, `--perf`). Only a full run proves the figure, and it
  takes 15–20 minutes for 30 parts, so run it in the background.

## Printable sheets (A4)

When the user wants the figure on paper (a handout, a poster, a PDF,
an A4 or Letter sheet), print it from the finished figure instead of
drawing again. Write a `sheet.mjs` next to the build: the poses to
capture as `drive.mjs` actions, the parts as CSS selectors over the
live SVG, and the words, every one of them true to the model. Then
run `node scripts/print.mjs sheet.mjs --format all --png` and read
every page. Read `references/print.md` first.

- **poster** (the default): one page, the name set large, the
  question, big numbers, the figure with numbered pins and every
  part with one line.
- **drawing**: one page, a technical drawing with a parts list, a
  specification, an inset and a strip of steps with the working
  parts lit.
- **detailed**: 2–4 pages, scaled to the machine: an overview with
  pins and round close-ups; "How it works" with each step as a large
  lit still (six per page, the why in the last slot); and for complex
  figures an "Assemblies" page with up to four large close-ups of
  part groups. A simple figure stays at 2 pages; `maxPages` caps it.

Offer all three when the user has not picked one. When the page
shows a figure that has a PDF, put a link to it on the card with
`pdfButton(href)` from `kit/pdf-button.mjs` inside the plate body;
print first, then rebuild the page.

## References

Vendored under `nomorals/skills/library/anatomy/references/` — read
as needed:

- `craft.md`: the look in depth, including the parts vocabulary with
  code, tone tables, painter's order patterns, round parts, pipes and
  harnesses, exploded views, screens on faces, and the anti-patterns
  that got figures rejected. Read it before designing.
- `kit.md`: the API, with signatures and examples.
- `motion.md`: live behaviour (loops, springs, tours, pointer,
  keyboard, accessibility, reduced motion), plus envelopes for
  one-shot events, exploded views, depth re-sorting and arming, in
  vanilla JS and React.
- `react.md`: the React/Next.js server/client split, element props
  and the large-payload key-warning trap.
- `verify.md`: the `capture.mjs` and `drive.mjs` commands, the review
  checklist (including shaders) and common faults.
- `webgl.md`: shaders inside the drawing — when one earns its place,
  the layer stack, the camera contract, ray-marching in the iso
  camera, the full API with a minimal page, the craft rules with the
  bugs that taught them, the light theme, and how to verify them.
- `walkthrough.md`: how the test rig was designed, step by step,
  from concept to code.
- `3d.md`: the optional 3D mode.
- `print.md`: printable A4 or Letter sheets from a built figure.
- `publish.md`: sharing a finished figure in the community gallery
  with `scripts/publish.mjs`, only when the user asks; nothing is
  pushed for them.

## Provenance

Ported from the Anatomy skill (https://github.com/wheresryan22/anatomy,
MIT, Copyright (c) 2026 Ryan). The skill markdown is adapted into
Devon's format; the `kit/`, `references/` and `scripts/` directories
are vendored as-is from that repo (zero dependencies, Node built-ins
only). The eight upstream examples are not vendored (25 MB of images
and builds) — they live at the upstream repo. Where Devon is named
(artifact delivery, her web surfaces), that is Devon's own grounding.
