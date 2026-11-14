---
id: remotion-video
title: Remotion video — the video-as-code idiom, engine deferred
summary: The Remotion way of making video (compositions in React, Studio preview first, render only when asked, captions, maps, multimedia, SaaS rendering) as Devon's own routing skill — written from the pattern, not copied, because the upstream skills repo carries no licence file. The Remotion engine itself is NOT installed in Devon and the rung is deferred; her own motion engine is the working answer today.
topics: remotion, video, react video, studio, render, captions, maps, saas
triggers: remotion, make a video with remotion, remotion studio, render video, video as code, remotion composition
layer: media
version: 1
---

# Remotion video

Remotion is "video as code": compositions written in React, a live
Studio preview, and programmatic rendering to MP4. This skill is
Devon's router for that idiom — when to use it, what the flow is,
and where it stops.

## Read this first: the engine is deferred

**Devon does not have Remotion.** There is no `remotion` string
anywhere in her repo, no React stack, no Remotion project, and no
`remotion` package installed. The engine rung is **deferred** — it
needs Node + React + a licence check, and it duplicates her own
motion engine. When someone asks Devon to make a video, her working
answer today is her own engine: `nomorals/media/motion_studio/`
(studio, montage, kenburns, typography, visualizer, explainers,
grading, project) + `nomorals/media/edit_engine/` (render, keyframes,
transitions, timeline) + ffmpeg + Blender. Build the video there
unless the user explicitly wants a Remotion project — in which case
say plainly that the Remotion rung is not built yet, and what building
it would take, before starting.

**Licence note.** Remotion the engine is free for individuals; a
company licence is required in some cases — check
remotion.dev/docs/license before any commercial use. The upstream
skills repo (https://github.com/remotion-dev/skills) carries **no
LICENSE file**, so no file from it is copied here: this skill is
written in Devon's own words from the pattern the skills describe.
If the licence grant for the upstream skills is ever confirmed, the
skill texts can be adopted verbatim then; until then, this file is
the adoption.

## Preserve user changes

Users may make edits in the code outside of the conversation. If you
detect a surprising change made in the meanwhile, do not overwrite
it — assume it was intentional, or ask for confirmation.

## The flow, in order

### Creating a video

When the user asks to make, create or build a new video or
composition, the work starts with a new Remotion project (or the
existing one, if there is one). A project is a React app with
compositions: each composition names its frames, frame rate, width
and height, and the React markup that draws each frame. Keep
compositions small and named for what they are; one scene per
composition, composed together for the full video.

### Writing the markup

Remotion markup is React that draws one frame at a time from the
current frame number — no timers, no `requestAnimationFrame`, no
state that changes outside the frame. Derive everything (positions,
opacity, scale) from the frame. Text, images, audio, video, GIFs,
fonts and transitions all have their own idioms; look them up in the
Remotion docs rather than guessing. Video matting, background
removal and placing content behind a subject are markup work, not
post-processing.

### Open the preview

Do NOT render by default. When the user asks to "make" a video, they
want to see the interactive preview, not a file. As soon as the
project can run, start Studio and open the preview in the browser
before building or editing the composition. Keep it open while you
work so the user can watch progress and steer.

- `npx remotion studio` opens the browser automatically.
- With an in-app browser, or to get the URL: `npx remotion studio
  --no-open` starts the server and prints its URL — open that exact
  URL and verify Studio loads, then verify the video preview loads
  for the composition. A specific composition lives at
  `/[composition-id]`, for example
  `http://localhost:3000/MapAnimation`.

### Render the video

Only render if the user is very explicit — "Render the video",
"Export", "Give me the MP4". The preview also has a rendering
interface, which is friendlier than the command line; consider using
it instead. From the command line: `npx remotion render`. Rendering
options (codec, quality, scale, frame ranges, transparent video) are
chosen per render — look them up rather than assuming defaults.

### Captions

Word-level captions come from transcription with timestamps
(Whisper-class tooling) mapped onto the composition's frames. Keep
caption timing frame-accurate: a caption that lags the audio by even
a few frames reads as broken.

### Maps

Static maps, animated routes and markers, geographic explainers,
Mapbox / MapLibre / MapTiler tiles, GeoJSON, and 3D geographic
flyovers are map compositions. Tiles come from the provider the
project is configured for; never hard-code a provider key into a
composition.

### Multimedia

Trimming, cropping, metadata and format work on video in the browser
is multimedia work. It runs on the machine, not in the chat — probe
the file first, then cut.

### Interactivity

New markup is editable in Studio by default. When writing or
restructuring markup, keep it editable: parameters the user can tune
in Studio beat hard-coded values.

### SaaS, automation and applications

Remotion-powered SaaS apps (a `<Player>` embed, rendering on Lambda,
Vercel, Cloudflare, via Express, or client-side rendering) and the
SaaS templates are a separate body of knowledge from making one
video. Reach for it only when the ask is an app or an automation,
not a single render.

### Looking up APIs and documentation

Remotion has ~1,200 pages of docs. Look up the current page for the
API in question before writing code against it — do not write
Remotion code from memory.

### Upgrading

Upgrading Remotion, its related packages, compatible Mediabunny
packages and installed Remotion skills follows the upgrade guide for
the version in the project. Read the installed version first; the
guide is a walk from one version to the next, so the starting point
decides which sections apply.

## What this skill does NOT do

- It does not install Remotion, React, or any package. That is the
  deferred engine rung.
- It does not render video on Devon's own engine — that is the
  `motion-design` skill and the motion engine's own docs.
- It does not promise a Remotion render on a machine without the
  rung. If the user wants one, the honest answer is the rung first,
  then the video.

## Provenance

Written in Devon's own words from the pattern of the Remotion agent
skills (https://github.com/remotion-dev/skills). No upstream file is
copied: the upstream skills repo has no LICENSE file, so verbatim
adoption is unconfirmed. The Remotion engine licence (free for
individuals, company licence in some cases) is separate from the
skills' licence status. Where Devon is named (her own motion engine
as the working answer, the deferred rung), that is Devon's own
grounding.
