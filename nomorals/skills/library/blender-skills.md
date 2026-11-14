---
id: blender-skills
title: Blender skills — 94 pro workflows for live Blender work
summary: Devon's Blender router: 94 pro Blender skills (modeling, animation, rigging, materials, lighting, rendering, styles, genres, export) vendored from the arjun988 blender-skills pack (MIT, v1.3.0), run against a LIVE Blender session through the MCP for Blender server — the add-on is the bridge, and the live-session organ is nomorals/media/blender_live.py.
topics: blender, 3d, modeling, animation, rigging, materials, lighting, rendering, sculpting, geometry nodes, export, game art
triggers: blender, 3d model, mesh, rig, animate, render, sculpt, geometry nodes, low poly, hard surface, archviz, game asset, export fbx, export glb
layer: media
version: 1
---

# Blender skills

This skill is Devon's Blender router. It applies whenever she works in
Blender — modeling, animation, rigging, materials, lighting, rendering,
or exporting a 3D asset. The 94 skills vendored under this skill's
directory are the pro workflows; this file is the router that picks
the right one and the live session that runs it.

## The bridge — read this first

The skills do NOT control Blender on their own. They run against a
LIVE Blender session through the MCP for Blender server
(ahujasid/mcp-for-blender, MIT) over Devon's MCP client (server
`blender` — `uvx mcp-for-blender`). The operator's step, once:

1. `uvx mcp-for-blender install-addon` — installs the add-on into Blender.
2. In Blender, enable the add-on and start its server.
3. Devon connects with `nomorals/media/blender_live.py`
   (`BlenderLiveSession.connect()`).

Devon NEVER starts Blender herself. When no live Blender is reachable
the organ says so plainly (`ADDON_SETUP`) instead of pretending. The
batch-render rung (`nomorals/media/edit_engine/render3d_blender.py`,
headless bpy scripts rendered to frames) is a SEPARATE rung alongside
this one — it renders a scene she is given; this one steers a session
she can see and iterate on.

## The live session

`BlenderLiveSession` (in `nomorals/media/blender_live.py`) is the
organ. Four tools, all through the MCP server:

- `run_code(code)` — run Blender Python in the live session
  (`execute_blender_code`). Work in small steps, print what you need.
- `screenshot(mode="angles")` — see the scene (`look`): `viewport`,
  `camera`, `angles` (an auto-framed multi-view sheet — the default,
  one call from several sides), or `frames` (an animation strip);
  shading `solid`, `material`, `rendered`, `wireframe`, `xray`.
- `scene_summary(fields, query, root)` — the compact scene facts
  (`get_scene_info`): one line per object, the fields you pick.
- `status()` — the connected Blender (`get_addon_status`): version,
  add-on version, what libraries and generators are switched on.

`iterate(goal, code_fn)` is the loop: run → screenshot → score →
adjust, bounded (3 passes by default, the MCP client's per-call
timeout on every call). The score is the scene summary checked
against the goal's named things; for animation goals the brain's
`suggest` feeds the motion-score rung
(`nomorals/media/directed/motion_score.py`) as the target. The loop
stops early when the scene matches, and reports when it does not —
it never runs forever.

## The 94 skills

Vendored as-is under `blender-skills/skills/` (one directory per
skill, `SKILL.md` + its `references/`), plus the pack's shared
`references/`. They are reference material, not Devon's voice — do not
summarize them down; the detail is the product. The `.cursor/skills/`
copy upstream is byte-identical to `.claude/skills/`; only one copy is
vendored. Version pin: **blender-skills v1.3.0** (94 skills, 175
references) — re-check the pin on update.

The skills fall into rough groups; pick by the task, not by the list:

- **Core craft:** `blender-modeler`, `blender-director`,
  `hard-surface`, `procedural-modeling`, `geometry-nodes`,
  `asset-optimization`, `export-pipeline`, `lod-pipeline`, `qa-review`.
- **Characters:** `character-artist`, `character-archetypes`,
  `creature-artist`, `rigging`, `animation`, `cloth-sim`,
  `hair-groom`, `collision-proxy`.
- **Look:** `materials`, `lighting`, `lookdev`, `rendering`,
  `compositing`, `camera-cinematography`, `uv-workflow`,
  `texture-painting`.
- **Worlds:** `environment-artist`, `archviz`, `biome-worlds`,
  `fantasy-worlds`, `historical-worlds`, `apocalypse-worlds`,
  `liminal-space-style`, `isometric-style`.
- **Styles (one skill each):** `anime-style`, `cartoon-style`,
  `pixel-art-style`, `lowpoly-style`, `realistic-style`,
  `painterly-style`, `hand-painted-style`, `hd-2d-style`,
  `psx-horror-style`, `chibi-style`, `manga-style`, `noir-style`,
  `neon-retrofuturism`, `frutiger-aero-style`, and the mood skills
  (`dark-gritty-mood`, `cozy-wholesome-mood`, `dream-surreal-mood`…).
- **Genres:** `genre-rpg`, `genre-soulslike`, `genre-metroidvania-roguelike`,
  `genre-survival`, `genre-stealth`, `genre-action-combat`,
  `genre-racing-sports`, `genre-open-world-sandbox`, and the rest.
- **Engines:** `godot-export`, `unreal-export` (or the engine's
  own export skill when the target is named).

## Honest gaps

- The MCP server's `generate_3d` / `generate_gaussian_world` /
  `search_assets` / `import_asset` tools need the operator's own
  generator keys or Premium — the organ does not call them, and the
  skills that assume them say so in their own text.
- `generate_gaussian_world` needs Blender 5.3+; Devon has Blender
  4.2.3. Do not promise it.
- TRELLIS.2 (image-to-3D) and UniRig (auto-rigging) are separate
  provision rungs, not part of this pack — a mesh the skills need
  comes from the operator, from TRELLIS.2, or from the asset tools.
