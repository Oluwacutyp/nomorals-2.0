# Games parity ledger — Echelon Wave 9 (games & graphics patterns)

What Wave 9 took from the graphics/mocap donors, where it already
lives in Devon, and what was deliberately not taken. **Patterns only:
no graphics engine was ported and no code changed for this wave.**
Base: `8de448e` (Devon main). Donor evidence: echelon-mining
REPORT §2.9 and notes/05 §§11–12, 19–20 (the plan's "notes/02 §§7–8"
citation is stale — notes/02 §§7–8 are figcraft/soundcraft; the mocap
and point-cloud sections live in notes/05). All donor repos are
unlicensed, so nothing was copied — shapes only.

## The donors (primary records)

* **bevy-mocap** (notes/05 §20) — `CSVAnimationLoader`, a Bevy
  `AssetLoader` driven by a data-only `ClipInfo` JSON descriptor
  (csv path, `root_name`, optional fps, optional channel mapping).
  It parses Unreal LiveLink-format CSV (ARKit blendshape channels)
  and emits an `AnimationClip`: morph-target curves from name-mapped
  channels, bone curves from mapping entries of the form
  `{path, axis, scale?, offset?} → channel`. Errors are a hand-rolled
  typed enum (`CSVAnimationLoaderError`, 7 variants). The insight:
  one engine's capture format treated as an interchange format
  another engine consumes, via a descriptor, not code.
* **storyteller-bevy** (notes/05 §19) — Bevy/WASM studio with a
  `studio-bvh/` sibling for BVH motion-capture support: captured
  motion as a loadable asset, again.
* **cloudcam** (notes/05 §12) — Azure Kinect → CPU point cloud
  (calibration XY tables in `pointcloud/`) → a hand-rolled binary
  ZeroMQ protocol (command words, ≤3000 points per packet).
* **CloudCamPlugin** (notes/05 §11) — the UE consumer: a data-only
  `.uplugin` descriptor declaring one Runtime module;
  `ZeroMqPollingThread` polls off the game thread and a private
  `RingBuffer` decouples the poll thread from the actor, which
  converts buffered points into position/color textures for a
  Niagara system. UE plugin discipline across the set
  (EventCenterPlugin, ActorSpawningPlugin, CloudCamPlugin,
  HyperJail — REPORT §2.9): data-only descriptors, one module per
  concern, streaming work off the main thread behind a bounded
  buffer.

## What was taken — and where it already lives in Devon

1. **Data-only descriptors declare content; one solver/engine
   consumes them.** Already Devon's games idiom, named not rebuilt:
   * `games/declarative.py` — frozen dataclasses `AffixDesc` /
     `SkillDesc` / `EnemyDesc`; `AFFIX_CATALOG` (12 affixes as data)
     and `ROLE_POOL` (8 enemy roles) are plain data, and one
     constraint solver (`solve()`) composes legal combinations.
     The module's own docstring states the discipline: content is
     data, "a one-line data entry, never a code change".
   * `games/enemies.py` — encounter tuning as data tables
     (`GRADE_POOLS`, `DIFFICULTY_MODS`, `POTIONS`, `SET_CHANCE`, …).
   * `games/gear.py`, `games/skills.py` — item/skill descriptors as
     frozen dataclasses, behaviour in shared engines.
   * `games/games/base.py` — the `MultiGame` contract: game code is
     pure, all state lives in `room.state` (a plain JSON-able dict),
     and the engine owns routing/timers/persistence without ever
     parsing game rules. Descriptor and engine never mix.
   * One module per concern across the package (`economy.py`,
     `matchmaking.py`, `seasons.py`, `gear.py`, `progression.py`,
     `tournaments.py`, …) — the UE plugin shape at package scale.
2. **Motion as a loadable data asset (the asset-loader shape).**
   The consumer half of the bevy-mocap pattern lives in
   `media/directed/`:
   * `media/directed/pose_rig.py` — `PoseTrack` (dataclass): a
     keypoint trajectory (COCO-18 body + 21-point hands, DWPose
     layout, normalized coordinates) as a data asset. It renders to
     a skeleton pose video — the exact input MimicMotion /
     AnimateAnyone consume — or drives the CPU mesh-warp animator.
   * `media/directed/motion_score.py` — natural language → a JSON
     motion score → compiled onto the rig as a `PoseTrack`: a
     descriptor in, an asset out, the loader/compiler split the
     donor's `CSVAnimationLoader` + `ClipInfo` exemplifies.
   * `media/directed/animator.py`, `lipsync.py`, `camera.py` are
     the consumers of those assets, each its own module.
   **Honest limit:** the capture-ingestion half did **not** land.
   No LiveLink/ARKit CSV or BVH mocap importer exists anywhere in
   `nomorals/` (grep-verified at this base). Devon's pose tracks
   are authored parametrically or choreographed by the brain —
   never imported from a captured performance. This note does not
   claim mocap ingestion; it records the asset side only.
3. **Tracking output as a data asset.** `media/scene_intel/`:
   `track.py` emits `Detection` dataclasses (frame time, bbox,
   track id, OSNet re-ID embedding) from YOLOv8 + ByteTrack-style
   association; `segment.py` emits scene segments as dataclasses;
   `characters.py` and `score.py` consume them. The cloudcam
   pipeline shape — sensor-side processing reduced to a typed
   stream of records a separate consumer renders/scores — with
   film files, not a depth camera, as the source.
4. **Streaming work decoupled behind a bounded, inspectable
   buffer.** Devon's form is not a ring buffer; it is
   `media/edit_engine/render_queue.py`'s `RenderQueue`: a
   file-backed queue workers claim segments from (`claim_ttl_s`,
   300s default), with stale-claim stealing when a worker dies —
   producer and renderer separated by a durable buffer, the same
   decoupling CloudCamPlugin's poll-thread + ring buffer gives a
   live stream, adapted to batch rendering.

## What was deliberately NOT taken

* **No graphics engine (Bevy or Unreal).** Devon's games are chat
  games by design (plan verdict V12): `games/` is pure text/emoji —
  grep-verified at this base, zero `pygame`/PIL imports anywhere in
  the package; rendering is text formatting. There is no engine to
  port into, and bolting one on would serve no surface Devon has.
* **No point-cloud streaming, Kinect capture, or ZeroMQ binary
  protocol.** Devon has no depth-camera surface; `scene_intel`
  works on finished film files. The pipeline *shape* was taken
  (item 3 above); the transport and hardware were not.
* **No Niagara / texture-upload render path** — a consequence of
  the engine decision above, not an oversight.
* **HyperJail** (VT-x/Blueprint VTuber rig) — a study artifact per
  REPORT §2.9, not an acquisition.
* **No donor code at all.** Every donor here is unlicensed
  (notes/05 license ledger); Wave 9 is patterns-only by the plan,
  so this ledger is the entire deliverable.

## The visual-ambition route (plan §1.5e pointer)

The arena's visual ambitions do not route through a game engine.
They route through Wave 1's media stack: §1.5d explainers shipped;
§1.5e — the 3D rung — remains the **owner's open choice** (the
camera3d paths are carried in the media stack and refuse by name
until he picks). Nothing in this wave pre-empts that decision.

## Status

*verified in tests* — n/a (no code changed). *Verified by survey* —
every Devon file named above was read/grepped at base `8de448e`
for this ledger. *Not yet verified* — mocap capture ingestion
(LiveLink/BVH) does not exist and is not scheduled; it enters the
queue only if a surface ever needs captured-performance input.
