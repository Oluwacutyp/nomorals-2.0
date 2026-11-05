# Studio parity — echelon ports vs the Craft incumbents

Per the integration plan (P-rules): each lifted system names its
incumbent surface here; a row moves to *shipped* only with its
acceptance evidence, and the rows that remain are the burn-down
queue. Sections land with their waves.

## Motion & compositing — EffectCraft → `motion_studio` (plan §2.3)

| Row | State | Evidence / note |
|---|---|---|
| 1.3a MotionProject document (D1 idiom: hash, seal, save/load, @uid segments) | shipped (Wave 4) | `motion_studio/project.py`; round-trip + snapshot-identity tests in `tests/test_echelon_motion_port.py` |
| 1.3a Property tree: path-addressed properties, frame-grid keyframes | shipped (Wave 4) | `MOTION_PROPERTIES` registry; unknown path/easing is a build error; key times FrameGrid-exact |
| 1.3a Compile totality: engine keyframes vs CPU-tier materialization, per-segment record | shipped (Wave 4) | compile record + `motion:<uid>` rungs merged into `render_document`'s honesty record |
| 1.3a Ledger gains the project hash | shipped (Wave 4) | `record_ledger({"kind": "motion_project", "project_hash": …})` |
| 1.3a `directed/motion_score.py` as the expression-to-track compiler / NL front end | standing, unchanged in role | its easing library (`EASE_FUNCS`/`ease_value`) is the property layer's expression vocabulary; pose compilation itself stays with `pose_rig` |
| Motion brush anchored by MaskRef (1.1d carrier) | shipped (Wave 4) | `brush/zoom` on image segments; compile record names the mask ladder rung |
| Typography consumes measured word timings (§2.4 label discipline) | shipped (Wave 4) | `typography.to_words` carries `timing_source` per word (`measured`/`estimated`/`authored`); lyric renders record the counts |
| 1.3b EffectCraft's dual JS surfaces (`crates/expr` value expressions, `crates/script` project scripting) | **considered, deliberately not ported** | Devon's split already exists and is named instead: value expressions = the keyframe/easing expression layer (1.2c + `directed/motion_score.py` easings, consumed by PropertyTrack sampling); project scripting = the brain driving ToolSpecs (P1). An embedded JS runtime would be a second, weaker agent surface — Devon's scripting surface is the agent herself. |
| 3D rung (`camera3d/*` property group) | **deferred — owner's choice** | paths are carried by MotionProject and refuse at compile with the reason; no stub renderer (plan §1.5e/§6.3: Blender-class external engine vs a Rust rung is a hardware-bearing decision) |
| Procedural paths on video segments (pan/rotation/blur over moving footage) | open remainder | video segments animate through the engine keyframe registry (`transform/zoom`, `opacity`) only; the CPU tier materializes stills/text. Named in the compile error, never silently dropped |
| Procedural engines beyond `visualizer` (explainer family: diagrams, equations-as-text, chart builds) | deferred to plan §1.5d | MotionProject documents are their planned home; the compile error names the deferral |
| Transition kinds wipe/circle through session directives | open remainder | engine session directives interpret cut/fade/dissolve/flash/dip/whip_pan; wipeleft/circleopen raise at compile, named |
