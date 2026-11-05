# Studio parity — echelon ports vs the Craft incumbents

Per the integration plan (P-rules): each lifted system names its
incumbent surface here; a row moves to *shipped* only with its
acceptance evidence, and the rows that remain are the burn-down
queue. Sections land with their waves.

## Effects — EffectCraft → `edit_engine/effects` (plan §1.2b/§1.2c/§1.1c(d)/§1.1d)

| Row | State | Evidence / note |
|---|---|---|
| Property registry: every effect param declared at `effects/<name>/<param>` (kind, default, range, keyframable family) in `EFFECT_PROPERTIES`; `KEYFRAMABLE_PARAMS` derives from it | shipped (Wave 5) | `effects.py`; registry totality + enforcement tests in `tests/test_echelon_effects_port.py`; served as data by the `studio_effect_properties` tool |
| Param validation at the chain funnel: undeclared param / kind / range / enum violations raise naming the property path | shipped (Wave 5) | `validate_effect_params` runs inside `_effect_filter`, the one funnel every builder call passes through |
| Keyframe binding through the registry — `sampled` family: declared params without an expression rung (blur radius, grain strength, rgb shift, ken-burns pan, beat-pulse, shake, flash level) discretize on the FrameGrid into constant-value windowed sub-effects of the same builder | shipped (Wave 5) | `keyframes.classify_track` + `discretize(fps=…)`; two-paths-one-result oracle vs a caller-built windowed assembly in the wave tests; a same-named static effect becomes the sampled base (two or more = ambiguity error) |
| Regional effects anchored on MaskRef: an effect's `mask` param (the 1.1d carrier, same shape as ImageDocument ops and the motion brush) resolves through the imggen ladder against the clip's window-start frame; application is alphamerge(mask × effect-alpha) + overlay over the untouched base, inside the single render graph | shipped (Wave 5) | `_prepare_effects`/`_chain_masked` in `render.py`; inside/outside region oracles + rung records (`box+feather`, `file`) in the wave tests. **Not maskedmerge**: on this ffmpeg maskedmerge blends per-plane against the mask's constant chroma and washes the unmasked region with the effected stream — proven on red/green fixtures before choosing the composite |
| Grades as Document state: `look_lock` (grade name or CDL dict, grain) becomes `color_grade`/`grain` effects on every base-track clip, consuming the canonical chain (`color_grade_chain` / studio_color CDL bake) in the single render graph — no separate re-encode | shipped (Wave 5) | `render._look_lock_effects`; applied vs carried split in the derivative manifest (palette/lens/name steer generation+QC, not the compositor — reported as carried, never as applied); overlay tracks carry their own effects — the look grades the program track |
| Per-effect honesty: every application recorded in `effects_applied` (producing rung, window, source — clip / look_lock / keyframe:<param> — and the mask rung + anchor note when regional); an `effects` stage entry joins `rungs` | shipped (Wave 5) | assemble/overlay/apply_effect all record through `_fx_record`; wave tests pin builder/structural/keyframe-sampled kinds |
| 3D effects (`camera3d`) | **deferred — owner's choice** | named in `DEFERRED_EFFECTS`: builders, keyframe targets and `apply_effect` refuse with the §1.5e/§6.3 reason; no stub renderer |
| Mask tracking across frames | open remainder | a regional mask anchors to the clip's window-start frame and holds for the clip; it does not follow motion. The anchor is stated in every masked record |
| Animated CDL components (grade curves) | open remainder | `color_grade` params are cuts, not curves — neither `grade` nor `cdl` is keyframable, and the registry says why. Per-component CDL keyframe paths are a future registry extension, not a silent gap |
| Regional `velocity_ramp` | refused, named | velocity is structural (chain rewrite); a masked ramp raises naming the reason instead of dropping the mask |

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
