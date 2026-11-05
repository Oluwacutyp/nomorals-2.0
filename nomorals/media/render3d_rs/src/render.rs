//! The rasterizer: Scene3D frame state in, linear-RGB pixels out.
//!
//! Pipeline per frame: tessellate each visible mesh node → world
//! transform (positions by the world matrix, normals by its
//! inverse-transpose) → view transform → Sutherland–Hodgman clip
//! against the near/far planes in view space → project with the
//! state's own projection matrix → edge-function rasterization
//! with perspective-correct interpolation → per-fragment shading.
//!
//! Shading (all linear, f64; sRGB encoding happens only at the
//! PNG boundary):
//! * `flat` — unlit: `color + emissive`.
//! * `lambert` — `emissive + color * (ambient + Σ diffuse)`.
//! * `phong` — lambert plus `Σ light.color * intensity * att *
//!   spot * pow(max(R·V, 0), shininess)` for lights facing the
//!   fragment.
//! * point/spot attenuation is `1 / (1 + d²)`; a spot's cone is
//!   `spot_angle` wide (full angle), smoothed across
//!   `spot_penumbra` of the half-angle. Directional lights do not
//!   attenuate.
//! * a mesh node with no material renders with the default
//!   material: lambert, color (0.8, 0.8, 0.8).
//!
//! Transparency: opacity-1 triangles draw into the depth buffer in
//! document order (depth ties keep the first drawn — document
//! order is the tiebreak rule). Translucent triangles draw after,
//! far-to-near by centroid view depth (ties: document order),
//! depth-tested but not depth-written, blended source-over.
//! Draw order is fixed; there is no threading and no hash-map
//! iteration anywhere in the render path. Same state in → same
//! pixels out.

use crate::geom::*;
use crate::json::Value;
use crate::png::linear_to_byte;

#[derive(Clone, Copy)]
pub struct Material {
    pub kind: u8, // 0 flat, 1 lambert, 2 phong
    pub color: Vec3,
    pub opacity: f64,
    pub shininess: f64,
    pub emissive: Vec3,
}

impl Material {
    fn from_json(value: Option<&Value>) -> Material {
        let default = Material {
            kind: 1,
            color: [0.8, 0.8, 0.8],
            opacity: 1.0,
            shininess: 32.0,
            emissive: [0.0, 0.0, 0.0],
        };
        let Some(value) = value else { return default };
        let kind = match value.get("kind").and_then(|v| v.as_str()) {
            Some("flat") => 0,
            Some("phong") => 2,
            _ => 1,
        };
        Material {
            kind,
            color: value
                .get("color")
                .and_then(|v| v.as_vec3())
                .unwrap_or(default.color),
            opacity: value
                .get("opacity")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0),
            shininess: value
                .get("shininess")
                .and_then(|v| v.as_f64())
                .unwrap_or(32.0),
            emissive: value
                .get("emissive")
                .and_then(|v| v.as_vec3())
                .unwrap_or([0.0, 0.0, 0.0]),
        }
    }
}

#[derive(Clone, Copy)]
struct Light {
    kind: u8, // 0 ambient, 1 directional, 2 point, 3 spot
    color: Vec3,
    intensity: f64,
    position: Vec3,
    direction: Vec3,
    spot_cos_inner: f64,
    spot_cos_outer: f64,
}

#[derive(Clone, Copy)]
struct RVert {
    world: Vec3,
    view: Vec3,
    normal: Vec3,
}

struct DrawTri {
    verts: [RVert; 3],
    material: Material,
    order: usize,
}

pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

fn lights_from_state(state: &Value) -> Vec<Light> {
    let mut lights = Vec::new();
    let Some(nodes) = state.get("nodes").and_then(|n| n.as_obj()) else {
        return lights;
    };
    for (_, node) in nodes {
        if node.get("visible").and_then(|v| v.as_bool()) == Some(false) {
            continue;
        }
        let Some(light) = node.get("light") else { continue };
        if light.is_null() {
            continue;
        }
        let kind = match light.get("kind").and_then(|v| v.as_str()) {
            Some("ambient") => 0,
            Some("directional") => 1,
            Some("spot") => 3,
            _ => 2,
        };
        let angle = light
            .get("spot_angle")
            .and_then(|v| v.as_f64())
            .unwrap_or(45.0);
        let penumbra = light
            .get("spot_penumbra")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);
        let half = angle.to_radians() / 2.0;
        let inner = half * (1.0 - penumbra.clamp(0.0, 1.0));
        lights.push(Light {
            kind,
            color: light
                .get("color")
                .and_then(|v| v.as_vec3())
                .unwrap_or([1.0, 1.0, 1.0]),
            intensity: light
                .get("intensity")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0),
            position: light
                .get("position")
                .and_then(|v| v.as_vec3())
                .unwrap_or([0.0, 0.0, 0.0]),
            direction: light
                .get("direction")
                .and_then(|v| v.as_vec3())
                .unwrap_or([0.0, 0.0, -1.0]),
            spot_cos_inner: inner.cos(),
            spot_cos_outer: half.cos(),
        });
    }
    lights
}

impl Value {
    fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }
}

fn shade(
    material: &Material,
    pos: Vec3,
    normal: Vec3,
    cam_pos: Vec3,
    lights: &[Light],
) -> Vec3 {
    if material.kind == 0 {
        return v_add(material.color, material.emissive);
    }
    let n = v_norm(normal);
    let mut diffuse_irr = [0.0, 0.0, 0.0];
    let mut specular = [0.0, 0.0, 0.0];
    let view_dir = v_norm(v_sub(cam_pos, pos));
    for light in lights {
        let scaled = v_scale(light.color, light.intensity);
        if light.kind == 0 {
            diffuse_irr = v_add(diffuse_irr, scaled);
            continue;
        }
        let (l, att) = if light.kind == 1 {
            (v_scale(light.direction, -1.0), 1.0)
        } else {
            let to_light = v_sub(light.position, pos);
            let dist = v_len(to_light);
            if dist <= 1e-12 {
                continue;
            }
            (
                v_scale(to_light, 1.0 / dist),
                1.0 / (1.0 + dist * dist),
            )
        };
        let mut factor = att;
        if light.kind == 3 {
            let cos_ang = v_dot(v_norm(v_sub(pos, light.position)), light.direction);
            if light.spot_cos_inner > light.spot_cos_outer {
                let t = ((cos_ang - light.spot_cos_outer)
                    / (light.spot_cos_inner - light.spot_cos_outer))
                    .clamp(0.0, 1.0);
                factor *= t * t * (3.0 - 2.0 * t);
            } else if cos_ang < light.spot_cos_outer {
                factor = 0.0;
            }
            if factor <= 0.0 {
                continue;
            }
        }
        let ndotl = v_dot(n, l);
        if ndotl <= 0.0 {
            continue;
        }
        diffuse_irr = v_add(diffuse_irr, v_scale(scaled, factor * ndotl));
        if material.kind == 2 {
            // R = 2(N·L)N − L; specular rides the same cone/att
            let r = v_sub(v_scale(n, 2.0 * ndotl), l);
            let rv = v_dot(r, view_dir).max(0.0);
            if rv > 0.0 {
                specular = v_add(
                    specular,
                    v_scale(scaled, factor * rv.powf(material.shininess)),
                );
            }
        }
    }
    let mut out = v_add(material.emissive, specular);
    for i in 0..3 {
        out[i] += material.color[i] * diffuse_irr[i];
    }
    out
}

/// Clip a polygon (view-space verts) against one half-space
/// `keep(p) >= 0`, interpolating all carried attributes.
fn clip_plane(poly: &[RVert], keep: impl Fn(&RVert) -> f64) -> Vec<RVert> {
    if poly.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(poly.len() + 1);
    let mut prev = poly[poly.len() - 1];
    let mut prev_d = keep(&prev);
    for &cur in poly {
        let cur_d = keep(&cur);
        if cur_d >= 0.0 {
            if prev_d < 0.0 {
                out.push(lerp_vert(&prev, &cur, prev_d / (prev_d - cur_d)));
            }
            out.push(cur);
        } else if prev_d >= 0.0 {
            out.push(lerp_vert(&prev, &cur, prev_d / (prev_d - cur_d)));
        }
        prev = cur;
        prev_d = cur_d;
    }
    out
}

fn lerp_vert(a: &RVert, b: &RVert, t: f64) -> RVert {
    let mix = |x: Vec3, y: Vec3| v_add(x, v_scale(v_sub(y, x), t));
    RVert {
        world: mix(a.world, b.world),
        view: mix(a.view, b.view),
        normal: mix(a.normal, b.normal),
    }
}

#[derive(Clone, Copy)]
struct ScreenVert {
    x: f64,
    y: f64,
    inv_w: f64,
    world_over_w: Vec3,
    normal_over_w: Vec3,
    view_over_w: Vec3,
}

/// Render one frame state (the core's `scene_state_at` dict) to
/// an sRGB PNG frame.
pub fn render_state(state: &Value, width: u32, height: u32) -> Result<Frame, String> {
    let camera = state
        .get("camera")
        .ok_or("frame state has no camera record")?;
    let view_mat = camera
        .get("view_matrix")
        .and_then(|v| v.as_mat4())
        .ok_or("camera record has no view_matrix")?;
    let proj_mat = camera
        .get("projection_matrix")
        .and_then(|v| v.as_mat4())
        .ok_or("camera record has no projection_matrix")?;
    let cam_pos = camera
        .get("position")
        .and_then(|v| v.as_vec3())
        .ok_or("camera record has no position")?;
    let near = camera.get("near").and_then(|v| v.as_f64()).unwrap_or(0.1);
    let far = camera.get("far").and_then(|v| v.as_f64()).unwrap_or(1000.0);
    let background = state
        .get("background")
        .and_then(|v| v.as_vec3())
        .unwrap_or([0.0, 0.0, 0.0]);
    let lights = lights_from_state(state);

    let (w, h) = (width as usize, height as usize);
    let mut color: Vec<Vec3> = vec![background; w * h];
    let mut depth: Vec<f64> = vec![f64::INFINITY; w * h];

    // gather draw triangles in document order
    let mut opaque: Vec<DrawTri> = Vec::new();
    let mut translucent: Vec<DrawTri> = Vec::new();
    let mut order = 0usize;
    if let Some(nodes) = state.get("nodes").and_then(|n| n.as_obj()) {
        for (_, node) in nodes {
            if node.get("visible").and_then(|v| v.as_bool()) == Some(false) {
                continue;
            }
            let Some(mesh_value) = node.get("mesh") else { continue };
            if mesh_value.is_null() {
                continue;
            }
            let mesh = MeshData::from_json(mesh_value)?;
            let tris = tessellate(&mesh)?;
            let world_mat = node
                .get("world_matrix")
                .and_then(|v| v.as_mat4())
                .ok_or("mesh node has no world_matrix")?;
            let nmat = normal_matrix(&world_mat);
            let material = Material::from_json(node.get("material"));
            for tri in tris {
                let mut verts = [RVert {
                    world: [0.0; 3],
                    view: [0.0; 3],
                    normal: [0.0; 3],
                }; 3];
                for (i, slot) in verts.iter_mut().enumerate() {
                    let world = mat_point(&world_mat, tri.p[i]);
                    *slot = RVert {
                        world,
                        view: mat_point(&view_mat, world),
                        normal: mat3_dir(&nmat, tri.n[i]),
                    };
                }
                let draw = DrawTri {
                    verts,
                    material,
                    order,
                };
                order += 1;
                if material.opacity >= 1.0 {
                    opaque.push(draw);
                } else {
                    translucent.push(draw);
                }
            }
        }
    }

    for tri in &opaque {
        rasterize(
            tri, &proj_mat, near, far, cam_pos, &lights, width, height,
            &mut color, &mut depth, false,
        );
    }
    translucent.sort_by(|a, b| {
        let za = centroid_view_depth(a);
        let zb = centroid_view_depth(b);
        zb.total_cmp(&za).then(a.order.cmp(&b.order))
    });
    for tri in &translucent {
        rasterize(
            tri, &proj_mat, near, far, cam_pos, &lights, width, height,
            &mut color, &mut depth, true,
        );
    }

    let mut rgb = vec![0u8; w * h * 3];
    for (i, px) in color.iter().enumerate() {
        rgb[i * 3] = linear_to_byte(px[0]);
        rgb[i * 3 + 1] = linear_to_byte(px[1]);
        rgb[i * 3 + 2] = linear_to_byte(px[2]);
    }
    Ok(Frame { width, height, rgb })
}

fn centroid_view_depth(tri: &DrawTri) -> f64 {
    (tri.verts[0].view[2] + tri.verts[1].view[2] + tri.verts[2].view[2]) / 3.0
}

#[allow(clippy::too_many_arguments)]
fn rasterize(
    tri: &DrawTri,
    proj_mat: &Mat4,
    near: f64,
    far: f64,
    cam_pos: Vec3,
    lights: &[Light],
    width: u32,
    height: u32,
    color: &mut [Vec3],
    depth: &mut [f64],
    blend: bool,
) {
    // view-space clipping: near plane (-z >= near), far (-z <= far)
    let poly = clip_plane(&tri.verts, |v| -v.view[2] - near);
    if poly.len() < 3 {
        return;
    }
    let poly = clip_plane(&poly, |v| far + v.view[2]);
    if poly.len() < 3 {
        return;
    }
    for k in 1..poly.len() - 1 {
        let fan = [poly[0], poly[k], poly[k + 1]];
        rasterize_one(
            &fan, tri.material, proj_mat, cam_pos, lights, width, height,
            color, depth, blend,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn rasterize_one(
    verts: &[RVert; 3],
    material: Material,
    proj_mat: &Mat4,
    cam_pos: Vec3,
    lights: &[Light],
    width: u32,
    height: u32,
    color: &mut [Vec3],
    depth: &mut [f64],
    blend: bool,
) {
    let (wf, hf) = (width as f64, height as f64);
    let mut sv = [ScreenVert {
        x: 0.0,
        y: 0.0,
        inv_w: 0.0,
        world_over_w: [0.0; 3],
        normal_over_w: [0.0; 3],
        view_over_w: [0.0; 3],
    }; 3];
    for (i, v) in verts.iter().enumerate() {
        let clip = mat_point4(proj_mat, v.view);
        if clip[3] <= 0.0 {
            return; // behind the eye after clipping — should not happen
        }
        let inv_w = 1.0 / clip[3];
        let ndc_x = clip[0] * inv_w;
        let ndc_y = clip[1] * inv_w;
        sv[i] = ScreenVert {
            x: (ndc_x * 0.5 + 0.5) * wf,
            y: (0.5 - ndc_y * 0.5) * hf,
            inv_w,
            world_over_w: v_scale(v.world, inv_w),
            normal_over_w: v_scale(v.normal, inv_w),
            view_over_w: v_scale(v.view, inv_w),
        };
    }
    let (x0, y0) = (sv[0].x, sv[0].y);
    let (x1, y1) = (sv[1].x, sv[1].y);
    let (x2, y2) = (sv[2].x, sv[2].y);
    let area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
    if area == 0.0 {
        return;
    }
    let min_x = x0.min(x1).min(x2).floor().max(0.0) as i64;
    let max_x = x0.max(x1).max(x2).ceil().min(wf - 1.0) as i64;
    let min_y = y0.min(y1).min(y2).floor().max(0.0) as i64;
    let max_y = y0.max(y1).max(y2).ceil().min(hf - 1.0) as i64;
    if min_x > max_x || min_y > max_y {
        return;
    }
    let edge = |ax: f64, ay: f64, bx: f64, by: f64, px: f64, py: f64| {
        (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    };
    for py_i in min_y..=max_y {
        for px_i in min_x..=max_x {
            let px = px_i as f64 + 0.5;
            let py = py_i as f64 + 0.5;
            let w0 = edge(x1, y1, x2, y2, px, py) / area;
            let w1 = edge(x2, y2, x0, y0, px, py) / area;
            let w2 = edge(x0, y0, x1, y1, px, py) / area;
            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                continue;
            }
            let inv_w = w0 * sv[0].inv_w + w1 * sv[1].inv_w + w2 * sv[2].inv_w;
            if inv_w <= 0.0 {
                continue;
            }
            let recip = 1.0 / inv_w;
            let interp = |attr: &[Vec3; 3]| -> Vec3 {
                [
                    (w0 * attr[0][0] + w1 * attr[1][0] + w2 * attr[2][0]) * recip,
                    (w0 * attr[0][1] + w1 * attr[1][1] + w2 * attr[2][1]) * recip,
                    (w0 * attr[0][2] + w1 * attr[1][2] + w2 * attr[2][2]) * recip,
                ]
            };
            let view_pos = interp(&[
                sv[0].view_over_w,
                sv[1].view_over_w,
                sv[2].view_over_w,
            ]);
            let frag_depth = -view_pos[2];
            let idx = (py_i as usize) * (width as usize) + (px_i as usize);
            if !(frag_depth < depth[idx]) {
                continue;
            }
            let world_pos = interp(&[
                sv[0].world_over_w,
                sv[1].world_over_w,
                sv[2].world_over_w,
            ]);
            let normal = interp(&[
                sv[0].normal_over_w,
                sv[1].normal_over_w,
                sv[2].normal_over_w,
            ]);
            let shaded = shade(&material, world_pos, normal, cam_pos, lights);
            if blend {
                let a = material.opacity;
                let dst = color[idx];
                color[idx] = [
                    shaded[0] * a + dst[0] * (1.0 - a),
                    shaded[1] * a + dst[1] * (1.0 - a),
                    shaded[2] * a + dst[2] * (1.0 - a),
                ];
            } else {
                depth[idx] = frag_depth;
                color[idx] = shaded;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse;

    fn state_json(body: &str) -> Value {
        parse(body).expect("state json")
    }

    /// Camera at (0,0,5) looking at the origin, fov 90, 8x8 canvas:
    /// view matrix = translate(0,0,-5); projection focal 1.
    fn cam_state(nodes: &str) -> Value {
        state_json(&format!(
            r#"{{"frame": 0, "seconds": 0.0, "background": [0.0, 0.0, 0.0],
            "camera": {{"name": "c", "position": [0, 0, 5],
              "view_matrix": [1,0,0,0, 0,1,0,0, 0,0,1,-5, 0,0,0,1],
              "projection_matrix": [1,0,0,0, 0,1,0,0, 0,0,-1.2,-2.2, 0,0,-1,0],
              "fov": 90.0, "near": 1.0, "far": 11.0, "aspect": 1.0}},
            "nodes": {{ {nodes} }}, "materials": {{}}}}"#
        ))
    }

    #[test]
    fn lambert_facing_light_is_exact() {
        // plane at z=0 facing +Z; directional light shining -Z
        // (surface->light = +Z = the normal): N·L = 1 exactly.
        let state = cam_state(
            r#""quad": {"kind": "mesh", "parent": null,
              "world_matrix": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
              "world_position": [0,0,0], "visible": true,
              "mesh": {"kind": "primitive", "primitive": "plane",
                       "params": {"size": 4.0}, "vertices": [], "faces": [],
                       "normals": []},
              "material": {"kind": "lambert", "color": [0.5, 0.25, 1.0],
                           "opacity": 1.0, "shininess": 32.0,
                           "emissive": [0, 0, 0]},
              "light": null},
            "sun": {"kind": "light", "parent": null,
              "world_matrix": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
              "world_position": [0,0,5], "visible": true, "mesh": null,
              "material": null,
              "light": {"kind": "directional", "color": [1, 1, 1],
                        "intensity": 1.0, "spot_angle": 45.0,
                        "spot_penumbra": 0.5, "direction": [0, 0, -1],
                        "position": [0, 0, 5]}}"#,
        );
        let frame = render_state(&state, 8, 8).expect("render");
        // center pixel: linear (0.5, 0.25, 1.0) -> sRGB (188, 137, 255)
        let idx = (4 * 8 + 4) * 3;
        assert_eq!(&frame.rgb[idx..idx + 3], &[188, 137, 255]);
    }

    #[test]
    fn nearer_triangle_wins_the_overlap() {
        // red cube in front of a bigger blue one, both flat
        let cube = |dx: f64, dz: f64, color: &str| {
            format!(
                r#""world_matrix": [1,0,0,{dx}, 0,1,0,0, 0,0,1,{dz}, 0,0,0,1],
                "world_position": [{dx},0,{dz}], "visible": true,
                "mesh": {{"kind": "primitive", "primitive": "cube",
                         "params": {{"size": 2.0}}, "vertices": [], "faces": [],
                         "normals": []}},
                "material": {{"kind": "flat", "color": {color},
                             "opacity": 1.0, "shininess": 32.0,
                             "emissive": [0, 0, 0]}}, "light": null"#
            )
        };
        let state = cam_state(&format!(
            r#""far": {{"kind": "mesh", "parent": null, {}}},
            "near": {{"kind": "mesh", "parent": null, {}}}"#,
            cube(0.0, -3.0, "[0, 0, 1]"),
            cube(0.0, 0.0, "[1, 0, 0]"),
        ));
        let frame = render_state(&state, 8, 8).expect("render");
        let idx = (4 * 8 + 4) * 3;
        assert_eq!(
            &frame.rgb[idx..idx + 3],
            &[255, 0, 0],
            "the nearer (red) cube must win the center pixel"
        );
    }

    #[test]
    fn empty_scene_is_background() {
        let state = cam_state("");
        let frame = render_state(&state, 4, 4).expect("render");
        assert!(frame.rgb.iter().all(|&b| b == 0));
    }
}
