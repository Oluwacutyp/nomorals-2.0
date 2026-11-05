//! Vector/matrix math (the Scene3D core's conventions: row-major
//! 4x4, column vectors, camera looks down -Z) and tessellation —
//! every mesh, primitive or explicit, becomes a triangle soup with
//! per-vertex normals in local space. The rasterizer never sees a
//! primitive; it only sees triangles.

use crate::json::Value;

pub type Vec3 = [f64; 3];
pub type Mat4 = [f64; 16];

pub fn v_add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn v_sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn v_scale(a: Vec3, s: f64) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub fn v_dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn v_cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn v_len(a: Vec3) -> f64 {
    v_dot(a, a).sqrt()
}

pub fn v_norm(a: Vec3) -> Vec3 {
    let len = v_len(a);
    if len <= 0.0 {
        [0.0, 0.0, 0.0]
    } else {
        v_scale(a, 1.0 / len)
    }
}

pub fn mat_point(m: &Mat4, p: Vec3) -> Vec3 {
    [
        m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
        m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
        m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11],
    ]
}

/// Apply a 4x4 to a point, returning the full homogeneous result.
pub fn mat_point4(m: &Mat4, p: Vec3) -> [f64; 4] {
    [
        m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
        m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
        m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11],
        m[12] * p[0] + m[13] * p[1] + m[14] * p[2] + m[15],
    ]
}

/// The normal matrix: inverse-transpose of the world matrix's
/// upper 3x3, as a row-major 3x3. Degenerate (zero-determinant)
/// transforms fall back to the raw upper 3x3 — a named compromise:
/// shading on a collapsed axis is approximate, never a crash.
pub fn normal_matrix(m: &Mat4) -> [f64; 9] {
    let (a, b, c) = (m[0], m[1], m[2]);
    let (d, e, f) = (m[4], m[5], m[6]);
    let (g, h, i) = (m[8], m[9], m[10]);
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if det.abs() < 1e-12 {
        return [a, b, c, d, e, f, g, h, i];
    }
    let inv = 1.0 / det;
    // inverse (row-major), then transposed on write
    let inv3 = [
        (e * i - f * h) * inv,
        (c * h - b * i) * inv,
        (b * f - c * e) * inv,
        (f * g - d * i) * inv,
        (a * i - c * g) * inv,
        (c * d - a * f) * inv,
        (d * h - e * g) * inv,
        (b * g - a * h) * inv,
        (a * e - b * d) * inv,
    ];
    [
        inv3[0], inv3[3], inv3[6], inv3[1], inv3[4], inv3[7], inv3[2], inv3[5],
        inv3[8],
    ]
}

pub fn mat3_dir(m: &[f64; 9], v: Vec3) -> Vec3 {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

/// One tessellated triangle: positions + normals, local space.
#[derive(Debug, Clone, Copy)]
pub struct Tri {
    pub p: [Vec3; 3],
    pub n: [Vec3; 3],
}

/// A mesh payload as read from the frame-state record.
pub struct MeshData {
    pub kind: String,
    pub primitive: String,
    pub params: Vec<(String, f64)>,
    pub vertices: Vec<Vec3>,
    pub faces: Vec<Vec<usize>>,
    pub normals: Vec<Vec3>,
}

impl MeshData {
    pub fn from_json(value: &Value) -> Result<MeshData, String> {
        let mut params = Vec::new();
        if let Some(obj) = value.get("params").and_then(|p| p.as_obj()) {
            for (k, v) in obj {
                params.push((
                    k.clone(),
                    v.as_f64()
                        .ok_or_else(|| format!("primitive param {k} is not a number"))?,
                ));
            }
        }
        let mut vertices = Vec::new();
        if let Some(items) = value.get("vertices").and_then(|v| v.as_arr()) {
            for item in items {
                vertices.push(
                    item.as_vec3()
                        .ok_or_else(|| "mesh vertex is not a 3-vector".to_string())?,
                );
            }
        }
        let mut normals = Vec::new();
        if let Some(items) = value.get("normals").and_then(|v| v.as_arr()) {
            for item in items {
                normals.push(
                    item.as_vec3()
                        .ok_or_else(|| "mesh normal is not a 3-vector".to_string())?,
                );
            }
        }
        let mut faces = Vec::new();
        if let Some(items) = value.get("faces").and_then(|v| v.as_arr()) {
            for item in items {
                let corners = item
                    .as_arr()
                    .ok_or_else(|| "mesh face is not an array".to_string())?;
                let mut face = Vec::with_capacity(corners.len());
                for corner in corners {
                    let idx = corner
                        .as_f64()
                        .ok_or_else(|| "mesh face index is not a number".to_string())?;
                    face.push(idx as usize);
                }
                faces.push(face);
            }
        }
        Ok(MeshData {
            kind: value
                .get("kind")
                .and_then(|v| v.as_str())
                .unwrap_or("primitive")
                .to_string(),
            primitive: value
                .get("primitive")
                .and_then(|v| v.as_str())
                .unwrap_or("cube")
                .to_string(),
            params,
            vertices,
            faces,
            normals,
        })
    }

    pub fn param(&self, key: &str, default: f64) -> f64 {
        self.params
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| *v)
            .unwrap_or(default)
    }
}

fn quad(tris: &mut Vec<Tri>, corners: [Vec3; 4], normal: Vec3) {
    let n = [normal, normal, normal];
    tris.push(Tri {
        p: [corners[0], corners[1], corners[2]],
        n,
    });
    tris.push(Tri {
        p: [corners[0], corners[2], corners[3]],
        n,
    });
}

fn tessellate_cube(mesh: &MeshData) -> Vec<Tri> {
    let size = mesh.param("size", 1.0);
    let (hx, hy, hz) = (
        mesh.param("width", size) / 2.0,
        mesh.param("height", size) / 2.0,
        mesh.param("depth", size) / 2.0,
    );
    let mut tris = Vec::with_capacity(12);
    // +X / -X
    quad(&mut tris, [[hx, -hy, hz], [hx, -hy, -hz], [hx, hy, -hz], [hx, hy, hz]], [1.0, 0.0, 0.0]);
    quad(&mut tris, [[-hx, -hy, -hz], [-hx, -hy, hz], [-hx, hy, hz], [-hx, hy, -hz]], [-1.0, 0.0, 0.0]);
    // +Y / -Y
    quad(&mut tris, [[-hx, hy, hz], [hx, hy, hz], [hx, hy, -hz], [-hx, hy, -hz]], [0.0, 1.0, 0.0]);
    quad(&mut tris, [[-hx, -hy, -hz], [hx, -hy, -hz], [hx, -hy, hz], [-hx, -hy, hz]], [0.0, -1.0, 0.0]);
    // +Z / -Z
    quad(&mut tris, [[-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]], [0.0, 0.0, 1.0]);
    quad(&mut tris, [[hx, -hy, -hz], [-hx, -hy, -hz], [-hx, hy, -hz], [hx, hy, -hz]], [0.0, 0.0, -1.0]);
    tris
}

fn tessellate_plane(mesh: &MeshData) -> Vec<Tri> {
    let size = mesh.param("size", 1.0);
    let (hw, hh) = (
        mesh.param("width", size) / 2.0,
        mesh.param("height", size) / 2.0,
    );
    let mut tris = Vec::with_capacity(2);
    quad(
        &mut tris,
        [[-hw, -hh, 0.0], [hw, -hh, 0.0], [hw, hh, 0.0], [-hw, hh, 0.0]],
        [0.0, 0.0, 1.0],
    );
    tris
}

fn ring_point(radius: f64, y: f64, theta: f64) -> Vec3 {
    [radius * theta.cos(), y, radius * theta.sin()]
}

fn tessellate_sphere(mesh: &MeshData) -> Vec<Tri> {
    let radius = mesh.param("radius", 1.0);
    let segments = (mesh.param("segments", 32.0) as usize).clamp(3, 256);
    let rings = (mesh.param("rings", (segments / 2) as f64) as usize).clamp(2, 128);
    let mut verts: Vec<Vec3> = Vec::with_capacity((rings + 1) * (segments + 1));
    for i in 0..=rings {
        let phi = std::f64::consts::PI * (i as f64) / (rings as f64);
        let y = radius * phi.cos();
        let rad = radius * phi.sin();
        for j in 0..=segments {
            let theta = 2.0 * std::f64::consts::PI * (j as f64) / (segments as f64);
            verts.push([rad * theta.cos(), y, rad * theta.sin()]);
        }
    }
    let at = |i: usize, j: usize| verts[i * (segments + 1) + j];
    let mut tris = Vec::with_capacity(rings * segments * 2);
    for i in 0..rings {
        for j in 0..segments {
            let (a, b) = (at(i, j), at(i, j + 1));
            let (c, d) = (at(i + 1, j), at(i + 1, j + 1));
            let n = |p: Vec3| v_norm(p);
            tris.push(Tri { p: [a, b, c], n: [n(a), n(b), n(c)] });
            tris.push(Tri { p: [b, d, c], n: [n(b), n(d), n(c)] });
        }
    }
    tris
}

fn tessellate_cylinder(mesh: &MeshData) -> Vec<Tri> {
    let radius = mesh.param("radius", 0.5);
    let height = mesh.param("height", 1.0);
    let segments = (mesh.param("segments", 32.0) as usize).clamp(3, 256);
    let capped = mesh.param("caps", 1.0) != 0.0;
    let (top_y, bot_y) = (height / 2.0, -height / 2.0);
    let theta = |j: usize| 2.0 * std::f64::consts::PI * (j as f64) / (segments as f64);
    let mut tris = Vec::with_capacity(segments * 4);
    for j in 0..segments {
        let (t0, t1) = (theta(j), theta(j + 1));
        let n0 = [t0.cos(), 0.0, t0.sin()];
        let n1 = [t1.cos(), 0.0, t1.sin()];
        let (tp0, tp1) = (ring_point(radius, top_y, t0), ring_point(radius, top_y, t1));
        let (bt0, bt1) = (ring_point(radius, bot_y, t0), ring_point(radius, bot_y, t1));
        tris.push(Tri { p: [tp0, tp1, bt0], n: [n0, n1, n0] });
        tris.push(Tri { p: [tp1, bt1, bt0], n: [n1, n1, n0] });
        if capped {
            let up = [0.0, 1.0, 0.0];
            let down = [0.0, -1.0, 0.0];
            tris.push(Tri {
                p: [[0.0, top_y, 0.0], tp1, tp0],
                n: [up, up, up],
            });
            tris.push(Tri {
                p: [[0.0, bot_y, 0.0], bt0, bt1],
                n: [down, down, down],
            });
        }
    }
    tris
}

fn tessellate_cone(mesh: &MeshData) -> Vec<Tri> {
    let radius = mesh.param("radius", 0.5);
    let height = mesh.param("height", 1.0);
    let segments = (mesh.param("segments", 32.0) as usize).clamp(3, 256);
    let capped = mesh.param("caps", 1.0) != 0.0;
    let apex: Vec3 = [0.0, height / 2.0, 0.0];
    let base_y = -height / 2.0;
    // analytic side normal at angle t: normalize((h*cos, r, h*sin))
    let side_n = |t: f64| v_norm([height * t.cos(), radius, height * t.sin()]);
    let mut tris = Vec::with_capacity(segments * 2);
    for j in 0..segments {
        let t0 = 2.0 * std::f64::consts::PI * (j as f64) / (segments as f64);
        let t1 = 2.0 * std::f64::consts::PI * ((j + 1) as f64) / (segments as f64);
        let mid = (t0 + t1) / 2.0;
        let b0 = ring_point(radius, base_y, t0);
        let b1 = ring_point(radius, base_y, t1);
        tris.push(Tri {
            p: [apex, b1, b0],
            n: [side_n(mid), side_n(t1), side_n(t0)],
        });
        if capped {
            let down = [0.0, -1.0, 0.0];
            tris.push(Tri {
                p: [[0.0, base_y, 0.0], b0, b1],
                n: [down, down, down],
            });
        }
    }
    tris
}

fn tessellate_explicit(mesh: &MeshData) -> Result<Vec<Tri>, String> {
    if mesh.vertices.is_empty() || mesh.faces.is_empty() {
        return Err("explicit mesh carries no vertices/faces".to_string());
    }
    let has_normals = mesh.normals.len() == mesh.vertices.len();
    let mut tris = Vec::new();
    for face in &mesh.faces {
        if face.len() < 3 {
            return Err("explicit mesh face has fewer than 3 corners".to_string());
        }
        for &idx in face {
            if idx >= mesh.vertices.len() {
                return Err(format!(
                    "explicit mesh face index {idx} out of range"
                ));
            }
        }
        // fan triangulation around corner 0 (faces are convex by
        // the authoring contract; the core validates the indices)
        for k in 1..face.len() - 1 {
            let idx = [face[0], face[k], face[k + 1]];
            let p = [
                mesh.vertices[idx[0]],
                mesh.vertices[idx[1]],
                mesh.vertices[idx[2]],
            ];
            let n = if has_normals {
                [
                    mesh.normals[idx[0]],
                    mesh.normals[idx[1]],
                    mesh.normals[idx[2]],
                ]
            } else {
                let face_n = v_norm(v_cross(v_sub(p[1], p[0]), v_sub(p[2], p[0])));
                [face_n, face_n, face_n]
            };
            tris.push(Tri { p, n });
        }
    }
    Ok(tris)
}

/// Tessellate any mesh payload into local-space triangles.
pub fn tessellate(mesh: &MeshData) -> Result<Vec<Tri>, String> {
    match mesh.kind.as_str() {
        "explicit" => tessellate_explicit(mesh),
        "primitive" => match mesh.primitive.as_str() {
            "cube" => Ok(tessellate_cube(mesh)),
            "sphere" => Ok(tessellate_sphere(mesh)),
            "plane" => Ok(tessellate_plane(mesh)),
            "cylinder" => Ok(tessellate_cylinder(mesh)),
            "cone" => Ok(tessellate_cone(mesh)),
            other => Err(format!("unknown primitive {other:?}")),
        },
        other => Err(format!("unknown mesh kind {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prim(name: &str, params: &[(&str, f64)]) -> MeshData {
        MeshData {
            kind: "primitive".to_string(),
            primitive: name.to_string(),
            params: params.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            vertices: Vec::new(),
            faces: Vec::new(),
            normals: Vec::new(),
        }
    }

    #[test]
    fn primitive_triangle_counts() {
        assert_eq!(tessellate(&prim("cube", &[])).unwrap().len(), 12);
        assert_eq!(tessellate(&prim("plane", &[])).unwrap().len(), 2);
        // sphere: rings * segments * 2 with defaults 32 x 16
        assert_eq!(tessellate(&prim("sphere", &[])).unwrap().len(), 32 * 16 * 2);
        // cylinder capped: (2 side + 2 cap) * segments
        assert_eq!(
            tessellate(&prim("cylinder", &[("segments", 8.0)])).unwrap().len(),
            32
        );
        // cone capped: (1 side + 1 cap) * segments
        assert_eq!(
            tessellate(&prim("cone", &[("segments", 8.0)])).unwrap().len(),
            16
        );
    }

    #[test]
    fn cube_face_normals_agree_with_winding() {
        for tri in tessellate(&prim("cube", &[("size", 2.0)])).unwrap() {
            let geo = v_cross(v_sub(tri.p[1], tri.p[0]), v_sub(tri.p[2], tri.p[0]));
            assert!(
                v_dot(geo, tri.n[0]) > 0.0,
                "cube face winding disagrees with its normal"
            );
            for p in tri.p {
                for c in p {
                    assert!(c.abs() <= 1.0 + 1e-12);
                }
            }
        }
    }

    #[test]
    fn sphere_normals_are_radial() {
        for tri in tessellate(&prim("sphere", &[("radius", 2.0), ("segments", 8.0)])).unwrap()
        {
            for (p, n) in tri.p.iter().zip(tri.n.iter()) {
                if v_len(*p) > 1e-9 {
                    let expect = v_norm(*p);
                    assert!(v_dot(expect, *n) > 0.999);
                    assert!((v_len(*p) - 2.0).abs() < 1e-9);
                }
            }
        }
    }

    #[test]
    fn explicit_fan_and_face_normals() {
        let mesh = MeshData {
            kind: "explicit".to_string(),
            primitive: String::new(),
            params: Vec::new(),
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            faces: vec![vec![0, 1, 2, 3]],
            normals: Vec::new(),
        };
        let tris = tessellate(&mesh).unwrap();
        assert_eq!(tris.len(), 2);
        assert_eq!(tris[0].n[0], [0.0, 0.0, 1.0]);
    }

    #[test]
    fn normal_matrix_undoes_nonuniform_scale() {
        // scale (2,1,1): a 45-degree normal must tilt away from X
        let world: Mat4 = [
            2.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let nm = normal_matrix(&world);
        let n = v_norm(mat3_dir(&nm, v_norm([1.0, 1.0, 0.0])));
        assert!(n[0] < n[1], "normal must tilt toward the unscaled axis");
        assert!((n[0] - 0.4472).abs() < 1e-3);
    }
}
