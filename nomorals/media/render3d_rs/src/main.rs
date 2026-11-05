//! devon-render3d — the native Rust 3D renderer rung (§1.5e).
//!
//! Protocol (deliberately boring): argv[1] is a job JSON file —
//! `{"width": W, "height": H, "out_dir": dir, "pattern":
//! "f_%04d.png", "frames": [{"frame": N, "state": <the core's
//! scene_state_at dict>}, ...]}` — and the result is one JSON
//! object on stdout. The Python core evaluates every frame state
//! (its deterministic frame function is the one scene semantics);
//! this binary only rasterizes the states it is given. It never
//! re-evaluates a scene and never invents a frame.

mod geom;
mod json;
mod png;
mod render;

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

fn pattern_name(pattern: &str, frame: i64) -> String {
    // the pipeline's pattern shape: %d or %0Nd inside a literal name
    if let Some(start) = pattern.find('%') {
        let rest = &pattern[start + 1..];
        if let Some(dpos) = rest.find('d') {
            let spec = &rest[..dpos];
            let num = if let Some(width) = spec.strip_prefix('0') {
                let w: usize = width.parse().unwrap_or(0);
                format!("{:0width$}", frame, width = w)
            } else {
                format!("{frame}")
            };
            return format!("{}{}{}", &pattern[..start], num, &rest[dpos + 1..]);
        }
    }
    pattern.to_string()
}

fn run() -> Result<String, String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 2 && args[1] == "--version" {
        return Ok(format!("devon-render3d {}", env!("CARGO_PKG_VERSION")));
    }
    if args.len() != 2 {
        return Err(
            "usage: render3d_rs <job.json> | --version".to_string(),
        );
    }
    let started = Instant::now();
    let text = std::fs::read_to_string(&args[1])
        .map_err(|e| format!("cannot read job {}: {e}", args[1]))?;
    let job = json::parse(&text)?;
    let width = job
        .get("width")
        .and_then(|v| v.as_f64())
        .ok_or("job has no width")? as u32;
    let height = job
        .get("height")
        .and_then(|v| v.as_f64())
        .ok_or("job has no height")? as u32;
    if width == 0 || height == 0 {
        return Err("job width/height must be positive".to_string());
    }
    let out_dir = PathBuf::from(
        job.get("out_dir")
            .and_then(|v| v.as_str())
            .ok_or("job has no out_dir")?,
    );
    std::fs::create_dir_all(&out_dir)
        .map_err(|e| format!("cannot create {}: {e}", out_dir.display()))?;
    let pattern = job
        .get("pattern")
        .and_then(|v| v.as_str())
        .unwrap_or("f_%04d.png");
    let frames = job
        .get("frames")
        .and_then(|v| v.as_arr())
        .ok_or("job has no frames array")?;

    let mut rendered = Vec::with_capacity(frames.len());
    for entry in frames {
        let frame_no = entry
            .get("frame")
            .and_then(|v| v.as_f64())
            .ok_or("frame entry has no frame number")? as i64;
        let state = entry.get("state").ok_or("frame entry has no state")?;
        let frame_start = Instant::now();
        let frame = render::render_state(state, width, height)?;
        let bytes = png::encode_png(frame.width, frame.height, &frame.rgb);
        let name = pattern_name(pattern, frame_no);
        let path = out_dir.join(&name);
        std::fs::write(&path, &bytes)
            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        rendered.push(format!(
            "{{\"frame\": {}, \"file\": {}, \"bytes\": {}, \"render_ms\": {:.3}}}",
            frame_no,
            json_escape(&name),
            bytes.len(),
            frame_start.elapsed().as_secs_f64() * 1000.0
        ));
    }
    Ok(format!(
        "{{\"ok\": true, \"engine\": \"devon-render3d\", \"version\": {}, \
         \"frames\": [{}], \"total_ms\": {:.3}}}",
        json_escape(env!("CARGO_PKG_VERSION")),
        rendered.join(", "),
        started.elapsed().as_secs_f64() * 1000.0
    ))
}

fn json_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn main() {
    let mut stdout = std::io::stdout().lock();
    match run() {
        Ok(line) => {
            let _ = writeln!(stdout, "{line}");
        }
        Err(message) => {
            let _ = writeln!(
                stdout,
                "{{\"ok\": false, \"error\": {}}}",
                json_escape(&message)
            );
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_formatting() {
        assert_eq!(pattern_name("f_%04d.png", 7), "f_0007.png");
        assert_eq!(pattern_name("f_%04d.png", 12345), "f_12345.png");
        assert_eq!(pattern_name("frame_%d.png", 3), "frame_3.png");
        assert_eq!(pattern_name("still.png", 3), "still.png");
    }
}
