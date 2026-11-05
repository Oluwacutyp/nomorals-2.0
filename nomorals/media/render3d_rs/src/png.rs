//! The pinned PNG encoder: 8-bit RGB, filter 0 on every scanline,
//! zlib stream of *stored* (uncompressed) deflate blocks. No
//! compression heuristics, no timestamps, no ancillary chunks —
//! the same pixels always produce the same bytes, on any host,
//! from any build of this source. That is the encoder half of the
//! rung's determinism contract; the other half is the rasterizer's
//! fixed evaluation order.

const PNG_SIG: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];

fn crc32(data: &[u8]) -> u32 {
    // IEEE CRC-32, table-free bitwise form — the files are small
    // and the encoder runs once per frame; clarity over speed.
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for chunk in data.chunks(5552) {
        for &byte in chunk {
            a += byte as u32;
            b += a;
        }
        a %= MOD;
        b %= MOD;
    }
    (b << 16) | a
}

fn write_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

fn zlib_stored(data: &[u8]) -> Vec<u8> {
    // zlib header 0x78 0x01: 32K window, no compression, FCHECK ok.
    let mut out = vec![0x78, 0x01];
    let mut chunks: Vec<&[u8]> = data.chunks(65535).collect();
    if chunks.is_empty() {
        chunks.push(&[]);
    }
    let last = chunks.len() - 1;
    for (i, block) in chunks.iter().enumerate() {
        out.push(if i == last { 0x01 } else { 0x00 }); // BFINAL, BTYPE=00
        let len = block.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(block);
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

/// Encode an RGB frame (row-major, 3 bytes per pixel) as PNG bytes.
pub fn encode_png(width: u32, height: u32, rgb: &[u8]) -> Vec<u8> {
    assert_eq!(
        rgb.len(),
        (width as usize) * (height as usize) * 3,
        "rgb buffer must be width*height*3"
    );
    let mut raw = Vec::with_capacity((width as usize * 3 + 1) * height as usize);
    let stride = width as usize * 3;
    for y in 0..height as usize {
        raw.push(0); // filter: none
        raw.extend_from_slice(&rgb[y * stride..(y + 1) * stride]);
    }
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit, RGB, deflate, std filters, no interlace
    let mut out = Vec::with_capacity(raw.len() + 64);
    out.extend_from_slice(&PNG_SIG);
    write_chunk(&mut out, b"IHDR", &ihdr);
    write_chunk(&mut out, b"IDAT", &zlib_stored(&raw));
    write_chunk(&mut out, b"IEND", &[]);
    out
}

/// sRGB electro-optical transfer (the output encoding): linear
/// [0,1] → sRGB [0,1]. Values outside [0,1] clamp — no tone mapping.
pub fn srgb_encode(linear: f64) -> f64 {
    let c = linear.clamp(0.0, 1.0);
    if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

pub fn linear_to_byte(linear: f64) -> u8 {
    (srgb_encode(linear) * 255.0 + 0.5).floor() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_known_vector() {
        // CRC-32 of "123456789" is the catalogue check value.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(adler32(b"123456789"), 0x091E_01DE);
    }

    #[test]
    fn png_structure_and_stored_roundtrip() {
        let rgb: Vec<u8> = (0..4 * 2 * 3).map(|i| i as u8).collect();
        let png = encode_png(4, 2, &rgb);
        assert_eq!(&png[..8], &PNG_SIG);
        // walk chunks, inflate the stored stream by hand, compare
        let mut pos = 8;
        let mut idat: Vec<u8> = Vec::new();
        let mut saw_iend = false;
        while pos < png.len() {
            let len = u32::from_be_bytes(png[pos..pos + 4].try_into().unwrap()) as usize;
            let kind = &png[pos + 4..pos + 8];
            let data = &png[pos + 8..pos + 8 + len];
            let crc = u32::from_be_bytes(
                png[pos + 8 + len..pos + 12 + len].try_into().unwrap(),
            );
            let mut crc_input = kind.to_vec();
            crc_input.extend_from_slice(data);
            assert_eq!(crc, crc32(&crc_input), "chunk CRC must verify");
            if kind == b"IDAT" {
                idat.extend_from_slice(data);
            }
            if kind == b"IEND" {
                saw_iend = true;
            }
            pos += 12 + len;
        }
        assert!(saw_iend);
        assert_eq!(&idat[..2], &[0x78, 0x01]);
        let mut raw: Vec<u8> = Vec::new();
        let mut p = 2;
        loop {
            let flag = idat[p];
            let len = u16::from_le_bytes(idat[p + 1..p + 3].try_into().unwrap()) as usize;
            raw.extend_from_slice(&idat[p + 5..p + 5 + len]);
            p += 5 + len;
            if flag == 0x01 {
                break;
            }
        }
        let adler = u32::from_be_bytes(idat[p..p + 4].try_into().unwrap());
        assert_eq!(adler, adler32(&raw));
        // filtered scanlines: filter byte 0, then the pixels verbatim
        let mut expect: Vec<u8> = Vec::new();
        for y in 0..2 {
            expect.push(0);
            expect.extend_from_slice(&rgb[y * 12..(y + 1) * 12]);
        }
        assert_eq!(raw, expect);
    }

    #[test]
    fn srgb_endpoints_and_midpoint() {
        assert_eq!(linear_to_byte(0.0), 0);
        assert_eq!(linear_to_byte(1.0), 255);
        assert_eq!(linear_to_byte(-3.0), 0);
        assert_eq!(linear_to_byte(7.5), 255);
        // 0.5 linear -> 0.7354 sRGB -> 188 (the golden-test anchor)
        assert_eq!(linear_to_byte(0.5), 188);
        assert_eq!(linear_to_byte(0.25), 137);
    }
}
