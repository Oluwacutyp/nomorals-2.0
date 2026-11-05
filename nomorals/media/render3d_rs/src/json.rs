//! A small order-preserving JSON parser — the only reader the
//! renderer needs. Objects keep insertion order (a `Vec` of pairs,
//! never a hash map): the Scene3D frame state's node order is the
//! document order, and draw order is this renderer's depth-tie
//! rule, so order is a determinism input, not a detail.

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Value>),
    Obj(Vec<(String, Value)>),
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Num(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&Vec<Value>> {
        match self {
            Value::Arr(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_obj(&self) -> Option<&Vec<(String, Value)>> {
        match self {
            Value::Obj(pairs) => Some(pairs),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// A JSON array of numbers as a Vec3 (None unless exactly 3).
    pub fn as_vec3(&self) -> Option<[f64; 3]> {
        let items = self.as_arr()?;
        if items.len() != 3 {
            return None;
        }
        Some([
            items[0].as_f64()?,
            items[1].as_f64()?,
            items[2].as_f64()?,
        ])
    }

    /// A JSON array of numbers as a row-major 4x4 matrix.
    pub fn as_mat4(&self) -> Option<[f64; 16]> {
        let items = self.as_arr()?;
        if items.len() != 16 {
            return None;
        }
        let mut out = [0.0f64; 16];
        for (i, item) in items.iter().enumerate() {
            out[i] = item.as_f64()?;
        }
        Some(out)
    }
}

pub fn parse(input: &str) -> Result<Value, String> {
    let bytes = input.as_bytes();
    let mut parser = Parser { bytes, pos: 0 };
    parser.skip_ws();
    let value = parser.parse_value()?;
    parser.skip_ws();
    if parser.pos != bytes.len() {
        return Err(format!(
            "trailing input at byte {} of the JSON document",
            parser.pos
        ));
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len()
            && matches!(self.bytes[self.pos], b' ' | b'\t' | b'\n' | b'\r')
        {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn parse_value(&mut self) -> Result<Value, String> {
        match self.peek() {
            Some(b'{') => self.parse_obj(),
            Some(b'[') => self.parse_arr(),
            Some(b'"') => Ok(Value::Str(self.parse_string()?)),
            Some(b't') => self.parse_lit("true", Value::Bool(true)),
            Some(b'f') => self.parse_lit("false", Value::Bool(false)),
            Some(b'n') => self.parse_lit("null", Value::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_num(),
            _ => Err(format!("unexpected input at byte {}", self.pos)),
        }
    }

    fn parse_lit(&mut self, lit: &str, value: Value) -> Result<Value, String> {
        if self.bytes[self.pos..].starts_with(lit.as_bytes()) {
            self.pos += lit.len();
            Ok(value)
        } else {
            Err(format!("invalid literal at byte {}", self.pos))
        }
    }

    fn parse_num(&mut self) -> Result<Value, String> {
        let start = self.pos;
        while self.pos < self.bytes.len()
            && matches!(self.bytes[self.pos], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
        {
            self.pos += 1;
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| "invalid number encoding".to_string())?;
        text.parse::<f64>()
            .map(Value::Num)
            .map_err(|_| format!("invalid number {text:?} at byte {start}"))
    }

    fn parse_string(&mut self) -> Result<String, String> {
        // caller has peeked the opening quote
        self.pos += 1;
        let mut out = String::new();
        loop {
            let c = self
                .peek()
                .ok_or_else(|| "unterminated string".to_string())?;
            self.pos += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let esc = self
                        .peek()
                        .ok_or_else(|| "unterminated escape".to_string())?;
                    self.pos += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.parse_hex4()?;
                            if (0xD800..0xDC00).contains(&hi) {
                                // high surrogate: a \uDC00..\uDFFF must follow
                                if self.peek() == Some(b'\\') {
                                    self.pos += 1;
                                    if self.peek() == Some(b'u') {
                                        self.pos += 1;
                                        let lo = self.parse_hex4()?;
                                        let cp = 0x10000
                                            + (((hi - 0xD800) as u32) << 10)
                                            + (lo.wrapping_sub(0xDC00) as u32);
                                        out.push(
                                            char::from_u32(cp)
                                                .unwrap_or('\u{FFFD}'),
                                        );
                                        continue;
                                    }
                                }
                                out.push('\u{FFFD}');
                            } else {
                                out.push(
                                    char::from_u32(hi as u32)
                                        .unwrap_or('\u{FFFD}'),
                                );
                            }
                        }
                        _ => {
                            return Err(format!(
                                "invalid escape at byte {}",
                                self.pos - 1
                            ))
                        }
                    }
                }
                _ => {
                    // raw byte: copy the full UTF-8 sequence through
                    let start = self.pos - 1;
                    let len = utf8_len(c);
                    let end = (start + len).min(self.bytes.len());
                    let text = std::str::from_utf8(&self.bytes[start..end])
                        .map_err(|_| "invalid UTF-8 in string".to_string())?;
                    out.push_str(text);
                    self.pos = end;
                }
            }
        }
    }

    fn parse_hex4(&mut self) -> Result<u16, String> {
        if self.pos + 4 > self.bytes.len() {
            return Err("truncated \\u escape".to_string());
        }
        let text = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4])
            .map_err(|_| "invalid \\u escape".to_string())?;
        self.pos += 4;
        u16::from_str_radix(text, 16)
            .map_err(|_| format!("invalid \\u escape {text:?}"))
    }

    fn parse_arr(&mut self) -> Result<Value, String> {
        self.pos += 1; // [
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Value::Arr(items));
        }
        loop {
            self.skip_ws();
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Value::Arr(items));
                }
                _ => return Err(format!("expected , or ] at byte {}", self.pos)),
            }
        }
    }

    fn parse_obj(&mut self) -> Result<Value, String> {
        self.pos += 1; // {
        let mut pairs = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Value::Obj(pairs));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(format!("expected object key at byte {}", self.pos));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(format!("expected : at byte {}", self.pos));
            }
            self.pos += 1;
            self.skip_ws();
            let value = self.parse_value()?;
            pairs.push((key, value));
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Value::Obj(pairs));
                }
                _ => return Err(format!("expected , or }} at byte {}", self.pos)),
            }
        }
    }
}

fn utf8_len(lead: u8) -> usize {
    if lead < 0x80 {
        1
    } else if lead >> 5 == 0b110 {
        2
    } else if lead >> 4 == 0b1110 {
        3
    } else {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nested_and_preserves_order() {
        let value = parse(r#"{"b": 1, "a": [true, null, -2.5e2], "s": "xé"}"#)
            .expect("parse");
        let obj = value.as_obj().expect("obj");
        let keys: Vec<&str> = obj.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["b", "a", "s"]);
        assert_eq!(value.get("a").unwrap().as_arr().unwrap()[2].as_f64(), Some(-250.0));
    }

    #[test]
    fn string_escapes_and_surrogates() {
        let value = parse(r#""a\nbé😀""#).expect("parse");
        assert_eq!(value.as_str(), Some("a\nbé😀"));
    }

    #[test]
    fn rejects_garbage_with_offsets() {
        assert!(parse("{").is_err());
        assert!(parse("[1,]").is_err());
        assert!(parse("1 2").is_err());
        assert!(parse("").is_err());
    }

    #[test]
    fn vec3_and_mat4_shapes() {
        assert_eq!(parse("[1, 2, 3]").unwrap().as_vec3(), Some([1.0, 2.0, 3.0]));
        assert_eq!(parse("[1, 2]").unwrap().as_vec3(), None);
        let m = parse("[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]").unwrap();
        assert_eq!(m.as_mat4().unwrap()[0], 1.0);
    }
}
