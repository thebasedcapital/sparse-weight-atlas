//! Minimal .npy reader for the fixed little-endian C-order dtypes of the contract:
//! uint16 (`<u2`), int32 (`<i4`), int64 (`<i8`), float32 (`<f4`).

use anyhow::{bail, Context, Result};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub enum NpyData {
    U16(Vec<u16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
}

#[derive(Debug, Clone)]
pub struct Npy {
    pub shape: Vec<usize>,
    pub data: NpyData,
}

impl Npy {
    pub fn as_u16(&self) -> &[u16] {
        match &self.data {
            NpyData::U16(v) => v,
            _ => panic!("npy: not uint16"),
        }
    }

    pub fn as_i32(&self) -> &[i32] {
        match &self.data {
            NpyData::I32(v) => v,
            _ => panic!("npy: not int32"),
        }
    }

    pub fn as_i64(&self) -> &[i64] {
        match &self.data {
            NpyData::I64(v) => v,
            _ => panic!("npy: not int64"),
        }
    }

    pub fn as_f32(&self) -> &[f32] {
        match &self.data {
            NpyData::F32(v) => v,
            _ => panic!("npy: not float32"),
        }
    }
}

fn field_value<'a>(hdr: &'a str, key: &str) -> Option<&'a str> {
    let k = format!("'{key}'");
    let i = hdr.find(&k)? + k.len();
    let rest = &hdr[i..];
    let c = rest.find(':')? + 1;
    Some(rest[c..].trim_start())
}

fn parse_quoted(hdr: &str, key: &str) -> Result<String> {
    let v = field_value(hdr, key).with_context(|| format!("npy header: missing {key}"))?;
    let q = v.chars().next().with_context(|| format!("npy header: empty {key}"))?;
    if q != '\'' && q != '"' {
        bail!("npy header: {key} is not a quoted string");
    }
    let end = v[1..]
        .find(q)
        .with_context(|| format!("npy header: unterminated {key}"))?;
    Ok(v[1..1 + end].to_string())
}

fn parse_shape(hdr: &str) -> Result<Vec<usize>> {
    let v = field_value(hdr, "shape").context("npy header: missing shape")?;
    let l = v.find('(').context("npy header: bad shape")?;
    let r = v.find(')').context("npy header: bad shape")?;
    let mut out = Vec::new();
    for part in v[l + 1..r].split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        out.push(p.parse::<usize>().with_context(|| format!("npy header: bad dim {p:?}"))?);
    }
    Ok(out)
}

pub fn read_npy(path: &Path) -> Result<Npy> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        bail!("{}: not a .npy file", path.display());
    }
    let major = bytes[6];
    let (hlen, hoff) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize),
        2 | 3 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12usize,
        ),
        _ => bail!("{}: unsupported npy major version {}", path.display(), major),
    };
    let hdr = std::str::from_utf8(&bytes[hoff..hoff + hlen])
        .with_context(|| format!("{}: npy header is not utf-8", path.display()))?;
    let descr = parse_quoted(hdr, "descr")?;
    let fortran = field_value(hdr, "fortran_order").unwrap_or("False").starts_with("True");
    if fortran {
        bail!("{}: fortran_order arrays are not supported", path.display());
    }
    let shape = parse_shape(hdr)?;
    let n: usize = shape.iter().product();
    let body = &bytes[hoff + hlen..];
    let data = match descr.as_str() {
        "<u2" => {
            if body.len() < 2 * n {
                bail!("{}: truncated uint16 body", path.display());
            }
            NpyData::U16((0..n).map(|i| u16::from_le_bytes([body[2 * i], body[2 * i + 1]])).collect())
        }
        "<i4" => {
            if body.len() < 4 * n {
                bail!("{}: truncated int32 body", path.display());
            }
            NpyData::I32(
                (0..n)
                    .map(|i| i32::from_le_bytes([body[4 * i], body[4 * i + 1], body[4 * i + 2], body[4 * i + 3]]))
                    .collect(),
            )
        }
        "<i8" => {
            if body.len() < 8 * n {
                bail!("{}: truncated int64 body", path.display());
            }
            NpyData::I64(
                (0..n)
                    .map(|i| {
                        let mut b = [0u8; 8];
                        b.copy_from_slice(&body[8 * i..8 * i + 8]);
                        i64::from_le_bytes(b)
                    })
                    .collect(),
            )
        }
        "<f4" => {
            if body.len() < 4 * n {
                bail!("{}: truncated float32 body", path.display());
            }
            NpyData::F32(
                (0..n)
                    .map(|i| f32::from_le_bytes([body[4 * i], body[4 * i + 1], body[4 * i + 2], body[4 * i + 3]]))
                    .collect(),
            )
        }
        other => bail!("{}: unsupported dtype {other:?}", path.display()),
    };
    Ok(Npy { shape, data })
}
