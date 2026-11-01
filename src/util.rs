use anyhow::{Context, Result};
use serde::{Serialize, de::DeserializeOwned};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Layout de pastas do projeto.
pub struct Paths {
    pub raw_cot: PathBuf,
    pub raw_b3: PathBuf,
    pub raw_nefin: PathBuf,
    pub raw_bcb: PathBuf,
    pub processed: PathBuf,
    pub bars: PathBuf,
    pub panel: PathBuf,
    pub results: PathBuf,
    pub diag: PathBuf,
}

impl Paths {
    pub fn new(data: &Path, results: &Path) -> Self {
        let raw = data.join("raw");
        let processed = data.join("processed");
        Paths {
            raw_cot: raw.join("cotahist"),
            raw_b3: raw.join("b3"),
            raw_nefin: raw.join("nefin"),
            raw_bcb: raw.join("bcb"),
            bars: processed.join("bars.bin"),
            panel: processed.join("panel.bin"),
            processed,
            results: results.to_path_buf(),
            diag: results.join("diag"),
        }
    }

    pub fn ensure(&self) -> std::io::Result<()> {
        for d in [
            &self.raw_cot,
            &self.raw_b3,
            &self.raw_nefin,
            &self.raw_bcb,
            &self.processed,
            &self.results,
            &self.diag,
        ] {
            std::fs::create_dir_all(d)?;
        }
        Ok(())
    }
}

pub fn save_bin<T: Serialize>(v: &T, p: &Path) -> Result<()> {
    let w = BufWriter::new(File::create(p).with_context(|| format!("criando {}", p.display()))?);
    bincode::serialize_into(w, v)?;
    Ok(())
}

pub fn load_bin<T: DeserializeOwned>(p: &Path) -> Result<T> {
    let r = BufReader::new(
        File::open(p).with_context(|| format!("abrindo {} (rode a etapa anterior do pipeline)", p.display()))?,
    );
    Ok(bincode::deserialize_from(r)?)
}

// ---------- datas como u32 AAAAMMDD ----------

pub fn year(d: u32) -> u32 {
    d / 10000
}

pub fn ym(d: u32) -> u32 {
    d / 100
}

/// AAAAMM + k meses
pub fn ym_add(ym: u32, k: i32) -> u32 {
    let t = (ym / 100) as i32 * 12 + (ym % 100) as i32 - 1 + k;
    (t.div_euclid(12) * 100 + t.rem_euclid(12) + 1) as u32
}

/// dias desde 1970-01-01 -> (ano, mês, dia). Algoritmo de H. Hinnant.
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400;
    let (m, d) = (m as i64, d as i64);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

pub fn to_days(d: u32) -> i64 {
    days_from_civil((d / 10000) as i64, (d / 100) % 100, d % 100)
}

pub fn from_days(z: i64) -> u32 {
    let (y, m, d) = civil_from_days(z);
    (y as u32) * 10000 + m * 100 + d
}

pub fn add_days(d: u32, n: i64) -> u32 {
    from_days(to_days(d) + n)
}

pub fn systime_to_date(t: SystemTime) -> u32 {
    let s = t.duration_since(UNIX_EPOCH).map(|x| x.as_secs()).unwrap_or(0);
    from_days((s / 86_400) as i64)
}

pub fn today() -> u32 {
    systime_to_date(SystemTime::now())
}

/// "dd/mm/aaaa" -> AAAAMMDD (datas-sentinela tipo 31/12/9999 viram None)
pub fn parse_ddmmyyyy(s: &str) -> Option<u32> {
    let mut it = s.trim().split('/');
    let d: u32 = it.next()?.trim().parse().ok()?;
    let m: u32 = it.next()?.trim().parse().ok()?;
    let y: u32 = it.next()?.trim().get(0..4)?.parse().ok()?;
    if !(1900..=2100).contains(&y) || !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    Some(y * 10000 + m * 100 + d)
}

/// "aaaa-mm-dd" -> AAAAMMDD
pub fn parse_iso(s: &str) -> Option<u32> {
    let s = s.trim().trim_matches('"');
    let mut it = s.split('-');
    let y: u32 = it.next()?.parse().ok()?;
    let m: u32 = it.next()?.parse().ok()?;
    let d: u32 = it.next()?.get(0..2)?.parse().ok()?;
    Some(y * 10000 + m * 100 + d)
}

pub fn fmt_date(d: u32) -> String {
    format!("{:04}-{:02}-{:02}", d / 10000, (d / 100) % 100, d % 100)
}

pub fn fmt_ym(ym: u32) -> String {
    format!("{:04}-{:02}", ym / 100, ym % 100)
}

/// Número no formato da B3 ("1.234,56"). Sem vírgula, lê no formato padrão.
pub fn parse_br_num(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let t: String = if s.contains(',') {
        s.chars()
            .filter(|&c| c != '.' && c != ' ')
            .map(|c| if c == ',' { '.' } else { c })
            .collect()
    } else {
        s.to_string()
    };
    t.parse::<f64>().ok().filter(|v| v.is_finite())
}

/// Nome de arquivo seguro + hash curto (evita colisão entre nomes parecidos).
pub fn slug(s: &str) -> String {
    let base: String = s
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{base}_{:08x}", (h & 0xffff_ffff) as u32)
}

pub fn pct(x: f64) -> String {
    if x.is_finite() { format!("{:+.1}%", x * 100.0) } else { "n/d".into() }
}
