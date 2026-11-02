//! The holdout ledger (`results/holdout_ledger.csv`, M5). It outlives runs and records:
//!
//! - `seen`: the last month of data that Kairós programs have already seen;
//! - `protocol`: a protocol registered (by hash) the first time a program declares it. Its
//!   holdout is reserved: while it is pending, no program sees those months;
//! - `opened`: a holdout opened by `open_once`, with the months it saw and its result;
//! - `withdrawn`: a pending protocol given up before opening (`kairos ledger withdraw`).
//!
//! Rules: a holdout opens once; a new protocol's holdout must start after every month already
//! seen (the *watermark*); pending holdouts fence the data of every program.
//!
//! The ledger keeps honest people honest: it is a plain file that its owner can edit, and it
//! only knows what Kairós programs saw, not what was looked at with other tools.

use std::path::{Path, PathBuf};

pub const FILE: &str = "holdout_ledger.csv";
const HEADER: &str = "kind,protocol,sha256,holdout_from,data_until,date,note";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Seen,
    Protocol,
    Opened,
    Withdrawn,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Seen => "seen",
            Kind::Protocol => "protocol",
            Kind::Opened => "opened",
            Kind::Withdrawn => "withdrawn",
        }
    }
    fn parse(s: &str) -> Option<Kind> {
        Some(match s {
            "seen" => Kind::Seen,
            "protocol" => Kind::Protocol,
            "opened" => Kind::Opened,
            "withdrawn" => Kind::Withdrawn,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub kind: Kind,
    /// protocol name (empty for `seen`)
    pub protocol: String,
    /// protocol hash (empty for `seen`)
    pub sha256: String,
    /// first month of the holdout, YYYYMM (0 for `seen`)
    pub holdout_from: u32,
    /// last month of data seen, YYYYMM (0 for `protocol` and `withdrawn`)
    pub data_until: u32,
    /// YYYYMMDD
    pub date: u32,
    pub note: String,
}

impl Row {
    pub fn new(kind: Kind, protocol: &str, sha256: &str, holdout_from: u32, data_until: u32, date: u32, note: &str) -> Row {
        Row { kind, protocol: protocol.into(), sha256: sha256.into(), holdout_from, data_until, date, note: note.into() }
    }

    /// Who saw the data, for messages: `the opening of "x" on 2026-11-01`.
    pub fn who(&self) -> String {
        match self.kind {
            Kind::Seen => format!("a Kairós program on {} ({})", fmt_date(self.date), self.note),
            Kind::Opened => format!("the opening of the holdout of \"{}\" on {}", self.protocol, fmt_date(self.date)),
            Kind::Protocol | Kind::Withdrawn => format!("protocol \"{}\"", self.protocol),
        }
    }
}

pub fn fmt_ym(ym: u32) -> String {
    if ym == 0 { String::new() } else { format!("{:04}-{:02}", ym / 100, ym % 100) }
}

pub fn fmt_date(d: u32) -> String {
    format!("{:04}-{:02}-{:02}", d / 10000, d / 100 % 100, d % 100)
}

/// `2019-01` (or empty: 0) as YYYYMM.
pub fn parse_ym(s: &str) -> Option<u32> {
    if s.is_empty() {
        return Some(0);
    }
    let (y, m) = s.split_once('-')?;
    let (y, m): (u32, u32) = (y.parse().ok()?, m.parse().ok()?);
    ((1..=12).contains(&m) && y >= 1000).then_some(y * 100 + m)
}

fn parse_date(s: &str) -> Option<u32> {
    let mut it = s.split('-');
    let (y, m, d): (u32, u32, u32) = (it.next()?.parse().ok()?, it.next()?.parse().ok()?, it.next()?.parse().ok()?);
    (it.next().is_none() && (1..=12).contains(&m) && (1..=31).contains(&d)).then_some(y * 10000 + m * 100 + d)
}

/// The month after `ym` (YYYYMM).
pub fn next_month(ym: u32) -> u32 {
    if ym % 100 == 12 { (ym / 100 + 1) * 100 + 1 } else { ym + 1 }
}

/// The month before `ym` (YYYYMM).
pub fn prev_month(ym: u32) -> u32 {
    if ym % 100 == 1 { (ym / 100 - 1) * 100 + 12 } else { ym - 1 }
}

fn clean(s: &str) -> String {
    s.replace([',', '\n', '\r'], ";")
}

pub fn render(rows: &[Row]) -> String {
    let mut s = String::from(HEADER);
    s.push('\n');
    for r in rows {
        s.push_str(&format!(
            "{},{},{},{},{},{},{}\n",
            r.kind.name(),
            clean(&r.protocol),
            clean(&r.sha256),
            fmt_ym(r.holdout_from),
            fmt_ym(r.data_until),
            fmt_date(r.date),
            clean(&r.note)
        ));
    }
    s
}

pub fn parse(text: &str) -> Result<Vec<Row>, String> {
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.splitn(7, ',').collect();
        let bad = || format!("line {}: `{line}`", i + 1);
        if f.len() != 7 {
            return Err(bad());
        }
        let kind = Kind::parse(f[0]).ok_or_else(bad)?;
        let holdout_from = parse_ym(f[3]).ok_or_else(bad)?;
        let data_until = parse_ym(f[4]).ok_or_else(bad)?;
        let date = parse_date(f[5]).ok_or_else(bad)?;
        rows.push(Row { kind, protocol: f[1].into(), sha256: f[2].into(), holdout_from, data_until, date, note: f[6].into() });
    }
    Ok(rows)
}

/// The ledger: in a file (`path`) or, for tests and embedding, only in memory.
#[derive(Debug, Default)]
pub struct Ledger {
    pub rows: Vec<Row>,
    path: Option<PathBuf>,
}

impl Ledger {
    pub fn in_memory(rows: Vec<Row>) -> Ledger {
        Ledger { rows, path: None }
    }

    /// Reads the ledger at `path`. A missing file starts with `seed()` (the holdouts opened before
    /// the ledger existed), written on the first change.
    pub fn open(path: &Path, seed: impl FnOnce() -> Vec<Row>) -> Result<Ledger, String> {
        let rows = match std::fs::read_to_string(path) {
            Ok(txt) => parse(&txt).map_err(|e| format!("cannot read the holdout ledger {}: {e}", path.display()))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => seed(),
            Err(e) => return Err(format!("cannot read the holdout ledger {}: {e}", path.display())),
        };
        Ok(Ledger { rows, path: Some(path.to_path_buf()) })
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    fn save(&self) -> Result<(), String> {
        let Some(p) = &self.path else { return Ok(()) };
        if let Some(dir) = p.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
            }
        }
        std::fs::write(p, render(&self.rows)).map_err(|e| format!("cannot write the holdout ledger {}: {e}", p.display()))
    }

    pub fn push(&mut self, r: Row) -> Result<(), String> {
        self.rows.push(r);
        self.save()
    }

    /// Replaces the note of the opening of `sha` (its result, once known).
    pub fn set_opened_note(&mut self, sha: &str, note: &str) -> Result<(), String> {
        if let Some(r) = self.rows.iter_mut().rev().find(|r| r.kind == Kind::Opened && r.sha256 == sha) {
            r.note = note.into();
        }
        self.save()
    }

    /// The last month any Kairós program (or an opening) has seen; 0 if none.
    pub fn watermark(&self) -> u32 {
        self.rows.iter().filter(|r| matches!(r.kind, Kind::Seen | Kind::Opened)).map(|r| r.data_until).max().unwrap_or(0)
    }

    /// The first record that saw month `ym`.
    pub fn seen_by(&self, ym: u32) -> Option<&Row> {
        self.rows.iter().find(|r| matches!(r.kind, Kind::Seen | Kind::Opened) && r.data_until >= ym)
    }

    pub fn find(&self, kind: Kind, sha: &str) -> Option<&Row> {
        self.rows.iter().find(|r| r.kind == kind && r.sha256 == sha)
    }

    /// Protocols whose holdout is still reserved: registered, neither opened nor withdrawn, and
    /// not spoiled (no month of the holdout was seen).
    pub fn pending(&self) -> Vec<&Row> {
        let w = self.watermark();
        self.rows
            .iter()
            .filter(|r| {
                r.kind == Kind::Protocol
                    && r.holdout_from > w
                    && self.find(Kind::Opened, &r.sha256).is_none()
                    && self.find(Kind::Withdrawn, &r.sha256).is_none()
            })
            .collect()
    }

    /// The last month programs may see while the pending holdouts stay reserved, and the
    /// protocol that sets it.
    pub fn fence(&self) -> Option<(u32, &Row)> {
        self.pending().into_iter().min_by_key(|r| r.holdout_from).map(|r| (prev_month(r.holdout_from), r))
    }

    /// Registers a protocol (idempotent). A new holdout must start after the watermark: months
    /// already seen cannot be a holdout.
    pub fn register(&mut self, name: &str, sha: &str, holdout_from: u32, today: u32, rule: &str) -> Result<(), String> {
        if self.find(Kind::Protocol, sha).is_some() {
            return Ok(());
        }
        let w = self.watermark();
        if holdout_from <= w {
            let who = self.seen_by(holdout_from).map(Row::who).unwrap_or_default();
            return Err(format!(
                "the holdout of protocol \"{name}\" starts in {}, but the data up to {} was already seen by {who}; \
                 a new holdout must start in {} or later",
                fmt_ym(holdout_from),
                fmt_ym(w),
                fmt_ym(next_month(w))
            ));
        }
        self.push(Row::new(Kind::Protocol, name, sha, holdout_from, 0, today, rule))
    }

    /// Records that a program saw the data up to `until` (only when it moves the watermark).
    pub fn saw(&mut self, until: u32, today: u32, who: &str) -> Result<(), String> {
        if until > self.watermark() {
            self.push(Row::new(Kind::Seen, "", "", 0, until, today, who))?;
        }
        Ok(())
    }

    /// Gives up a pending protocol: its months stop being fenced (and can never be its holdout).
    pub fn withdraw(&mut self, which: &str, today: u32) -> Result<String, String> {
        let pending: Vec<Row> =
            self.pending().into_iter().filter(|r| r.protocol == which || (which.len() >= 8 && r.sha256.starts_with(which))).cloned().collect();
        match pending.as_slice() {
            [] => Err(format!("no pending protocol named \"{which}\" (see `kairos ledger`)")),
            [r] => {
                self.push(Row::new(Kind::Withdrawn, &r.protocol, &r.sha256, r.holdout_from, 0, today, "withdrawn before opening"))?;
                Ok(format!("protocol \"{}\" ({}) withdrawn: its holdout from {} is no longer reserved", r.protocol, &r.sha256[..8], fmt_ym(r.holdout_from)))
            }
            _ => Err(format!("more than one pending protocol is named \"{which}\": give the start of its hash instead")),
        }
    }

    /// Text for `kairos ledger`.
    pub fn describe(&self) -> String {
        let mut s = String::new();
        if let Some(p) = &self.path {
            s.push_str(&format!("holdout ledger: {}\n", p.display()));
        }
        if self.rows.is_empty() {
            s.push_str("empty: no Kairós program has seen the data yet\n");
            return s;
        }
        for r in &self.rows {
            let what = match r.kind {
                Kind::Seen => format!("saw data up to {}", fmt_ym(r.data_until)),
                Kind::Protocol => format!("protocol \"{}\" {} reserves the holdout from {}", r.protocol, short(&r.sha256), fmt_ym(r.holdout_from)),
                Kind::Opened => format!(
                    "opened the holdout of \"{}\" {} ({}..{})",
                    r.protocol,
                    short(&r.sha256),
                    fmt_ym(r.holdout_from),
                    fmt_ym(r.data_until)
                ),
                Kind::Withdrawn => format!("withdrew protocol \"{}\" {}", r.protocol, short(&r.sha256)),
            };
            let note = if r.note.is_empty() { String::new() } else { format!(" — {}", r.note) };
            s.push_str(&format!("  {}  {what}{note}\n", fmt_date(r.date)));
        }
        let w = self.watermark();
        if w > 0 {
            s.push_str(&format!("months seen up to {}: a new holdout must start in {} or later\n", fmt_ym(w), fmt_ym(next_month(w))));
        }
        match self.fence() {
            Some((f, r)) => s.push_str(&format!(
                "data visible to programs: up to {} (the holdout of \"{}\" is reserved)\n",
                fmt_ym(f),
                r.protocol
            )),
            None => s.push_str("no pending holdout: programs see all the data\n"),
        }
        s
    }
}

fn short(sha: &str) -> String {
    format!("[{}]", sha.get(..8).unwrap_or(sha))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lab_seed() -> Vec<Row> {
        vec![Row::new(Kind::Opened, "busca-de-alpha (quant-b3)", "e83eb427", 201901, 202610, 20261101, "nenhum alpha aprovado")]
    }

    #[test]
    fn months() {
        assert_eq!(next_month(201812), 201901);
        assert_eq!(next_month(201901), 201902);
        assert_eq!(prev_month(201901), 201812);
        assert_eq!(prev_month(201912), 201911);
    }

    #[test]
    fn round_trip() {
        let mut rows = lab_seed();
        rows.push(Row::new(Kind::Seen, "", "", 0, 202611, 20261205, "program momentum, with a comma"));
        rows.push(Row::new(Kind::Protocol, "p,1", "abcdef0123", 202701, 0, 20261205, "accept if alpha > 0"));
        let txt = render(&rows);
        let back = parse(&txt).unwrap();
        assert_eq!(back.len(), 3);
        assert_eq!(back[0], rows[0]);
        assert_eq!(back[1].note, "program momentum; with a comma");
        assert_eq!(back[2].protocol, "p;1");
        assert_eq!(back[2].holdout_from, 202701);
        assert!(parse("kind\nnope,a,b,,,2026-01-01,x\n").is_err());
        assert!(parse("kind\nseen,,,,2026-13,2026-01-01,x\n").is_err());
    }

    #[test]
    fn a_new_holdout_starts_after_everything_seen() {
        let mut l = Ledger::in_memory(lab_seed());
        assert_eq!(l.watermark(), 202610);
        // the lab's search opened 2019+: a holdout in that period is refused
        let e = l.register("p", "aaaa", 201901, 20261102, "").unwrap_err();
        assert!(e.contains("2026-11 or later"), "{e}");
        assert!(e.contains("busca-de-alpha"), "{e}");
        l.register("p", "aaaa", 202611, 20261102, "").unwrap();
        // idempotent
        l.register("p", "aaaa", 202611, 20261103, "").unwrap();
        assert_eq!(l.rows.len(), 2);
        // pending: the data is fenced at 2026-10
        assert_eq!(l.fence().map(|f| f.0), Some(202610));
        // seeing data within the fence does not spoil it
        l.saw(202610, 20261103, "x").unwrap();
        assert_eq!(l.rows.len(), 2);
        assert_eq!(l.pending().len(), 1);
    }

    #[test]
    fn opening_spoils_overlapping_reservations_and_withdrawal_frees_months() {
        let mut l = Ledger::in_memory(lab_seed());
        l.register("p", "aaaa", 202611, 20261102, "").unwrap();
        l.register("q", "bbbb", 202701, 20261102, "").unwrap();
        assert_eq!(l.fence().map(|f| f.0), Some(202610));
        // opening p sees up to 2027-12: q's holdout is spoiled, nothing stays reserved
        l.push(Row::new(Kind::Opened, "p", "aaaa", 202611, 202712, 20280105, "")).unwrap();
        assert!(l.pending().is_empty());
        assert_eq!(l.fence(), None);
        assert_eq!(l.watermark(), 202712);
        // withdrawing
        let mut l = Ledger::in_memory(lab_seed());
        l.register("p", "aaaa1111", 202611, 20261102, "").unwrap();
        assert!(l.withdraw("nope", 20261103).is_err());
        l.withdraw("p", 20261103).unwrap();
        assert_eq!(l.fence(), None);
        assert!(l.withdraw("p", 20261104).is_err());
    }
}
