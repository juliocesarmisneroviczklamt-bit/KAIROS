//! Download e leitura das séries históricas COTAHIST da B3 (layout de largura fixa, 245 colunas).

use crate::http;
use crate::store::{Bar, BarStore, Series};
use crate::util;
use anyhow::{Context, Result, anyhow};
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

/// Códigos BDI aceitos. Além do lote padrão (02) entram as situações especiais (recuperação
/// judicial etc.), senão a ação some da base justamente quando está morrendo.
const ALLOWED_BDI: [u8; 8] = [2, 5, 6, 7, 8, 9, 11, 58];
/// Classes de ação, pelo primeiro token do campo ESPECI. Ficam de fora FII, ETF (CI), BDR,
/// direitos, recibos e bônus.
const ALLOWED_CLASS: [&str; 11] = [
    "ON", "PN", "PNA", "PNB", "PNC", "PND", "PNE", "PNF", "PNG", "PNH", "UNT",
];
/// ETFs guardados só como referência (e ativo da estratégia de tendência no paper trading).
pub const BENCH: [&str; 1] = ["BOVA11"];

pub fn fetch(dir: &Path, from: u32, to: Option<u32>, threads: usize) -> Result<()> {
    let today = util::today();
    let to = to.unwrap_or(util::year(today));
    let agent = http::agent()?;
    let years: Vec<u32> = (from..=to).collect();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads.max(1)).build()?;
    println!("COTAHIST {from}..{to} -> {}", dir.display());
    let errs: Vec<String> = pool.install(|| {
        years
            .par_iter()
            .filter_map(|&y| {
                let dest = dir.join(format!("COTAHIST_A{y}.ZIP"));
                if is_complete(&dest, y, today) {
                    return None;
                }
                let url = format!("https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{y}.ZIP");
                match http::download(&agent, &url, &dest).and_then(|n| check_zip(&dest).map(|_| n)) {
                    Ok(n) => {
                        println!("  {y}: {:.1} MB", n as f64 / 1e6);
                        None
                    }
                    Err(e) => Some(format!("{y}: {e:#}")),
                }
            })
            .collect()
    });
    if !errs.is_empty() {
        for e in &errs {
            eprintln!("  ERRO {e}");
        }
        return Err(anyhow!("{} downloads falharam", errs.len()));
    }
    Ok(())
}

/// Arquivo de ano fechado baixado depois do fim do ano, ou do ano corrente baixado hoje.
fn is_complete(p: &Path, y: u32, today: u32) -> bool {
    let Ok(meta) = p.metadata() else { return false };
    if meta.len() < 1000 {
        return false;
    }
    let Ok(m) = meta.modified() else { return false };
    let d = util::systime_to_date(m);
    if y < util::year(today) { d > y * 10000 + 1231 } else { d >= today }
}

fn check_zip(p: &Path) -> Result<()> {
    let mut f = std::fs::File::open(p)?;
    let mut b = [0u8; 2];
    f.read_exact(&mut b)?;
    if &b != b"PK" {
        let _ = std::fs::remove_file(p);
        return Err(anyhow!("não é ZIP: {}", p.display()));
    }
    Ok(())
}

pub struct Rec {
    pub ticker: String,
    pub name: String,
    pub isin: String,
    pub class: String,
    pub bar: Bar,
}

#[derive(Default)]
pub struct FileStats {
    pub lines: u64,
    pub accepted: u64,
    pub rejected_class: HashMap<String, u64>,
    pub isin_types: HashMap<String, u64>,
}

#[inline]
fn num(b: &[u8]) -> u64 {
    let mut v: u64 = 0;
    for &c in b {
        if c.is_ascii_digit() {
            v = v * 10 + (c - b'0') as u64;
        }
    }
    v
}

fn latin1(b: &[u8]) -> String {
    b.iter().map(|&c| c as char).collect::<String>().trim().to_string()
}

/// Lê um ZIP anual e devolve só ações/units do mercado à vista (TPMERC 010) cotadas em R$.
pub fn parse_zip(path: &Path) -> Result<(Vec<Rec>, FileStats)> {
    let f = std::fs::File::open(path).with_context(|| format!("abrindo {}", path.display()))?;
    let mut z = zip::ZipArchive::new(f).with_context(|| format!("zip inválido {}", path.display()))?;
    let entry = z.by_index(0)?;
    let mut rd = BufReader::with_capacity(1 << 20, entry);
    let mut buf: Vec<u8> = Vec::with_capacity(512);
    let mut recs = Vec::new();
    let mut st = FileStats::default();
    loop {
        buf.clear();
        if rd.read_until(b'\n', &mut buf)? == 0 {
            break;
        }
        st.lines += 1;
        let l = buf.as_slice();
        // TIPREG=01, TPMERC=010 (vista), MODREF=R$
        if l.len() < 245 || &l[0..2] != b"01" || &l[24..27] != b"010" || &l[52..54] != b"R$" {
            continue;
        }
        let bdi = num(&l[10..12]) as u8;
        let ticker = latin1(&l[12..24]);
        let bench = BENCH.contains(&ticker.as_str());
        if !bench && !ALLOWED_BDI.contains(&bdi) {
            continue;
        }
        let especi = latin1(&l[39..49]);
        // nos anos 90 o marcador vinha grudado na classe: "PNA*", "PNB*ED"
        let class = especi.split_whitespace().next().unwrap_or("").split('*').next().unwrap_or("").to_string();
        if !bench && !ALLOWED_CLASS.contains(&class.as_str()) {
            if bdi == 2 {
                *st.rejected_class.entry(class).or_default() += 1;
            }
            continue;
        }
        if num(&l[108..121]) == 0 {
            continue;
        }
        let fat = num(&l[210..217]).max(1) as f64;
        let px = |a: usize, b: usize| num(&l[a..b]) as f64 / 100.0 / fat;
        let isin = latin1(&l[230..242]);
        *st.isin_types.entry(isin.get(6..9).unwrap_or("").to_string()).or_default() += 1;
        st.accepted += 1;
        recs.push(Rec {
            ticker,
            name: latin1(&l[27..39]),
            isin,
            class,
            bar: Bar {
                date: num(&l[2..10]) as u32,
                open: px(56, 69),
                high: px(69, 82),
                low: px(82, 95),
                avg: px(95, 108),
                close: px(108, 121),
                bid: px(121, 134),
                ask: px(134, 147),
                trades: num(&l[147..152]) as u32,
                qty: num(&l[152..170]) as f64,
                vol_brl: num(&l[170..188]) as f64 / 100.0,
                fatcot: fat as u32,
                bdi,
                especi,
            },
        });
    }
    Ok((recs, st))
}

pub fn ingest(raw_dir: &Path) -> Result<BarStore> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(raw_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|x| x.to_str())
                .map(|x| x.eq_ignore_ascii_case("zip"))
                .unwrap_or(false)
        })
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(anyhow!("nenhum COTAHIST em {} (rode fetch-cotahist)", raw_dir.display()));
    }
    println!("Lendo {} arquivos COTAHIST em paralelo...", files.len());
    let parsed: Vec<(PathBuf, Result<(Vec<Rec>, FileStats)>)> =
        files.par_iter().map(|p| (p.clone(), parse_zip(p))).collect();

    struct Acc {
        isins: Vec<String>,
        names: Vec<(String, u32, u32)>,
        class_n: HashMap<String, u32>,
        bars: Vec<Bar>,
    }
    let mut acc: HashMap<String, Acc> = HashMap::new();
    let mut rej: HashMap<String, u64> = HashMap::new();
    let mut isin_t: HashMap<String, u64> = HashMap::new();
    let mut total = 0u64;
    for (path, res) in parsed {
        let (recs, st) = res.with_context(|| format!("lendo {}", path.display()))?;
        let fname = path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
        println!("  {fname:<22} linhas={:>9}  ações/units={:>7}", st.lines, st.accepted);
        total += st.accepted;
        for (k, v) in st.rejected_class {
            *rej.entry(k).or_default() += v;
        }
        for (k, v) in st.isin_types {
            *isin_t.entry(k).or_default() += v;
        }
        for r in recs {
            let a = acc.entry(r.ticker).or_insert_with(|| Acc {
                isins: Vec::new(),
                names: Vec::new(),
                class_n: HashMap::new(),
                bars: Vec::new(),
            });
            if !a.isins.contains(&r.isin) {
                a.isins.push(r.isin);
            }
            let d = r.bar.date;
            if let Some(e) = a.names.iter_mut().find(|e| e.0 == r.name) {
                e.1 = e.1.min(d);
                e.2 = e.2.max(d);
            } else {
                a.names.push((r.name, d, d));
            }
            *a.class_n.entry(r.class).or_default() += 1;
            a.bars.push(r.bar);
        }
    }

    let series: Vec<Series> = acc
        .into_iter()
        .map(|(ticker, mut a)| {
            a.bars.sort_by_key(|b| b.date);
            let mut bars: Vec<Bar> = Vec::with_capacity(a.bars.len());
            for b in a.bars {
                if bars.last().map(|l| l.date) == Some(b.date) {
                    let l = bars.last_mut().unwrap();
                    if b.vol_brl > l.vol_brl {
                        *l = b;
                    }
                } else {
                    bars.push(b);
                }
            }
            let class = a
                .class_n
                .into_iter()
                .max_by_key(|(_, n)| *n)
                .map(|(c, _)| c)
                .unwrap_or_default();
            a.names.sort_by_key(|x| x.1);
            Series { ticker, aliases: Vec::new(), isins: a.isins, names: a.names, class, bars }
        })
        .collect();

    let (series, merges) = chain_renames(series);
    let nbars: usize = series.iter().map(|s| s.bars.len()).sum();
    println!(
        "Total: {total} registros -> {} séries, {nbars} pregões ({} trocas de código encadeadas por ISIN)",
        series.len(),
        merges.len()
    );
    for m in merges.iter().take(12) {
        println!("  {m}");
    }
    let mut rv: Vec<_> = rej.into_iter().collect();
    rv.sort_by(|a, b| b.1.cmp(&a.1));
    println!(
        "Classes rejeitadas (BDI 02): {}",
        rv.iter().take(15).map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(" ")
    );
    let mut iv: Vec<_> = isin_t.into_iter().collect();
    iv.sort_by(|a, b| b.1.cmp(&a.1));
    println!(
        "Tipos de ISIN aceitos: {}",
        iv.iter().take(15).map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(" ")
    );
    Ok(BarStore { series })
}

/// Tickers que compartilham ISIN, sem sobreposição de datas e com intervalo curto entre um e
/// outro, são a mesma ação que trocou de código (ex.: VVAR3 -> VIIA3). Junta numa série só.
fn chain_renames(series: Vec<Series>) -> (Vec<Series>, Vec<String>) {
    fn find(p: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while p[r] != r {
            r = p[r];
        }
        let mut y = x;
        while p[y] != r {
            let nx = p[y];
            p[y] = r;
            y = nx;
        }
        r
    }
    let n = series.len();
    let mut parent: Vec<usize> = (0..n).collect();
    let mut pairs = Vec::new();
    {
        let mut by_isin: HashMap<&str, Vec<usize>> = HashMap::new();
        for (i, s) in series.iter().enumerate() {
            for isin in &s.isins {
                if isin.len() == 12 && isin.starts_with("BR") {
                    by_isin.entry(isin.as_str()).or_default().push(i);
                }
            }
        }
        for (_, mut v) in by_isin {
            if v.len() < 2 {
                continue;
            }
            v.sort_by_key(|&i| series[i].first());
            for w in v.windows(2) {
                let (a, b) = (w[0], w[1]);
                let (la, fb) = (series[a].last(), series[b].first());
                if la < fb && util::to_days(fb) - util::to_days(la) <= 45 {
                    pairs.push((a, b));
                }
            }
        }
    }
    for (a, b) in pairs {
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra != rb {
            parent[rb] = ra;
        }
    }
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().push(i);
    }
    let mut slots: Vec<Option<Series>> = series.into_iter().map(Some).collect();
    let mut out = Vec::with_capacity(n);
    let mut log = Vec::new();
    for (_, mut idx) in groups {
        if idx.len() == 1 {
            out.push(slots[idx[0]].take().unwrap());
            continue;
        }
        idx.sort_by_key(|&i| slots[i].as_ref().map(|s| s.first()).unwrap_or(0));
        let mut segs: Vec<Series> = idx.iter().map(|&i| slots[i].take().unwrap()).collect();
        log.push(segs.iter().map(|s| s.ticker.as_str()).collect::<Vec<_>>().join(" -> "));
        let mut merged = segs.pop().unwrap(); // o segmento mais recente dá o código
        let mut bars: Vec<Bar> = Vec::new();
        for s in segs {
            merged.aliases.push(s.ticker);
            merged.aliases.extend(s.aliases);
            for i in s.isins {
                if !merged.isins.contains(&i) {
                    merged.isins.push(i);
                }
            }
            merged.names.extend(s.names);
            bars.extend(s.bars);
        }
        bars.append(&mut merged.bars);
        bars.sort_by_key(|b| b.date);
        bars.dedup_by_key(|b| b.date);
        merged.bars = bars;
        merged.names.sort_by_key(|x| x.1);
        out.push(merged);
    }
    out.sort_by(|a, b| a.ticker.cmp(&b.ticker));
    log.sort();
    (out, log)
}
