//! Teste 1: replicar o fator de momentum (WML) do NEFIN com os nossos dados, e medir o que
//! sobra pra quem só pode comprar (long-only), depois de custos.

use crate::macrodata::{self, Factors};
use crate::panel::Panel;
use crate::stats;
use crate::universe::{self, LiqRule};
use crate::util::{self, Paths, pct};
use anyhow::Result;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;

#[derive(clap::Args, Clone, Debug)]
pub struct Args {
    /// Regra de liquidez da elegibilidade
    #[arg(long, value_enum, default_value_t = LiqRule::Dias)]
    pub liq: LiqRule,
    /// Volume financeiro diário mínimo em R$ (NEFIN usa 500 mil)
    #[arg(long, default_value_t = 500_000.0)]
    pub min_vol: f64,
    /// Custo por lado em bps (emolumentos + spread + slippage), usado nas regressões do long-only
    #[arg(long, default_value_t = 25.0)]
    pub cost_bps: f64,
    /// Mínimo de ações elegíveis pra formar as carteiras do mês
    #[arg(long, default_value_t = 15)]
    pub min_stocks: usize,
}

struct Row {
    ym: u32,
    n: usize,
    w: f64,
    l: f64,
    mkt_ew: f64,
    lo_gross: f64,
    turnover: f64,
    bh_mkt: f64,
}

/// (média a.a., vol a.a., t de Newey-West da média, Sharpe)
fn ann(x: &[f64]) -> (f64, f64, f64, f64) {
    let m = stats::mean(x);
    let s = stats::std(x);
    let t = stats::ols_nw(x, &[], stats::nw_lags(x.len())).map(|o| o.t[0]).unwrap_or(f64::NAN);
    (m * 12.0, s * 12f64.sqrt(), t, m / s * 12f64.sqrt())
}

fn corr_finite(x: &[f64], y: &[f64]) -> (f64, usize) {
    let (a, b): (Vec<f64>, Vec<f64>) =
        x.iter().zip(y).filter(|(p, q)| p.is_finite() && q.is_finite()).map(|(p, q)| (*p, *q)).unzip();
    (stats::corr(&a, &b), a.len())
}

pub fn run(p: &Paths, a: &Args) -> Result<()> {
    let panel: Panel = util::load_bin(&p.panel)?;
    let cdi_d = macrodata::load_cdi(&p.raw_bcb)?;
    let nef_ew_d = macrodata::load_factors(&p.raw_nefin.join("nefin_factors_equally_weighted.csv"), false)?;
    let nef_vw_d = macrodata::load_factors(&p.raw_nefin.join("nefin_factors.csv"), false)?;
    let nef_ew_m = macrodata::load_factors(&p.raw_nefin.join("nefin_factors_equally_weighted_monthly.csv"), true)?;
    let nef_vw_m = macrodata::load_factors(&p.raw_nefin.join("nefin_factors_monthly.csv"), true)?;

    // calendário mensal
    let dates = &panel.dates;
    let mut months: Vec<u32> = Vec::new();
    let mut mrange: Vec<(usize, usize)> = Vec::new();
    let mut date_month: Vec<usize> = Vec::with_capacity(dates.len());
    for (i, &d) in dates.iter().enumerate() {
        let m = util::ym(d);
        if months.last() != Some(&m) {
            months.push(m);
            mrange.push((i, i + 1));
        } else if let Some(r) = mrange.last_mut() {
            r.1 = i + 1;
        }
        date_month.push(months.len() - 1);
    }
    let nm = months.len();
    let nt = panel.tks.len();

    // retorno mensal de cada papel (0 nos dias sem negócio, como no NEFIN)
    let mut mret = vec![vec![0.0f64; nm]; nt];
    let mut mtraded = vec![vec![false; nm]; nt];
    for (t, tk) in panel.tks.iter().enumerate() {
        for d in &tk.days {
            let m = date_month[d.di as usize];
            mret[t][m] = (1.0 + mret[t][m]) * (1.0 + d.r) - 1.0;
            mtraded[t][m] = true;
        }
    }
    let mut cdi_m: BTreeMap<u32, f64> = BTreeMap::new();
    for (&d, &r) in &cdi_d {
        *cdi_m.entry(util::ym(d)).or_insert(1.0) *= 1.0 + r;
    }
    for v in cdi_m.values_mut() {
        *v -= 1.0;
    }

    let mut elig_cache: HashMap<u32, Vec<usize>> = HashMap::new();
    let mut rows: Vec<Row> = Vec::new();
    let mut daily: Vec<(u32, f64, f64, f64)> = Vec::new();
    let mut prev_hold: Vec<(usize, f64)> = Vec::new();
    let mut extremes: Vec<(usize, u32, f64)> = Vec::new();

    for mi in 12..nm {
        let ym = months[mi];
        let y = ym / 100;
        let elig = elig_cache.entry(y).or_insert_with(|| universe::eligible(&panel, y, a.liq, a.min_vol));
        // precisa ter negociado no mês anterior (está viva na formação da carteira)
        let alive: Vec<usize> = elig.iter().copied().filter(|&t| mtraded[t][mi - 1]).collect();
        if alive.len() < a.min_stocks {
            prev_hold.clear();
            continue;
        }
        // sinal: retorno acumulado de t-12 a t-2 (pula o mês t-1)
        let mut sig: Vec<(usize, f64)> = alive
            .iter()
            .map(|&t| (t, (mi - 12..=mi - 2).map(|m| 1.0 + mret[t][m]).product::<f64>() - 1.0))
            .collect();
        sig.sort_by(|p1, p2| p1.1.total_cmp(&p2.1));
        let n = sig.len();
        let k3 = ((n as f64) / 3.0).round() as usize;
        let losers: Vec<usize> = sig[..k3].iter().map(|x| x.0).collect();
        let winners: Vec<usize> = sig[n - k3..].iter().map(|x| x.0).collect();

        // retornos diários equal-weighted (rebalanceamento diário, como o NEFIN)
        let (a0, b0) = mrange[mi];
        for &t in &alive {
            let days = &panel.tks[t].days;
            let lo = days.partition_point(|d| (d.di as usize) < a0);
            for d in days[lo..].iter().take_while(|d| (d.di as usize) < b0) {
                if d.r.abs() > 0.5 {
                    extremes.push((t, dates[d.di as usize], d.r));
                }
            }
        }
        let daily_ew = |set: &[usize]| -> Vec<f64> {
            let mut acc = vec![0.0f64; b0 - a0];
            for &t in set {
                let days = &panel.tks[t].days;
                let lo = days.partition_point(|d| (d.di as usize) < a0);
                for d in &days[lo..] {
                    let di = d.di as usize;
                    if di >= b0 {
                        break;
                    }
                    acc[di - a0] += d.r;
                }
            }
            let inv = 1.0 / set.len() as f64;
            acc.into_iter().map(|v| v * inv).collect()
        };
        let (wd, ld, ed) = (daily_ew(&winners), daily_ew(&losers), daily_ew(&alive));
        let comp = |v: &[f64]| v.iter().map(|r| 1.0 + r).product::<f64>() - 1.0;
        for i in 0..(b0 - a0) {
            daily.push((dates[a0 + i], wd[i], ld[i], ed[i]));
        }

        // long-only: compra os vencedores em pesos iguais e segura o mês (buy-and-hold)
        let lo_gross = winners.iter().map(|&t| mret[t][mi]).sum::<f64>() / winners.len() as f64;
        let bh_mkt = alive.iter().map(|&t| mret[t][mi]).sum::<f64>() / alive.len() as f64;
        let neww = 1.0 / winners.len() as f64;
        let mut old: HashMap<usize, f64> = prev_hold.iter().copied().collect();
        let mut turnover = 0.0;
        for &t in &winners {
            turnover += (neww - old.remove(&t).unwrap_or(0.0)).abs();
        }
        turnover += old.values().sum::<f64>();
        let tot: f64 = winners.iter().map(|&t| neww * (1.0 + mret[t][mi])).sum();
        prev_hold = winners.iter().map(|&t| (t, neww * (1.0 + mret[t][mi]) / tot)).collect();

        rows.push(Row { ym, n, w: comp(&wd), l: comp(&ld), mkt_ew: comp(&ed), lo_gross, turnover, bh_mkt });
    }

    // ------------------------------------------------------------------ relatório
    let mut rep = String::new();
    macro_rules! out {
        ($($t:tt)*) => {{
            let s = format!($($t)*);
            println!("{s}");
            rep.push_str(&s);
            rep.push('\n');
        }};
    }
    let first = rows.first().map(|r| r.ym).unwrap_or(0);
    let last = rows.last().map(|r| r.ym).unwrap_or(0);
    out!("");
    out!("================ MOMENTUM NA B3 ================");
    out!(
        "Universo estilo NEFIN: regra de liquidez {:?}, volume mínimo R$ {:.0}. Meses: {} ({}..{})",
        a.liq,
        a.min_vol,
        rows.len(),
        util::fmt_ym(first),
        util::fmt_ym(last)
    );
    let mut by_year: BTreeMap<u32, (usize, usize)> = BTreeMap::new();
    for r in &rows {
        let e = by_year.entry(r.ym / 100).or_insert((0, 0));
        e.0 += r.n;
        e.1 += 1;
    }
    out!(
        "Ações elegíveis (média no ano): {}",
        by_year
            .iter()
            .filter(|(y, _)| *y % 5 == 0 || **y == last / 100)
            .map(|(y, (s, c))| format!("{y}: {}", s / c))
            .collect::<Vec<_>>()
            .join(" | ")
    );

    extremes.sort_by(|x, y| y.2.abs().total_cmp(&x.2.abs()));
    out!(
        "Retornos diários |r| > 50% dentro do universo elegível: {} (maiores: {})",
        extremes.len(),
        extremes
            .iter()
            .take(8)
            .map(|(t, d, r)| format!("{} {} {}", panel.tks[*t].ticker, util::fmt_date(*d), pct(*r)))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let mut ecsv = String::from("ticker,data,retorno\n");
    for (t, d, r) in &extremes {
        let _ = writeln!(ecsv, "{},{},{:.4}", panel.tks[*t].ticker, util::fmt_date(*d), r);
    }
    std::fs::write(p.diag.join("extremos_no_universo.csv"), ecsv)?;

    // --- 1. replicação
    let sample: Vec<&Row> = rows.iter().filter(|r| nef_ew_m.contains_key(&r.ym)).collect();
    let ours: Vec<f64> = sample.iter().map(|r| r.w - r.l).collect();
    let n_ew: Vec<f64> = sample.iter().map(|r| nef_ew_m[&r.ym].wml).collect();
    let n_vw: Vec<f64> = sample.iter().map(|r| nef_vw_m.get(&r.ym).map(|f| f.wml).unwrap_or(f64::NAN)).collect();
    let our_mkt: Vec<f64> = sample.iter().map(|r| r.mkt_ew).collect();
    let n_mkt: Vec<f64> = sample.iter().map(|r| nef_ew_m[&r.ym].rm_rf + nef_ew_m[&r.ym].rf).collect();
    let our_d: HashMap<u32, f64> = daily.iter().map(|&(d, w, l, _)| (d, w - l)).collect();
    let (mut xd, mut yd_ew, mut yd_vw) = (Vec::new(), Vec::new(), Vec::new());
    for (d, f) in &nef_ew_d {
        if let Some(&o) = our_d.get(d) {
            xd.push(o);
            yd_ew.push(f.wml);
            yd_vw.push(nef_vw_d.get(d).map(|g| g.wml).unwrap_or(f64::NAN));
        }
    }
    let (c_ew, nn) = corr_finite(&ours, &n_ew);
    let (c_vw, _) = corr_finite(&ours, &n_vw);
    let (c_d, nd) = corr_finite(&xd, &yd_ew);
    let (c_dvw, _) = corr_finite(&xd, &yd_vw);
    let (c_mkt, _) = corr_finite(&our_mkt, &n_mkt);
    out!("");
    out!("--- 1. Replicação do WML do NEFIN (valida o motor) ---");
    out!(
        "Correlação mensal  com WML NEFIN equal-weighted: {c_ew:.3} | value-weighted: {c_vw:.3}   ({nn} meses)"
    );
    out!("Correlação diária  com WML NEFIN equal-weighted: {c_d:.3} | value-weighted: {c_dvw:.3}   ({nd} dias)");
    out!("Correlação mensal do nosso mercado EW com o mercado NEFIN (VW): {c_mkt:.3}");

    // --- 2. prêmio de momentum (long-short, sem custos)
    out!("");
    out!("--- 2. Prêmio de momentum long-short (vencedores - perdedores, sem custos) ---");
    out!("{:<14} {:>8} {:>8} {:>7} {:>7} | {:>8} {:>7}", "período", "média", "vol", "t(NW)", "Sharpe", "NEFIN", "t(NW)");
    let periods: [(u32, u32, &str); 4] = [
        (200101, 201012, "2001-2010"),
        (201101, 202012, "2011-2020"),
        (202101, 999912, "2021-hoje"),
        (0, 999912, "2001-hoje"),
    ];
    for (p0, p1, name) in periods {
        let idx: Vec<usize> = (0..sample.len()).filter(|&i| sample[i].ym >= p0 && sample[i].ym <= p1).collect();
        if idx.len() < 12 {
            continue;
        }
        let x: Vec<f64> = idx.iter().map(|&i| ours[i]).collect();
        let z: Vec<f64> = idx.iter().map(|&i| n_ew[i]).collect();
        let (m, s, t, sh) = ann(&x);
        let (mz, _, tz, _) = ann(&z);
        out!("{name:<14} {:>8} {:>8} {t:>7.2} {sh:>7.2} | {:>8} {tz:>7.2}", pct(m), pct(s), pct(mz));
    }
    let pre: Vec<f64> = rows.iter().filter(|r| r.ym < 200101).map(|r| r.w - r.l).collect();
    if pre.len() >= 24 {
        let (m, _, t, _) = ann(&pre);
        out!(
            "{:<14} {:>8} {:>8} {t:>7.2} {:>7} | (fora da amostra do NEFIN)",
            format!("{}-2000", util::fmt_ym(first).get(0..4).unwrap_or("")),
            pct(m),
            "",
            ""
        );
    }
    let fx = |f: fn(&Factors) -> f64| -> Vec<f64> { sample.iter().map(|r| f(&nef_ew_m[&r.ym])).collect() };
    let (rm, smb, hml, iml, wml_n, rf) = (
        fx(|f| f.rm_rf),
        fx(|f| f.smb),
        fx(|f| f.hml),
        fx(|f| f.iml),
        fx(|f| f.wml),
        fx(|f| f.rf),
    );
    if let Some(o) = stats::ols_nw(&ours, &[&rm, &smb, &hml, &iml], stats::nw_lags(ours.len())) {
        out!(
            "WML nosso explicado por mercado/tamanho/valor/iliquidez? alpha {} a.a. (t={:.2}), R2={:.2}",
            pct(o.coef[0] * 12.0),
            o.t[0],
            o.r2
        );
    }

    // --- 3. long-only
    out!("");
    out!("--- 3. Long-only nos vencedores (o que dá pra fazer sem alugar ação) ---");
    let lo_cdi: Vec<f64> = sample.iter().map(|r| cdi_m.get(&r.ym).copied().unwrap_or(f64::NAN)).collect();
    let turn: Vec<f64> = sample.iter().map(|r| r.turnover).collect();
    out!(
        "Giro médio: {:.0}% da carteira por mês (compra+venda); custo anual ≈ giro x 12 x custo por lado",
        100.0 * stats::mean(&turn)
    );
    out!(
        "{:<22} {:>8} {:>8} {:>8} {:>8} {:>12} {:>6} {:>9}",
        "estratégia", "CAGR", "vol", "Sharpe*", "maxDD", "vs univ.EW", "t", "%m>CDI"
    );
    let univ: Vec<f64> = sample.iter().map(|r| r.bh_mkt).collect();
    let line = |name: &str, x: &[f64]| -> String {
        let ex_cdi: Vec<f64> = x.iter().zip(&lo_cdi).map(|(a, b)| a - b).collect();
        let sh = stats::mean(&ex_cdi) / stats::std(&ex_cdi) * 12f64.sqrt();
        let ex_u: Vec<f64> = x.iter().zip(&univ).map(|(a, b)| a - b).collect();
        let (mu, _, tu, _) = ann(&ex_u);
        let beat = ex_cdi.iter().filter(|v| **v > 0.0).count() as f64 / ex_cdi.len() as f64;
        format!(
            "{name:<22} {:>8} {:>8} {sh:>8.2} {:>8} {:>12} {tu:>6.2} {:>8.0}%",
            pct(stats::cagr(x, 12.0)),
            pct(stats::std(x) * 12f64.sqrt()),
            pct(stats::max_dd(x)),
            pct(mu),
            100.0 * beat
        )
    };
    let mut net_main: Vec<f64> = Vec::new();
    for c in [0.0, 10.0, 25.0, 50.0, 100.0] {
        let net: Vec<f64> = sample.iter().map(|r| r.lo_gross - r.turnover * c / 1e4).collect();
        out!("{}", line(&format!("vencedores, custo {c:.0}bps"), &net));
        if (c - a.cost_bps).abs() < 1e-9 {
            net_main = net;
        }
    }
    if net_main.is_empty() {
        net_main = sample.iter().map(|r| r.lo_gross - r.turnover * a.cost_bps / 1e4).collect();
    }
    out!("{}", line("universo EW (B&H)", &univ));
    let nef_mkt: Vec<f64> = sample.iter().map(|r| nef_vw_m.get(&r.ym).map(|f| f.rm_rf + f.rf).unwrap_or(f64::NAN)).collect();
    if nef_mkt.iter().all(|v| v.is_finite()) {
        out!("{}", line("mercado NEFIN (VW)", &nef_mkt));
    }
    out!("{}", line("CDI", &lo_cdi));
    out!("* Sharpe sobre o CDI. 'vs univ.EW' = excesso médio a.a. sobre o universo elegível equal-weighted.");

    let y_ex: Vec<f64> = net_main.iter().zip(&rf).map(|(a, b)| a - b).collect();
    let lags = stats::nw_lags(y_ex.len());
    out!("");
    out!("Regressão do long-only (custo {:.0}bps) contra os fatores NEFIN (equal-weighted):", a.cost_bps);
    if let Some(o) = stats::ols_nw(&y_ex, &[&rm, &smb, &hml, &iml], lags) {
        out!(
            "  4 fatores (Mkt, SMB, HML, IML):       alpha {} a.a. (t={:.2})  betas Mkt {:.2} SMB {:.2} HML {:.2} IML {:.2}  R2={:.2}",
            pct(o.coef[0] * 12.0),
            o.t[0],
            o.coef[1],
            o.coef[2],
            o.coef[3],
            o.coef[4],
            o.r2
        );
    }
    if let Some(o) = stats::ols_nw(&y_ex, &[&rm, &smb, &hml, &iml, &wml_n], lags) {
        out!(
            "  5 fatores (+ WML do NEFIN):           alpha {} a.a. (t={:.2})  beta WML {:.2} (t={:.2})  R2={:.2}",
            pct(o.coef[0] * 12.0),
            o.t[0],
            o.coef[5],
            o.t[5],
            o.r2
        );
    }

    // --- arquivos
    let mut csv = String::from(
        "mes,n_elegiveis,W,L,WML,WML_NEFIN_EW,WML_NEFIN_VW,mercado_EW,longonly_bruto,giro,universo_EW_bh,cdi\n",
    );
    for r in &rows {
        let _ = writeln!(
            csv,
            "{},{},{:.6},{:.6},{:.6},{},{},{:.6},{:.6},{:.4},{:.6},{}",
            util::fmt_ym(r.ym),
            r.n,
            r.w,
            r.l,
            r.w - r.l,
            nef_ew_m.get(&r.ym).map(|f| format!("{:.6}", f.wml)).unwrap_or_default(),
            nef_vw_m.get(&r.ym).map(|f| format!("{:.6}", f.wml)).unwrap_or_default(),
            r.mkt_ew,
            r.lo_gross,
            r.turnover,
            r.bh_mkt,
            cdi_m.get(&r.ym).map(|v| format!("{v:.6}")).unwrap_or_default()
        );
    }
    std::fs::write(p.results.join("momentum_mensal.csv"), csv)?;
    let mut dcsv = String::from("data,W,L,WML,mercado_EW,WML_NEFIN_EW,WML_NEFIN_VW\n");
    for &(d, w, l, m) in &daily {
        let _ = writeln!(
            dcsv,
            "{},{w:.6},{l:.6},{:.6},{m:.6},{},{}",
            util::fmt_date(d),
            w - l,
            nef_ew_d.get(&d).map(|f| format!("{:.6}", f.wml)).unwrap_or_default(),
            nef_vw_d.get(&d).map(|f| format!("{:.6}", f.wml)).unwrap_or_default()
        );
    }
    std::fs::write(p.results.join("momentum_diario.csv"), dcsv)?;
    std::fs::write(p.results.join("momentum_relatorio.txt"), &rep)?;
    println!("\nArquivos: {}\\momentum_mensal.csv, momentum_diario.csv, momentum_relatorio.txt", p.results.display());
    Ok(())
}
