//! Simulador mensal com execução realista e IR de pessoa física.
//!
//! - Custo por papel: emolumentos B3 + meio spread real (mediana dos últimos 21 pregões, a partir
//!   da melhor compra/venda no fechamento) + impacto de mercado k·σ·√(ordem/volume diário).
//! - Atraso de execução opcional (lag em pregões).
//! - Capacidade: posição máxima = cap_adv × volume diário mediano.
//! - IR (regras atuais, aplicadas ao período todo): 15% sobre o ganho líquido mensal em ações à
//!   vista; isenção quando as vendas do mês somam até R$ 20 mil; prejuízo compensado nos meses
//!   seguintes; custo médio de aquisição; JCP com 15% retido na fonte; dividendos isentos.
//! - O que não está em ação fica no CDI.
//! - Registra patrimônio diário, operações e IR mês a mês (usado também pelo paper trading).

use crate::common::{Data, days_in};
use crate::panel::Day;
use crate::stats;
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Copy, Debug)]
pub enum Costs {
    Fixed { bps: f64 },
    Real { fee: f64, impact_k: f64 },
}

#[derive(Clone, Debug)]
pub struct SimCfg {
    /// capital inicial em R$ nominais da data de início
    pub capital0: f64,
    pub lag: usize,
    pub costs: Costs,
    pub tax: bool,
    pub cap_adv: Option<f64>,
}

/// Pesos-alvo da carteira do mês `mi` (soma <= 1; o resto fica no CDI).
pub struct Plan {
    pub mi: usize,
    pub w: Vec<(usize, f64)>,
}

#[derive(Clone, Debug)]
pub struct Trade {
    /// pregão em cujo fechamento a ordem foi executada (AAAAMMDD)
    pub date: u32,
    pub ticker: String,
    pub buy: bool,
    pub value: f64,
    pub price: f64,
    pub cost: f64,
}

#[derive(Clone, Debug)]
pub struct TaxMonth {
    pub ym: u32,
    pub sales: f64,
    pub realized: f64,
    pub tax: f64,
    pub exempt: bool,
}

#[derive(Default)]
pub struct SimOut {
    pub mi: Vec<usize>,
    /// retorno mensal da riqueza total, já descontados custos e IR pago
    pub ret: Vec<f64>,
    pub turnover: Vec<f64>,
    pub exposure: Vec<f64>,
    pub cost_paid: f64,
    pub traded: f64,
    pub tax_paid: f64,
    pub jcp_withheld: f64,
    pub final_value: f64,
    /// valor final se vender tudo no fim pagando o IR do ganho não realizado
    pub liq_value: f64,
    pub exempt_months: usize,
    pub taxed_months: usize,
    /// fração do valor-alvo cortada por falta de liquidez (capacidade)
    pub capped_frac: f64,
    /// patrimônio e caixa no fim de cada pregão: (índice de data, patrimônio, caixa)
    pub daily: Vec<(usize, f64, f64)>,
    pub trades: Vec<Trade>,
    pub taxes: Vec<TaxMonth>,
    /// posições no fim: (papel, valor)
    pub holdings: Vec<(usize, f64)>,
}

impl SimOut {
    /// custo médio por real negociado, em bps
    pub fn avg_cost_bps(&self) -> f64 {
        if self.traded > 0.0 { self.cost_paid / self.traded * 1e4 } else { 0.0 }
    }
}

#[derive(Clone, Copy)]
struct Liq {
    half_spread: f64,
    sigma: f64,
    adv: f64,
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    Some(v[v.len() / 2])
}

/// Liquidez do papel com dados até o pregão s-1 (nada do futuro).
fn liq(d: &Data, t: usize, s: usize) -> Liq {
    let days = &d.panel.tks[t].days;
    let hi = days.partition_point(|x| (x.di as usize) < s);
    let w63 = &days[hi.saturating_sub(63)..hi];
    let adv = median(w63.iter().map(|x| x.vol).collect()).unwrap_or(0.0);
    let rs: Vec<f64> = w63.iter().map(|x| x.r).collect();
    let sigma = stats::std(&rs);
    let spr = median(
        days[hi.saturating_sub(21)..hi]
            .iter()
            .map(|x| x.spr as f64)
            .filter(|v| v.is_finite() && *v >= 0.0)
            .collect(),
    );
    Liq {
        // sem cotação de compra/venda: assume spread de 4% (conservador)
        half_spread: (spr.unwrap_or(0.04).max(0.0002) / 2.0).min(0.10),
        sigma: if sigma.is_finite() { sigma.min(0.20) } else { 0.03 },
        adv: adv.max(1.0),
    }
}

/// último preço sem ajuste do papel até o pregão `upto` (inclusive)
fn last_px(d: &Data, t: usize, upto: usize) -> f64 {
    days_in(&d.panel.tks[t].days, 0, upto + 1).last().map(|x| x.px as f64).unwrap_or(f64::NAN)
}

struct Pos {
    val: f64,
    basis: f64,
}

pub fn simulate(d: &Data, plans: &[Plan], cfg: &SimCfg) -> SimOut {
    let nd = d.panel.dates.len();
    let mut out = SimOut::default();
    let mut cache: HashMap<(usize, usize), Liq> = HashMap::new();
    let mut get_liq = |t: usize, s: usize| *cache.entry((t, s)).or_insert_with(|| liq(d, t, s));
    let cost = |l: Liq, amt: f64| -> f64 {
        match cfg.costs {
            Costs::Fixed { bps } => amt * bps / 1e4,
            Costs::Real { fee, impact_k } => amt * (fee + l.half_spread + impact_k * l.sigma * (amt / l.adv).sqrt()),
        }
    };
    let mut cash = cfg.capital0;
    let mut pos: BTreeMap<usize, Pos> = BTreeMap::new();
    let mut carry = 0.0f64;
    let (mut capped, mut targeted) = (0.0f64, 0.0f64);

    for plan in plans {
        let raw_s = d.cal.mrange[plan.mi].0 + cfg.lag;
        if raw_s == 0 || raw_s > nd {
            break;
        }
        let (s, e) = d.window(plan.mi, cfg.lag);
        let trade_di = s - 1;
        let trade_date = d.panel.dates[trade_di];
        let v0 = cash + pos.values().map(|p| p.val).sum::<f64>();
        if v0 <= 0.0 {
            break;
        }
        // alvos (com limite de capacidade)
        let mut target: BTreeMap<usize, f64> = BTreeMap::new();
        for &(t, w) in &plan.w {
            let mut tv = w * v0;
            targeted += tv;
            if let Some(k) = cfg.cap_adv {
                let cap = k * get_liq(t, s).adv;
                if tv > cap {
                    capped += tv - cap;
                    tv = cap;
                }
            }
            *target.entry(t).or_insert(0.0) += tv;
        }
        // vendas primeiro
        let (mut sales, mut realized, mut traded) = (0.0f64, 0.0f64, 0.0f64);
        let eps = 1e-7 * v0;
        let held: Vec<usize> = pos.keys().copied().collect();
        for t in held {
            let tv = target.get(&t).copied().unwrap_or(0.0);
            let l = get_liq(t, s);
            let p = pos.get_mut(&t).unwrap();
            if p.val > tv + eps {
                let sell = p.val - tv;
                let c = cost(l, sell);
                let b_out = p.basis * sell / p.val;
                realized += sell - c - b_out;
                p.basis -= b_out;
                p.val -= sell;
                cash += sell - c;
                out.cost_paid += c;
                sales += sell;
                traded += sell;
                out.trades.push(Trade {
                    date: trade_date,
                    ticker: d.panel.tks[t].ticker.clone(),
                    buy: false,
                    value: sell,
                    price: last_px(d, t, trade_di),
                    cost: c,
                });
            }
        }
        pos.retain(|_, p| p.val > eps);
        // compras, limitadas ao caixa disponível
        let mut buys: Vec<(usize, f64, f64)> = Vec::new();
        for (&t, &tv) in &target {
            let cur = pos.get(&t).map(|p| p.val).unwrap_or(0.0);
            if tv > cur + eps {
                let amt = tv - cur;
                buys.push((t, amt, cost(get_liq(t, s), amt)));
            }
        }
        let need: f64 = buys.iter().map(|b| b.1 + b.2).sum();
        let scale = if need > cash { (cash / need).max(0.0) } else { 1.0 };
        for (t, amt, c) in buys {
            let (amt, c) = (amt * scale, c * scale);
            if amt <= 0.0 {
                continue;
            }
            cash -= amt + c;
            out.cost_paid += c;
            traded += amt;
            out.trades.push(Trade {
                date: trade_date,
                ticker: d.panel.tks[t].ticker.clone(),
                buy: true,
                value: amt,
                price: last_px(d, t, trade_di),
                cost: c,
            });
            let p = pos.entry(t).or_insert(Pos { val: 0.0, basis: 0.0 });
            p.val += amt;
            p.basis += amt + c;
        }
        // patrimônio logo depois das ordens (fechamento do pregão da execução)
        let nav_trade = cash + pos.values().map(|p| p.val).sum::<f64>();
        match out.daily.last_mut() {
            Some(last) if last.0 == trade_di => *last = (trade_di, nav_trade, cash),
            _ => out.daily.push((trade_di, nav_trade, cash)),
        }
        // carrega a carteira em [s, e), pregão a pregão: preço muda a posição, provento vira caixa
        let mut cursors: Vec<(usize, &[Day], usize)> =
            pos.keys().map(|&t| (t, days_in(&d.panel.tks[t].days, s, e), 0usize)).collect();
        for di in s..e {
            let mut income = 0.0f64;
            for (t, sl, k) in cursors.iter_mut() {
                if *k < sl.len() && sl[*k].di as usize == di {
                    let x = &sl[*k];
                    *k += 1;
                    let p = pos.get_mut(t).unwrap();
                    let (div, jcp) = (x.div as f64, x.jcp as f64);
                    let jcp_net = if cfg.tax { 0.85 * jcp } else { jcp };
                    if cfg.tax {
                        out.jcp_withheld += p.val * 0.15 * jcp;
                    }
                    income += p.val * (div + jcp_net);
                    p.val *= 1.0 + (x.r - div - jcp);
                }
            }
            cash = cash * (1.0 + d.cdi_d[di]) + income;
            out.daily.push((di, cash + pos.values().map(|p| p.val).sum::<f64>(), cash));
        }
        // IR do mês das vendas
        if cfg.tax && sales > 0.0 {
            let mut tax = 0.0;
            let exempt = sales <= 20_000.0;
            if exempt {
                out.exempt_months += 1;
                if realized < 0.0 {
                    carry -= realized;
                }
            } else {
                out.taxed_months += 1;
                let net = realized - carry;
                if net > 0.0 {
                    tax = 0.15 * net;
                    cash -= tax;
                    out.tax_paid += tax;
                    carry = 0.0;
                } else {
                    carry = -net;
                }
            }
            if let Some(last) = out.daily.last_mut() {
                last.1 -= tax;
                last.2 -= tax;
            }
            out.taxes.push(TaxMonth { ym: d.ym(plan.mi), sales, realized, tax, exempt });
        }
        if e > s {
            let v1 = cash + pos.values().map(|p| p.val).sum::<f64>();
            out.mi.push(plan.mi);
            out.ret.push(v1 / v0 - 1.0);
            out.turnover.push(traded / v0);
            out.exposure.push(plan.w.iter().map(|x| x.1).sum());
        }
        out.traded += traded;
    }
    out.final_value = cash + pos.values().map(|p| p.val).sum::<f64>();
    let unreal: f64 = pos.values().map(|p| p.val - p.basis).sum();
    out.liq_value = if cfg.tax { out.final_value - 0.15 * (unreal - carry).max(0.0) } else { out.final_value };
    out.capped_frac = if targeted > 0.0 { capped / targeted } else { 0.0 };
    out.holdings = pos.iter().map(|(&t, p)| (t, p.val)).collect();
    out
}
