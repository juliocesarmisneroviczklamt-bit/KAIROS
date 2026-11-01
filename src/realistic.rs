//! Estudo B: quanto do momentum long-only sobra com execução realista e IR de pessoa física.
//! Custo real = emolumentos B3 (0,03%) + meio spread real + impacto 0,5·σ·√(ordem/volume).
//! Capital em reais de hoje, deflacionado pelo IPCA até o início da simulação.

use crate::backtest::{self, Costs, Plan, SimCfg, SimOut};
use crate::common::{self, Data, Form, Report};
use crate::risk;
use crate::stats;
use crate::util::{self, Paths, pct};
use anyhow::Result;
use std::fmt::Write as _;

const REAL: Costs = Costs::Real { fee: 0.0003, impact_k: 0.5 };

fn cagr_from(final_value: f64, initial: f64, months: usize) -> f64 {
    (final_value / initial).powf(12.0 / months as f64) - 1.0
}

pub fn run(p: &Paths) -> Result<()> {
    let d = Data::load(p)?;
    let all = risk::liquid_forms(&d, 0.0);
    let (w_d, _) = risk::daily_series(&d, &all);
    let start = d.cal.idx_from(200101);
    let forms: Vec<Form> = all.into_iter().filter(|f| f.mi >= start).collect();
    let plain = risk::winner_plans(&d, &forms, None);
    let managed = risk::winner_plans(&d, &forms, Some(&w_d));
    let ym0 = d.ym(forms[0].mi);
    let defl = d.ipca_factor_from(ym0);
    let strategies: [(&str, &Vec<Plan>); 2] = [("vencedores", &plain), ("vencedores c/ controle de vol", &managed)];

    let mut rep = Report::default();
    rep.line("");
    rep.line("================ B. EXECUÇÃO REALISTA + IR (pessoa física) ================");
    rep.line(format!(
        "IPCA de {} até hoje: {:.2}x. Os capitais estão em reais de hoje (R$ 300 mil hoje = R$ {:.0} em {}).",
        util::fmt_ym(ym0),
        defl,
        300_000.0 / defl,
        util::fmt_ym(ym0)
    ));
    rep.line("Custo real = emolumentos B3 0,03% + meio spread (melhor compra/venda no fechamento) + impacto 0,5·σ·√(ordem/volume diário).");
    rep.line("IR: regras de hoje no período todo: 15% sobre o ganho líquido do mês, isento se as vendas do mês somam até R$ 20 mil, prejuízo compensa ganho futuro, JCP com 15% retido, dividendo isento.");

    // 1) custos e atraso, sem IR, R$ 300 mil de hoje
    rep.line("");
    rep.line("1) Custos e atraso de execução (sem IR, R$ 300 mil de hoje)");
    rep.line(format!(
        "{:<31} {:<34} {:>8} {:>8} {:>8} {:>12} {:>7}",
        "estratégia", "execução", "CAGR", "Sharpe*", "maxDD", "custo/lado", "giro"
    ));
    let c300 = 300_000.0 / defl;
    let mut csv = String::from("bloco,estrategia,config,capital_hoje,cagr,sharpe,maxdd,custo_bps,giro,cagr_ir,cagr_ir_liquidando,meses_isentos,meses_tributados\n");
    for (sname, plans) in strategies {
        for (cname, costs, lag) in [
            ("custo fixo 25 bps, no fechamento", Costs::Fixed { bps: 25.0 }, 0),
            ("custo real, no fechamento", REAL, 0),
            ("custo real, 1 pregão de atraso", REAL, 1),
        ] {
            let o = backtest::simulate(&d, plans, &SimCfg { capital0: c300, lag, costs, tax: false, cap_adv: None });
            let cdi: Vec<f64> = o.mi.iter().map(|&m| d.cdi_m[m]).collect();
            let pf = common::perf(&o.ret, &cdi);
            rep.line(format!(
                "{sname:<31} {cname:<34} {:>8} {:>8.2} {:>8} {:>9.0} bps {:>6.0}%",
                pct(pf.cagr),
                pf.sharpe,
                pct(pf.mdd),
                o.avg_cost_bps(),
                100.0 * stats::mean(&o.turnover)
            ));
            let _ = writeln!(
                csv,
                "custos,{sname},{cname},300000,{:.5},{:.3},{:.4},{:.1},{:.3},,,,",
                pf.cagr,
                pf.sharpe,
                pf.mdd,
                o.avg_cost_bps(),
                stats::mean(&o.turnover)
            );
        }
    }

    // 2) IR por tamanho de capital (custo real, 1 pregão de atraso)
    rep.line("");
    rep.line("2) IR por tamanho de capital (custo real, 1 pregão de atraso)");
    rep.line(format!(
        "{:<12} {:<31} {:>10} {:>12} {:>14} {:>16} {:>11}",
        "capital hoje", "estratégia", "antes IR", "após IR*", "vendendo tudo", "meses isent/trib", "custo/lado"
    ));
    let mut ref_run: Option<SimOut> = None;
    for cap in [50_000.0, 300_000.0, 3_000_000.0] {
        let c0 = cap / defl;
        for (sname, plans) in strategies {
            let pre = backtest::simulate(&d, plans, &SimCfg { capital0: c0, lag: 1, costs: REAL, tax: false, cap_adv: None });
            let pos = backtest::simulate(&d, plans, &SimCfg { capital0: c0, lag: 1, costs: REAL, tax: true, cap_adv: None });
            let n = pos.ret.len();
            let (a, b, c) = (
                cagr_from(pre.final_value, c0, n),
                cagr_from(pos.final_value, c0, n),
                cagr_from(pos.liq_value, c0, n),
            );
            rep.line(format!(
                "R$ {:<9} {sname:<31} {:>10} {:>12} {:>14} {:>9}/{:<6} {:>7.0} bps",
                format!("{:.0} mil", cap / 1000.0),
                pct(a),
                pct(b),
                pct(c),
                pos.exempt_months,
                pos.taxed_months,
                pos.avg_cost_bps()
            ));
            let _ = writeln!(
                csv,
                "ir,{sname},custo real lag 1,{cap},{a:.5},,,{:.1},,{b:.5},{c:.5},{},{}",
                pos.avg_cost_bps(),
                pos.exempt_months,
                pos.taxed_months
            );
            if ref_run.is_none() {
                ref_run = Some(pos);
            }
        }
    }
    rep.line("* após IR pago mês a mês, sem vender a carteira no fim (o IR do ganho não realizado fica diferido).");

    // 3) referências no mesmo período, após IR
    let r = ref_run.expect("simulação de referência");
    let n = r.mi.len();
    let w_cdi: f64 = r.mi.iter().map(|&m| 1.0 + d.cdi_m[m]).product();
    let w_mkt: f64 = r.mi.iter().map(|&m| d.nef_vw[m].map(|f| 1.0 + f.rm_rf + f.rf).unwrap_or(1.0 + d.cdi_m[m])).product();
    let w_ipca: f64 = r.mi.iter().map(|&m| 1.0 + d.ipca.get(&d.ym(m)).copied().unwrap_or(0.0)).product();
    let after = |w: f64| if w > 1.0 { 1.0 + 0.85 * (w - 1.0) } else { w };
    let g = |w: f64| w.powf(12.0 / n as f64) - 1.0;
    rep.line("");
    rep.line(format!(
        "3) Referências no mesmo período ({} a {}), compradas no início e vendidas no fim:",
        util::fmt_ym(d.ym(r.mi[0])),
        util::fmt_ym(d.ym(*r.mi.last().unwrap()))
    ));
    rep.line(format!("   CDI (tipo Tesouro Selic, 15% de IR no resgate): {} antes do IR, {} depois", pct(g(w_cdi)), pct(g(after(w_cdi)))));
    rep.line(format!(
        "   Mercado NEFIN (tipo ETF de índice, 15% no resgate, sem isenção): {} antes, {} depois",
        pct(g(w_mkt)),
        pct(g(after(w_mkt)))
    ));
    rep.line(format!("   Inflação (IPCA): {} ao ano", pct(g(w_ipca))));

    std::fs::write(p.results.join("realista_resumo.csv"), csv)?;
    std::fs::write(p.results.join("realista_relatorio.txt"), &rep.text)?;
    Ok(())
}
