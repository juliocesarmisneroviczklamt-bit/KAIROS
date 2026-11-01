//! Registro de testes. Toda estratégia avaliada entra aqui, e o resultado é descontado pelo
//! número de tentativas (Deflated Sharpe Ratio, Bailey & López de Prado 2014). Testou 20
//! coisas? O melhor Sharpe tem que bater o que o melhor de 20 dariam por puro acaso.

use crate::stats;
use anyhow::Result;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Trial {
    pub estudo: String,
    pub nome: String,
    pub params: String,
    pub periodo: String,
    pub n: usize,
    /// Sharpe mensal (média/desvio dos retornos em excesso)
    pub sr_m: f64,
    pub skew: f64,
    /// curtose (3 = normal)
    pub kurt: f64,
}

impl Trial {
    pub fn new(estudo: &str, nome: &str, params: &str, periodo: &str, excess: &[f64]) -> Self {
        let (sr_m, skew, kurt) = moments(excess);
        Trial {
            estudo: estudo.into(),
            nome: nome.into(),
            params: params.into(),
            periodo: periodo.into(),
            n: excess.len(),
            sr_m,
            skew,
            kurt,
        }
    }
    fn key(&self) -> String {
        format!("{}|{}|{}|{}", self.estudo, self.nome, self.params, self.periodo)
    }
}

pub fn moments(x: &[f64]) -> (f64, f64, f64) {
    let m = stats::mean(x);
    let s = stats::std(x);
    let n = x.len() as f64;
    let sk = x.iter().map(|v| ((v - m) / s).powi(3)).sum::<f64>() / n;
    let ku = x.iter().map(|v| ((v - m) / s).powi(4)).sum::<f64>() / n;
    (m / s, sk, ku)
}

fn esc(s: &str) -> String {
    s.replace([',', '\n'], ";")
}

pub fn load(path: &Path) -> Vec<Trial> {
    let Ok(txt) = std::fs::read_to_string(path) else { return Vec::new() };
    txt.lines()
        .skip(1)
        .filter_map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            if f.len() < 8 {
                return None;
            }
            Some(Trial {
                estudo: f[0].into(),
                nome: f[1].into(),
                params: f[2].into(),
                periodo: f[3].into(),
                n: f[4].parse().ok()?,
                sr_m: f[5].parse().ok()?,
                skew: f[6].parse().ok()?,
                kurt: f[7].parse().ok()?,
            })
        })
        .collect()
}

/// Grava (ou atualiza) a tentativa e devolve o registro inteiro.
pub fn upsert(path: &Path, t: &Trial) -> Result<Vec<Trial>> {
    let mut all = load(path);
    all.retain(|x| x.key() != t.key());
    all.push(t.clone());
    let mut s = String::from("estudo,nome,params,periodo,n_meses,sharpe_mensal,assimetria,curtose\n");
    for x in &all {
        s.push_str(&format!(
            "{},{},{},{},{},{:.6},{:.4},{:.4}\n",
            esc(&x.estudo),
            esc(&x.nome),
            esc(&x.params),
            esc(&x.periodo),
            x.n,
            x.sr_m,
            x.skew,
            x.kurt
        ));
    }
    std::fs::write(path, s)?;
    Ok(all)
}

fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t * (-z * z - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98 + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
        .exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

pub fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Inversa da normal padrão (algoritmo de Acklam).
pub fn norm_inv(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let pl = 0.02425;
    if p < pl {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - pl {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Sharpe esperado do melhor de `n` testes sem habilidade nenhuma (variância `var` entre testes).
pub fn expected_max_sr(n: usize, var: f64) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let g = 0.577_215_664_9;
    let nf = n as f64;
    var.sqrt() * ((1.0 - g) * norm_inv(1.0 - 1.0 / nf) + g * norm_inv(1.0 - 1.0 / (nf * std::f64::consts::E)))
}

pub struct Deflated {
    pub n_trials: usize,
    /// Sharpe anual que o melhor de N testes daria por acaso
    pub sr0_ann: f64,
    /// probabilidade de o Sharpe verdadeiro ser maior que isso (DSR)
    pub dsr: f64,
}

pub fn deflate(t: &Trial, all: &[Trial]) -> Deflated {
    let n = all.len().max(1);
    let srs: Vec<f64> = all.iter().map(|x| x.sr_m).filter(|v| v.is_finite()).collect();
    let var = if srs.len() >= 3 { stats::std(&srs).powi(2) } else { 1.0 / t.n.max(2) as f64 };
    let sr0 = expected_max_sr(n, var);
    let denom = (1.0 - t.skew * t.sr_m + (t.kurt - 1.0) / 4.0 * t.sr_m * t.sr_m).max(1e-9).sqrt();
    let z = (t.sr_m - sr0) * ((t.n as f64) - 1.0).max(1.0).sqrt() / denom;
    Deflated { n_trials: n, sr0_ann: sr0 * 12f64.sqrt(), dsr: norm_cdf(z) }
}

/// Mostra todos os testes já feitos, descontados pelo número final de tentativas.
pub fn print_all(path: &Path) -> Result<()> {
    let all = load(path);
    if all.is_empty() {
        println!("Registro vazio ({}).", path.display());
        return Ok(());
    }
    println!("{} testes no registro ({}):", all.len(), path.display());
    println!(
        "{:<16} {:<31} {:<10} {:>6} {:>12} {:>8} {:>6}",
        "estudo", "teste", "período", "meses", "Sharpe a.a.", "acaso*", "DSR"
    );
    for t in &all {
        let d = deflate(t, &all);
        println!(
            "{:<16} {:<31} {:<10} {:>6} {:>12.2} {:>8.2} {:>5.0}%",
            t.estudo,
            t.nome,
            t.periodo,
            t.n,
            t.sr_m * 12f64.sqrt(),
            d.sr0_ann,
            100.0 * d.dsr
        );
    }
    println!("* Sharpe que o melhor de N testes daria sem habilidade nenhuma. DSR = chance de o Sharpe verdadeiro superar isso.");
    Ok(())
}

/// Registra e devolve a linha de texto com o desconto por múltiplos testes.
pub fn record_and_describe(path: &Path, t: &Trial) -> Result<String> {
    let all = upsert(path, t)?;
    let d = deflate(t, &all);
    Ok(format!(
        "{}: Sharpe {:.2} a.a. | {} testes no registro; o melhor deles daria ~{:.2} por acaso | DSR {:.0}%",
        t.nome,
        t.sr_m * 12f64.sqrt(),
        d.n_trials,
        d.sr0_ann,
        100.0 * d.dsr
    ))
}
