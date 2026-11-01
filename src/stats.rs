//! Estatística básica + OLS com erros-padrão de Newey-West (robustos a autocorrelação).

pub fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    x.iter().sum::<f64>() / x.len() as f64
}

pub fn std(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return f64::NAN;
    }
    let m = mean(x);
    (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n - 1) as f64).sqrt()
}

pub fn corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 3 {
        return f64::NAN;
    }
    let (mx, my) = (mean(&x[..n]), mean(&y[..n]));
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (a, b) = (x[i] - mx, y[i] - my);
        sxy += a * b;
        sxx += a * a;
        syy += b * b;
    }
    sxy / (sxx * syy).sqrt()
}

/// Retorno composto anualizado; `ppy` = períodos por ano.
pub fn cagr(x: &[f64], ppy: f64) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    let w: f64 = x.iter().map(|r| 1.0 + r).product();
    w.powf(ppy / x.len() as f64) - 1.0
}

pub fn max_dd(x: &[f64]) -> f64 {
    let (mut w, mut peak, mut mdd) = (1.0f64, 1.0f64, 0.0f64);
    for r in x {
        w *= 1.0 + r;
        peak = peak.max(w);
        mdd = mdd.min(w / peak - 1.0);
    }
    mdd
}

/// Regra de bolso de Newey-West para o número de defasagens.
pub fn nw_lags(n: usize) -> usize {
    (4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize
}

pub struct Ols {
    /// [intercepto, betas...]
    pub coef: Vec<f64>,
    pub t: Vec<f64>,
    pub r2: f64,

}

/// y = a + X b + e, com t-stats de Newey-West.
pub fn ols_nw(y: &[f64], xs: &[&[f64]], lags: usize) -> Option<Ols> {
    let n = y.len();
    let k = xs.len() + 1;
    if n <= k + 2 || xs.iter().any(|x| x.len() != n) {
        return None;
    }
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut r = Vec::with_capacity(k);
            r.push(1.0);
            for x in xs {
                r.push(x[i]);
            }
            r
        })
        .collect();
    let mut xtx = vec![vec![0.0; k]; k];
    let mut xty = vec![0.0; k];
    for i in 0..n {
        for a in 0..k {
            xty[a] += rows[i][a] * y[i];
            for b in 0..k {
                xtx[a][b] += rows[i][a] * rows[i][b];
            }
        }
    }
    let inv = invert(&xtx)?;
    let beta: Vec<f64> = (0..k).map(|a| (0..k).map(|b| inv[a][b] * xty[b]).sum()).collect();
    let u: Vec<f64> = (0..n).map(|i| y[i] - (0..k).map(|a| rows[i][a] * beta[a]).sum::<f64>()).collect();
    let mut s = vec![vec![0.0; k]; k];
    for i in 0..n {
        for a in 0..k {
            for b in 0..k {
                s[a][b] += u[i] * u[i] * rows[i][a] * rows[i][b];
            }
        }
    }
    for l in 1..=lags.min(n - 1) {
        let w = 1.0 - l as f64 / (lags as f64 + 1.0);
        for i in l..n {
            let uu = u[i] * u[i - l];
            for a in 0..k {
                for b in 0..k {
                    s[a][b] += w * uu * (rows[i][a] * rows[i - l][b] + rows[i - l][a] * rows[i][b]);
                }
            }
        }
    }
    let v = matmul(&matmul(&inv, &s), &inv);
    let t = (0..k).map(|a| beta[a] / v[a][a].sqrt()).collect();
    let ym = mean(y);
    let sst: f64 = y.iter().map(|v| (v - ym) * (v - ym)).sum();
    let sse: f64 = u.iter().map(|v| v * v).sum();
    Some(Ols { coef: beta, t, r2: 1.0 - sse / sst })
}

fn invert(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let k = m.len();
    let mut a: Vec<Vec<f64>> = m
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let mut row = r.clone();
            row.extend((0..k).map(|j| if i == j { 1.0 } else { 0.0 }));
            row
        })
        .collect();
    for c in 0..k {
        let piv = (c..k).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[piv][c].abs() < 1e-14 {
            return None;
        }
        a.swap(c, piv);
        let d = a[c][c];
        for v in a[c].iter_mut() {
            *v /= d;
        }
        for r in 0..k {
            if r == c {
                continue;
            }
            let f = a[r][c];
            if f != 0.0 {
                for j in 0..2 * k {
                    let t = a[c][j];
                    a[r][j] -= f * t;
                }
            }
        }
    }
    Some(a.into_iter().map(|r| r[k..].to_vec()).collect())
}

fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (n, m, p) = (a.len(), b.len(), b[0].len());
    let mut c = vec![vec![0.0; p]; n];
    for i in 0..n {
        for k in 0..m {
            for j in 0..p {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}
