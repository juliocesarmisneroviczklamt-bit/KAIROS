use serde::{Deserialize, Serialize};

/// Um pregão de um papel. Preços por ação, em R$, sem ajuste (já divididos pelo FATCOT).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Bar {
    pub date: u32,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub avg: f64,
    /// melhor oferta de compra e de venda no fechamento (por ação)
    pub bid: f64,
    pub ask: f64,
    /// volume financeiro (R$)
    pub vol_brl: f64,
    /// quantidade de ações negociadas
    pub qty: f64,
    pub trades: u32,
    pub fatcot: u32,
    pub bdi: u8,
    /// campo ESPECI cru (classe + marcadores ex-evento + segmento de listagem)
    pub especi: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Series {
    /// código atual (o mais recente, se houve troca de código)
    pub ticker: String,
    /// códigos anteriores encadeados por ISIN
    pub aliases: Vec<String>,
    pub isins: Vec<String>,
    /// nomes de pregão (NOMRES) com primeira e última data em que apareceram
    pub names: Vec<(String, u32, u32)>,
    /// ON, PN, PNA, ..., UNT
    pub class: String,
    pub bars: Vec<Bar>,
}

impl Series {
    pub fn issuer(&self) -> String {
        issuer_of(&self.ticker)
    }
    pub fn first(&self) -> u32 {
        self.bars.first().map(|b| b.date).unwrap_or(0)
    }
    pub fn last(&self) -> u32 {
        self.bars.last().map(|b| b.date).unwrap_or(0)
    }
    /// índice do último pregão com data <= d
    pub fn idx_on_or_before(&self, d: u32) -> Option<usize> {
        let i = self.bars.partition_point(|b| b.date <= d);
        if i == 0 { None } else { Some(i - 1) }
    }
    /// índice do primeiro pregão com data > d
    pub fn idx_after(&self, d: u32) -> Option<usize> {
        let i = self.bars.partition_point(|b| b.date <= d);
        if i < self.bars.len() { Some(i) } else { None }
    }
}

pub fn issuer_of(t: &str) -> String {
    t.chars().take(4).collect()
}

#[derive(Serialize, Deserialize, Default)]
pub struct BarStore {
    pub series: Vec<Series>,
}
