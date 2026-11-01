//! quant-b3: laboratório de pesquisa de alpha na B3.
//!
//! Biblioteca usada pelo executável `quant-b3` e pelo runtime da linguagem Kairós (crate `kairos/`),
//! que lê os mesmos dados e é julgada pelo mesmo simulador, registro de testes e paper trading.

pub mod b3events;
pub mod backtest;
pub mod common;
pub mod cotahist;
pub mod http;
pub mod macrodata;
pub mod momentum;
pub mod panel;
pub mod paper;
pub mod realistic;
pub mod registry;
pub mod risk;
pub mod search;
pub mod smallcaps;
pub mod stats;
pub mod store;
pub mod universe;
pub mod util;
