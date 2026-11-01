use anyhow::{Context, Result, anyhow};
use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Cliente HTTP com TLS nativo do Windows (schannel): nada de C pra compilar.
pub fn agent() -> Result<ureq::Agent> {
    let tls = native_tls::TlsConnector::new().context("inicializando TLS (schannel)")?;
    Ok(ureq::AgentBuilder::new()
        .tls_connector(Arc::new(tls))
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) quant-b3/0.1")
        .timeout_connect(Duration::from_secs(30))
        .timeout_read(Duration::from_secs(180))
        .build())
}

fn backoff(attempt: u32) {
    thread::sleep(Duration::from_millis(700u64 << attempt));
}

pub fn get_text(agent: &ureq::Agent, url: &str) -> Result<String> {
    let mut last = String::new();
    for attempt in 0..5 {
        match agent.get(url).call() {
            Ok(resp) => match resp.into_string() {
                Ok(s) => return Ok(s),
                Err(e) => last = e.to_string(),
            },
            Err(ureq::Error::Status(code, _)) if code == 404 || code == 400 => {
                return Err(anyhow!("HTTP {code}: {url}"));
            }
            Err(e) => last = e.to_string(),
        }
        backoff(attempt);
    }
    Err(anyhow!("falhou após 5 tentativas: {url}: {last}"))
}

pub fn download(agent: &ureq::Agent, url: &str, dest: &Path) -> Result<u64> {
    let tmp = dest.with_extension("part");
    let mut last = String::new();
    for attempt in 0..5 {
        let res: Result<u64> = (|| {
            let resp = agent.get(url).call()?;
            let mut reader = resp.into_reader();
            let mut f = std::fs::File::create(&tmp)?;
            Ok(std::io::copy(&mut reader, &mut f)?)
        })();
        match res {
            Ok(n) => {
                std::fs::rename(&tmp, dest)?;
                return Ok(n);
            }
            Err(e) => {
                last = format!("{e:#}");
                backoff(attempt);
            }
        }
    }
    let _ = std::fs::remove_file(&tmp);
    Err(anyhow!("download falhou: {url}: {last}"))
}
