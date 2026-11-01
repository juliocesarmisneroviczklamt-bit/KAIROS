//! Downloads over HTTPS with the system's native TLS (schannel on Windows, Security.framework on
//! macOS, OpenSSL elsewhere). Only with the `net` feature (on by default); without it this crate
//! is pure Rust and every download returns an error, which is how the Kairós language uses it.

use anyhow::Result;
use std::path::Path;

#[cfg(feature = "net")]
pub use with_net::*;

#[cfg(feature = "net")]
mod with_net {
    use super::*;
    use anyhow::{Context, anyhow};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    pub type Agent = ureq::Agent;

    /// Cliente HTTP com o TLS nativo do sistema: nada de C pra compilar no Windows.
    pub fn agent() -> Result<Agent> {
        let tls = native_tls::TlsConnector::new().context("inicializando TLS")?;
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

    pub fn get_text(agent: &Agent, url: &str) -> Result<String> {
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

    pub fn download(agent: &Agent, url: &str, dest: &Path) -> Result<u64> {
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
}

/// Built without the `net` feature: no downloads.
#[cfg(not(feature = "net"))]
pub struct Agent;

#[cfg(not(feature = "net"))]
fn no_net() -> anyhow::Error {
    anyhow::anyhow!("quant-b3 foi compilado sem o recurso `net`: downloads indisponíveis")
}

#[cfg(not(feature = "net"))]
pub fn agent() -> Result<Agent> {
    Err(no_net())
}

#[cfg(not(feature = "net"))]
pub fn get_text(_agent: &Agent, _url: &str) -> Result<String> {
    Err(no_net())
}

#[cfg(not(feature = "net"))]
pub fn download(_agent: &Agent, _url: &str, _dest: &Path) -> Result<u64> {
    Err(no_net())
}
