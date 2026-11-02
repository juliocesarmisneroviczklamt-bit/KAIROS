//! `cfg(kairos_jit)`: the native back-end (Cranelift JIT) is built where Cranelift supports the
//! target. Keep in sync with the target-specific dependencies in Cargo.toml.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(kairos_jit)");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let jit = match arch.as_str() {
        "x86_64" => matches!(os.as_str(), "linux" | "macos" | "windows" | "freebsd"),
        "aarch64" => matches!(os.as_str(), "linux" | "macos"),
        _ => false,
    };
    if jit {
        println!("cargo::rustc-cfg=kairos_jit");
    }
}
