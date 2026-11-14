fn main() {
    // The build script runs on the host: check the *target* architecture (cross-compiling the
    // x86_64 `asm` feature from an aarch64 Mac is fine).
    #[cfg(feature = "asm")]
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("x86_64") {
        eprintln!("Currently feature `asm` can only be enabled on x86_64 arch.");
        std::process::exit(1);
    }
}
