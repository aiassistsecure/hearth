//! `hearth runtime` — fetch a prebuilt llama-server instead of hazing the
//! operator with a compiler.
//!
//! "Install a CUDA toolchain and build from source" is not a prerequisite a
//! drop-in replacement gets to have — half of Ollama's adoption story is that
//! it ships as one binary. llama.cpp publishes prebuilt server binaries on
//! every release; hearth already owns a resuming, digest-conscious downloader.
//! This module connects the two.
//!
//! The one honest wrinkle, checked against the live release listing rather
//! than assumed: **llama.cpp ships prebuilt CUDA builds for Windows but not
//! for Linux.** On a Linux NVIDIA box the zero-compile path is the VULKAN
//! build, which runs on the NVIDIA driver's own Vulkan at a modest cost
//! against CUDA. So the policy is: prebuilt Vulkan by default — working in
//! minutes — and the CUDA source build documented as the *optimization*,
//! never the prerequisite.

use std::path::{Path, PathBuf};

/// What kind of box is this, as far as picking a runtime build goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Platform {
    pub os: Os,
    pub arch: Arch,
    /// An NVIDIA driver is present (`nvidia-smi` runs). On Linux this selects
    /// the Vulkan build — the driver ships its own Vulkan ICD, so no CUDA
    /// toolkit is needed.
    pub nvidia: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Os {
    Linux,
    Mac,
    Windows,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    X64,
    Arm64,
}

impl Platform {
    /// Detect the current box.
    pub fn detect() -> Platform {
        let os = if cfg!(target_os = "macos") {
            Os::Mac
        } else if cfg!(target_os = "windows") {
            Os::Windows
        } else {
            Os::Linux
        };
        let arch = if cfg!(target_arch = "aarch64") {
            Arch::Arm64
        } else {
            Arch::X64
        };
        let nvidia = std::process::Command::new("nvidia-smi")
            .arg("-L")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        Platform { os, arch, nvidia }
    }
}

/// Where hearth publishes its own CUDA build of llama-server — the one
/// upstream does not ship for Linux. Built by CI on a stock runner (compiling
/// CUDA needs the toolkit, not a GPU) and preferred over Vulkan when present.
pub const HEARTH_CUDA_RELEASE: &str =
    "https://github.com/Eth-Interchained/hearth/releases/download/runtime-cuda";

/// The hearth-built CUDA asset for a platform, when one can exist.
/// Only Linux — Windows already has upstream CUDA, macOS has Metal.
pub fn hearth_cuda_asset(p: Platform) -> Option<String> {
    match (p.os, p.arch, p.nvidia) {
        (Os::Linux, Arch::X64, true) => Some(format!(
            "{HEARTH_CUDA_RELEASE}/llama-server-cuda-linux-x64.tar.gz"
        )),
        _ => None,
    }
}

/// Which UPSTREAM release asset serves this platform best without a compiler.
///
/// Pure, and the whole policy lives here so it is testable against the real
/// asset names from the release listing:
///
///   * macOS builds carry Metal already — one build per arch, done.
///   * Linux + NVIDIA gets the VULKAN build as the FALLBACK: upstream ships no
///     Linux CUDA prebuilt, and Vulkan rides the driver the box already has.
///     hearth''s own CI-built CUDA asset is tried first (`hearth_cuda_asset`).
///   * Linux without a GPU gets the plain CPU build.
///   * Windows + NVIDIA does have an upstream CUDA prebuilt; use it.
///
/// Returns the asset name pattern with `{tag}` where the release tag goes.
pub fn asset_pattern(p: Platform) -> &'static str {
    match (p.os, p.arch, p.nvidia) {
        (Os::Mac, Arch::Arm64, _) => "llama-{tag}-bin-macos-arm64.tar.gz",
        (Os::Mac, Arch::X64, _) => "llama-{tag}-bin-macos-x64.tar.gz",
        (Os::Linux, Arch::X64, true) => "llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz",
        (Os::Linux, Arch::X64, false) => "llama-{tag}-bin-ubuntu-x64.tar.gz",
        (Os::Linux, Arch::Arm64, true) => "llama-{tag}-bin-ubuntu-vulkan-arm64.tar.gz",
        (Os::Linux, Arch::Arm64, false) => "llama-{tag}-bin-ubuntu-arm64.tar.gz",
        (Os::Windows, Arch::X64, true) => "llama-{tag}-bin-win-cuda-12.4-x64.zip",
        (Os::Windows, Arch::X64, false) => "llama-{tag}-bin-win-cpu-x64.zip",
        (Os::Windows, Arch::Arm64, _) => "llama-{tag}-bin-win-cpu-arm64.zip",
    }
}

/// A note for the operator about what they got and what they could upgrade to.
/// Printed, because a silent perf tradeoff is a mystery benchmark later.
pub fn tradeoff_note(p: Platform) -> Option<&'static str> {
    match (p.os, p.nvidia) {
        (Os::Linux, true) => Some(
            "this is the Vulkan build — zero compile, runs on the NVIDIA driver you \
             already have. For maximum throughput, a CUDA source build is the \
             optimization (cmake -DGGML_CUDA=ON); hearth will prefer it automatically \
             if you install one on PATH.",
        ),
        _ => None,
    }
}

/// Where the fetched runtime lives: `$HEARTH_HOME/runtime`.
pub fn runtime_dir(hearth_home: &Path) -> PathBuf {
    hearth_home.join("runtime")
}

/// The llama-server inside a fetched runtime, if one has been fetched.
pub fn fetched_server(hearth_home: &Path) -> Option<PathBuf> {
    // Release tarballs unpack to build/bin/llama-server; we normalize to
    // runtime/bin at extract time, and check both in case of older fetches.
    for rel in ["bin/llama-server", "build/bin/llama-server"] {
        let p = runtime_dir(hearth_home).join(rel);
        if p.exists() {
            return Some(p);
        }
    }
    None
}


/// Where hearth keeps its isolated vLLM environment.
pub fn vllm_dir(hearth_home: &Path) -> PathBuf {
    runtime_dir(hearth_home).join("vllm")
}

/// The vLLM CLI inside hearth's managed virtual environment, if installed.
pub fn fetched_vllm(hearth_home: &Path) -> Option<PathBuf> {
    let dir = vllm_dir(hearth_home);
    let candidates = if cfg!(target_os = "windows") {
        vec![dir.join("Scripts").join("vllm.exe")]
    } else {
        vec![dir.join("bin").join("vllm")]
    };
    candidates.into_iter().find(|p| p.exists())
}

/// Resolve the vLLM CLI with the same operator-first policy as llama-server:
///
///   1. HEARTH_VLLM
///   2. vllm on PATH
///   3. hearth-managed isolated runtime
pub fn resolve_vllm(hearth_home: &Path, path_has_vllm: bool) -> Resolved {
    if let Ok(explicit) = std::env::var("HEARTH_VLLM") {
        return Resolved::Explicit(PathBuf::from(explicit));
    }
    if path_has_vllm {
        return Resolved::OnPath;
    }
    match fetched_vllm(hearth_home) {
        Some(p) => Resolved::Fetched(p),
        None => Resolved::Missing,
    }
}

/// Which llama-server should a command use, in order of preference:
///
///   1. `HEARTH_LLAMA_SERVER` — the operator said exactly which one.
///   2. `llama-server` on PATH — an operator-installed build (a CUDA source
///      build lands here) beats our fetched fallback, because it was chosen.
///   3. The runtime `hearth runtime` fetched.
///
/// PATH beating the fetched build is deliberate: the fetched Vulkan build is
/// the floor, not the ceiling, and an operator who compiled CUDA should win
/// without unsetting anything.
pub fn resolve_server(hearth_home: &Path, path_has_server: bool) -> Resolved {
    if let Ok(explicit) = std::env::var("HEARTH_LLAMA_SERVER") {
        return Resolved::Explicit(PathBuf::from(explicit));
    }
    if path_has_server {
        return Resolved::OnPath;
    }
    match fetched_server(hearth_home) {
        Some(p) => Resolved::Fetched(p),
        None => Resolved::Missing,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resolved {
    Explicit(PathBuf),
    OnPath,
    Fetched(PathBuf),
    /// Nothing anywhere — and the error should say `hearth runtime` now
    /// exists, not "go compile".
    Missing,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `HEARTH_LLAMA_SERVER` is process-wide, and `resolve_server` READS it
    /// before anything else — so every test that calls `resolve_server` shares
    /// state with every test that sets it, even the ones that never touch it.
    ///
    /// The comment that used to sit on the env test said the opposite:
    /// "tests in this file do not race on this var because only this one
    /// touches it." True of WRITERS, and irrelevant — the race is with the
    /// READER. Rust runs a test binary's tests as parallel threads in ONE
    /// process, so with unlucky scheduling
    /// `resolution_order_lets_an_operators_cuda_build_win` observed the var
    /// mid-set and got `Explicit(...)` where it asserted `Missing`. It failed
    /// exactly that way in CI on the v0.4.4 tag, having passed on v0.4.3 with
    /// identical code — which is what a scheduling race looks like from the
    /// outside, and why the test was believed correct for two releases.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Take the lock and guarantee a clean slate. `into_inner` on poison
    /// because a panic in one test must not cascade into "all the others
    /// failed too", which hides the one that actually broke.
    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        let g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("HEARTH_LLAMA_SERVER");
        std::env::remove_var("HEARTH_VLLM");
        g
    }

    fn plat(os: Os, arch: Arch, nvidia: bool) -> Platform {
        Platform { os, arch, nvidia }
    }

    // Asset names below are from the REAL b10673 release listing, checked
    // live — not invented to match the code.

    #[test]
    fn linux_with_nvidia_gets_vulkan_because_no_linux_cuda_prebuilt_exists() {
        // The fact that prompted this module: llama.cpp ships win-cuda but no
        // ubuntu-cuda. Vulkan rides the NVIDIA driver's own ICD — zero
        // compile, which is the entire point.
        assert_eq!(
            asset_pattern(plat(Os::Linux, Arch::X64, true)),
            "llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz"
        );
        assert!(tradeoff_note(plat(Os::Linux, Arch::X64, true))
            .unwrap()
            .contains("CUDA source build is the optimization"));
    }

    #[test]
    fn linux_without_a_gpu_gets_the_cpu_build_not_a_vulkan_it_cannot_use() {
        assert_eq!(
            asset_pattern(plat(Os::Linux, Arch::X64, false)),
            "llama-{tag}-bin-ubuntu-x64.tar.gz"
        );
        assert!(tradeoff_note(plat(Os::Linux, Arch::X64, false)).is_none());
    }

    #[test]
    fn macos_needs_no_gpu_flag_because_metal_is_in_the_build() {
        for nvidia in [true, false] {
            assert_eq!(
                asset_pattern(plat(Os::Mac, Arch::Arm64, nvidia)),
                "llama-{tag}-bin-macos-arm64.tar.gz"
            );
        }
    }

    #[test]
    fn windows_with_nvidia_gets_the_cuda_build_that_actually_exists_there() {
        assert_eq!(
            asset_pattern(plat(Os::Windows, Arch::X64, true)),
            "llama-{tag}-bin-win-cuda-12.4-x64.zip"
        );
    }

    #[test]
    fn resolution_order_lets_an_operators_cuda_build_win() {
        let _env = env_guard();
        // Unique per process: a fixed shared name under temp_dir is the same
        // class of hazard as the shared env var, one `cargo test` away from
        // biting.
        let dir = std::env::temp_dir().join(format!("hearth-rt-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        // Nothing anywhere: Missing, and the caller points at `hearth runtime`.
        assert_eq!(resolve_server(&dir, false), Resolved::Missing);

        // A fetched runtime is found…
        std::fs::create_dir_all(runtime_dir(&dir).join("bin")).unwrap();
        std::fs::write(runtime_dir(&dir).join("bin/llama-server"), b"x").unwrap();
        assert!(matches!(resolve_server(&dir, false), Resolved::Fetched(_)));

        // …but PATH beats it: an operator who compiled CUDA chose to, and
        // must win without unsetting anything.
        assert_eq!(resolve_server(&dir, true), Resolved::OnPath);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_explicit_env_var_beats_everything() {
        // Holds ENV_LOCK for the whole set/read/remove window. Without it the
        // window is visible to every other test that calls `resolve_server`.
        let _env = env_guard();
        std::env::set_var("HEARTH_LLAMA_SERVER", "/opt/llama/llama-server");
        let r = resolve_server(Path::new("/nowhere"), true);
        std::env::remove_var("HEARTH_LLAMA_SERVER");
        assert_eq!(
            r,
            Resolved::Explicit(PathBuf::from("/opt/llama/llama-server"))
        );
    }


    #[test]
    fn vllm_resolution_finds_hearths_managed_environment() {
        let _env = env_guard();
        let dir = std::env::temp_dir().join(format!("hearth-vllm-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(resolve_vllm(&dir, false), Resolved::Missing);

        let cli = if cfg!(target_os = "windows") {
            vllm_dir(&dir).join("Scripts").join("vllm.exe")
        } else {
            vllm_dir(&dir).join("bin").join("vllm")
        };
        std::fs::create_dir_all(cli.parent().unwrap()).unwrap();
        std::fs::write(&cli, b"x").unwrap();

        assert_eq!(resolve_vllm(&dir, false), Resolved::Fetched(cli));
        assert_eq!(resolve_vllm(&dir, true), Resolved::OnPath);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn explicit_vllm_override_beats_path_and_managed_runtime() {
        let _env = env_guard();
        std::env::set_var("HEARTH_VLLM", "/opt/vllm/bin/vllm");
        let r = resolve_vllm(Path::new("/nowhere"), true);
        std::env::remove_var("HEARTH_VLLM");
        assert_eq!(
            r,
            Resolved::Explicit(PathBuf::from("/opt/vllm/bin/vllm"))
        );
    }
}
