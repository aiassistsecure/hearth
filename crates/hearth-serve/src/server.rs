//! The llama-server child: spawn, watch, stop.
//!
//! hearth does not do inference. llama.cpp has spent years on kernels and
//! samplers; the supervisor's job is to keep that work running and tell
//! the truth about whether it is. One child per model, one port per child,
//! stdout/stderr captured to files so a crash leaves evidence.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

/// Runtime backend selected for one supervised model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// GGUF via llama.cpp's llama-server.
    LlamaCpp,
    /// Hugging Face safetensors directory via vLLM's OpenAI-compatible server.
    Vllm,
}

/// Everything needed to start one model's serving child.
#[derive(Debug, Clone)]
pub struct ServerSpec {
    /// Model name as declared ("muse", "llama3:latest").
    pub model: String,
    /// Runtime-facing model path: GGUF file for llama.cpp, model directory for vLLM.
    pub gguf: PathBuf,
    /// Backend that owns inference for this child.
    pub backend: Backend,
    /// Port to serve on.
    pub port: u16,
    /// The llama-server binary. Default: "llama-server" on PATH.
    pub binary: PathBuf,
    /// Context size. 0 = llama-server's default.
    ///
    /// Note this is the TOTAL context divided across slots by llama-server, so
    /// `--ctx 8192 --parallel 4` gives each concurrent request 2048 — not 8192
    /// each. Sizing it as if every slot got the full window is the most common
    /// way a "parallel" server starts truncating prompts.
    pub ctx: u32,
    /// Concurrent request slots. This is what makes a server able to answer two
    /// callers at once; llama-server defaults to ONE, which means every request
    /// after the first waits for the one in front of it.
    ///
    /// hearth defaults to 8 because a production node fronting a router is
    /// answering more than one caller, and a serving stack that silently
    /// serializes looks exactly like a slow model. Each slot costs its own KV
    /// cache, which is why the budget wants to know.
    pub parallel: u32,
    /// Layers to offload to the GPU. `None` means all of them (`-1`).
    ///
    /// llama.cpp's own default is 0 — CPU. A model runner that quietly runs a
    /// 20 GiB model on the CPU is not slow by a little; it is two orders of
    /// magnitude off, and it looks like a hardware problem rather than a
    /// missing flag. If the model did not fit the planner would already have
    /// refused it, so "all of them" is the honest default here.
    pub gpu_layers: Option<i32>,
    /// Lock the model's pages into RAM (`--mlock`). OFF by default, and the
    /// default matters: with all layers offloaded to the GPU, the host copy of
    /// the weights is cold after load — mlock pins it in RAM anyway, turning
    /// droppable page cache into unreclaimable locked memory. On a box whose
    /// RAM is the same size as its card (a 48 GiB A6000 host with 48 GiB RAM,
    /// say), locking two 20 GiB models is a system-OOM assist, and it landed
    /// on exactly that box within hours of shipping as a default.
    ///
    /// Turn it on for CPU inference, where a paged-out weight page really does
    /// turn the first token after an idle period into a disk read.
    pub mlock: bool,
    /// Continuous batching. Only meaningful with more than one slot, and the
    /// reason parallel slots actually help instead of just interleaving.
    pub cont_batching: bool,
    /// Extra args passed through verbatim, after ours.
    pub extra_args: Vec<String>,
    /// Directory for stdout/stderr capture files.
    pub log_dir: PathBuf,
}

impl ServerSpec {
    pub fn new(model: impl Into<String>, gguf: impl Into<PathBuf>, port: u16) -> ServerSpec {
        ServerSpec {
            model: model.into(),
            gguf: gguf.into(),
            backend: Backend::LlamaCpp,
            port,
            binary: PathBuf::from("llama-server"),
            ctx: 0,
            parallel: 8,
            gpu_layers: None,
            mlock: false,
            cont_batching: true,
            extra_args: Vec::new(),
            log_dir: std::env::temp_dir(),
        }
    }

    /// Construct a vLLM-backed safetensors model.
    pub fn new_safetensors(
        model: impl Into<String>,
        model_dir: impl Into<PathBuf>,
        port: u16,
    ) -> ServerSpec {
        let mut spec = ServerSpec::new(model, model_dir, port);
        spec.backend = Backend::Vllm;
        spec.binary = PathBuf::from("vllm");
        spec.parallel = 0;
        spec.cont_batching = false;
        spec
    }

    pub fn endpoint(&self) -> String {
        format!("127.0.0.1:{}", self.port)
    }

    /// The argv we will actually run — pure, so tests can assert it without
    /// spawning anything.
    pub fn argv(&self) -> Vec<String> {
        let mut args = match self.backend {
            Backend::LlamaCpp => {
                let mut args = vec![
                    "-m".to_string(),
                    self.gguf.display().to_string(),
                    "--host".to_string(),
                    "127.0.0.1".to_string(),
                    "--port".to_string(),
                    self.port.to_string(),
                ];
                if self.ctx > 0 {
                    args.push("-c".to_string());
                    args.push(self.ctx.to_string());
                }
                // All layers on the GPU unless told otherwise. llama.cpp defaults to 0.
                args.push("--n-gpu-layers".to_string());
                args.push(match self.gpu_layers {
                    Some(n) => n.to_string(),
                    None => "-1".to_string(),
                });
                if self.parallel > 0 {
                    args.push("--parallel".to_string());
                    args.push(self.parallel.to_string());
                }
                if self.cont_batching && self.parallel > 1 {
                    args.push("--cont-batching".to_string());
                }
                if self.mlock {
                    args.push("--mlock".to_string());
                }
                args
            }
            Backend::Vllm => {
                let mut args = vec![
                    "serve".to_string(),
                    self.gguf.display().to_string(),
                    "--host".to_string(),
                    "127.0.0.1".to_string(),
                    "--port".to_string(),
                    self.port.to_string(),
                    "--served-model-name".to_string(),
                    self.model.clone(),
                ];
                if self.ctx > 0 {
                    args.push("--max-model-len".to_string());
                    args.push(self.ctx.to_string());
                }
                args
            }
        };
        // Caller's args go LAST so an operator can override any default above.
        args.extend(self.extra_args.iter().cloned());
        args
    }

    fn log_path(&self, stream: &str) -> PathBuf {
        // Model names can carry ':' and '/'; keep filenames boring.
        let safe: String = self
            .model
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' || c == '.' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        self.log_dir.join(format!("hearth-{safe}-{stream}.log"))
    }
}

/// A running (or exited) serving child.
pub struct ServerChild {
    pub spec: ServerSpec,
    child: Child,
}

impl ServerChild {
    /// Spawn the child. Fails fast if the model path is missing.
    pub fn spawn(spec: ServerSpec) -> Result<ServerChild, String> {
        if !spec.gguf.exists() {
            return Err(format!("model path not found: {}", spec.gguf.display()));
        }
        let stdout =
            std::fs::File::create(spec.log_path("out")).map_err(|e| format!("log file: {e}"))?;
        let stderr =
            std::fs::File::create(spec.log_path("err")).map_err(|e| format!("log file: {e}"))?;
        let mut cmd = Command::new(&spec.binary);
        // A fetched runtime keeps its shared libraries next to the binary;
        // without this, a prebuilt llama-server dies on ggml .so lookups and
        // reads as "started, then died".
        if let Some(dir) = spec.binary.parent() {
            if dir.components().count() > 0 {
                let existing = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
                let joined = if existing.is_empty() {
                    dir.display().to_string()
                } else {
                    format!("{}:{existing}", dir.display())
                };
                cmd.env("LD_LIBRARY_PATH", joined);
            }
        }
        let child = cmd
            .args(spec.argv())
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr))
            .spawn()
            .map_err(|e| format!("spawn {}: {e}", spec.binary.display()))?;
        Ok(ServerChild { spec, child })
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Has the process exited? `None` = still running.
    pub fn exit_code(&mut self) -> Option<Option<i32>> {
        match self.child.try_wait() {
            Ok(Some(status)) => Some(status.code()),
            Ok(None) => None,
            Err(_) => Some(None),
        }
    }

    /// Stop deliberately. SIGKILL is honest here: llama-server holds no
    /// state worth a graceful window, and a supervisor that waits politely
    /// on a wedged child is a supervisor that hangs.
    pub fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    /// Where the child's stderr went, for post-mortems.
    pub fn stderr_log(&self) -> PathBuf {
        self.spec.log_path("err")
    }
}

impl Drop for ServerChild {
    fn drop(&mut self) {
        // Never leak a serving child past its supervisor.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Find a free port by asking the OS for one and releasing it. Racy in
/// principle, fine in practice for a single supervisor per host.
pub fn free_port() -> Result<u16, String> {
    std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .map_err(|e| e.to_string())
}

/// Does `binary` exist and run? Checked once at startup so a missing
/// llama-server is one clear error, not one confusing error per model.
pub fn runtime_available(binary: &Path) -> bool {
    Command::new(binary)
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argv_is_exactly_what_we_run() {
        // Exact, deliberately. A loose assertion here would not have noticed
        // that hearth shipped WITHOUT --n-gpu-layers or --parallel, which meant
        // llama.cpp's own defaults applied: layers on the CPU and one slot.
        // This test failing is the correct response to changing that contract.
        let mut spec = ServerSpec::new("muse", "/models/muse.gguf", 8080);
        spec.ctx = 4096;
        spec.extra_args = vec!["-t".into(), "2".into()];
        assert_eq!(
            spec.argv(),
            vec![
                "-m",
                "/models/muse.gguf",
                "--host",
                "127.0.0.1",
                "--port",
                "8080",
                "-c",
                "4096",
                // Production defaults, in the order argv() emits them.
                "--n-gpu-layers",
                "-1",
                "--parallel",
                "8",
                "--cont-batching",
                // The operator's own arguments last, so they win.
                "-t",
                "2"
            ]
        );
    }

    #[test]
    fn vllm_argv_serves_a_safetensors_directory() {
        let mut spec = ServerSpec::new_safetensors("imagine", "/models/imagine-v8", 9000);
        spec.ctx = 8192;
        assert_eq!(
            spec.argv(),
            vec![
                "serve",
                "/models/imagine-v8",
                "--host",
                "127.0.0.1",
                "--port",
                "9000",
                "--served-model-name",
                "imagine",
                "--max-model-len",
                "8192"
            ]
        );
    }

    #[test]
    fn zero_ctx_means_no_flag() {
        let spec = ServerSpec::new("muse", "/models/muse.gguf", 8080);
        assert!(!spec.argv().contains(&"-c".to_string()));
    }

    #[test]
    fn log_names_are_boring() {
        let spec = ServerSpec::new("hf:owner/repo:Q4_K_M", "/m.gguf", 1);
        let p = spec.log_path("err");
        let name = p.file_name().unwrap().to_str().unwrap();
        assert_eq!(name, "hearth-hf_owner_repo_Q4_K_M-err.log");
    }

    #[test]
    fn missing_gguf_fails_before_spawn() {
        let spec = ServerSpec::new("ghost", "/definitely/not/here.gguf", 1);
        let err = match ServerChild::spawn(spec) {
            Err(e) => e,
            Ok(_) => panic!("spawn succeeded on a missing gguf"),
        };
        assert!(err.contains("model path not found"), "{err}");
    }

    #[test]
    fn missing_binary_is_one_clear_answer() {
        assert!(!runtime_available(Path::new(
            "/definitely/not/llama-server"
        )));
    }

    #[test]
    fn a_real_child_runs_exits_and_reports() {
        // Use /bin/sh as a stand-in runtime: spawns, sleeps briefly, exits 7.
        let dir = tempfile::tempdir().unwrap();
        let gguf = dir.path().join("fake.gguf");
        std::fs::write(&gguf, b"not a real gguf").unwrap();
        let mut spec = ServerSpec::new("fake", &gguf, 1);
        spec.binary = PathBuf::from("/bin/sh");
        spec.extra_args = vec!["-c".into(), "exit 7".into()];
        // /bin/sh ignores our -m/--host args? No — it errors on them, which
        // is fine: we only assert the child lifecycle, not the exit value.
        spec.log_dir = dir.path().to_path_buf();
        let mut child = ServerChild::spawn(spec).unwrap();
        assert!(child.pid() > 0);
        // Wait for exit (bounded).
        let mut code = None;
        for _ in 0..50 {
            if let Some(c) = child.exit_code() {
                code = Some(c);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(code.is_some(), "child never exited");
    }
}

#[cfg(test)]
mod production_defaults {
    use super::*;

    fn argv_of(spec: &ServerSpec) -> Vec<String> {
        spec.argv()
    }

    fn pair(argv: &[String], flag: &str) -> Option<String> {
        argv.iter()
            .position(|a| a == flag)
            .map(|i| argv[i + 1].clone())
    }

    #[test]
    fn all_layers_go_to_the_gpu_by_default() {
        // llama.cpp's own default is 0 — CPU. hearth shipped without this flag
        // at all, so a 20 GiB model ran wherever the build happened to default,
        // and CPU inference on a serving node reads as broken hardware rather
        // than as a missing argument.
        let argv = argv_of(&ServerSpec::new("m", "/m.gguf", 8080));
        assert_eq!(pair(&argv, "--n-gpu-layers").as_deref(), Some("-1"));
    }

    #[test]
    fn a_node_answers_more_than_one_caller_by_default() {
        // llama-server defaults to ONE slot: every request after the first
        // waits for the one in front of it, which looks exactly like a slow
        // model rather than a serialized one.
        let argv = argv_of(&ServerSpec::new("m", "/m.gguf", 8080));
        assert_eq!(pair(&argv, "--parallel").as_deref(), Some("8"));
        assert!(
            argv.iter().any(|a| a == "--cont-batching"),
            "extra slots without continuous batching just queue differently"
        );
    }

    #[test]
    fn one_slot_means_no_continuous_batching() {
        // With a single slot the flag is noise, and noise in an argv is how a
        // reader stops trusting the rest of it.
        let spec = ServerSpec {
            parallel: 1,
            ..ServerSpec::new("m", "/m.gguf", 8080)
        };
        assert!(!spec.argv().iter().any(|a| a == "--cont-batching"));
    }

    #[test]
    fn the_operators_own_arguments_win() {
        // Defaults are a starting point, not a policy. Ours go first so the
        // caller's --n-gpu-layers overrides rather than conflicts.
        let spec = ServerSpec {
            extra_args: vec!["--n-gpu-layers".into(), "20".into()],
            ..ServerSpec::new("m", "/m.gguf", 8080)
        };
        let argv = spec.argv();
        let ours = argv.iter().position(|a| a == "--n-gpu-layers").unwrap();
        let theirs = argv.iter().rposition(|a| a == "--n-gpu-layers").unwrap();
        assert!(theirs > ours, "the caller's copy must come last to win");
        assert_eq!(argv[theirs + 1], "20");
    }

    #[test]
    fn gpu_layers_can_be_pinned_including_to_cpu() {
        // A CPU-only box is a real deployment; it just must be chosen, not
        // arrived at by accident.
        let spec = ServerSpec {
            gpu_layers: Some(0),
            ..ServerSpec::new("m", "/m.gguf", 8080)
        };
        assert_eq!(pair(&spec.argv(), "--n-gpu-layers").as_deref(), Some("0"));
    }

    #[test]
    fn mlock_is_opt_in_because_it_pins_host_ram_the_gpu_path_does_not_need() {
        // Shipped as an always-on default first, and it hit a real box within
        // hours: 48 GiB A6000 host with 48 GiB RAM, two 20 GiB models — mlock
        // pinned the cold host copies and helped the system toward OOM. With
        // full GPU offload the host copy is droppable page cache; locking it
        // buys nothing and costs the model's size in RAM.
        assert!(!ServerSpec::new("m", "/m.gguf", 8080)
            .argv()
            .iter()
            .any(|a| a == "--mlock"));
        // Still there for CPU inference, where it earns its keep.
        let spec = ServerSpec {
            mlock: true,
            ..ServerSpec::new("m", "/m.gguf", 8080)
        };
        assert!(spec.argv().iter().any(|a| a == "--mlock"));
    }

    #[test]
    fn the_context_is_still_bound_to_the_port_and_model() {
        // The defaults must not have displaced the arguments that identify
        // WHICH model on WHICH port — an argv bug here looks like the model
        // being broken.
        let spec = ServerSpec {
            ctx: 8192,
            ..ServerSpec::new("muse", "/models/muse.gguf", 9001)
        };
        let argv = spec.argv();
        assert_eq!(pair(&argv, "-m").as_deref(), Some("/models/muse.gguf"));
        assert_eq!(pair(&argv, "--port").as_deref(), Some("9001"));
        assert_eq!(pair(&argv, "--host").as_deref(), Some("127.0.0.1"));
        assert_eq!(pair(&argv, "-c").as_deref(), Some("8192"));
    }
}
