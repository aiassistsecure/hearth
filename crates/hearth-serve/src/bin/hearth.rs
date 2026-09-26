//! hearth — keep declared models warm, and tell the truth about which are.
//!
//!   hearth serve --model NAME (--gguf FILE | --safetensors DIR) [--port N]
//!                [--vram-gib N] [--ctx N] [--binary PATH] [--total-gib N] [--once]
//!   hearth status
//!   hearth why MODEL
//!   hearth as-of MODEL SEQ
//!   hearth verify
//!
//! `status`, `why`, `as-of` and `verify` read the on-disk spine directly —
//! they answer from the recorded history whether or not a supervisor is
//! running, because the history is the database, not a process's memory.

use std::io::Write;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// SIGINT/SIGTERM latch. The handler only flips a bool — all real work
/// (stop children, record `unloaded`, flush the spine) happens on the
/// supervise loop, because a signal handler that touches a database is a
/// signal handler that corrupts one. Found live: without this, killing
/// the supervisor leaked the llama-server child and lost the final events.
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_sig: i32) {
    SHUTDOWN.store(true, Ordering::SeqCst);
}

fn install_signal_handlers() {
    // SAFETY: on_signal is async-signal-safe — it writes one atomic bool.
    unsafe {
        extern "C" {
            fn signal(signum: i32, handler: extern "C" fn(i32)) -> usize;
        }
        const SIGINT: i32 = 2;
        const SIGTERM: i32 = 15;
        signal(SIGINT, on_signal);
        signal(SIGTERM, on_signal);
    }
}

use hearth_core::{Budget, Declared, GIB};
use hearth_serve::server::{free_port, runtime_available, Backend, ServerSpec};
use hearth_serve::{hearth_home, Supervisor};
use hearth_store::Spine;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("--version") | Some("-V") => {
            println!("hearth {}", env!("CARGO_PKG_VERSION"));
            println!("commands: up|preload|pull|serve|run|runtime|status|why|as-of|verify");
            return ExitCode::SUCCESS;
        }
        Some("serve") => cmd_serve(&args[1..]),
        Some("run") => cmd_run(&args[1..]),
        Some("status") => cmd_status(),
        Some("why") => cmd_why(&args[1..]),
        Some("as-of") => cmd_as_of(&args[1..]),
        Some("verify") => cmd_verify(),
        Some("pull") => cmd_pull(&args[1..]),
        Some("up") => cmd_up(&args[1..]),
        Some("preload") => cmd_preload(&args[1..]),
        Some("runtime") => cmd_runtime(&args[1..]),
        _ => {
            eprintln!(
                "usage: hearth up|preload|pull|serve|run|runtime|status|why|as-of|verify (see crate docs)"
            );
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("hearth: {e}");
            ExitCode::FAILURE
        }
    }
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn open_spine() -> Result<Spine, String> {
    let dir = hearth_home().join("spine");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Spine::open(&dir)
}

/// `hearth pull <reference>` — fetch weights, verify the digest, record it.
///
/// The recording is the part no other tool does. Months later, "where did this
/// file come from" is `hearth why <model>` rather than an archaeology project.
/// A tunable: command-line flag wins, then the environment, then the default.
///
/// Flags beat env deliberately — an operator typing a flag is expressing intent
/// NOW, while an env var is standing configuration. The reverse order makes a
/// debugging session fight the deployment.
fn tunable<T: std::str::FromStr>(args: &[String], flag_name: &str, env: &str, default: T) -> T {
    if let Some(v) = flag(args, flag_name) {
        if let Ok(t) = v.parse() {
            return t;
        }
        eprintln!("hearth: {flag_name} {v:?} is not a valid value — using the default");
    }
    if let Ok(v) = std::env::var(env) {
        if let Ok(t) = v.parse() {
            return t;
        }
        eprintln!("hearth: {env}={v:?} is not a valid value — using the default");
    }
    default
}

/// Every value of a repeatable flag, accepting both `--k v` and `--k=v` —
/// Mark writes `--preload-model=name`, scripts often write `--preload-model name`,
/// and a flag that only honours one spelling silently drops the other.
fn flag_all(args: &[String], name: &str) -> Vec<String> {
    let mut out = Vec::new();
    let prefix = format!("{name}=");
    let mut i = 0;
    while i < args.len() {
        if let Some(v) = args[i].strip_prefix(&prefix) {
            out.push(v.to_string());
        } else if args[i] == name {
            if let Some(v) = args.get(i + 1) {
                out.push(v.clone());
                i += 1;
            }
        }
        i += 1;
    }
    out
}

/// One-shot form of `flag_all` — last occurrence wins, both spellings.
fn flag_eq(args: &[String], name: &str) -> Option<String> {
    flag_all(args, name).pop()
}

/// Warm one resident model: a single-token generation straight at its
/// endpoint. Returns milliseconds on success.
fn warm_one(endpoint: &str, timeout: Duration) -> Result<u128, String> {
    use std::io::{Read, Write};
    let addr: std::net::SocketAddr = endpoint
        .parse()
        .map_err(|e| format!("bad endpoint {endpoint}: {e}"))?;
    let started = std::time::Instant::now();
    let mut s = std::net::TcpStream::connect_timeout(&addr, timeout)
        .map_err(|e| format!("connect: {e}"))?;
    // Generous read deadline rather than none: warmup is the one place a
    // bound is right, because it is our OWN throwaway request — a warmup that
    // hangs must not wedge the warmer thread for the rest of the night.
    let _ = s.set_read_timeout(Some(Duration::from_secs(300)));
    let body = hearth_serve::warmup::warmup_request_body();
    let req = format!(
        "POST /completion HTTP/1.1\r\nhost: {endpoint}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes())
        .map_err(|e| format!("write: {e}"))?;
    let mut buf = Vec::new();
    let _ = s.read_to_end(&mut buf);
    let head = String::from_utf8_lossy(&buf);
    let status: Option<u16> = head.split_whitespace().nth(1).and_then(|c| c.parse().ok());
    if hearth_serve::warmup::warmup_succeeded(status) {
        Ok(started.elapsed().as_millis())
    } else {
        Err(format!(
            "warmup answered {status:?}: {}",
            head.chars().take(120).collect::<String>()
        ))
    }
}

/// Warm an OpenAI-compatible child directly. vLLM does not expose llama.cpp's
/// /completion route, so mixed fleets use this for safetensors children.
fn warm_one_openai(endpoint: &str, model: &str, timeout: Duration) -> Result<u128, String> {
    use std::io::{Read, Write};

    let addr: std::net::SocketAddr = endpoint
        .parse()
        .map_err(|e| format!("bad endpoint {endpoint}: {e}"))?;
    let started = std::time::Instant::now();
    let mut s = std::net::TcpStream::connect_timeout(&addr, timeout)
        .map_err(|e| format!("connect: {e}"))?;
    let _ = s.set_read_timeout(Some(Duration::from_secs(300)));
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "stream": false
    })
    .to_string();
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nhost: {endpoint}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes())
        .map_err(|e| format!("write: {e}"))?;
    let mut buf = Vec::new();
    let _ = s.read_to_end(&mut buf);
    let head = String::from_utf8_lossy(&buf);
    let status: Option<u16> = head.split_whitespace().nth(1).and_then(|c| c.parse().ok());
    if matches!(status, Some(200..=299)) {
        Ok(started.elapsed().as_millis())
    } else {
        Err(format!(
            "warmup answered {status:?}: {}",
            head.chars().take(120).collect::<String>()
        ))
    }
}

/// `hearth preload NAME [NAME…] | '*'` — warm models on a RUNNING fleet.
///
/// Goes through the gateway's /residency to find endpoints, then fires the
/// one-token generation at each. `*` warms every model that is ready.
fn cmd_preload(args: &[String]) -> Result<(), String> {
    let port: u16 = tunable(args, "--port", "HEARTH_PORT", 11434);
    // Positionals are what remains after flags AND their values. A naive
    // "everything not starting with --" filter swallowed the VALUE of
    // `--port 18266` as a model name and then reported that 18266 was not
    // declared — true, useless, and confusing. Found by running it.
    let names: Vec<String> = {
        let mut out = Vec::new();
        let mut i = 0;
        while i < args.len() {
            if args[i].starts_with("--") {
                // `--k=v` is one token; `--k v` is two.
                i += if args[i].contains('=') { 1 } else { 2 };
            } else {
                out.push(args[i].clone());
                i += 1;
            }
        }
        out
    };
    if names.is_empty() {
        return Err(
            "usage: hearth preload MODEL [MODEL…] | '*'   (against a running `hearth up`)".into(),
        );
    }

    // Ask the running fleet what exists and where.
    let health = hearth_serve::probe::probe_http(
        &format!("127.0.0.1:{port}"),
        "/residency",
        Duration::from_secs(3),
    );
    let body = match &health {
        hearth_serve::probe::ProbeResult::Ok => {
            // probe_http discards bodies; fetch it plainly.
            fetch_local_json(port, "/residency")?
        }
        other => {
            return Err(format!(
                "no hearth gateway answering on 127.0.0.1:{port} ({other:?}) — start one with `hearth up`"
            ))
        }
    };
    let v: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| format!("/residency was not json: {e}"))?;
    let models = v["models"].as_array().cloned().unwrap_or_default();

    let wanted_all = names.iter().any(|n| n == "*");
    let mut warmed = 0usize;
    let mut skipped: Vec<String> = Vec::new();
    for m in &models {
        let name = m["model"].as_str().unwrap_or("");
        if !wanted_all && !names.iter().any(|n| n == name) {
            continue;
        }
        if m["ready"] != serde_json::json!(true) {
            skipped.push(format!("{name} ({})", m["state"].as_str().unwrap_or("?")));
            continue;
        }
        let endpoint = m["endpoint"].as_str().unwrap_or("");
        match warm_one(endpoint, Duration::from_secs(5)) {
            Ok(ms) => {
                println!("warmed {name} in {ms}ms");
                warmed += 1;
            }
            Err(e) => eprintln!("hearth: {name} warmup FAILED — {e}"),
        }
    }
    if !wanted_all {
        for n in &names {
            if !models
                .iter()
                .any(|m| m["model"] == serde_json::json!(n.as_str()))
            {
                eprintln!("hearth: {n} is not declared on this fleet — check `hearth preload '*'` output or /v1/models");
            }
        }
    }
    for s in &skipped {
        eprintln!("hearth: skipped {s} — not ready; it will be warm the moment it turns resident if the fleet was started with preload on");
    }
    if warmed == 0 && skipped.is_empty() {
        return Err("nothing matched — nothing warmed".into());
    }
    Ok(())
}

/// GET a small JSON body off the local gateway. Bounded, plain, no deps.
fn fetch_local_json(port: u16, path: &str) -> Result<String, String> {
    use std::io::{Read, Write};
    let mut s = std::net::TcpStream::connect_timeout(
        &format!("127.0.0.1:{port}").parse().unwrap(),
        Duration::from_secs(3),
    )
    .map_err(|e| format!("connect: {e}"))?;
    let _ = s.set_read_timeout(Some(Duration::from_secs(5)));
    s.write_all(
        format!("GET {path} HTTP/1.1\r\nhost: 127.0.0.1:{port}\r\nconnection: close\r\n\r\n")
            .as_bytes(),
    )
    .map_err(|e| format!("write: {e}"))?;
    let mut buf = Vec::new();
    let _ = s.read_to_end(&mut buf);
    let raw = String::from_utf8_lossy(&buf);
    raw.split_once("\r\n\r\n")
        .map(|(_, b)| b.to_string())
        .ok_or_else(|| "no body in the gateway's answer".into())
}

/// POST JSON to the local OpenAI-compatible gateway.
fn post_local_json(port: u16, path: &str, body: &str) -> Result<String, String> {
    use std::io::{Read, Write};

    let mut s = std::net::TcpStream::connect_timeout(
        &format!("127.0.0.1:{port}").parse().unwrap(),
        Duration::from_secs(3),
    )
    .map_err(|e| format!("connect: {e}"))?;
    let _ = s.set_read_timeout(Some(Duration::from_secs(600)));

    let req = format!(
        "POST {path} HTTP/1.1\r\nhost: 127.0.0.1:{port}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes())
        .map_err(|e| format!("write: {e}"))?;

    let mut buf = Vec::new();
    s.read_to_end(&mut buf)
        .map_err(|e| format!("read: {e}"))?;

    let raw = String::from_utf8_lossy(&buf);
    let (head, body) = raw
        .split_once("\r\n\r\n")
        .ok_or_else(|| "no body in the gateway's answer".to_string())?;
    let status = head.split_whitespace().nth(1).unwrap_or("?");
    if !status.starts_with('2') {
        return Err(format!(
            "gateway answered HTTP {status}: {}",
            body.chars().take(500).collect::<String>()
        ));
    }
    Ok(body.to_string())
}

/// `hearth run MODEL` — an Ollama-style interactive terminal on top of the
/// same OpenAI-compatible gateway used by applications. The REPL is a client,
/// not a second inference path: residency, routing and provenance remain under
/// the normal hearth supervisor.
fn cmd_run(args: &[String]) -> Result<(), String> {
    use std::io::{self, BufRead};

    let model = args
        .first()
        .filter(|s| !s.starts_with("--"))
        .ok_or("usage: hearth run MODEL [--port N]")?
        .clone();
    let port: u16 = tunable(args, "--port", "HEARTH_PORT", 11434);

    // Fail fast with a useful message before dropping the user into a prompt.
    let models = fetch_local_json(port, "/v1/models").map_err(|e| {
        format!(
            "no hearth gateway answering on 127.0.0.1:{port} ({e}) — start the fleet with `hearth up`"
        )
    })?;
    let listed: serde_json::Value =
        serde_json::from_str(&models).map_err(|e| format!("/v1/models was not json: {e}"))?;
    let exists = listed["data"]
        .as_array()
        .map(|xs| xs.iter().any(|m| m["id"].as_str() == Some(model.as_str())))
        .unwrap_or(false);
    if !exists {
        return Err(format!(
            "model {model:?} is not declared by the running fleet — check `hearth status` or GET /v1/models"
        ));
    }

    println!("hearth run {model}");
    println!("  /clear  clear conversation · /exit  quit");
    println!();

    let stdin = io::stdin();
    let mut input = stdin.lock();
    let mut messages: Vec<serde_json::Value> = Vec::new();

    loop {
        print!(">>> ");
        std::io::stdout().flush().map_err(|e| e.to_string())?;

        let mut line = String::new();
        if input.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
            println!();
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match line {
            "/exit" | "/quit" => break,
            "/clear" => {
                messages.clear();
                println!("conversation cleared");
                continue;
            }
            _ => {}
        }

        messages.push(serde_json::json!({
            "role": "user",
            "content": line
        }));
        let request = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": false
        });
        let body = post_local_json(port, "/v1/chat/completions", &request.to_string())?;
        let response: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| format!("gateway returned invalid json: {e}"))?;
        if let Some(err) = response.get("error") {
            return Err(format!("generation failed: {err}"));
        }
        let answer = response["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| {
                format!(
                    "gateway answer had no choices[0].message.content: {}",
                    body.chars().take(500).collect::<String>()
                )
            })?
            .to_string();

        println!("{answer}");
        println!();
        messages.push(serde_json::json!({
            "role": "assistant",
            "content": answer
        }));
    }

    Ok(())
}

/// `hearth runtime` — fetch a prebuilt llama-server. No compiler, no CUDA
/// toolkit, no prerequisite beyond curl and tar.
///
/// Preference on Linux+NVIDIA: hearth's OWN CI-built CUDA binary (upstream
/// ships none for Linux), falling back to upstream's Vulkan build — which
/// rides the driver's own ICD. Everything else gets upstream's native build.
/// The llama-server this box should use, and a clear error when none exists.
/// Order: HEARTH_LLAMA_SERVER > PATH > the runtime `hearth runtime` fetched.
fn default_binary() -> Result<std::path::PathBuf, String> {
    use hearth_pull::runtime as rt;
    let on_path = std::process::Command::new("llama-server")
        .arg("--version")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    match rt::resolve_server(&hearth_home(), on_path) {
        rt::Resolved::Explicit(p) => Ok(p),
        rt::Resolved::OnPath => Ok("llama-server".into()),
        rt::Resolved::Fetched(p) => Ok(p),
        rt::Resolved::Missing => Err(
            "no llama-server found. Run `hearth runtime` to fetch a prebuilt one \
             (no compiler needed), or install llama.cpp yourself and put \
             llama-server on PATH."
                .into(),
        ),
    }
}

fn cmd_runtime(args: &[String]) -> Result<(), String> {
    let target = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .map(String::as_str)
        .unwrap_or("all");

    match target {
        "all" => {
            install_llama_runtime(args)?;
            install_vllm_runtime(args)?;
            Ok(())
        }
        "llama" | "llama.cpp" => install_llama_runtime(args),
        "vllm" => install_vllm_runtime(args),
        other => Err(format!(
            "unknown runtime {other:?}; use: hearth runtime [all|llama|vllm] [--force]"
        )),
    }
}

/// Resolve the vLLM CLI for safetensors serving.
/// Order: HEARTH_VLLM > PATH > the isolated runtime installed by hearth runtime.
fn default_vllm_binary() -> Result<std::path::PathBuf, String> {
    use hearth_pull::runtime as rt;

    let on_path = std::process::Command::new("vllm")
        .arg("--version")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    match rt::resolve_vllm(&hearth_home(), on_path) {
        rt::Resolved::Explicit(p) => Ok(p),
        rt::Resolved::OnPath => Ok("vllm".into()),
        rt::Resolved::Fetched(p) => Ok(p),
        rt::Resolved::Missing => Err(
            "no vLLM runtime found. Run `hearth runtime` to install HEARTH's managed vLLM environment, or set HEARTH_VLLM to an existing vllm CLI."
                .into(),
        ),
    }
}

fn install_llama_runtime(args: &[String]) -> Result<(), String> {
    use hearth_pull::runtime as rt;

    let plat = rt::Platform::detect();
    let home = hearth_home();
    let dir = rt::runtime_dir(&home);
    let bin_dir = dir.join("bin");
    std::fs::create_dir_all(&bin_dir).map_err(|e| e.to_string())?;

    let force = args.iter().any(|a| a == "--force");
    if let Some(existing) = rt::fetched_server(&home) {
        if !force {
            println!("runtime already fetched at {}", existing.display());
            println!("  (re-fetch with: hearth runtime --force)");
            return Ok(());
        }
    }

    // 1. hearth's own CUDA build, when this box can use one.
    let mut fetched_from = String::new();
    let mut archive = dir.join("runtime.tar.gz");
    if let Some(cuda_url) = rt::hearth_cuda_asset(plat) {
        eprintln!(
            "hearth: trying the CUDA build (built by hearth CI — upstream ships none for linux)…"
        );
        let req = hearth_pull::curl::Request::get(&cuda_url).to_file(&archive);
        match hearth_pull::curl::fetch_file(&req, true) {
            Ok(_) => fetched_from = cuda_url,
            Err(e) => {
                eprintln!("hearth: no CUDA build available yet ({})", first_line(&e.0));
                eprintln!(
                    "hearth: falling back to upstream Vulkan — zero compile, runs on your driver"
                );
            }
        }
    }

    // 2. Upstream's best zero-compile build for this platform.
    if fetched_from.is_empty() {
        let tag = latest_llama_tag()?;
        let asset = rt::asset_pattern(plat).replace("{tag}", &tag);
        let url = format!("https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{asset}");
        eprintln!("hearth: fetching {asset} …");
        archive = dir.join(&asset);
        let req = hearth_pull::curl::Request::get(&url).to_file(&archive);
        hearth_pull::curl::fetch_file(&req, true).map_err(|e| e.0)?;
        fetched_from = url;
    }

    // 3. Extract. Modern bsdtar/libarchive auto-detects both .tar.gz and .zip.
    // Do not force gzip with -z: Windows llama.cpp releases are ZIP archives.
    let out = std::process::Command::new("tar")
        .args([
            "-xf",
            &archive.display().to_string(),
            "-C",
            &dir.display().to_string(),
        ])
        .output()
        .map_err(|e| format!("could not run tar: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "extracting {}: {}",
            archive.display(),
            String::from_utf8_lossy(&out.stderr)
        ));
    }

    // 4. Normalize: whatever layout the tarball used, the server and its
    //    shared libraries end up in runtime/bin. FOUND, not guessed — the
    //    first attempt hardcoded build/bin and bin, and the real b10673
    //    tarball unpacks to llama-<tag>/ with everything flat inside it.
    let server_name = if cfg!(target_os = "windows") {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    let found = find_file(&dir, server_name, 3).ok_or_else(|| {
        format!(
            "the archive did not contain {server_name} anywhere under {}",
            dir.display()
        )
    })?;
    let src = found.parent().unwrap_or(&dir).to_path_buf();
    let mut moved = 0usize;
    if src != bin_dir {
        for entry in std::fs::read_dir(&src).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            if entry.path().is_dir() {
                continue;
            }
            let to = bin_dir.join(entry.file_name());
            if to.exists() {
                std::fs::remove_file(&to)
                    .map_err(|e| format!("removing stale runtime file {}: {e}", to.display()))?;
            }
            std::fs::copy(entry.path(), &to).map_err(|e| {
                format!(
                    "copying runtime file {} to {}: {e}",
                    entry.path().display(),
                    to.display()
                )
            })?;
            moved += 1;
        }
    }
    let server = bin_dir.join(server_name);
    if !server.exists() {
        return Err(format!(
            "the archive did not contain {server_name} where expected — look in {}",
            dir.display()
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&server, std::fs::Permissions::from_mode(0o755));
    }
    let _ = std::fs::remove_file(&archive);

    // 5. Smoke it. A binary that cannot even print its version must not be
    //    reported as an installed runtime.
    let ver = std::process::Command::new(&server)
        .arg("--version")
        .env("LD_LIBRARY_PATH", &bin_dir)
        .output();
    match ver {
        Ok(o) if o.status.success() || !o.stderr.is_empty() => {
            let v = String::from_utf8_lossy(if o.stdout.is_empty() {
                &o.stderr
            } else {
                &o.stdout
            });
            println!("runtime ready: {}", server.display());
            println!("  {}", first_line(v.trim()));
            println!("  from {fetched_from}  ({moved} file(s))");
        }
        Ok(o) => {
            return Err(format!(
                "fetched, but llama-server --version failed (exit {:?}) — a missing \
                 system library is the usual cause; run it by hand to see which: \
                 LD_LIBRARY_PATH={} {} --version",
                o.status.code(),
                bin_dir.display(),
                server.display()
            ));
        }
        Err(e) => return Err(format!("fetched, but could not run it: {e}")),
    }
    if let Some(note) = rt::tradeoff_note(plat) {
        if !fetched_from.contains("runtime-cuda") {
            println!("  note: {note}");
        }
    }
    println!("  `hearth up` and `hearth serve` will use it automatically.");
    Ok(())
}

fn install_vllm_runtime(args: &[String]) -> Result<(), String> {
    use hearth_pull::runtime as rt;

    let home = hearth_home();
    let dir = rt::vllm_dir(&home);
    let force = args.iter().any(|a| a == "--force");

    if let Some(existing) = rt::fetched_vllm(&home) {
        if !force {
            println!("vLLM runtime already installed at {}", existing.display());
            println!("  (re-install with: hearth runtime vllm --force)");
            return Ok(());
        }
        std::fs::remove_dir_all(&dir)
            .map_err(|e| format!("removing old vLLM runtime {}: {e}", dir.display()))?;
    }

    if !cfg!(target_os = "linux") {
        return Err(
            "HEARTH-managed vLLM is currently supported on Linux; set HEARTH_VLLM to a working vllm CLI on this platform"
                .into(),
        );
    }

    let python = ["python3", "python"]
        .into_iter()
        .find(|name| {
            std::process::Command::new(name)
                .arg("--version")
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        })
        .ok_or("python3 is required to install HEARTH's managed vLLM runtime")?;

    if let Some(parent) = dir.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    eprintln!(
        "hearth: creating isolated vLLM runtime at {} …",
        dir.display()
    );
    let status = std::process::Command::new(python)
        .args(["-m", "venv"])
        .arg(&dir)
        .status()
        .map_err(|e| format!("could not create vLLM virtualenv with {python}: {e}"))?;
    if !status.success() {
        return Err(format!(
            "{python} -m venv failed — install the Python venv module for this interpreter"
        ));
    }

    let python_bin = dir.join("bin").join("python");
    eprintln!("hearth: installing vLLM into the isolated runtime …");
    let status = std::process::Command::new(&python_bin)
        .args(["-m", "pip", "install", "--upgrade", "vllm"])
        .status()
        .map_err(|e| format!("could not run pip for vLLM: {e}"))?;
    if !status.success() {
        return Err("installing vLLM into HEARTH's isolated runtime failed".into());
    }

    let vllm = rt::fetched_vllm(&home).ok_or_else(|| {
        format!(
            "vLLM installed, but its CLI was not found under {}",
            dir.display()
        )
    })?;
    let ver = std::process::Command::new(&vllm)
        .arg("--version")
        .output()
        .map_err(|e| format!("vLLM installed but could not be executed: {e}"))?;
    if !ver.status.success() {
        return Err(format!(
            "vLLM installed but --version failed: {}",
            first_line(String::from_utf8_lossy(&ver.stderr).trim())
        ));
    }

    let shown = if ver.stdout.is_empty() {
        String::from_utf8_lossy(&ver.stderr)
    } else {
        String::from_utf8_lossy(&ver.stdout)
    };
    println!("vLLM runtime ready: {}", vllm.display());
    println!("  {}", first_line(shown.trim()));
    println!("  HEARTH will use it automatically for safetensors models.");
    Ok(())
}

/// Breadth-limited search for a file by name. Depth-capped so a hostile or
/// malformed archive cannot walk us anywhere expensive.
fn find_file(root: &std::path::Path, name: &str, depth: u8) -> Option<std::path::PathBuf> {
    if depth == 0 {
        return None;
    }
    let entries = std::fs::read_dir(root).ok()?;
    let mut dirs = Vec::new();
    for e in entries.flatten() {
        let p = e.path();
        if p.is_file() && e.file_name() == name {
            return Some(p);
        }
        if p.is_dir() {
            dirs.push(p);
        }
    }
    dirs.into_iter()
        .find_map(|d| find_file(&d, name, depth - 1))
}

fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or(s)
}

/// The newest llama.cpp release tag, from the GitHub API via curl.
fn latest_llama_tag() -> Result<String, String> {
    let body = hearth_pull::curl::fetch_string(
        &hearth_pull::curl::Request::get(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=1",
        )
        .header("Accept", "application/vnd.github+json")
        .header("User-Agent", "hearth"),
    )
    .map_err(|e| e.0)?;
    let v: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| format!("release listing was not json: {e}"))?;
    v.get(0)
        .and_then(|r| r.get("tag_name"))
        .and_then(|t| t.as_str())
        .map(String::from)
        .ok_or_else(|| "no releases in the listing".into())
}

fn cmd_pull(args: &[String]) -> Result<(), String> {
    let reference = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .ok_or("usage: hearth pull REFERENCE   (e.g. tinyllama, ollama:library/llama3:latest, file:./muse.gguf)")?;

    let cfg = hearth_pull::PullConfig {
        blobs_dir: match flag(args, "--blobs") {
            Some(d) => d.into(),
            None => hearth_home().join("blobs"),
        },
        // A silent multi-gigabyte pause is indistinguishable from a hang.
        progress: !args.iter().any(|a| a == "--quiet"),
        verify_existing: args.iter().any(|a| a == "--verify-existing"),
    };

    let spine = open_spine()?;
    eprintln!("hearth: pulling {reference} …");
    let out = hearth_pull::pull(reference, &cfg, &spine)?;
    spine.flush();

    let gib = out.bytes as f64 / GIB as f64;
    if out.already_had_it {
        println!("{} already here — {:.2} GiB", out.model, gib);
    } else {
        println!("{} pulled and verified — {:.2} GiB", out.model, gib);
    }
    println!("  from  {}", out.source);
    println!("  at    {}", out.model_path.display());
    println!();
    if out.model_path.is_dir() {
        println!(
            "  hearth serve --model {} --safetensors {}",
            out.model,
            out.model_path.display()
        );
    } else {
        println!(
            "  hearth serve --model {} --gguf {}",
            out.model,
            out.model_path.display()
        );
    }
    Ok(())
}

/// `hearth up` — a whole fleet behind one OpenAI-compatible port.
///
/// This is the shape that replaces a model runner rather than supplementing
/// one: declare a roster, hearth refuses what will not fit, brings up the rest,
/// and answers on a single endpoint — routing by the `model` field the way
/// every OpenAI client already sends it.
///
/// The part nothing else does is what happens when it CANNOT serve you. A
/// model still loading is a 503 with Retry-After; a model whose GPU the host
/// reclaimed is a 503 that says so in the body; a model that will never fit on
/// this card is a 409, because a retryable status there is a router hammering a
/// box that is arithmetically incapable of answering.
fn cmd_up(args: &[String]) -> Result<(), String> {
    // Everything after a bare `--` belongs to llama-server, not to hearth.
    // This is the ONLY route by which an operator reaches the runtime's own
    // knobs — `-ub`, `--cache-reuse`, `-fa`, `--metrics`. Before this split
    // existed, `hearth up` read its known flags and silently ignored the rest,
    // so fleet.conf's documented `extra` directive was a no-op: the flags were
    // parsed by start.sh, appended to hearth's argv, and dropped on the floor.
    // Found by `ps -ww -eo args | grep llama-server` on a production box whose
    // fleet.conf said `extra --jinja` and whose children had no `--jinja`.
    let (args, passthrough) = split_passthrough(args);
    let specs = collect_models(args)?;
    if specs.is_empty() {
        return Err(
            "usage: hearth up --model NAME=[gguf:|safetensors:]PATH[:GIB][@CTX] [--model …] \
                    [--port 11434] [--total-gib 48]"
                .into(),
        );
    }
    let api_port: u16 = tunable(args, "--port", "HEARTH_PORT", 11434);
    let total_gib: u64 = flag(args, "--total-gib")
        .map(|v| v.parse().map_err(|e| format!("--total-gib: {e}")))
        .transpose()?
        .unwrap_or(24);
    // Every knob: flag beats env beats default, so a deployment sets
    // HEARTH_* once and a debugging session overrides per-run.
    let ctx: u32 = tunable(args, "--ctx", "HEARTH_CTX", 0);
    // Concurrent slots per model. llama-server's own default is 1, which
    // serializes every caller behind the one in front of them. 8 because a
    // production node fronting a router is answering more than one caller,
    // and each slot's KV cache is already inside the declared budget.
    let parallel: u32 = tunable(args, "--parallel", "HEARTH_PARALLEL", 8);
    let gpu_layers: Option<i32> = {
        let v: i32 = tunable(args, "--gpu-layers", "HEARTH_GPU_LAYERS", -1);
        if v < 0 {
            None
        } else {
            Some(v)
        }
    };
    // mlock is opt-in: with full GPU offload the host weight copy is droppable
    // page cache, and locking it costs the model's size in RAM for nothing.
    // For CPU inference (--gpu-layers 0) it is worth turning on.
    let mlock: bool = args.iter().any(|a| a == "--mlock")
        || std::env::var("HEARTH_MLOCK").ok().as_deref() == Some("1");

    let budget = Budget {
        total_bytes: total_gib * GIB,
        reserve_bytes: 2 * GIB,
    };
    // Declaration order IS priority order. The operator listed these in the
    // order that matters to them; first fit, never best fit.
    // The PLANNER must see the same ctx the child will actually run with, or
    // it budgets KV for a context nobody uses — in either direction.
    let declared: Vec<Declared> = specs
        .iter()
        .map(|s| Declared {
            model: s.name.clone(),
            weights_bytes: s.gib * GIB,
            kv_bytes: match s.backend {
                Backend::LlamaCpp => kv_bytes_for_model(&s.path, s.ctx.unwrap_or(ctx), parallel),
                Backend::Vllm => 0,
            },
        })
        .collect();

    install_signal_handlers();
    let sup = Arc::new(Mutex::new(Supervisor::new(open_spine()?, budget, declared)));

    // Start only what the budget admitted. Anything refused stays declared and
    // visible — /residency reports it with the exact shortfall, rather than it
    // silently not existing.
    let log_dir = hearth_home().join("logs");
    std::fs::create_dir_all(&log_dir).map_err(|e| e.to_string())?;
    for ms in &specs {
        let name = &ms.name;
        let port = free_port()?;
        let mut spec = match ms.backend {
            Backend::LlamaCpp => ServerSpec::new(name, &ms.path, port),
            Backend::Vllm => ServerSpec::new_safetensors(name, &ms.path, port),
        };
        spec.ctx = ms.ctx.unwrap_or(ctx);
        if ms.backend == Backend::LlamaCpp {
            spec.parallel = parallel;
            spec.gpu_layers = gpu_layers;
            spec.mlock = mlock;
            // `extra` remains llama.cpp-specific. Passing llama flags to
            // vLLM would turn a mixed fleet into a runtime argument failure.
            spec.extra_args = passthrough.to_vec();
        }
        spec.binary = match flag(args, "--binary") {
            Some(b) => b.into(),
            None => match ms.backend {
                Backend::LlamaCpp => default_binary()?,
                Backend::Vllm => default_vllm_binary()?,
            },
        };
        spec.log_dir = log_dir.clone();
        if !runtime_available(&spec.binary) {
            let hint = match ms.backend {
                Backend::LlamaCpp => "run `hearth runtime llama` or pass --binary",
                Backend::Vllm => "run `hearth runtime vllm` or pass --binary",
            };
            return Err(format!(
                "runtime not available: {} — {hint}",
                spec.binary.display()
            ));
        }
        // Say the ctx THIS model got. With per-model ctx the one-line banner
        // below can only report the default, and a fleet where the banner says
        // 150000 while a model is actually running 32768 is a banner that
        // misleads exactly when it matters.
        let eff_ctx = spec.ctx;
        match sup.lock().unwrap().start(spec) {
            Ok(()) => {
                let ctx_note = if eff_ctx == 0 {
                    "ctx runtime default".to_string()
                } else if ms.backend == Backend::LlamaCpp {
                    format!(
                        "ctx {eff_ctx} across {parallel} slot(s) = {}/request",
                        eff_ctx / parallel.max(1)
                    )
                } else {
                    format!("ctx {eff_ctx} · vLLM batching")
                };
                let backend = match ms.backend {
                    Backend::LlamaCpp => "llama.cpp/GGUF",
                    Backend::Vllm => "vLLM/safetensors",
                };
                eprintln!(
                    "hearth: {name} loading via {backend} on 127.0.0.1:{port} — {ctx_note} …"
                );
            }
            // A refusal is not a crash. Say it and keep going: the rest of the
            // fleet is still worth serving.
            Err(e) => eprintln!("hearth: {name} not started — {e}"),
        }
    }

    // Auto-preload: which models get a first-token warmup the moment they
    // turn resident. Default is ALL admitted models — preloading is the
    // product; opting out (--preload-models=0) is the special case.
    let preload_n: Option<usize> = flag_eq(args, "--preload-models")
        .or_else(|| std::env::var("HEARTH_PRELOAD_MODELS").ok())
        .and_then(|v| v.parse().ok());
    let preload_named: Vec<String> = {
        let mut named = flag_all(args, "--preload-model");
        if let Ok(env_named) = std::env::var("HEARTH_PRELOAD_MODEL") {
            named.extend(
                env_named
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty()),
            );
        }
        named
    };
    let declared_names: Vec<String> = specs.iter().map(|s| s.name.clone()).collect();
    for unknown in hearth_serve::warmup::unknown_targets(&declared_names, &preload_named) {
        // A typo that silently warms nothing is a cold model discovered by
        // the first user of the day. Loud, not fatal.
        eprintln!("hearth: --preload-model {unknown}: not declared on this fleet — ignoring");
    }
    let warm_list =
        hearth_serve::warmup::warmup_targets(&declared_names, preload_n, &preload_named);

    let addr = format!("127.0.0.1:{api_port}");
    let server = hearth_api::Server::bind(&addr)?;

    // SIGTERM latches SHUTDOWN, but the accept loop below is BLOCKED inside
    // accept(2), which only returns when a connection arrives — so a TERM was
    // ignored until the next request happened to come in, and every clean stop
    // ended in somebody's kill -9, which loses the final unloaded events from
    // the spine. This watcher unblocks accept the moment the latch flips, by
    // connecting to our own port once. Found by `start.sh down` timing out at
    // 10 seconds of TERM on a healthy gateway.
    {
        let addr = addr.clone();
        std::thread::spawn(move || {
            while !SHUTDOWN.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(250));
            }
            let _ = std::net::TcpStream::connect(&addr);
        });
    }
    println!("hearth up on http://{addr}");
    // Print what was actually chosen. A server whose concurrency you have to
    // infer from a config file is a server nobody tunes.
    println!(
        "  {} slot(s)/model · {} · ctx {}",
        parallel,
        match gpu_layers {
            Some(n) => format!("{n} GPU layer(s)"),
            None => "all layers on GPU".to_string(),
        },
        if ctx == 0 {
            "runtime default".to_string()
        } else {
            format!("{ctx} shared across slots")
        },
    );
    println!("  POST /v1/chat/completions   routed by the \"model\" field");
    println!("  GET  /v1/models             what is declared, and what is ready");
    println!("  GET  /residency             the truth the OpenAI shape cannot carry");

    // The warmer: one thread, working through the preload list IN ORDER —
    // warming simultaneously would contend for the GPU while models are still
    // loading, turning a warmup into a slowdown. It waits for each model to
    // turn resident (no deadline: a 32B over a network fabric takes what it
    // takes), fires one throwaway token, and reports the time. After this,
    // the first REAL request finds the graph built and the KV cache ready.
    if !warm_list.is_empty() {
        println!(
            "  preload: {} model(s) will be warmed as they turn resident",
            warm_list.len()
        );
        let warm_backends: std::collections::HashMap<String, Backend> = specs
            .iter()
            .map(|s| (s.name.clone(), s.backend))
            .collect();
        let warmer = Arc::clone(&sup);
        std::thread::spawn(move || {
            for model in warm_list {
                loop {
                    if SHUTDOWN.load(Ordering::SeqCst) {
                        return;
                    }
                    let route = {
                        let s = match warmer.lock() {
                            Ok(s) => s,
                            Err(_) => return,
                        };
                        s.fleet().route(&model, hearth_serve::now_ms())
                    };
                    match route {
                        hearth_core::fleet::Route::Ready { endpoint, .. } => {
                            let result = match warm_backends.get(&model) {
                                Some(Backend::Vllm) => {
                                    warm_one_openai(&endpoint, &model, Duration::from_secs(5))
                                }
                                _ => warm_one(&endpoint, Duration::from_secs(5)),
                            };
                            match result {
                                Ok(ms) => eprintln!(
                                    "hearth: warmed {model} in {ms}ms — first real request will be hot"
                                ),
                                Err(e) => eprintln!("hearth: {model} warmup FAILED — {e}"),
                            }
                            break;
                        }
                        // Never coming (refused, failed, undeclared): move on
                        // rather than wait for a residency that cannot happen.
                        hearth_core::fleet::Route::NotAdmitted { .. }
                        | hearth_core::fleet::Route::NotDeclared { .. }
                        | hearth_core::fleet::Route::Failed { .. } => break,
                        _ => std::thread::sleep(Duration::from_millis(500)),
                    }
                }
            }
        });
    }

    // The supervisor ticks on its own thread so a long generation never stalls
    // residency tracking, and so the accept loop is never the thing holding
    // the fleet lock.
    let ticker = Arc::clone(&sup);
    std::thread::spawn(move || {
        while !SHUTDOWN.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(500));
            let mut s = match ticker.lock() {
                Ok(s) => s,
                Err(_) => break,
            };
            let n = s.tick();
            if n > 0 {
                eprintln!("hearth: {n} transition(s) recorded");
                eprint!("{}", s.report());
            }
        }
    });

    // One thread per connection, bounded. The accept loop used to handle
    // requests inline, which meant a single generation blocked every other
    // caller — parallel slots inside llama-server buy nothing if the gateway
    // in front of them is a queue of one.
    //
    // Bounded rather than unbounded because an unbounded spawn is a way to
    // convert a traffic spike into an OOM. Over the cap we answer 503 with a
    // Retry-After, which is the honest thing: we are busy, come back.
    let inflight = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let max_inflight: usize = tunable(args, "--max-inflight", "HEARTH_MAX_INFLIGHT", 64);

    for conn in server.incoming() {
        if SHUTDOWN.load(Ordering::SeqCst) {
            break;
        }
        let mut stream = match conn {
            Ok(s) => s,
            Err(e) => {
                eprintln!("hearth: accept failed: {e}");
                continue;
            }
        };

        if inflight.load(Ordering::SeqCst) >= max_inflight {
            let body = serde_json::json!({
                "error": {
                    "message": format!(
                        "hearth is at its in-flight limit ({max_inflight}) — this is \
                         backpressure, not a failure"),
                    "type": "server_busy",
                    "retryable": true,
                    "operator_fault": false,
                }
            })
            .to_string();
            let _ =
                stream.write_all(hearth_api::http::render_response(503, &body, Some(1)).as_bytes());
            continue;
        }

        let sup = Arc::clone(&sup);
        let inflight = Arc::clone(&inflight);
        inflight.fetch_add(1, Ordering::SeqCst);
        std::thread::spawn(move || {
            // Decrement no matter how this thread leaves, or the limit ratchets
            // down until the gateway refuses everything.
            struct Guard(Arc<std::sync::atomic::AtomicUsize>);
            impl Drop for Guard {
                fn drop(&mut self) {
                    self.0.fetch_sub(1, Ordering::SeqCst);
                }
            }
            let _guard = Guard(inflight);

            let req = match hearth_api::http::read_request(&stream) {
                Ok(r) => r,
                Err(e) => {
                    let body = serde_json::json!({ "error": { "message": e } }).to_string();
                    let _ = stream
                        .write_all(hearth_api::http::render_response(400, &body, None).as_bytes());
                    return;
                }
            };

            // Decide under the lock; act OUTSIDE it. A generation can take
            // minutes, and holding the fleet lock across one would stall every
            // other request and the supervisor with it.
            let decision = {
                let s = match sup.lock() {
                    Ok(s) => s,
                    Err(_) => return,
                };
                hearth_api::decide(&req.path, &req.body, s.fleet(), hearth_serve::now_ms())
            };

            match decision {
                hearth_api::Decision::Proxy { model, endpoint } => {
                    if let Err(e) = hearth_api::http::proxy(
                        &endpoint,
                        &req,
                        &mut stream,
                        Duration::from_secs(5),
                    ) {
                        eprintln!("hearth: proxy to {model} at {endpoint} failed: {e}");
                        let body = serde_json::json!({
                            "error": { "message": e, "type": "upstream_unreachable" }
                        })
                        .to_string();
                        let _ = stream.write_all(
                            hearth_api::http::render_response(502, &body, None).as_bytes(),
                        );
                    }
                }
                hearth_api::Decision::Answer(r) => {
                    let _ = stream.write_all(
                        hearth_api::http::render_response(r.status, &r.body, r.retry_after)
                            .as_bytes(),
                    );
                }
            }
        });
    }

    eprintln!("hearth: shutting down — recording unloaded, reaping children");
    sup.lock().unwrap().stop_all();
    Ok(())
}

/// `--model NAME=/path/to.gguf` or `--model NAME=/path/to.gguf:20` (GiB).
///
/// The size is what the budget plans against. Without it we guess 4 GiB, which
/// is deliberately conservative: over-guessing refuses models that would fit,
/// and that is a worse failure than admitting one that is slightly tight.
/// `--model NAME=/path/to.gguf[:GIB][@CTX]`
///
/// PER-MODEL CONTEXT, BECAUSE ONE GLOBAL `--ctx` IS THE WRONG SHAPE.
///
/// `--ctx` is a single value stamped onto every child in the start loop, and
/// KV cache is computed from it per model — so a fleet mixing a 1M-context
/// Kimi-Linear with an 8B granite had to pick ONE number for both. Pick it for
/// the big model and the small one wastes the card; pick it for the small one
/// and the big one is pointlessly clamped. On a 48 GiB card with three models
/// that is the difference between all three resident and one refused.
///
/// `@CTX` is optional and falls back to the global `--ctx`, so every existing
/// invocation means exactly what it meant before.
///
/// `@` rather than a third colon: `:GIB:CTX` would be two bare positional
/// numbers whose order you have to remember, and it would collide with the
/// digits-only test that keeps a Windows drive letter from being read as a
/// size. `@` cannot appear in either field.
/// One `--model` spec, parsed. A named struct rather than a tuple because the
/// fourth field made the tuple genuinely hard to read at the call sites — and
/// `gib`/`ctx` are both numbers, so positional access is exactly where a
/// silent mix-up would live.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelSpec {
    name: String,
    path: String,
    backend: Backend,
    gib: u64,
    /// `None` inherits the fleet-wide `--ctx`.
    ctx: Option<u32>,
}

/// Split argv at the first bare `--`: hearth's own flags on the left, the
/// runtime's on the right. Pure so the contract can be asserted without a
/// fleet. A missing `--` means no passthrough — never a guess about which
/// unknown flags "look like" llama-server's.
fn split_passthrough(args: &[String]) -> (&[String], &[String]) {
    match args.iter().position(|a| a == "--") {
        Some(i) => (&args[..i], &args[i + 1..]),
        None => (args, &[]),
    }
}

fn collect_models(args: &[String]) -> Result<Vec<ModelSpec>, String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--model" {
            let spec = args
                .get(i + 1)
                .ok_or("--model needs NAME=[gguf:|safetensors:]PATH[:GIB][@CTX]")?;
            let (name, rest) = spec.split_once('=').ok_or_else(|| {
                format!("--model {spec}: expected NAME=[gguf:|safetensors:]PATH[:GIB][@CTX]")
            })?;
            // Explicit backend prefixes are optional. No prefix keeps the
            // historical GGUF meaning, including Windows paths such as C:\\models\\m.gguf.
            let (backend, rest) = if let Some(path) = rest.strip_prefix("safetensors:") {
                (Backend::Vllm, path)
            } else if let Some(path) = rest.strip_prefix("gguf:") {
                (Backend::LlamaCpp, path)
            } else {
                (Backend::LlamaCpp, rest)
            };
            // Context first, off the right: it is the only field introduced by
            // '@', so taking it before the size keeps the size parsing below
            // byte-for-byte what it was.
            let (rest, ctx) = match rest.rsplit_once('@') {
                Some((r, c)) if !c.is_empty() && c.chars().all(|ch| ch.is_ascii_digit()) => {
                    let parsed: u32 = c
                        .parse()
                        .map_err(|e| format!("--model {spec}: context {c}: {e}"))?;
                    if parsed == 0 {
                        return Err(format!(
                            "--model {spec}: @0 is not a context — omit @CTX to inherit --ctx"
                        ));
                    }
                    (r, Some(parsed))
                }
                Some((_, c)) => {
                    return Err(format!(
                        "--model {spec}: expected digits after '@', got {c:?}"
                    ))
                }
                None => (rest, None),
            };
            // Split the size off the RIGHT, so a Windows path's drive colon is
            // not mistaken for a size separator.
            let (path, gib) = match rest.rsplit_once(':') {
                Some((p, g)) if g.chars().all(|c| c.is_ascii_digit()) && !g.is_empty() => {
                    (p, g.parse().unwrap_or(4))
                }
                _ => (rest, 4),
            };
            if name.is_empty() || path.is_empty() {
                return Err(format!("--model {spec}: name and path are both required"));
            }
            out.push(ModelSpec {
                name: name.to_string(),
                path: path.to_string(),
                backend,
                gib,
                ctx,
            });
            i += 2;
        } else {
            i += 1;
        }
    }
    Ok(out)
}

/// The KV cache a model at `ctx_flag`/`parallel` will actually allocate,
/// read from its own GGUF header.
///
/// THIS IS THE NUMBER THE PLANNER WAS MISSING. `Declared.kv_bytes` has
/// existed since `budget::plan` was written, and every caller here passed
/// `0` — so two models could each fit the card on weight size alone while
/// the KV cache neither one budgeted for exhausted it anyway. On
/// 2026-08-28 that put a production A6000 through exactly this:
/// `muse-local:latest` at `--parallel 8` and no explicit `--ctx` spent
/// ~14 GiB on KV cache nothing had declared, `qwen2.5:14b`'s own CUDA
/// allocation failed against what was left, and `llama-server` exited with
/// no error text. `/residency` reported "30.0 / 42.0 GiB held" throughout —
/// correct about weights, silent about the number that mattered.
///
/// Failure to read the header is NOT fatal — the model still fails an
/// unreadable-header case exactly as it did before this existed (declared
/// at weight size alone) rather than refusing to start over a header this
/// reader could not parse. It is loud on stderr, though: a silent 0 here is
/// the same silent gap this function exists to close.
fn kv_bytes_for_model(gguf_path: &str, ctx_flag: u32, parallel: u32) -> u64 {
    let file = match std::fs::File::open(gguf_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!(
                "hearth: could not open {gguf_path} to size its KV cache ({e}) — \
                 budgeting weights only for this model, same as before KV accounting existed"
            );
            return 0;
        }
    };
    let mut reader = std::io::BufReader::new(file);
    match hearth_core::read_kv_shape(&mut reader) {
        Ok(shape) => {
            let total_ctx = hearth_core::total_ctx_tokens(ctx_flag, shape.context_length, parallel);
            shape.kv_bytes_for(total_ctx)
        }
        Err(e) => {
            eprintln!(
                "hearth: could not read KV shape from {gguf_path} ({e}) — \
                 budgeting weights only for this model, same as before KV accounting existed"
            );
            0
        }
    }
}

fn cmd_serve(args: &[String]) -> Result<(), String> {
    let model = flag(args, "--model").ok_or("--model is required")?;
    let gguf = flag(args, "--gguf");
    let safetensors = flag(args, "--safetensors");
    let (model_path, is_safetensors) = match (gguf, safetensors) {
        (Some(path), None) => (path, false),
        (None, Some(path)) => (path, true),
        (Some(_), Some(_)) => {
            return Err("choose exactly one of --gguf FILE or --safetensors DIR".into())
        }
        (None, None) => return Err("one of --gguf FILE or --safetensors DIR is required".into()),
    };
    let port: u16 = match flag(args, "--port") {
        Some(p) => p.parse().map_err(|e| format!("--port: {e}"))?,
        None => free_port()?,
    };
    let vram_gib: u64 = flag(args, "--vram-gib")
        .map(|v| v.parse().map_err(|e| format!("--vram-gib: {e}")))
        .transpose()?
        .unwrap_or(4);
    let total_gib: u64 = flag(args, "--total-gib")
        .map(|v| v.parse().map_err(|e| format!("--total-gib: {e}")))
        .transpose()?
        .unwrap_or(24);
    let ctx: u32 = flag(args, "--ctx")
        .map(|v| v.parse().map_err(|e| format!("--ctx: {e}")))
        .transpose()?
        .unwrap_or(0);
    let once = args.iter().any(|a| a == "--once");

    let mut spec = if is_safetensors {
        ServerSpec::new_safetensors(&model, &model_path, port)
    } else {
        ServerSpec::new(&model, &model_path, port)
    };
    spec.ctx = ctx;
    spec.binary = match flag(args, "--binary") {
        Some(b) => b.into(),
        None if is_safetensors => default_vllm_binary()?,
        None => default_binary()?,
    };
    spec.log_dir = hearth_home().join("logs");
    std::fs::create_dir_all(&spec.log_dir).map_err(|e| e.to_string())?;

    if !runtime_available(&spec.binary) {
        let hint = if is_safetensors {
            "run `hearth runtime vllm` to install HEARTH's managed vLLM runtime or pass --binary"
        } else {
            "install llama.cpp or pass --binary"
        };
        return Err(format!(
            "runtime not available: {} — {hint}",
            spec.binary.display()
        ));
    }

    let budget = Budget {
        total_bytes: total_gib * GIB,
        reserve_bytes: 2 * GIB,
    };
    let kv_bytes = if is_safetensors {
        // vLLM owns KV allocation for safetensors today. Weight admission is
        // still deterministic; config-derived KV accounting is a follow-up.
        0
    } else {
        kv_bytes_for_model(&model_path, ctx, spec.parallel)
    };
    let declared = vec![Declared {
        model: model.clone(),
        weights_bytes: vram_gib * GIB,
        kv_bytes,
    }];

    install_signal_handlers();
    let mut sup = Supervisor::new(open_spine()?, budget, declared);
    sup.start(spec)?;
    eprintln!(
        "hearth: {model} loading via {} on 127.0.0.1:{port} …",
        if is_safetensors {
            "vLLM/safetensors"
        } else {
            "llama.cpp/GGUF"
        }
    );

    let endpoint = sup.wait_ready(&model, Duration::from_secs(600))?;
    println!("resident {model} http://{endpoint} (OpenAI-compatible at /v1)");

    if once {
        sup.stop_all();
        return Ok(());
    }

    while !SHUTDOWN.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(500));
        let n = sup.tick();
        if n > 0 {
            eprintln!("hearth: {n} transition(s) recorded");
            eprint!("{}", sup.report());
        }
    }
    eprintln!("hearth: shutting down — recording unloaded, reaping children");
    sup.stop_all();
    Ok(())
}

fn cmd_status() -> Result<(), String> {
    let spine = open_spine()?;
    let events = spine.all_latest();
    if events.is_empty() {
        println!("no models in the spine yet — run `hearth serve`");
        return Ok(());
    }
    println!("{:<24} {:<14} {:>8}  facts", "MODEL", "STATE", "SEQ");
    for ev in events {
        println!(
            "{:<24} {:<14} {:>8}  {}",
            ev.model, ev.kind, ev.seq, ev.facts
        );
    }
    println!("\nhead {}", spine.head());
    Ok(())
}

fn cmd_why(args: &[String]) -> Result<(), String> {
    let model = args.first().ok_or("usage: hearth why MODEL")?;
    let spine = open_spine()?;
    let chain = spine.why(model);
    // Where the bytes came from is a different chain from why it is (not)
    // resident, and both are part of the answer. Pull events live in their own
    // collection; for one commit nothing read them, so this command claimed
    // "no history" about a model it had just downloaded and verified.
    let provenance = spine.provenance(model);

    if chain.is_empty() && provenance.is_empty() {
        return Err(format!("{model} has no history in the spine"));
    }

    if !provenance.is_empty() {
        println!("where {model} came from:\n");
        for (i, ev) in provenance.iter().enumerate() {
            let arrow = if i == 0 { "●" } else { "└─" };
            println!("{arrow} seq {:>5}  {:<14} {}", ev.seq, ev.kind, ev.facts);
        }
        if !chain.is_empty() {
            println!();
        }
    }

    if chain.is_empty() {
        println!("(never served — the bytes are here, nothing has loaded them)");
        return Ok(());
    }

    println!("why {model} — causal chain, newest first:\n");
    for (i, ev) in chain.iter().enumerate() {
        let arrow = if i == 0 { "●" } else { "└─" };
        println!("{arrow} seq {:>5}  {:<14} {}", ev.seq, ev.kind, ev.facts);
    }
    Ok(())
}

fn cmd_as_of(args: &[String]) -> Result<(), String> {
    let model = args.first().ok_or("usage: hearth as-of MODEL SEQ")?;
    let seq: u64 = args
        .get(1)
        .ok_or("usage: hearth as-of MODEL SEQ")?
        .parse()
        .map_err(|e| format!("SEQ: {e}"))?;
    let spine = open_spine()?;
    match spine.state_as_of(model, seq) {
        Some(ev) => {
            println!(
                "as of seq {seq}: {} was `{}` {}",
                ev.model, ev.kind, ev.facts
            );
            Ok(())
        }
        None => Err(format!("{model} had no state at seq {seq}")),
    }
}

fn cmd_verify() -> Result<(), String> {
    let spine = open_spine()?;
    let (checked, failures) = spine.verify();
    if failures.is_empty() {
        println!("verify ok — {checked} nodes checked, history intact");
        Ok(())
    } else {
        Err(format!(
            "verify FAILED — {} of {checked} nodes corrupt: {failures:?}",
            failures.len()
        ))
    }
}

#[cfg(test)]
mod passthrough_tests {
    use super::split_passthrough;

    fn v(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn everything_after_the_separator_is_the_runtimes() {
        let args = v(&[
            "--model",
            "m=/m.gguf:20",
            "--parallel",
            "2",
            "--",
            "--cache-reuse",
            "256",
            "-ub",
            "2048",
        ]);
        let (own, rt) = split_passthrough(&args);
        assert_eq!(own, &v(&["--model", "m=/m.gguf:20", "--parallel", "2"])[..]);
        assert_eq!(rt, &v(&["--cache-reuse", "256", "-ub", "2048"])[..]);
    }

    #[test]
    fn no_separator_means_no_passthrough_not_a_guess() {
        // `--cache-reuse` is unknown to hearth, but without `--` it is NOT
        // forwarded. Silently forwarding unknown flags would turn a typo in a
        // hearth flag into a llama-server crash at spawn.
        let args = v(&["--model", "m=/m.gguf:20", "--cache-reuse", "256"]);
        let (own, rt) = split_passthrough(&args);
        assert_eq!(own.len(), 4);
        assert!(rt.is_empty());
    }

    #[test]
    fn a_hearth_flag_after_the_separator_belongs_to_the_runtime() {
        // `--port` after `--` is llama-server's --port, not hearth's. hearth
        // owns the child port; an operator who passes it through gets exactly
        // what they asked for, appended last where argv() lets it win.
        let args = v(&["--port", "11434", "--", "--port", "9"]);
        let (own, rt) = split_passthrough(&args);
        assert_eq!(own, &v(&["--port", "11434"])[..]);
        assert_eq!(rt, &v(&["--port", "9"])[..]);
    }

    #[test]
    fn only_the_first_separator_splits() {
        let args = v(&["--", "-c", "--", "x"]);
        let (own, rt) = split_passthrough(&args);
        assert!(own.is_empty());
        assert_eq!(rt, &v(&["-c", "--", "x"])[..]);
    }
}

#[cfg(test)]
mod model_spec_tests {
    use super::collect_models;

    fn m(spec: &str) -> Vec<super::ModelSpec> {
        collect_models(&["--model".to_string(), spec.to_string()]).expect("should parse")
    }

    #[test]
    fn per_model_ctx_is_read_and_the_size_still_is_too() {
        let out = m("gpt-oss:20b=/blobs/sha256-27cd:12@32768");
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].name, "gpt-oss:20b",
            "a colon in the NAME is not a size"
        );
        assert_eq!(out[0].path, "/blobs/sha256-27cd");
        assert_eq!(out[0].gib, 12);
        assert_eq!(out[0].ctx, Some(32768));
    }

    #[test]
    fn omitting_ctx_inherits_the_global_one() {
        // THE BACK-COMPAT GUARD. Every fleet.conf written before @CTX existed
        // must mean exactly what it meant, or an upgrade silently reshapes a
        // production fleet's KV budget.
        let out = m("muse-local:latest=/blobs/sha256-71b5:12");
        assert_eq!(out[0].name, "muse-local:latest");
        assert_eq!(out[0].gib, 12);
        assert_eq!(
            out[0].ctx, None,
            "None is what makes the global --ctx apply"
        );
    }

    #[test]
    fn ctx_without_a_size_works_and_the_size_defaults() {
        let out = m("small=/blobs/sha256-aaa@8192");
        assert_eq!(out[0].path, "/blobs/sha256-aaa");
        assert_eq!(out[0].gib, 4, "documented default when :GIB is absent");
        assert_eq!(out[0].ctx, Some(8192));
    }

    #[test]
    fn a_windows_drive_colon_is_still_not_a_size() {
        let out = m(r"win=C:\models\m.gguf:20@16384");
        assert_eq!(out[0].path, r"C:\models\m.gguf");
        assert_eq!(out[0].gib, 20);
        assert_eq!(out[0].ctx, Some(16384));
    }

    #[test]
    fn a_mixed_fleet_keeps_each_models_own_ctx() {
        let args: Vec<String> = [
            "--model",
            "kimi=/blobs/a:35@65536",
            "--model",
            "granite=/blobs/b:9",
            "--model",
            "gpt-oss:20b=/blobs/c:12@32768",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let out = collect_models(&args).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].ctx, Some(65536));
        assert_eq!(out[1].ctx, None, "inherits the global default");
        assert_eq!(out[2].ctx, Some(32768));
    }

    #[test]
    fn a_nonsense_ctx_is_refused_rather_than_silently_ignored() {
        // Silently dropping "@abc" would budget KV for the global ctx while
        // the operator believed they had pinned one.
        assert!(collect_models(&["--model".into(), "x=/p:12@abc".into()]).is_err());
        // @0 is ambiguous: it is the sentinel for "runtime default" internally,
        // so accepting it would mean two spellings of the same thing.
        assert!(collect_models(&["--model".into(), "x=/p:12@0".into()]).is_err());
    }

    #[test]
    fn name_and_path_are_both_still_required() {
        assert!(collect_models(&["--model".into(), "=/p:12".into()]).is_err());
        assert!(collect_models(&["--model".into(), "n=".into()]).is_err());
        assert!(collect_models(&["--model".into(), "no-equals-sign".into()]).is_err());
    }
}
