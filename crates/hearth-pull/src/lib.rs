//! `hearth pull` — get the bytes, prove they are the right bytes, and write
//! down where they came from.
//!
//! This is the half of the aggregator that was missing. [`hearth_resolve`] is
//! deliberately network-free: it turns a reference into blob URLs, a chosen
//! quantization and a byte count, with `serde` and `serde_json` as its entire
//! dependency list. That is what lets the VRAM planner refuse a model *before*
//! forty gigabytes move. This crate is what moves them.
//!
//! Three properties, and the third is the one nobody else offers:
//!
//! 1. **Digest-verified.** Every blob is hashed and compared against the digest
//!    the registry published. A download that does not match is deleted, not
//!    kept — a corrupt GGUF fails much later as an unexplainable tokenizer
//!    error, three layers away from the truth.
//! 2. **Resumable.** A 40 GiB transfer that dies at 39 GiB resumes. Restarting
//!    is not a recovery strategy, it is a way to never finish on a bad link.
//! 3. **Recorded.** `PullStarted` and `PullCompleted` land in the spine with
//!    the resolved source, so months later "where did this file come from" is a
//!    query — `hearth why <model>` — instead of an archaeology project. Those
//!    two transitions already existed in [`hearth_store`]; the history was
//!    ready before the downloader was.

pub mod curl;
pub mod registry;
pub mod runtime;

use std::path::{Path, PathBuf};

use hearth_core::sha256;
use hearth_resolve::{plan::ModelFormat, Reference};
use hearth_store::{EventRef, Spine, Transition};

pub use registry::{Blob, Fetched};

/// Where blobs live, and how loud to be about it.
#[derive(Debug, Clone)]
pub struct PullConfig {
    /// Content-addressed blob directory, e.g. `~/.hearth/blobs`.
    pub blobs_dir: PathBuf,
    /// Draw curl's progress meter. A silent twenty-minute pause is
    /// indistinguishable from a hang, and a downloader that looks hung gets
    /// killed halfway.
    pub progress: bool,
    /// Re-verify a blob that is already on disk instead of trusting its name.
    /// Off by default: re-hashing 40 GiB on every start is a real cost, and the
    /// filename IS the digest. On when you have reason to distrust the disk.
    pub verify_existing: bool,
}

impl Default for PullConfig {
    fn default() -> Self {
        PullConfig {
            blobs_dir: PathBuf::from("./blobs"),
            progress: true,
            verify_existing: false,
        }
    }
}

/// What a pull did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pulled {
    pub model: String,
    /// Resolved origin, as recorded in the spine.
    pub source: String,
    /// Selected model format.
    pub format: ModelFormat,
    /// Runtime-facing path: a GGUF file or a materialized safetensors model directory.
    pub model_path: PathBuf,
    /// Backward-compatible primary weights path. For safetensors this is the
    /// first weight shard; use model_path for serving.
    pub weights_path: PathBuf,
    pub bytes: u64,
    /// True when everything was already present and verified — so a second
    /// `pull` of the same model is honest about having done nothing.
    pub already_had_it: bool,
}

/// Content-addressed path for a digest. The filename IS the digest, which is
/// what makes "do we already have this" a `stat` rather than a re-hash.
pub fn blob_path(blobs_dir: &Path, digest: &str) -> PathBuf {
    blobs_dir.join(format!("sha256-{}", sha256::normalize(digest)))
}

/// Bytes as something a person can compare to a download page.
pub fn human_bytes(n: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    let f = n as f64;
    if f >= GIB {
        format!("{:.2} GiB", f / GIB)
    } else if f >= MIB {
        format!("{:.1} MiB", f / MIB)
    } else {
        format!("{n} B")
    }
}

/// Hash a file, showing that it is still working.
///
/// The step this replaces was a bare `sha256::hex_digest_file`, which on a
/// 35 GiB blob is minutes of total silence immediately after curl prints
/// `100.0%`. That is the exact failure `PullConfig::progress` was written to
/// prevent, and the verify step was the one place the pull path did not honour
/// it.
///
/// Throttled to roughly twice a second on one rewritten line, so it reads like
/// the transfer meter above it instead of scrolling a wall of text.
fn digest_file_loud(path: &Path, label: &str, progress: bool) -> Result<String, String> {
    use std::io::Write;
    let started = std::time::Instant::now();
    let mut last = std::time::Instant::now();
    let digest = sha256::hex_digest_file_with_progress(path, |done, total| {
        if !progress {
            return;
        }
        if last.elapsed() < std::time::Duration::from_millis(500) && done < total {
            return;
        }
        last = std::time::Instant::now();
        let secs = started.elapsed().as_secs_f64().max(0.001);
        let rate = (done as f64 / secs) as u64;
        let pct = if total > 0 {
            format!("{:5.1}%", done as f64 / total as f64 * 100.0)
        } else {
            "  ?  ".to_string()
        };
        // \r, not \n: one line, rewritten in place.
        let _ = write!(
            std::io::stderr(),
            "\rhearth: verifying {label} {pct} ({} / {}) {}/s   ",
            human_bytes(done),
            human_bytes(total),
            human_bytes(rate)
        );
        let _ = std::io::stderr().flush();
    })
    .map_err(|e| format!("could not hash {}: {e}", path.display()))?;
    if progress {
        let secs = started.elapsed().as_secs_f64();
        let len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let rate = if secs > 0.0 {
            (len as f64 / secs) as u64
        } else {
            0
        };
        let _ = writeln!(
            std::io::stderr(),
            "\rhearth: verified {label} — {} in {:.1}s ({}/s)      ",
            human_bytes(len),
            secs,
            human_bytes(rate)
        );
    }
    Ok(digest)
}

/// Fetch one blob to its content-addressed home, verifying the digest.
///
/// Downloads to a `.partial` sibling and renames only after the digest
/// matches. Without that, an interrupted transfer leaves a file whose *name*
/// claims a digest its *contents* do not have — and every later run trusts the
/// name. The rename is the commit.
pub fn fetch_blob(blob: &Blob, cfg: &PullConfig) -> Result<(PathBuf, u64, bool), String> {
    match &blob.digest {
        Some(digest) => fetch_blob_verified(blob, digest, cfg),
        None => fetch_blob_self_verified(blob, cfg),
    }
}

/// The original path: a digest is already known, so the final home is known
/// before a single byte moves, and every byte is checked against it.
fn fetch_blob_verified(
    blob: &Blob,
    digest: &str,
    cfg: &PullConfig,
) -> Result<(PathBuf, u64, bool), String> {
    let final_path = blob_path(&cfg.blobs_dir, digest);

    if final_path.exists() {
        let len = std::fs::metadata(&final_path).map(|m| m.len()).unwrap_or(0);
        if cfg.verify_existing {
            let actual = digest_file_loud(&final_path, &blob.name, cfg.progress)?;
            if !sha256::matches(digest, &actual) {
                // On disk under a digest it does not have. Say so loudly and
                // re-fetch rather than quietly serving the wrong weights.
                eprintln!(
                    "hearth: {} does not match its own name — refetching",
                    final_path.display()
                );
                let _ = std::fs::remove_file(&final_path);
            } else {
                return Ok((final_path, len, true));
            }
        } else if blob.size_bytes == 0 || len == blob.size_bytes {
            return Ok((final_path, len, true));
        } else {
            // Right name, wrong size: a previous run was interrupted after the
            // rename, or something truncated it. Not trustworthy.
            eprintln!(
                "hearth: {} is {len} bytes, expected {} — refetching",
                final_path.display(),
                blob.size_bytes
            );
            let _ = std::fs::remove_file(&final_path);
        }
    }

    let partial = final_path.with_extension("partial");
    download_to(&partial, blob, cfg)?;

    let actual = digest_file_loud(&partial, &blob.name, cfg.progress)?;
    if !sha256::matches(digest, &actual) {
        // Delete it. Keeping a blob that failed verification is how a corrupt
        // download becomes a permanent, cached, corrupt download.
        let _ = std::fs::remove_file(&partial);
        return Err(format!(
            "digest mismatch for {}\n  expected sha256:{}\n  got      sha256:{}\n  \
             the file has been deleted; a resumed transfer over a proxy that \
             rewrote a range will do this, so a plain retry is worth one attempt",
            blob.name,
            sha256::normalize(digest),
            actual
        ));
    }

    let len = std::fs::metadata(&partial).map(|m| m.len()).unwrap_or(0);
    std::fs::rename(&partial, &final_path)
        .map_err(|e| format!("could not commit {}: {e}", final_path.display()))?;
    Ok((final_path, len, false))
}

/// No digest was published to check against — a bare URL, or a HuggingFace
/// file the tree listing did not mark as LFS.
///
/// The final home cannot be known before downloading, because the digest
/// that names it is the thing being discovered. So this downloads to a
/// TEMPORARY name (never a content-addressed one — a partial file must never
/// sit under the name its finished content will claim), hashes what actually
/// arrived, and only then computes the path a verified blob would have had
/// from the start. If that path already exists, this exact content was
/// already fetched under some other name, and the new download is discarded
/// as a duplicate rather than kept as a second copy of the same bytes.
///
/// This is self-consistency, not verification: the file matches its own
/// name by construction, which is trivially true of every file and proves
/// nothing about whether the content is what the operator meant to fetch.
/// Callers that need real verification pin a digest — the URL scheme's
/// `#sha256:HEX` fragment exists for exactly that.
fn fetch_blob_self_verified(blob: &Blob, cfg: &PullConfig) -> Result<(PathBuf, u64, bool), String> {
    std::fs::create_dir_all(&cfg.blobs_dir).map_err(|e| e.to_string())?;
    let safe_name = blob
        .name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>();
    let partial = cfg
        .blobs_dir
        .join(format!("{safe_name}.unverified.partial"));

    download_to(&partial, blob, cfg)?;

    let actual = digest_file_loud(&partial, &blob.name, cfg.progress)?;
    let final_path = blob_path(&cfg.blobs_dir, &actual);

    if final_path.exists() {
        // Already have this exact content under its rightful name — this
        // download was redundant, not wrong.
        let _ = std::fs::remove_file(&partial);
        let len = std::fs::metadata(&final_path).map(|m| m.len()).unwrap_or(0);
        return Ok((final_path, len, true));
    }

    let len = std::fs::metadata(&partial).map(|m| m.len()).unwrap_or(0);
    std::fs::rename(&partial, &final_path)
        .map_err(|e| format!("could not commit {}: {e}", final_path.display()))?;
    eprintln!(
        "hearth: {} has no published digest — stored as sha256:{actual}, \
         self-consistent only (not checked against a source, because the \
         source published none)",
        blob.name
    );
    Ok((final_path, len, false))
}

/// Download one blob to `dest`, resuming a partial transfer if one is
/// already there. Shared by both verification paths — they differ in what
/// happens to the bytes afterward, not in how the bytes arrive.
fn download_to(dest: &Path, blob: &Blob, cfg: &PullConfig) -> Result<(), String> {
    let req = {
        let mut r = curl::Request::get(&blob.url).to_file(dest);
        for (k, v) in &blob.headers {
            r = r.header(k, v);
        }
        // Resume only if there is something to resume; `-C -` on a missing file
        // is fine, but on a zero-byte one some servers answer 416.
        if dest.exists() && std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0) > 0 {
            r = r.resuming();
        }
        r
    };
    curl::fetch_file(&req, cfg.progress).map_err(|e| e.0)?;
    Ok(())
}

fn safe_model_dir_name(model: &str) -> String {
    model
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '.' { c } else { '_' })
        .collect()
}

fn materialize_safetensors_package(
    model: &str,
    files: &[(String, PathBuf)],
    cfg: &PullConfig,
) -> Result<PathBuf, String> {
    let dir = cfg
        .blobs_dir
        .join("models")
        .join(safe_model_dir_name(model));
    std::fs::create_dir_all(&dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;

    for (relative, source) in files {
        let rel = Path::new(relative);
        if rel.is_absolute()
            || rel.components().any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return Err(format!("refusing unsafe model path {relative:?}"));
        }
        let dest = dir.join(rel);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("creating {}: {e}", parent.display()))?;
        }
        if dest.exists() {
            let _ = std::fs::remove_file(&dest);
        }
        if std::fs::hard_link(source, &dest).is_err() {
            std::fs::copy(source, &dest)
                .map_err(|e| format!("materializing {}: {e}", dest.display()))?;
        }
    }

    Ok(dir)
}

/// Pull a model reference, recording the whole thing in the spine.
///
/// The recording is not decoration. `PullStarted` goes in *before* any bytes
/// move, so a pull that dies halfway leaves evidence that it was attempted and
/// from where — the alternative is a half-empty blob directory and no idea
/// which model it belongs to.
pub fn pull(reference: &str, cfg: &PullConfig, spine: &Spine) -> Result<Pulled, String> {
    let reference = Reference::parse(reference).map_err(|e| e.0)?;
    let model = reference.display_name();
    let source = registry::source_string(&reference);

    // A local file is already here. Recording a pull for it would be a lie.
    if let Reference::Local { path } = &reference {
        let p = PathBuf::from(path);
        if !p.exists() {
            return Err(format!("no file at {path}"));
        }
        let bytes = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
        return Ok(Pulled {
            model,
            source,
            format: ModelFormat::Gguf,
            model_path: p.clone(),
            weights_path: p,
            bytes,
            already_had_it: true,
        });
    }

    let started = spine
        .record(
            &model,
            &Transition::PullStarted {
                source: source.clone(),
            },
            &[],
        )
        .map_err(|e| format!("could not record the pull: {e}"))?;

    // Resolving is a network round trip against a registry that can be slow or
    // rate-limited, and it produced no output at all — so the first thing the
    // operator saw was a pause before any transfer had been announced.
    if cfg.progress {
        eprintln!("hearth: resolving {source} …");
    }

    let fetched = registry::resolve_blobs(&reference).map_err(|e| {
        // The failure is worth recording too — "we tried and the registry said
        // no" is a different fact from "nobody ever tried".
        let _ = spine.record(
            &model,
            &Transition::PullStarted {
                source: format!("{source} — FAILED: {e}"),
            },
            std::slice::from_ref(&started),
        );
        e
    })?;

    // Announce the PLAN before moving bytes. "1 file, 35.10 GiB" up front is
    // the difference between a transfer the operator can wait out and one they
    // assume is stuck; and on a multi-part model it is the only way to know
    // that a meter reaching 100.0% is part one of three, not the end.
    let n = fetched.blobs.len();
    if cfg.progress {
        let total: u64 = fetched.blobs.iter().map(|b| b.size_bytes).sum();
        eprintln!(
            "hearth: {n} file(s), {} to fetch — then a sha256 verify of each",
            human_bytes(total)
        );
    }

    let mut primary_weight: Option<(PathBuf, u64)> = None;
    let mut package_files: Vec<(String, PathBuf)> = Vec::new();
    let mut cached_all = true;
    let mut actual_weight_bytes = 0u64;
    for (i, blob) in fetched.blobs.iter().enumerate() {
        if cfg.progress {
            eprintln!(
                "hearth: [{}/{n}] {} ({})",
                i + 1,
                blob.name,
                if blob.size_bytes > 0 {
                    human_bytes(blob.size_bytes)
                } else {
                    "size unknown".to_string()
                }
            );
        }
        let (path, len, cached) = fetch_blob(blob, cfg)?;
        if cfg.progress && cached {
            eprintln!("hearth: [{}/{n}] already on disk, skipped", i + 1);
        }
        if !cached {
            cached_all = false;
        }
        package_files.push((blob.name.clone(), path.clone()));
        if blob.is_weights {
            actual_weight_bytes = actual_weight_bytes.saturating_add(len);
            if primary_weight.is_none() {
                primary_weight = Some((path, len));
            }
        }
    }

    let (weights_path, first_weight_len) = primary_weight.ok_or_else(|| {
        format!(
            "{model} resolved with no weights layer — the registry returned \
             {} blob(s), none of them the model",
            fetched.blobs.len()
        )
    })?;

    let model_path = match fetched.format {
        ModelFormat::Gguf => weights_path.clone(),
        ModelFormat::SafeTensors => {
            materialize_safetensors_package(&model, &package_files, cfg)?
        }
    };
    let bytes = if fetched.weights_bytes > 0 {
        fetched.weights_bytes
    } else if actual_weight_bytes > 0 {
        actual_weight_bytes
    } else {
        first_weight_len
    };

    spine
        .record(
            &model,
            &Transition::PullCompleted {
                path: model_path.display().to_string(),
                size_bytes: bytes,
            },
            &[EventRef {
                hash: started.hash,
                seq: started.seq,
            }],
        )
        .map_err(|e| format!("could not record completion: {e}"))?;

    Ok(Pulled {
        model,
        source,
        format: fetched.format,
        model_path,
        weights_path,
        bytes,
        already_had_it: cached_all,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blob_path_is_its_digest_so_presence_is_a_stat() {
        let p = blob_path(Path::new("/b"), "sha256:ABC123");
        assert_eq!(p, Path::new("/b/sha256-abc123"));
        // Both spellings of the same digest land on the same file, or a cache
        // would miss every time the registry changed its formatting.
        assert_eq!(p, blob_path(Path::new("/b"), "abc123"));
    }

    #[test]
    fn a_local_reference_is_not_recorded_as_a_pull() {
        let dir = std::env::temp_dir().join("hearth-pull-local");
        std::fs::create_dir_all(&dir).unwrap();
        let gguf = dir.join("muse.gguf");
        std::fs::write(&gguf, b"weights").unwrap();

        let spine = Spine::in_memory();
        let out = pull(
            &format!("file:{}", gguf.display()),
            &PullConfig::default(),
            &spine,
        )
        .unwrap();

        assert!(out.already_had_it);
        assert_eq!(out.bytes, 7);
        assert!(
            spine.latest(&out.model).is_none(),
            "nothing was fetched, so claiming a pull would be a lie"
        );
        std::fs::remove_file(&gguf).ok();
    }

    #[test]
    fn a_missing_local_file_is_refused_before_anything_is_recorded() {
        let spine = Spine::in_memory();
        let err = pull(
            "file:/definitely/not/here.gguf",
            &PullConfig::default(),
            &spine,
        )
        .unwrap_err();
        assert!(err.contains("no file at"), "{err}");
        assert_eq!(spine.seq(), 0, "nothing should have been written");
    }

    #[test]
    fn a_digest_mismatch_deletes_the_file_rather_than_caching_corruption() {
        // The blob is served by a local file:// URL whose contents do not match
        // the digest we claim. curl handles file:// so this needs no network.
        let dir = std::env::temp_dir().join("hearth-pull-mismatch");
        std::fs::create_dir_all(&dir).unwrap();
        let served = dir.join("served.bin");
        std::fs::write(&served, b"not what we asked for").unwrap();

        let blobs = dir.join("blobs");
        let cfg = PullConfig {
            blobs_dir: blobs.clone(),
            progress: false,
            verify_existing: false,
        };
        let blob = Blob {
            name: "weights".into(),
            url: format!("file://{}", served.display()),
            digest: Some(
                "sha256:0000000000000000000000000000000000000000000000000000000000000000".into(),
            ),
            size_bytes: 0,
            headers: vec![],
            is_weights: true,
        };

        let err = fetch_blob(&blob, &cfg).unwrap_err();
        assert!(err.contains("digest mismatch"), "{err}");
        let digest = blob.digest.as_deref().unwrap();
        assert!(
            !blob_path(&blobs, digest).exists(),
            "a blob that failed verification must not be left on disk"
        );
        assert!(
            !blob_path(&blobs, digest).with_extension("partial").exists(),
            "and neither must the partial"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_matching_digest_commits_the_file_under_its_digest() {
        let dir = std::env::temp_dir().join("hearth-pull-match");
        std::fs::create_dir_all(&dir).unwrap();
        let body = b"the actual weights, honestly";
        let served = dir.join("served.bin");
        std::fs::write(&served, body).unwrap();

        let blobs = dir.join("blobs");
        let cfg = PullConfig {
            blobs_dir: blobs.clone(),
            progress: false,
            verify_existing: false,
        };
        let blob = Blob {
            name: "weights".into(),
            url: format!("file://{}", served.display()),
            digest: Some(sha256::hex_digest(body)),
            size_bytes: body.len() as u64,
            headers: vec![],
            is_weights: true,
        };

        let (path, len, cached) = fetch_blob(&blob, &cfg).unwrap();
        assert!(!cached);
        assert_eq!(len, body.len() as u64);
        assert_eq!(std::fs::read(&path).unwrap(), body);

        // Second call is a stat, not a download.
        let (_, _, cached2) = fetch_blob(&blob, &cfg).unwrap();
        assert!(cached2, "the filename IS the digest — presence is enough");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_cached_blob_of_the_wrong_size_is_refetched() {
        let dir = std::env::temp_dir().join("hearth-pull-truncated");
        std::fs::create_dir_all(&dir).unwrap();
        let body = b"complete contents here";
        let served = dir.join("served.bin");
        std::fs::write(&served, body).unwrap();

        let blobs = dir.join("blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        let digest = sha256::hex_digest(body);
        // A truncated file already sitting under the right name — what an
        // interrupted run leaves behind if it renames before verifying.
        std::fs::write(blob_path(&blobs, &digest), b"trunc").unwrap();

        let cfg = PullConfig {
            blobs_dir: blobs.clone(),
            progress: false,
            verify_existing: false,
        };
        let blob = Blob {
            name: "weights".into(),
            url: format!("file://{}", served.display()),
            digest: Some(digest),
            size_bytes: body.len() as u64,
            headers: vec![],
            is_weights: true,
        };
        let (path, len, cached) = fetch_blob(&blob, &cfg).unwrap();
        assert!(!cached, "the size did not match, so it was not trusted");
        assert_eq!(len, body.len() as u64);
        assert_eq!(std::fs::read(&path).unwrap(), body);
        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- fetch_blob_self_verified: no published digest to check against ----

    #[test]
    fn no_digest_still_lands_the_file_under_its_own_computed_sha256() {
        // A bare URL pull: nothing published a digest, so `blob.digest` is
        // `None`. The file must still end up content-addressed — hearth
        // computes the digest itself rather than trusting a caller-supplied
        // filename for the final path.
        let dir = std::env::temp_dir().join("hearth-pull-no-digest");
        std::fs::create_dir_all(&dir).unwrap();
        let body = b"weights nobody published a digest for";
        let served = dir.join("served.bin");
        std::fs::write(&served, body).unwrap();

        let blobs = dir.join("blobs");
        let cfg = PullConfig {
            blobs_dir: blobs.clone(),
            progress: false,
            verify_existing: false,
        };
        let blob = Blob {
            name: "model.gguf".into(),
            url: format!("file://{}", served.display()),
            digest: None,
            size_bytes: 0,
            headers: vec![],
            is_weights: true,
        };

        let (path, len, cached) = fetch_blob(&blob, &cfg).unwrap();
        assert!(!cached);
        assert_eq!(len, body.len() as u64);
        assert_eq!(std::fs::read(&path).unwrap(), body);

        let expected_name = format!("sha256-{}", sha256::hex_digest(body));
        assert_eq!(
            path.file_name().unwrap().to_str().unwrap(),
            expected_name,
            "the final path must be the digest hearth computed, not a guess"
        );
        assert!(
            !blobs.join("model.gguf.unverified.partial").exists(),
            "the working file must not survive under its temporary name"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_second_no_digest_pull_of_identical_content_is_recognised_as_cached() {
        // Two different URLs (or the same URL fetched twice under different
        // callers) that happen to serve byte-identical content must land on
        // the SAME blob, not two copies — that is the entire point of
        // content addressing, and the no-digest path has to preserve it even
        // though it does not know the address until after downloading.
        let dir = std::env::temp_dir().join("hearth-pull-no-digest-dup");
        std::fs::create_dir_all(&dir).unwrap();
        let body = b"identical weights served from two places";
        let served = dir.join("served.bin");
        std::fs::write(&served, body).unwrap();

        let blobs = dir.join("blobs");
        let cfg = PullConfig {
            blobs_dir: blobs.clone(),
            progress: false,
            verify_existing: false,
        };
        let blob = Blob {
            name: "model.gguf".into(),
            url: format!("file://{}", served.display()),
            digest: None,
            size_bytes: 0,
            headers: vec![],
            is_weights: true,
        };

        let (first_path, ..) = fetch_blob(&blob, &cfg).unwrap();
        let (second_path, _, cached) = fetch_blob(&blob, &cfg).unwrap();
        assert!(
            cached,
            "identical content must be recognised, not re-stored"
        );
        assert_eq!(first_path, second_path);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_unsafe_blob_name_does_not_escape_the_blobs_directory() {
        // blob.name comes from a URL path or a registry listing — untrusted
        // input. A name containing "../" must not let the temporary working
        // file land outside blobs_dir.
        let dir = std::env::temp_dir().join("hearth-pull-unsafe-name");
        std::fs::create_dir_all(&dir).unwrap();
        let body = b"contents";
        let served = dir.join("served.bin");
        std::fs::write(&served, body).unwrap();

        let blobs = dir.join("blobs");
        let cfg = PullConfig {
            blobs_dir: blobs.clone(),
            progress: false,
            verify_existing: false,
        };
        let blob = Blob {
            name: "../../etc/model.gguf".into(),
            url: format!("file://{}", served.display()),
            digest: None,
            size_bytes: 0,
            headers: vec![],
            is_weights: true,
        };

        let (path, ..) = fetch_blob(&blob, &cfg).unwrap();
        assert!(
            path.starts_with(&blobs),
            "the final path must stay under blobs_dir, got {}",
            path.display()
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}


#[cfg(test)]
mod safetensors_package_tests {
    use super::*;

    #[test]
    fn safetensors_package_materializes_repository_paths() {
        let root = std::env::temp_dir().join(format!("hearth-st-package-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let blobs = root.join("blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        let a = blobs.join("a");
        let b = blobs.join("b");
        std::fs::write(&a, b"weights").unwrap();
        std::fs::write(&b, b"config").unwrap();

        let cfg = PullConfig {
            blobs_dir: blobs,
            progress: false,
            verify_existing: false,
        };
        let dir = materialize_safetensors_package(
            "Interchained/imagine-v8",
            &[
                ("model.safetensors".into(), a),
                ("config.json".into(), b),
            ],
            &cfg,
        )
        .unwrap();

        assert_eq!(std::fs::read(dir.join("model.safetensors")).unwrap(), b"weights");
        assert_eq!(std::fs::read(dir.join("config.json")).unwrap(), b"config");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn safetensors_package_rejects_parent_traversal() {
        let root = std::env::temp_dir().join(format!("hearth-st-traversal-{}", std::process::id()));
        let blobs = root.join("blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        let src = blobs.join("a");
        std::fs::write(&src, b"x").unwrap();
        let cfg = PullConfig { blobs_dir: blobs, progress: false, verify_existing: false };
        let err = materialize_safetensors_package(
            "m",
            &[("../escape".into(), src)],
            &cfg,
        )
        .unwrap_err();
        assert!(err.contains("unsafe model path"));
        let _ = std::fs::remove_dir_all(&root);
    }
}
