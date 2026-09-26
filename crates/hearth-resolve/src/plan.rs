//! From a reference to a list of bytes to fetch.
//!
//! Both catalogs eventually hand you the same thing — GGUF files over HTTPS —
//! but they describe them completely differently, and neither description is
//! usable as-is.
//!
//! **Ollama** serves an OCI manifest. The GGUF is one layer among several
//! (there is also a template, a params blob, a licence), identified by media
//! type. Everything is content-addressed, so the digest comes free.
//!
//! **HuggingFace** hands you a directory listing and no opinion. A popular
//! repo has eight quantizations of the same model and it is on you to pick;
//! big models are split into `00001-of-00003` parts that are useless
//! individually. Sizes are often absent unless you ask for them.
//!
//! This module is the part that turns either of those into "fetch these bytes,
//! expect this digest" — and it is pure, so every rule in it is testable
//! without a network.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The on-disk model format selected from a catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFormat {
    Gguf,
    SafeTensors,
}

/// One file to fetch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Blob {
    pub url: String,
    /// Where it lands, relative to the model's directory.
    pub filename: String,
    /// `sha256:...` when the catalog tells us. Ollama always does because it
    /// is content-addressed; HuggingFace usually does not.
    pub digest: Option<String>,
    pub size_bytes: Option<u64>,
}

/// Everything needed to materialize one model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FetchPlan {
    /// Stable identity — the reference's key.
    pub key: String,
    pub display_name: String,
    /// Runtime-facing model format. GGUF is a single/model-part artifact;
    /// safetensors is a Transformers model directory with weights + metadata.
    pub format: ModelFormat,
    /// In order. Multi-part weights must be fetched and kept in sequence.
    pub blobs: Vec<Blob>,
    /// Which quantization we ended up with, and whether the caller chose it.
    /// Surfaced because "you asked for a model and got Q4_K_M" is information
    /// somebody will eventually need to explain a quality difference.
    pub quant: Option<String>,
    pub quant_was_chosen_for_you: bool,
}

impl FetchPlan {
    pub fn total_bytes(&self) -> Option<u64> {
        // All or nothing: a partial sum silently understates the download and
        // would let the VRAM planner admit something that does not fit.
        self.blobs.iter().map(|b| b.size_bytes).sum()
    }

    pub fn is_multipart(&self) -> bool {
        self.blobs.len() > 1
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolveError(pub String);

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ResolveError {}

fn err(m: impl Into<String>) -> ResolveError {
    ResolveError(m.into())
}

/* -------------------------------------------------------------- ollama -- */

/// The layer media type that holds actual weights. The others in an Ollama
/// manifest are the prompt template, the parameters and the licence, and
/// downloading them as though they were the model is a 30-byte "model".
pub const OLLAMA_MODEL_MEDIA_TYPE: &str = "application/vnd.ollama.image.model";

/// Turn an Ollama OCI manifest into a fetch plan.
///
/// `base` is the registry root, e.g. `https://registry.ollama.ai`.
pub fn plan_from_ollama_manifest(
    manifest: &Value,
    base: &str,
    namespace: &str,
    name: &str,
    key: String,
    display_name: String,
) -> Result<FetchPlan, ResolveError> {
    let layers = manifest
        .get("layers")
        .and_then(|l| l.as_array())
        .ok_or_else(|| err("manifest has no layers array"))?;

    let model_layers: Vec<&Value> = layers
        .iter()
        .filter(|l| l.get("mediaType").and_then(|m| m.as_str()) == Some(OLLAMA_MODEL_MEDIA_TYPE))
        .collect();

    if model_layers.is_empty() {
        let seen: Vec<&str> = layers
            .iter()
            .filter_map(|l| l.get("mediaType").and_then(|m| m.as_str()))
            .collect();
        return Err(err(format!(
            "no {OLLAMA_MODEL_MEDIA_TYPE} layer in this manifest — saw: {}",
            if seen.is_empty() {
                "nothing".to_string()
            } else {
                seen.join(", ")
            }
        )));
    }

    let mut blobs = Vec::with_capacity(model_layers.len());
    for (i, layer) in model_layers.iter().enumerate() {
        let digest = layer
            .get("digest")
            .and_then(|d| d.as_str())
            .ok_or_else(|| err("a model layer has no digest"))?;
        let size = layer.get("size").and_then(|s| s.as_u64());
        blobs.push(Blob {
            url: format!(
                "{}/v2/{namespace}/{name}/blobs/{digest}",
                base.trim_end_matches('/')
            ),
            filename: if model_layers.len() == 1 {
                "model.gguf".to_string()
            } else {
                format!("model-{:05}.gguf", i + 1)
            },
            digest: Some(digest.to_string()),
            size_bytes: size,
        });
    }

    Ok(FetchPlan {
        key,
        display_name,
        format: ModelFormat::Gguf,
        blobs,
        // Ollama tags encode the quantization in the tag itself; the manifest
        // does not name it separately, so claiming one would be inventing it.
        quant: None,
        quant_was_chosen_for_you: false,
    })
}

/* --------------------------------------------------------- huggingface -- */

/// Preference order when the caller did not pin a quantization.
///
/// Q4_K_M first because it is the one the community converged on: close to
/// half the size of the f16 weights, and the quality loss is not the thing you
/// notice. Descending from there through the K-quants, then the big ones.
/// F16 is last on purpose — it is correct and almost always the wrong default,
/// since it will not fit next to anything else on one card.
///
/// The block-float formats sit at the END deliberately. When a repo ships one
/// it is usually the REFERENCE release (gpt-oss is published in MXFP4, not
/// converted to it), so it must be selectable — but a repo carrying both an
/// MXFP4 and a K-quant should still land on the K-quant, because MXFP4 has no
/// native path on pre-Blackwell hardware and gets dequantized to compute.
/// Putting them last means behaviour changes only for repos that previously
/// had NO recognised quantization at all.
pub const QUANT_PREFERENCE: &[&str] = &[
    "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "Q4_0", "Q3_K_M", "F16", "MXFP4",
    "NVFP4", "MXFP8",
];

/// Files that live in a GGUF repo but are not the model you serve.
///
/// A draft head for speculative decoding (`eagle3-*`), a vision projector
/// (`mmproj-*`) and a LoRA adapter are all valid GGUF, and all far smaller
/// than the weights — so a chooser that ranks by quantization name will
/// cheerfully return one.
///
/// Observed on `ggml-org/gpt-oss-20b-GGUF`, which holds exactly three GGUFs:
///
/// ```text
/// eagle3-gpt-oss-20b-Q8_0.gguf     0.92 GB   <- draft head
/// eagle3-gpt-oss-20b-BF16.gguf     1.72 GB   <- draft head
/// gpt-oss-20b-MXFP4.gguf          12.11 GB   <- THE MODEL
/// ```
///
/// An unpinned pull resolved to the 0.92 GB draft head, because `Q8_0` is in
/// `QUANT_PREFERENCE` and `MXFP4` was not even recognised. That download
/// succeeds, verifies, loads, and serves a draft head as though it were
/// gpt-oss-20b. Wrong weights that work are worse than a failure.
pub fn is_auxiliary_gguf(path: &str) -> bool {
    let stem = path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase();
    const MARKERS: &[&str] = &["eagle3", "draft", "mmproj", "lora", "vocab-only"];
    MARKERS.iter().any(|m| stem.contains(m))
}

/// One file in a HuggingFace repo listing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoFile {
    pub path: String,
    pub size_bytes: Option<u64>,
}

/// Pick the GGUF files for one quantization.
///
/// Returns every part of a multi-part model, in order, because half of a
/// split GGUF is not a model — and picking one part and calling it done is a
/// download that succeeds and then fails to load with a confusing error.
pub fn pick_gguf(
    files: &[RepoFile],
    wanted: Option<&str>,
) -> Result<(Vec<RepoFile>, String, bool), ResolveError> {
    let all_ggufs: Vec<&RepoFile> = files
        .iter()
        .filter(|f| f.path.to_ascii_lowercase().ends_with(".gguf"))
        .collect();
    if all_ggufs.is_empty() {
        return Err(err(
            "no .gguf files in this repo — hearth serves GGUF, so a repo of \
             safetensors needs converting first",
        ));
    }

    // Drop draft heads, vision projectors and adapters before ranking. They
    // are never the model being served, and being small they otherwise win on
    // a quantization name the real weights do not carry.
    //
    // Falls back to the unfiltered list when filtering would leave nothing, so
    // a repo containing ONLY auxiliary files still reports on what it has
    // rather than claiming the repo is empty.
    let ggufs: Vec<&RepoFile> = {
        let primary: Vec<&RepoFile> = all_ggufs
            .iter()
            .copied()
            .filter(|f| !is_auxiliary_gguf(&f.path))
            .collect();
        if primary.is_empty() {
            all_ggufs.clone()
        } else {
            primary
        }
    };

    let available = available_quants(&ggufs);

    let (chosen, was_chosen_for_you) = match wanted {
        Some(w) => {
            let up = w.to_ascii_uppercase();
            if !available.iter().any(|q| q == &up) {
                return Err(err(format!(
                    "no {up} in this repo — available: {}",
                    if available.is_empty() {
                        "none detected".into()
                    } else {
                        available.join(", ")
                    }
                )));
            }
            (up, false)
        }
        None => {
            let pick = QUANT_PREFERENCE
                .iter()
                .find(|p| available.iter().any(|q| q == *p))
                .map(|p| p.to_string())
                // A repo with exactly one GGUF and an unrecognized name is
                // still perfectly usable — take it rather than refusing.
                .or_else(|| (ggufs.len() == 1).then(|| "unknown".to_string()))
                .ok_or_else(|| {
                    err(format!(
                        "could not choose between: {} — pin one with @QUANT",
                        available.join(", ")
                    ))
                })?;
            (pick, true)
        }
    };

    let mut parts: Vec<RepoFile> = if chosen == "unknown" {
        ggufs.iter().map(|f| (*f).clone()).collect()
    } else {
        ggufs
            .iter()
            .filter(|f| quant_of(&f.path).as_deref() == Some(chosen.as_str()))
            .map(|f| (*f).clone())
            .collect()
    };

    // Sorted so `00002-of-00003` follows `00001-of-00003` regardless of the
    // order the API listed them in.
    parts.sort_by(|a, b| a.path.cmp(&b.path));

    if let Some(expected) = parts.iter().find_map(|f| total_parts(&f.path)) {
        if parts.len() as u32 != expected {
            return Err(err(format!(
                "{chosen} is split into {expected} parts but only {} are present — \
                 an incomplete split GGUF will download fine and then fail to load",
                parts.len()
            )));
        }
    }

    Ok((parts, chosen, was_chosen_for_you))
}

/// Select a complete Hugging Face safetensors package.
///
/// Unlike GGUF, a safetensors model is not one opaque file: Transformers
/// needs the weight file(s) plus configuration and tokenizer metadata. We
/// therefore return a model directory plan, not "the largest .safetensors".
pub fn pick_safetensors(files: &[RepoFile]) -> Result<Vec<RepoFile>, ResolveError> {
    let mut weights: Vec<RepoFile> = files
        .iter()
        .filter(|f| {
            let p = f.path.to_ascii_lowercase();
            p.ends_with(".safetensors")
                && !p.contains("adapter")
                && !p.contains("lora")
                && !p.contains("optimizer")
        })
        .cloned()
        .collect();

    if weights.is_empty() {
        return Err(err("no .gguf or .safetensors model weights in this repo"));
    }

    weights.sort_by(|a, b| a.path.cmp(&b.path));

    // If the repo publishes a safetensors index, keep it. Sharded
    // Transformers loaders use this to map tensor names to shard files.
    const SUPPORT_FILES: &[&str] = &[
        "model.safetensors.index.json",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "chat_template.json",
    ];

    let mut package = weights;
    for f in files {
        let name = f.path.rsplit('/').next().unwrap_or(&f.path);
        if SUPPORT_FILES.iter().any(|wanted| name == *wanted)
            && !package.iter().any(|p| p.path == f.path)
        {
            package.push(f.clone());
        }
    }

    package.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(package)
}

/// Pick a Hugging Face model without changing existing GGUF preference.
///
/// If the caller pins a quantization, that is an explicit GGUF request.
/// Without a pin, GGUF remains preferred when present; safetensors is the
/// fallback for native Transformers/vLLM repositories such as Imagine.
pub fn pick_hf_model(
    files: &[RepoFile],
    wanted_quant: Option<&str>,
) -> Result<(Vec<RepoFile>, ModelFormat, Option<String>, bool), ResolveError> {
    if wanted_quant.is_some()
        || files
            .iter()
            .any(|f| f.path.to_ascii_lowercase().ends_with(".gguf"))
    {
        let (parts, quant, chosen) = pick_gguf(files, wanted_quant)?;
        return Ok((parts, ModelFormat::Gguf, Some(quant), chosen));
    }

    let files = pick_safetensors(files)?;
    Ok((files, ModelFormat::SafeTensors, None, false))
}

/// Every quantization this repo appears to offer.
pub fn available_quants(files: &[&RepoFile]) -> Vec<String> {
    let mut out: Vec<String> = files.iter().filter_map(|f| quant_of(&f.path)).collect();
    out.sort();
    out.dedup();
    out
}

/// The quantization named in a filename, if one is.
///
/// GGUF naming is a convention, not a standard: `model.Q4_K_M.gguf`,
/// `model-q4_k_m.gguf`, `Model.IQ3_XS.gguf`. Matching on the token between
/// separators handles all of them without a regex dependency.
pub fn quant_of(path: &str) -> Option<String> {
    let stem = path.rsplit('/').next().unwrap_or(path);
    let stem = stem
        .strip_suffix(".gguf")
        .or_else(|| stem.strip_suffix(".GGUF"))?;
    stem.split(['.', '-', '_'])
        .collect::<Vec<_>>()
        .windows(1)
        .filter_map(|w| w.first().copied())
        // Reassemble multi-token quants (Q4_K_M splits on '_') by scanning the
        // original stem for each known name instead of trusting the split.
        .next()
        .and_then(|_| {
            let upper = stem.to_ascii_uppercase();
            KNOWN_QUANTS
                .iter()
                .filter(|q| is_token(&upper, q))
                // Longest match wins: Q4_K_M must not be reported as Q4_K.
                .max_by_key(|q| q.len())
                .map(|q| q.to_string())
        })
}

/// Every quantization name we recognize, longest-match-wins at lookup.
const KNOWN_QUANTS: &[&str] = &[
    "Q2_K", "Q2_K_L", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q3_K_XL", "Q3_K", "Q4_0", "Q4_1", "Q4_K_S",
    "Q4_K_M", "Q4_K_L", "Q4_K_XL", "Q4_K", "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M", "Q5_K_L", "Q5_K",
    "Q6_K_L", "Q6_K", "Q8_K", "Q8_0", "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
    "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "IQ4_XS", "IQ4_NL", "TQ1_0", "TQ2_0",
    // Block-float formats. MXFP4 is not a community repack: gpt-oss is
    // PUBLISHED in it, so a resolver that does not know the name cannot pull
    // the reference weights at all -- `@MXFP4` failed with "no MXFP4 in this
    // repo" while `gpt-oss-20b-MXFP4.gguf` sat in the listing.
    "MXFP4", "MXFP8", "NVFP4", "FP8", "F16", "FP16", "BF16", "F32",
];

/// Is `needle` present in `hay` as a whole token rather than a substring?
/// Without this, `Q4_K` matches inside `Q4_K_M` and the wrong file gets picked.
fn is_token(hay: &str, needle: &str) -> bool {
    let bytes = hay.as_bytes();
    let n = needle.as_bytes();
    let sep = |b: u8| !(b.is_ascii_alphanumeric());
    let mut i = 0;
    while let Some(pos) = hay[i..].find(needle) {
        let start = i + pos;
        let end = start + n.len();
        let before_ok = start == 0 || sep(bytes[start - 1]);
        let after_ok = end == bytes.len() || sep(bytes[end]);
        if before_ok && after_ok {
            return true;
        }
        i = start + 1;
        if i >= hay.len() {
            break;
        }
    }
    false
}

/// `...-00001-of-00003.gguf` -> 3
pub fn total_parts(path: &str) -> Option<u32> {
    let stem = path.rsplit('/').next()?;
    let idx = stem.to_ascii_lowercase().find("-of-")?;
    let after = &stem[idx + 4..];
    let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// Where in HuggingFace a set of files lives.
///
/// A struct rather than four more positional `&str`s: `(owner, repo,
/// revision)` are all strings, so transposing two of them compiles perfectly
/// and produces a download URL for a repo that does not exist. Named fields
/// make that mistake unrepresentable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfSource<'a> {
    pub owner: &'a str,
    pub repo: &'a str,
    pub revision: &'a str,
}

/// What we settled on, and whether the caller or hearth decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantChoice {
    pub quant: String,
    pub chosen_for_you: bool,
}

/// Build the plan once the files are chosen.
pub fn plan_from_hf_files(
    src: HfSource<'_>,
    files: Vec<RepoFile>,
    choice: QuantChoice,
    key: String,
    display_name: String,
) -> FetchPlan {
    let (owner, repo, revision) = (src.owner, src.repo, src.revision);
    let QuantChoice {
        quant,
        chosen_for_you,
    } = choice;
    let blobs = files
        .into_iter()
        .map(|f| Blob {
            url: format!(
                "https://huggingface.co/{owner}/{repo}/resolve/{revision}/{}",
                f.path
            ),
            filename: f.path.rsplit('/').next().unwrap_or(&f.path).to_string(),
            // HuggingFace does not put a content digest in the file listing.
            // Saying None is honest; inventing one would make verification a
            // ritual that always passes.
            digest: None,
            size_bytes: f.size_bytes,
        })
        .collect();

    FetchPlan {
        key,
        display_name,
        format: ModelFormat::Gguf,
        blobs,
        quant: Some(quant),
        quant_was_chosen_for_you: chosen_for_you,
    }
}


#[cfg(test)]
mod safetensors_tests {
    use super::*;

    fn files() -> Vec<RepoFile> {
        vec![
            RepoFile { path: "config.json".into(), size_bytes: Some(100) },
            RepoFile { path: "tokenizer.json".into(), size_bytes: Some(200) },
            RepoFile { path: "model.safetensors.index.json".into(), size_bytes: Some(50) },
            RepoFile { path: "model-00001-of-00002.safetensors".into(), size_bytes: Some(10) },
            RepoFile { path: "model-00002-of-00002.safetensors".into(), size_bytes: Some(11) },
            RepoFile { path: "adapter_model.safetensors".into(), size_bytes: Some(1) },
            RepoFile { path: "README.md".into(), size_bytes: Some(1) },
        ]
    }

    #[test]
    fn safetensors_selection_keeps_weights_and_runtime_metadata() {
        let picked = pick_safetensors(&files()).unwrap();
        let names: Vec<&str> = picked.iter().map(|f| f.path.as_str()).collect();
        assert!(names.contains(&"config.json"));
        assert!(names.contains(&"tokenizer.json"));
        assert!(names.contains(&"model.safetensors.index.json"));
        assert!(names.contains(&"model-00001-of-00002.safetensors"));
        assert!(names.contains(&"model-00002-of-00002.safetensors"));
        assert!(!names.contains(&"adapter_model.safetensors"));
        assert!(!names.contains(&"README.md"));
    }

    #[test]
    fn unpinned_hf_falls_back_to_safetensors_when_no_gguf_exists() {
        let (_, format, quant, chosen) = pick_hf_model(&files(), None).unwrap();
        assert_eq!(format, ModelFormat::SafeTensors);
        assert_eq!(quant, None);
        assert!(!chosen);
    }

    #[test]
    fn gguf_still_wins_when_both_formats_exist() {
        let mut both = files();
        both.push(RepoFile {
            path: "model.Q4_K_M.gguf".into(),
            size_bytes: Some(123),
        });
        let (picked, format, quant, _) = pick_hf_model(&both, None).unwrap();
        assert_eq!(format, ModelFormat::Gguf);
        assert_eq!(quant.as_deref(), Some("Q4_K_M"));
        assert_eq!(picked.len(), 1);
    }
}
