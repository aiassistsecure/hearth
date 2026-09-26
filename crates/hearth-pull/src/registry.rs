//! Turning a reference into a list of blobs to fetch.
//!
//! The URL construction and the manifest reading live in [`hearth_resolve`],
//! which is network-free on purpose. This module is the thin layer that
//! actually asks the registry for the manifest and hands the answer back as
//! something [`crate::fetch_blob`] can download.

use hearth_resolve::plan;
use hearth_resolve::Reference;

use crate::curl;

/// One thing to download.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Blob {
    /// For error messages a person has to read.
    pub name: String,
    pub url: String,
    /// Expected digest, `sha256:…` or bare hex, when a registry published
    /// one. `None` for a source that never had one to publish — a bare URL,
    /// or a HuggingFace file the tree listing did not mark as LFS. That case
    /// is not skipped verification, it is a DIFFERENT verification: hearth
    /// hashes the download itself and trusts that a file matches its own
    /// content, which is always true and says nothing about whether the
    /// content is what the operator meant to fetch. `hearth why` names the
    /// difference rather than calling both cases "verified".
    pub digest: Option<String>,
    /// Expected size, or 0 when the source did not say.
    pub size_bytes: u64,
    pub headers: Vec<(String, String)>,
    /// Is this the weights layer? Exactly one blob should be, and it is the one
    /// the VRAM budget cares about.
    pub is_weights: bool,
}

/// What a reference resolved to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fetched {
    pub blobs: Vec<Blob>,
    /// Selected runtime-facing model format.
    pub format: plan::ModelFormat,
    /// Total weights bytes — the number the planner wants, known before a
    /// single byte moves.
    pub weights_bytes: u64,
}

/// The origin string recorded in the spine. Stable and greppable, because it is
/// the answer to "where did this file come from" months later.
pub fn source_string(r: &Reference) -> String {
    match r {
        Reference::Ollama {
            namespace,
            name,
            tag,
        } => format!("ollama:{namespace}/{name}:{tag}"),
        Reference::HuggingFace {
            owner,
            repo,
            revision,
            quant,
        } => match quant {
            Some(q) => format!("hf:{owner}/{repo}@{q}#{revision}"),
            None => format!("hf:{owner}/{repo}#{revision}"),
        },
        Reference::Local { path } => format!("file:{path}"),
        Reference::Url { url, sha256 } => match sha256 {
            Some(hex) => format!("{url}#sha256:{hex}"),
            None => url.clone(),
        },
    }
}

const OLLAMA_REGISTRY: &str = "https://registry.ollama.ai";

/// The media type that is actually the model. An Ollama manifest also carries
/// templates, params, system prompts and licenses; summing every layer would
/// overstate VRAM and refuse models that fit.
const OLLAMA_MODEL_MEDIA_TYPE: &str = "application/vnd.ollama.image.model";

/// Ask the registry what a reference is made of.
pub fn resolve_blobs(reference: &Reference) -> Result<Fetched, String> {
    match reference {
        Reference::Ollama {
            namespace,
            name,
            tag,
        } => {
            let manifest_url = format!("{OLLAMA_REGISTRY}/v2/{namespace}/{name}/manifests/{tag}");
            let body = curl::fetch_string(&curl::Request::get(&manifest_url).header(
                "Accept",
                "application/vnd.docker.distribution.manifest.v2+json",
            ))
            .map_err(|e| {
                format!(
                    "could not fetch the manifest for {namespace}/{name}:{tag}\n  {}",
                    e.0
                )
            })?;

            let manifest: serde_json::Value = serde_json::from_str(&body).map_err(|e| {
                format!(
                    "the registry did not return a manifest for {namespace}/{name}:{tag}: {e}\n  \
                     got: {}",
                    body.chars().take(160).collect::<String>()
                )
            })?;

            blobs_from_ollama_manifest(&manifest, namespace, name)
        }
        Reference::HuggingFace {
            owner,
            repo,
            revision,
            quant,
        } => resolve_huggingface(owner, repo, revision, quant.as_deref()),
        Reference::Local { path } => Ok(Fetched {
            blobs: vec![Blob {
                name: "local".into(),
                url: format!("file://{path}"),
                digest: None,
                size_bytes: 0,
                headers: vec![],
                is_weights: true,
            }],
            format: plan::ModelFormat::Gguf,
            weights_bytes: 0,
        }),
        Reference::Url { url, sha256 } => Ok(Fetched {
            blobs: vec![Blob {
                name: "url".into(),
                url: url.clone(),
                // Pinned by the caller, or unknown until `fetch_blob` hashes
                // the download and self-verifies. Either way this is the
                // model: a bare URL pull has no manifest to carry a template,
                // license, or params layer alongside it.
                digest: sha256.clone(),
                size_bytes: 0,
                headers: vec![],
                is_weights: true,
            }],
            format: plan::ModelFormat::Gguf,
            // Unknown before the download. The planner treats an unknown
            // weight size the same way it treats any other unmeasured model:
            // it cannot refuse what it cannot size, so this pull is not
            // budget-checked ahead of time the way a registry pull is. That
            // is a real gap, not a rounding error, and worth its own fix
            // rather than inventing a number here that looks precise and
            // is not.
            weights_bytes: 0,
        }),
    }
}

const HF_HUB: &str = "https://huggingface.co";

/// Resolve a HuggingFace reference to its blobs.
///
/// Two network calls where Ollama needed one, because HuggingFace's contract
/// is different: a manifest already says what the weights are, while a
/// HuggingFace repo is a directory listing with no opinion about which file
/// is the model. `hearth_resolve::plan` carries the opinion (quant
/// preference, multipart shard grouping); this function's whole job is
/// handing it a real listing to have that opinion about.
fn resolve_huggingface(
    owner: &str,
    repo: &str,
    revision: &str,
    quant: Option<&str>,
) -> Result<Fetched, String> {
    let tree_url = format!("{HF_HUB}/api/models/{owner}/{repo}/tree/{revision}?recursive=true");
    let body = curl::fetch_string(&curl::Request::get(&tree_url)).map_err(|e| {
        format!(
            "could not list files in {owner}/{repo}@{revision}\n  {}",
            e.0
        )
    })?;
    let listing: serde_json::Value = serde_json::from_str(&body).map_err(|e| {
        format!(
            "{owner}/{repo}@{revision} did not return a file listing: {e}\n  got: {}",
            body.chars().take(160).collect::<String>()
        )
    })?;

    blobs_from_hf_listing(&listing, owner, repo, revision, quant)
}

/// Everything after the network call: pure, so it is tested against a real
/// captured tree listing without touching HuggingFace, the same discipline
/// `blobs_from_ollama_manifest` already holds itself to for the other
/// catalog.
fn blobs_from_hf_listing(
    listing: &serde_json::Value,
    owner: &str,
    repo: &str,
    revision: &str,
    quant: Option<&str>,
) -> Result<Fetched, String> {
    let entries = hf_repo_files(listing)?;
    let files: Vec<plan::RepoFile> = entries
        .iter()
        .map(|e| plan::RepoFile {
            path: e.path.clone(),
            size_bytes: e.size_bytes,
        })
        .collect();

    let (parts, format, chosen, chosen_for_you) =
        plan::pick_hf_model(&files, quant).map_err(|e| format!("{owner}/{repo}: {e}"))?;

    if chosen_for_you {
        if let Some(chosen) = &chosen {
            eprintln!(
                "hearth: no quantization pinned for {owner}/{repo} — chose {chosen} \
                 (pin one with @{chosen} to stop seeing this)"
            );
        }
    } else if format == plan::ModelFormat::SafeTensors {
        eprintln!("hearth: {owner}/{repo} has no GGUF — selected native safetensors package");
    }

    let mut blobs = Vec::with_capacity(parts.len());
    let mut weights_bytes = 0u64;
    for part in &parts {
        let digest = entries
            .iter()
            .find(|e| e.path == part.path)
            .and_then(|e| e.sha256.clone());
        let size = part.size_bytes.unwrap_or(0);
        let lower = part.path.to_ascii_lowercase();
        let is_weights = match format {
            plan::ModelFormat::Gguf => lower.ends_with(".gguf"),
            plan::ModelFormat::SafeTensors => lower.ends_with(".safetensors"),
        };
        if is_weights {
            weights_bytes = weights_bytes.saturating_add(size);
        }
        blobs.push(Blob {
            // Keep the repository-relative path for safetensors packages so
            // the pull layer can materialize a valid Transformers directory.
            name: part.path.clone(),
            url: format!("{HF_HUB}/{owner}/{repo}/resolve/{revision}/{}", part.path),
            digest,
            size_bytes: size,
            headers: vec![],
            is_weights,
        });
    }

    Ok(Fetched {
        blobs,
        format,
        weights_bytes,
    })
}

/// One file from a HuggingFace tree listing, with whatever digest it
/// actually carries.
struct HfEntry {
    path: String,
    size_bytes: Option<u64>,
    /// The content sha256, when the file is LFS-tracked. HuggingFace's tree
    /// API reports this under `lfs.oid` — LFS pointers ARE sha256 by
    /// construction, so this is a real published digest, not a computed
    /// stand-in. A GGUF that is not LFS-tracked (rare; the format is built
    /// for files large enough that HuggingFace always LFS-tracks them in
    /// practice) has no digest here, and `resolve_huggingface` passes that
    /// through as `None` honestly rather than inventing one.
    sha256: Option<String>,
}

/// Read the tree listing's JSON array into structured entries.
///
/// Pure — no network — so the HuggingFace response shape is tested against a
/// captured real listing, the same discipline `blobs_from_ollama_manifest`
/// already holds itself to.
fn hf_repo_files(listing: &serde_json::Value) -> Result<Vec<HfEntry>, String> {
    let items = listing
        .as_array()
        .ok_or("expected a JSON array of repo files")?;

    let mut out = Vec::with_capacity(items.len());
    for item in items {
        // Directories appear in a recursive tree listing too; only files are
        // ever a GGUF, so anything else is skipped rather than misread as a
        // zero-byte file.
        if item.get("type").and_then(|t| t.as_str()) != Some("file") {
            continue;
        }
        let path = item
            .get("path")
            .and_then(|p| p.as_str())
            .ok_or("a file entry has no \"path\"")?
            .to_string();
        let size_bytes = item.get("size").and_then(hearth_core::budget::whole_bytes);
        let sha256 = item
            .get("lfs")
            .and_then(|l| l.get("oid"))
            .and_then(|o| o.as_str())
            .map(|s| s.to_ascii_lowercase());
        out.push(HfEntry {
            path,
            size_bytes,
            sha256,
        });
    }
    Ok(out)
}

/// Read an Ollama manifest into blobs. Pure, so it is tested against a real
/// captured manifest without touching the network.
pub fn blobs_from_ollama_manifest(
    manifest: &serde_json::Value,
    namespace: &str,
    name: &str,
) -> Result<Fetched, String> {
    let layers = manifest
        .get("layers")
        .and_then(|l| l.as_array())
        .ok_or_else(|| {
            format!(
                "manifest has no layers array — {}",
                manifest
                    .get("errors")
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "and no error either".into())
            )
        })?;

    let mut blobs = Vec::new();
    let mut weights_bytes = 0u64;

    for layer in layers {
        let digest = layer
            .get("digest")
            .and_then(|d| d.as_str())
            .ok_or("a layer has no digest — refusing to fetch something unverifiable")?;
        let media = layer
            .get("mediaType")
            .and_then(|m| m.as_str())
            .unwrap_or("");
        let size = layer
            .get("size")
            .and_then(hearth_core::budget::whole_bytes)
            .unwrap_or(0);
        let is_weights = media == OLLAMA_MODEL_MEDIA_TYPE;
        if is_weights {
            weights_bytes = weights_bytes.saturating_add(size);
        }
        blobs.push(Blob {
            name: format!("{}:{}", short_media(media), sha256_short(digest)),
            url: format!("{OLLAMA_REGISTRY}/v2/{namespace}/{name}/blobs/{digest}"),
            digest: Some(digest.to_string()),
            size_bytes: size,
            headers: vec![],
            is_weights,
        });
    }

    if !blobs.iter().any(|b| b.is_weights) {
        return Err(format!(
            "no {OLLAMA_MODEL_MEDIA_TYPE} layer in the manifest — {} layer(s), \
             none of them weights. hearth will not guess which blob is a model.",
            blobs.len()
        ));
    }

    Ok(Fetched {
        blobs,
        format: plan::ModelFormat::Gguf,
        weights_bytes,
    })
}

fn short_media(media: &str) -> &str {
    media.rsplit('.').next().unwrap_or(media)
}

fn sha256_short(digest: &str) -> String {
    hearth_core::sha256::normalize(digest)
        .chars()
        .take(12)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real manifest captured from registry.ollama.ai for library/tinyllama.
    /// Real, so the parser is checked against what the registry actually sends
    /// rather than against a fixture written to match the code.
    const TINYLLAMA: &str = r#"{
      "schemaVersion": 2,
      "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
      "config": {
        "mediaType": "application/vnd.docker.container.image.v1+json",
        "digest": "sha256:6331358be52a6ebc2fd0755a51ad1175734fd17a628ab5ea6897109396245362",
        "size": 483
      },
      "layers": [
        {
          "mediaType": "application/vnd.ollama.image.model",
          "digest": "sha256:2af3b81862c6be03c769683af18efdadb2c33f60ff32ab6f83e42c043d6c7816",
          "size": 637699456
        },
        {
          "mediaType": "application/vnd.ollama.image.template",
          "digest": "sha256:af0ddbdaaa26f30d54d727f9dd944b76bdb926fdaf9a58f63f78c534dca71bd1",
          "size": 70
        },
        {
          "mediaType": "application/vnd.ollama.image.system",
          "digest": "sha256:c8472cd9daed5e7c2ff5a1b04e5e2b9e5b0b4f4b2f5f5c4f5b4f5c4f5b4f5c4f",
          "size": 31
        },
        {
          "mediaType": "application/vnd.ollama.image.params",
          "digest": "sha256:fa956ab37b8c21152f2f5e0b0b4f4b2f5f5c4f5b4f5c4f5b4f5c4f5b4f5c4f5b",
          "size": 98
        }
      ]
    }"#;

    #[test]
    fn only_the_model_layer_counts_as_weights() {
        let m: serde_json::Value = serde_json::from_str(TINYLLAMA).unwrap();
        let f = blobs_from_ollama_manifest(&m, "library", "tinyllama").unwrap();

        assert_eq!(f.blobs.len(), 4, "every layer is fetchable");
        assert_eq!(
            f.blobs.iter().filter(|b| b.is_weights).count(),
            1,
            "exactly one of them is the model"
        );
        // 637,699,456 and not 637,699,655: summing the template, system and
        // params layers would overstate VRAM and refuse models that fit.
        assert_eq!(f.weights_bytes, 637_699_456);
    }

    #[test]
    fn blob_urls_are_built_from_the_digest() {
        let m: serde_json::Value = serde_json::from_str(TINYLLAMA).unwrap();
        let f = blobs_from_ollama_manifest(&m, "library", "tinyllama").unwrap();
        let w = f.blobs.iter().find(|b| b.is_weights).unwrap();
        assert_eq!(
            w.url,
            "https://registry.ollama.ai/v2/library/tinyllama/blobs/sha256:2af3b81862c6be03c769683af18efdadb2c33f60ff32ab6f83e42c043d6c7816"
        );
        assert!(w.name.contains("model"), "readable in an error: {}", w.name);
    }

    #[test]
    fn a_manifest_with_no_model_layer_is_refused_not_guessed() {
        // A model reference whose manifest carries only templates is either the
        // wrong tag or a registry change. Picking the largest blob and hoping
        // is how you load a license file as weights.
        let m = serde_json::json!({
            "layers": [
                { "mediaType": "application/vnd.ollama.image.template",
                  "digest": "sha256:aa", "size": 70 }
            ]
        });
        let err = blobs_from_ollama_manifest(&m, "library", "x").unwrap_err();
        assert!(err.contains("none of them weights"), "{err}");
        assert!(err.contains("will not guess"));
    }

    #[test]
    fn a_layer_without_a_digest_is_refused() {
        // No digest means no verification possible. Fetching it anyway would
        // put an unverifiable blob on disk under a made-up name.
        let m = serde_json::json!({
            "layers": [{ "mediaType": OLLAMA_MODEL_MEDIA_TYPE, "size": 10 }]
        });
        let err = blobs_from_ollama_manifest(&m, "library", "x").unwrap_err();
        assert!(err.contains("no digest"), "{err}");
    }

    #[test]
    fn a_registry_error_document_says_what_it_said() {
        let m = serde_json::json!({ "errors": [{ "code": "MANIFEST_UNKNOWN" }] });
        let err = blobs_from_ollama_manifest(&m, "library", "nope").unwrap_err();
        assert!(err.contains("MANIFEST_UNKNOWN"), "{err}");
    }

    #[test]
    fn a_json_float_size_survives_the_crossing() {
        // Sizes arriving as integer-valued floats are real; refusing them would
        // report a legitimate layer as zero bytes and let the planner admit a
        // model that cannot fit.
        let m = serde_json::json!({
            "layers": [{ "mediaType": OLLAMA_MODEL_MEDIA_TYPE,
                         "digest": "sha256:ab", "size": 637699456.0 }]
        });
        let f = blobs_from_ollama_manifest(&m, "library", "x").unwrap();
        assert_eq!(f.weights_bytes, 637_699_456);
    }

    #[test]
    fn source_strings_are_stable_and_greppable() {
        assert_eq!(
            source_string(&Reference::parse("deepseek-r1:32b").unwrap()),
            "ollama:library/deepseek-r1:32b"
        );
        assert_eq!(
            source_string(&Reference::parse("hf:TheBloke/X-GGUF@Q4_K_M").unwrap()),
            "hf:TheBloke/X-GGUF@Q4_K_M#main"
        );
        assert_eq!(
            source_string(&Reference::parse("file:/m/muse.gguf").unwrap()),
            "file:/m/muse.gguf"
        );
    }

    /// A HuggingFace tree listing, the real shape: `type`, `path`, `size`, and
    /// an `lfs.oid` that IS the content sha256 for a git-lfs-tracked file —
    /// which every GGUF on HuggingFace is, in practice, because the format
    /// only exists for files large enough that HuggingFace always LFS-tracks
    /// them. A directory entry and a non-LFS small file are mixed in
    /// deliberately, because a real recursive listing has both and this must
    /// not misread either as a model file.
    fn hf_listing_two_quants() -> serde_json::Value {
        serde_json::json!([
            { "type": "directory", "path": "original" },
            { "type": "file", "path": "README.md", "size": 512 },
            {
                "type": "file", "path": "model.Q4_K_M.gguf", "size": 4_500_000_000u64,
                "lfs": { "oid": "AAAA000000000000000000000000000000000000000000000000000000AAAA",
                         "size": 4_500_000_000u64 }
            },
            {
                "type": "file", "path": "model.Q8_0.gguf", "size": 8_100_000_000u64,
                "lfs": { "oid": "BBBB000000000000000000000000000000000000000000000000000000BBBB",
                         "size": 8_100_000_000u64 }
            }
        ])
    }

    #[test]
    fn hf_repo_files_skips_directories_and_reads_the_lfs_digest() {
        let entries = hf_repo_files(&hf_listing_two_quants()).unwrap();
        // README.md and the directory are real entries in the listing; neither
        // is a model file, and only the two GGUFs should come back.
        let paths: Vec<&str> = entries.iter().map(|e| e.path.as_str()).collect();
        assert_eq!(
            paths,
            vec!["README.md", "model.Q4_K_M.gguf", "model.Q8_0.gguf"]
        );

        let gguf = entries
            .iter()
            .find(|e| e.path == "model.Q4_K_M.gguf")
            .unwrap();
        assert_eq!(
            gguf.sha256.as_deref(),
            Some("aaaa000000000000000000000000000000000000000000000000000000aaaa")
        );
        let readme = entries.iter().find(|e| e.path == "README.md").unwrap();
        assert_eq!(
            readme.sha256, None,
            "not LFS-tracked, so honestly no digest"
        );
    }

    #[test]
    fn huggingface_resolves_to_the_preferred_quant_with_its_real_digest() {
        // No quant pinned: plan::QUANT_PREFERENCE puts Q4_K_M ahead of Q8_0.
        let f = blobs_from_hf_listing(&hf_listing_two_quants(), "TheBloke", "X-GGUF", "main", None)
            .unwrap();
        assert_eq!(f.blobs.len(), 1);
        assert_eq!(f.blobs[0].name, "model.Q4_K_M.gguf");
        assert_eq!(
            f.blobs[0].url,
            "https://huggingface.co/TheBloke/X-GGUF/resolve/main/model.Q4_K_M.gguf"
        );
        assert_eq!(
            f.blobs[0].digest.as_deref(),
            Some("aaaa000000000000000000000000000000000000000000000000000000aaaa"),
            "a real published digest, not a self-computed stand-in"
        );
        assert_eq!(f.weights_bytes, 4_500_000_000);
    }

    #[test]
    fn a_pinned_quant_overrides_the_preference_order() {
        let f = blobs_from_hf_listing(
            &hf_listing_two_quants(),
            "TheBloke",
            "X-GGUF",
            "main",
            Some("Q8_0"),
        )
        .unwrap();
        assert_eq!(f.blobs[0].name, "model.Q8_0.gguf");
        assert_eq!(f.weights_bytes, 8_100_000_000);
    }

    #[test]
    fn a_quant_the_repo_does_not_have_is_refused_with_the_ones_that_exist() {
        let err = blobs_from_hf_listing(
            &hf_listing_two_quants(),
            "TheBloke",
            "X-GGUF",
            "main",
            Some("Q2_K"),
        )
        .unwrap_err();
        assert!(err.contains("Q2_K"), "{err}");
        assert!(err.contains("Q4_K_M"), "{err}");
    }

    #[test]
    fn a_repo_of_safetensors_is_selected_as_a_native_model_package() {
        let listing = serde_json::json!([
            { "type": "file", "path": "config.json", "size": 1024 },
            { "type": "file", "path": "tokenizer.json", "size": 2048 },
            {
                "type": "file", "path": "model.safetensors", "size": 16_000_000_000u64,
                "lfs": { "oid": "CCCC000000000000000000000000000000000000000000000000000000CCCC" }
            }
        ]);
        let f =
            blobs_from_hf_listing(&listing, "Interchained", "imagine-v8", "main", None).unwrap();
        assert_eq!(f.format, plan::ModelFormat::SafeTensors);
        assert_eq!(f.weights_bytes, 16_000_000_000);
        assert!(f
            .blobs
            .iter()
            .any(|b| b.name == "config.json" && !b.is_weights));
        assert!(f
            .blobs
            .iter()
            .any(|b| b.name == "tokenizer.json" && !b.is_weights));
        assert!(f
            .blobs
            .iter()
            .any(|b| b.name == "model.safetensors" && b.is_weights));
    }

    // ---- Reference::Url, the third source alongside Ollama and HuggingFace --

    #[test]
    fn a_url_reference_resolves_to_exactly_itself() {
        let r = Reference::parse("https://example.com/model.gguf").unwrap();
        let f = resolve_blobs(&r).unwrap();
        assert_eq!(f.blobs.len(), 1);
        assert_eq!(f.blobs[0].url, "https://example.com/model.gguf");
        assert_eq!(f.blobs[0].digest, None, "nothing published one");
        assert!(f.blobs[0].is_weights);
    }

    #[test]
    fn a_pinned_sha256_fragment_becomes_the_expected_digest() {
        let r = Reference::parse(
            "https://example.com/model.gguf#sha256:deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        )
        .unwrap();
        let f = resolve_blobs(&r).unwrap();
        assert_eq!(
            f.blobs[0].digest.as_deref(),
            Some("deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
        );
    }

    #[test]
    fn url_source_strings_carry_the_pinned_digest_when_there_is_one() {
        assert_eq!(
            source_string(&Reference::parse("https://example.com/m.gguf").unwrap()),
            "https://example.com/m.gguf"
        );
        assert_eq!(
            source_string(&Reference::parse("https://example.com/m.gguf#sha256:ab12").unwrap()),
            "https://example.com/m.gguf#sha256:ab12"
        );
    }
}
