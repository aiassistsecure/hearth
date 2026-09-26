# hearth

**Deterministic model residency.** Keep declared models warm, and tell the truth about which ones are.

Not an inference engine. llama.cpp and vLLM have spent years on kernels, samplers and tokenizers, and none of that is the problem. The problem is that no serving stack will *promise* a model stays loaded, and none of them can say *why* one stopped being.

## The night this came from

A PIN operator on one rented RTX A6000 stopped answering. Requests hung, then failed. Four pull requests landed against the streaming path in a single evening and none of them were the cause — because the cause was never visible. Three completely different failures were arriving as the same timeout:

- the runtime evicted a model to free VRAM,
- the host detached the GPU and gave it to another tenant,
- a 32B model was simply still loading.

One of those is a capacity problem you own. One is your provider's and no configuration will touch it. One is not a problem at all. A timeout cannot tell you which, so all three got "fixed" repeatedly and none of them went away.

Worse, the operator was being **scored down** for a card their host reclaimed. A reputation system fed that kind of data slowly deletes its own honest operators.

## What it does

```
$ hearth status
42.0 / 44.2 GiB held  (5 declared, 2 admitted)
  muse-local:latest            resident for 14203s
  deepseek-r1:32b              loading for 47s
  gemma4:26b                   not admitted — short by 14.8 GiB
  qwen3.6:27b                  not admitted — short by 15.8 GiB
  gemma4-extract:31b           not admitted — short by 17.8 GiB
```

Three things, none of which you can get today:

**1. Residency is a named state with a named reason.**

```
Unknown    never probed
Loading    weights materializing — with elapsed, so it can say how long
Resident   loaded AND answering AND accounted for
Lost       with a reason: Evicted · GpuDetached · ProcessExited · Unhealthy
Failed     won't load, and what the runtime said
Stopped    unloaded on purpose
```

`Evicted` and `GpuDetached` are the two states nothing else reports, and they were the two that mattered. They are distinguished by one bit — whether the GPU was still present when the probe failed — and that bit is the entire diagnosis.

**2. The card's size is arithmetic, checked before anything loads.**

Declare four 20 GiB models on a 48 GiB card and no runtime errors. It loads, evicts, loads, evicts, forever, and presents as "the models got slow." hearth refuses the fifth model at declare time and tells you it was short by 17.8 GiB. Nothing is ever evicted to make room for a load — if it doesn't fit, the honest answer is that it doesn't fit.

**3. Routers get an answer they can act on.**

| answer | what a router should do |
|---|---|
| `Ready` | send it |
| `Warming{for_ms}` | wait, or try elsewhere — but do **not** fault this node |
| `Lost{GpuDetached}` | try elsewhere, and do **not** score this operator down |
| `Lost{Evicted}` | try elsewhere; this box is over-committed |
| `NotAdmitted{short}` | stop asking — it will never fit here |
| `Unknown` | we genuinely don't know yet, and we say so |

## Design rules

**Nothing fails on a clock.** A 32B materializing over a network fabric can legitimately spend minutes before its first token, and killing it at an arbitrary deadline turns a slow success into a fast failure. `Loading` reports how long it has been loading; deciding what to do about that belongs to whoever knows if a human is waiting. Progress is reported — patience is a policy, not a constant.

**`Loading` is not `Ready`.** The most common way a serving stack lies is routing to something still coming up and calling the inevitable timeout an error.

**Declaration order is priority order.** First fit, never best fit. Reordering to squeeze in one more model would silently demote whatever the operator listed first, and on a serving box first means most important. A planner that outsmarts the operator surprises them at 3am.

**The reserve is never planned into.** Weights aren't the whole cost — KV cache grows with context and parallelism, the CUDA context is hundreds of megabytes, and fragmentation is real on a card that's been up for weeks.

## Install

| package | registry | what you get |
|---|---|---|
| [`hearth-serve`](crates/hearth-serve/README.md) | [crates.io](https://crates.io/crates/hearth-serve) | `cargo install hearth-serve` — the `hearth` binary |
| [`hearth-core`](crates/hearth-core/README.md) | [crates.io](https://crates.io/crates/hearth-core) | the decision procedure, no GPU needed |
| [`hearth-store`](crates/hearth-store/README.md) | [crates.io](https://crates.io/crates/hearth-store) | the NEDB spine |
| [`hearth-resolve`](crates/hearth-resolve/README.md) | [crates.io](https://crates.io/crates/hearth-resolve) | Ollama + HuggingFace resolution |
| [`hearth-pull`](crates/hearth-pull/README.md) | [crates.io](https://crates.io/crates/hearth-pull) | fetch weights, digest-verified and recorded |
| [`hearth-api`](crates/hearth-api/README.md) | [crates.io](https://crates.io/crates/hearth-api) | one OpenAI-compatible port in front of the fleet |
| [`@interchained/hearth`](crates/hearth-node/README.md) | [npm](https://www.npmjs.com/package/@interchained/hearth) | Node bindings, prebuilt for 4 platforms |
| [`hearth-engine`](crates/hearth-py/README.md) | [PyPI](https://pypi.org/project/hearth-engine/) | Python bindings, one abi3 wheel per platform |

Each package page is written for the language you land in — the API is the same core either way, but the key names are idiomatic per language and [should not be copied between them](examples/README.md#one-thing-to-watch).

## Status

`hearth-core` — state machine, VRAM planner, fleet routing. Pure logic, no GPU required.
`hearth-resolve` — one reference syntax over the Ollama registry and HuggingFace GGUF/safetensors repositories, verified against the live registries. **Resolution only: it plans blob URLs and byte counts, and fetches nothing** — `serde` and `serde_json` are its entire dependency list.
`hearth-store` — **the NEDB spine.** Every residency transition is a bi-temporal, causally-linked, tamper-evident event in an embedded [NEDB](https://github.com/Eth-Interchained/nedb). Not a log on the side: the supervisor's memory IS the database.
`hearth-pull` — the downloader. Digest-verified, resumable, and every fetch recorded in the spine with its origin. Real: 637 MB from `registry.ollama.ai` in 33s, and a 64-byte corruption at offset 300M — same file size — caught by re-hashing and healed.
`hearth-api` — the gateway. One OpenAI-compatible port routing by the `model` field, where the HTTP status carries the diagnosis: warming is a 503 with Retry-After, a reclaimed GPU is a 503 saying it was not the operator's doing, and a model that will never fit is a **409** — because a retryable status there is a router hammering a box that is arithmetically incapable of answering.
`hearth-serve` — the supervisor. `llama-server` children, honest health probes, `gpu_present` via nvidia-smi (an honest `None` on CPU boxes), SIGTERM records `unloaded` and reaps children. Plus the `hearth` CLI.

Proven end-to-end on a real model: `hearth serve` brought a GGUF resident under a real `llama-server` (warmup measured, not guessed), served real completion tokens, took a SIGTERM, and a **fresh process** then read the whole story back off disk:

```
$ hearth why stories
● seq     3  unloaded       {}
└─ seq     2  resident       {"endpoint":"127.0.0.1:18080","warmup_ms":256}
└─ seq     1  loading        {"endpoint":"127.0.0.1:18080","pid":10121}
└─ seq     0  declared       {"admitted":true,"vram_bytes":1073741824}

$ hearth as-of stories 2
as of seq 2: stories was `resident` {"endpoint":"127.0.0.1:18080","warmup_ms":256}

$ hearth verify
verify ok — 4 nodes checked, history intact
```

*What was resident **as of** seq 2* is a real query against a real causal chain. When a model goes cold at 3am you get the answer instead of a theory — and `verify` proves nobody rewrote it. No other serving stack can print that.

Next: HuggingFace fetch (resolve already plans it) · streaming verified against a real llama-server · napi + PyO3 bindings for store/serve/api.

## Integration

hearth speaks OpenAI-compatible, so [pin-clientd](https://github.com/aiassistsecure/pin-clientd) works with it today: set `apiMode: "openai"` and point `inferenceUri` at hearth. `/residency` then adds the truth that the OpenAI shape has no way to express.

## Model formats

hearth keeps the inference runtime separate from residency supervision:

- **GGUF** stays on `llama-server` exactly as before.
- **Hugging Face safetensors** are pulled as a complete Transformers package
  (weights/shards, index, config, tokenizer metadata) and served with vLLM.

When a Hugging Face repository carries both GGUF and safetensors, the existing
GGUF quantization preference remains authoritative. When it carries no GGUF,
hearth falls back to safetensors.

```bash
# Native HF/safetensors example
hearth pull hf:Interchained/imagine-v8

# `hearth pull` prints the materialized directory. Serve it with:
hearth serve --model imagine-v8 --safetensors ~/.hearth/blobs/models/Interchained_imagine-v8

# Or declare it in a fleet (GGUF remains the no-prefix default)
# model muse=/models/muse.gguf:20
# model imagine-v8=safetensors:/models/Interchained_imagine-v8:3@16384
```

Safetensors serving expects the `vllm` CLI on PATH (or an explicit
`--binary`). The same supervisor, health probing, causal spine, and
OpenAI-compatible gateway remain in front of either backend.

### Mixed fleets and interactive use

`hearth up` accepts backend-qualified model declarations:

```bash
hearth up \
  --model muse=gguf:/models/muse.gguf:20@16384 \
  --model imagine-v8=safetensors:/models/Interchained_imagine-v8:3@16384
```

The `gguf:` prefix is optional for backward compatibility. Safetensors
entries select vLLM; GGUF entries select llama.cpp. The same syntax works in
`$HEARTH_HOME/fleet.conf` through `scripts/start.sh`.

Once a fleet is running, `hearth run MODEL` opens a small interactive terminal
against HEARTH's own `/v1/chat/completions` gateway:

```bash
hearth run imagine-v8
```

Use `/clear` to reset the conversation and `/exit` to quit. The REPL is only
a client of the normal gateway, so residency, routing and provenance stay on
the same hardened serving path.

## Build

Every snippet in every README here is a **runnable example** — see [`examples/`](examples/README.md). The numbers in the docs were pasted out of real runs, not written by hand.

```bash
cargo test --workspace   # no GPU needed
cargo run -p hearth-core --example quickstart

# serve a model under supervision (needs llama.cpp's llama-server)
hearth serve --model muse --gguf ./muse.gguf --port 8080
hearth status && hearth why muse && hearth verify
```

---

© Interchained LLC · BUSL-1.1 (converts to Apache-2.0 on 2030-08-27)
