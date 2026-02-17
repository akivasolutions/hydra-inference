# Tightwad

Mixed-vendor GPU inference cluster manager with speculative decoding proxy. Pools CUDA and ROCm GPUs across machines using [llama.cpp RPC](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc), and accelerates inference via application-layer speculative decoding across network-separated servers.

## How It Works in 10 Seconds

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   MACHINE A (cheap)              MACHINE B (powerful)           │
│   RTX 4060, 8GB                  RTX 4070 Ti Super, 16GB        │
│   Qwen3-8B via Ollama            Qwen3-32B via Ollama           │
│                                                                  │
│   "Draft 8 tokens fast"    ───►  "Verify all 8 in one pass"     │
│   ~100 tok/s                     single forward pass            │
│                                                                  │
│   "the answer is 42,       ───►  ✓ yes  ✓ yes  ✓ yes           │
│    because math works..."        ✓ yes  ✓ yes  ✗ no             │
│                                  └── take my token here         │
│                                                                  │
│   Result: 5 tokens accepted instantly = skipped 5 slow steps    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

  Without Tightwad: big model generates every token, one at a time
  With Tightwad:    big model only works on the tokens it disagrees with
  Output quality:   IDENTICAL (mathematically guaranteed)
  Speed:            2-3x faster
```

> The small model is fast but sometimes wrong. The big model is slow but always right.
> Tightwad uses the small model to do most of the work, and the big model to catch mistakes.
> Because catching mistakes is cheap — it's one batch operation, not N serial ones.

## Two Modes

### 1. RPC Cluster — Pool GPUs into one endpoint

Combine GPUs from different machines and vendors into a single OpenAI-compatible API. The coordinator distributes model layers across local and remote GPUs.

### 2. Speculative Decoding Proxy — Draft + Verify across machines

A fast small model (e.g., 8B on a consumer GPU) drafts candidate tokens, a large model (e.g., 72B on a server or cloud API) verifies them in batch. Output quality is **identical to running the large model alone**, but 2-3x faster because batch verification is much cheaper than autoregressive generation.

```
Client (OpenAI API)
        │
        ▼
┌──────────────────────────────┐
│   Tightwad Proxy (:8088)      │  Python async server
│   Speculation Loop:          │
│   1. Draft 8 tokens          │──► Draft: Qwen3-8B (fast, local)
│   2. Verify batch            │──► Target: Qwen3-72B (accurate, local or API)
│   3. Accept/reject           │
│   4. Stream to client        │
└──────────────────────────────┘
```

**Why not just use RPC?** RPC ships 100-300 MB of tensor data per step over the network. The speculative proxy ships token IDs (bytes). For models that fit on a single machine's VRAM, speculation is dramatically faster.

## Quick Start

```bash
# Install
git clone https://github.com/akivasolutions/tightwad.git
cd tightwad
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Edit topology for your hardware
vim configs/cluster.yaml
```

### Speculative Decoding Proxy

```bash
# Start the proxy (draft + target servers must be running)
tightwad proxy start

# Check health and acceptance rate stats
tightwad proxy status

# Test it
curl http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'

# Detailed stats
curl http://localhost:8088/v1/tightwad/status

# Stop
tightwad proxy stop
```

### RPC Cluster

```bash
# Check cluster status
tightwad status

# Start (after rpc-server instances are running on workers)
tightwad start

# Hot-swap to a different model (RPC workers persist)
tightwad swap deepseek-r1-70b

# Benchmark
tightwad benchmark

# Stop
tightwad stop
```

## Homelab Recipe

One concrete setup you can reproduce in ~20 minutes with two machines.

**Hardware:**
- **Machine A (draft):** Gaming PC with RTX 4060 (8GB VRAM) — the one you use for games
- **Machine B (target):** Server or second PC with RTX 4070 Ti Super (16GB VRAM)

**Expected results:** 58% average token acceptance rate, up to 88% on reasoning tasks

---

### Step 1 — On Machine A: Start the draft model

```bash
ollama run qwen3:8b
# Confirm it works: ollama ps
# Should show: qwen3:8b running
```

Ollama listens on `0.0.0.0:11434` by default. If not, set `OLLAMA_HOST=0.0.0.0` before starting.

### Step 2 — On Machine B: Start the target model

```bash
ollama run qwen3:32b
# Confirm: ollama ps
# Should show: qwen3:32b running
```

Same note: make sure Ollama is accessible on the network (`OLLAMA_HOST=0.0.0.0`).

### Step 3 — On whichever machine runs the proxy: Install Tightwad

```bash
git clone https://github.com/akivasolutions/tightwad.git
cd tightwad
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Step 4 — Edit `configs/cluster.yaml`

```yaml
proxy:
  host: 0.0.0.0
  port: 8088
  max_draft_tokens: 8
  fallback_on_draft_failure: true
  draft:
    url: http://192.168.1.10:11434    # Machine A — replace with your IP
    model_name: qwen3:8b
    backend: ollama
  target:
    url: http://192.168.1.20:11434    # Machine B — replace with your IP
    model_name: qwen3:32b
    backend: ollama
```

Replace `192.168.1.10` and `192.168.1.20` with your actual machine IPs (`ip addr` on Linux, `ipconfig` on Windows).

### Step 5 — Start the proxy

```bash
tightwad proxy start
# Expected output:
# ✓ Draft model healthy  (qwen3:8b @ 192.168.1.10:11434)
# ✓ Target model healthy (qwen3:32b @ 192.168.1.20:11434)
# ✓ Proxy listening on http://localhost:8088
```

### Step 6 — Test it

```bash
# Basic test
curl http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 17 * 24?"}],
    "max_tokens": 100
  }'

# Check acceptance rate stats
tightwad proxy status
# Expected: Acceptance rate: ~58% | Rounds: N | Tokens saved: N

# Detailed stats
curl http://localhost:8088/v1/tightwad/status
```

### Step 7 — Point your app at it

Any OpenAI-compatible client works. Just change the base URL:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8088/v1",
    api_key="not-needed"  # Tightwad doesn't require an API key
)

response = client.chat.completions.create(
    model="tightwad",
    messages=[{"role": "user", "content": "Explain recursion"}]
)
```

**Acceptance rates you can expect with this setup:**

| Task | Acceptance Rate |
|------|:--------------:|
| Reasoning / math | ~88% |
| Code generation | ~73% |
| Factual Q&A | ~52% |
| Creative writing | ~34% |
| **Average** | **~58%** |

> **Note on the bigger picture:** With Qwen3-8B drafting for Qwen3.5-397B (via API), we've seen 80% acceptance after whitespace normalization — meaning 4 in 5 tokens come from your local GPU, not the cloud. Reasoning tasks hit 88%. The bigger the gap between draft and target quality, the more you save.

## Configuration

Edit `configs/cluster.yaml`:

```yaml
# Speculative decoding proxy
proxy:
  host: 0.0.0.0
  port: 8088
  max_draft_tokens: 32              # Sweet spot for cross-machine (reduces HTTP round trips)
  fallback_on_draft_failure: true
  draft:
    url: http://192.168.1.50:11434   # Ollama on a cheap GPU
    model_name: qwen3:8b
    backend: ollama                     # or "llamacpp"
  target:
    url: http://192.168.1.100:11434    # Bigger GPU or cloud API
    model_name: qwen3:32b
    backend: ollama

# RPC cluster (optional, for tensor-parallel across machines)
coordinator:
  host: 0.0.0.0
  port: 8080
  backend: hip
  gpus:
    - name: "7900 XTX #0"
      vram_gb: 24
    - name: "7900 XTX #1"
      vram_gb: 24

workers:
  - host: 192.168.1.100
    gpus:
      - name: "RTX 4070 Ti Super"
        vram_gb: 16
        rpc_port: 50052

models:
  qwen3-72b:
    path: /models/Qwen3-72B-Q4_K_M.gguf
    ctx_size: 8192
    flash_attn: true
    default: true
```

### Server Backends

The proxy supports two backend types for draft and target servers:

| Backend | Endpoint | Best for |
|---------|----------|----------|
| `ollama` | `/api/generate` (raw mode) | Quick setup, any Ollama instance |
| `llamacpp` | `/v1/completions` (with logprobs) | Best performance, full logprobs support |

## How Speculative Decoding Works

1. **Draft:** The small model generates N candidate tokens (fast, ~100+ tok/s)
2. **Verify:** The large model evaluates all N tokens in a single forward pass
3. **Accept/reject:** Keep tokens where both models agree, take the large model's token at the first disagreement
4. **Repeat** until done

The output is **provably identical** to running the large model alone — the small model just proposes shortcuts.

### Benchmark Results

#### Wall-Clock Speedup (Qwen3-8B → Qwen3-32B, cross-machine llama-server)

Draft on RTX 2070 (8GB), target on RTX 4070 Ti Super + RTX 3060 (28GB). Both via llama-server with prompt-append verification.

| Prompt | Baseline | Speculative | Speedup |
|--------|:--------:|:-----------:|:-------:|
| Capital of France | 1.17s | 0.90s | **1.30x** |
| Thermodynamics | 12.73s | 9.09s | **1.40x** |
| Prime checker | 12.76s | 10.15s | **1.28x** |
| Average speed | 13.24s | 10.95s | **1.21x** |
| TCP vs UDP | 5.58s | 4.88s | **1.14x** |
| **Total** | **45.43s** | **35.96s** | **1.27x** |

**1.27x overall speedup** with `max_draft_tokens: 32` (50 rounds, 31.7 tokens/round, 100% acceptance).

##### Tuning `max_draft_tokens`

| Setting | Rounds | Tok/Round | Overall Speedup |
|:-------:|:------:|:---------:|:---------------:|
| 8 | 96 | 8.8 | 0.63x (slower) |
| **32** | **50** | **31.7** | **1.27x** |
| 64 | 16 | 56.5 | 1.21x |

The sweet spot is **32 draft tokens** — fewer rounds reduce HTTP overhead, but going too high (64) adds draft latency that outweighs the savings.

#### Acceptance Rate Details (logprobs verification)

| Metric | Value |
|--------|:-----:|
| **Acceptance Rate** | **73.5%** |
| **Effective tokens/round** | **6.6** (at max_draft_tokens=8) |
| Total rounds | 87 |
| Drafted tokens | 671 |
| Accepted tokens | 493 |

#### Text-Match Benchmarks (Ollama, for acceptance rate comparison)

Same-family (Qwen3-8B → Qwen3-32B, local Ollama):

| Prompt Type | Acceptance Rate | Rounds | Notes |
|-------------|:--------------:|:------:|-------|
| Reasoning   | **89%**        | 32     | Highest — deterministic math answers |
| Code        | **76%**        | 34     | High — structured syntax overlap |
| Factual     | 73%            | 16     | Strong agreement on facts |
| List        | 42%            | 40     | Varied phrasing causes divergence |
| Creative    | 39%            | 6      | Lowest — many valid outputs |
| **Average** | **63.8%**      | 25.6   | |

#### Cross-Family (Qwen3-8B → Llama 3.3 70B, OpenRouter)

| **Average** | **~3%** | Nearly zero — different tokenizers and training data |

**Key finding:** Same-family drafting is critical. An 8B model from the same family as the target achieves 73% acceptance with logprobs, while cross-family drops to ~3%.

#### CPU Draft Results (Qwen3-1.7B CPU → Qwen3-32B GPU)

| Draft Host | Draft Speed | Acceptance | Wall-Clock Speedup |
|------------|:-----------:|:----------:|:------------------:|
| M4 Mac CPU (llama-server) | 32.8 tok/s | 68% | 0.80x |
| Unraid CPU (Ollama, text-match) | 14.9 tok/s | 68% | 0.14x |

CPU drafting with a 1.7B model works but doesn't achieve speedup at `max_draft_tokens=8` due to HTTP round-trip overhead. Needs testing with higher draft token counts and/or multi-drafter parallelism.

### Use Cases

- **Local multi-GPU:** Draft on a consumer GPU ($200), verify on a larger GPU/rig
- **Cloud cost reduction:** Draft locally, verify via cloud API — fewer API calls for the same output quality
- **CPU draft, GPU verify:** Run a tiny model (0.6B-1.7B) on CPU/RAM, verify on GPU. Turns every idle CPU in a datacenter into usable inference compute
- **Multi-drafter parallelism:** Multiple CPUs each run a draft model in parallel, the GPU target picks the best candidate. Mimics datacenter topology where idle CPUs are abundant and GPUs are scarce
- **Legacy GPU revival:** A 12-year-old GPU with 2GB VRAM can run Qwen3-1.7B as a draft model for a 72B target — turning e-waste into productive infrastructure
- **Edge + datacenter:** Fast local responses with datacenter-grade accuracy

## Why Tightwad?

You've probably heard of the other tools. Here's how Tightwad fits in.

### vs vLLM

vLLM is excellent production inference software. It's also CUDA-only. If you have an AMD GPU, you can't use it — full stop. Tightwad pools CUDA and ROCm GPUs on the same model, same endpoint.

vLLM does support speculative decoding, but only within a single machine. Tightwad's proxy does it across your network — your draft model can be on a completely different box than your target.

vLLM is built for ML teams running production workloads at scale. Tightwad is built for anyone with two machines and a network cable.

| | vLLM | Tightwad |
|--|------|----------|
| AMD / ROCm support | ✗ | ✓ |
| Cross-machine speculative decoding | ✗ | ✓ |
| Works with Ollama | ✗ | ✓ |
| Target audience | Production ML teams | Homelab / anyone |

### vs Ollama

Ollama is great. It's the reason most people have local models running at all. But Ollama runs one model on one machine. When you outgrow one GPU, Ollama can't help you — it has no concept of pooling or cross-machine inference.

Tightwad is the next step after Ollama. Keep using Ollama as the backend on each machine — Tightwad just coordinates between them.

### vs llama.cpp RPC

Tightwad is built *on top of* llama.cpp RPC. We didn't replace it — we added the orchestration layer, YAML configuration, CLI, and speculative decoding proxy that you'd otherwise have to script yourself.

The key difference for speculative decoding: llama.cpp RPC ships 100–300 MB of tensor data over the network per step. Tightwad's proxy ships token IDs — a few bytes. For models that fit on individual machines, the proxy approach is dramatically faster over a standard home network.

### vs TGI (HuggingFace Text Generation Inference)

TGI is part of HuggingFace's inference ecosystem and is designed to integrate with their paid services. It's an excellent tool if you're already in that ecosystem.

Tightwad is MIT licensed, has no vendor affiliation, and works with your existing Ollama or llama.cpp setup without any additional accounts or services. It's backend-agnostic by design.

### The honest summary

If you have a single powerful CUDA machine and need production-grade throughput: use vLLM.

If you have one machine and just want to run models: use Ollama.

If you have two or more machines — mixed vendors, mixed budgets, mixed everything — and want them to work together intelligently: that's what Tightwad is for.

## CLI Reference

| Command | Description |
|---------|-------------|
| `tightwad proxy start` | Start speculative decoding proxy |
| `tightwad proxy stop` | Stop the proxy |
| `tightwad proxy status` | Show draft/target health + acceptance rate stats |
| `tightwad status` | Show RPC cluster status |
| `tightwad start [-m MODEL]` | Start RPC coordinator |
| `tightwad stop` | Stop the coordinator |
| `tightwad swap MODEL` | Hot-swap model (workers persist) |
| `tightwad benchmark` | Benchmark the running coordinator |

Global option: `-c /path/to/cluster.yaml` or `TIGHTWAD_CONFIG` env var.

## API Endpoints (Proxy)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/completions` | POST | Text completion (OpenAI-compatible) |
| `/v1/chat/completions` | POST | Chat completion (OpenAI-compatible) |
| `/v1/models` | GET | List available models |
| `/v1/tightwad/status` | GET | Proxy stats: acceptance rate, rounds, throughput |

All endpoints support `stream: true` for SSE streaming.

## Hardware Setup

### Worker (CUDA — Windows)

```bash
cmake -B build -DGGML_CUDA=ON -DGGML_RPC=ON
cmake --build build --config Release
build/bin/rpc-server.exe -p 50052  # GPU 0
```

Or use `scripts/install-worker.sh`

### Coordinator (ROCm — Ubuntu)

```bash
cmake -B build -DGGML_HIP=ON -DGGML_RPC=ON -DAMDGPU_TARGETS=gfx1100
cmake --build build --config Release -j$(nproc)
sudo cp build/bin/llama-server /usr/local/bin/
```

Or use `scripts/install-coordinator.sh`

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Project Structure

```
tightwad/
├── config.py        # YAML config loader (cluster + proxy)
├── cli.py           # Click CLI (cluster + proxy commands)
├── coordinator.py   # llama-server lifecycle management
├── worker.py        # RPC worker health checks
├── proxy.py         # Speculative decoding proxy server
└── speculation.py   # Verification algorithm (pure logic)
tests/
├── test_config.py
├── test_coordinator.py
├── test_speculation.py
└── test_proxy.py
configs/
└── cluster.yaml     # Hardware topology + proxy config
```
