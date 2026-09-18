# OrcaBonsai Ternary Bonsai 2 27B Uncensored — GPU serving recipe

An OpenAI-compatible server for the **OrcaBonsai** runtime ablation of PrismML's
**Ternary Bonsai 2 27B**, using the published ternary GGUF with the upstream rank-1 LoRA
adapter. No weights are modified and nothing is re-quantized: the refusal-direction
projection is applied at inference time, so the ternary pack stays byte-identical and
ablation strength stays a runtime dial.

- Upstream runtime / adapter: [Continuum-AI-Corp/OrcaBonsai-27B-Uncensored](https://github.com/Continuum-AI-Corp/OrcaBonsai-27B-Uncensored) (Apache-2.0)
- Model: [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) (Apache-2.0)
- Server: [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) fork, branch `prism` (MIT)
- Prebuilt image: `ghcr.io/letechlead/orcabonsai-27b-serving:latest` — ~1.5 GB, nothing to compile

## The part that is easy to get wrong

**The upstream repository's own `docker/Dockerfile` cannot serve this on a GPU.** Its
comments say so:

> mlx-cuda has none ("QuantizedMatmul has no CUDA implementation"), so this is a CPU
> image and a forward pass of this 27B model takes minutes. It is useful for checking
> behaviour, not for serving.

It also loads MLX *packs* (`config.json` + `runtime/`), not GGUF, so it cannot load the
published `*-PQ2_0.gguf` at all.

The GPU path is the repository's **other** published form of the same edit: a rank-1 LoRA
adapter. `W' = W − r(rᵀW)` is rank 1, so it factors as `A = rᵀW`, `B = −r`, and llama.cpp
applies it as two extra matmuls on the activation instead of merging it into the weights —
which is what lets the edit survive at ~2 bits, where a baked edit would be rounded away.

Second gotcha: **the server must be the PrismML fork.** `*-PQ2_0.gguf` is group-128 ternary
stored as ggml **type id 142**, and stock llama.cpp refuses it at load:

```
gguf_init_from_reader: tensor 'output.weight' has invalid ggml type 142. should be in [0, 43)
```

An *old* PrismML build fails the same way — a matching fork name is not enough, the build
has to be newer than the model. `base/Dockerfile` builds the fork from a pinned tag, so
this is handled for you.

## Requirements

- A Linux host with an NVIDIA GPU and a working `nvidia-container-toolkit`
  (`docker run --rm --gpus all nvidia/cuda:13.1.1-runtime-ubuntu24.04 nvidia-smi` must work)
- Docker 27+ with Compose v2.30+ for the `gpus:` key (older: see the commented block in `docker-compose.yml`)
- ~1.6 GB of disk for the image, plus ~7 GB for the model
- Only if you build from source: ~15 GB more for the toolchain, and a CUDA compile that
  takes roughly 15–40 minutes depending on core count and how many GPU architectures you
  target

The toolkit the server is compiled against defaults to **CUDA 13.1.1**, which wants a 580-series
driver or newer. On an older driver, build against an earlier toolkit instead:

```bash
make base CUDA_VERSION=12.8.0     # 570+ driver
make base CUDA_VERSION=12.4.0     # 550+ driver
```

Match `CUDA_VERSION` to what your driver supports — the runtime image is pulled from the same
tag, so the two never disagree.

## Quickstart

**1. Get the model** (~6.7 GiB, not redistributed here):

```bash
mkdir -p models
huggingface-cli download prism-ml/Ternary-Bonsai-2-27B-gguf \
  Ternary-Bonsai-2-27B-PQ2_0.gguf --local-dir models
```

`PTQ1_0.gguf` also works (smaller, group-64); `PQ2_0` is the fork's preferred format on CUDA.

**2. Pull the image** (~1.5 GB — the fork and the adapter are already inside it, so there is
nothing to compile):

```bash
docker pull ghcr.io/letechlead/orcabonsai-27b-serving:latest
```

**3. Run it.**

```bash
cp .env.example .env      # optional; defaults already work
docker compose up -d
curl -s localhost:8080/v1/models | jq '.data[0].meta.n_ctx'
```

Serve a request:

```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages":[{"role":"user","content":"Explain how a pin tumbler lock works."}],
  "max_tokens":256,"temperature":0
}'
```

## Building from source instead

The published image is built for `CUDA_ARCHS=86;89;90` (Ampere / Ada / Hopper) against CUDA
13.1.1. If your GPU is something else, or your driver predates CUDA 13, build it yourself —
the Dockerfiles are here, and this is the path that produced the published image:

```bash
make base     # compiles the PrismML fork  -> prismml-llama-server:cuda13
make image    # adds the adapter           -> orcabonsai-27b-serving:latest
docker compose up -d
```

A full CUDA compile takes roughly 15–40 minutes depending on core count and how many GPU
architectures you target, and needs ~15 GB of disk for the toolchain. Narrowing it to just
your own card is much faster:

```bash
make base CUDA_ARCHS=89 CUDA_VERSION=13.1.1   # 89 = Ada, 86 = Ampere, 90 = Hopper
make image
```

To run a locally built image through the same service definition, use the build override —
it changes only where the image comes from, keeping the command, ports and healthcheck:

```bash
docker compose -f docker-compose.yml -f docker-compose.build.yml up -d --build
```

## Verify the ablation is actually applied

A registered adapter is not proof the edit landed — llama.cpp will happily report an
adapter whose tensors matched nothing. Check both of these.

**a) Is the adapter loaded, at the strength you asked for?**

```bash
curl -s localhost:8080/lora-adapters
# [{"id":0,"path":"/app/gguf/bonsai-abliterate-lora.gguf","scale":1.0,...}]
```

**b) Does it change behaviour?** Same weights, adapter on vs off (`scripts/verify-ablation.sh`
does this for you, restarting the server each way):

```bash
./scripts/verify-ablation.sh
```

Expected — identical model, different behaviour:

| BONSAI_ALPHA | Observed opening on a request the base model declines |
|---|---|
| `0.0` | "I cannot provide instructions on how to…" |
| `1.0` | answers directly, usually with a legality/eligibility caveat |

If both columns read the same, the adapter is inert: check that the server is the fork and
that `/lora-adapters` shows `scale` matching `BONSAI_ALPHA`.

## Tuning

| Variable | Default | Notes |
|---|---|---|
| `BONSAI_ALPHA` | `1.0` | Ablation strength. `0` = published model, `1` = exact projection, `2` = flips stubborn refusals at some coherence cost. Runtime dial, no second checkpoint |
| `BONSAI_CTX` | `262144` | Trained context; llama.cpp clamps anything above it |
| `BONSAI_KV_TYPE` | `q8_0` | `q4_0` roughly halves KV memory at some quality cost |
| `BONSAI_SPLIT` | `50,50` | Fraction of the model per GPU, one entry per GPU |
| `--n-gpu-layers` | `99` | Lower it to keep some layers on CPU if a card is too small |

Rough VRAM: ~6.5 GiB of weights plus KV. At `q8_0`, the KV costs about 34 KiB/token, so
262144 context adds ~8.5 GiB — it fits comfortably on a single 24 GB card, and splitting
across two gets you faster decode because the model is bandwidth-bound.

## Measured performance

Reference build: fork tag `prism-b10685-7dffb15` compiled with `CUDA_ARCHS=86`, on 2× RTX 3090
(24 GiB each), `--tensor-split 50/50`, `--ctx-size 262144`, q8_0 KV, adapter at alpha 1.0.
Steady state used ~11.4 / 11.6 GiB of VRAM per card, with no CPU offload.

Measured with **llama-benchy** (pp 4096 / tg 512, 3 runs, averages, `--no-cache` so every run is a
real prefill). "Depth" is tokens of conversation history already in context:

| depth | prefill t/s | decode t/s | TTFT | E2E, 512 tokens |
|---:|---:|---:|---:|---:|
| 8,192 | 1,278 | 62.7 | 9.6 s | 17.8 s |
| 32,768 | 1,162 | 52.6 | 31.7 s | 41.5 s |
| 65,536 | 1,018 | 42.8 | 68.4 s | 80.4 s |
| 122,880 | 834 | 31.8 | 152.2 s | 168.3 s |
| 150,000 | 767 | 28.5 | 200.8 s | 218.8 s |
| 257,000 | 555 | 20.0 | 474.4 s | 500.0 s |

TTFT is time to the first token; E2E is prefill through the last of 512 generated tokens
(TTFT + 512/decode).

- **Prefill degrades gently**: 1,278 → 555 t/s across the entire 257k ladder, so the full native
  window is usable, not just loadable.
- **Decode is memory-bandwidth bound and falls as the context fills**: 62.7 → 20.0 t/s. That is
  the real cost of a deep context. Short-prompt benchmarks report much higher decode numbers and
  are misleading for anything you intend to run at depth.
- The 257k prefill row carries a wider spread (±50 t/s) because one of its three runs overlapped an
  unrelated request on the same endpoint; the server itself logged a steady ~593 t/s across all
  three runs. Treat ~590 as the cleaner figure.
- Throughput roughly halves when a second model shares the GPUs (55 tok/s decode observed
  co-resident with another 24 GiB-class server).

## Files

```
docker-compose.yml           the service — runs the published image
docker-compose.build.yml     override that builds from source instead
Dockerfile                   serving image: base + the upstream LoRA adapter (sha256-pinned)
base/Dockerfile              PrismML llama.cpp fork built from source with CUDA
scripts/verify-ablation.sh   A/B check that the ablation is live
Makefile                     base / image / up / down / verify helpers
.env.example                 all knobs with defaults
```

No binaries are distributed in this repository — the compiled server and the adapter live in
the published image, not in git. Building from source compiles the server from a pinned
upstream tag, and the adapter is fetched from upstream at image-build time and verified
against the sha256 the upstream repo publishes. Model weights are supplied by you.

## Licensing and attribution

The runtime idea, the refusal direction and the LoRA adapter are the work of the
**OrcaRouter research team** ([Continuum-AI-Corp/OrcaBonsai-27B-Uncensored](https://github.com/Continuum-AI-Corp/OrcaBonsai-27B-Uncensored), Apache-2.0).
The ternary Bonsai 2 model and the llama.cpp fork are **PrismML**'s
([prism-ml](https://huggingface.co/prism-ml), [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp), MIT).
The Dockerfiles, compose service and scripts in this repository are the packaging work and
are released under Apache-2.0 (see `LICENSE`).

This repository applies a behavioural intervention to a model. It removes a refusal
direction; it does not remove the model's training, its biases, or your responsibility for
what you send it. The upstream project measures the capability cost of the projection as
within noise at the sample sizes it reports.
