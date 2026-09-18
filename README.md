# OrcaBonsai Ternary Bonsai 2 27B Uncensored — GPU serving recipe

An OpenAI-compatible server for the **OrcaBonsai** runtime ablation of PrismML's
**Ternary Bonsai 2 27B**, using the published ternary GGUF with the upstream rank-1 LoRA
adapter. No weights are modified and nothing is re-quantized: the refusal-direction
projection is applied at inference time, so the ternary pack stays byte-identical and
ablation strength stays a runtime dial.

- Upstream runtime / adapter: [Continuum-AI-Corp/OrcaBonsai-27B-Uncensored](https://github.com/Continuum-AI-Corp/OrcaBonsai-27B-Uncensored) (Apache-2.0)
- Model: [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) (Apache-2.0)
- Server: [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) fork, branch `prism` (MIT)

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
- ~15 GB of disk for the build, plus ~7 GB for the model
- Patience for the first build: a CUDA llama.cpp compile takes roughly 15–40 minutes
  depending on core count and how many GPU architectures you target

## Quickstart

**1. Get the model** (~6.7 GiB, not redistributed here):

```bash
mkdir -p models
huggingface-cli download prism-ml/Ternary-Bonsai-2-27B-gguf \
  Ternary-Bonsai-2-27B-PQ2_0.gguf --local-dir models
```

`PTQ1_0.gguf` also works (smaller, group-64); `PQ2_0` is the fork's preferred format on CUDA.

**2. Build the two images.** `make base` compiles the fork; `make image` layers the adapter on top.

```bash
make base     # -> prismml-llama-server:cuda13   (long: full CUDA build)
make image    # -> orcabonsai-27b-serving:latest (seconds)
```

Or without `make`, targeting only your own GPU for a much faster compile:

```bash
docker build -f base/Dockerfile -t prismml-llama-server:cuda13 \
  --build-arg CUDA_ARCHS=89 base/          # 89 = Ada, 86 = Ampere, 90 = Hopper
docker build -t orcabonsai-27b-serving:latest .
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

## Files

```
Dockerfile             serving image: base + the upstream LoRA adapter (sha256-pinned)
base/Dockerfile        PrismML llama.cpp fork built from source with CUDA
docker-compose.yml     the service
scripts/verify-ablation.sh   A/B check that the ablation is live
.env.example           all knobs with defaults
```

No binaries are distributed in this repository. `base/Dockerfile` compiles the server from
a pinned upstream tag, and the adapter is fetched from upstream at image-build time and
verified against the sha256 the upstream repo publishes. Model weights are supplied by you.

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
