# syntax=docker/dockerfile:1
#
# Serving image: OrcaBonsai Ternary Bonsai 2 27B Uncensored.
#
# The ablation is a rank-1 LoRA adapter published by the OrcaRouter team. This image
# is the runtime that applies it: it layers the adapter onto the PrismML llama.cpp
# server (built by base/Dockerfile) and nothing else.
#
# WHY NOT THE UPSTREAM docker/Dockerfile
#   The OrcaBonsai repository ships its own docker/Dockerfile, and by its own
#   description it is CPU-only: it runs the MLX pack, and MLX has no CUDA kernel for
#   the pack's quantized matmul ("QuantizedMatmul has no CUDA implementation"), so a
#   forward pass takes minutes. It also loads MLX packs, not GGUF. It cannot serve
#   the published GGUF on a GPU. The adapter is the GPU path.
#
# NO BINARIES ARE DISTRIBUTED HERE
#   The only artifact fetched at build time is the adapter, straight from upstream,
#   pinned by the sha256 the upstream repository publishes. If upstream ever changes
#   the file, this build fails instead of silently producing a different runtime.

ARG BASE_IMAGE=prismml-llama-server:cuda13
FROM ${BASE_IMAGE}

ARG ORCABONSAI_REF=main
# sha256 published in the OrcaBonsai README for gguf/bonsai-abliterate-lora.gguf.
ARG LORA_SHA256=f1669534803d340a496015f5c45125f3437b4d13ec764f40e34488ce83967f42

WORKDIR /app

RUN mkdir -p /app/gguf \
 && curl -fsSL -o /app/gguf/bonsai-abliterate-lora.gguf \
      "https://github.com/Continuum-AI-Corp/OrcaBonsai-27B-Uncensored/raw/${ORCABONSAI_REF}/gguf/bonsai-abliterate-lora.gguf" \
 && echo "${LORA_SHA256}  /app/gguf/bonsai-abliterate-lora.gguf" | sha256sum -c - \
 && ls -l /app/gguf/bonsai-abliterate-lora.gguf

ENTRYPOINT ["/usr/local/bin/llama-server"]
CMD ["--help"]
