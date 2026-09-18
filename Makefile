# Convenience wrapper. Both targets are plain docker build commands — read the
# README if you prefer to run them by hand, or want to narrow CUDA_ARCHS.

BASE_IMAGE      ?= prismml-llama-server:cuda13
SERVICE_IMAGE   ?= orcabonsai-27b-serving:latest
# CUDA toolkit the server is compiled against. It must be a version your NVIDIA driver
# supports: CUDA 13.x needs a 580-series driver or newer. On an older driver use
# 12.8.0 (needs 570+) or 12.4.0 (needs 550+).
CUDA_VERSION    ?= 13.1.1
# 86 = Ampere (RTX 30xx/A10), 89 = Ada (RTX 40xx/L4), 90 = Hopper (H100).
# Single arch builds much faster; add 75 (Turing), 80 (A100), 100/120 (Blackwell) as needed.
CUDA_ARCHS      ?= 86;89;90

.PHONY: base image up down logs verify clean

## Compile the PrismML llama.cpp fork from source (slow, run once).
base:
	docker build -f base/Dockerfile -t $(BASE_IMAGE) \
	  --build-arg CUDA_VERSION=$(CUDA_VERSION) \
	  --build-arg CUDA_ARCHS='$(CUDA_ARCHS)' base/

## Layer the upstream LoRA adapter onto the base image (fast).
image:
	docker build -t $(SERVICE_IMAGE) .

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

## A/B check that the ablation is live.
verify:
	./scripts/verify-ablation.sh

## Remove the images built here.
clean:
	-docker rmi $(SERVICE_IMAGE) $(BASE_IMAGE)
