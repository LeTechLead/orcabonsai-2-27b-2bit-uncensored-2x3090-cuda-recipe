# Convenience wrapper. `make up` runs the published image (nothing to build);
# `make base && make image` compiles from source instead. Both are plain docker
# commands — read the README if you prefer to run them by hand.

PUBLISHED_IMAGE ?= ghcr.io/letechlead/orcabonsai-27b-serving:latest
BASE_IMAGE      ?= prismml-llama-server:cuda13
LOCAL_IMAGE     ?= orcabonsai-27b-serving:local
# CUDA toolkit the server is compiled against. It must be a version your NVIDIA driver
# supports: CUDA 13.x needs a 580-series driver or newer. On an older driver use
# 12.8.0 (needs 570+) or 12.4.0 (needs 550+).
CUDA_VERSION    ?= 13.1.1
# 86 = Ampere (RTX 30xx/A10), 89 = Ada (RTX 40xx/L4), 90 = Hopper (H100).
# Single arch builds much faster; add 75 (Turing), 80 (A100), 100/120 (Blackwell) as needed.
CUDA_ARCHS      ?= 86;89;90

.PHONY: pull base image up up-build down logs verify clean

## Fetch the published image (the default path; nothing is compiled).
pull:
	docker pull $(PUBLISHED_IMAGE)

## Compile the PrismML llama.cpp fork from source (slow, run once).
base:
	docker build -f base/Dockerfile -t $(BASE_IMAGE) \
	  --build-arg CUDA_VERSION=$(CUDA_VERSION) \
	  --build-arg CUDA_ARCHS='$(CUDA_ARCHS)' base/

## Layer the upstream LoRA adapter onto the base image (fast).
image:
	docker build -t $(LOCAL_IMAGE) .

up:
	docker compose up -d

## Build from source and run that, instead of the published image.
up-build: base image
	docker compose -f docker-compose.yml -f docker-compose.build.yml up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

## A/B check that the ablation is live.
verify:
	./scripts/verify-ablation.sh

## Remove the images built locally (leaves the published image alone).
clean:
	-docker rmi $(LOCAL_IMAGE) $(BASE_IMAGE)
