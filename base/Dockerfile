# syntax=docker/dockerfile:1
#
# Base image: the PrismML fork of llama.cpp, built from source with CUDA.
#
# WHY THIS FORK IS MANDATORY
#   The ternary Bonsai GGUF files use the fork's own low-bit container formats.
#   *-PQ2_0.gguf is group-128 ternary stored as ggml type id 142, and stock
#   llama.cpp rejects it at load time:
#       gguf_init_from_reader: tensor 'output.weight' has invalid ggml type 142.
#       should be in [0, 43)
#   The fork's own README documents PQ2_0 (id 142) as the format to prefer on
#   CUDA/HIP/Metal/CPU for these models, and warns against mixing its ggml-*
#   libraries with a stock llama.cpp build.
#
# WHY FROM SOURCE
#   No prebuilt binaries are redistributed here. The image contains only what this
#   build produces. Pin PRISM_REF for reproducibility.
#
# CUDA_ARCHS: set to your GPU(s). Common values:
#   75 = Turing (T4/2080)   80 = A100   86 = Ampere consumer (RTX 30xx, A10)
#   89 = Ada (RTX 40xx, L4) 90 = Hopper (H100)   100/120 = Blackwell
#   Builds are much faster with a single arch, e.g. --build-arg CUDA_ARCHS=89

ARG CUDA_VERSION=13.1.1
ARG UBUNTU_VERSION=24.04

# ─────────────────────────────── build stage ───────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

# Fork tag, not a branch: reproducible builds.
ARG PRISM_REF=prism-b10685-7dffb15
ARG CUDA_ARCHS=86;89;90

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The driver stub is what a BUILD host links against: the real libcuda.so.1 exists only on
# the machine that runs the container, where the NVIDIA container runtime mounts it in.
# GGML_CUDA references driver API symbols (cuMemCreate, cuMemAddressReserve, cuDeviceGet*),
# so the linker needs the toolkit's stub. Without it the build dies with
#   libcuda.so.1, needed by libggml-cuda.so, not found
#   undefined reference to `cuMemCreate'
# Linking against the stub records the usual libcuda.so.1 dependency, which resolves to the
# real driver at runtime.
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so

WORKDIR /src
RUN git clone --depth 1 --branch "${PRISM_REF}" \
        https://github.com/PrismML-Eng/llama.cpp.git . \
    && git rev-parse HEAD > /src/PRISM_COMMIT

# Tests and examples are off: we ship a server, and building them is both slower and a
# source of link/install failures. GGML_NATIVE=OFF keeps the build off the build host's CPU.
RUN cmake -B build -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DGGML_BUILD_SERVER=ON \
        -DLLAMA_CURL=OFF -DGGML_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DGGML_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_BUILD_EXAMPLES=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
    && cmake --build build --config Release --target llama-server -j"$(nproc)"

# Artifacts are copied directly rather than with `cmake --install`: installing would also
# try to install binaries that were deliberately not built, and fail.
RUN mkdir -p /install/bin /install/lib \
    && cp build/bin/llama-server /install/bin/ \
    && cp -a build/bin/*.so* /install/lib/ \
    && chmod +x /install/bin/llama-server

# ────────────────────────────── runtime stage ──────────────────────────────
# The CUDA *runtime* image supplies libcudart.so.13 / libcublas.so.13 /
# libcublasLt.so.13 — exactly what the fork's binaries link against.
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY --from=builder /src/PRISM_COMMIT /etc/prism-commit

RUN ldconfig && test -x /usr/local/bin/llama-server

ENV PATH="/usr/local/bin:${PATH}"
ENV LD_LIBRARY_PATH=/usr/local/lib

EXPOSE 8080
ENTRYPOINT ["llama-server"]
CMD ["--help"]
