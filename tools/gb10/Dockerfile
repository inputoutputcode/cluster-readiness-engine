# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.01-py3
FROM ${BASE_IMAGE}

ARG NCCL_TESTS_REF=v2.16.9
ARG MPI_HOME=/usr/local/mpi
ARG NCCL_HOME=/usr
RUN apt-get update && apt-get install -y --no-install-recommends \
      git make g++ openssh-server ibverbs-utils libibverbs-dev iproute2 \
    && rm -rf /var/lib/apt/lists/*
RUN test -x "${MPI_HOME}/bin/mpicxx" \
    && git clone --depth 1 --branch "${NCCL_TESTS_REF}" \
      https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests \
    && make -C /opt/nccl-tests -j4 MPI=1 NAME_SUFFIX=_mpi \
      MPI_HOME="${MPI_HOME}" NCCL_HOME="${NCCL_HOME}" CUDA_HOME=/usr/local/cuda \
      NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121" \
    && install -m755 /opt/nccl-tests/build/all_reduce_perf_mpi /opt/nccl-tests/build/alltoall_perf_mpi /usr/local/bin/ \
    && /usr/local/mpi/bin/mpirun --version

COPY gb10_c2c.cu /tmp/gb10_c2c.cu
COPY gb10_llama32_1b.py /usr/local/bin/gb10-llama32-1b.py
RUN nvcc -O3 -std=c++17 -gencode=arch=compute_121,code=sm_121 \
      /tmp/gb10_c2c.cu -o /usr/local/bin/gb10-c2c \
    && chmod 0755 /usr/local/bin/gb10-c2c /usr/local/bin/gb10-llama32-1b.py \
    && python3 -c 'import torch, transformers'
