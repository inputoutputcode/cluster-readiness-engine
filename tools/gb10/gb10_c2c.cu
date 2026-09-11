// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_OK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(2); } } while (0)

__global__ void consume(const uint8_t* input, uint8_t* output, size_t bytes) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < bytes) output[i] = input[i] ^ 0x5a;
}

__global__ void produce(uint8_t* output, size_t bytes) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < bytes) output[i] = static_cast<uint8_t>((i * 17 + 23) & 0xff);
}

static size_t parse_size(const std::string& text) {
  char* end = nullptr;
  unsigned long long value = std::strtoull(text.c_str(), &end, 10);
  if (!end || end == text.c_str()) return 0;
  if (*end == 'K') value *= 1024ULL;
  else if (*end == 'M') value *= 1024ULL * 1024ULL;
  else if (*end == 'G') value *= 1024ULL * 1024ULL * 1024ULL;
  else if (*end != '\0') return 0;
  return static_cast<size_t>(value);
}

struct SharedBuffer {
  uint8_t* host = nullptr;
  uint8_t* device = nullptr;
  std::string type;
};

static SharedBuffer allocate_buffer(size_t bytes, bool managed) {
  SharedBuffer b;
  if (managed) {
    CUDA_OK(cudaMallocManaged(reinterpret_cast<void**>(&b.host), bytes));
    b.device = b.host;
    b.type = "managed";
  } else {
    CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&b.host), bytes, cudaHostAllocMapped));
    CUDA_OK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&b.device), b.host, 0));
    b.type = "pinned";
  }
  return b;
}

static void release_buffer(SharedBuffer& b) {
  if (b.type == "managed") CUDA_OK(cudaFree(b.host));
  else CUDA_OK(cudaFreeHost(b.host));
}

static bool verify_cpu_to_gpu(const std::vector<uint8_t>& got) {
  for (size_t i = 0; i < got.size(); ++i)
    if (got[i] != (static_cast<uint8_t>((i * 13 + 7) & 0xff) ^ 0x5a)) return false;
  return true;
}

static bool verify_gpu_to_cpu(const uint8_t* got, size_t bytes) {
  for (size_t i = 0; i < bytes; ++i)
    if (got[i] != static_cast<uint8_t>((i * 17 + 23) & 0xff)) return false;
  return true;
}

static bool run(size_t bytes, int samples, bool managed) {
  SharedBuffer shared = allocate_buffer(bytes, managed);
  uint8_t* device_out = nullptr;
  CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&device_out), bytes));
  std::vector<uint8_t> host_out(bytes);
  const int threads = 256;
  const int blocks = static_cast<int>((bytes + threads - 1) / threads);

  double cpu_to_gpu_us = 0.0;
  bool cpu_to_gpu_ok = true;
  for (int sample = 0; sample < samples; ++sample) {
    for (size_t i = 0; i < bytes; ++i) shared.host[i] = static_cast<uint8_t>((i * 13 + 7) & 0xff);
    CUDA_OK(cudaDeviceSynchronize());
    auto start = std::chrono::steady_clock::now();
    consume<<<blocks, threads>>>(shared.device, device_out, bytes);
    CUDA_OK(cudaDeviceSynchronize());
    cpu_to_gpu_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    CUDA_OK(cudaMemcpy(host_out.data(), device_out, bytes, cudaMemcpyDeviceToHost));
    cpu_to_gpu_ok = cpu_to_gpu_ok && verify_cpu_to_gpu(host_out);
  }

  double gpu_to_cpu_us = 0.0;
  bool gpu_to_cpu_ok = true;
  for (int sample = 0; sample < samples; ++sample) {
    produce<<<blocks, threads>>>(shared.device, bytes);
    CUDA_OK(cudaDeviceSynchronize());
    auto start = std::chrono::steady_clock::now();
    gpu_to_cpu_ok = gpu_to_cpu_ok && verify_gpu_to_cpu(shared.host, bytes);
    gpu_to_cpu_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
  }

  const double cpu_avg = cpu_to_gpu_us / samples;
  const double gpu_avg = gpu_to_cpu_us / samples;
  std::printf("NVCRE_C2C_RESULT direction=cpuToGPU memory=%s size_bytes=%zu bandwidth_gbps=%.2f latency_us=%.2f samples=%d verified=%s\n",
      shared.type.c_str(), bytes, bytes / cpu_avg / 1000.0, cpu_avg, samples, cpu_to_gpu_ok ? "true" : "false");
  std::printf("NVCRE_C2C_RESULT direction=gpuToCPU memory=%s size_bytes=%zu bandwidth_gbps=%.2f latency_us=%.2f samples=%d verified=%s\n",
      shared.type.c_str(), bytes, bytes / gpu_avg / 1000.0, gpu_avg, samples, gpu_to_cpu_ok ? "true" : "false");
  std::fflush(stdout);

  CUDA_OK(cudaFree(device_out));
  release_buffer(shared);
  return cpu_to_gpu_ok && gpu_to_cpu_ok;
}

int main(int argc, char** argv) {
  std::string sizes = "4M,64M,1G";
  int samples = 5;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) sizes = argv[++i];
    else if (std::strcmp(argv[i], "--samples") == 0 && i + 1 < argc) samples = std::atoi(argv[++i]);
    else { std::fprintf(stderr, "usage: %s [--sizes 4M,64M,1G] [--samples 5]\n", argv[0]); return 2; }
  }
  if (samples < 1) return 2;

  cudaDeviceProp prop{};
  CUDA_OK(cudaGetDeviceProperties(&prop, 0));
  int managed = 0, pageable = 0, host_map = 0;
  CUDA_OK(cudaDeviceGetAttribute(&managed, cudaDevAttrConcurrentManagedAccess, 0));
  CUDA_OK(cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, 0));
  CUDA_OK(cudaDeviceGetAttribute(&host_map, cudaDevAttrCanMapHostMemory, 0));
  std::printf("NVCRE_C2C_CAPABILITY device=%s concurrent_managed=%d pageable_memory=%d mapped_host=%d\n",
      prop.name, managed, pageable, host_map);
  if (!managed || !host_map) {
    std::fprintf(stderr, "GB10 C2C requires concurrent managed and mapped host memory\n");
    return 1;
  }

  bool ok = true;
  size_t start = 0;
  while (start < sizes.size()) {
    size_t comma = sizes.find(',', start);
    size_t bytes = parse_size(sizes.substr(start, comma - start));
    if (bytes == 0) return 2;
    ok = run(bytes, samples, true) && ok;
    ok = run(bytes, samples, false) && ok;
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return ok ? 0 : 1;
}
