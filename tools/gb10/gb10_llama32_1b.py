#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Random-initialized Llama 3.2 1B distributed training smoke workload."""

import argparse
from datetime import datetime
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import LlamaConfig, LlamaForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sequence-length", type=int, default=1024)
    return parser.parse_args()


def stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()

    config = LlamaConfig(
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=131072,
        rope_theta=500000.0,
        torch_dtype="bfloat16",
        use_cache=False,
    )
    model = LlamaForCausalLM(config).to(device=local_rank, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    parameter_count = sum(p.numel() for p in model.parameters())

    if rank == 0:
        print(
            f"using world size: {world}, data-parallel size: {world}, "
            "context-parallel size: 1, tensor-model-parallel size: 1, "
            "pipeline-model-parallel size: 1",
            flush=True,
        )

    generator = torch.Generator(device=local_rank).manual_seed(1234 + rank)
    for step in range(1, args.steps + 1):
        tokens = torch.randint(
            0, config.vocab_size, (1, args.sequence_length + 1),
            device=local_rank, generator=generator,
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(input_ids=tokens[:, :-1], labels=tokens[:, 1:])
            loss = output.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if rank == world - 1:
            tflops = 6.0 * parameter_count * args.sequence_length / (elapsed_ms / 1000.0) / 1e12
            print(
                f"[{stamp()}] iteration {step:8d}/ {args.steps:8d} | "
                f"consumed samples: {step * world} | "
                f"elapsed time per iteration (ms): {elapsed_ms:.1f} | "
                f"throughput per GPU (TFLOP/s/GPU): {tflops:.1f} | "
                f"lm loss: {loss.detach().float().item():.6E}",
                flush=True,
            )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
