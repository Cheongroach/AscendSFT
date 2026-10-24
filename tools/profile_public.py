#!/usr/bin/env python3
"""公开随机张量采集示例；无需任何历史训练代码或数据。"""

import argparse
from pathlib import Path

import torch
import torch_npu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("profile-output"))
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()
    if args.steps < 5:
        parser.error("At least five steps are needed for wait=2, warmup=1, active=2")
    torch.manual_seed(42)
    torch.npu.set_device(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 1024), torch.nn.GELU(), torch.nn.Linear(1024, 256)
    ).npu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    inputs = torch.randn(16, 256, device="npu")
    target = torch.randn_like(inputs)
    profiler = torch_npu.profiler
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(wait=2, warmup=1, active=2, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(str(args.output)),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as recording:
        for _ in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(inputs) - target).square().mean()
            loss.backward()
            optimizer.step()
            recording.step()
    torch.npu.synchronize()
    print("Profiling finished. Inspect the generated local trace and CSV files.")


if __name__ == "__main__":
    main()
