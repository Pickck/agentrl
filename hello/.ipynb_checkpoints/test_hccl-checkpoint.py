import torch
import torch_npu
import torch.distributed as dist
import os

def main():
    dist.init_process_group(backend="hccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.npu.set_device(rank)

    x = torch.ones(10).npu() * (rank + 1)
    dist.all_reduce(x)

    print(f"rank {rank}, result: {x}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()