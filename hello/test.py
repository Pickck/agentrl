import os
import torch
import torch.distributed as dist
import torch_npu

dist.init_process_group("hccl")

local_rank = int(os.environ["LOCAL_RANK"])

torch.npu.set_device(local_rank)

print(
    "rank",
    dist.get_rank(),
    "local_rank",
    local_rank,
    "device",
    torch.npu.current_device()
)

dist.barrier()

print("ok")