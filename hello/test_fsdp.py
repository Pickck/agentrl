import os
import torch
import torch.distributed as dist
import torch_npu

from transformers import AutoModelForCausalLM
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

dist.init_process_group("hccl")

rank = dist.get_rank()
torch.npu.set_device(rank)

model = AutoModelForCausalLM.from_pretrained(
    "/mnt/workspace/models/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.bfloat16,
)

model = model.npu()

fsdp_model = FSDP(model)

print(f"rank {rank} fsdp ok")

dist.barrier()