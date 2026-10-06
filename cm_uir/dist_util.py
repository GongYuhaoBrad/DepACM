import torch as th
import torch.distributed as dist
import os
GPUS_PER_NODE = 3

SETUP_RETRY_COUNT = 3


def setup_dist():
    if th.cuda.is_available():
            dist.init_process_group(
                backend='nccl',
                init_method='tcp://127.0.0.1:12346',
                world_size=1,
                rank=0
            )
            print("[INFO] Single-GPU DDP: initialized 1-process process group (nccl)")
    else:
        dist.init_process_group(
            backend='gloo',
            init_method='tcp://127.0.0.1:12346',
            world_size=1,
            rank=0
        )
def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        pa=p.clone()
        
def load_state_dict(path, **kwargs):
   
    return th.load( path,**kwargs)

