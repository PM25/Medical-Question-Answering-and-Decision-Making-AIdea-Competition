import torch
import random
import numpy as np


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def get_device(_id=0):
    assert torch.cuda.device_count() > _id
    print("===== [Device Information] =====")

    if _id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(_id)
        _id = torch.cuda.current_device()
        device = torch.device(f"cuda:{_id}")
        print(f"There are {torch.cuda.device_count()} GPU(s) available.")
        print(f"Device name: {torch.cuda.get_device_name(_id)} (gpu_id={_id})")
        print("[GPU Memory Usage]")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), "GB")
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")

    print("=" * 32)
    return device
