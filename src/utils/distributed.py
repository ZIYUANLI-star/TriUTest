import torch, os
def bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
