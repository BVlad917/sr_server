import torch
import numpy as np


def np_rgb_uint_2_tensor_norm(img, device=torch.device("cpu")):
    """
    Numpy array, range [0, 255], channels-last format --> PyTorch tensor, range [0, 1], channels-first, on GPU.
    Also has support for batch numpy array.
    """
    img = img / 255.  # normalize to [0, 1]
    img = np.transpose(img, axes=(2, 0, 1) if len(img.shape) == 3 else (0, 3, 1, 2))  # (C, H, W) or (B, C, H, W)
    img = torch.from_numpy(img).float()  # PyTorch tensor
    if len(img.shape) == 3:
        img = img.unsqueeze(0)  # add batch dimension if needed
    return img.to(device)


def tensor_norm_2_np_rgb_uint(img):
    """
    PyTorch tensor, range [0, 1], channels-first --> Numpy array, range [0, 255], channels-last format, on CPU.
    Also has support for batch tensor.
    """
    img = img.squeeze().float().cpu().clamp_(0, 1).numpy()
    img = np.transpose(img, axes=(1, 2, 0) if len(img.shape) == 3 else (0, 2, 3, 1))
    img = (img * 255.0).round().astype(np.uint8)
    return img
