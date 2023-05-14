import os
import torch
from functools import partial

from image_utils import np_rgb_uint_2_tensor_norm, tensor_norm_2_np_rgb_uint
from sisr_architectures.swin2_sr.network_swin2sr import Swin2SR
from sisr_architectures.swin_ir.network_swinir import SwinIR
from sisr_architectures.RRDBNet_arch import RRDBNet as CustomRRDBNet
from basicsr.archs.rrdbnet_arch import RRDBNet as BasicSRRRDBNet

SCALE = 4  # super-resolution scale
SWIN_WINDOW_SIZE = 8  # window size used for SwinIR and Swin2SR for Real Image Super-Resolution
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device to run SISR model
ARCHITECTURES_DIR = "./sisr_architectures/"  # directory with architecture classes
MODEL_WEIGHTS = "./sisr_weights/"  # directory with model weights

# model name to the class needed for that model
NAME_2_CLASS = {
    "swin_ir": SwinIR,
    "swin2_sr": Swin2SR,
    "esrgan": CustomRRDBNet,
    "real_esrgan": BasicSRRRDBNet,
    "bsrgan": CustomRRDBNet,
}
# model name to the class needed for that model
NAME_2_WEIGHT_FILES = {
    "swin_ir": "swin_ir.pth",
    "swin2_sr": "swin2_sr.pth",
    "esrgan": "esrgan.pth",
    "real_esrgan": "real_esrgan.pth",
    "bsrgan": "bsrgan.pth",
}
# model name to the arguments needed to define that model
NAME_2_ARGS = {
    "swin_ir":
        {"upscale": 4, "in_chans": 3, "img_size": 64, "window_size": 8,
         "img_range": 1., "depths": [6, 6, 6, 6, 6, 6, 6, 6, 6], "embed_dim": 240,
         "num_heads": [8, 8, 8, 8, 8, 8, 8, 8, 8], "mlp_ratio": 2,
         "upsampler": 'nearest+conv', "resi_connection": '3conv'},
    "swin2_sr":
        {"upscale": 4, "in_chans": 3, "img_size": 64, "window_size": 8,
         "img_range": 1., "depths": [6, 6, 6, 6, 6, 6], "embed_dim": 180, "num_heads": [6, 6, 6, 6, 6, 6],
         "mlp_ratio": 2, "upsampler": 'nearest+conv', "resi_connection": '1conv'},
    "esrgan":
        {"in_nc": 3, "out_nc": 3, "nf": 64, "nb": 23, "gc": 32},
    "real_esrgan":
        {"num_in_ch": 3, "num_out_ch": 3},
    "bsrgan":
        {"in_nc": 3, "out_nc": 3, "nf": 64, "nb": 23, "gc": 32},
}


@torch.no_grad()
def run_sr_inference(lr_img, model, scale, window_size=None):
    """
    Run SISR inference on a low resolution image with a given model, scale, and window size (if it is the case)
    :param lr_img: LR image to be super resoluted; tensor (B, C, H, W), RGB format
    :param model: SISR model; Pytorch Module
    :param scale: raise image resolution by this factor; int
    :param window_size: size of the window used in Swin Transformer Block (for SwinIR and Swin2SR)
    :return: super resoluted image; tensor (B, C, 4 * H, 4 * W), RGB format
    """
    _, _, h_old, w_old = lr_img.shape
    h_pad = 0 if window_size is None else (window_size - h_old % window_size) % window_size
    w_pad = 0 if window_size is None else (window_size - w_old % window_size) % window_size

    lr_img = torch.cat([lr_img, torch.flip(lr_img, [2])], 2)[:, :, :h_old + h_pad, :]
    lr_img = torch.cat([lr_img, torch.flip(lr_img, [3])], 3)[:, :, :, :w_old + w_pad]

    sr_img = model(lr_img)
    sr_img = sr_img[..., :h_old * scale, :w_old * scale]
    return sr_img


def sisr_forward(img, device, sisr_model, scale, window_size):
    """
    Apply image super-resolution on given image(s) as a numpy array(s)
    :param img: image for super-resolution; np array ((B), H, W, C), RGB format
    :param device: device to run SISR inference on; string or torch.device() instance
    :param sisr_model: SISR model; PyTorch module
    :param scale: SISR scale
    :param window_size: size of window used in Swin Transformer Blocks (for SwinIR and Swin2SR)
    :return: super resoluted images; np array(s) ((B), H * 4, W * 4, C), RGB format
    """
    img_tensor = np_rgb_uint_2_tensor_norm(img, device=device)
    img_sr_tensor = run_sr_inference(lr_img=img_tensor, model=sisr_model, scale=scale, window_size=window_size)
    sr_img = tensor_norm_2_np_rgb_uint(img_sr_tensor)
    return sr_img


def load_model(model_name, device=DEVICE, eval_model=True):
    """
    Load a pretrained model.
    Available options: ["swin_ir", "swin2_sr", "esrgan", "real_esrgan", "bsrgan"]
    """
    assert model_name in NAME_2_CLASS, f"ERROR: No architecture for given model: {model_name}"
    assert model_name in NAME_2_WEIGHT_FILES, f"ERROR: No weights for given model: {model_name}"
    assert model_name in NAME_2_ARGS, f"ERROR: No arguments for given model: {model_name}"

    arch = NAME_2_CLASS[model_name]
    weights_file_name = NAME_2_WEIGHT_FILES[model_name]
    model_kwargs = NAME_2_ARGS[model_name]

    model = arch(**model_kwargs)
    weights_file_path = os.path.join(MODEL_WEIGHTS, weights_file_name)
    weights = torch.load(weights_file_path)
    weights_key = "params_ema" if "params_ema" in weights else "params" if "params" in weights else None
    model.load_state_dict(weights[weights_key] if weights_key else weights, strict=True)
    model = model.to(device)
    model = model.eval() if eval_model else model.train()
    return model


def get_sisr_forward_fn(model_name, device=DEVICE):
    """
    Get a partial function corresponding to SISR inference on a given model. Can be used as a simple
    method which is applied directly on a numpy image (or batch of numpy images).
    Available options: ["swin_ir", "swin2_sr", "esrgan", "real_esrgan", "bsrgan"]
    """
    model = load_model(model_name=model_name, device=device, eval_model=True)
    return partial(sisr_forward,
                   device=device,
                   sisr_model=model,
                   scale=SCALE,
                   window_size=SWIN_WINDOW_SIZE if "swin" in model_name else None)
