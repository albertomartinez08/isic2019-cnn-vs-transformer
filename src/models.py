"""Model factory (timm, ImageNet-1k pretrained) and model-cost utilities."""
import timm
import torch

import config as C


def create(key: str, pretrained: bool = True) -> torch.nn.Module:
    name = C.MODELS[key]
    return timm.create_model(name, pretrained=pretrained, num_classes=C.NUM_CLASSES)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_flops(model: torch.nn.Module, device) -> float:
    """GFLOPs for one 224x224 image (multiply-adds counted as 2 FLOPs by torch)."""
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    x = torch.zeros(1, 3, C.IMG_SIZE, C.IMG_SIZE, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 1e9


def gradcam_target_layer(key: str, model: torch.nn.Module):
    """Return (target_layers, reshape_transform) for pytorch-grad-cam."""
    if key == "resnet50":
        return [model.layer4[-1]], None
    if key == "efficientnet_b0":
        return [model.conv_head], None
    if key == "swin_t":
        def reshape(t):  # timm swin blocks return (B, H, W, C)
            if t.dim() == 4:
                return t.permute(0, 3, 1, 2)
            b, n, c = t.shape
            s = int(n ** 0.5)
            return t.reshape(b, s, s, c).permute(0, 3, 1, 2)
        return [model.layers[-1].blocks[-1].norm2], reshape
    raise ValueError(key)
