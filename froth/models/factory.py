"""Construct architectures explicitly; no silent substitution between models."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import torch
from torch import nn

from froth.config import SAM_MODELS


def load_checkpoint(path):
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def load_model_weights(model, path):
    state = load_checkpoint(path)
    if isinstance(state, dict) and "mask_decoder" in state:
        if not hasattr(model, "mask_decoder"):
            raise ValueError("A decoder-only SAM checkpoint cannot initialize this architecture")
        model.mask_decoder.load_state_dict(state["mask_decoder"], strict=True)
        return
    for key in ("model_state", "model_state_dict", "state_dict", "model", "net_state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    if not isinstance(state, dict):
        raise ValueError("Expected a tensor state_dict or a supported checkpoint dictionary")
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def build_model(config, pretrained=None):
    spec = config["model"]
    name = spec["name"]
    pretrained = spec.get("pretrained", False) if pretrained is None else pretrained
    if name == "unet":
        from .unet import UNet
        model = UNet()
    elif name == "deeplabv3":
        from torchvision.models.segmentation import deeplabv3_resnet50
        from torchvision.models.segmentation.deeplabv3 import DeepLabHead
        model = deeplabv3_resnet50(weights="DEFAULT" if pretrained else None,
                                 weights_backbone=None)
        model.classifier = DeepLabHead(2048, 2)
    elif name == "deeplabv3plus":
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise RuntimeError("Install requirements/cnn.txt for DeepLabV3+.") from exc
        model = smp.DeepLabV3Plus(encoder_name=spec.get("encoder", "resnet50"),
                                 encoder_weights="imagenet" if pretrained else None,
                                 encoder_output_stride=spec.get("output_stride", 16),
                                 in_channels=3, classes=2)
    elif name == "fcn":
        from torchvision.models.segmentation import fcn_resnet50
        model = fcn_resnet50(weights="DEFAULT" if pretrained else None, weights_backbone=None)
        model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)
    elif name == "sbsnet":
        root = spec.get("external_root")
        if not root or not Path(root).is_dir():
            raise RuntimeError("Set model.external_root to the SBSNet-pytorch folder containing net/.")
        sys.path.insert(0, str(Path(root).resolve()))
        module_name, class_name = spec.get("class_path", "net.sbsnet.SBSNet:SBSNet").split(":")
        model = getattr(importlib.import_module(module_name), class_name)(num_classes=spec.get("output_channels", 2))
    elif name in SAM_MODELS:
        module_name = "segment_anything_hq" if name == "hq_sam" else "segment_anything"
        try:
            registry = importlib.import_module(module_name).sam_model_registry
        except ImportError as exc:
            raise RuntimeError(f"Install the {module_name} implementation described in docs/environment.md.") from exc
        base = spec.get("base_checkpoint")
        if not base or not Path(base).is_file():
            raise FileNotFoundError("Set model.base_checkpoint to the matching SAM/HQ-SAM/MedSAM base weights.")
        model = registry[spec.get("variant", "vit_b")](checkpoint=None)
        load_model_weights(model, base)
    else:
        raise ValueError(f"No builder for {name!r}")
    return model


def configure_fine_tuning(model, mode):
    groups = {"decoder_only": ("mask_decoder",), "decoder+prompt": ("mask_decoder", "prompt_encoder"),
              "prompt_only": ("prompt_encoder",), "encoder_only": ("image_encoder",),
              "full": ("image_encoder", "prompt_encoder", "mask_decoder")}
    if mode not in groups:
        raise ValueError(f"Unknown fine-tuning mode: {mode}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name in groups[mode]:
        for parameter in getattr(model, name).parameters():
            parameter.requires_grad_(True)


def logits_from_output(output):
    if isinstance(output, dict):
        return output["out"]
    if not torch.is_tensor(output):
        raise TypeError("Expected tensor logits or a torchvision {'out': logits} result")
    return output


def foreground_probability(logits, loss_name):
    if loss_name == "cross_entropy":
        return logits.softmax(dim=1)[:, 1]
    # SBSNet is trained with two identical binary target channels, so their
    # probabilities are averaged rather than treated as background/foreground.
    return logits.sigmoid().mean(dim=1)
