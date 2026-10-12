"""Explicit experiment configuration, independent of the current directory."""

from __future__ import annotations

import copy
import json
from pathlib import Path


MODELS = ("unet", "sbsnet", "deeplabv3", "deeplabv3plus", "fcn", "sam", "hq_sam", "medsam")
SAM_MODELS = ("sam", "hq_sam", "medsam")


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    # Checkpoint and external-code paths in JSON are relative to the repo,
    # rather than the shell's working directory.
    root = path.parent.parent if path.parent.name == "configs" else path.parent
    for key in ("base_checkpoint", "external_root"):
        value = config["model"].get(key)
        if value:
            p = Path(value).expanduser()
            config["model"][key] = str(p if p.is_absolute() else (root / p).resolve())
    return config


def validate_config(config):
    name = config["model"]["name"]
    if name not in MODELS:
        raise ValueError(f"Unsupported maintained model {name!r}; see docs/model_support.md")
    data, train = config["data"], config["training"]
    size = data.get("image_size")
    if name not in SAM_MODELS:
        if not isinstance(size, list) or len(size) != 2 or min(size) <= 0:
            raise ValueError("data.image_size must be [height, width] in positive pixels")
        if name == "unet" and any(int(x) % 32 for x in size):
            raise ValueError("The supplied U-Net requires image dimensions divisible by 32")
    if train["epochs"] < 1 or train["batch_size"] < 1 or train["lr"] <= 0:
        raise ValueError("epochs, batch_size and learning rate must be positive")
    if not 0 < config["inference"].get("threshold", 0.5) < 1:
        raise ValueError("The probability threshold must be between 0 and 1")
    if name in SAM_MODELS and train["batch_size"] != 1:
        raise ValueError("SAM-family training uses batch_size=1 with optional gradient accumulation")
    if data.get("crop_size") and name == "unet" and data["crop_size"] % 32:
        raise ValueError("U-Net training crop size must be divisible by 32")
    if train.get("accumulation_steps", 1) < 1:
        raise ValueError("accumulation_steps must be positive")
    if name in SAM_MODELS and train.get("prompt_source") not in {"full_image", "ground_truth"}:
        raise ValueError("SAM training must declare full_image or ground_truth box prompts")
    return config


def with_overrides(config, **overrides):
    result = copy.deepcopy(config)
    for key, value in overrides.items():
        if value is not None:
            result["training"][key] = value
    return validate_config(result)
