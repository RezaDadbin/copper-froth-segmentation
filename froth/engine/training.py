"""Shared CNN and SAM-family training with explicit validation and saved history."""

from __future__ import annotations

import random
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from froth.config import SAM_MODELS
from froth.data.dataset import FrothDataset, NativeFrothDataset, native_collate
from froth.data.labelme import read_samples
from froth.io import provenance, sha256, write_csv, write_json
from froth.metrics import binary_metrics
from froth.models.factory import build_model, configure_fine_tuning, foreground_probability, logits_from_output
from froth.models.sam_family import prompted_logits


def select_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BCEDiceLoss(nn.Module):
    def forward(self, logits, target):
        probability = logits.sigmoid()
        dice = (2 * (probability * target).sum() + 1) / (probability.sum() + target.sum() + 1)
        return 0.5 * F.binary_cross_entropy_with_logits(logits, target) + 0.5 * (1 - dice)


def make_loss(config, samples, device):
    name = config["training"]["loss"]
    if name == "cross_entropy":
        weights = None
        if config["training"].get("class_weights") == "estimate":
            counts = np.zeros(2, dtype=np.int64)
            dataset = FrothDataset(samples[:200], config["data"])
            for _, mask in dataset:
                counts += np.bincount(mask.numpy().ravel(), minlength=2)
            weights = 1 / np.clip(counts / counts.sum(), 1e-6, 1)
            weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=weights)
    if name == "bce_dice":
        return BCEDiceLoss()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unknown loss: {name}")


def run_epoch(model, loader, config, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    family = config["model"]["name"]
    if training and family in SAM_MODELS:
        # Frozen image/prompt encoders keep evaluation behavior (e.g. dropout).
        for module_name in ("image_encoder", "prompt_encoder", "mask_decoder"):
            module = getattr(model, module_name)
            module.train(any(p.requires_grad for p in module.parameters()))
    accumulation = config["training"].get("accumulation_steps", 1)
    if training:
        optimizer.zero_grad(set_to_none=True)
    rows, total_loss, total_images = [], 0.0, 0
    for step, batch in enumerate(loader):
        with torch.set_grad_enabled(training):
            if family in SAM_MODELS:
                image, mask = batch[0]
                logits = prompted_logits(model, image, mask, family, config["training"]["prompt_source"])
                target = torch.as_tensor(mask, device=device).float()[None, None]
                probability = logits.sigmoid()[:, 0]
                loss = criterion(logits, target)
                targets = target[:, 0]
            else:
                images, targets = (x.to(device) for x in batch)
                logits = logits_from_output(model(images))
                if logits.shape[-2:] != targets.shape[-2:]:
                    logits = F.interpolate(logits, targets.shape[-2:], mode="bilinear", align_corners=False)
                if config["training"]["loss"] == "cross_entropy":
                    loss = criterion(logits, targets)
                else:
                    target = targets.float()[:, None].expand_as(logits)
                    loss = criterion(logits, target)
                probability = foreground_probability(logits, config["training"]["loss"])
            if training:
                # Normalize the final partial accumulation group by its real size.
                group_start = (step // accumulation) * accumulation
                group_size = min(accumulation, len(loader) - group_start)
                (loss / group_size).backward()
                if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                    clip = config["training"].get("gradient_clip", 0)
                    if clip:
                        nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        predictions = (probability.detach() > config["inference"]["threshold"]).cpu().numpy()
        truths = targets.detach().cpu().numpy()
        total_loss += float(loss.detach()) * len(predictions)
        total_images += len(predictions)
        rows.extend(binary_metrics(p, g, distances=False) for p, g in zip(predictions, truths))
        every = config["training"].get("log_every", 25)
        if (step + 1) % every == 0 or step + 1 == len(loader):
            print(f"  {'train' if training else 'validation'} {step+1}/{len(loader)} batches", flush=True)
    return {"loss": total_loss / total_images,
            **{key: float(np.mean([row[key] for row in rows])) for key in ("iou", "miou", "dice", "pixel_acc")}}


def train(config, dataset_root, manifest, output, device="auto", limit_train=None, limit_val=None, pretrained=None):
    output = Path(output)
    if (output / "last.pt").exists() or (output / "history.csv").exists():
        raise FileExistsError(f"Choose a new experiment directory; existing training output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    train_samples = read_samples(dataset_root, manifest, "train")[:limit_train]
    val_samples = read_samples(dataset_root, manifest, "test")[:limit_val]
    options = config["training"]
    set_seed(options.get("seed", 1337))
    device = select_device(device)
    model = build_model(config, pretrained=pretrained).to(device)
    sam = config["model"]["name"] in SAM_MODELS
    if sam:
        configure_fine_tuning(model, options.get("fine_tune", "decoder_only"))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer_type = options.get("optimizer", "adam")
    optimizer_args = dict(lr=options["lr"], weight_decay=options.get("weight_decay", 0))
    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(parameters, momentum=0.9, nesterov=True, **optimizer_args)
    else:
        optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[optimizer_type]
        optimizer = optimizer_class(parameters, **optimizer_args)
    criterion = make_loss(config, train_samples, device)
    workers = options.get("num_workers", 0)
    common = dict(batch_size=options["batch_size"], num_workers=workers, pin_memory=device.type == "cuda")
    if sam:
        common["collate_fn"] = native_collate
        train_ds, val_ds = NativeFrothDataset(train_samples), NativeFrothDataset(val_samples)
    else:
        train_ds, val_ds = FrothDataset(train_samples, config["data"], True), FrothDataset(val_samples, config["data"])
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    metadata = provenance(config, manifest, device=str(device), train_images=len(train_ds), validation_images=len(val_ds),
                          train_image_names=[s.image_name for s in train_samples],
                          validation_image_names=[s.image_name for s in val_samples],
                          pretrained_initialization=config["model"].get("pretrained", False) if pretrained is None else pretrained)
    if config["model"].get("base_checkpoint"):
        metadata["base_checkpoint_sha256"] = sha256(config["model"]["base_checkpoint"])
    metadata["input_sha256"] = {
        sample.image_name: {"image": sha256(sample.image_path), "annotation": sha256(sample.annotation_path)}
        for sample in [*train_samples, *val_samples]}
    write_json(output / "run.json", metadata)
    run_hash = sha256(output / "run.json")
    best, stale, history = -float("inf"), 0, []
    metric = options.get("selection_metric", "iou")
    for epoch in range(1, options["epochs"] + 1):
        start = time.monotonic()
        if options.get("poly_power"):
            lr = options["lr"] * (1 - (epoch - 1) / options["epochs"]) ** options["poly_power"]
            for group in optimizer.param_groups:
                group["lr"] = lr
        tr = run_epoch(model, train_loader, config, criterion, device, optimizer)
        va = run_epoch(model, val_loader, config, criterion, device)
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "seconds": time.monotonic()-start,
               **{f"train_{k}": v for k, v in tr.items()}, **{f"validation_{k}": v for k, v in va.items()}}
        history.append(row)
        write_csv(output / "history.csv", history)
        state = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                 "epoch": epoch, "config": config, "validation": va, "training_run_sha256": run_hash}
        torch.save(state, output / "last.pt")
        if va[metric] > best:
            best, stale = va[metric], 0
            torch.save(state, output / "best.pt")
        else:
            stale += 1
        print(f"Epoch {epoch}/{options['epochs']}: loss={tr['loss']:.5f}, validation {metric}={va[metric]:.5f}", flush=True)
        patience = options.get("early_stop_patience", 0)
        if patience and stale >= patience:
            break
    from froth.visualization.training import render_history
    render_history(output / "history.csv", output / "figures/training_curves")
    return output / "best.pt"
