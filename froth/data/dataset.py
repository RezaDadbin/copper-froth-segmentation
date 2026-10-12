"""Tensor dataset with synchronized image/mask augmentation."""

from __future__ import annotations

import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .labelme import load_sample, transform_pair


def image_tensor(image, normalize=False):
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
    if normalize:
        mean = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        tensor = (tensor - mean) / std
    return tensor


class FrothDataset(Dataset):
    def __init__(self, samples, config, training=False):
        self.samples, self.config, self.training = samples, config, training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image, mask = load_sample(sample)
        image, mask, _ = transform_pair(image, mask, self.config)
        crop_size = self.config.get("crop_size") if self.training else None
        if crop_size:
            factor = random.uniform(*self.config.get("scale_range", [0.7, 1.7]))
            h, w = mask.shape
            h, w = max(2, int(h * factor)), max(2, int(w * factor))
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            ph, pw = max(0, crop_size-h), max(0, crop_size-w)
            image = cv2.copyMakeBorder(image, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
            mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
            y, x = random.randint(0, image.shape[0]-crop_size), random.randint(0, image.shape[1]-crop_size)
            image, mask = image[y:y+crop_size, x:x+crop_size], mask[y:y+crop_size, x:x+crop_size]
        if self.training and random.random() < self.config.get("horizontal_flip_probability", 0):
            image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
        return image_tensor(image, self.config.get("normalize", False)), torch.from_numpy(mask.copy()).long()


def native_collate(batch):
    """Keep native SAM samples as a list; dimensions may differ between images."""
    return batch


class NativeFrothDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return load_sample(self.samples[index])
