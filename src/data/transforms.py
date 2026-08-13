"""Image preprocessing and the three augmentation levels (plan §3, poster panel 8).

Geometry is fixed across every experiment: resize 128, then 112 crop (random
when training, center when evaluating), normalized with mean = std = 0.5 so
inputs land in roughly [-1, 1]. Only the *augmentation level* varies, and it
varies alone -- that is what makes E7 a clean measurement of the poster's
"use strong data augmentation" tip rather than a confound.

Levels (plan §3):
    none    crop only -- the control
    basic   + horizontal flip
    strong  + color jitter 0.4, random grayscale p=0.2, Gaussian blur p=0.5,
              random erasing p=0.25

A note on why this matters more for InfoNCE than for the pairwise losses: with
PK sampling, InfoNCE's denominator sees every other batch sample, so any cue
that trivially identifies an image (fixed crop, fixed color cast) lets the
network shortcut the discrimination task. Augmentation removes the shortcut.
"""

from __future__ import annotations

from torchvision import transforms

RESIZE: int = 128
CROP: int = 112
_MEAN = (0.5, 0.5, 0.5)
_STD = (0.5, 0.5, 0.5)

AUGMENTATION_LEVELS = ("none", "basic", "strong")


def build_transform(level: str = "basic", train: bool = True):
    """Build the preprocessing pipeline.

    Args:
        level: one of `AUGMENTATION_LEVELS`. Ignored when `train=False` --
            evaluation is always deterministic, otherwise the same checkpoint
            would score differently on consecutive runs and no comparison in the
            matrix would be trustworthy.
        train: random crop + augmentation if True, else center crop only.

    Returns:
        A torchvision transform mapping PIL image -> normalized float tensor
        of shape (3, 112, 112).

    Raises:
        ValueError: on an unknown level.
    """
    if level not in AUGMENTATION_LEVELS:
        raise ValueError(f"level must be one of {AUGMENTATION_LEVELS}, got {level!r}")

    if not train:
        return transforms.Compose(
            [
                transforms.Resize(RESIZE),
                transforms.CenterCrop(CROP),
                transforms.ToTensor(),
                transforms.Normalize(_MEAN, _STD),
            ]
        )

    steps: list = [transforms.Resize(RESIZE), transforms.RandomCrop(CROP)]

    if level in ("basic", "strong"):
        steps.append(transforms.RandomHorizontalFlip(p=0.5))

    if level == "strong":
        steps += [
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5
            ),
        ]

    steps += [transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]

    if level == "strong":
        # After ToTensor: RandomErasing operates on tensors, not PIL images.
        steps.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)))

    return transforms.Compose(steps)
