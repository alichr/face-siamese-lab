"""The shared-weight encoder f_theta (plan §5, poster panels 1 & 3).

    resnet18 -> global average pool -> Linear(512 -> d) -> BatchNorm1d(d) -> L2 normalize

The poster draws two towers; there is only one `nn.Module` here, called twice.
Shared weights are not a constraint imposed on the architecture -- they *are* the
architecture. Two separately-parameterized towers could place the same face at
two unrelated points and the similarity would mean nothing.

No ImageNet pretraining by default. E1 compares three losses at equal budget; a
pretrained backbone would hand all three most of the representation before any
of them applied a gradient, compressing exactly the gap being measured.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from src.losses.geometry import l2_normalize

BACKBONES = {
    "resnet18": (models.resnet18, 512),
    "resnet50": (models.resnet50, 2048),
}


class Encoder(nn.Module):
    """Maps a face image to an embedding z in R^d, optionally on the unit sphere.

    Attributes:
        embedding_dim: d, the output dimensionality.
        normalize: whether the forward pass L2-normalizes (E6 turns this off).
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        embedding_dim: int = 128,
        normalize: bool = True,
        pretrained: bool = False,
    ) -> None:
        """
        Args:
            backbone: `"resnet18"` (default) or `"resnet50"`.
            embedding_dim: d, default 128 (E8 sweeps 64/128/256/512).
            normalize: L2-normalize the output. Default True; `False` is E6's
                ablation of the poster's "normalize embeddings" tip.
            pretrained: load ImageNet weights. Default False -- see module docstring.

        Raises:
            ValueError: on an unknown backbone or non-positive embedding_dim.
        """
        super().__init__()

        if backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {sorted(BACKBONES)}, got {backbone!r}")
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim}")

        ctor, feature_dim = BACKBONES[backbone]
        weights = "DEFAULT" if pretrained else None
        net = ctor(weights=weights)

        # Drop the ImageNet classifier; keep everything through the global average
        # pool, which is already the (N, feature_dim, 1, 1) tensor we want.
        net.fc = nn.Identity()
        self.backbone = net

        self.head = nn.Linear(feature_dim, embedding_dim)

        # BatchNorm on the embedding, before normalization. It centers and scales
        # each dimension, which stops the head from drifting to a large-magnitude
        # regime where the subsequent L2 normalize would squash all the useful
        # variation into a narrow cone.
        self.bn = nn.BatchNorm1d(embedding_dim)

        self.embedding_dim = embedding_dim
        self.normalize = normalize
        self.backbone_name = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images.

        Args:
            x: (N, 3, H, W) normalized image tensor.

        Returns:
            (N, d) embeddings; unit-norm rows when `self.normalize` is True.
        """
        features = self.backbone(x)
        z = self.bn(self.head(features))
        return l2_normalize(z) if self.normalize else z

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        return (
            f"Encoder({self.backbone_name}, d={self.embedding_dim}, "
            f"normalize={self.normalize}, {self.num_parameters() / 1e6:.1f}M params)"
        )


def build_encoder(cfg: dict) -> Encoder:
    """Construct an Encoder from a resolved config's `model` block."""
    return Encoder(
        backbone=cfg.get("backbone", "resnet18"),
        embedding_dim=cfg.get("embedding_dim", 128),
        normalize=cfg.get("normalize", True),
        pretrained=cfg.get("pretrained", False),
    )
