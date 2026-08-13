"""Encoder tests (plan §5) — shared weights, normalization, output shape."""

from __future__ import annotations

import pytest
import torch

from src.models.encoder import Encoder, build_encoder


@pytest.fixture(autouse=True)
def _determinism() -> None:
    torch.manual_seed(0)


def _images(n: int = 4) -> torch.Tensor:
    return torch.randn(n, 3, 112, 112)


@pytest.mark.parametrize("d", [64, 128, 256, 512])
def test_output_shape_matches_embedding_dim(d: int) -> None:
    """E8 sweeps d; the head must actually honour it."""
    model = Encoder(embedding_dim=d).eval()
    assert model(_images()).shape == (4, d)


def test_output_is_unit_norm_when_normalize_true() -> None:
    model = Encoder(embedding_dim=128, normalize=True).eval()
    norms = model(_images()).norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_output_is_not_unit_norm_when_normalize_false() -> None:
    """E6's ablation must genuinely change the geometry, not just a flag."""
    model = Encoder(embedding_dim=128, normalize=False).eval()
    norms = model(_images(8)).norm(dim=-1)
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-2)


def test_shared_weights_are_literal(  ) -> None:
    """Poster panel 3: both 'branches' are one module called twice.

    The same input through two calls must give the identical embedding -- if the
    module held any per-call state, the siamese property would be broken.
    """
    model = Encoder().eval()
    x = _images()
    a = model(x)
    b = model(x)
    assert torch.allclose(a, b, atol=1e-6)


def test_two_inputs_through_one_module_equals_batched_call() -> None:
    """Encoding x1 and x2 separately == encoding them concatenated (eval mode)."""
    model = Encoder().eval()
    x1, x2 = _images(3), _images(3)
    separate = torch.cat([model(x1), model(x2)])
    together = model(torch.cat([x1, x2]))
    assert torch.allclose(separate, together, atol=1e-5)


def test_gradients_reach_the_backbone() -> None:
    """Catches a detached head or a frozen backbone."""
    model = Encoder()
    model(_images()).sum().backward()
    first_conv = model.backbone.conv1.weight
    assert first_conv.grad is not None and first_conv.grad.abs().sum() > 0


def test_batchnorm_makes_training_and_eval_differ() -> None:
    """BN1d uses batch stats in train mode and running stats in eval.

    This is why the Phase 4 Stage B equivalence test must run in eval mode.
    """
    model = Encoder()
    x = _images(8)
    model.train()
    train_out = model(x)
    model.eval()
    eval_out = model(x)
    assert not torch.allclose(train_out, eval_out, atol=1e-4)


def test_build_encoder_from_config() -> None:
    model = build_encoder({"backbone": "resnet18", "embedding_dim": 64, "normalize": False})
    assert model.embedding_dim == 64 and model.normalize is False


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"backbone": "vgg16"}, "backbone must be"), ({"embedding_dim": 0}, "embedding_dim")],
)
def test_invalid_config_rejected(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Encoder(**kwargs)


def test_not_pretrained_by_default() -> None:
    """E1 compares losses at equal budget; pretrained weights would compress the gap."""
    a = Encoder(pretrained=False)
    b = Encoder(pretrained=False)
    assert not torch.allclose(a.backbone.conv1.weight, b.backbone.conv1.weight)
