"""Phase 4 critical gate, driven from pytest.

The equivalence test itself must run under `torchrun` with 3 processes, which
pytest cannot express in-process. This module launches it as a subprocess and
asserts a clean exit, so `pytest tests/` covers the critical gate rather than
leaving it as a command someone has to remember to run.

Skipped when fewer than 3 GPUs are visible.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "tests" / "ddp_equivalence.py"

needs_3_gpus = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 3,
    reason="needs 3 CUDA devices",
)


@needs_3_gpus
@pytest.mark.slow
def test_ddp_equivalence_gate() -> None:
    """Single-GPU vs 3-GPU: loss AND gradients must match, and the negative
    control must fail. Any binding check failing exits non-zero."""
    torchrun = shutil.which("torchrun", path=str(Path(sys.executable).parent)) or "torchrun"

    result = subprocess.run(
        [torchrun, "--nproc_per_node=3", str(SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    output = result.stdout + result.stderr

    assert result.returncode == 0, f"Phase 4 gate FAILED:\n{output}"
    assert "ALL PHASE 4 GATE CHECKS PASSED" in output, output

    # The negative control must genuinely have run and failed -- otherwise the
    # gradient check could be passing vacuously.
    assert "negative control: detached gather FAILS" in output, output


def test_gather_is_identity_when_not_distributed() -> None:
    """Single-process runs share the same code path; the gather must be a no-op."""
    from src.engine.gather import gather_labels, gather_with_grad, get_rank, get_world_size

    z = torch.randn(8, 16, requires_grad=True)
    labels = torch.arange(8)

    assert gather_with_grad(z) is z
    assert torch.equal(gather_labels(labels), labels)
    assert get_rank() == 0 and get_world_size() == 1


def test_global_infonce_inputs_single_process() -> None:
    """rank_offset must be 0 outside DDP, so self-masking hits the right entry."""
    from src.engine.gather import global_infonce_inputs

    z = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    labels = torch.arange(4).repeat_interleave(2)
    pool, pool_labels, offset = global_infonce_inputs(z, labels)

    assert offset == 0
    assert torch.equal(pool, z) and torch.equal(pool_labels, labels)


def test_pk_sampler_ranks_get_disjoint_identities() -> None:
    """Global negatives are only 'more negatives' if the ranks hold different
    identities. Overlapping ranks would make some gathered rows positives."""
    import numpy as np

    from src.data.pk_sampler import PKSampler

    labels = np.repeat(np.arange(60), 6)
    samplers = [
        PKSampler(labels, p=8, k=4, seed=0, rank=r, world_size=3, num_batches=5)
        for r in range(3)
    ]
    for s in samplers:
        s.set_epoch(0)

    for step_batches in zip(*[iter(s) for s in samplers]):
        id_sets = [set(labels[np.asarray(b)].tolist()) for b in step_batches]
        assert all(len(s) == 8 for s in id_sets)
        assert not (id_sets[0] & id_sets[1]), "rank 0 and 1 share an identity"
        assert not (id_sets[0] & id_sets[2]), "rank 0 and 2 share an identity"
        assert not (id_sets[1] & id_sets[2]), "rank 1 and 2 share an identity"
        assert len(id_sets[0] | id_sets[1] | id_sets[2]) == 24


def test_pk_sampler_rejects_impossible_world_size() -> None:
    import numpy as np

    from src.data.pk_sampler import PKSampler

    labels = np.repeat(np.arange(10), 6)
    with pytest.raises(ValueError, match="disjoint slots per step"):
        PKSampler(labels, p=8, k=4, world_size=3)
