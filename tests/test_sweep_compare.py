"""Tests for the Phase 5 orchestration helpers.

These matter because a parsing bug in `compare.py` would silently mis-plot the
whole matrix -- e.g. reading tau=0.5 as 0.05 would reorder the E4 sweep and
invent a U-shape that is not in the data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from compare import SWEEPS, _numeric  # noqa: E402
from sweep import Job, is_complete, is_ddp, write_status  # noqa: E402


# --- run-name -> swept value parsing ------------------------------------------


@pytest.mark.parametrize(
    ("name", "experiment", "expected"),
    [
        ("e2_contrastive_m0.25", "e2", 0.25),
        ("e2_contrastive_m1.5", "e2", 1.5),
        ("e3_triplet_a0.1_random", "e3", 0.1),
        ("e3_triplet_a0.4_batch-hard", "e3", 0.4),
        ("e4_infonce_tau0.03", "e4", 0.03),
        ("e4_infonce_tau0.5", "e4", 0.5),
        ("e5_infonce_batch64", "e5", 64.0),
        ("e5_infonce_global768", "e5", 768.0),
        ("e8_dim512", "e8", 512.0),
    ],
)
def test_swept_value_parsed_from_run_name(name: str, experiment: str, expected: float) -> None:
    pattern = SWEEPS[experiment][1]
    assert _numeric(name, pattern) == expected


def test_tau_0_5_is_not_confused_with_0_05() -> None:
    """The E4 sweep's shape depends on this ordering being right."""
    pattern = SWEEPS["e4"][1]
    values = [
        _numeric(f"e4_infonce_tau{t}", pattern) for t in ("0.03", "0.05", "0.07", "0.1", "0.2", "0.5")
    ]
    assert values == [0.03, 0.05, 0.07, 0.1, 0.2, 0.5]
    assert values == sorted(values)


def test_unparseable_name_returns_none() -> None:
    assert _numeric("e2_contrastive_nomargin", SWEEPS["e2"][1]) is None


# --- sweep bookkeeping --------------------------------------------------------


def _write_config(tmp_path: Path, name: str, cfg: dict) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_is_complete_false_without_metrics(tmp_path: Path) -> None:
    cfg = {"output_dir": str(tmp_path / "run"), "loss": {"name": "infonce"}}
    assert not is_complete(_write_config(tmp_path, "a", cfg))


def test_is_complete_true_with_real_metrics(tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    (out / "metrics.json").write_text(json.dumps({"internal": {"auc": 0.97}}))
    cfg = {"output_dir": str(out), "loss": {"name": "infonce"}}
    assert is_complete(_write_config(tmp_path, "b", cfg))


def test_is_complete_false_on_truncated_json(tmp_path: Path) -> None:
    """A run killed mid-write must be retried, not counted as done."""
    out = tmp_path / "run"
    out.mkdir()
    (out / "metrics.json").write_text('{"internal": {"auc": 0.9')
    cfg = {"output_dir": str(out), "loss": {"name": "infonce"}}
    assert not is_complete(_write_config(tmp_path, "c", cfg))


def test_ddp_configs_are_identified(tmp_path: Path) -> None:
    """The global-negatives row needs torchrun and must not go into the sweep."""
    local = _write_config(tmp_path, "loc", {"loss": {"name": "infonce", "negatives": "local"}})
    glob = _write_config(tmp_path, "glo", {"loss": {"name": "infonce", "negatives": "global"}})
    assert not is_ddp(local)
    assert is_ddp(glob)


def test_status_table_renders_every_job(tmp_path: Path, monkeypatch) -> None:
    import sweep

    status = tmp_path / "status.md"
    monkeypatch.setattr(sweep, "STATUS_PATH", status)

    jobs = [
        Job(config=Path("configs/a.yaml"), name="a", status="done", gpu=0, seconds=120.0,
            headline="LFW 0.9012±0.01"),
        Job(config=Path("configs/b.yaml"), name="b", status="running", gpu=1),
        Job(config=Path("configs/c.yaml"), name="c", status="FAILED", error="exit 1"),
    ]
    write_status(jobs, started=0.0)
    text = status.read_text()

    for job in jobs:
        assert f"`{job.name}`" in text
    assert "LFW 0.9012" in text
    assert "FAILED" in text


def test_all_matrix_configs_exist_and_differ_in_one_variable() -> None:
    """The point of generating configs: only the swept variable may change."""
    configs = sorted(Path("configs").glob("e4_infonce_tau*.yaml"))
    if not configs:
        pytest.skip("configs not generated")

    loaded = [yaml.safe_load(p.read_text()) for p in configs]
    taus = {c["loss"]["temperature"] for c in loaded}
    assert len(taus) == len(loaded), "temperatures are not distinct"

    # Everything except temperature and output_dir must be identical.
    def stripped(cfg: dict) -> str:
        c = yaml.safe_load(yaml.safe_dump(cfg))
        c["loss"].pop("temperature")
        c.pop("output_dir")
        return yaml.safe_dump(c, sort_keys=True)

    assert len({stripped(c) for c in loaded}) == 1, "E4 configs differ in more than τ"
