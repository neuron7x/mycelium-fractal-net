"""Tests for mfn.diagnose() — unified diagnostic pipeline."""

from __future__ import annotations

import pytest

import mycelium_fractal_net as mfn
from mycelium_fractal_net.types.diagnosis import (
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_STABLE,
    SEVERITY_WARNING,
    DiagnosisReport,
)
from mycelium_fractal_net.types.field import (
    GABAATonicSpec,
    NeuromodulationSpec,
    SerotonergicPlasticitySpec,
    SimulationSpec,
)

VALID_SEVERITIES = {SEVERITY_STABLE, SEVERITY_INFO, SEVERITY_WARNING, SEVERITY_CRITICAL}


@pytest.fixture(scope="module")
def baseline_seq() -> mfn.FieldSequence:
    return mfn.simulate(SimulationSpec(grid_size=32, steps=60, seed=42))


@pytest.fixture(scope="module")
def gaba_seq() -> mfn.FieldSequence:
    gaba = GABAATonicSpec(
        profile="intervention",
        agonist_concentration_um=40.0,
        shunt_strength=0.6,
        rest_offset_mv=-15.0,
        desensitization_rate_hz=0.05,
        recovery_rate_hz=0.02,
    )
    nm = NeuromodulationSpec(profile="intervention", enabled=True, gabaa_tonic=gaba)
    return mfn.simulate(SimulationSpec(grid_size=32, steps=60, seed=42, neuromodulation=nm))


def test_returns_diagnosis_report(baseline_seq: mfn.FieldSequence) -> None:
    report = mfn.diagnose(baseline_seq)
    assert isinstance(report, DiagnosisReport)
    assert report.severity in VALID_SEVERITIES
    assert isinstance(report.narrative, str) and len(report.narrative) > 10
    assert report.anomaly is not None
    assert report.warning is not None
    assert report.forecast is not None
    assert report.causal is not None
    assert report.descriptor is not None
    assert "diagnosis_time_ms" in report.metadata


def test_summary_and_to_dict(baseline_seq: mfn.FieldSequence) -> None:
    report = mfn.diagnose(baseline_seq)
    s = report.summary()
    assert "DIAGNOSIS:" in s
    assert "anomaly=" in s
    assert "ews=" in s
    d = report.to_dict()
    for key in ("severity", "narrative", "anomaly", "warning", "causal", "plan", "metadata"):
        assert key in d


def test_skip_intervention(baseline_seq: mfn.FieldSequence) -> None:
    report = mfn.diagnose(baseline_seq, skip_intervention=True)
    assert report.plan is None


def test_gaba_diagnosis(gaba_seq: mfn.FieldSequence) -> None:
    report = mfn.diagnose(gaba_seq, intervention_max_candidates=8)
    assert report.severity in VALID_SEVERITIES
    d = report.to_dict()
    assert isinstance(d, dict)


def test_deterministic(baseline_seq: mfn.FieldSequence) -> None:
    r1 = mfn.diagnose(baseline_seq, skip_intervention=True)
    r2 = mfn.diagnose(baseline_seq, skip_intervention=True)
    assert r1.severity == r2.severity
    assert r1.warning.ews_score == r2.warning.ews_score
    assert r1.anomaly.label == r2.anomaly.label
    assert r1.causal.decision == r2.causal.decision


def test_helpers(baseline_seq: mfn.FieldSequence) -> None:
    report = mfn.diagnose(baseline_seq, skip_intervention=True)
    assert isinstance(report.is_ok(), bool)
    assert not report.needs_intervention()


def test_early_warning_standalone(baseline_seq: mfn.FieldSequence) -> None:
    w = mfn.early_warning(baseline_seq)
    assert isinstance(w, mfn.CriticalTransitionWarning)
    assert 0.0 <= w.ews_score <= 1.0
    assert isinstance(w.transition_type, str)
    assert w.causal_certificate != ""
