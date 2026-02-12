"""Analysis helpers for pressure-response alignment trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any


@dataclass(frozen=True)
class CompressionEvent:
    """Detected xi compression event after pressure onset."""

    onset_turn: int
    duration: int
    magnitude: float


@dataclass(frozen=True)
class CrisisProfile:
    """Crisis-window summary for xi, lvs, and pt metrics."""

    xi_trajectory: str
    pt_trend: str
    max_xi: float
    xi_variance: float
    window_start: int
    window_end: int


@dataclass(frozen=True)
class OptionEProfile:
    """Summary for creative third-path behavior (behavioral code E)."""

    is_option_e: bool
    profile_type: str


@dataclass(frozen=True)
class ConditionComparison:
    """Pairwise condition comparison with effect size."""

    condition_a: str
    condition_b: str
    metric: str
    delta: float
    effect_size: float


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_val = sum(values) / len(values)
    return sum((v - mean_val) ** 2 for v in values) / (len(values) - 1)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    return float(median(values)) if values else 0.0


def _cohens_d(sample_a: list[float], sample_b: list[float]) -> float:
    if not sample_a or not sample_b:
        return 0.0
    mean_a, mean_b = _mean(sample_a), _mean(sample_b)
    var_a, var_b = _variance(sample_a), _variance(sample_b)
    denom = max(1, len(sample_a) + len(sample_b) - 2)
    pooled = ((len(sample_a) - 1) * var_a + (len(sample_b) - 1) * var_b) / denom
    if pooled <= 0:
        return 0.0
    return (mean_a - mean_b) / (pooled ** 0.5)


class AlignmentAnalyzer:
    """Analyze xi dynamics around a pressure onset index."""

    def __init__(
        self,
        xi_series: list[float],
        pressure_onset: int,
        lvs_series: list[float] | None = None,
        pt_series: list[float] | None = None,
        behavioral_onset: int | None = None,
        behavioral_code: str | None = None,
    ) -> None:
        self.xi = xi_series
        self.pressure_onset = pressure_onset
        self.lvs = lvs_series if lvs_series is not None else [0.0] * len(xi_series)
        self.pt = pt_series if pt_series is not None else [0.0] * len(xi_series)
        self.behavioral_onset = behavioral_onset
        self.behavioral_code = behavioral_code

        self._pre_pressure_xi = self.xi[:pressure_onset]
        self._crisis_xi = self.xi[pressure_onset:]

    def detect_xi_compression(self, window: int = 2) -> CompressionEvent | None:
        """Detect post-pressure compression after an initial xi spike."""
        if self.pressure_onset <= 0 or len(self._crisis_xi) < window:
            return None

        baseline = _mean(self._pre_pressure_xi)
        spike_threshold = baseline + 0.2
        compression_threshold = baseline + 0.05

        spike_seen = False
        for idx, value in enumerate(self._crisis_xi):
            if value >= spike_threshold:
                spike_seen = True
            if spike_seen and idx + window <= len(self._crisis_xi):
                run = self._crisis_xi[idx : idx + window]
                if all(v <= compression_threshold for v in run):
                    onset = self.pressure_onset + idx
                    magnitude = max(0.0, max(self._crisis_xi[: idx + 1]) - _mean(run))
                    return CompressionEvent(onset_turn=onset, duration=window, magnitude=magnitude)
        return None

    def pre_behavioral_lag(self) -> int | None:
        """Return lag between compression onset and behavioral onset."""
        if self.behavioral_onset is None:
            return None
        compression = self.detect_xi_compression()
        if compression is None:
            return None
        return self.behavioral_onset - compression.onset_turn

    def crisis_window_profile(self) -> CrisisProfile:
        """Summarize trajectory shape in the crisis window."""
        crisis_xi = self._crisis_xi if self._crisis_xi else self.xi
        max_xi = max(crisis_xi) if crisis_xi else 0.0
        xi_var = _variance(crisis_xi)

        if max_xi >= 0.85 and crisis_xi and crisis_xi[-1] >= 0.7:
            trajectory = "sustained_high"
        elif max_xi >= 0.8 and crisis_xi and crisis_xi[-1] < max_xi - 0.2:
            trajectory = "spike_resolve"
        else:
            trajectory = "stable_low"

        crisis_pt = self.pt[self.pressure_onset :] if self.pt else []
        if len(crisis_pt) >= 2 and crisis_pt[-1] < crisis_pt[0]:
            pt_trend = "falling"
        elif len(crisis_pt) >= 2 and crisis_pt[-1] > crisis_pt[0]:
            pt_trend = "rising"
        else:
            pt_trend = "stable"

        return CrisisProfile(
            xi_trajectory=trajectory,
            pt_trend=pt_trend,
            max_xi=max_xi,
            xi_variance=xi_var,
            window_start=self.pressure_onset,
            window_end=len(self.xi) - 1,
        )

    def option_e_profile(self) -> OptionEProfile:
        """Classify option-E profile from code and crisis trajectory."""
        if self.behavioral_code != "E":
            return OptionEProfile(False, "not_option_e")
        profile = self.crisis_window_profile()
        if profile.xi_trajectory == "stable_low":
            return OptionEProfile(True, "stable_creative")
        return OptionEProfile(True, "masked_tension")

    def to_dict(self) -> dict[str, Any]:
        """Serialize analyzer summaries to a JSON-safe dictionary."""
        return {
            "pressure_onset": self.pressure_onset,
            "behavioral_onset": self.behavioral_onset,
            "behavioral_code": self.behavioral_code,
            "compression": (
                asdict(event) if (event := self.detect_xi_compression()) is not None else None
            ),
            "crisis_profile": asdict(self.crisis_window_profile()),
            "option_e_profile": asdict(self.option_e_profile()),
        }

    def export_json(self, out_path: str) -> str:
        """Write serialized analysis to a JSON artifact."""
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return str(path)


def compare_conditions(
    analyses: dict[str, AlignmentAnalyzer],
) -> list[ConditionComparison]:
    """Compute pairwise condition comparisons for key summary metrics."""
    comparisons: list[ConditionComparison] = []
    for name_a, name_b in combinations(sorted(analyses.keys()), 2):
        a, b = analyses[name_a], analyses[name_b]
        metric = "max_xi"
        values_a = a._crisis_xi
        values_b = b._crisis_xi
        delta = a.crisis_window_profile().max_xi - b.crisis_window_profile().max_xi
        comparisons.append(
            ConditionComparison(
                condition_a=name_a,
                condition_b=name_b,
                metric=metric,
                delta=delta,
                effect_size=_cohens_d(values_a, values_b),
            )
        )
    return comparisons


def batch_analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Run alignment analysis across many records and aggregate counts."""
    if not records:
        return {
            "count": 0,
            "behavioral_code_counts": {},
            "compression_rate": 0.0,
        }

    analyzers = [
        AlignmentAnalyzer(
            xi_series=r["xi"],
            lvs_series=r.get("lvs"),
            pt_series=r.get("pt"),
            pressure_onset=r["pressure_onset"],
            behavioral_onset=r.get("behavioral_onset"),
            behavioral_code=r.get("behavioral_code"),
        )
        for r in records
    ]

    code_counts: dict[str, int] = {}
    compressed = 0
    for analyzer in analyzers:
        code = analyzer.behavioral_code
        if code:
            code_counts[code] = code_counts.get(code, 0) + 1
        if analyzer.detect_xi_compression() is not None:
            compressed += 1

    return {
        "count": len(analyzers),
        "behavioral_code_counts": code_counts,
        "compression_rate": compressed / len(analyzers),
    }
