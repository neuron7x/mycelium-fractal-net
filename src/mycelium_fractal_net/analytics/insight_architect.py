"""
Fractal Insight Architect v2.0

Transforms semi-structured observations into actionable, multi-scale insights
that follow the template described in the problem statement. The implementation
is intentionally lightweight and deterministic so it can run inside automated
tests without external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


class InsufficientDataError(ValueError):
    """
    Raised when provided observations cannot produce a full micro/meso/macro
    stack. Includes up to ``max_clarifications`` follow-up questions.
    """

    def __init__(self, clarifications: list[str]):
        super().__init__("insufficient data for insight generation")
        self.clarifications = clarifications


@dataclass
class LevelPattern:
    """Normalized representation of a pattern at a given scale."""

    name: str
    metric: float | None = None
    evidence: str | None = None


@dataclass
class Insight:
    """Structured insight payload."""

    name: str
    summary: str
    micro: str
    meso: str
    macro: str
    invariant: str
    steps: list[str]
    validation: str
    boundaries: str

    def format(self) -> str:
        """Render the insight using the mandated template."""
        lines = [
            f"**[{self.name}]**: {self.summary}",
            "",
            "**Фрактальна структура**:",
            f"- **Мікро**: {self.micro}",
            f"- **Мезо**: {self.meso}",
            f"- **Макро**: {self.macro}",
            "",
            f"**Інваріант**: {self.invariant}",
            "",
            "**Операційні кроки**:",
        ]
        lines.extend(self.steps)
        lines.extend(
            [
                "",
                f"**Валідація**: {self.validation}",
                f"**Межі**: {self.boundaries}",
            ]
        )
        return "\n".join(lines)


@dataclass
class _NormalizedData:
    levels: dict[str, list[LevelPattern]]
    tensions: list[str]
    goal: str | None


class FractalInsightArchitect:
    """
    Deterministic insight generator that follows the "ФРАКТАЛЬНИЙ ІНСАЙТ-АРХІТЕКТОР v2.0"
    protocol from the problem statement.
    """

    def __init__(self, max_clarifications: int = 3):
        self.max_clarifications = max(1, min(max_clarifications, 3))

    # Public API -----------------------------------------------------------------
    def generate(self, data: Mapping[str, Any] | Sequence[Mapping[str, Any]], *, principle_name: str | None = None) -> Insight:
        """
        Generate an insight. Expects either a mapping with keys ``micro``,
        ``meso``, ``macro`` or a list of mappings each containing ``level``.
        """
        normalized = self._normalize(data)
        missing_levels = [lvl for lvl in ("micro", "meso", "macro") if not normalized.levels[lvl]]
        if missing_levels:
            raise InsufficientDataError(self._build_clarifications(missing_levels))

        metrics = [p.metric for items in normalized.levels.values() for p in items if p.metric is not None]
        invariant_threshold = self._compute_threshold(metrics)

        name = self._build_name(principle_name, normalized)
        summary = self._build_summary(normalized)
        micro_text = self._describe_level("Мікро", normalized.levels["micro"])
        meso_text = self._describe_level("Мезо", normalized.levels["meso"])
        macro_text = self._describe_level("Макро", normalized.levels["macro"])
        invariant = self._build_invariant(invariant_threshold)
        steps = self._build_steps(normalized, invariant_threshold)
        validation = self._build_validation()
        boundaries = self._build_boundaries()

        return Insight(
            name=name,
            summary=summary,
            micro=micro_text,
            meso=meso_text,
            macro=macro_text,
            invariant=invariant,
            steps=steps,
            validation=validation,
            boundaries=boundaries,
        )

    def format_insight(self, data: Mapping[str, Any] | Sequence[Mapping[str, Any]], *, principle_name: str | None = None) -> str:
        """Generate and return a formatted string insight."""
        return self.generate(data, principle_name=principle_name).format()

    # Normalization --------------------------------------------------------------
    def _normalize(self, data: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> _NormalizedData:
        levels = {"micro": [], "meso": [], "macro": []}
        tensions: list[str] = []
        goal = None

        if isinstance(data, Mapping):
            goal = data.get("goal")
            tensions = list(data.get("tensions", []) or data.get("tension_points", []))
            for lvl in levels:
                levels[lvl] = self._normalize_entries(data.get(lvl, []))
        else:
            for item in data:
                if not isinstance(item, Mapping):
                    continue
                lvl = str(item.get("level", "")).lower()
                if lvl in levels:
                    levels[lvl].extend(self._normalize_entries([item]))
                if item.get("tension"):
                    tensions.append(str(item["tension"]))
            goal = None

        return _NormalizedData(levels=levels, tensions=tensions, goal=goal)

    def _normalize_entries(self, entries: Iterable[Any]) -> list[LevelPattern]:
        normalized: list[LevelPattern] = []
        for entry in entries:
            if isinstance(entry, str):
                normalized.append(LevelPattern(name=entry))
                continue
            if isinstance(entry, Mapping):
                name = (
                    entry.get("pattern")
                    or entry.get("name")
                    or entry.get("description")
                    or entry.get("signal")
                )
                if not name:
                    continue
                metric = entry.get("metric")
                if metric is None and isinstance(entry.get("value"), (int, float)):
                    metric = float(entry["value"])
                evidence = entry.get("evidence") or entry.get("example")
                normalized.append(LevelPattern(name=str(name), metric=self._safe_float(metric), evidence=evidence))
        return normalized

    # Builders -------------------------------------------------------------------
    def _build_clarifications(self, missing_levels: list[str]) -> list[str]:
        templates = {
            "micro": "Надайте базові мікропатерни (правила/сигнали) та їхні метрики.",
            "meso": "Опишіть взаємодії між елементами на мезорівні та рівень узгодженості.",
            "macro": "Додайте системні наслідки на макрорівні з вимірюваними метриками.",
        }
        clarifications = [templates[lvl] for lvl in missing_levels]
        return clarifications[: self.max_clarifications]

    def _build_name(self, principle_name: str | None, normalized: _NormalizedData) -> str:
        if principle_name:
            return principle_name.strip().upper()
        micro = normalized.levels["micro"][0].name
        macro = normalized.levels["macro"][0].name
        return f"{micro} → {macro}".upper()

    def _build_summary(self, normalized: _NormalizedData) -> str:
        macro = normalized.levels["macro"][0].name
        tension = normalized.tensions[0] if normalized.tensions else "каскадне посилення"
        goal = normalized.goal or "стабілізувати систему"
        return f"{tension} призводить до {macro}; мета — {goal}."

    def _describe_level(self, label: str, patterns: list[LevelPattern]) -> str:
        primary = patterns[0]
        metric_text = f" (метрика: {primary.metric:.3f})" if primary.metric is not None else ""
        example = f" (приклад: {primary.evidence})" if primary.evidence else ""
        extra = f"; повторюваність {len(patterns)}x" if len(patterns) > 1 else ""
        return f"{label.lower()}-правило — {primary.name}{metric_text}{example}{extra}."

    def _build_invariant(self, threshold: float) -> str:
        return f"Стійкий при змінах >20%; поріг чутливості {threshold:.3f} для ключових метрик."

    def _build_steps(self, normalized: _NormalizedData, threshold: float) -> list[str]:
        micro_metric = self._first_metric(normalized.levels["micro"])
        meso_metric = self._first_metric(normalized.levels["meso"])
        goal = normalized.goal or "результат"
        return [
            f"1. Зняти базову мікрометріку ({self._format_metric(micro_metric)}), охоплення ≥80%.",
            f"2. Синхронізувати мезорівень через A/B-тест (ціль: -50% варіації; базово {self._format_metric(meso_metric)}).",
            f"3. Моніторити макропоказники щотижня; очікуване покращення {goal} на 15% упродовж 1-2 тижнів; реагувати при відхиленні >{threshold:.3f}.",
        ]

    def _build_validation(self) -> str:
        return "A/B-тест + контрольна група; контрприклад: якщо зміни випадкові та не масштабуються, гіпотеза відкидається."

    def _build_boundaries(self) -> str:
        return "Не працює в хаотичних системах із випадковістю >70%; сигнал ризику — відсутність повторюваних патернів."

    # Utilities ------------------------------------------------------------------
    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _compute_threshold(metrics: list[float]) -> float:
        if not metrics:
            return 0.1
        max_metric = max(abs(m) for m in metrics if m is not None) or 1.0
        return round(max_metric * 0.2, 3)

    @staticmethod
    def _first_metric(patterns: list[LevelPattern]) -> float | None:
        for p in patterns:
            if p.metric is not None:
                return p.metric
        return None

    @staticmethod
    def _format_metric(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "n/a"
