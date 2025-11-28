"""
Custom exceptions for numerical stability control.

These exceptions provide clear error signaling when numerical computations
fail due to instability, value range violations, or NaN/Inf occurrences.

Reference: docs/ARCHITECTURE.md Section 2 (stability requirements)
"""

from __future__ import annotations


class StabilityError(Exception):
    """
    Base exception for numerical stability issues.

    Raised when a numerical computation produces unstable results,
    such as diverging values or oscillations.

    Attributes:
        message: Description of the stability issue.
        step: Integration step at which instability occurred (optional).
        value: The problematic value that triggered the error (optional).
    """

    def __init__(
        self,
        message: str,
        step: int | None = None,
        value: float | None = None,
    ) -> None:
        self.step = step
        self.value = value
        details = []
        if step is not None:
            details.append(f"step={step}")
        if value is not None:
            details.append(f"value={value:.6g}")
        detail_str = f" ({', '.join(details)})" if details else ""
        super().__init__(f"{message}{detail_str}")


class ValueOutOfRangeError(StabilityError):
    """
    Exception for values exceeding physically/logically allowed ranges.

    Raised when computed values exceed predefined bounds, such as:
    - Membrane potentials outside [-95, +40] mV
    - Field densities outside [0, 1]
    - Negative concentrations

    Attributes:
        message: Description of the range violation.
        value: The out-of-range value.
        min_bound: Minimum allowed value (optional).
        max_bound: Maximum allowed value (optional).
    """

    def __init__(
        self,
        message: str,
        value: float,
        min_bound: float | None = None,
        max_bound: float | None = None,
        step: int | None = None,
    ) -> None:
        self.min_bound = min_bound
        self.max_bound = max_bound
        bounds_str = ""
        if min_bound is not None and max_bound is not None:
            bounds_str = f", expected [{min_bound:.6g}, {max_bound:.6g}]"
        elif min_bound is not None:
            bounds_str = f", min={min_bound:.6g}"
        elif max_bound is not None:
            bounds_str = f", max={max_bound:.6g}"
        super().__init__(f"{message}: value={value:.6g}{bounds_str}", step=step, value=value)


class NumericalInstabilityError(StabilityError):
    """
    Exception for NaN or Infinity occurrences.

    Raised when NaN or Inf values appear in numerical computations,
    indicating a severe numerical breakdown requiring immediate action.

    Attributes:
        message: Description of the instability.
        field_name: Name of the field/variable with NaN/Inf (optional).
        nan_count: Number of NaN values found (optional).
        inf_count: Number of Inf values found (optional).
    """

    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        nan_count: int | None = None,
        inf_count: int | None = None,
        step: int | None = None,
    ) -> None:
        self.field_name = field_name
        self.nan_count = nan_count
        self.inf_count = inf_count
        details = []
        if field_name:
            details.append(f"field={field_name}")
        if nan_count is not None:
            details.append(f"NaN={nan_count}")
        if inf_count is not None:
            details.append(f"Inf={inf_count}")
        detail_str = f" ({', '.join(details)})" if details else ""
        super().__init__(f"{message}{detail_str}", step=step)
