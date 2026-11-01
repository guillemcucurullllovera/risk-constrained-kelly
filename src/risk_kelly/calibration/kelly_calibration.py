"""Fractional-Kelly coefficient calibration (F-26).

The Kelly criterion gives the theoretically optimal fraction of bankroll to
risk per trade.  In practice:
  1. Full Kelly produces large drawdowns (log-utility maximizer is risk-neutral
     about ruin in the short term; operators are not).
  2. The correct fraction of Kelly = f that achieves a target ruin probability.

Formula (continuous approximation from Kelly/Vince):
  P(ruin) ≈ exp(−2 × f × μ / σ²) for small μ/σ

Solving for f:
  f = −ln(P_ruin_target) × σ² / (2 × μ)

Where:
  μ = mean OOS return per trade (from walk-forward runs)
  σ² = variance of OOS returns per trade
  P_ruin_target = operator-set target ruin probability (e.g. 0.01 = 1 in 100 years)

Output:
  Fractional-Kelly coefficient in (0, 1) — validated, ready for signed config.
  The output is a RECOMMENDATION — the operator reviews and signs before committing.

Notes:
  - This is not a free optimization: the fraction is determined by risk tolerance
    (ruin probability target) and observed strategy variance — not by fitting to
    historical data.
  - The ruin-probability target is governance-determined (Francisco's call).
  - If the OOS returns show insufficient edge (μ ≤ 0), Kelly returns 0 (no trade).
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence


class KellyCalibrationError(ValueError):
    """Kelly calibration cannot produce a valid fraction — see message for reason."""


@dataclass(frozen=True)
class KellyCalibrationResult:
    """Output of compute_kelly_fraction().

    fraction: the recommended fractional-Kelly coefficient, in (0, 1).
    mean_oos_return: sample mean of the OOS return series.
    variance_oos_return: sample variance of the OOS return series.
    target_ruin_probability: the target P(ruin) that was solved against.
    implied_ruin_probability: the actual P(ruin) at the returned fraction
        (may differ slightly from target due to Decimal rounding).
    n_observations: number of OOS return observations used.
    warning: optional warning if the fraction was clamped or the edge is thin.
    """
    fraction: Decimal
    mean_oos_return: Decimal
    variance_oos_return: Decimal
    target_ruin_probability: float
    implied_ruin_probability: float
    n_observations: int
    warning: str = ""


# ---------------------------------------------------------------------------
# Core formula
# ---------------------------------------------------------------------------

def compute_kelly_fraction(
    oos_returns: Sequence[Decimal],
    target_ruin_probability: float = 0.01,
    max_fraction: Decimal = Decimal("0.50"),
    min_observations: int = 20,
) -> KellyCalibrationResult:
    """Compute the fractional-Kelly coefficient from walk-forward OOS returns.

    Args:
        oos_returns: per-trade OOS net returns (net of transaction costs per
            WalkForwardResult.oos_returns). Must have honest costs applied.
        target_ruin_probability: target P(ruin) — e.g. 0.01 means 1% chance of
            ruin, which typically maps to a ~100-year horizon for long-only
            strategies. This is a governance decision (operator-set), not data.
        max_fraction: hard ceiling on the returned fraction. Default 0.50
            (half-Kelly maximum). Prevents over-sizing from lucky OOS samples.
        min_observations: minimum OOS trades for reliable estimation.
            Raises KellyCalibrationError if fewer trades observed.

    Returns:
        KellyCalibrationResult with the recommended fraction and diagnostics.

    Raises:
        KellyCalibrationError: if returns are insufficient, edge is negative,
            or target_ruin_probability is not in (0, 1).
    """
    if not 0.0 < target_ruin_probability < 1.0:
        raise KellyCalibrationError(
            f"target_ruin_probability {target_ruin_probability} must be in (0, 1)"
        )
    if len(oos_returns) < min_observations:
        raise KellyCalibrationError(
            f"Need at least {min_observations} OOS return observations; "
            f"got {len(oos_returns)}. Run more walk-forward windows."
        )

    floats = [float(r) for r in oos_returns]
    mu = statistics.mean(floats)
    if len(floats) >= 2:
        var = statistics.variance(floats)
    else:
        var = 0.0

    n = len(floats)
    mean_dec = Decimal(str(round(mu, 8)))
    var_dec = Decimal(str(round(var, 8)))

    # Negative or zero edge → no trade (Kelly = 0)
    if mu <= 0.0:
        return KellyCalibrationResult(
            fraction=Decimal("0"),
            mean_oos_return=mean_dec,
            variance_oos_return=var_dec,
            target_ruin_probability=target_ruin_probability,
            implied_ruin_probability=1.0,
            n_observations=n,
            warning=(
                "Non-positive mean OOS return — strategy has no measurable edge. "
                "Kelly fraction is 0 (no trade). Review strategy parameters."
            ),
        )

    if var <= 0.0:
        raise KellyCalibrationError(
            "OOS return variance is zero — all returns are identical. "
            "This indicates a data problem (non-representative or too-few OOS windows)."
        )

    # f = −ln(P_ruin) × σ² / (2 × μ)
    raw_f = (-math.log(target_ruin_probability) * var) / (2.0 * mu)
    raw_f_dec = Decimal(str(round(raw_f, 6)))

    warning = ""
    if raw_f_dec >= max_fraction:
        raw_f_dec = max_fraction
        warning = (
            f"Raw Kelly fraction {round(raw_f, 4)} exceeds max_fraction cap {max_fraction}. "
            "Clamped. Consider reviewing strategy edge or increasing the ruin target."
        )
    elif raw_f_dec <= Decimal("0"):
        raw_f_dec = Decimal("0")
        warning = "Computed fraction is non-positive — no trade recommended."

    # Verify fraction ∈ (0, 1) before committing
    if raw_f_dec > Decimal("1"):
        raw_f_dec = Decimal("0.5")
        warning = "Computed fraction > 1 (full Kelly) — clamped to 0.5 (half-Kelly safety cap)."

    # Implied ruin probability at the returned fraction
    if float(raw_f_dec) > 0 and mu > 0 and var > 0:
        implied_ruin = math.exp(-2.0 * float(raw_f_dec) * mu / var)
    else:
        implied_ruin = 1.0

    return KellyCalibrationResult(
        fraction=raw_f_dec,
        mean_oos_return=mean_dec,
        variance_oos_return=var_dec,
        target_ruin_probability=target_ruin_probability,
        implied_ruin_probability=implied_ruin,
        n_observations=n,
        warning=warning,
    )


def compute_kelly_fraction_discrete(
    oos_returns: Sequence[Decimal],
    half_kelly: bool = True,
    max_fraction: Decimal = Decimal("0.25"),
    min_observations: int = 20,
) -> KellyCalibrationResult:
    """Compute Kelly fraction via numerical bisection for heavy-tailed strategies.

    The continuous-diffusion approximation (compute_kelly_fraction) breaks down
    when the return distribution is highly non-normal — e.g. put spreads, which
    have ~90% small wins and occasional large losses. This function solves the
    true discrete Kelly optimality condition:

      E[r / (1 + f × r)] = 0   →   f = argmax E[log(1 + f × r)]

    via bisection, then applies half-Kelly (f / 2) for operational prudence.

    Args:
        oos_returns:    per-trade OOS returns (return on risk capital).
        half_kelly:     if True (default), return f* / 2 (conservative sizing).
        max_fraction:   hard ceiling (default 0.25 = half-Kelly cap for discretionary).
        min_observations: minimum trades for reliable estimation.

    Returns:
        KellyCalibrationResult with the discrete Kelly fraction.

    Raises:
        KellyCalibrationError: if returns are insufficient or have no positive edge.
    """
    if len(oos_returns) < min_observations:
        raise KellyCalibrationError(
            f"Need at least {min_observations} OOS return observations; "
            f"got {len(oos_returns)}. Run more walk-forward windows."
        )

    floats = [float(r) for r in oos_returns]
    mu = statistics.mean(floats)
    var = statistics.variance(floats) if len(floats) >= 2 else 0.0
    n = len(floats)
    mean_dec = Decimal(str(round(mu, 8)))
    var_dec = Decimal(str(round(var, 8)))

    if mu <= 0.0:
        return KellyCalibrationResult(
            fraction=Decimal("0"),
            mean_oos_return=mean_dec,
            variance_oos_return=var_dec,
            target_ruin_probability=0.01,
            implied_ruin_probability=1.0,
            n_observations=n,
            warning=(
                "Non-positive mean OOS return — strategy has no measurable edge. "
                "Kelly fraction is 0 (no trade)."
            ),
        )

    if var <= 0.0:
        raise KellyCalibrationError(
            "OOS return variance is zero — all returns identical. Data problem."
        )

    # Bisection on g(f) = E[r / (1 + f × r)] = 0
    min_r = min(floats)
    if min_r >= 0.0:
        raise KellyCalibrationError(
            "All OOS returns are non-negative — no variance to calibrate against."
        )
    f_upper = min(0.9999, 1.0 / abs(min_r) * 0.9999)

    def g(f: float) -> float:
        return sum(r / (1.0 + f * r) for r in floats) / n

    g_lo = g(1e-10)
    g_hi = g(f_upper)
    if g_lo <= 0.0:
        return KellyCalibrationResult(
            fraction=Decimal("0"),
            mean_oos_return=mean_dec,
            variance_oos_return=var_dec,
            target_ruin_probability=0.01,
            implied_ruin_probability=1.0,
            n_observations=n,
            warning="Discrete Kelly has no solution > 0 (edge insufficient relative to loss tail).",
        )

    lo, hi = 1e-10, f_upper
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if g(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    full_kelly = (lo + hi) / 2.0

    f_result = full_kelly / 2.0 if half_kelly else full_kelly
    f_dec = Decimal(str(round(min(f_result, float(max_fraction)), 6)))

    warning = (
        f"Discrete Kelly (bisection, half-Kelly={'yes' if half_kelly else 'no'}). "
        "Continuous ruin-probability approximation is not valid for put-spread "
        "return distributions; implied_ruin_probability is a rough estimate only."
    )
    if f_dec >= max_fraction:
        warning += f" Clamped to max_fraction={max_fraction}."

    # Approximate implied ruin for reporting (continuous formula, caveat noted in warning)
    if float(f_dec) > 0 and mu > 0 and var > 0:
        implied_ruin = math.exp(-2.0 * float(f_dec) * mu / var)
    else:
        implied_ruin = 1.0

    return KellyCalibrationResult(
        fraction=f_dec,
        mean_oos_return=mean_dec,
        variance_oos_return=var_dec,
        target_ruin_probability=0.01,
        implied_ruin_probability=implied_ruin,
        n_observations=n,
        warning=warning,
    )


def validate_kelly_result(result: KellyCalibrationResult) -> None:
    """Raise KellyCalibrationError if the result is not suitable for signing.

    Checks:
      - fraction must be in (0, 0.50] — never full Kelly, never zero in production
      - implied_ruin_probability ≤ 2 × target (within acceptable tolerance)
      - no active warning (warnings indicate the result needs review)
    """
    if result.fraction <= Decimal("0"):
        raise KellyCalibrationError(
            "fraction is 0 — strategy has no edge; do not commit to signed config"
        )
    if result.fraction > Decimal("0.50"):
        raise KellyCalibrationError(
            f"fraction {result.fraction} exceeds 0.50 — never commit a fraction "
            "above half-Kelly to signed config (tail risk too high)"
        )
    if result.implied_ruin_probability > 2.0 * result.target_ruin_probability:
        raise KellyCalibrationError(
            f"implied ruin probability {result.implied_ruin_probability:.4f} is more than "
            f"2x the target {result.target_ruin_probability} — recalibrate"
        )
    if result.warning:
        raise KellyCalibrationError(
            f"Kelly result has an unresolved warning — review before signing: {result.warning}"
        )
