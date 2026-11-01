"""sizing_ceiling — the Sinclair operator-discipline CEILING that wraps RCK (R3).

Provenance: clean-room reimplementation from Euan Sinclair, *Volatility Trading*
(2nd ed.), Ch. 8 (money management), plus the estimation-error literature it draws
on — Chopra & Ziemba (1993, "The effect of errors in means, variances and
covariances on optimal portfolio choice"), Thorp/MacLean/Ziemba (fractional Kelly,
growth-security), and the OU-process optimal-betting result. Patterns and equations
only; no code copied. MIT/Apache-clean by construction (pure python + math).

------------------------------------------------------------------------------
WHAT THIS IS (and is NOT)
------------------------------------------------------------------------------
A *ceiling* layer. RCK (Rockafellar-Uryasev / Busseti-Ryu-Boyd drawdown-constrained
Kelly) sizes the bet on a scenario set; THIS module clamps that size from above with
operator discipline the mean-variance optimizer cannot see:

    shrunk fraction-of-Kelly ceiling
  + Chopra-Ziemba mean haircut (deflate E[r] BEFORE the optimizer sees it)
  + small-sample / negative-payoff-skew shrink
  + OU mean-reversion scaling cap (peak ~sqrt(2) SD, REDUCE beyond)
  + regime-change kill switch

It is ADVISORY / research-only. It certifies NO edge (CPCV + purged folds + deflated
Sharpe stay upstream) and emits NO live order. It sizes a bet; it never places one.

------------------------------------------------------------------------------
METHODS-NOT-NUMBERS (R3 rail b)
------------------------------------------------------------------------------
Every risk knob — kelly_fraction, mean_shrink, estimation_coef, skew_coef,
mr_scale_coef — is an UNSIGNED explicit caller argument (fed from the trial registry;
signed only at T8 / §5). This module hard-codes NO §5 risk constant. The one default,
`sd_cap = sqrt(2)`, is the THEORETICAL peak of the OU optimal-holding envelope (a
structural math constant, not a tuned risk parameter), and is overridable.

------------------------------------------------------------------------------
OPEN TENSION — SURFACED, NOT SILENTLY STACKED (R3 rail h)
------------------------------------------------------------------------------
Feeding RCK a deflated mean AND constraining it with lambda both add conservatism.
Stacking them blindly double-counts. `apply_ceiling` therefore RECORDS every haircut
it applies in `CeilingResult.applied_haircuts` (name, multiplicative factor) so the
mean-haircut-vs-lambda double-conservatism is inspectable and can be set JOINTLY,
not discovered after the fact. Callers must reconcile the two, not layer them by reflex.

Governance note (R3 rail i): Sleeve C's sacred clock-driven roll may warrant its own
lambda / ceiling exemption — that is an OPEN governance question for §5/T8, not
decided here.

------------------------------------------------------------------------------
BUG NOTE — DO NOT PROPAGATE kelly.py:73
------------------------------------------------------------------------------
heimdall/core/sizing/kelly.py:73-76 carries a pre-PA-030 DOUBLE-COUNT: it divides by
max_loss_per_unit at line 73 (dollars) and AGAIN at line 76 (contracts). The
continuous-Kelly ceiling here divides by the VARIANCE exactly once (f* = mu/sigma2) —
a single inverse-variance scaling. A regression test asserts that doubling sigma2
HALVES the ceiling (not quarters it), which a double-divide would fail.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


class SizingCeilingError(ValueError):
    """Fail-closed STOP: non-finite input, sigma2<=0, a fraction outside (0,1),
    a non-positive wealth/sd_cap, or a negative haircut coefficient. Never fall
    through with a bad size."""


# ── value objects ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CeilingInputs:
    """Estimator moments behind a sizing decision.

    mu       : excess (over risk-free) expected return of the sleeve. RAW in-sample
               unless the caller pre-deflated it; `apply_ceiling` deflates internally
               via the Chopra-Ziemba haircut, so pass the naive estimate here.
    sigma2   : return variance (> 0).
    skew     : payoff skew (negative for the short-convexity sleeve).
    sample_n : number of observations behind the estimate (>= 1).
    """
    mu: float
    sigma2: float
    skew: float
    sample_n: int


@dataclass(frozen=True)
class CeilingResult:
    f_ceiling: float
    applied_haircuts: tuple[tuple[str, float], ...]  # ordered (name, factor) — surfaced
    killed: bool
    reason: str


# ── validation helpers ────────────────────────────────────────────────────
def _finite(name: str, x: float) -> float:
    if not math.isfinite(x):
        raise SizingCeilingError(f"{name} must be finite; got {x!r}")
    return x


def _fraction_open_unit(name: str, x: float) -> float:
    """(0, 1) exclusive — never full Kelly, never zero. Mirrors kelly.py contract."""
    _finite(name, x)
    if not (0.0 < x < 1.0):
        raise SizingCeilingError(
            f"{name} {x} must be strictly in (0, 1) — never full Kelly, never zero"
        )
    return x


def _shrink_unit(name: str, x: float) -> float:
    """[0, 1) — 0 is the identity (no shrink; caller owns that choice), 1 would zero
    the mean (kill the edge) so it is refused."""
    _finite(name, x)
    if not (0.0 <= x < 1.0):
        raise SizingCeilingError(f"{name} {x} must be in [0, 1)")
    return x


def _nonneg_coef(name: str, x: float) -> float:
    _finite(name, x)
    if x < 0.0:
        raise SizingCeilingError(f"{name} {x} must be >= 0")
    return x


# ── (1) continuous-time full-Kelly — the HARD upper bound the book never touches ──
def continuous_kelly_ceiling(mu: float, sigma2: float) -> float:
    """Full continuous-time Kelly leverage f* = mu / sigma2 (mu = excess return).

    This is the growth-optimal bet and the ABSOLUTE ceiling: the book always sizes a
    FRACTION of it. Growth g(f) = mu*f - 0.5*sigma2*f^2 is symmetric about f* (see
    `growth_rate`), so over- and under-betting by the same amount cost the same
    growth — but overbetting adds pure variance with no growth gain, hence
    underbetting dominates overbetting under estimation uncertainty.

    Single inverse-variance scaling (NOT the kelly.py:73 double divide).
    """
    _finite("mu", mu)
    _finite("sigma2", sigma2)
    if sigma2 <= 0.0:
        raise SizingCeilingError(f"sigma2 {sigma2} must be > 0 (fail-closed)")
    return mu / sigma2


def growth_rate(f: float, mu: float, sigma2: float) -> float:
    """Log-growth of a bet at fraction f: g(f) = mu*f - 0.5*sigma2*f^2. Maximised at
    f* = mu/sigma2; symmetric penalty either side (demonstrates the ceiling rationale)."""
    _finite("f", f)
    _finite("mu", mu)
    _finite("sigma2", sigma2)
    if sigma2 <= 0.0:
        raise SizingCeilingError(f"sigma2 {sigma2} must be > 0 (fail-closed)")
    return mu * f - 0.5 * sigma2 * f * f


# ── (2) Chopra-Ziemba mean haircut — deflate E[r] BEFORE the optimizer sees it ──
def mean_haircut_chopra_ziemba(mu_raw: float, shrink: float) -> float:
    """Return the deflated mean (1 - shrink) * mu_raw.

    Errors in means : variances : covariances hurt the optimizer ~ 20 : 2 : 1
    (Chopra-Ziemba 1993): a biased E[r] makes the optimizer CONFIDENTLY oversize.
    So RCK must consume a deflated / Bayesian-posterior mean, NEVER the naive
    in-sample estimate — estimated parameters systematically overbet (Medo-Pismak-
    Zhang). shrink = 0.5 is the "poor man's Bayesian" half-Kelly: average the
    measured edge with the zero-edge hypothesis. Monotone-decreasing in shrink.
    """
    _finite("mu_raw", mu_raw)
    _shrink_unit("shrink", shrink)
    return (1.0 - shrink) * mu_raw


# ── (3) small-sample + negative-skew shrink ────────────────────────────────
def shrink_for_estimation_and_skew(
    f: float,
    sample_n: int,
    skew: float,
    *,
    estimation_coef: float,
    skew_coef: float,
) -> float:
    """Shrink a fraction f for (a) small-sample uncertainty in f (std(f) ~ 1/sqrt(n))
    and (b) negative payoff skew.

        estimation_factor = clamp01(1 - estimation_coef / sqrt(n))
        skew_factor       = clamp01(1 - skew_coef * max(0, -skew))
        f_out             = f * estimation_factor * skew_factor

    Less data -> larger haircut. Negative skew (short-convexity) -> larger haircut,
    because a mean-variance Kelly UNDERSTATES ruin when the payoff is left-skewed
    (the blow-up tail is fatter than the variance implies). Favourable (positive)
    skew earns no extra cut. Both coefficients are UNSIGNED caller args.
    """
    _finite("f", f)
    if not isinstance(sample_n, int) or sample_n < 1:
        raise SizingCeilingError(f"sample_n {sample_n!r} must be an int >= 1")
    _finite("skew", skew)
    _nonneg_coef("estimation_coef", estimation_coef)
    _nonneg_coef("skew_coef", skew_coef)

    estimation_factor = 1.0 - estimation_coef / math.sqrt(sample_n)
    downside_skew = max(0.0, -skew)
    skew_factor = 1.0 - skew_coef * downside_skew

    estimation_factor = min(1.0, max(0.0, estimation_factor))
    skew_factor = min(1.0, max(0.0, skew_factor))
    return f * estimation_factor * skew_factor


# ── (4) OU mean-reversion scaling cap — peak ~sqrt(2) SD, REDUCE beyond ─────
def mr_scaling_cap(
    deviation_sd: float,
    wealth: float,
    *,
    scale_coef: float,
    sd_cap: float = math.sqrt(2),
) -> float:
    """Optimal Ornstein-Uhlenbeck (mean-reverting) holding, envelope-capped.

        position = -scale_coef * wealth * z * exp( -z^2 / (2 * sd_cap^2) )

    where z = deviation_sd (signed, in SD units). The magnitude rises with the
    deviation, PEAKS at |z| = sd_cap (default sqrt(2), the theoretical OU peak), then
    DECREASES beyond it: past the cap the wealth you have already lost (mark-to-market)
    dominates the improving edge, so you REDUCE, not add — Sinclair's "when in trouble,
    double, but only to a point". The sign is contrarian (above the mean -> short).

    Proebsting's-paradox guard: `wealth` MUST be the current mark-to-market wealth, not
    entry wealth — resizing on entry wealth chases the position to unbounded leverage.
    """
    _finite("deviation_sd", deviation_sd)
    _finite("wealth", wealth)
    _finite("sd_cap", sd_cap)
    _nonneg_coef("scale_coef", scale_coef)
    if wealth <= 0.0:
        raise SizingCeilingError(f"wealth {wealth} must be > 0 (wiped out -> no size)")
    if sd_cap <= 0.0:
        raise SizingCeilingError(f"sd_cap {sd_cap} must be > 0")

    z = deviation_sd
    envelope = math.exp(-(z * z) / (2.0 * sd_cap * sd_cap))
    return -scale_coef * wealth * z * envelope


def _mr_overshoot_attenuation(deviation_sd: float, sd_cap: float) -> float:
    """Multiplicative attenuation in (0, 1] applied ONLY when |z| > sd_cap: the ratio
    of the OU envelope magnitude at |z| to its magnitude at the cap. At/inside the cap
    -> 1.0 (the Kelly/haircut layers govern); past the cap -> shrinks toward 0."""
    z = abs(deviation_sd)
    if z <= sd_cap:
        return 1.0
    peak = sd_cap * math.exp(-(sd_cap * sd_cap) / (2.0 * sd_cap * sd_cap))
    here = z * math.exp(-(z * z) / (2.0 * sd_cap * sd_cap))
    return min(1.0, here / peak)


# ── (5) regime-change kill — stop doubling into a genuine revaluation ──────
def regime_change_kill(gate_signal: bool, is_fundamental_revaluation: bool) -> bool:
    """Return True (KILL: stop scaling, size to 0) only when the gate flags a regime
    change AND that change is a genuine fundamental revaluation, not noise.

    Both inputs are gate-supplied and deterministic (no look-ahead). A regime change
    that is mere noise does NOT kill (you may double into it, to the OU cap); a
    fundamental revaluation of a short-vol loser does — "double, but only to a point,
    and never into a real regime break".
    """
    return bool(gate_signal) and bool(is_fundamental_revaluation)


# ── (6) apply_ceiling — min(RCK, Sinclair ceiling) with the kill switch ────
def apply_ceiling(
    rck_b: float,
    ceiling_inputs: CeilingInputs,
    *,
    kelly_fraction: float,
    mean_shrink: float,
    estimation_coef: float,
    skew_coef: float,
    gate_signal: bool = False,
    is_fundamental_revaluation: bool = False,
    mr_deviation_sd: float | None = None,
    wealth: float | None = None,
    mr_scale_coef: float | None = None,
    sd_cap: float = math.sqrt(2),
) -> CeilingResult:
    """Clamp an RCK size from above with the Sinclair ceiling, honouring the kill switch.

    Pipeline (each factor RECORDED in applied_haircuts so nothing stacks silently):
      0. kill switch: fundamental revaluation -> f_ceiling = 0, killed=True.
      1. deflate the mean (Chopra-Ziemba)  -> mu_deflated = (1-mean_shrink)*mu.
      2. full-Kelly from the deflated mean -> f_full = mu_deflated / sigma2.
         (non-positive edge -> no bet, f_ceiling = 0.)
      3. shrunk fraction-of-Kelly          -> f_frac = kelly_fraction * f_full.
      4. estimation + skew shrink.
      5. OU overshoot attenuation (only if MR params supplied and |z| > sd_cap).
      6. Sinclair ceiling = the result of 1-5; f_ceiling = min(rck_b, ceiling).

    Because mean_shrink deflates mu, kelly_fraction < 1, and steps 4-5 attenuate,
    the ceiling is a TRUE upper bound: for a positive-edge sleeve (mu > 0),
    f_ceiling <= mu/sigma2 (raw full Kelly). All risk knobs are UNSIGNED caller args.
    """
    _finite("rck_b", rck_b)
    ci = ceiling_inputs
    _finite("mu", ci.mu)
    _finite("sigma2", ci.sigma2)
    if ci.sigma2 <= 0.0:
        raise SizingCeilingError(f"sigma2 {ci.sigma2} must be > 0 (fail-closed)")
    _finite("skew", ci.skew)
    if not isinstance(ci.sample_n, int) or ci.sample_n < 1:
        raise SizingCeilingError(f"sample_n {ci.sample_n!r} must be an int >= 1")

    kelly_fraction = _fraction_open_unit("kelly_fraction", kelly_fraction)
    _shrink_unit("mean_shrink", mean_shrink)
    _nonneg_coef("estimation_coef", estimation_coef)
    _nonneg_coef("skew_coef", skew_coef)

    # 0. kill switch — sacred stop, wins over everything
    if regime_change_kill(gate_signal, is_fundamental_revaluation):
        return CeilingResult(
            f_ceiling=0.0,
            applied_haircuts=(("regime_change_kill", 0.0),),
            killed=True,
            reason="fundamental regime revaluation — stop scaling the short-vol loser",
        )

    haircuts: list[tuple[str, float]] = []

    # 1. deflate the mean
    mu_deflated = mean_haircut_chopra_ziemba(ci.mu, mean_shrink)
    haircuts.append(("mean_shrink", 1.0 - mean_shrink))

    # 2. full-Kelly from the deflated mean; non-positive edge -> no bet
    f_full = continuous_kelly_ceiling(mu_deflated, ci.sigma2)
    if f_full <= 0.0:
        return CeilingResult(
            f_ceiling=0.0,
            applied_haircuts=tuple(haircuts),
            killed=False,
            reason="non-positive deflated edge — no bet",
        )

    # 3. shrunk fraction-of-Kelly
    f_frac = kelly_fraction * f_full
    haircuts.append(("kelly_fraction", kelly_fraction))

    # 4. estimation + skew shrink
    f_shrunk = shrink_for_estimation_and_skew(
        f_frac, ci.sample_n, ci.skew,
        estimation_coef=estimation_coef, skew_coef=skew_coef,
    )
    haircuts.append(("estimation_skew", (f_shrunk / f_frac) if f_frac != 0.0 else 0.0))

    # 5. OU overshoot attenuation (optional)
    if mr_deviation_sd is not None or wealth is not None or mr_scale_coef is not None:
        if mr_deviation_sd is None or wealth is None or mr_scale_coef is None:
            raise SizingCeilingError(
                "MR cap needs mr_deviation_sd, wealth and mr_scale_coef together"
            )
        # validate the MR inputs via the standalone method (fail-closed reuse)
        mr_scaling_cap(mr_deviation_sd, wealth, scale_coef=mr_scale_coef, sd_cap=sd_cap)
        atten = _mr_overshoot_attenuation(mr_deviation_sd, sd_cap)
        f_shrunk = f_shrunk * atten
        haircuts.append(("mr_overshoot", atten))

    ceiling = f_shrunk

    # 6. clamp the RCK size to the ceiling
    if rck_b <= ceiling:
        f_ceiling = rck_b
        haircuts.append(("rck_clamp", rck_b))
        reason = "RCK size binds (below the Sinclair ceiling)"
    else:
        f_ceiling = ceiling
        reason = "Sinclair ceiling binds (RCK clamped down)"

    return CeilingResult(
        f_ceiling=f_ceiling,
        applied_haircuts=tuple(haircuts),
        killed=False,
        reason=reason,
    )


__all__ = [
    "SizingCeilingError",
    "CeilingInputs",
    "CeilingResult",
    "continuous_kelly_ceiling",
    "growth_rate",
    "mean_haircut_chopra_ziemba",
    "shrink_for_estimation_and_skew",
    "mr_scaling_cap",
    "regime_change_kill",
    "apply_ceiling",
]
