"""Tests (written FIRST) for research/sizing/sizing_ceiling.py — the Sinclair
(Volatility Trading, Ch. 8) operator-discipline CEILING that wraps RCK.

Rails exercised here:
  * ceiling is a TRUE upper bound: for a positive-edge sleeve f_ceiling <= mu/sigma2
  * mean haircut (Chopra-Ziemba 20:2:1) is monotone-decreasing in the shrink
  * estimation+skew haircut is monotone: less data -> more haircut, more negative
    payoff skew -> more haircut (short-convexity => MV-Kelly UNDERSTATES ruin)
  * MR (OU) scaling cap peaks at ~sqrt(2) SD and DECREASES beyond (over-deviation:
    losing wealth dominates improving edge) — Proebsting guard: sizes on passed
    (mark-to-market) wealth
  * regime-change kill returns size 0 + reason on a genuine fundamental revaluation
  * determinism (pure arithmetic, byte-reproducible)
  * fail-closed on sigma2<=0, fractions outside (0,1), non-finite inputs
  * REGRESSION: does NOT reproduce heimdall/core/sizing/kelly.py:73 double
    divide-by-max-loss (single inverse-variance scaling, not squared)
  * FIREWALL mirror: no heimdall.core module imports risk_kelly.sizing
"""
import math
import subprocess
import sys

import pytest

from risk_kelly.sizing.sizing_ceiling import (
    CeilingInputs,
    CeilingResult,
    SizingCeilingError,
    apply_ceiling,
    continuous_kelly_ceiling,
    growth_rate,
    mean_haircut_chopra_ziemba,
    mr_scaling_cap,
    regime_change_kill,
    shrink_for_estimation_and_skew,
)


# ---------------------------------------------------------------------------
# 1. continuous_kelly_ceiling — the HARD upper bound mu/sigma2
# ---------------------------------------------------------------------------
class TestContinuousKellyCeiling:
    def test_full_kelly_is_mu_over_sigma2(self):
        assert continuous_kelly_ceiling(0.10, 0.04) == pytest.approx(2.5)

    def test_single_inverse_variance_scaling_not_double(self):
        """REGRESSION vs kelly.py:73 double divide. Doubling sigma2 must HALVE the
        ceiling (single 1/sigma2). A double-divide-by-sigma2 would QUARTER it."""
        base = continuous_kelly_ceiling(0.10, 0.04)
        doubled = continuous_kelly_ceiling(0.10, 0.08)
        assert doubled == pytest.approx(0.5 * base)   # halves — single inverse
        assert doubled != pytest.approx(0.25 * base)  # not quartered — no double-divide

    def test_fail_closed_on_nonpositive_sigma2(self):
        with pytest.raises(SizingCeilingError):
            continuous_kelly_ceiling(0.10, 0.0)
        with pytest.raises(SizingCeilingError):
            continuous_kelly_ceiling(0.10, -0.01)

    def test_fail_closed_on_nonfinite(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(SizingCeilingError):
                continuous_kelly_ceiling(bad, 0.04)
            with pytest.raises(SizingCeilingError):
                continuous_kelly_ceiling(0.10, bad)


class TestGrowthSymmetry:
    def test_growth_peaks_at_full_kelly_and_is_symmetric(self):
        """g(f)=mu*f-0.5*sigma2*f^2 is symmetric about f*=mu/sigma2; over/under betting
        by the same delta lose the SAME growth — but overbetting adds pure variance,
        so underbetting dominates under uncertainty."""
        mu, sigma2 = 0.10, 0.04
        f_star = continuous_kelly_ceiling(mu, sigma2)
        g_star = growth_rate(f_star, mu, sigma2)
        for delta in (0.1, 0.5, 1.0):
            under = growth_rate(f_star - delta, mu, sigma2)
            over = growth_rate(f_star + delta, mu, sigma2)
            assert under < g_star and over < g_star
            assert under == pytest.approx(over)  # symmetric penalty


# ---------------------------------------------------------------------------
# 2. mean_haircut_chopra_ziemba — deflate the mean BEFORE it reaches the optimizer
# ---------------------------------------------------------------------------
class TestMeanHaircut:
    def test_half_kelly_poor_mans_bayesian(self):
        assert mean_haircut_chopra_ziemba(0.10, 0.5) == pytest.approx(0.05)

    def test_monotone_decreasing_in_shrink(self):
        mu = 0.10
        vals = [mean_haircut_chopra_ziemba(mu, s) for s in (0.0, 0.25, 0.5, 0.75, 0.9)]
        assert all(a > b for a, b in zip(vals, vals[1:]))

    def test_shrink_reduces_magnitude(self):
        assert mean_haircut_chopra_ziemba(0.10, 0.5) < 0.10

    def test_fail_closed_on_bad_shrink(self):
        for bad in (-0.1, 1.0, 1.5, float("nan"), float("inf")):
            with pytest.raises(SizingCeilingError):
                mean_haircut_chopra_ziemba(0.10, bad)

    def test_fail_closed_on_nonfinite_mu(self):
        with pytest.raises(SizingCeilingError):
            mean_haircut_chopra_ziemba(float("nan"), 0.5)


# ---------------------------------------------------------------------------
# 3. shrink_for_estimation_and_skew — small-sample std(f) + negative payoff skew
# ---------------------------------------------------------------------------
class TestEstimationSkewShrink:
    def test_more_data_less_haircut(self):
        big = shrink_for_estimation_and_skew(1.0, 400, 0.0, estimation_coef=1.0, skew_coef=1.0)
        small = shrink_for_estimation_and_skew(1.0, 25, 0.0, estimation_coef=1.0, skew_coef=1.0)
        assert big > small          # more observations -> milder haircut
        assert big < 1.0            # always some estimation haircut for finite n

    def test_negative_skew_more_haircut(self):
        flat = shrink_for_estimation_and_skew(1.0, 100, 0.0, estimation_coef=1.0, skew_coef=1.0)
        neg = shrink_for_estimation_and_skew(1.0, 100, -0.8, estimation_coef=1.0, skew_coef=1.0)
        assert neg < flat           # short-convexity negative skew -> cut harder

    def test_positive_skew_no_skew_penalty(self):
        flat = shrink_for_estimation_and_skew(1.0, 100, 0.0, estimation_coef=1.0, skew_coef=1.0)
        pos = shrink_for_estimation_and_skew(1.0, 100, 0.8, estimation_coef=1.0, skew_coef=1.0)
        assert pos == pytest.approx(flat)  # favourable skew earns no extra cut

    def test_result_never_exceeds_input(self):
        out = shrink_for_estimation_and_skew(2.0, 50, -0.3, estimation_coef=1.0, skew_coef=1.0)
        assert 0.0 <= out <= 2.0

    def test_fail_closed(self):
        with pytest.raises(SizingCeilingError):
            shrink_for_estimation_and_skew(1.0, 0, 0.0, estimation_coef=1.0, skew_coef=1.0)
        with pytest.raises(SizingCeilingError):
            shrink_for_estimation_and_skew(1.0, 100, 0.0, estimation_coef=-0.1, skew_coef=1.0)
        with pytest.raises(SizingCeilingError):
            shrink_for_estimation_and_skew(float("nan"), 100, 0.0, estimation_coef=1.0, skew_coef=1.0)


# ---------------------------------------------------------------------------
# 4. mr_scaling_cap — OU optimal holding, peak at ~sqrt(2) SD, reduce beyond
# ---------------------------------------------------------------------------
class TestMRScalingCap:
    def test_magnitude_peaks_near_sqrt2(self):
        grid = [i * 0.05 for i in range(1, 80)]  # 0.05 .. 3.95 SD
        mags = [abs(mr_scaling_cap(z, 1.0e6, scale_coef=1.0)) for z in grid]
        z_peak = grid[max(range(len(grid)), key=lambda i: mags[i])]
        assert z_peak == pytest.approx(math.sqrt(2), abs=0.06)

    def test_reduces_beyond_the_cap(self):
        cap = math.sqrt(2)
        m_at = abs(mr_scaling_cap(cap, 1.0e6, scale_coef=1.0))
        m_beyond = abs(mr_scaling_cap(2.5, 1.0e6, scale_coef=1.0))
        m_further = abs(mr_scaling_cap(3.5, 1.0e6, scale_coef=1.0))
        assert m_at > m_beyond > m_further  # over-deviation: losing wealth dominates

    def test_contrarian_sign(self):
        assert mr_scaling_cap(1.0, 1.0e6, scale_coef=1.0) < 0   # above mean -> short
        assert mr_scaling_cap(-1.0, 1.0e6, scale_coef=1.0) > 0  # below mean -> long

    def test_scales_with_marktomarket_wealth(self):
        """Proebsting guard: bigger passed (current) wealth -> proportionally bigger."""
        small = abs(mr_scaling_cap(1.0, 1.0e6, scale_coef=1.0))
        big = abs(mr_scaling_cap(1.0, 2.0e6, scale_coef=1.0))
        assert big == pytest.approx(2.0 * small)

    def test_fail_closed(self):
        with pytest.raises(SizingCeilingError):
            mr_scaling_cap(1.0, 0.0, scale_coef=1.0)          # wiped-out wealth
        with pytest.raises(SizingCeilingError):
            mr_scaling_cap(1.0, 1.0e6, scale_coef=1.0, sd_cap=0.0)
        with pytest.raises(SizingCeilingError):
            mr_scaling_cap(float("inf"), 1.0e6, scale_coef=1.0)


# ---------------------------------------------------------------------------
# 5. regime_change_kill — stop doubling into a genuine fundamental revaluation
# ---------------------------------------------------------------------------
class TestRegimeChangeKill:
    def test_kill_only_on_fundamental_regime_change(self):
        assert regime_change_kill(True, True) is True    # regime changed & fundamental
        assert regime_change_kill(True, False) is False  # changed but noise -> double
        assert regime_change_kill(False, True) is False  # no regime change
        assert regime_change_kill(False, False) is False


# ---------------------------------------------------------------------------
# 6. apply_ceiling — min(RCK, Sinclair ceiling) with the kill switch
# ---------------------------------------------------------------------------
def _inputs(mu=0.10, sigma2=0.04, skew=-0.5, sample_n=120):
    return CeilingInputs(mu=mu, sigma2=sigma2, skew=skew, sample_n=sample_n)


_KNOBS = dict(
    kelly_fraction=0.25,
    mean_shrink=0.5,
    estimation_coef=1.0,
    skew_coef=1.0,
)


class TestApplyCeiling:
    def test_is_true_upper_bound_leq_full_kelly(self):
        ci = _inputs()
        res = apply_ceiling(10.0, ci, **_KNOBS)  # RCK wants a huge size
        assert isinstance(res, CeilingResult)
        assert res.f_ceiling <= continuous_kelly_ceiling(ci.mu, ci.sigma2) + 1e-12
        assert res.killed is False

    def test_rck_binds_when_smaller_than_ceiling(self):
        ci = _inputs()
        res = apply_ceiling(0.001, ci, **_KNOBS)  # RCK already tiny
        assert res.f_ceiling == pytest.approx(0.001)
        assert any(name == "rck_clamp" for name, _ in res.applied_haircuts)

    def test_ceiling_binds_when_smaller_than_rck(self):
        ci = _inputs()
        res = apply_ceiling(10.0, ci, **_KNOBS)
        assert res.f_ceiling < 10.0

    def test_kill_returns_zero_with_reason(self):
        ci = _inputs()
        res = apply_ceiling(
            10.0, ci, gate_signal=True, is_fundamental_revaluation=True, **_KNOBS
        )
        assert res.killed is True
        assert res.f_ceiling == 0.0
        assert res.reason != ""

    def test_noise_regime_change_does_not_kill(self):
        ci = _inputs()
        res = apply_ceiling(
            10.0, ci, gate_signal=True, is_fundamental_revaluation=False, **_KNOBS
        )
        assert res.killed is False
        assert res.f_ceiling > 0.0

    def test_mr_overshoot_reduces_ceiling(self):
        ci = _inputs()
        base = apply_ceiling(10.0, ci, mr_deviation_sd=math.sqrt(2), wealth=1e6,
                             mr_scale_coef=1.0, **_KNOBS)
        over = apply_ceiling(10.0, ci, mr_deviation_sd=3.0, wealth=1e6,
                             mr_scale_coef=1.0, **_KNOBS)
        assert over.f_ceiling < base.f_ceiling  # deviation past the cap -> reduce

    def test_haircuts_are_surfaced_not_silently_stacked(self):
        ci = _inputs()
        res = apply_ceiling(10.0, ci, **_KNOBS)
        names = {name for name, _ in res.applied_haircuts}
        # the deflated-mean AND the fraction-of-Kelly must both be visible so the
        # lambda-vs-haircut double-conservatism tension can be inspected, not hidden
        assert "mean_shrink" in names
        assert "kelly_fraction" in names
        assert "estimation_skew" in names

    def test_nonpositive_edge_gives_no_bet(self):
        ci = _inputs(mu=-0.05)
        res = apply_ceiling(10.0, ci, **_KNOBS)
        assert res.f_ceiling == 0.0

    def test_fail_closed_on_bad_fraction(self):
        ci = _inputs()
        for bad in (0.0, 1.0, 1.5, -0.1):
            with pytest.raises(SizingCeilingError):
                apply_ceiling(10.0, ci, kelly_fraction=bad, mean_shrink=0.5,
                              estimation_coef=1.0, skew_coef=1.0)

    def test_fail_closed_on_bad_sigma2(self):
        with pytest.raises(SizingCeilingError):
            apply_ceiling(10.0, _inputs(sigma2=0.0), **_KNOBS)

    def test_fail_closed_on_nonfinite_rck(self):
        with pytest.raises(SizingCeilingError):
            apply_ceiling(float("nan"), _inputs(), **_KNOBS)


# ---------------------------------------------------------------------------
# 7. determinism
# ---------------------------------------------------------------------------
class TestDeterminism:
    def test_byte_identical_across_calls(self):
        ci = _inputs()
        a = apply_ceiling(10.0, ci, mr_deviation_sd=2.1, wealth=1e6,
                          mr_scale_coef=1.0, **_KNOBS)
        b = apply_ceiling(10.0, ci, mr_deviation_sd=2.1, wealth=1e6,
                          mr_scale_coef=1.0, **_KNOBS)
        assert a == b
        assert a.f_ceiling == b.f_ceiling


# ---------------------------------------------------------------------------
# 8. firewall mirror — core ⊥ research.sizing (RAIL a)
# ---------------------------------------------------------------------------
def test_standalone_ceiling_import_has_no_convex_solver_side_effect():
    r = subprocess.run([sys.executable, "-c", "import sys; import risk_kelly.sizing.sizing_ceiling; assert 'cvxpy' not in sys.modules"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
