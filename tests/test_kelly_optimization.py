"""Tests for research/calibration/kelly_optimization.py (PA-030 optimizer).

Pins the panel-converged methodology (2026-07-02): plug-in full Kelly via
bisection vs the closed-form two-outcome solution; the measured c* shrinkage
(incl. the docstring claim that bootstrap-AVERAGED growth argmax ≈ plug-in
argmax — no shrinkage); the drawdown guard (Wilson upper bound, isotonic clip,
stress saturation at −1, monotonicity in stress/D_kill); the fail-closed data
contract (r < −1 and r > 1 raise, never clip); the selection tree (NOT_SIGNABLE
paths, MC_UNSTABLE, floor rounding, binding-constraint naming); walk-forward
purge (expiry ≤ decision date) and hard risk gates; seed derivation vectors;
and run-to-run determinism of the full result object.

All data is deterministic and synthetic — no network, no RNG without a fixed
seed (NF-08).
"""
from __future__ import annotations

import math
import random
from datetime import date, timedelta
from decimal import Decimal

import pytest

from risk_kelly.calibration.kelly_calibration import KellyCalibrationError
from risk_kelly.calibration.kelly_optimization import (
    DEFAULTS,
    KellyOptimizationResult,
    KellyParams,
    SetResult,
    WalkforwardRefit,
    WalkforwardResult,
    WalkforwardRuleMetrics,
    _composite_fit_at,
    _gen_paths,
    _path_mdds,
    _quantile,
    _shrinkage,
    _stability_disagrees,
    _stress_s1,
    derive_seed,
    empirical_growth,
    finalize_status,
    full_kelly,
    optimize_kelly_fraction,
    stationary_bootstrap_indices,
    walkforward_validation,
    wilson_upper,
)

_SHA = "a" * 64  # fixed fake snapshot hash for seed derivation


def _fast_params(**overrides) -> KellyParams:
    """Small-B params so the full pipeline runs in seconds in tests."""
    base = dict(
        b_shrink=60, b_dd=200, b_shrink_wf=30, b_dd_wf=80,
        horizon=20, horizon_report=25,
        warmup_trades=70, refit_cadence=13,
    )
    base.update(overrides)
    return KellyParams(**base)


def _pattern_returns(n: int = 200) -> list:
    """Deterministic 9-wins/1-loss cycle: +0.06 ×9, −0.45 ×1.

    Two-outcome closed form: f* = p/a − q/b = 0.9/0.45 − 0.1/0.06 = 2 − 5/3 = 1/3.
    """
    cycle = [0.06] * 9 + [-0.45]
    return (cycle * ((n // 10) + 1))[:n]


def _dates(n: int, start: date = date(2000, 1, 5)) -> tuple[list, list]:
    entries = [start + timedelta(days=28 * i) for i in range(n)]
    expiries = [e + timedelta(days=28) for e in entries]
    return entries, expiries


# ---------------------------------------------------------------------------
# full_kelly — bisection vs closed form
# ---------------------------------------------------------------------------

class TestFullKelly:
    def test_two_outcome_closed_form(self):
        # 40 × +0.5 and 40 × −0.4: f* = p/a − q/b = 0.5/0.4 − 0.5/0.5 = 0.25
        returns = [0.5] * 40 + [-0.4] * 40
        f, capped = full_kelly(returns)
        assert not capped
        assert abs(f - 0.25) < 1e-6

    def test_pattern_closed_form(self):
        f, capped = full_kelly(_pattern_returns(200))
        assert not capped
        assert abs(f - (1.0 / 3.0)) < 1e-6

    def test_nonpositive_mean_returns_zero(self):
        returns = [0.01, -0.01] * 50  # mean exactly 0
        f, capped = full_kelly(returns)
        assert f == 0.0 and not capped

    def test_negated_returns_zero(self):
        f, _ = full_kelly([-r for r in _pattern_returns(100)])
        assert f == 0.0

    def test_kelly_capped_when_root_beyond_domain(self):
        # 99 small wins, one tiny loss → h(f_hi) > 0 → parked at the 0.5 cap
        returns = [0.05] * 99 + [-0.01]
        f, capped = full_kelly(returns)
        assert capped and f == 0.5

    def test_single_full_max_loss_is_finite(self):
        # a −1 print among wins must yield a finite, domain-bounded result
        returns = [0.03] * 99 + [-1.0]
        f, capped = full_kelly(returns)
        assert math.isfinite(f) and 0.0 < f <= 0.5

    def test_growth_curve_ruin_guard(self):
        assert empirical_growth([-1.0, 0.5], 1.0) == float("-inf")


# ---------------------------------------------------------------------------
# Data contract — raise, never clip
# ---------------------------------------------------------------------------

class TestDataContract:
    def test_below_minus_one_raises(self):
        rs = [Decimal("0.05")] * 70 + [Decimal("-1.5")]
        with pytest.raises(KellyCalibrationError, match="max-loss accounting"):
            optimize_kelly_fraction(rs, data_snapshot_sha256=_SHA, params=_fast_params())

    def test_above_plus_one_raises(self):
        rs = [Decimal("0.05")] * 70 + [Decimal("1.2")]
        with pytest.raises(KellyCalibrationError, match="structure"):
            optimize_kelly_fraction(rs, data_snapshot_sha256=_SHA, params=_fast_params())

    def test_constant_series_raises(self):
        rs = [Decimal("0.05")] * 100
        with pytest.raises(KellyCalibrationError, match="zero-variance"):
            optimize_kelly_fraction(rs, data_snapshot_sha256=_SHA, params=_fast_params())

    def test_empty_raises(self):
        with pytest.raises(KellyCalibrationError, match="empty"):
            optimize_kelly_fraction([], data_snapshot_sha256=_SHA, params=_fast_params())


# ---------------------------------------------------------------------------
# NOT_SIGNABLE gates
# ---------------------------------------------------------------------------

class TestNotSignable:
    def test_too_few_observations(self):
        rs = [Decimal(str(r)) for r in _pattern_returns(59)]
        res = optimize_kelly_fraction(rs, data_snapshot_sha256=_SHA, params=_fast_params())
        assert res.status == "NOT_SIGNABLE"
        assert res.f_rec is None
        assert any("n<60" in c for c in res.reason_codes)

    def test_too_few_losses(self):
        rs = [Decimal("0.05"), Decimal("0.04")] * 50  # varying, zero losses
        res = optimize_kelly_fraction(rs, data_snapshot_sha256=_SHA, params=_fast_params())
        assert res.status == "NOT_SIGNABLE"
        assert any("losses<5" in c for c in res.reason_codes)

    def test_nonpositive_edge(self):
        rs = [Decimal(str(-r)) for r in _pattern_returns(200)]
        res = optimize_kelly_fraction(rs, data_snapshot_sha256=_SHA, params=_fast_params())
        assert res.status == "NOT_SIGNABLE"
        assert any("nonpositive_edge" in c for c in res.reason_codes)

    def test_guard_floor_breach(self):
        # p_kill = 0 is unattainable on the Wilson UPPER bound (it is > 0 even
        # at k = 0), so even f = 0.005 breaches → guard undefined → NOT_SIGNABLE.
        rs = [Decimal(str(r)) for r in _pattern_returns(200)]
        res = optimize_kelly_fraction(
            rs, data_snapshot_sha256=_SHA, params=_fast_params(p_kill=0.0))
        assert res.status == "NOT_SIGNABLE"
        assert any("guard_floor_breach" in c for c in res.reason_codes)


# ---------------------------------------------------------------------------
# Selection tree, composition, rounding
# ---------------------------------------------------------------------------

class TestSelection:
    def test_candidate_on_clean_data(self):
        rs = [Decimal(str(r)) for r in _pattern_returns(200)]
        res = optimize_kelly_fraction(rs, data_snapshot_sha256=_SHA, params=_fast_params())
        assert res.status == "CANDIDATE"
        assert res.f_rec is not None
        assert Decimal("0.005") <= res.f_rec <= Decimal("0.25")
        assert res.binding_constraint  # names the binding term

    def test_frec_is_min_of_pipes_and_cap(self):
        rs = [Decimal(str(r)) for r in _pattern_returns(200)]
        res = optimize_kelly_fraction(rs, data_snapshot_sha256=_SHA, params=_fast_params())
        pipes = [s.f_pipe for s in res.sets] + [res.params.f_cap]
        assert res.f_raw == min(pipes)

    def test_floor_rounding_direction(self):
        # 0.123456… must floor to 0.12345, never round up
        assert Decimal("0.123456").quantize(
            Decimal("0.00001"), rounding="ROUND_FLOOR") == Decimal("0.12345")
        rs = [Decimal(str(r)) for r in _pattern_returns(200)]
        res = optimize_kelly_fraction(rs, data_snapshot_sha256=_SHA, params=_fast_params())
        assert res.f_rec <= Decimal(str(res.f_raw))

    def test_mc_unstable_path(self):
        # Impossible tolerance forces the stability gate to fire (|Δ| ≥ 0 > −1)
        rs = [Decimal(str(r)) for r in _pattern_returns(200)]
        res = optimize_kelly_fraction(
            rs, data_snapshot_sha256=_SHA, params=_fast_params(stability_tol=-1.0))
        assert res.status == "MC_UNSTABLE"
        assert res.f_rec is None

    def test_deterministic_across_runs(self):
        rs = [Decimal(str(r)) for r in _pattern_returns(200)]
        a = optimize_kelly_fraction(rs, data_snapshot_sha256=_SHA, params=_fast_params())
        b = optimize_kelly_fraction(rs, data_snapshot_sha256=_SHA, params=_fast_params())
        assert a == b  # full frozen-dataclass equality, seeds included

    def test_different_snapshot_hash_changes_seeds(self):
        rs = [Decimal(str(r)) for r in _pattern_returns(200)]
        a = optimize_kelly_fraction(rs, data_snapshot_sha256=_SHA, params=_fast_params())
        b = optimize_kelly_fraction(rs, data_snapshot_sha256="b" * 64, params=_fast_params())
        assert dict(a.seeds) != dict(b.seeds)


# ---------------------------------------------------------------------------
# Shrinkage — the docstring claims are regression-pinned
# ---------------------------------------------------------------------------

class TestShrinkage:
    def test_bootstrap_averaged_growth_argmax_is_plugin_argmax(self):
        """Docstring claim (a): averaging G_b(f) across resamples ≈ G(f), so its
        argmax provides no shrinkage. Pinned so nobody 'simplifies' c* back to it."""
        returns = _pattern_returns(200)
        f_star, _ = full_kelly(returns)
        rng = random.Random(derive_seed(_SHA, "test-claim-a"))
        f_grid = [0.02 * k for k in range(1, 26)]  # 0.02..0.50
        n = len(returns)
        avg = []
        resamples = []
        for _ in range(300):
            idx = stationary_bootstrap_indices(rng, n, n, 6)
            resamples.append([returns[i] for i in idx])
        for f in f_grid:
            avg.append((sum(empirical_growth(s, f) for s in resamples) / 300, f))
        f_avg_argmax = max(avg)[1]
        assert abs(f_avg_argmax - f_star) < 0.05  # within grid coarseness of f̂*

    def test_c_star_shrinks_below_one_on_noisy_data(self):
        rs = [Decimal(str(r)) for r in _pattern_returns(200)]
        res = optimize_kelly_fraction(rs, data_snapshot_sha256=_SHA, params=_fast_params())
        for s in res.sets:
            assert 0.01 <= s.c_star <= 1.0
            assert s.f_shrunk == min(s.c_star * s.f_star, 0.5)

    def test_r_curve_recorded(self):
        rs = [Decimal(str(r)) for r in _pattern_returns(200)]
        res = optimize_kelly_fraction(rs, data_snapshot_sha256=_SHA, params=_fast_params())
        full = res.sets[0]
        assert len(full.r_curve) == 100
        cs = [c for c, _ in full.r_curve]
        assert cs == sorted(cs)


# ---------------------------------------------------------------------------
# Wilson bound + guard mechanics
# ---------------------------------------------------------------------------

class TestGuard:
    def test_wilson_upper_zero_events(self):
        # k=0, n=100, z=1.6449: U = (z²/2n · 2) / (1 + z²/n) ≈ 0.026342
        assert abs(wilson_upper(0, 100) - 0.026342) < 1e-4

    def test_wilson_upper_exceeds_p_hat(self):
        for k, n in ((0, 50), (3, 100), (25, 200)):
            assert wilson_upper(k, n) > k / n

    def test_wilson_monotone_in_k(self):
        us = [wilson_upper(k, 100) for k in range(0, 30)]
        assert us == sorted(us)

    def test_stress_s1_saturates_at_minus_one(self):
        assert _stress_s1([-0.9, -0.5, 0.4], 1.25) == [-1.0, -0.625, 0.4]

    def test_path_mdd_ruin(self):
        # f=1.0 against r=−1 → multiplier 0 → RUIN → MDD = 1.0
        mdds = _path_mdds([-1.0, 0.05], [[0, 1, 1]], 1.0)
        assert mdds == [1.0]

    def test_path_mdd_simple_sequence(self):
        # One path [win, loss]: MDD = 1 − (1 + f·r_loss) at f where loss < win
        f = 0.1
        mdds = _path_mdds([0.05, -0.45], [[0, 1]], f)
        assert abs(mdds[0] - (1.0 - (1.0 + f * -0.45))) < 1e-12

    def test_f_dd_monotone_in_stress_multiplier(self):
        rs = [Decimal(str(r)) for r in _pattern_returns(200)]
        # Force the guard to bind by making the shrunk-Kelly term huge
        res_lo = optimize_kelly_fraction(
            rs, data_snapshot_sha256=_SHA, params=_fast_params(stress_mult=1.0))
        res_hi = optimize_kelly_fraction(
            rs, data_snapshot_sha256=_SHA, params=_fast_params(stress_mult=1.5))
        for lo, hi in zip(res_lo.sets, res_hi.sets):
            if lo.f_dd is not None and hi.f_dd is not None:
                assert hi.f_dd <= lo.f_dd

    def test_f_dd_monotone_in_d_kill(self):
        rs = [Decimal(str(r)) for r in _pattern_returns(200)]
        res_tight = optimize_kelly_fraction(
            rs, data_snapshot_sha256=_SHA, params=_fast_params(d_kill=0.10))
        res_loose = optimize_kelly_fraction(
            rs, data_snapshot_sha256=_SHA, params=_fast_params(d_kill=0.30))
        for t, l in zip(res_tight.sets, res_loose.sets):
            if t.f_dd is not None and l.f_dd is not None:
                assert t.f_dd <= l.f_dd

    def test_incumbent_risk_line_present(self):
        rs = [Decimal(str(r)) for r in _pattern_returns(200)]
        res = optimize_kelly_fraction(rs, data_snapshot_sha256=_SHA, params=_fast_params())
        full = res.sets[0]
        assert len(full.incumbent_guard_line) == 2  # both binding cells
        for cell, p_hat, ub in full.incumbent_guard_line:
            assert cell in ("L6_S0", "L6_S1")
            assert 0.0 <= p_hat <= ub <= 1.0


# ---------------------------------------------------------------------------
# Seeds (NF-08)
# ---------------------------------------------------------------------------

class TestSeeds:
    def test_derivation_vector(self):
        # Regression pin: derivation formula must never silently change.
        expected = int.from_bytes(
            __import__("hashlib").sha256(
                f"{_SHA}|kelly-opt-1.0|shrink|E_full".encode()).digest()[:8], "big")
        assert derive_seed(_SHA, "shrink", "E_full") == expected

    def test_all_seeds_surfaced(self):
        rs = [Decimal(str(r)) for r in _pattern_returns(200)]
        res = optimize_kelly_fraction(rs, data_snapshot_sha256=_SHA, params=_fast_params())
        assert len(res.seeds) > 0
        purposes = [p for p, _ in res.seeds]
        assert any("shrink" in p for p in purposes)
        assert any("paths" in p for p in purposes)
        assert any("stability" in p for p in purposes)

    def test_bootstrap_deterministic(self):
        a = stationary_bootstrap_indices(random.Random(42), 100, 50, 6)
        b = stationary_bootstrap_indices(random.Random(42), 100, 50, 6)
        assert a == b and len(a) == 50 and all(0 <= i < 100 for i in a)


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

class TestWalkforward:
    def test_purge_excludes_inflight_trade(self):
        """A trade expiring AFTER the decision date must not be in the training
        set. Constructed so leakage would change the answer: the in-flight trade
        (index warmup−1) expires one day after trade warmup's entry."""
        n = 120
        params = _fast_params(warmup_trades=70)
        returns = [Decimal(str(r)) for r in _pattern_returns(n)]
        entries, expiries = _dates(n)
        # Make trade 69 expire one day AFTER trade 70's entry
        expiries[69] = entries[70] + timedelta(days=1)
        wf = walkforward_validation(
            returns, entries, expiries, data_snapshot_sha256=_SHA, params=params)
        first = wf.refits[0]
        assert first.k == 70
        # Train set = trades with expiry ≤ entry of trade 70 → 69 of the first 70
        assert first.n_train == 69

    def test_expiry_on_decision_date_included(self):
        n = 120
        params = _fast_params(warmup_trades=70)
        returns = [Decimal(str(r)) for r in _pattern_returns(n)]
        entries, expiries = _dates(n)
        expiries[69] = entries[70]  # settles exactly on the decision date
        wf = walkforward_validation(
            returns, entries, expiries, data_snapshot_sha256=_SHA, params=params)
        assert wf.refits[0].n_train == 70

    def test_rules_and_gates_present(self):
        n = 150
        returns = [Decimal(str(r)) for r in _pattern_returns(n)]
        entries, expiries = _dates(n)
        wf = walkforward_validation(
            returns, entries, expiries, data_snapshot_sha256=_SHA,
            params=_fast_params(warmup_trades=70))
        names = [r.rule for r in wf.rules]
        assert names == ["optimized", "incumbent", "half_kelly", "full_kelly"]
        assert isinstance(wf.passed, bool)
        assert wf.gate_no_ruin  # capped fractions on r ≥ −1 cannot ruin

    def test_fallback_counted_for_short_oos_arm(self):
        # Early windows: last-30% slice has < min_obs → composite falls back
        n = 150
        returns = [Decimal(str(r)) for r in _pattern_returns(n)]
        entries, expiries = _dates(n)
        wf = walkforward_validation(
            returns, entries, expiries, data_snapshot_sha256=_SHA,
            params=_fast_params(warmup_trades=70))
        # train at k=70 has ~70 trades → last-30% ≈ 21 < 60 → must fall back
        assert wf.fallback_count >= 1
        assert wf.refits[0].fell_back_to_full


# ---------------------------------------------------------------------------
# Final status composition
# ---------------------------------------------------------------------------

def _passing_wf(refit_count: int = 1) -> WalkforwardResult:
    """A structurally valid, gate-passing walk-forward result. Constructed
    directly so the finalize_status success branches are asserted
    UNCONDITIONALLY — the previous fixture-driven versions of these tests
    guarded every assert behind `if wf.passed:` and the fixture never passed
    gate (iii), leaving PROPOSED / PENDING_GOVERNANCE with zero coverage."""
    refits = tuple(
        WalkforwardRefit(k=70 + 13 * i, entry_date=f"2005-0{i + 1}-01",
                         n_train=70 + 13 * i, f_applied=0.04,
                         fell_back_to_full=False)
        for i in range(refit_count)
    )
    rules = tuple(
        WalkforwardRuleMetrics(
            rule=r, total_log_growth=0.10, annualized_growth=0.05,
            realized_mdd=0.03, time_under_water_frac=0.20,
            per_decade_growth=(), ruined=False)
        for r in ("optimized", "incumbent", "half_kelly", "full_kelly")
    )
    return WalkforwardResult(
        refits=refits, rules=rules, gate_no_ruin=True,
        gate_mdd_below_dkill=True, gate_mdd_not_worse_than_incumbent=True,
        passed=True, fallback_count=0, refit_instability_warnings=(), seeds=())


class TestFinalizeStatus:
    def _candidate(self) -> KellyOptimizationResult:
        rs = [Decimal(str(r)) for r in _pattern_returns(200)]
        return optimize_kelly_fraction(rs, data_snapshot_sha256=_SHA, params=_fast_params())

    def _wf(self, params) -> object:
        n = 150
        returns = [Decimal(str(r)) for r in _pattern_returns(n)]
        entries, expiries = _dates(n)
        return walkforward_validation(
            returns, entries, expiries, data_snapshot_sha256=_SHA, params=params)

    def test_pending_governance_when_dkill_unconfirmed(self):
        params = _fast_params(warmup_trades=70)  # d_kill_confirmed=False default
        opt = self._candidate()
        status, codes = finalize_status(opt, _passing_wf(), params)
        assert status == "PENDING_GOVERNANCE"
        assert "d_kill_unconfirmed_placeholder" in codes

    def test_proposed_when_dkill_confirmed(self):
        params = _fast_params(warmup_trades=70, d_kill_confirmed=True)
        opt = self._candidate()
        status, codes = finalize_status(opt, _passing_wf(), params)
        assert status == "PROPOSED" and codes == ()

    def test_zero_refits_fails_closed(self):
        # Zero refits = zero validation evidence: a wf whose gates all pass
        # vacuously (empty applied series) must NOT reach PROPOSED — it is
        # evidentially identical to wf=None (walkforward_missing). Reachable
        # only with overridden warmup/min_obs (short-history recalibrations).
        params = _fast_params(warmup_trades=70, d_kill_confirmed=True)
        opt = self._candidate()
        status, codes = finalize_status(opt, _passing_wf(refit_count=0), params)
        assert status == "VALIDATION_FAILED"
        assert "walkforward_no_refits" in codes

    def test_walkforward_missing_fails_closed(self):
        params = _fast_params(d_kill_confirmed=True)
        opt = self._candidate()
        status, codes = finalize_status(opt, None, params)
        assert status == "VALIDATION_FAILED"
        assert "walkforward_missing" in codes

    def test_validation_failed_propagates(self):
        # d_kill < 0 makes gate (ii) unpassable even for a rule that sits out
        # entirely (realized MDD 0.0 > −0.1) — forces the failure path.
        params = _fast_params(warmup_trades=70, d_kill=-0.1)
        opt = self._candidate()
        wf = self._wf(params)
        assert not wf.passed
        status, codes = finalize_status(opt, wf, params)
        assert status == "VALIDATION_FAILED"
        assert "wf_mdd_exceeds_d_kill" in codes

    def test_not_signable_propagates(self):
        rs = [Decimal(str(-r)) for r in _pattern_returns(200)]
        opt = optimize_kelly_fraction(rs, data_snapshot_sha256=_SHA, params=_fast_params())
        status, _ = finalize_status(opt, None, _fast_params())
        assert status == "NOT_SIGNABLE"


# ---------------------------------------------------------------------------
# Stability gate — Decimal boundary semantics (REVIEW F3)
# ---------------------------------------------------------------------------

class TestStabilityTolerance:
    def test_exact_one_grid_step_passes(self):
        # Spec §5(6): MC_UNSTABLE only if |Δ| > 0.005 — Δ = 0.005 exactly PASSES.
        # In binary float, |0.250−0.245| > 0.005 (spurious fire) while
        # |0.040−0.035| < 0.005: the float compare made the pre-registered
        # tolerance value-dependent. 0/49 adjacent grid pairs may disagree.
        step = Decimal("0.005")
        for k in range(1, 50):
            lo = Decimal("0.00500") * k
            hi = lo + step
            assert not _stability_disagrees(hi.quantize(Decimal("0.00001")),
                                            lo.quantize(Decimal("0.00001")), 0.005)

    def test_beyond_one_grid_step_fires(self):
        assert _stability_disagrees(Decimal("0.05500"), Decimal("0.04500"), 0.005)

    def test_none_fires(self):
        assert _stability_disagrees(None, Decimal("0.04000"), 0.005)


# ---------------------------------------------------------------------------
# E_oos slice convention (REVIEW F4 — INTERIM adjudication pin)
# ---------------------------------------------------------------------------

class TestOOSSliceConvention:
    def test_round_convention_pinned(self):
        """E_oos = last round(0.3·n) trades — the repo's oos_returns()
        convention. Spec §2's formula text says ceil() while citing this
        convention (self-contradictory); round() is the INTERIM resolution
        pending operator adjudication (REVIEW_FINDINGS_KELLY_OPT_v1 F4).
        n=395: round(118.5)=118 (banker's) vs ceil=119 — this pin detects a
        silent flip of the convention in either direction."""
        rs = [Decimal(str(r)) for r in _pattern_returns(395)]
        res = optimize_kelly_fraction(rs, data_snapshot_sha256=_SHA, params=_fast_params())
        assert res.sets[1].name == "E_oos"
        assert res.sets[1].n == 118  # round; ceil would give 119


# ---------------------------------------------------------------------------
# Risk tables at f_rec (REVIEW F5)
# ---------------------------------------------------------------------------

class TestRiskTablesAtFrec:
    def test_mdd_table_evaluated_at_frec_not_fpipe(self):
        """Force the signing cap to bind (f_cap < f_pipe): the mdd_at_frec
        table must be computed at f_rec = cap, not at the set's own f_pipe."""
        rs = [Decimal(str(r)) for r in _pattern_returns(200)]
        params = _fast_params(f_cap=0.05)
        res = optimize_kelly_fraction(rs, data_snapshot_sha256=_SHA, params=params)
        assert res.status == "CANDIDATE"
        assert res.f_rec == Decimal("0.05000")
        full = res.sets[0]
        assert full.f_pipe > 0.05  # the cap genuinely binds
        # Recompute the L6_S0 row from primitives at f_rec and at f_pipe.
        returns = [float(r) for r in rs]
        seed = derive_seed(_SHA, "paths", "E_full", params.block_len,
                           params.horizon, params.b_dd)
        paths = _gen_paths(random.Random(seed), len(returns), params.b_dd,
                           params.horizon, params.block_len)
        p50_at_frec = _quantile(sorted(_path_mdds(returns, paths, float(res.f_rec))), 0.50)
        p50_at_fpipe = _quantile(sorted(_path_mdds(returns, paths, full.f_pipe)), 0.50)
        stored_p50 = full.mdd_at_frec[0][1]
        assert stored_p50 == pytest.approx(p50_at_frec, abs=1e-12)
        assert p50_at_fpipe != pytest.approx(p50_at_frec, abs=1e-9)  # discriminating

    def test_tables_filled_only_on_candidate(self):
        rs = [Decimal(str(-r)) for r in _pattern_returns(200)]
        res = optimize_kelly_fraction(rs, data_snapshot_sha256=_SHA, params=_fast_params())
        assert res.status == "NOT_SIGNABLE"
        for s in res.sets:
            assert s.mdd_at_frec == () and s.terminal_wealth == ()


# ---------------------------------------------------------------------------
# Walk-forward replay fidelity (REVIEW F2 / F14)
# ---------------------------------------------------------------------------

def _returns_with_dead_tail(n_good: int = 180, n_bad: int = 100) -> list:
    """Good pattern edge first, then a negative-drift tail with plenty of
    losses: late refit windows have an oos arm that passes the STRUCTURAL
    gates (n ≥ 60, losses ≥ 5) but has nonpositive edge."""
    bad_cycle = [-0.02] * 7 + [0.01] * 3
    return _pattern_returns(n_good) + (bad_cycle * ((n_bad // 10) + 1))[:n_bad]


class TestWalkforwardReplayFidelity:
    def test_nonstructural_oos_unsignability_sits_out(self):
        """Spec §6 authorizes the full-window fallback ONLY for the structural
        gates. A non-structural unsignable oos arm (nonpositive_edge) means the
        deployed rule is NOT_SIGNABLE (§2) → the replay must sit the window out
        at f = 0, not keep trading the full-window fraction precisely where the
        recent-data arm says the edge is gone."""
        n = 280
        returns = [Decimal(str(r)) for r in _returns_with_dead_tail()]
        entries, expiries = _dates(n)
        wf = walkforward_validation(
            returns, entries, expiries, data_snapshot_sha256=_SHA,
            params=_fast_params(warmup_trades=210))
        sit_outs = [r for r in wf.refits
                    if r.sit_out_reason.startswith("not_signable:oos")]
        assert wf.sit_out_count >= 1
        assert sit_outs, f"no non-structural sit-out; refits={wf.refits}"
        for r in sit_outs:
            assert r.f_applied == 0.0
            assert not r.fell_back_to_full
        # fallback_count counts STRUCTURAL fallbacks only
        assert wf.fallback_count == sum(1 for r in wf.refits if r.fell_back_to_full)

    def test_below_floor_composite_sits_out(self, monkeypatch):
        """§5 step 4: f_rec < 0.005 is NOT_SIGNABLE at deployment — the replay
        must not trade at fractions the deployed rule can never emit."""
        import risk_kelly.calibration.kelly_optimization as ko

        def tiny_pipe_run_set(name, data, sha, params, prefix, log, **kw):
            return SetResult(
                name=name, n=len(data), n_losses=20, signable=True, reasons=(),
                f_star=0.01, kelly_capped=False, c_star=0.30, f_shrunk=0.003,
                f_dd=0.005, f_pipe=0.003, binding_term="shrunk_kelly",
                r_curve=(), f_star_b_deciles=(), frac_f_star_b_zero=0.0,
                frac_f_star_b_capped=0.0, p_boot_mean_nonpos=0.0,
                quantile_growth_curve=(), guard_matrix=(),
                incumbent_guard_line=(), mdd_at_frec=(), terminal_wealth=(),
                time_under_water=(), warnings=())

        monkeypatch.setattr(ko, "_run_set", tiny_pipe_run_set)
        f, _, sit_out = _composite_fit_at(
            _pattern_returns(200), _SHA, 70, _fast_params(), [])
        assert f == 0.0
        assert sit_out == "below_guard_grid_floor"

    def test_per_refit_comparator_f_table_emitted(self):
        n = 150
        returns = [Decimal(str(r)) for r in _pattern_returns(n)]
        entries, expiries = _dates(n)
        wf = walkforward_validation(
            returns, entries, expiries, data_snapshot_sha256=_SHA,
            params=_fast_params(warmup_trades=70))
        for r in wf.refits:
            assert r.f_incumbent == pytest.approx(0.08923)
            if not r.sit_out_reason and r.n_train >= 60:
                assert r.f_full_kelly >= r.f_half_kelly >= 0.0


# ---------------------------------------------------------------------------
# Shrinkage rule ordering (REVIEW F11)
# ---------------------------------------------------------------------------

class TestShrinkageRuleOrder:
    def test_mean_nonpos_checked_before_two_loss_park(self):
        """Spec §3(a) order: mean ≤ 0 → f*_b = 0 BEFORE the <2-loss park.
        Data crafted so single-loss resamples have nonpositive mean (one −0.9
        outweighs 79 × +0.001): they must resolve to 0 (fail-closed), never to
        the 0.5 cap; zero-loss resamples (positive mean) still park at 0.5."""
        returns = [0.001] * 79 + [-0.9]
        rng = random.Random(derive_seed(_SHA, "test-f11"))
        _, _, _, f_star_bs, mean_nonpos = _shrinkage(
            returns, rng, 300, 6, [0.01 * k for k in range(1, 101)])
        assert mean_nonpos == sum(1 for v in f_star_bs if v == 0.0)
        assert mean_nonpos >= 1                      # single-loss resamples → 0
        assert any(v == 0.5 for v in f_star_bs)      # zero-loss parks remain
        # No resample with a loss present may sit at the 0.5 park: with one
        # loss the mean is negative → must be 0.0 under the spec order.
        assert all(v in (0.0, 0.5) for v in f_star_bs)


# ---------------------------------------------------------------------------
# Spec §10 minimum test set (REVIEW F13)
# ---------------------------------------------------------------------------

class TestSpecMinimumPins:
    def test_s3_transform_saturates_at_minus_one(self):
        # s3_pos replaces the trade at that path position with r = −1.
        mdds = _path_mdds([0.05, 0.05], [[0, 1]], 0.5, s3_pos=[1])
        assert mdds[0] == pytest.approx(0.5, abs=1e-12)  # 1 − (1 − 0.5)
        assert _path_mdds([0.05, 0.05], [[0, 1]], 1.0, s3_pos=[1]) == [1.0]  # ruin

    def test_crn_paths_shared_across_stress_cells(self):
        """§3(b) CRN: one path set per (set, L, H, B) reused across every
        stress transform. Pinned two ways: the seeds log carries exactly ONE
        L=6 binding-path seed per run per set, and the guard matrix contains
        S0/S1/S15/S3 rows — four stress cells, one path generation."""
        rs = [Decimal(str(r)) for r in _pattern_returns(200)]
        params = _fast_params()
        res = optimize_kelly_fraction(rs, data_snapshot_sha256=_SHA, params=params)
        for set_name in ("E_full", "E_oos"):
            main_l6 = [p for p, _ in res.seeds
                       if p == f"paths({set_name},L=6,H={params.horizon},B={params.b_dd})"]
            assert len(main_l6) == 1
        cells = {r.cell for r in res.sets[0].guard_matrix}
        assert {"L6_S0", "L6_S1", "L6_S15", "L6_S3"} <= cells

    def test_isotonic_clip_consistency(self):
        """The guard decision must follow the RUNNING MAX of the Wilson UB
        (isotonic clip), not the raw per-f value: recompute the clip from the
        fingerprinted guard matrix and check f_dd matches the clipped rule."""
        rs = [Decimal(str(r)) for r in _pattern_returns(200)]
        params = _fast_params()
        res = optimize_kelly_fraction(rs, data_snapshot_sha256=_SHA, params=params)
        full = res.sets[0]
        d_idx = params.d_diag_grid.index(params.d_kill)
        s_idx = params.d_diag_grid.index(params.d_soft)
        rows = [r for r in full.guard_matrix if r.cell in ("L6_S0", "L6_S1")]
        by_f: dict = {}
        for r in rows:
            by_f.setdefault(r.f, []).append(r)
        run_max: dict = {}
        feasible_max = None
        for f in sorted(by_f):
            ok = True
            for r in by_f[f]:
                kk, ks = (r.cell, "k"), (r.cell, "s")
                run_max[kk] = max(run_max.get(kk, 0.0), r.stats[d_idx][3])
                run_max[ks] = max(run_max.get(ks, 0.0), r.stats[s_idx][3])
                if run_max[kk] > params.p_kill or run_max[ks] > params.p_soft:
                    ok = False
            if not ok:
                break
            feasible_max = f
            if f >= full.f_shrunk:
                break
        assert full.f_dd == feasible_max



# ---------------------------------------------------------------------------
# Driver report assembly (REVIEW F1 / F9 / F10 / F15)
# ---------------------------------------------------------------------------



def _report_meta() -> dict:
    return dict(spx_sha="s" * 64, vix_sha="v" * 64, series_sha="r" * 64,
                snapshot_sha="a" * 64, n_trades=200, n_losses=20,
                n_losses_oos=6, date_range=["2000-01-05", "2015-01-01"])


class TestDriverReport:
    def _candidate(self, **overrides):
        rs = [Decimal(str(r)) for r in _pattern_returns(200)]
        params = _fast_params(**overrides)
        return optimize_kelly_fraction(
            rs, data_snapshot_sha256=_SHA, params=params), params
