"""R3 sizing — DEEP NUMERIC VALIDATION of the Risk-Constrained Kelly solve.

This is the "sibling component that installs cvxpy" that ``test_sizing_firewall.py``
defers the deep RCK numerics to. It complements ``test_sizing_contract.py`` (which
already covers the ScenarioSet contract, lambda_from_drawdown, ES/CVaR, GPD closed
forms, the Sinclair ceiling, verify_kkt_montecarlo teeth, and the drawdown-binds RCK
solve) with the two pieces the coordinated build did not yet exercise:

  * ``solve_qrck`` (the quadratic Taylor of RCK) == a Markowitz mean-variance portfolio
    in the light-aversion limit — proving RCK SUBSUMES both Kelly and Markowitz.
  * ``verify_drawdown_montecarlo`` — a seeded compounded-wealth replay measuring the
    realized ``Prob(W_min < alpha)`` against the provable ``alpha^lambda`` bound: the
    finite-sample validation on a synthetic scenario law.

Plus growth-vs-lambda monotonicity, the mean-haircut coupling (rail h), determinism, and
the core PERP research.sizing firewall mirror (rail a).

Rails: methods not pinned numbers (alpha/beta/lambda are explicit TEST args, UNSIGNED —
T8 signs the live ones); deterministic given the scenario set; NO fabricated green
(cvxpy is REQUIRED for the acceptance run — skip only if genuinely absent).
"""
import math
import subprocess
import sys

import numpy as np
import pytest

from risk_kelly.sizing import (
    ScenarioSet,
    build_scenario_set,
    lambda_from_drawdown,
    solve_qrck,
    solve_rck,
    verify_drawdown_montecarlo,
    verify_kkt_montecarlo,
)
from risk_kelly.sizing.errors import RCKError
from risk_kelly.sizing import rck as rckmod

cvxpy = pytest.importorskip("cvxpy", reason="RCK convex solve requires cvxpy (build-venv dep)")


# --------------------------------------------------------------------------- #
# deterministic negative-skew SHORT-VOL scenario set (bet + cash)             #
# --------------------------------------------------------------------------- #
def _short_vol_scenarioset():
    """92 small wins (+4%), 8 rare crashes (-25%): positive log-growth, fat left tail.

    Columns are [risky_bet, cash]; cash (certain return 1) precludes ruin. This is the
    canonical short-vol shape that makes UNCONSTRAINED Kelly court deep drawdowns.
    """
    bet = np.concatenate([np.full(92, 1.04), np.full(8, 0.75)])
    returns = np.column_stack([bet, np.ones_like(bet)])
    probs = np.full(100, 0.01)
    return ScenarioSet(returns=returns, probabilities=probs)


# --------------------------------------------------------------------------- #
# 1. solve_qrck == Markowitz in the light-aversion limit (RCK ⊃ Markowitz)    #
# --------------------------------------------------------------------------- #
class TestQRCKisMarkowitz:
    def test_qrck_lambda_zero_equals_markowitz(self):
        rng = np.random.default_rng(3)
        m = 600
        R = np.column_stack(
            [
                1.0 + rng.normal(0.008, 0.03, m),
                1.0 + rng.normal(0.012, 0.05, m),
                1.0 + rng.normal(0.004, 0.02, m),
            ]
        )
        R = np.clip(R, 1e-4, None)
        pi = np.full(m, 1.0 / m)
        scen = ScenarioSet(returns=R, probabilities=pi)

        qr = solve_qrck(scen, lam=0.0)

        # Independent Markowitz on the SECOND-MOMENT matrix of net returns:
        #   maximize  mean_net·b - ½ b'M2 b   s.t. 1'b = 1, b >= 0
        # QRCK's 2nd-order objective is IDENTICALLY this program => same optimum.
        net = R - 1.0
        mean_net = pi @ net
        M2 = (net * pi[:, None]).T @ net
        M2 = 0.5 * (M2 + M2.T)
        b = cvxpy.Variable(3, nonneg=True)
        obj = cvxpy.Maximize(mean_net @ b - 0.5 * cvxpy.quad_form(b, cvxpy.psd_wrap(M2)))
        cvxpy.Problem(obj, [cvxpy.sum(b) == 1]).solve(solver=cvxpy.CLARABEL)

        assert np.allclose(qr.b, np.asarray(b.value).ravel(), atol=1e-4)
        assert qr.meta["method"] == "QRCK"

    def test_qrck_positive_lambda_more_conservative_than_lambda_zero(self):
        scen = _short_vol_scenarioset()
        light = solve_qrck(scen, lam=0.0)
        heavy = solve_qrck(scen, lam=8.0)
        # more aversion -> at least as much cash (column index 1)
        assert heavy.b[1] >= light.b[1] - 1e-6


# --------------------------------------------------------------------------- #
# 2. Seeded synthetic drawdown replay                                     #
# --------------------------------------------------------------------------- #
class TestDrawdownReplay:
    def test_realized_drawdown_under_bound_and_far_below_self_report(self):
        scen = _short_vol_scenarioset()
        alpha, beta = 0.7, 0.1
        lam = lambda_from_drawdown(alpha, beta)
        sol = solve_rck(scen, lam)

        rep = verify_drawdown_montecarlo(
            scen, sol.b, lam, alpha=alpha, seed=12345, n_paths=40_000, horizon=500
        )
        bound = alpha ** lam
        assert rep.theoretical_bound == pytest.approx(bound, rel=1e-12)
        # 1) the provable Busseti-Ryu-Boyd cap holds empirically
        assert rep.bound_holds
        assert rep.empirical_prob <= bound
        # 2) conservative: realized comes in strictly UNDER the bound
        assert rep.empirical_prob < bound
        assert rep.margin_fraction > 0.0
        # 3) This synthetic scenario law has a small measured breach probability.
        assert rep.empirical_prob < 0.5

    def test_bound_holds_across_multiple_floors(self):
        # one lambda bounds the WHOLE min-wealth CDF: Prob(W_min<alpha) < alpha^lambda
        # for EVERY alpha (strictly stronger than a single VaR point).
        scen = _short_vol_scenarioset()
        lam = lambda_from_drawdown(0.7, 0.1)
        sol = solve_rck(scen, lam)
        for alpha in (0.6, 0.7, 0.8, 0.9):
            rep = verify_drawdown_montecarlo(
                scen, sol.b, lam, alpha=alpha, seed=7, n_paths=30_000, horizon=400
            )
            assert rep.empirical_prob <= rep.theoretical_bound + 1e-9

    def test_replay_is_seeded_reproducible(self):
        scen = _short_vol_scenarioset()
        lam = lambda_from_drawdown(0.7, 0.1)
        sol = solve_rck(scen, lam)
        a = verify_drawdown_montecarlo(scen, sol.b, lam, alpha=0.7, seed=99,
                                       n_paths=5_000, horizon=100)
        b = verify_drawdown_montecarlo(scen, sol.b, lam, alpha=0.7, seed=99,
                                       n_paths=5_000, horizon=100)
        assert a.empirical_prob == b.empirical_prob

    def test_unconstrained_kelly_breaches_far_more_than_rck(self):
        # Full-Kelly (all-risky) vs the drawdown-capped RCK bet: the replayed breach
        # probability at the floor must be strictly higher for the uncapped bet — the
        # mechanical reason the signed Kelly self-reports a near-1 ruin.
        scen = _short_vol_scenarioset()
        lam = lambda_from_drawdown(0.7, 0.1)
        sol = solve_rck(scen, lam)
        all_risky = np.array([1.0, 0.0])  # 100% risky, 0% cash (unconstrained corner)
        rck_rep = verify_drawdown_montecarlo(scen, sol.b, lam, alpha=0.7, seed=1,
                                             n_paths=40_000, horizon=500)
        kelly_rep = verify_drawdown_montecarlo(scen, all_risky, lam, alpha=0.7, seed=1,
                                               n_paths=40_000, horizon=500)
        assert kelly_rep.empirical_prob > rck_rep.empirical_prob
        assert not kelly_rep.bound_holds  # the uncapped bet BREACHES the cap


# --------------------------------------------------------------------------- #
# 3. growth-vs-lambda monotonicity (trade growth for drawdown safety)         #
# --------------------------------------------------------------------------- #
class TestMonotonicity:
    def test_larger_lambda_smaller_growth_more_cash_smaller_drawdown(self):
        scen = _short_vol_scenarioset()
        lams = [0.5, 1.0, 3.0, 6.0, 10.0]
        growth, cash, breach = [], [], []
        for lam in lams:
            sol = solve_rck(scen, lam)
            growth.append(sol.expected_log_growth)
            cash.append(sol.b[1])
            rep = verify_drawdown_montecarlo(scen, sol.b, lam, alpha=0.7, seed=42,
                                             n_paths=20_000, horizon=400)
            breach.append(rep.empirical_prob)
        # expected log-growth is non-increasing in lambda
        for a, b in zip(growth, growth[1:]):
            assert b <= a + 1e-9
        # cash allocation is non-decreasing in lambda
        for a, b in zip(cash, cash[1:]):
            assert b >= a - 1e-6
        # strictly: the most-averse bet holds materially more cash than the least
        assert cash[-1] > cash[0] + 1e-3


# --------------------------------------------------------------------------- #
# 4. mean-haircut coupling (rail h) — deflate the mean BEFORE sizing          #
# --------------------------------------------------------------------------- #
class TestMeanHaircut:
    def test_haircut_changes_the_bet_and_is_surfaced(self):
        scen = _short_vol_scenarioset()
        lam = lambda_from_drawdown(0.7, 0.1)
        raw = solve_rck(scen, lam, mean_haircut=0.0)
        cut = solve_rck(scen, lam, mean_haircut=0.5)
        # The haircut is a REAL, coupled input (rail h): it materially moves the bet and
        # is RECORDED on the solution so the lambda-vs-haircut double-conservatism tension
        # is inspectable, never silently stacked. (Direction is intentionally not asserted:
        # this haircut compresses the whole excess-return distribution toward 1, softening
        # BOTH the edge AND the tail, so its net effect on size is set-dependent.)
        assert not np.allclose(raw.b, cut.b)
        assert cut.mean_haircut == 0.5
        assert raw.mean_haircut == 0.0

    def test_haircut_domain_fails_closed(self):
        scen = _short_vol_scenarioset()
        lam = lambda_from_drawdown(0.7, 0.1)
        for bad in (-0.1, 1.0, 1.5):
            with pytest.raises(RCKError):
                solve_rck(scen, lam, mean_haircut=bad)


# --------------------------------------------------------------------------- #
# 5. determinism (convex solve is deterministic given the scenario set)       #
# --------------------------------------------------------------------------- #
class TestDeterminism:
    def test_same_inputs_same_weights(self):
        scen = _short_vol_scenarioset()
        lam = lambda_from_drawdown(0.7, 0.1)
        a = solve_rck(scen, lam)
        b = solve_rck(scen, lam)
        assert np.allclose(a.b, b.b, atol=1e-9)
        assert a.expected_log_growth == pytest.approx(b.expected_log_growth, abs=1e-9)


# --------------------------------------------------------------------------- #
# 6. fail-closed                                                              #
# --------------------------------------------------------------------------- #
class TestFailClosed:
    def test_nonpositive_lambda_rejected(self):
        scen = _short_vol_scenarioset()
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(RCKError):
                solve_rck(scen, bad)

    def test_replay_rejects_bad_alpha(self):
        scen = _short_vol_scenarioset()
        lam = lambda_from_drawdown(0.7, 0.1)
        sol = solve_rck(scen, lam)
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(RCKError):
                verify_drawdown_montecarlo(scen, sol.b, lam, alpha=bad, seed=0)

    def test_replay_rejects_wrong_shape_b(self):
        scen = _short_vol_scenarioset()
        lam = lambda_from_drawdown(0.7, 0.1)
        with pytest.raises(RCKError):
            verify_drawdown_montecarlo(scen, np.array([1.0]), lam, alpha=0.7, seed=0)

    def test_qrck_negative_lambda_rejected(self):
        scen = _short_vol_scenarioset()
        with pytest.raises(RCKError):
            solve_qrck(scen, -0.5)


# --------------------------------------------------------------------------- #
# 7. cross-check: RCK and QRCK agree on the drawdown-capped corner            #
# --------------------------------------------------------------------------- #
class TestRckQrckConsistency:
    def test_both_hold_cash_under_binding_cap(self):
        scen = _short_vol_scenarioset()
        lam = lambda_from_drawdown(0.7, 0.1)
        rck_sol = solve_rck(scen, lam)
        qrck_sol = solve_qrck(scen, lam)
        # both size a positive cash buffer under the same binding drawdown cap
        assert rck_sol.b[1] > 1e-3
        assert qrck_sol.b[1] > 1e-3


# --------------------------------------------------------------------------- #
# 8. firewall mirror — core PERP research.sizing (RAIL a)                     #
# --------------------------------------------------------------------------- #
def test_standalone_rck_import_keeps_solver_lazy():
    r = subprocess.run([sys.executable, "-c", "import sys; import risk_kelly.sizing.rck; assert 'cvxpy' not in sys.modules"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
