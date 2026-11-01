"""R3 sizing — Robust / drawdown-Constrained Kelly (RCK) convex sizer.

Provenance: clean-room reimplementation of Busseti, Ryu & Boyd, *Risk-Constrained
Kelly Gambling* (2016), and the Rockafellar-Uryasev CVaR duality it builds on.
Equations only; the convex program is expressed in cvxpy (Apache-2.0) and solved
by Clarabel/SCS (Apache-2.0/MIT). No solver source is vendored. ECOS is GPLv3
and is not a dependency of this standalone release.

------------------------------------------------------------------------------
THE PROGRAM
------------------------------------------------------------------------------
Given a `ScenarioSet` (returns R, probabilities p) — produced by `evt` from the
conditional-EVT tail — RCK sizes the bet `b` (allocation over assets, one of
which is cash) by:

    maximize    E[ log(R · b) ]              (log-growth / Kelly objective)
    subject to  b >= 0,  1ᵀb = 1
                E[ (R · b)^{-λ} ] <= 1        (drawdown / risk constraint)

The risk constraint is the Busseti-Ryu-Boyd bound: with λ = log β / log α the EXACT
program  E[(R·b)^{-λ}] ≤ 1  GUARANTEES  Prob( minimum wealth W_min < α ) <= β  — a
drawdown floor α held with confidence 1-β. Both the objective (concave) and the
constraint (convex: a negative power of an affine, positive argument) are DCP-compliant,
so the solve is deterministic to solver tolerance. NUMERICAL CAVEAT: cvxpy canonicalizes
the negative power with a RATIONAL approximation (a ~1e-5 reduction-level slack, warned at
solve time), so the returned bet satisfies the moment cap to that tolerance
(`_DRAWDOWN_FEASIBILITY_TOL`), not to machine precision; `solve_rck` re-checks the EXACT
moment in numpy and fails closed on any MATERIAL overshoot, and the seeded
`verify_drawdown_montecarlo` replay is the independent empirical check on the realized bound.

------------------------------------------------------------------------------
STATUS / RAILS
------------------------------------------------------------------------------
  * UNSIGNED — λ (and hence α, β) and the optional mean-haircut are explicit
    caller args feeding the trial registry; NO §5 constant is pinned (T8 signs).
  * DETERMINISTIC — the convex solve is deterministic; the KKT Monte-Carlo
    re-check is SEEDED and byte-reproducible given the scenario set.
  * ADVISORY — RCK sizes a bet on a scenario set; it does NOT certify an edge
    (CPCV + purged folds + deflated Sharpe stay upstream) and emits NO order.
  * DOUBLE-CONSERVATISM (rail h) — feeding RCK a haircut mean AND capping with λ
    both add conservatism; `solve_rck` takes an explicit `mean_haircut` and
    RECORDS it on the solution so the tension is surfaced, never silently stacked.
  * cvxpy is a GUARDED import (below) — importing this module never pulls cvxpy;
    only `solve_rck` needs it, and it fails closed with `SolverUnavailableError`.
  * GOVERNANCE (rail i) — Sleeve C's sacred clock-driven roll may warrant its own
    λ / exemption; that is an OPEN §5/T8 question, NOT decided here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ScenarioSet is defined ONCE, in evt — imported here (the shared evt→rck contract).
from risk_kelly.sizing.evt import ScenarioSet
from risk_kelly.sizing.errors import (
    DrawdownConstraintError,
    RCKError,
    SolverUnavailableError,
)

__all__ = [
    "lambda_from_drawdown",
    "solve_rck",
    "solve_qrck",
    "verify_kkt_montecarlo",
    "verify_drawdown_montecarlo",
    "RCKSolution",
    "KKTReport",
    "DrawdownReplay",
]

# cvxpy canonicalizes the convex constraint  E[(R·b)^{-λ}] ≤ 1  by approximating the
# fractional/negative exponent with a RATIONAL power (a UserWarning it emits), which
# introduces a reduction-level slack of ~1e-5 in the drawdown moment — its sign in λ is
# essentially arbitrary, so a genuinely-optimal bet can land at exact-moment 1.0000072.
# Any feasibility CHECK on the returned bet must therefore use a tolerance that comfortably
# exceeds that approximation error, or it will false-negative a valid solve. 1e-4 clears it
# by an order of magnitude while still being far tighter than any material drawdown loosening
# (over a horizon H a 1e-4 per-step overshoot degrades the bound by ~(1+1e-4)^H — negligible).
_DRAWDOWN_FEASIBILITY_TOL = 1e-4


# --------------------------------------------------------------------------- #
# λ from a drawdown floor (α) and its allowed breach probability (β)          #
# --------------------------------------------------------------------------- #
def lambda_from_drawdown(alpha: float, beta: float) -> float:
    """RCK risk-aversion λ from a drawdown floor α and breach probability β.

    Busseti-Ryu-Boyd: the constraint  E[(R·b)^{-λ}] <= 1  with

        λ = log(β) / log(α)

    implies  Prob(W_min < α) <= β.  α ∈ (0,1) is the wealth floor (e.g. 0.7 = a
    30% max drawdown), β ∈ (0,1) the tolerated probability of breaching it. Both
    are UNSIGNED research inputs (T8 signs them). Tighter floor (α→1) or smaller
    β → larger λ → more conservative sizing.
    """
    if not (0.0 < alpha < 1.0):
        raise RCKError(f"alpha (drawdown floor) must be in (0,1); got {alpha!r}")
    if not (0.0 < beta < 1.0):
        raise RCKError(f"beta (breach probability) must be in (0,1); got {beta!r}")
    lam = math.log(beta) / math.log(alpha)
    if not (np.isfinite(lam) and lam > 0.0):
        raise RCKError(f"computed lambda is not a positive finite number: {lam!r}")
    return lam


# --------------------------------------------------------------------------- #
# solution + verifier value objects                                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RCKSolution:
    """Result of the RCK convex solve."""

    b: np.ndarray                 # optimal allocation (n_assets,)
    lam: float                    # λ used
    expected_log_growth: float    # E[log(R·b)] at b*
    drawdown_moment: float        # E[(R·b)^{-λ}] at b*  (binds ⇔ ≈ 1)
    drawdown_binds: bool          # constraint active within tolerance
    status: str                   # solver status
    mean_haircut: float = 0.0     # haircut applied to the edge (rail h)
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class KKTReport:
    """Seeded Monte-Carlo optimality / feasibility re-check of a candidate b."""

    feasible: bool                # b >= 0, 1ᵀb = 1, drawdown ≤ 1 (within tol)
    is_optimum: bool              # no seeded feasible perturbation improved growth
    max_improvement: float        # largest growth gain found (≤ tol ⇒ optimum)
    drawdown_moment: float        # E[(R·b)^{-λ}]
    drawdown_binds: bool          # constraint active
    sum_b: float
    min_b: float
    seed: int
    n_probes: int


# --------------------------------------------------------------------------- #
# cvxpy — GUARDED import (module import never requires it)                     #
# --------------------------------------------------------------------------- #
def _require_cvxpy():
    try:
        import cvxpy  # noqa: WPS433 (deliberate lazy import — fail closed)
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise SolverUnavailableError(
            "cvxpy is not installed — RCK cannot solve the convex program. "
            "Install it into the build venv (`pip install cvxpy` — MIT/Apache/BSD) "
            "to run the sizer; the rest of the sizing package imports without it."
        ) from exc
    return cvxpy


def _haircut_returns(returns: np.ndarray, mean_haircut: float) -> np.ndarray:
    """Deflate the EDGE of each scenario: r → 1 + (1-h)(r-1) (rail h).

    Shrinks every scenario's excess return toward the no-edge point 1.0, per
    Chopra-Ziemba. h=0 is the identity. Applied to the RCK input, and recorded on
    the solution so the mean-haircut-vs-λ double-conservatism stays visible.
    """
    if not (0.0 <= mean_haircut < 1.0):
        raise RCKError(f"mean_haircut must be in [0,1); got {mean_haircut!r}")
    if mean_haircut == 0.0:
        return returns
    return 1.0 + (1.0 - mean_haircut) * (returns - 1.0)


def solve_rck(
    scenarios: ScenarioSet,
    lam: float,
    *,
    mean_haircut: float = 0.0,
    solver: Optional[str] = None,
    binds_tol: float = 1e-5,
    verbose: bool = False,
) -> RCKSolution:
    """Solve the drawdown-constrained Kelly program for the given ScenarioSet.

    λ is the risk-aversion (see `lambda_from_drawdown`). `mean_haircut` optionally
    deflates the edge before sizing (rail h). Fails closed:
      * `SolverUnavailableError` if cvxpy is absent,
      * `RCKError` if the program is infeasible / the solver does not converge.
    Deterministic to solver tolerance (convex, no randomness).
    """
    if not (np.isfinite(lam) and lam > 0.0):
        raise RCKError(f"lambda must be a positive finite number; got {lam!r}")
    cp = _require_cvxpy()

    R = _haircut_returns(np.asarray(scenarios.returns, dtype=np.float64), mean_haircut)
    p = np.asarray(scenarios.probabilities, dtype=np.float64)
    n_assets = R.shape[1]

    b = cp.Variable(n_assets, nonneg=True)
    g = R @ b  # affine, positive on the feasible set
    objective = cp.Maximize(p @ cp.log(g))
    constraints = [
        cp.sum(b) == 1,
        p @ cp.power(g, -float(lam)) <= 1.0,  # convex: negative power of positive affine
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=verbose)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise DrawdownConstraintError(
            f"RCK program did not solve to optimality (status={problem.status!r}); "
            "the drawdown constraint may be infeasible for this scenario set / λ"
        )
    b_val = np.asarray(b.value, dtype=np.float64)
    b_val = np.clip(b_val, 0.0, None)
    s = b_val.sum()
    if s <= 0.0:
        raise RCKError("solver returned a degenerate all-zero allocation")
    b_val = b_val / s  # renormalize away tiny solver slack

    # Re-evaluate the EXACT drawdown moment in numpy at the true λ on the returns the solver
    # ACTUALLY constrained (the haircut R, not the raw scenarios) — this checks the cvxpy
    # rational-power approximation of the exponent. A tiny overshoot (~1e-5, either sign) is the
    # reduction-level slack and is accepted; a MATERIAL overshoot means the solver did not satisfy
    # its OWN constraint, so fail closed rather than certify an over-risked bet.
    g_star = R @ b_val
    constrained_dd = float(np.sum(p * g_star ** (-float(lam))))
    if constrained_dd > 1.0 + _DRAWDOWN_FEASIBILITY_TOL:
        raise DrawdownConstraintError(
            f"RCK solve returned status={problem.status!r} but the exact drawdown moment "
            f"E[(R·b)^-λ]={constrained_dd!r} on the constrained returns exceeds 1 by more than "
            f"the solver tolerance ({_DRAWDOWN_FEASIBILITY_TOL}) — cap not satisfied, fail closed"
        )
    # REPORTED drawdown_moment is on the RAW scenarios (the real tail). With mean_haircut>0 the
    # haircut compresses the tail too, so the raw-return moment can exceed 1 even though the solve
    # satisfied its (softer) constraint — that mean-haircut-vs-λ double-conservatism is SURFACED
    # here (rail h), not silently stacked; the caller must reconcile the two (see docstring).
    dd = scenarios.drawdown_moment(b_val, lam)
    return RCKSolution(
        b=b_val,
        lam=float(lam),
        expected_log_growth=scenarios.expected_log_growth(b_val),
        drawdown_moment=dd,
        drawdown_binds=bool(dd >= 1.0 - binds_tol),  # active ⇔ within binds_tol of the cap
        status=str(problem.status),
        mean_haircut=float(mean_haircut),
    )


# --------------------------------------------------------------------------- #
# verify_kkt_montecarlo — seeded feasibility + optimality re-check            #
# --------------------------------------------------------------------------- #
def verify_kkt_montecarlo(
    scenarios: ScenarioSet,
    b: np.ndarray,
    lam: float,
    *,
    seed: int,
    n_probes: int = 2000,
    step: float = 1e-3,
    tol: float = 1e-6,
) -> KKTReport:
    """Independently re-check a candidate allocation `b` (SEEDED, reproducible).

    Confirms primal feasibility (b ≥ 0, 1ᵀb = 1, drawdown moment ≤ 1) and probes
    optimality: it draws `n_probes` seeded feasible tangent perturbations
    (Σd = 0, kept non-negative and drawdown-feasible) and checks that none raises
    E[log(R·b)] by more than `tol` — a Monte-Carlo KKT/optimality certificate that
    does NOT need cvxpy, so it independently audits the convex solve. Also reports
    whether the drawdown constraint BINDS (rail g / acceptance criterion).
    """
    if not (np.isfinite(lam) and lam > 0.0):
        raise RCKError(f"lambda must be a positive finite number; got {lam!r}")
    b = np.asarray(b, dtype=np.float64).ravel()
    n_assets = scenarios.n_assets
    if b.shape != (n_assets,):
        raise RCKError(f"b must have shape ({n_assets},); got {b.shape}")

    sum_b = float(b.sum())
    min_b = float(b.min())
    # feasibility
    feasible_simplex = abs(sum_b - 1.0) <= 1e-6 and min_b >= -1e-9
    try:
        base_growth = scenarios.expected_log_growth(b)
        dd = scenarios.drawdown_moment(b, lam)
    except Exception:  # infeasible growth (non-positive R·b somewhere)
        return KKTReport(
            feasible=False, is_optimum=False, max_improvement=math.inf,
            drawdown_moment=math.inf, drawdown_binds=False,
            sum_b=sum_b, min_b=min_b, seed=seed, n_probes=0,
        )
    # NOTE: tolerance is _DRAWDOWN_FEASIBILITY_TOL (1e-4), NOT 1e-6 — the tighter bound
    # false-negatived optimal solves whose exact moment sits at 1.0000072 because of the
    # cp.power rational approximation (see the module constant). This aligns the feasibility
    # check with the drawdown-binds tolerance used below.
    feasible = bool(feasible_simplex and dd <= 1.0 + _DRAWDOWN_FEASIBILITY_TOL)

    rng = np.random.default_rng(seed)
    max_improvement = 0.0
    if n_assets >= 2:
        for _ in range(n_probes):
            d = rng.standard_normal(n_assets)
            d -= d.mean()  # keep on the Σb = 1 simplex tangent
            nrm = np.linalg.norm(d)
            if nrm == 0.0:
                continue
            d = d / nrm
            cand = b + step * d
            if np.any(cand < 0.0):
                continue
            cand = cand / cand.sum()
            try:
                cand_dd = scenarios.drawdown_moment(cand, lam)
                if cand_dd > 1.0 + _DRAWDOWN_FEASIBILITY_TOL:  # violates the drawdown constraint
                    continue
                gain = scenarios.expected_log_growth(cand) - base_growth
            except Exception:
                continue
            if gain > max_improvement:
                max_improvement = gain

    return KKTReport(
        feasible=feasible,
        is_optimum=bool(feasible and max_improvement <= tol),
        max_improvement=float(max_improvement),
        drawdown_moment=float(dd),
        drawdown_binds=bool(abs(dd - 1.0) <= 1e-4),
        sum_b=sum_b,
        min_b=min_b,
        seed=seed,
        n_probes=n_probes,
    )


# --------------------------------------------------------------------------- #
# solve_qrck — the quadratic (Taylor) RCK that SUBSUMES Markowitz              #
# --------------------------------------------------------------------------- #
def solve_qrck(
    scenarios: ScenarioSet,
    lam: float,
    *,
    mean_haircut: float = 0.0,
    solver: Optional[str] = None,
    binds_tol: float = 1e-5,
    verbose: bool = False,
) -> RCKSolution:
    """Quadratic-RCK: the 2nd-order Taylor of the RCK program around ``R·b ≈ 1``.

    With ``x = R·b`` and ``log x ≈ (x-1) - (x-1)²/2``, the RCK objective becomes the
    concave quadratic ``E[(x-1)] - ½E[(x-1)²]`` and the drawdown moment
    ``E[x^{-λ}] ≤ 1`` linearizes (2nd order) to ``E[x-1] ≥ ½(λ+1)E[(x-1)²]``. The
    objective is *identically* a Markowitz mean / second-moment portfolio, so in the
    light-aversion limit ``λ → 0`` (the risk constraint slack) QRCK's optimum IS a
    Markowitz mean-variance portfolio — i.e. RCK subsumes both Kelly and Markowitz,
    and QRCK is the bridge (see ``test_sizing_rck.py::TestQRCKisMarkowitz``).

    ``λ ≥ 0`` is allowed here (unlike ``solve_rck``, which requires ``λ > 0``): ``λ=0``
    is the pure-Markowitz corner. Fails closed exactly like ``solve_rck``. The reported
    ``expected_log_growth`` / ``drawdown_moment`` are the TRUE (non-Taylor) functionals
    at the QRCK optimum, so QRCK and RCK solutions are directly comparable.
    """
    if not (np.isfinite(lam) and lam >= 0.0):
        raise RCKError(f"lambda must be a finite number >= 0; got {lam!r}")
    cp = _require_cvxpy()

    R = _haircut_returns(np.asarray(scenarios.returns, dtype=np.float64), mean_haircut)
    p = np.asarray(scenarios.probabilities, dtype=np.float64)
    n_assets = R.shape[1]

    b = cp.Variable(n_assets, nonneg=True)
    net = R @ b - 1.0                     # per-scenario excess (affine)
    e_net = p @ net                       # E[x-1]
    e_net2 = p @ cp.square(net)           # E[(x-1)^2] (2nd moment of net return)
    objective = cp.Maximize(e_net - 0.5 * e_net2)
    constraints = [cp.sum(b) == 1]
    if lam > 0.0:
        # Taylor of E[x^{-λ}] <= 1  ⇔  E[x-1] - ½(λ+1)E[(x-1)²] >= 0  (concave ≥ 0).
        constraints.append(e_net - 0.5 * (lam + 1.0) * e_net2 >= 0)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=verbose)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise DrawdownConstraintError(
            f"QRCK program did not solve to optimality (status={problem.status!r})"
        )
    b_val = np.clip(np.asarray(b.value, dtype=np.float64), 0.0, None)
    s = b_val.sum()
    if s <= 0.0:
        raise RCKError("QRCK solver returned a degenerate all-zero allocation")
    b_val = b_val / s

    if lam > 0.0:
        dd = scenarios.drawdown_moment(b_val, lam)
        binds = bool(dd >= 1.0 - binds_tol)  # active ⇔ within binds_tol of the cap
    else:
        dd = float("nan")               # drawdown moment undefined at λ=0 (Markowitz corner)
        binds = False
    return RCKSolution(
        b=b_val,
        lam=float(lam),
        expected_log_growth=scenarios.expected_log_growth(b_val),
        drawdown_moment=dd,
        drawdown_binds=binds,
        status=str(problem.status),
        mean_haircut=float(mean_haircut),
        meta={"method": "QRCK"},
    )


# --------------------------------------------------------------------------- #
# verify_drawdown_montecarlo — compounded-wealth W_min replay (the 0.9645 test)#
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DrawdownReplay:
    """Seeded Monte-Carlo replay of compounded wealth under a bet ``b`` (rail g).

    The DIRECT falsification of a self-reported ruin figure: it compounds i.i.d.
    draws from the scenario set under ``b`` and measures the realized
    ``Prob(W_min < alpha)`` against the RCK provable cap ``alpha ** lambda``.
    The mathematical bound concerns the scenario law; a finite Monte Carlo
    estimate can exceed it by sampling error and is not itself a guarantee.
    """

    alpha: float                 # drawdown floor tested (fraction of starting wealth)
    empirical_prob: float        # realized Prob(W_min < alpha) across the replayed paths
    theoretical_bound: float     # alpha ** lambda  (the Busseti-Ryu-Boyd cap)
    bound_holds: bool            # empirical_prob <= theoretical_bound
    margin_fraction: float       # (bound - empirical) / bound  (>0 ⇒ conservative)
    n_paths: int
    horizon: int
    seed: int


def verify_drawdown_montecarlo(
    scenarios: ScenarioSet,
    b: np.ndarray,
    lam: float,
    *,
    alpha: float,
    seed: int,
    n_paths: int = 20_000,
    horizon: int = 250,
) -> DrawdownReplay:
    """Replay compounded wealth under ``b`` and check the realized drawdown CDF.

    Simulates ``n_paths`` wealth paths of length ``horizon`` by i.i.d.-sampling the
    scenario set under its probabilities and compounding the per-scenario gross return
    ``R·b``; returns the realized ``Prob(W_min < alpha)`` and compares it to the provable
    ``alpha ** lambda`` bound. SEEDED and byte-reproducible given the scenario set. This is
    the drawdown-CDF half of the RCK self-verifier (``verify_kkt_montecarlo`` is the KKT /
    optimality half) — together they audit both optimality and the realized drawdown cap,
    a finite-sample diagnostic, not a certificate of a historical trading result.
    """
    if not (np.isfinite(lam) and lam > 0.0):
        raise RCKError(f"lambda must be a positive finite number; got {lam!r}")
    if not (0.0 < alpha < 1.0):
        raise RCKError(f"alpha (drawdown floor) must be in (0,1); got {alpha!r}")
    if n_paths < 1 or horizon < 1:
        raise RCKError("n_paths and horizon must be >= 1")

    b = np.asarray(b, dtype=np.float64).ravel()
    if b.shape != (scenarios.n_assets,):
        raise RCKError(f"b must have shape ({scenarios.n_assets},); got {b.shape}")

    gross = scenarios.growth(b)                     # (S,) per-scenario gross return (>0)
    p = np.asarray(scenarios.probabilities, dtype=np.float64)
    log_gross = np.log(gross)

    rng = np.random.default_rng(seed)
    idx = rng.choice(gross.shape[0], size=(n_paths, horizon), p=p)
    log_wealth = np.cumsum(log_gross[idx], axis=1)  # log W_t, t = 1..horizon
    # W_min over the whole path INCLUDING the starting wealth (log 0).
    min_log_wealth = np.minimum(log_wealth.min(axis=1), 0.0)
    w_min = np.exp(min_log_wealth)
    empirical = float(np.mean(w_min < alpha))
    bound = float(alpha ** lam)
    margin = float((bound - empirical) / bound) if bound > 0.0 else float("nan")

    return DrawdownReplay(
        alpha=float(alpha),
        empirical_prob=empirical,
        theoretical_bound=bound,
        bound_holds=bool(empirical <= bound + 1e-12),
        margin_fraction=margin,
        n_paths=int(n_paths),
        horizon=int(horizon),
        seed=int(seed),
    )
