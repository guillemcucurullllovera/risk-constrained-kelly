"""R3 sizing — the end-to-end EVT → RCK → Sinclair-ceiling WIRING.

This module is the single call that runs the one-directional R3 pipeline the package
docstring describes, joining the three components without merging them:

    pnl fills ─▶ evt.build_evt_scenario_set   (conditional-EVT / GPD-POT fat tail ⊕ body)
              ─▶ rck.solve_rck                 (drawdown-constrained Kelly: E[g^{-λ}] ≤ 1)
              ─▶ sizing_ceiling.apply_ceiling  (Sinclair fraction-of-Kelly clamp + haircut)
              ─▶ verify_drawdown_montecarlo    (seeded replay: realized Prob(W_min<α) ≤ β)

RCK CONSUMES the EVT fat-tailed distribution (EVT sizes the TAIL; RCK sizes the BET on the
resulting scenario set); the Sinclair ceiling is the CAP that clamps RCK's risky weight from
above with operator discipline the convex optimizer cannot see. The returned
``SizedPosition`` records EVERY layer (RCK weight, ceiling factors, which layer binds, the
replayed drawdown bound) so nothing stacks silently (rail h).

STATUS / RAILS:
  * UNSIGNED — α, β (→ λ), tail_fraction, and every Sinclair coefficient are explicit caller
    args feeding the trial registry; NO §5 constant is pinned here (T8 signs them).
  * ADVISORY — this SIZES a bet on a scenario set; it certifies NO edge (CPCV + purged folds +
    deflated Sharpe stay upstream) and emits NO live order.
  * DETERMINISTIC — the EVT splice and the MC replay are SEEDED; the convex solve is
    deterministic to solver tolerance.
  * core ⊥ research — nothing here is importable by `heimdall.core` (firewall-tested).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from risk_kelly.sizing.evt import ScenarioSet, build_evt_scenario_set
from risk_kelly.sizing.rck import (
    DrawdownReplay,
    RCKSolution,
    lambda_from_drawdown,
    solve_rck,
    verify_drawdown_montecarlo,
)
from risk_kelly.sizing.sizing_ceiling import (
    CeilingInputs,
    CeilingResult,
    apply_ceiling,
)

__all__ = ["SizedPosition", "size_position"]


@dataclass(frozen=True)
class SizedPosition:
    """The full R3 sizing decision — every layer surfaced (rail h)."""

    final_weight: float           # the size actually recommended (RCK ∧ Sinclair ceiling)
    rck_weight: float             # RCK risky-asset weight BEFORE the ceiling
    binding_layer: str            # "rck" | "sinclair_ceiling" | "regime_kill" | "no_bet"
    alpha: float                  # drawdown floor tested
    beta: float                   # tolerated breach probability
    lam: float                    # λ = log β / log α
    rck_solution: RCKSolution
    ceiling_result: CeilingResult
    drawdown_replay: Optional[DrawdownReplay] = None
    scenario_meta: dict = field(default_factory=dict)


def size_position(
    pnl_returns: np.ndarray,
    *,
    alpha: float,
    beta: float,
    # --- EVT splice knobs ---
    tail_fraction: float = 0.10,
    n_tail: int = 4000,
    evt_filter: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    gross_floor: float = 1e-3,
    # --- RCK knobs ---
    mean_haircut: float = 0.0,
    # --- Sinclair ceiling knobs (all UNSIGNED caller args) ---
    kelly_fraction: float,
    mean_shrink: float,
    estimation_coef: float,
    skew_coef: float,
    gate_signal: bool = False,
    is_fundamental_revaluation: bool = False,
    # --- verification ---
    seed: int,
    replay: bool = True,
    replay_paths: int = 20_000,
    replay_horizon: int = 250,
) -> SizedPosition:
    """Run the whole R3 pipeline on a per-period NET return series and size the bet.

    ``pnl_returns`` are per-period net returns on risk capital (e.g. a short-put-spread
    book: many small credits, a fat left tail). ``alpha``/``beta`` set the drawdown floor
    and its tolerated breach probability (→ λ). The EVT layer builds a fat-tailed
    ``[risky, cash]`` scenario set; RCK sizes the drawdown-capped bet on it; the Sinclair
    ceiling clamps the risky weight; the seeded replay measures the realized
    ``Prob(W_min<α)`` against the provable ``α^λ`` bound. Fails closed via the component
    errors (``EVTError`` / ``RCKError`` / ``SizingCeilingError``). DETERMINISTIC given
    ``seed`` and the inputs.
    """
    x = np.asarray(pnl_returns, dtype=np.float64).ravel()
    lam = lambda_from_drawdown(alpha, beta)

    # 1) EVT: splice the fat GPD tail onto the empirical body → ScenarioSet [risky, cash].
    scen: ScenarioSet = build_evt_scenario_set(
        x,
        tail_fraction=tail_fraction,
        n_tail=n_tail,
        seed=seed,
        filter=evt_filter,
        gross_floor=gross_floor,
    )

    # 2) RCK: solve the drawdown-constrained Kelly on the EVT scenario set.
    rck_sol = solve_rck(scen, lam, mean_haircut=mean_haircut)
    rck_weight = float(rck_sol.b[0])  # risky-asset weight (column 0); column 1 is cash

    # 3) Sinclair ceiling: clamp the RCK risky weight from above (moments from the body).
    #    mu/sigma2/skew are the RAW per-period moments; apply_ceiling deflates mu internally.
    mu = float(np.mean(x))
    sigma2 = float(np.var(x))
    # Fisher skew of the per-period returns (0 if degenerate variance).
    if sigma2 > 0.0:
        skew = float(np.mean(((x - mu) / np.sqrt(sigma2)) ** 3))
    else:
        skew = 0.0
    ceiling_inputs = CeilingInputs(mu=mu, sigma2=sigma2, skew=skew, sample_n=int(x.size))
    ceiling = apply_ceiling(
        rck_weight,
        ceiling_inputs,
        kelly_fraction=kelly_fraction,
        mean_shrink=mean_shrink,
        estimation_coef=estimation_coef,
        skew_coef=skew_coef,
        gate_signal=gate_signal,
        is_fundamental_revaluation=is_fundamental_revaluation,
    )

    if ceiling.killed:
        binding_layer = "regime_kill"
    elif ceiling.f_ceiling <= 0.0:
        binding_layer = "no_bet"
    elif ceiling.reason.startswith("RCK size binds"):
        binding_layer = "rck"
    else:
        binding_layer = "sinclair_ceiling"

    # 4) seeded drawdown replay on the FINAL sized bet [final_weight risky, rest cash].
    dd_replay: Optional[DrawdownReplay] = None
    if replay:
        final_w = float(np.clip(ceiling.f_ceiling, 0.0, 1.0))
        b_final = np.array([final_w, 1.0 - final_w], dtype=np.float64)
        dd_replay = verify_drawdown_montecarlo(
            scen, b_final, lam, alpha=alpha, seed=seed,
            n_paths=replay_paths, horizon=replay_horizon,
        )

    return SizedPosition(
        final_weight=float(ceiling.f_ceiling),
        rck_weight=rck_weight,
        binding_layer=binding_layer,
        alpha=float(alpha),
        beta=float(beta),
        lam=float(lam),
        rck_solution=rck_sol,
        ceiling_result=ceiling,
        drawdown_replay=dd_replay,
        scenario_meta=dict(scen.meta),
    )
