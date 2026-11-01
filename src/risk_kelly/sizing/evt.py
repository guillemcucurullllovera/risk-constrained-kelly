"""R3 sizing — conditional-EVT tail model + the ScenarioSet CONTRACT.

Provenance: clean-room reimplementation of the peaks-over-threshold (POT) tail
apparatus from McNeil, Frey & Embrechts, *Quantitative Risk Management* (Ch. 5,
7) — the GPD limit for excesses, the POT tail/VaR/ES closed forms, and the
coherent Expected-Shortfall estimator. Equations only; the GPD MLE delegates to
`scipy.stats.genpareto` (BSD). No black-box copy.

------------------------------------------------------------------------------
ROLE IN THE R3 PIPELINE (one-directional):

    real fills ─▶ fit_conditional_evt (GPD-POT left tail)
              ─▶ build_scenario_set  ─▶  ScenarioSet r
              ─▶ rck.solve_rck (convex sizer, drawdown cap)
              ─▶ sizing_ceiling.apply_ceiling (Sinclair clamp + haircut)

`ScenarioSet` is defined HERE, once, and imported by `rck` — it is the shared
contract between the tail producer (evt) and the bet sizer (rck). `evt` sizes the
TAIL; `rck` sizes the BET on the resulting scenario set.

------------------------------------------------------------------------------
STATUS / RAILS:
  * UNSIGNED — every knob (threshold / tail_fraction / xi) is an explicit caller
    arg feeding the trial registry; NO §5 constant is pinned here (T8 signs them).
  * ADVISORY — this sizes a bet on a scenario set; it does NOT certify an edge
    (CPCV + purged folds + deflated Sharpe stay upstream) and emits NO live order.
  * CONDITIONAL EVT — `fit_conditional_evt` accepts an optional `filter` hook to
    pre-whiten the series (ARMA-GARCH residuals, per McNeil-Frey conditional EVT).
    Absent a filter it fits the RAW series and the IID assumption is the caller's
    to defend — validation rail (g) stress-tests it with a regime-block bootstrap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy import stats

from risk_kelly.sizing.errors import EVTError, ScenarioSetError

__all__ = [
    "ScenarioSet",
    "GPDFit",
    "es_cvar",
    "fit_conditional_evt",
    "build_scenario_set",
    "build_evt_scenario_set",
]


# --------------------------------------------------------------------------- #
# ScenarioSet — the shared evt→rck contract                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScenarioSet:
    """A finite, weighted set of outcome scenarios — the input to the RCK sizer.

    `returns[i, j]` is the GROSS wealth multiplier (e.g. 1.03 for +3%) of asset j
    in scenario i; `probabilities[i]` is the (physical / haircut) probability of
    scenario i. The RCK program maximizes  E[ log(returns · b) ]  over allocations
    b >= 0, 1ᵀb = 1, subject to the drawdown moment  E[(returns·b)^{-λ}] ≤ 1.

    Frozen and array-immutable — deterministic, safely shareable between the
    producer (evt) and consumer (rck). Single-asset bets pass a 1-D return vector
    (see `build_scenario_set`), stored internally as an (S, 1) column.
    """

    returns: np.ndarray
    probabilities: np.ndarray
    asset_names: Optional[tuple[str, ...]] = None
    seed: Optional[int] = None
    provenance: str = ""
    tail_fit: Optional["GPDFit"] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        r = np.asarray(self.returns, dtype=np.float64)
        if r.ndim == 1:
            r = r.reshape(-1, 1)
        if r.ndim != 2 or r.shape[0] == 0 or r.shape[1] == 0:
            raise ScenarioSetError(
                f"returns must be a non-empty 2-D (S, A) array; got shape {r.shape}"
            )
        if not np.all(np.isfinite(r)):
            raise ScenarioSetError("returns contains non-finite values")
        if np.any(r <= 0.0):
            raise ScenarioSetError(
                "returns must be strictly positive GROSS multipliers (log-wealth needs r>0)"
            )

        p = np.asarray(self.probabilities, dtype=np.float64)
        if p.ndim != 1 or p.shape[0] != r.shape[0]:
            raise ScenarioSetError(
                f"probabilities must be 1-D of length S={r.shape[0]}; got shape {p.shape}"
            )
        if not np.all(np.isfinite(p)):
            raise ScenarioSetError("probabilities contains non-finite values")
        if np.any(p < 0.0):
            raise ScenarioSetError("probabilities must be non-negative")
        if not np.isclose(p.sum(), 1.0, rtol=0, atol=1e-9):
            raise ScenarioSetError(f"probabilities must sum to 1; got {p.sum()!r}")

        if self.asset_names is not None and len(self.asset_names) != r.shape[1]:
            raise ScenarioSetError(
                f"asset_names length {len(self.asset_names)} != n_assets {r.shape[1]}"
            )

        r = np.ascontiguousarray(r)
        p = np.ascontiguousarray(p)
        r.flags.writeable = False
        p.flags.writeable = False
        object.__setattr__(self, "returns", r)
        object.__setattr__(self, "probabilities", p)

    # -- shape ------------------------------------------------------------- #
    @property
    def n_scenarios(self) -> int:
        return int(self.returns.shape[0])

    @property
    def n_assets(self) -> int:
        return int(self.returns.shape[1])

    # -- growth functionals (consumed by rck) ------------------------------ #
    def growth(self, b: np.ndarray) -> np.ndarray:
        """Per-scenario portfolio gross return  g_i = returns[i] · b."""
        b = np.asarray(b, dtype=np.float64)
        if b.shape != (self.n_assets,):
            raise ScenarioSetError(
                f"allocation b must have shape ({self.n_assets},); got {b.shape}"
            )
        g = self.returns @ b
        if np.any(g <= 0.0):
            raise ScenarioSetError(
                "allocation produces a non-positive gross return in some scenario "
                "(log-wealth undefined) — infeasible for RCK"
            )
        return g

    def expected_log_growth(self, b: np.ndarray) -> float:
        """E[ log(returns · b) ] — the RCK objective."""
        return float(np.sum(self.probabilities * np.log(self.growth(b))))

    def drawdown_moment(self, b: np.ndarray, lam: float) -> float:
        """E[ (returns · b)^{-λ} ] — the RCK drawdown constraint LHS (≤ 1 binds)."""
        if lam <= 0.0:
            raise ScenarioSetError("lambda must be > 0 for the drawdown moment")
        return float(np.sum(self.probabilities * self.growth(b) ** (-float(lam))))


# --------------------------------------------------------------------------- #
# GPDFit — a fitted generalized-Pareto tail (McNeil POT)                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GPDFit:
    """Generalized-Pareto (POT) tail fit of the excesses over a threshold `u`.

    Parameterization matches McNeil-Frey-Embrechts:  excess Y = X - u | X > u
    has df  G(y) = 1 - (1 + xi·y/beta)^{-1/xi}  (xi != 0),  beta > 0.
    `n_exceedances / n_total` estimates the tail probability P(X > u).
    """

    xi: float          # shape ξ  (tail index; ξ>0 ↔ fat / power tail)
    beta: float        # scale β > 0
    threshold: float   # u
    n_exceedances: int  # N_u
    n_total: int        # n

    def __post_init__(self) -> None:
        if not np.isfinite(self.xi):
            raise EVTError(f"xi must be finite; got {self.xi!r}")
        if not (np.isfinite(self.beta) and self.beta > 0.0):
            raise EVTError(f"beta (scale) must be finite and > 0; got {self.beta!r}")
        if self.n_exceedances <= 0 or self.n_total <= 0:
            raise EVTError("n_exceedances and n_total must be positive")
        if self.n_exceedances > self.n_total:
            raise EVTError("n_exceedances cannot exceed n_total")

    @property
    def tail_prob(self) -> float:
        """P(X > u) ≈ N_u / n."""
        return self.n_exceedances / self.n_total

    def var(self, p: float) -> float:
        """POT VaR at level p (McNeil 7.18):
        VaR_p = u + (β/ξ)[ ((n/N_u)(1-p))^{-ξ} - 1 ].
        Requires p in the fitted tail (p > 1 - N_u/n).
        """
        if not (0.0 < p < 1.0):
            raise EVTError(f"p must be in (0,1); got {p!r}")
        if p <= 1.0 - self.tail_prob:
            raise EVTError(
                f"p={p} is below the fitted tail (need p > {1.0 - self.tail_prob}); "
                "quantile not covered by the POT fit"
            )
        ratio = (self.n_total / self.n_exceedances) * (1.0 - p)
        if abs(self.xi) < 1e-12:  # exponential limit
            return self.threshold - self.beta * np.log(ratio)
        return self.threshold + (self.beta / self.xi) * (ratio ** (-self.xi) - 1.0)

    def es(self, p: float) -> float:
        """POT Expected Shortfall at level p (McNeil 7.19):
        ES_p = VaR_p/(1-ξ) + (β - ξ·u)/(1-ξ),  finite only for ξ < 1.
        """
        if self.xi >= 1.0:
            raise EVTError(
                f"ES undefined: xi={self.xi} >= 1 implies an infinite-mean tail"
            )
        var_p = self.var(p)
        return var_p / (1.0 - self.xi) + (self.beta - self.xi * self.threshold) / (
            1.0 - self.xi
        )


# --------------------------------------------------------------------------- #
# es_cvar — empirical coherent Expected Shortfall / CVaR                        #
# --------------------------------------------------------------------------- #
def es_cvar(losses: np.ndarray, alpha: float) -> tuple[float, float]:
    """Empirical VaR and Expected Shortfall (CVaR) of a LOSS sample at level α.

    Returns (VaR_α, ES_α) where ES_α = E[ L | L >= VaR_α ] — the average of the
    worst (1-α) losses. ES is coherent (sub-additive); VaR is not. Larger loss =
    worse, so pass losses (for a short-vol book, losses = -pnl).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1); got {alpha!r}")
    x = np.asarray(losses, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("losses is empty")
    if not np.all(np.isfinite(x)):
        raise ValueError("losses contains non-finite values")
    var = float(np.quantile(x, alpha, method="linear"))
    tail = x[x >= var]
    if tail.size == 0:  # degenerate (all mass at/below var)
        es = var
    else:
        es = float(tail.mean())
    return var, es


# --------------------------------------------------------------------------- #
# fit_conditional_evt — GPD-POT fit of the (optionally filtered) tail          #
# --------------------------------------------------------------------------- #
def fit_conditional_evt(
    data: np.ndarray,
    *,
    tail: str = "lower",
    threshold: Optional[float] = None,
    tail_fraction: float = 0.10,
    filter: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> GPDFit:
    """Fit a GPD to the peaks-over-threshold of the LOSS side of `data`.

    tail="lower": model the left tail (losses = -data) — the short-vol drawdown
    side. tail="upper": model the right tail (losses = data). If `filter` is
    given it pre-whitens `data` (e.g. returns ARMA-GARCH standardized residuals)
    → McNeil-Frey CONDITIONAL EVT; absent, the RAW series is fitted (IID
    assumption is the caller's to defend — see validation rail (g)).

    `threshold`/`tail_fraction` and the resulting ξ are UNSIGNED research knobs.
    """
    if tail not in ("lower", "upper"):
        raise EVTError(f"tail must be 'lower' or 'upper'; got {tail!r}")
    if not (0.0 < tail_fraction < 1.0):
        raise EVTError(f"tail_fraction must be in (0,1); got {tail_fraction!r}")

    x = np.asarray(data, dtype=np.float64).ravel()
    if filter is not None:
        x = np.asarray(filter(x), dtype=np.float64).ravel()
    if x.size < 20:
        raise EVTError(f"need >= 20 observations to fit a GPD tail; got {x.size}")
    if not np.all(np.isfinite(x)):
        raise EVTError("data contains non-finite values")

    losses = -x if tail == "lower" else x
    n_total = losses.size

    if threshold is None:
        threshold = float(np.quantile(losses, 1.0 - tail_fraction))
    exceed = losses[losses > threshold] - threshold
    if exceed.size < 10:
        raise EVTError(
            f"too few exceedances ({exceed.size}) over threshold {threshold}; "
            "lower the threshold or supply more data"
        )

    # GPD MLE with location pinned to 0 (excesses start at 0). scipy: c=ξ, scale=β.
    xi, _loc, beta = stats.genpareto.fit(exceed, floc=0.0)
    if not (np.isfinite(xi) and np.isfinite(beta) and beta > 0.0):
        raise EVTError(f"GPD MLE returned invalid params xi={xi}, beta={beta}")

    return GPDFit(
        xi=float(xi),
        beta=float(beta),
        threshold=float(threshold),
        n_exceedances=int(exceed.size),
        n_total=int(n_total),
    )


# --------------------------------------------------------------------------- #
# build_scenario_set — assemble the evt→rck contract                          #
# --------------------------------------------------------------------------- #
def build_scenario_set(
    returns: np.ndarray,
    *,
    probabilities: Optional[np.ndarray] = None,
    asset_names: Optional[tuple[str, ...]] = None,
    seed: Optional[int] = None,
    provenance: str = "",
    tail_fit: Optional[GPDFit] = None,
) -> ScenarioSet:
    """Deterministically assemble a `ScenarioSet` from gross-return scenarios.

    `returns` is 1-D (single bet, shape (S,)) or 2-D (multi-asset, (S, A)).
    `probabilities` default to uniform. `seed` is recorded for provenance only —
    construction here is pure/deterministic (no RNG); a stochastic EVT-splice
    sampler (body ⊕ GPD tail) is a sibling producer that will seed its draw and
    feed its output through this same constructor.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    s = r.shape[0]
    if probabilities is None:
        p = np.full(s, 1.0 / s, dtype=np.float64)
    else:
        p = np.asarray(probabilities, dtype=np.float64).ravel()
    return ScenarioSet(
        returns=r,
        probabilities=p,
        asset_names=asset_names,
        seed=seed,
        provenance=provenance,
        tail_fit=tail_fit,
    )


# --------------------------------------------------------------------------- #
# build_evt_scenario_set — the evt→rck WIRING: fat GPD tail ⊕ empirical body   #
# --------------------------------------------------------------------------- #
def build_evt_scenario_set(
    pnl_returns: np.ndarray,
    *,
    tail_fraction: float = 0.10,
    n_tail: int = 4000,
    seed: int,
    filter: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    gross_floor: float = 1e-3,
    include_cash: bool = True,
    cash_name: str = "cash",
    risky_name: str = "risky",
    provenance: str = "",
) -> ScenarioSet:
    """Splice a conditional-EVT (GPD-POT) LEFT tail onto the empirical body → a ScenarioSet.

    This is the concrete evt→rck WIRING referenced by the pipeline docstring: the RCK
    sizer CONSUMES the fat-tailed distribution the EVT layer produces (``evt`` sizes the
    TAIL, ``rck`` sizes the BET). ``pnl_returns`` are per-period NET returns on risk
    capital (fractions; a +4% period is 0.04, a −25% period is −0.25). The construction is
    the McNeil semiparametric mixture:

      * fit a GPD to the peaks-over-threshold of the LOSS side (``fit_conditional_evt``,
        ``tail='lower'``) — optionally on ``filter``-whitened residuals (conditional EVT);
      * BODY = the empirical observations at/inside the threshold (kept verbatim), carrying
        total probability ``1 - tail_prob`` (tail_prob = N_u/n, the empirical exceedance rate);
      * TAIL = ``n_tail`` GPD-sampled excesses (SEEDED, reproducible) mapped back to losses
        ``u + excess`` → net returns ``−(u+excess)``, carrying total probability ``tail_prob``.

    Returns a two-asset ScenarioSet ``[risky, cash]`` (cash = certain gross 1.0, precludes
    ruin) of GROSS multipliers ``g = 1 + r``, ready for ``rck.solve_rck``. GPD tail draws
    are clipped to ``gross_floor`` so every risky gross stays strictly positive (log-wealth
    needs g>0); a genuinely defined-risk book should pass returns already bounded above −1.

    UNSIGNED: ``tail_fraction``/``n_tail``/the resulting ξ are explicit research knobs
    (T8 signs). DETERMINISTIC given ``seed`` and the input series.
    """
    if not (0.0 < tail_fraction < 1.0):
        raise EVTError(f"tail_fraction must be in (0,1); got {tail_fraction!r}")
    if n_tail < 1:
        raise EVTError(f"n_tail must be >= 1; got {n_tail!r}")
    if not (0.0 < gross_floor < 1.0):
        raise EVTError(f"gross_floor must be in (0,1); got {gross_floor!r}")

    x = np.asarray(pnl_returns, dtype=np.float64).ravel()
    if filter is not None:
        x = np.asarray(filter(x), dtype=np.float64).ravel()
    if x.size < 20:
        raise EVTError(f"need >= 20 observations to splice an EVT tail; got {x.size}")
    if not np.all(np.isfinite(x)):
        raise EVTError("pnl_returns contains non-finite values")

    fit = fit_conditional_evt(x, tail="lower", tail_fraction=tail_fraction)
    u = fit.threshold                       # threshold on LOSSES (= -returns)
    tail_prob = fit.tail_prob               # N_u / n

    losses = -x
    body_returns = x[losses <= u]           # empirical body (verbatim)
    if body_returns.size == 0:
        raise EVTError("empty body after thresholding — lower tail_fraction")

    # SEEDED GPD tail draws → net returns on the loss side, clipped so gross stays > 0.
    rng = np.random.default_rng(seed)
    excess = stats.genpareto.rvs(
        fit.xi, loc=0.0, scale=fit.beta, size=int(n_tail), random_state=rng
    )
    excess = np.clip(np.asarray(excess, dtype=np.float64), 0.0, None)
    tail_returns = -(u + excess)            # net loss returns (< -u)
    tail_gross = np.clip(1.0 + tail_returns, gross_floor, None)

    body_gross = np.clip(1.0 + body_returns, gross_floor, None)

    risky_gross = np.concatenate([body_gross, tail_gross])
    # mixture weights: body carries (1-tail_prob), tail carries tail_prob, uniform within each.
    w_body = np.full(body_gross.size, (1.0 - tail_prob) / body_gross.size)
    w_tail = np.full(tail_gross.size, tail_prob / tail_gross.size)
    probs = np.concatenate([w_body, w_tail])
    probs = probs / probs.sum()             # renormalize away float drift

    if include_cash:
        returns = np.column_stack([risky_gross, np.ones_like(risky_gross)])
        asset_names: Optional[tuple[str, ...]] = (risky_name, cash_name)
    else:
        returns = risky_gross.reshape(-1, 1)
        asset_names = (risky_name,)

    return ScenarioSet(
        returns=returns,
        probabilities=probs,
        asset_names=asset_names,
        seed=seed,
        provenance=provenance or "build_evt_scenario_set(GPD-POT left-tail ⊕ empirical body)",
        tail_fit=fit,
        meta={
            "tail_prob": float(tail_prob),
            "threshold_loss": float(u),
            "n_body": int(body_gross.size),
            "n_tail": int(tail_gross.size),
            "tail_fraction": float(tail_fraction),
        },
    )
