"""risk_kelly.sizing — R3 drawdown-constrained position sizing (research).

IMPORT FIREWALL (hard):
  The deterministic money path (`heimdall.core`) MUST NOT import anything under
  `risk_kelly.sizing`. `grep 'import.*research' heimdall/core` stays empty
  — CI-enforced by import-linter's `core-independence` contract AND by
  tests/research/test_sizing_firewall.py (mirrors the FF-1 core-perp-research
  check). This package SIZES a bet on a scenario set; it does NOT certify an edge
  and emits NO live order. The dependency is one-directional:

      research.sizing ─may import→ numpy / scipy / (cvxpy, lazily in rck)
      heimdall.core   ─imports→    NOTHING under research                (verified)

ONE-DIRECTIONAL PIPELINE:

      real fills ─▶ evt.fit_conditional_evt   (conditional-EVT / GPD-POT tail)
                 ─▶ evt.build_scenario_set     ─▶  ScenarioSet r
                 ─▶ rck.solve_rck              (convex sizer, drawdown cap E[g^{-λ}]≤1)
                 ─▶ sizing_ceiling.apply_ceiling (Sinclair clamp + mean haircut)

  `ScenarioSet` is the SHARED CONTRACT between the tail producer (evt) and the bet
  sizer (rck): defined once in `evt.py`, imported by `rck.py`. `evt` sizes the
  TAIL (EVT); `rck` sizes the BET (RCK) on the resulting scenario set.

PARAMETER-FREE / UNSIGNED status (rail b):
  Every risk knob — the drawdown floor α, breach prob β, the RCK exponent λ, the
  GPD shape ξ, the Sinclair fraction, the mean haircut, the tail threshold — is an
  explicit caller argument feeding the trial registry, NEVER a pinned §5 constant.
  T8 signs them. Nothing here retires the signed Kelly (PA-043) on its own: this
  is the research infrastructure that WILL, once §5/T8 sign the numbers.

CVXPY IS OPTIONAL AT IMPORT (rail c/d):
  Importing this package (and `evt` + `sizing_ceiling`) requires only numpy/scipy.
  `rck` DEFERS the cvxpy import into `solve_rck`, which fails closed with
  `SolverUnavailableError` when the solver is absent — so the firewall/evt/ceiling
  tests run without the dependency.

COORDINATION NOTE:
  `sizing_ceiling.py` (the Sinclair ceiling, built by a sibling component) carries
  its OWN typed error `SizingCeilingError` (a ValueError); `evt`/`rck` use the
  `errors` module's `SizingError` hierarchy. The two are intentionally not merged
  (that would require editing the sibling module). Both are re-exported below.

OPEN GOVERNANCE (rail i): Sleeve C's sacred clock-driven roll may warrant its own
λ / ceiling exemption — an OPEN §5/T8 question, not decided in this package.
"""
from risk_kelly.sizing import (  # noqa: F401  (submodule re-export)
    errors,
    evt,
    pipeline,
    rck,
    sizing_ceiling,
)

# --- evt: conditional-EVT tail + the ScenarioSet contract ------------------ #
from risk_kelly.sizing.evt import (  # noqa: F401
    GPDFit,
    ScenarioSet,
    build_evt_scenario_set,
    build_scenario_set,
    es_cvar,
    fit_conditional_evt,
)

# --- pipeline: the end-to-end EVT → RCK → Sinclair-ceiling wiring ----------- #
from risk_kelly.sizing.pipeline import (  # noqa: F401
    SizedPosition,
    size_position,
)

# --- rck: drawdown-constrained Kelly convex sizer -------------------------- #
from risk_kelly.sizing.rck import (  # noqa: F401
    RCKSolution,
    KKTReport,
    DrawdownReplay,
    lambda_from_drawdown,
    solve_rck,
    solve_qrck,
    verify_kkt_montecarlo,
    verify_drawdown_montecarlo,
)

# --- sizing_ceiling: Sinclair fraction-of-Kelly clamp (sibling component) --- #
from risk_kelly.sizing.sizing_ceiling import (  # noqa: F401
    CeilingInputs,
    CeilingResult,
    SizingCeilingError,
    apply_ceiling,
)

# --- errors ---------------------------------------------------------------- #
from risk_kelly.sizing.errors import (  # noqa: F401
    DrawdownConstraintError,
    EVTError,
    RCKError,
    ScenarioSetError,
    SizingError,
    SolverUnavailableError,
)

__all__ = [
    # submodules
    "evt",
    "rck",
    "pipeline",
    "sizing_ceiling",
    "errors",
    # contract + evt
    "ScenarioSet",
    "GPDFit",
    "build_scenario_set",
    "build_evt_scenario_set",
    "fit_conditional_evt",
    "es_cvar",
    # pipeline
    "size_position",
    "SizedPosition",
    # rck
    "lambda_from_drawdown",
    "solve_rck",
    "solve_qrck",
    "verify_kkt_montecarlo",
    "verify_drawdown_montecarlo",
    "RCKSolution",
    "KKTReport",
    "DrawdownReplay",
    # sizing_ceiling
    "apply_ceiling",
    "CeilingInputs",
    "CeilingResult",
    "SizingCeilingError",
    # errors
    "SizingError",
    "ScenarioSetError",
    "EVTError",
    "RCKError",
    "DrawdownConstraintError",
    "SolverUnavailableError",
]
