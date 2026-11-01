"""R3 sizing — typed error hierarchy (shared by evt / rck / sizing_ceiling).

All failures in the R3 drawdown-constrained sizing pipeline are typed so callers
(and the trial registry) can distinguish a bad contract from a missing solver from
a domain violation. Fail closed — never return a silently-wrong size.
"""
from __future__ import annotations


class SizingError(Exception):
    """Base for every R3 sizing failure."""


class ScenarioSetError(SizingError, ValueError):
    """The ScenarioSet contract is malformed (shape / probabilities / non-finite)."""


class EVTError(SizingError, ValueError):
    """Conditional-EVT / GPD-POT fit failure (bad threshold, no exceedances, ...)."""


class RCKError(SizingError):
    """Robust-Kelly (RCK) failure: bad drawdown params, infeasible program, ..."""


class DrawdownConstraintError(RCKError):
    """The drawdown constraint could not be satisfied / verified (binds check failed)."""


class SolverUnavailableError(RCKError):
    """cvxpy (the convex solver) is not installed — RCK cannot solve. Fail closed.

    This is deliberately NOT an ImportError: importing the sizing package must
    never require cvxpy; only *solving* does, and it must fail with a typed,
    actionable error rather than a raw import crash.
    """


# NOTE: the Sinclair ceiling layer (sizing_ceiling.py) carries its OWN typed
# error (`SizingCeilingError`, a ValueError) — it was built by a sibling
# component and is preserved as-is. The two hierarchies are intentionally NOT
# merged here (that would require editing the sibling's module); see the
# coordination note in this package's __init__.

__all__ = [
    "SizingError",
    "ScenarioSetError",
    "EVTError",
    "RCKError",
    "DrawdownConstraintError",
    "SolverUnavailableError",
]
