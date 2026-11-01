"""Pre-registered composite Kelly-fraction optimizer (PA-030, methodology v1.0).

Replaces the ad-hoc half-Kelly heuristic with ONE pre-registered composite rule:

    f_rec = floor_5dp( min( f_pipe(E_full), f_pipe(E_oos), F_CAP ) )
    f_pipe(E) = min( c*(E) · f̂*(E),  f_DD(E) )

where f̂* is the plug-in discrete full Kelly (bisection on the empirical
growth-curve first-order condition), c* is a MEASURED estimation-error
shrinkage multiplier (Section: shrinkage), and f_DD is a drawdown-guard
fraction computed on a one-sided 95% Wilson upper bound over stationary-
bootstrap paths (Section: guard). Both estimation sets must be signable;
every failure resolves to NOT_SIGNABLE (fraction null) — NEVER to the
incumbent and never to a second-best variant. n_trials = 1 for the sizing
layer (single pre-registered rule; comparators are non-selecting diagnostics).

Why the obvious alternatives fail (panel-measured, 2026-07-02 — keep this):
  (a) bootstrap-AVERAGED growth argmax equals the plug-in argmax identically —
      resample means of per-observation terms provide no shrinkage;
  (b) pointwise-QUANTILE growth selection is a t-test in disguise: it
      degenerates to the grid floor whenever P_boot(mean_b ≤ 0) > α (≈ 0.35
      on this data), i.e. it answers "is the edge significant", not "how much";
  (c) posterior-mean Bayesian shrinkage is linear in the uncertain mean and
      shrinks nothing without an arbitrary skeptical prior.
c* instead evaluates the RULE "bet c times the fitted Kelly" under resampled
estimation randomness and keeps the c that maximizes bootstrap-expected
realized growth on the empirical curve. Known limitation (documented upgrade
path, not built): c* double-uses the sample as truth and resampling source,
which is second-order optimistic; a nested bootstrap would correct it.

Determinism (NF-08): every RNG is random.Random(seed) with
seed = int.from_bytes(sha256("|".join([data_snapshot_sha256, MODULE_VERSION,
purpose, *params]).encode()).digest()[:8], "big"); one generator per purpose,
consumed in a single fixed loop order; every seed is echoed in the result.
No global RNG, no wall-clock, no dict/set iteration in any accumulation.

Research layer: floats internally (disclosed), Decimal only at the output
boundary. Pure functions — no I/O, no network, no clock.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_FLOOR
from statistics import NormalDist
from typing import Optional, Sequence

from risk_kelly.calibration.kelly_calibration import KellyCalibrationError

MODULE_VERSION = "kelly-opt-1.0"
METHODOLOGY_VERSION = "panel-converged-2026-07-02"

_ANNUALIZATION = 365.25 / 28.0          # trades per year at 28-day cadence
_WILSON_Z = NormalDist().inv_cdf(0.95)  # one-sided 95%
_RUIN_EPS = 1e-12
_INCUMBENT = 0.08923                    # currently signed value — comparison only


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KellyParams:
    """All methodology constants. [OPEN] fields need operator sign-off with the
    fraction; everything else is a FIXED methodological constant — changing any
    fixed constant is a new trial (see the trial-accounting block).
    """
    # --- [OPEN] governance inputs (Francisco signs these WITH the fraction) ---
    d_kill: float = 0.20            # [OPEN] program-kill max-drawdown budget
    d_kill_confirmed: bool = False  # [OPEN] True only when d_kill equals the committed §5 kill threshold
    p_kill: float = 0.05            # [OPEN] tolerated P(MDD ≥ d_kill) over the horizon
    d_soft: float = 0.10            # [OPEN] mandate-erosion drawdown
    p_soft: float = 0.25            # [OPEN] tolerated P(MDD ≥ d_soft)
    horizon: int = 65               # [OPEN] guard horizon in trades (~5y at 28d)
    stress_mult: float = 1.25       # [OPEN] crisis-slippage multiplier on losses (S1)
    f_cap: float = 0.25             # [OPEN] signing-convention cap (confirm)
    # --- fixed methodological constants ---
    f_grid_step: float = 0.005      # guard grid 0.005..0.250
    c_grid_step: float = 0.01       # shrinkage grid 0.01..1.00
    block_len: int = 6              # stationary-bootstrap mean block length (trades)
    b_shrink: int = 2000
    b_dd: int = 10000
    b_shrink_wf: int = 500
    b_dd_wf: int = 1000
    horizon_report: int = 130       # report-only second horizon
    min_obs: int = 60
    min_losses: int = 5
    oos_fraction: float = 0.30      # last-30% convention (matches oos_returns())
    warmup_trades: int = 130        # walk-forward warm-up (~10y)
    refit_cadence: int = 13         # walk-forward refit every ~1y of trades
    stability_tol: float = 0.005    # one guard grid step
    d_diag_grid: tuple = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


DEFAULTS = KellyParams()


# ---------------------------------------------------------------------------
# Result containers (frozen; mirror the report)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GuardRow:
    """Diagnostic row: exceedance stats for one (cell, f) across the D grid."""
    cell: str
    f: float
    stats: tuple  # tuple of (D, k, p_hat, wilson_ub) in d_diag_grid order


@dataclass(frozen=True)
class SetResult:
    name: str
    n: int
    n_losses: int
    signable: bool
    reasons: tuple                  # typed NOT_SIGNABLE reason codes for this set
    f_star: float
    kelly_capped: bool
    c_star: float
    f_shrunk: float                 # c* · f̂*
    f_dd: Optional[float]           # None = undefined (grid floor breaches)
    f_pipe: Optional[float]
    binding_term: str               # "shrunk_kelly" | "drawdown_guard:<target>:<cell>" | ""
    r_curve: tuple                  # ((c, R(c)), ...) — the shrinkage evidence
    f_star_b_deciles: tuple         # 11 quantiles of the bootstrap full-Kelly dist
    frac_f_star_b_zero: float
    frac_f_star_b_capped: float
    p_boot_mean_nonpos: float
    quantile_growth_curve: tuple    # DIAGNOSTIC_ONLY ((f, G_q10(f)), ...)
    guard_matrix: tuple             # tuple[GuardRow] — all cells × F × D grid
    incumbent_guard_line: tuple     # ((cell, p_hat, wilson_ub) at (D_kill, H) for f=incumbent)
    mdd_at_frec: tuple              # ((cell, p50, p90, p99, p_exceed_dkill_ub), ...) at f_rec (filled post-composition)
    terminal_wealth: tuple          # ((horizon, P5, P25, P50, P75, P95), ...) at f_rec (filled post-composition)
    time_under_water: tuple         # (median_frac, p95_frac) at f_rec, binding cells only (filled post-composition)
    warnings: tuple
    shrinkage_l_variants: tuple = ()  # ((L, c_star, f_shrunk), ...) report-only §7 sensitivity


@dataclass(frozen=True)
class KellyOptimizationResult:
    status: str                     # CANDIDATE | NOT_SIGNABLE | MC_UNSTABLE
    reason_codes: tuple
    f_rec: Optional[Decimal]        # 5-dp Decimal or None
    f_raw: Optional[float]
    binding_constraint: str
    sets: tuple                     # (SetResult E_full, SetResult E_oos)
    stability_f_rec: Optional[Decimal]
    seeds: tuple                    # ((purpose, seed), ...) every RNG used
    params: KellyParams
    data_snapshot_sha256: str
    incumbent: float = _INCUMBENT
    warnings: tuple = ()


@dataclass(frozen=True)
class WalkforwardRefit:
    k: int
    entry_date: str
    n_train: int
    f_applied: float
    fell_back_to_full: bool         # STRUCTURAL oos-slice fallback only (§6)
    sit_out_reason: str = ""        # non-empty → window traded at f=0 (fail-closed replay)
    f_incumbent: float = _INCUMBENT # per-rule per-refit f table (§6 report requirement)
    f_half_kelly: float = 0.0
    f_full_kelly: float = 0.0


@dataclass(frozen=True)
class WalkforwardRuleMetrics:
    rule: str
    total_log_growth: float
    annualized_growth: float
    realized_mdd: float
    time_under_water_frac: float
    per_decade_growth: tuple        # ((label, mean_log_growth), ...)
    ruined: bool


@dataclass(frozen=True)
class WalkforwardResult:
    refits: tuple                   # tuple[WalkforwardRefit]
    rules: tuple                    # tuple[WalkforwardRuleMetrics] — optimized first
    gate_no_ruin: bool
    gate_mdd_below_dkill: bool
    gate_mdd_not_worse_than_incumbent: bool
    passed: bool
    fallback_count: int             # structural oos-slice fallbacks ONLY (§6)
    refit_instability_warnings: tuple
    seeds: tuple
    sit_out_count: int = 0          # windows sat out: non-structural NOT_SIGNABLE replay
    below_floor_count: int = 0      # windows sat out: composite < guard grid floor (§5 step 4)


# ---------------------------------------------------------------------------
# Seed derivation (NF-08)
# ---------------------------------------------------------------------------

def derive_seed(data_snapshot_sha256: str, purpose: str, *params) -> int:
    """seed = first 8 bytes (big-endian) of sha256(snapshot|module|purpose|params)."""
    material = "|".join(
        [data_snapshot_sha256, MODULE_VERSION, purpose] + [str(p) for p in params]
    )
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


# ---------------------------------------------------------------------------
# Growth curve + plug-in full Kelly
# ---------------------------------------------------------------------------

def empirical_growth(returns: Sequence[float], f: float) -> float:
    """G(f) = mean log1p(f·r); −inf if any 1 + f·r ≤ 1e−12 (ruin on the sample)."""
    total = 0.0
    log1p = math.log1p
    for r in returns:
        x = f * r
        if 1.0 + x <= _RUIN_EPS:
            return float("-inf")
        total += log1p(x)
    return total / len(returns)


def full_kelly(returns: Sequence[float]) -> tuple[float, bool]:
    """Plug-in discrete full Kelly on (0, f_hi], f_hi = min(0.5, 0.999999/|min r|).

    Root of h(f) = Σ r_i/(1+f·r_i) = 0 (h strictly decreasing ⇒ unique).
    Returns (f̂*, capped): f̂* = 0 if mean ≤ 0; f̂* = f_hi flagged capped=True
    if h(f_hi) > 0 (root beyond domain). Bisection, 100 iterations.
    """
    n = len(returns)
    if n == 0:
        raise KellyCalibrationError("full_kelly: empty input")
    if math.fsum(returns) / n <= 0.0:
        return 0.0, False
    r_min = min(returns)
    f_hi = min(0.5, 0.999999 / abs(r_min)) if r_min < 0.0 else 0.5

    def h(f: float) -> float:
        return math.fsum(r / (1.0 + f * r) for r in returns)

    if h(f_hi) > 0.0:
        return f_hi, True
    lo, hi = 0.0, f_hi
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if h(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0, False


# ---------------------------------------------------------------------------
# Stationary bootstrap (Politis–Romano, circular)
# ---------------------------------------------------------------------------

def stationary_bootstrap_indices(rng, n: int, length: int, mean_block_len: float) -> list:
    """Circular stationary bootstrap index path of the given length over [0, n)."""
    p_new = 1.0 / mean_block_len
    u = rng.randrange(n)
    out = []
    append = out.append
    random = rng.random
    randrange = rng.randrange
    for _ in range(length):
        append(u)
        if random() < p_new:
            u = randrange(n)
        else:
            u = (u + 1) % n
    return out


# ---------------------------------------------------------------------------
# Shrinkage: c* = argmax_c bootstrap-expected realized growth of "bet c·f̂*"
# ---------------------------------------------------------------------------

def _shrinkage(returns: list, rng, b_shrink: int, block_len: int, c_grid: list):
    """Returns (c_star, r_max, r_curve, f_star_b list, mean_nonpos_count).
    Fail-closed tie-break: smaller c. Spec §3(a) rule ORDER: mean ≤ 0 → f*_b = 0
    is checked BEFORE the <2-loss park, so a degenerate resample that satisfies
    both resolves to 0 (fail-closed), never to the 0.5 cap."""
    n = len(returns)
    f_star_bs = []
    mean_nonpos = 0
    for _ in range(b_shrink):
        idx = stationary_bootstrap_indices(rng, n, n, block_len)
        sample = [returns[i] for i in idx]
        mu = math.fsum(sample) / n
        if mu <= 0.0:
            f_star_bs.append(0.0)
            mean_nonpos += 1
            continue
        n_loss = sum(1 for r in sample if r < 0.0)
        if n_loss < 2:
            f_star_bs.append(0.5)   # parked at the domain cap; R(c) penalizes it
            continue
        f_b, _ = full_kelly(sample)
        f_star_bs.append(f_b)

    growth_cache: dict = {0.0: 0.0}
    def g(f: float) -> float:
        v = growth_cache.get(f)
        if v is None:
            v = empirical_growth(returns, f)
            growth_cache[f] = v
        return v

    best_c, best_r = c_grid[0], float("-inf")
    r_curve = []
    for c in c_grid:
        total = 0.0
        for f_b in f_star_bs:
            total += g(min(c * f_b, 0.5))
        r = total / b_shrink
        r_curve.append((c, r))
        if r > best_r + 1e-15:      # strict improvement only; ties → smaller c
            best_c, best_r = c, r
    return best_c, best_r, tuple(r_curve), f_star_bs, mean_nonpos


# ---------------------------------------------------------------------------
# Wilson upper bound
# ---------------------------------------------------------------------------

def wilson_upper(k: int, n: int, z: float = _WILSON_Z) -> float:
    """One-sided (1−α) Wilson score UPPER bound for a binomial proportion."""
    if n <= 0:
        raise KellyCalibrationError("wilson_upper: n must be positive")
    p_hat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = p_hat + z2 / (2.0 * n)
    spread = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))
    return (centre + spread) / denom


# ---------------------------------------------------------------------------
# Drawdown guard
# ---------------------------------------------------------------------------

def _gen_paths(rng, n: int, b: int, horizon: int, block_len: float) -> list:
    """B index paths of the given horizon, generated once (common random numbers)."""
    return [stationary_bootstrap_indices(rng, n, horizon, block_len) for _ in range(b)]


def _stress_s1(returns: list, mult: float) -> list:
    """Amplify losses by `mult`, saturating at the structural −1 floor; wins unchanged."""
    return [max(-1.0, mult * r) if r < 0.0 else r for r in returns]


def _path_mdds(data: list, paths: list, f: float, s3_pos: Optional[list] = None) -> list:
    """Max drawdown per path at fraction f. s3_pos: per-path index whose trade is
    replaced by r = −1 (the S3 'a max-loss print will eventually happen' cell)."""
    log1p, exp = math.log1p, math.exp
    out = []
    for b, idx in enumerate(paths):
        pos = s3_pos[b] if s3_pos is not None else -1
        logw = 0.0
        peak = 0.0
        max_dd_log = 0.0
        ruined = False
        for t, i in enumerate(idx):
            r = -1.0 if t == pos else data[i]
            x = f * r
            if 1.0 + x <= _RUIN_EPS:
                ruined = True
                break
            logw += log1p(x)
            if logw > peak:
                peak = logw
            dd = peak - logw
            if dd > max_dd_log:
                max_dd_log = dd
        out.append(1.0 if ruined else 1.0 - exp(-max_dd_log))
    return out


def _exceedance(mdds: list, thresholds) -> list:
    """[(D, k, p_hat, wilson_ub), ...] for each threshold D."""
    b = len(mdds)
    rows = []
    for d in thresholds:
        k = sum(1 for m in mdds if m >= d)
        rows.append((d, k, k / b, wilson_upper(k, b)))
    return rows


# ---------------------------------------------------------------------------
# Per-set pipeline
# ---------------------------------------------------------------------------

def _f_grid(params: KellyParams) -> list:
    return [round(params.f_grid_step * k, 10) for k in range(1, 51)]


def _c_grid(params: KellyParams) -> list:
    return [round(params.c_grid_step * k, 10) for k in range(1, 101)]


def _quantile(sorted_vals: list, q: float) -> float:
    """Order-statistic quantile (0-indexed floor) — exact and version-stable."""
    return sorted_vals[min(int(q * len(sorted_vals)), len(sorted_vals) - 1)]


def _run_set(
    name: str,
    returns: list,
    data_sha: str,
    params: KellyParams,
    seed_prefix: str,
    seeds_log: list,
    skip_diagnostics: bool = False,
    b_shrink_override: Optional[int] = None,
    b_dd_override: Optional[int] = None,
) -> SetResult:
    """Run gates → f̂* → c* → guard for one estimation set.

    skip_diagnostics skips report-only guard cells and diagnostics; the
    stability rerun uses it at IDENTICAL B (spec: only the seeds change),
    while walk-forward additionally overrides B down to the wf budgets.
    """
    n = len(returns)
    n_losses = sum(1 for r in returns if r < 0.0)
    reasons = []
    warnings = []

    if n < params.min_obs:
        reasons.append(f"n<{params.min_obs}")
    if n_losses < params.min_losses:
        reasons.append(f"losses<{params.min_losses}")
    if reasons:
        return _unsignable_set(name, n, n_losses, tuple(reasons))

    f_star, capped = full_kelly(returns)
    if capped:
        warnings.append("kelly_capped: h(f_hi) > 0 — root beyond the 0.5 domain cap")
    if f_star <= 0.0:
        return _unsignable_set(name, n, n_losses, ("nonpositive_edge",),
                               f_star=0.0)

    # --- shrinkage ---
    purpose = f"{seed_prefix}shrink"
    seed = derive_seed(data_sha, purpose, name)
    seeds_log.append((f"{purpose}({name})", seed))
    rng = random.Random(seed)
    b_s = b_shrink_override if b_shrink_override is not None else params.b_shrink
    c_star, r_max, r_curve, f_star_bs, mean_nonpos_count = _shrinkage(
        returns, rng, b_s, params.block_len, _c_grid(params)
    )
    if r_max <= 0.0:
        return _unsignable_set(
            name, n, n_losses, ("no_positive_growth_rule_under_uncertainty",),
            f_star=f_star, kelly_capped=capped, c_star=c_star, r_curve=r_curve,
        )
    if c_star >= 1.0:
        warnings.append("c_grid_boundary: c* = 1.00 (no shrinkage indicated); grid not extended")

    f_shrunk = min(c_star * f_star, 0.5)

    # --- bootstrap-f* diagnostics ---
    srt = sorted(f_star_bs)
    deciles = tuple(_quantile(srt, q / 10.0) for q in range(0, 11))
    frac_zero = sum(1 for v in f_star_bs if v == 0.0) / len(f_star_bs)
    frac_cap = sum(1 for v in f_star_bs if v == 0.5) / len(f_star_bs)

    # P_boot(mean_b ≤ 0): counted directly inside _shrinkage (no proxy).
    p_mean_nonpos = mean_nonpos_count / len(f_star_bs)

    # --- drawdown guard ---
    fgrid = _f_grid(params)
    b_dd = b_dd_override if b_dd_override is not None else params.b_dd
    l_base = params.block_len

    purpose = f"{seed_prefix}paths"
    path_seed = derive_seed(data_sha, purpose, name, l_base, params.horizon, b_dd)
    seeds_log.append((f"{purpose}({name},L={l_base},H={params.horizon},B={b_dd})", path_seed))
    paths_l6 = _gen_paths(random.Random(path_seed), n, b_dd, params.horizon, l_base)

    s1 = _stress_s1(returns, params.stress_mult)
    binding_cells = [("L6_S0", returns, paths_l6, None), ("L6_S1", s1, paths_l6, None)]

    # Ascending feasibility scan on the Wilson UPPER bound with isotonic clip.
    upper_needed = f_shrunk  # guard never needs to certify above the shrunk term
    f_dd = None
    binding_term = ""
    run_max_kill = {c[0]: 0.0 for c in binding_cells}
    run_max_soft = {c[0]: 0.0 for c in binding_cells}
    guard_rows = []
    first_violation = None
    for f in fgrid:
        feasible = True
        for cell_name, data, paths, s3 in binding_cells:
            mdds = _path_mdds(data, paths, f, s3)
            exc = _exceedance(mdds, params.d_diag_grid)
            guard_rows.append(GuardRow(cell=cell_name, f=f, stats=tuple(exc)))
            u_kill = wilson_upper(sum(1 for m in mdds if m >= params.d_kill), len(mdds))
            u_soft = wilson_upper(sum(1 for m in mdds if m >= params.d_soft), len(mdds))
            run_max_kill[cell_name] = max(run_max_kill[cell_name], u_kill)   # isotonic clip
            run_max_soft[cell_name] = max(run_max_soft[cell_name], u_soft)
            if run_max_kill[cell_name] > params.p_kill:
                feasible = False
                if first_violation is None:
                    first_violation = (
                        f"drawdown_guard:K:D_kill={params.d_kill},p={params.p_kill},"
                        f"cell={cell_name},set={name}"
                    )
            if run_max_soft[cell_name] > params.p_soft:
                feasible = False
                if first_violation is None:
                    first_violation = (
                        f"drawdown_guard:S:D_soft={params.d_soft},p={params.p_soft},"
                        f"cell={cell_name},set={name}"
                    )
        if feasible:
            f_dd = f
        else:
            break
        if f >= upper_needed:
            break  # early stop: guard already certifies past the shrunk-Kelly term

    if f_dd is None:
        return _unsignable_set(
            name, n, n_losses, ("guard_floor_breach",),
            f_star=f_star, kelly_capped=capped, c_star=c_star, f_shrunk=f_shrunk,
            r_curve=r_curve, deciles=deciles, frac_zero=frac_zero,
            frac_cap=frac_cap, p_mean_nonpos=p_mean_nonpos,
            guard_rows=tuple(guard_rows), warnings=tuple(warnings),
        )

    if f_shrunk <= f_dd:
        f_pipe = f_shrunk
        term = "shrunk_kelly"
    else:
        f_pipe = f_dd
        term = first_violation or f"drawdown_guard:set={name}"

    # --- report-only diagnostics (skipped in fast mode) ---
    quantile_curve = ()
    incumbent_line = ()
    mdd_at = ()                    # filled post-composition at f_rec (spec §4/§7)
    terminal = ()
    tuw = ()
    shrink_l_variants = ()
    if not skip_diagnostics:
        # Incumbent risk line — the operator must see the risk of what is replaced.
        inc_rows = []
        for cell_name, data, paths, s3 in binding_cells:
            mdds = _path_mdds(data, paths, _INCUMBENT, s3)
            k = sum(1 for m in mdds if m >= params.d_kill)
            inc_rows.append((cell_name, k / len(mdds), wilson_upper(k, len(mdds))))
        incumbent_line = tuple(inc_rows)

        # Report-only cells: (L3,S0), (L12,S0), (L12,S1), (L6,S3) — thin over F.
        report_cells = []
        for l_alt in (3, 12):
            purpose = f"{seed_prefix}paths"
            sd = derive_seed(data_sha, purpose, name, l_alt, params.horizon, b_dd)
            seeds_log.append((f"{purpose}({name},L={l_alt},H={params.horizon},B={b_dd})", sd))
            p_alt = _gen_paths(random.Random(sd), n, b_dd, params.horizon, l_alt)
            report_cells.append((f"L{l_alt}_S0", returns, p_alt, None))
            if l_alt == 12:
                report_cells.append((f"L{l_alt}_S1", s1, p_alt, None))
        s3_seed = derive_seed(data_sha, f"{seed_prefix}s3pos", name, params.horizon, b_dd)
        seeds_log.append((f"{seed_prefix}s3pos({name})", s3_seed))
        s3_rng = random.Random(s3_seed)
        s3_pos = [s3_rng.randrange(params.horizon) for _ in range(b_dd)]
        report_cells.append(("L6_S3", returns, paths_l6, s3_pos))
        # Stress-1.5 report cell (§7 sensitivity grid: stress ∈ {1.0, 1.25, 1.5});
        # CRN — reuses paths_l6, only the data vector changes (§3(b)).
        report_cells.append(("L6_S15", _stress_s1(returns, 1.5), paths_l6, None))
        for cell_name, data, paths, s3 in report_cells:
            for f in fgrid:
                mdds = _path_mdds(data, paths, f, s3)
                guard_rows.append(GuardRow(cell=cell_name, f=f,
                                           stats=tuple(_exceedance(mdds, params.d_diag_grid))))
                if f >= upper_needed:
                    break

        # Terminal-wealth H=130 path seed is logged here (the paths themselves
        # are regenerated at f_rec post-composition — see _risk_tables_at).
        sd_h130 = derive_seed(data_sha, f"{seed_prefix}paths", name, l_base,
                              params.horizon_report, b_dd)
        seeds_log.append((f"{seed_prefix}paths({name},L={l_base},"
                          f"H={params.horizon_report},B={b_dd})", sd_h130))

        # Shrinkage-side L variants (§3/§7 report-only sensitivity): rerun c*
        # at L ∈ {3, 12} with derived seeds; B_s unchanged.
        shrink_l = []
        for l_alt in (3, 12):
            sd_l = derive_seed(data_sha, f"{seed_prefix}shrink-L{l_alt}", name)
            seeds_log.append((f"{seed_prefix}shrink-L{l_alt}({name})", sd_l))
            c_alt, _, _, _, _ = _shrinkage(
                returns, random.Random(sd_l), b_s, l_alt, _c_grid(params))
            shrink_l.append((l_alt, c_alt, min(c_alt * f_star, 0.5)))
        shrink_l_variants = tuple(shrink_l)

        # Pointwise-quantile growth curve — DIAGNOSTIC_ONLY edge-fragility
        # evidence (expected ≤ 0 at every f when the edge is weak; NOT a selector).
        purpose = f"{seed_prefix}shrink"
        qrng = random.Random(derive_seed(data_sha, f"{purpose}-qcurve", name))
        seeds_log.append((f"{purpose}-qcurve({name})",
                          derive_seed(data_sha, f"{purpose}-qcurve", name)))
        q_fs = [round(0.01 * k, 10) for k in range(1, 26)]
        resamples = []
        for _ in range(min(b_s, 500)):
            idx = stationary_bootstrap_indices(qrng, n, n, params.block_len)
            resamples.append([returns[i] for i in idx])
        qc = []
        for f in q_fs:
            gs = sorted(empirical_growth(s, f) for s in resamples)
            qc.append((f, _quantile(gs, 0.10)))
        quantile_curve = tuple(qc)

    return SetResult(
        name=name, n=n, n_losses=n_losses, signable=True, reasons=(),
        f_star=f_star, kelly_capped=capped, c_star=c_star, f_shrunk=f_shrunk,
        f_dd=f_dd, f_pipe=f_pipe, binding_term=term, r_curve=r_curve,
        f_star_b_deciles=deciles, frac_f_star_b_zero=frac_zero,
        frac_f_star_b_capped=frac_cap, p_boot_mean_nonpos=p_mean_nonpos,
        quantile_growth_curve=quantile_curve, guard_matrix=tuple(guard_rows),
        incumbent_guard_line=incumbent_line, mdd_at_frec=mdd_at,
        terminal_wealth=terminal, time_under_water=tuw, warnings=tuple(warnings),
        shrinkage_l_variants=shrink_l_variants,
    )


def _unsignable_set(name, n, n_losses, reasons, *, f_star=float("nan"),
                    kelly_capped=False, c_star=float("nan"),
                    f_shrunk=float("nan"), r_curve=(), deciles=(),
                    frac_zero=float("nan"), frac_cap=float("nan"),
                    p_mean_nonpos=float("nan"), guard_rows=(), warnings=()):
    return SetResult(
        name=name, n=n, n_losses=n_losses, signable=False, reasons=reasons,
        f_star=f_star, kelly_capped=kelly_capped, c_star=c_star,
        f_shrunk=f_shrunk, f_dd=None, f_pipe=None, binding_term="",
        r_curve=r_curve, f_star_b_deciles=deciles, frac_f_star_b_zero=frac_zero,
        frac_f_star_b_capped=frac_cap, p_boot_mean_nonpos=p_mean_nonpos,
        quantile_growth_curve=(), guard_matrix=tuple(guard_rows),
        incumbent_guard_line=(), mdd_at_frec=(), terminal_wealth=(),
        time_under_water=(), warnings=tuple(warnings),
    )


def _risk_tables_at(
    name: str,
    returns: list,
    data_sha: str,
    params: KellyParams,
    f: float,
) -> tuple:
    """MDD / terminal-wealth / TUW tables evaluated at fraction f.

    Spec §4/§7 require these AT f_rec, which is only known after the min-
    composition across sets — so they are filled post-composition, not inside
    the per-set pipeline. Regenerates the IDENTICAL CRN paths the guard used
    (same derived seeds; already logged by _run_set — nothing new to log).
    """
    n = len(returns)
    l_base = params.block_len
    b_dd = params.b_dd
    path_seed = derive_seed(data_sha, "paths", name, l_base, params.horizon, b_dd)
    paths_l6 = _gen_paths(random.Random(path_seed), n, b_dd, params.horizon, l_base)
    s1 = _stress_s1(returns, params.stress_mult)
    binding_cells = [("L6_S0", returns, paths_l6, None), ("L6_S1", s1, paths_l6, None)]

    mdd_rows = []
    for cell_name, data, paths, s3 in binding_cells:
        mdds = sorted(_path_mdds(data, paths, f, s3))
        k = sum(1 for m in mdds if m >= params.d_kill)
        mdd_rows.append((cell_name, _quantile(mdds, 0.50), _quantile(mdds, 0.90),
                         _quantile(mdds, 0.99), wilson_upper(k, len(mdds))))

    term_rows = []
    tuw_vals = []
    log1p = math.log1p
    for hz in (params.horizon, params.horizon_report):
        sd = derive_seed(data_sha, "paths", name, l_base, hz, b_dd)
        phz = paths_l6 if hz == params.horizon else _gen_paths(
            random.Random(sd), n, b_dd, hz, l_base)
        wealths = []
        for idx in phz:
            logw = 0.0
            peak = 0.0
            uw = 0
            for i in idx:
                x = f * returns[i]
                if 1.0 + x <= _RUIN_EPS:
                    logw = float("-inf")
                    break
                logw += log1p(x)
                if logw > peak:
                    peak = logw
                elif logw < peak:
                    uw += 1
            wealths.append(math.exp(logw) if logw != float("-inf") else 0.0)
            if hz == params.horizon:
                tuw_vals.append(uw / hz)
        wealths.sort()
        term_rows.append((hz, _quantile(wealths, 0.05), _quantile(wealths, 0.25),
                          _quantile(wealths, 0.50), _quantile(wealths, 0.75),
                          _quantile(wealths, 0.95)))
    tuw_sorted = sorted(tuw_vals)
    return (tuple(mdd_rows), tuple(term_rows),
            (_quantile(tuw_sorted, 0.50), _quantile(tuw_sorted, 0.95)))


def _stability_disagrees(stability_f_rec: Optional[Decimal], f_rec: Decimal,
                         tol: float) -> bool:
    """Spec §5(6)/§8: MC_UNSTABLE only if |Δf_rec| > tol (one guard grid step);
    Δ = tol exactly must PASS. Evaluated in Decimal — float subtraction of two
    5-dp Decimals is not exact at the boundary (e.g. float |0.250 − 0.245|
    > 0.005), which made the pre-registered tolerance value-dependent."""
    if stability_f_rec is None:
        return True
    return abs(stability_f_rec - f_rec) > Decimal(str(tol))


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

def _validate_data_contract(returns: list) -> None:
    """Raise KellyCalibrationError on contract breaks — never a report, never a clip."""
    if not returns:
        raise KellyCalibrationError("empty return series")
    for r in returns:
        if not math.isfinite(r):
            raise KellyCalibrationError(f"non-finite return {r!r} — data contract broken")
        if r < -1.0 - 1e-9:
            raise KellyCalibrationError(
                f"return {r} < -1: max-loss accounting upstream is broken — never clip"
            )
        if r > 1.0:
            raise KellyCalibrationError(
                f"return {r} > 1: impossible for the Sleeve-A OTM credit spread "
                "(max win = credit/(width-credit) < 1). This bound is structure-"
                "specific — revisit before reusing this module on other structures."
            )
    if len(set(returns)) == 1:
        raise KellyCalibrationError("zero-variance return series — data problem")


# ---------------------------------------------------------------------------
# Public API — optimization
# ---------------------------------------------------------------------------

def optimize_kelly_fraction(
    returns_full: Sequence[Decimal],
    *,
    data_snapshot_sha256: str,
    params: KellyParams = DEFAULTS,
) -> KellyOptimizationResult:
    """Run the pre-registered composite rule on E_full and its last-30% E_oos.

    Pure function: no I/O, no clock, no global state. Status here is
    CANDIDATE / NOT_SIGNABLE / MC_UNSTABLE; walk-forward gates and the D_kill
    governance confirmation are applied by finalize_status().
    """
    returns = [float(r) for r in returns_full]
    _validate_data_contract(returns)

    # E_oos slice convention: round() — the repo's pre-registered oos_returns()
    # convention (put_spread_pnl.py:267). Spec §2's formula text says ceil()
    # while citing that convention (the spec self-contradicts); resolved toward
    # round() INTERIM pending explicit adjudication (REVIEW_FINDINGS_KELLY_OPT_v1
    # F4), disclosed in the report's methodology_deviations. Changing it is a
    # new trial (§8).
    n_oos = max(1, round(len(returns) * params.oos_fraction))
    sets_def = [("E_full", returns), ("E_oos", returns[-n_oos:])]

    def _run_all(prefix: str) -> tuple[list, list]:
        log: list = []
        results = [
            # Stability rerun: independent seeds, IDENTICAL B (spec), no
            # report-only diagnostics (they don't affect the number).
            _run_set(name, data, data_snapshot_sha256, params, prefix, log,
                     skip_diagnostics=bool(prefix))
            for name, data in sets_def
        ]
        return results, log

    set_results, seeds_log = _run_all("")

    reason_codes = []
    for s in set_results:
        reason_codes.extend(f"{s.name}:{r}" for r in s.reasons)

    if any(not s.signable for s in set_results):
        return KellyOptimizationResult(
            status="NOT_SIGNABLE", reason_codes=tuple(reason_codes),
            f_rec=None, f_raw=None, binding_constraint="",
            sets=tuple(set_results), stability_f_rec=None,
            seeds=tuple(seeds_log), params=params,
            data_snapshot_sha256=data_snapshot_sha256,
        )

    def _compose(results: list) -> tuple[float, str]:
        candidates = [
            (r.f_pipe,
             r.binding_term if r.binding_term.startswith("drawdown_guard")
             else f"shrunk_kelly:set={r.name},c*={r.c_star},f*={round(r.f_star, 6)}")
            for r in results
        ]
        candidates.append((params.f_cap, f"signing_cap:{params.f_cap}"))
        # min() keeps the FIRST of equal keys → ties resolve in the fail-closed
        # pre-registered order: E_full, E_oos, signing cap.
        return min(candidates, key=lambda t: t[0])

    f_raw, binding = _compose(set_results)
    f_rec = Decimal(str(f_raw)).quantize(Decimal("0.00001"), rounding=ROUND_FLOOR)

    if f_rec < Decimal("0.005"):
        return KellyOptimizationResult(
            status="NOT_SIGNABLE",
            reason_codes=tuple(reason_codes) + ("f_rec_below_guard_grid_floor",),
            f_rec=None, f_raw=f_raw, binding_constraint=binding,
            sets=tuple(set_results), stability_f_rec=None,
            seeds=tuple(seeds_log), params=params,
            data_snapshot_sha256=data_snapshot_sha256,
        )

    # --- stability rerun: independent seeds, fast mode, must agree within tol ---
    stab_results, stab_log = _run_all("stability~")
    seeds_log.extend(stab_log)
    stability_f_rec = None
    if all(s.signable for s in stab_results):
        f_raw_b, _ = _compose(stab_results)
        stability_f_rec = Decimal(str(f_raw_b)).quantize(
            Decimal("0.00001"), rounding=ROUND_FLOOR)
    if _stability_disagrees(stability_f_rec, f_rec, params.stability_tol):
        return KellyOptimizationResult(
            status="MC_UNSTABLE",
            reason_codes=tuple(reason_codes) + (
                f"stability_rerun={stability_f_rec} vs {f_rec}, tol={params.stability_tol}",),
            f_rec=None, f_raw=f_raw, binding_constraint=binding,
            sets=tuple(set_results), stability_f_rec=stability_f_rec,
            seeds=tuple(seeds_log), params=params,
            data_snapshot_sha256=data_snapshot_sha256,
        )

    # Fill the §4/§7 risk tables AT f_rec (only known now, post-composition).
    f_eval = float(f_rec)
    filled_sets = []
    for s, (name, data) in zip(set_results, sets_def):
        mdd_at, terminal, tuw = _risk_tables_at(
            name, data, data_snapshot_sha256, params, f_eval)
        filled_sets.append(replace(
            s, mdd_at_frec=mdd_at, terminal_wealth=terminal, time_under_water=tuw))

    warnings = tuple(w for s in set_results for w in s.warnings)
    return KellyOptimizationResult(
        status="CANDIDATE", reason_codes=tuple(reason_codes),
        f_rec=f_rec, f_raw=f_raw, binding_constraint=binding,
        sets=tuple(filled_sets), stability_f_rec=stability_f_rec,
        seeds=tuple(seeds_log), params=params,
        data_snapshot_sha256=data_snapshot_sha256, warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Public API — walk-forward validation (validation-only; NEVER selection)
# ---------------------------------------------------------------------------

def _composite_fit_at(
    train: list, data_sha: str, k: int, params: KellyParams, seeds_log: list,
) -> tuple[float, bool, str]:
    """Fit the deployed composite rule on (train-full, last-30%-of-train).

    Within walk-forward ONLY: if the last-30% slice fails its STRUCTURAL gates
    (n<min_obs or losses<min_losses — structural for early windows), the
    composite falls back to the full-window component alone (counted).
    Deployment does NOT get this fallback.

    Any NON-structural unsignability on either arm (nonpositive_edge,
    no_positive_growth_rule_under_uncertainty, guard_floor_breach) means the
    deployed rule is NOT_SIGNABLE (§2: both sets must be signable) → the replay
    sits the window out at f = 0. §6 authorizes the fallback ONLY for the
    structural gates; falling back on edge failure would replay a rule more
    permissive than the deployed one, exactly in the windows where the
    recent-data arm says the edge is gone (fail-closed: f = 0). A composite
    below the guard grid floor is likewise NOT_SIGNABLE at deployment
    (§5 step 4) → sits out. Sit-outs are counted and disclosed separately
    from structural fallbacks.

    Returns (f, fell_back_structural, sit_out_reason); sit_out_reason is ""
    when the window trades.
    """
    n_oos = max(1, round(len(train) * params.oos_fraction))
    arms = [("full", train), ("oos", train[-n_oos:])]
    pipes = []
    fell_back = False
    for arm_name, data in arms:
        n_loss = sum(1 for r in data if r < 0.0)
        if len(data) < params.min_obs or n_loss < params.min_losses:
            if arm_name == "oos":
                fell_back = True
                continue
            # full window itself structurally unfit → sit out (fail-closed)
            return 0.0, False, "full_window_structurally_unfit"
        log: list = []
        sr = _run_set(f"wf{k}-{arm_name}", data, data_sha, params,
                      f"wf~{k}~{arm_name}~", log, skip_diagnostics=True,
                      b_shrink_override=params.b_shrink_wf,
                      b_dd_override=params.b_dd_wf)
        seeds_log.extend(log)
        if not sr.signable:
            return 0.0, fell_back, f"not_signable:{arm_name}:{','.join(sr.reasons)}"
        pipes.append(sr.f_pipe)
    if not pipes:  # defensive: unreachable (full arm never skips)
        return 0.0, fell_back, "no_signable_arm"
    f = min(min(pipes), params.f_cap)
    f = float(Decimal(str(f)).quantize(Decimal("0.00001"), rounding=ROUND_FLOOR))
    if f < params.f_grid_step:
        return 0.0, fell_back, "below_guard_grid_floor"
    return f, fell_back, ""


def walkforward_validation(
    returns_full: Sequence[Decimal],
    entry_dates: Sequence,
    expiry_dates: Sequence,
    *,
    data_snapshot_sha256: str,
    params: KellyParams = DEFAULTS,
) -> WalkforwardResult:
    """Expanding-window replay of the deployed composite rule vs comparators.

    Training set at refit date t_k = trades whose EXPIRY ≤ entry date of trade k
    (explicit purge: index slicing alone would leak the in-flight 28-DTE trade
    that settles on/about t_k). Comparators are pre-registered, replayed
    identically, and diagnostics-only.
    """
    returns = [float(r) for r in returns_full]
    _validate_data_contract(returns)
    n = len(returns)
    if not (len(entry_dates) == len(expiry_dates) == n):
        raise KellyCalibrationError("walkforward: dates and returns length mismatch")

    seeds_log: list = []
    refits: list = []
    fallback_count = 0
    sit_out_count = 0
    below_floor_count = 0
    instability = []

    # Rule f-paths, applied trade-by-trade from the first refit onward.
    applied: dict = {"optimized": [], "incumbent": [], "half_kelly": [], "full_kelly": []}
    applied_returns: list = []
    applied_dates: list = []

    k = params.warmup_trades
    last_f_opt = None
    while k < n:
        t_k = entry_dates[k]
        train = [returns[i] for i in range(n) if expiry_dates[i] <= t_k]
        if len(train) >= params.min_obs:
            f_opt, fell_back, sit_out = _composite_fit_at(
                train, data_snapshot_sha256, k, params, seeds_log)
            if fell_back:
                fallback_count += 1
            if sit_out == "below_guard_grid_floor":
                below_floor_count += 1
            elif sit_out:
                sit_out_count += 1
            f_star_t, _ = full_kelly(train)
            f_half = min(0.5 * f_star_t, params.f_cap)
            f_full = min(f_star_t, params.f_cap)
        else:
            f_opt, f_half, f_full = 0.0, 0.0, 0.0
            fell_back, sit_out = False, "train_below_min_obs"
        if last_f_opt is not None and abs(f_opt - last_f_opt) > 0.05:
            instability.append(
                f"refit@{t_k}: |Δf|={abs(f_opt - last_f_opt):.4f} > 0.05")
        last_f_opt = f_opt
        refits.append(WalkforwardRefit(
            k=k, entry_date=str(t_k), n_train=len(train),
            f_applied=f_opt, fell_back_to_full=fell_back,
            sit_out_reason=sit_out,
            f_half_kelly=f_half, f_full_kelly=f_full,
        ))
        block_end = min(k + params.refit_cadence, n)
        for i in range(k, block_end):
            applied["optimized"].append(f_opt)
            applied["incumbent"].append(_INCUMBENT)
            applied["half_kelly"].append(f_half)
            applied["full_kelly"].append(f_full)
            applied_returns.append(returns[i])
            applied_dates.append(entry_dates[i])
        k = block_end

    def _metrics(rule: str) -> WalkforwardRuleMetrics:
        fs = applied[rule]
        logw = 0.0
        peak = 0.0
        max_dd_log = 0.0
        under = 0
        ruined = False
        decade_totals: dict = {"1996-2005": [0.0, 0], "2006-2015": [0.0, 0],
                               "2016-2026": [0.0, 0]}
        for f, r, d in zip(fs, applied_returns, applied_dates):
            x = f * r
            if 1.0 + x <= _RUIN_EPS:
                ruined = True
                logw = float("-inf")
                break
            g = math.log1p(x)
            logw += g
            if logw > peak:
                peak = logw
            else:
                under += 1
            max_dd_log = max(max_dd_log, peak - logw)
            y = d.year if hasattr(d, "year") else int(str(d)[:4])
            key = "1996-2005" if y <= 2005 else ("2006-2015" if y <= 2015 else "2016-2026")
            decade_totals[key][0] += g
            decade_totals[key][1] += 1
        n_applied = len(fs)
        mean_g = (logw / n_applied) if (n_applied and not ruined) else float("-inf")
        return WalkforwardRuleMetrics(
            rule=rule,
            total_log_growth=logw,
            annualized_growth=(math.exp(_ANNUALIZATION * mean_g) - 1.0
                               if not ruined else -1.0),
            realized_mdd=1.0 if ruined else 1.0 - math.exp(-max_dd_log),
            time_under_water_frac=(under / n_applied) if n_applied else 0.0,
            per_decade_growth=tuple(
                (lbl, (tot / cnt) if cnt else 0.0)
                for lbl, (tot, cnt) in decade_totals.items()
            ),
            ruined=ruined,
        )

    rules = tuple(_metrics(r) for r in ("optimized", "incumbent", "half_kelly", "full_kelly"))
    opt, inc = rules[0], rules[1]
    gate_no_ruin = not opt.ruined
    gate_dkill = opt.realized_mdd <= params.d_kill
    gate_not_worse = opt.realized_mdd <= inc.realized_mdd
    return WalkforwardResult(
        refits=tuple(refits), rules=rules,
        gate_no_ruin=gate_no_ruin, gate_mdd_below_dkill=gate_dkill,
        gate_mdd_not_worse_than_incumbent=gate_not_worse,
        passed=gate_no_ruin and gate_dkill and gate_not_worse,
        fallback_count=fallback_count,
        refit_instability_warnings=tuple(instability),
        seeds=tuple(seeds_log),
        sit_out_count=sit_out_count,
        below_floor_count=below_floor_count,
    )


# ---------------------------------------------------------------------------
# Final status composition
# ---------------------------------------------------------------------------

def finalize_status(
    opt: KellyOptimizationResult,
    wf: Optional[WalkforwardResult],
    params: KellyParams,
) -> tuple[str, tuple]:
    """Compose the overall report status from optimization + validation + governance.

    PROPOSED           — all gates pass AND d_kill is confirmed against the
                         committed §5 kill threshold.
    PENDING_GOVERNANCE — all gates pass but d_kill is an unconfirmed placeholder
                         (§5 kill thresholds not yet committed in writing): the
                         conditional recommendation is shown but cannot be signed
                         until the operator confirms/replaces D_kill and re-runs.
    VALIDATION_FAILED  — a walk-forward hard gate failed (NOT_SIGNABLE).
    NOT_SIGNABLE / MC_UNSTABLE — propagated from the optimization stage.
    """
    if opt.status != "CANDIDATE":
        return opt.status, opt.reason_codes
    if wf is None:
        return "VALIDATION_FAILED", ("walkforward_missing",)
    if not wf.refits:
        # Zero refits = zero validation evidence — evidentially identical to a
        # missing walk-forward, so it fails closed the same way (NF-07).
        # Unreachable at the default constants (E_oos min_obs forces n ≥ 199
        # → ≥ 6 refits) but overridden warmup/min_obs are the documented path
        # for short-history recalibrations, where an empty refit table would
        # otherwise pass all three gates vacuously.
        return "VALIDATION_FAILED", ("walkforward_no_refits",)
    if not wf.passed:
        codes = [
            code for code, ok in (
                ("wf_ruin", wf.gate_no_ruin),
                ("wf_mdd_exceeds_d_kill", wf.gate_mdd_below_dkill),
                ("wf_mdd_worse_than_incumbent", wf.gate_mdd_not_worse_than_incumbent),
            ) if not ok
        ]
        return "VALIDATION_FAILED", tuple(codes)
    if not params.d_kill_confirmed:
        return "PENDING_GOVERNANCE", ("d_kill_unconfirmed_placeholder",)
    return "PROPOSED", ()
