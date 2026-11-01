# Mathematical scope

Busseti, Ryu & Boyd (2016): maximize sum p_i log(R_i b), with b>=0,
sum b=1 and sum p_i (R_i b)^(-lambda)<=1; lambda=log(beta)/log(alpha).
The moment condition gives an infinite-horizon wealth-floor bound under the
specified stationary IID scenario law, not an empirical guarantee for changing
markets. The original solver rechecks the exact moment after CVXPY's rational
power approximation and fails closed on material violations.

Original modules retain scenario validation, convex and quadratic approximations,
fractional/shrunken Kelly ceilings, bootstrap calibration, walk-forward validation
and Wilson drawdown diagnostics. Random feasible perturbations are a heuristic
optimality probe, not a proof of KKT stationarity. A separate release test computes
primal feasibility, dual-sign, stationarity and complementary-slackness residuals.
Private application firewall/report-driver tests are individually listed as
excluded integrations, not counted as passed.

Synthetic scenarios only. Solver tolerances are explicit; a mean haircut can alter the risk law, so a feasible haircut-law solution need not meet the raw-law bound. Monte Carlo confidence intervals describe seeded finite-horizon replays only; no trading profitability or production approval is claimed.
