import numpy as np
from risk_kelly.sizing.evt import ScenarioSet
from risk_kelly.sizing.rck import solve_rck,lambda_from_drawdown
from risk_kelly.kkt import residuals


def test_convex_solve_has_small_primal_dual_kkt_residuals():
    s=ScenarioSet(np.array([[1.2,1.],[0.8,1.]]),np.array([0.6,0.4]))
    lam=lambda_from_drawdown(0.7,0.1)
    result=solve_rck(s,lam,solver='CLARABEL')
    report=residuals(s,result.b,lam)
    assert report['primal_residual'] < 1e-4
    assert report['stationarity_residual'] < 2e-3
    assert report['complementarity_residual'] < 2e-3
    assert report['dual_sign_residual'] == 0
