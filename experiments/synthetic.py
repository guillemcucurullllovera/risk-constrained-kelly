import json
import numpy as np
from risk_kelly.sizing.evt import ScenarioSet
from risk_kelly.sizing.rck import solve_rck,lambda_from_drawdown,verify_drawdown_montecarlo
from risk_kelly.kkt import residuals

s=ScenarioSet(np.array([[1.2,1.],[0.8,1.]]),np.array([0.6,0.4]))
lam=lambda_from_drawdown(0.7,0.1)
result=solve_rck(s,lam,solver='CLARABEL')
replay=verify_drawdown_montecarlo(s,result.b,lam,alpha=0.7,seed=2026,n_paths=3000,horizon=80)
print(json.dumps({'seed':2026,'allocation':result.b.tolist(),'moment':result.drawdown_moment,
                 'kkt':residuals(s,result.b,lam),'finite_replay_breach':replay.empirical_prob,
                 'scenario_law_bound':replay.theoretical_bound},indent=2))
