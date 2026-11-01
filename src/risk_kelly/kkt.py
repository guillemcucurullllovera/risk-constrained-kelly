"""Explicit residuals for the finite-scenario RCK program (not a perturbation test)."""
import numpy as np
from scipy.optimize import lsq_linear


def residuals(scenarios,b,lam,active_tolerance=1e-4):
    b=np.asarray(b,float);R=scenarios.returns;p=scenarios.probabilities
    g=R@b
    if np.any(g<=0):raise ValueError('non-positive portfolio gross return')
    moment=float(p@g**(-lam));growth_gradient=R.T@(p/g)
    risk_gradient=-lam*(R.T@(p*g**(-lam-1)))
    active=moment>=1-active_tolerance
    zeros=np.flatnonzero(b<1e-7)
    columns=([risk_gradient] if active else [])+[np.ones(len(b))]
    columns += [-np.eye(len(b))[:,i] for i in zeros]
    lower=([0.] if active else [])+[-np.inf]+[0.]*len(zeros)
    upper=[np.inf]*len(lower)
    fit=lsq_linear(np.column_stack(columns),growth_gradient,bounds=(lower,upper),tol=1e-12)
    mu=float(fit.x[0]) if active else 0.
    eta=np.zeros(len(b));eta[zeros]=fit.x[(2 if active else 1):]
    return {'primal_residual':float(max(abs(b.sum()-1),max(0,-b.min()),max(0,moment-1))),
            'stationarity_residual':float(np.max(np.abs(np.column_stack(columns)@fit.x-growth_gradient))),
            'complementarity_residual':float(max(abs(mu*(moment-1)),np.max(np.abs(eta*b)))),
            'dual_sign_residual':float(max(0,-mu,-eta.min())), 'risk_multiplier':mu}
