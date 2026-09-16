"""
Verification of the Riemann-Liouville (and Caputo) half-derivative of f(x)=sqrt(x).

Closed form (Riemann-Liouville), 0<alpha<1, f(x)=x^beta:
    D^alpha[x^beta] = Gamma(beta+1)/Gamma(beta+1-alpha) * x^(beta-alpha)
With beta = alpha = 1/2:
    D^(1/2)[x^(1/2)] = Gamma(3/2)/Gamma(1) * x^0 = sqrt(pi)/2   (a CONSTANT)

So every method must converge to sqrt(pi)/2 ~= 0.886226925452758 at x = 0.5.
Caputo gives the same value here because f(0)=0.

Return-shape notes (probed on this venv):
  * differint.RL(alpha,f,0,1,N)            -> ndarray of length N (N x N matrix method)
  * differint.RLpoint / CaputoL1point      -> scalar
  * pyfod.fod.riemannliouville / caputo    -> dict with key 'fd' = the derivative value
  * numfracpy.RL_der1(f,t,dt,alpha)        -> scalar/array (last element at t)
"""
import json, math, time
import numpy as np
import differint.differint as df
import pyfod.fod as fod
import numfracpy as nfp

X = 0.5
ANALYTIC = math.sqrt(math.pi) / 2.0   # = Gamma(3/2)/Gamma(1)
f = lambda t: t ** 0.5
results = {"x": X, "analytic_sqrt_pi_over_2": ANALYTIC}

def run(key, fn):
    t0 = time.time()
    try:
        v = float(np.atleast_1d(fn()).astype(float).flatten()[-1])
        results[key] = {"value": v, "sec": round(time.time()-t0, 2),
                        "abs_err": abs(v-ANALYTIC), "rel_err": abs(v-ANALYTIC)/ANALYTIC}
    except Exception as e:
        results[key] = {"error": f"{type(e).__name__}: {e}", "sec": round(time.time()-t0, 2)}
    print(f"[done] {key}  {results[key].get('value', results[key].get('error'))}  ({results[key]['sec']}s)", flush=True)

# --- Riemann-Liouville -------------------------------------------------------
run("differint_RL",       lambda: df.RL(0.5, f, 0.0, 1.0, 5001))
run("differint_RLpoint",  lambda: df.RLpoint(0.5, f, 0.0, 1.0, 5001))
run("pyfod_RL",           lambda: fod.riemannliouville(f, lower=0.0, upper=X, dt=1e-4, alpha=0.5)["fd"])
run("numfracpy_RL_der1",  lambda: nfp.RL_der1(f, t=X, dt=1e-4, alpha=0.5))

# --- Caputo (equals R-L here since f(0)=0) -----------------------------------
run("differint_CaputoL1", lambda: df.CaputoL1point(0.5, f, 0.0, 1.0, 5001))
run("pyfod_Caputo",       lambda: fod.caputo(f, lower=0.0, upper=X, dt=1e-4, alpha=0.5)["fd"])

print(json.dumps(results, indent=2))
with open("C:/Users/MATTIA/source/hermes-dir/frac_results.json", "w") as fh:
    json.dump(results, fh, indent=2)
