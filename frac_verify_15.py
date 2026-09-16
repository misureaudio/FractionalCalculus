"""
Alpha = 3/2, f(x) = sqrt(x), evaluated at x = 0.5.

Analytic target (Riemann-Liouville):
    I^{1/2}[sqrt(x)] = 1/Gamma(1/2) * Int_0^x (x-t)^(-1/2) t^(1/2) dt = (sqrt(pi)/2) x
    D^{3/2}_RL[sqrt(x)] = 1/Gamma(1/2) * d^2/dx^2 [(sqrt(pi)/2) x] = 0   (EXACT)
This is the null-space case: x^beta with beta+1-alpha = 1/2+1-3/2 = 0.
Caputo order 3/2 is UNDEFINED for sqrt(x) (f'' = -1/(4 t^{3/2}) not in L^1).
Metric: ABSOLUTE error (relative is undefined at a zero target).
"""
import json, math, time
import numpy as np
import differint.differint as df
import pyfod.fod as fod
import numfracpy as nfp

X = 0.5
ALPHA = 1.5
ANALYTIC = 0.0
H = 1e-4
f = lambda t: t**0.5

results = {
    "x": X, "alpha": ALPHA, "analytic": ANALYTIC,
    "note": ("D^{3/2}_RL[sqrt(x)] = d^2/dx^2[(sqrt(pi)/2) x] = 0 exactly "
             "(null space: beta+1-alpha=0). Metric = abs_err.")
}

def run(key, fn):
    t0 = time.time()
    try:
        r = fn()
        v = float(np.atleast_1d(r).astype(float).flatten()[-1])
        results[key] = {"value": v, "sec": round(time.time()-t0, 2), "abs_err": abs(v-ANALYTIC)}
    except Exception as e:
        results[key] = {"error": f"{type(e).__name__}: {e}", "sec": round(time.time()-t0, 2)}
    print(f"[done] {key}: {results[key].get('value', results[key].get('error'))} ({results[key]['sec']}s)", flush=True)

def run_grid(key, fn):
    t0 = time.time()
    try:
        arr = np.asarray(fn(), dtype=float)
        xi = np.linspace(0.0, 1.0, len(arr))
        idx = int(np.argmin(np.abs(xi-X)))
        v = float(arr[idx])
        results[key] = {"value": v, "grid_x": float(xi[idx]), "sec": round(time.time()-t0, 2),
                        "abs_err": abs(v-ANALYTIC)}
    except Exception as e:
        results[key] = {"error": f"{type(e).__name__}: {e}", "sec": round(time.time()-t0, 2)}
    print(f"[done] {key}: {results[key].get('value', results[key].get('error'))} ({results[key]['sec']}s)", flush=True)

# ===== methods that SUPPORT alpha=1.5 =====
run("pyfod_grunwaldletnikov", lambda: fod.grunwaldletnikov(f, lower=0.0, upper=X, n=int(X/H), alpha=ALPHA)["fd"])
run("numfracpy_RL_der2", lambda: nfp.RL_der2(f, t=X, dt=H, alpha=ALPHA))
run("differint_GLpoint", lambda: df.GLpoint(ALPHA, f, 0.0, X, 4001))
run_grid("differint_GL_fft", lambda: df.GL(ALPHA, f, 0.0, 1.0, 4001))

# my own direct GL sum (independent of any package)
def gl_direct():
    M = int(X//H)
    m = np.arange(0, M+1)
    ratios = (ALPHA - np.arange(0, M))/np.arange(1, M+1)
    b = np.concatenate([[1.0], np.cumprod(ratios)])          # b[m] = binom(ALPHA, m)
    x = np.maximum(X - m*H, 0.0)
    terms = ((-1.0)**m) * b * np.sqrt(x)
    return terms.sum()/(H**ALPHA)
run("direct_GL_sum", gl_direct)

# ===== methods that should NOT support alpha=1.5 (record behavior) =====
run("differint_RL", lambda: df.RL(ALPHA, f, 0.0, 1.0, 2001))
run("differint_RLpoint", lambda: df.RLpoint(ALPHA, f, 0.0, 1.0, 2001))
run("pyfod_riemannliouville", lambda: fod.riemannliouville(f, lower=0.0, upper=X, dt=H, alpha=ALPHA)["fd"])
run("numfracpy_RL_der1", lambda: nfp.RL_der1(f, t=X, dt=H, alpha=ALPHA))
run("differint_CaputoL1", lambda: df.CaputoL1point(ALPHA, f, 0.0, 1.0, 2001))
run("pyfod_caputo", lambda: fod.caputo(f, lower=0.0, upper=X, dt=H, alpha=ALPHA)["fd"])
run("numfracpy_Caputo_der2", lambda: nfp.Caputo_der2(f, t=X, dt=H, alpha=ALPHA))

print(json.dumps(results, indent=2))
with open("C:/Users/MATTIA/source/hermes-dir/frac_results_15.json", "w") as fh:
    json.dump(results, fh, indent=2)
