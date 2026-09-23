"""S2Mamba Eq9/10/20 + Alg2 local adaptation, NOT author coefficients/RTL.

The fitting grid contains no labels/scene activations; all six cores share one
Softplus fit. Internal arithmetic is Q30/Q24, matching N0/N1 comparison context,
not the paper's complete 16-bit network. Only offline fitting uses real math.
"""
import math
import json
from pathlib import Path

Q = 1 << 30
LOG2E = round(math.log2(math.e) * Q)
LN2 = round(math.log(2) * Q)


def fit(seed=20260913, steps=5000):
    import numpy as np
    rng = np.random.default_rng(seed)
    x = rng.uniform(-16., 16., 4096)
    true = np.logaddexp(0., x)
    y = x * math.log2(math.e)
    u = np.floor(y)
    v = y - u
    basis = np.column_stack([v, np.ones_like(v)])
    initial = np.linalg.lstsq(basis, 2.**v, rcond=None)[0]

    def anneal(initial, objective):
        best = current = np.array(initial, dtype=float)
        best_loss = current_loss = initial_loss = objective(current)
        for i in range(steps):
            t = i / max(1, steps-1)
            candidate = current + rng.normal(0., .03 * (1-t)**2 + .00001, 2)
            if not (.1 < candidate[0] < 2. and -.2 < candidate[1] < 1.5):
                continue
            loss = objective(candidate)
            temperature = max(initial_loss, 1.e-5) * .2 * (1-t)**3 + 1.e-10
            if loss < current_loss or rng.random() < math.exp(min(0., (current_loss-loss)/temperature)):
                current, current_loss = candidate, loss
            if loss < best_loss:
                best, best_loss = candidate.copy(), loss
        return best, dict(initial_mse=float(initial_loss), final_mse=float(best_loss))

    def stage1(p):
        if p[1] <= 0 or p[0]+p[1] <= 0:
            return 1.e30
        return float(np.mean((np.log1p((p[0]*v+p[1])*2.**u)-true)**2))

    exp_p, first = anneal(initial, stage1)
    alpha = 1 + (exp_p[0]*v + exp_p[1])*2.**u
    w = np.floor(np.log2(alpha))
    kminus1 = alpha / 2.**w - 1
    basis2 = np.column_stack([kminus1, np.ones_like(kminus1)])
    target2 = true - w * math.log(2)
    init2 = np.linalg.lstsq(basis2, target2, rcond=None)[0]
    log_p, second = anneal(init2, lambda p: float(np.mean((basis2@p-target2)**2)))
    # Standalone Exp(Delta*A) uses its own linear 2^v approximation.
    grid = np.linspace(0, 1, 4096, endpoint=False)
    exp_state = np.linalg.lstsq(np.column_stack([grid, np.ones_like(grid)]), 2.**grid, rcond=None)[0]
    return dict(schema=1, method="S2Mamba-inspired Eq9/10/20 and two-stage Alg2 adaptation",
                seed=seed, steps=steps, samples=4096, fit_range=[-16.,16.], loss="mean squared Softplus error",
                notes="Local coefficients, not author coefficients; no WIS/SiLU/full 16-bit network reproduction",
                EXP_SOFT_Q30=[round(float(z)*Q) for z in exp_p],
                LOG_SOFT_Q30=[round(float(z)*Q) for z in log_p],
                EXP_STATE_Q30=[round(float(z)*Q) for z in exp_state],
                LOG2E_Q30=LOG2E, LN2_Q30=LN2, stage1=first, stage2=second,
                RTL_latency_registers=13, target_II=1)


def exp_q30(x24, coeff):
    y30 = (x24 * LOG2E) >> 24
    u, v = y30 >> 30, y30 & (Q-1)
    affine = (v * coeff[0] + (coeff[1] << 30)) >> 30
    if u < -63:
        return 0
    if u > 32:
        raise OverflowError("N2 positive exp exceeds 64-bit Q30 carrier")
    return affine << u if u >= 0 else affine >> -u


def online(qdt, sd, decay, scale, kf, params):
    if not -128 <= qdt <= 127:
        raise ValueError("INT8 dt required")
    magnitude = (abs(qdt)*sd+32) >> 6
    x24 = -magnitude if qdt < 0 else magnitude
    if abs(x24) > 16*(1<<24):
        raise ValueError("N2 fit domain [-16,16] exceeded; refit, do not silently extrapolate")
    alpha = Q + exp_q30(x24, params['EXP_SOFT_Q30'])
    w = alpha.bit_length()-1-30
    assert w >= 0
    k30 = alpha >> w
    lp = params['LOG_SOFT_Q30']
    delta30 = w*LN2 + (((k30-Q)*lp[0]) >> 30) + lp[1]
    delta24 = max(0, delta30+32) >> 6
    a = []
    for d in decay:
        mag = (delta24*d+(1<<23)) >> 24
        a.append(min(1<<24, (exp_q30(-mag, params['EXP_STATE_Q30'])+32) >> 6))
    raw_k = (delta24*scale+(1<<(63-kf))) >> (64-kf)
    return a, min((1<<19)-1, raw_k), raw_k > (1<<19)-1


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
