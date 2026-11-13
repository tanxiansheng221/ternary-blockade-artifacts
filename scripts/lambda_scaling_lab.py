# -*- coding: utf-8 -*-
import os as _aim_os
AIM_SEED = int(_aim_os.environ.get("AIM_SEED", "1"))
AIM_TAG = _aim_os.environ.get("AIM_TAG", "")
_OFF = (AIM_SEED - 1) * 2

# LAMBDA SCALING LAB (problem 26): sweep weight decay, measure generalization
# transition step T_gen -> extraction inefficiency curve. Pure numpy, CPU.
import numpy as np, time, json

R = lambda s: print(s, flush=True)
rng = np.random.default_rng(0 + _OFF)
P = 97; EMB, HID = 96, 192
STEPS, BATCH = 6000, 1024
lr = 2e-3
WDS = [0.1, 0.5, 1.0, 2.0, 5.0]

pairs = [(a, b) for a in range(P) for b in range(P)]
labels = np.array([(a + b) % P for a, b in pairs])
idx = rng.permutation(len(pairs))
ntr = len(pairs) // 2
tr, va = idx[:ntr], idx[ntr:]
Xa = np.array([pairs[i][0] for i in tr]); Xb = np.array([pairs[i][1] for i in tr]); Ya = labels[tr]
Va = np.array([pairs[i][0] for i in va]); Vb = np.array([pairs[i][1] for i in va]); Yv = labels[va]

def build():
    E = (rng.standard_normal((P, EMB)) * 0.5)
    W1 = (rng.standard_normal((2*EMB, HID)) / np.sqrt(2*EMB))
    W2 = (rng.standard_normal((HID, P)) / np.sqrt(HID))
    b1 = np.zeros(HID); b2 = np.zeros(P)
    return [E, W1, b1, W2, b2]

def fwd(p, pa, pb):
    E, W1, b1, W2, b2 = p
    Hx = np.concatenate([E[pa], E[pb]], 1)
    A1 = Hx @ W1 + b1
    Z = np.maximum(A1, 0)
    return Z @ W2 + b2, (Hx, A1, Z)

def acc(p, pa, pb, y):
    lg, _ = fwd(p, pa, pb)
    return (lg.argmax(1) == y).mean()

def run(wd):
    p = build()
    m = [np.zeros_like(x) for x in p]; v = [np.zeros_like(x) for x in p]
    t_adam = 0; T_mem = T_gen = None; tr_hist = []
    for step in range(1, STEPS+1):
        bi = rng.choice(ntr, size=BATCH, replace=True)
        logits, (Hx, A1, Z) = fwd(p, Xa[bi], Xb[bi])
        y = Ya[bi]; B = len(bi)
        lg = logits - logits.max(1, keepdims=True)
        Psm = np.exp(lg); Psm /= Psm.sum(1, keepdims=True)
        dlog = Psm.copy(); dlog[np.arange(B), y] -= 1; dlog /= B
        W1, W2 = p[1], p[3]
        dW2 = Z.T @ dlog; db2 = dlog.sum(0)
        dZ = dlog @ W2.T; dZ[A1 <= 0] = 0
        dW1 = Hx.T @ dZ; db1 = dZ.sum(0)
        dHx = dZ @ W1.T
        dE = np.zeros_like(p[0])
        np.add.at(dE, Xa[bi], dHx[:, :EMB])
        np.add.at(dE, Xb[bi], dHx[:, EMB:])
        grads = [dE, dW1, db1, dW2, db2]
        t_adam += 1
        for i, (pp, g) in enumerate(zip(p, grads)):
            m[i] = 0.9*m[i] + 0.1*g
            v[i] = 0.999*v[i] + 0.001*g*g
            pp -= lr*((m[i]/(1-0.9**t_adam))/(np.sqrt(v[i]/(1-0.999**t_adam))+1e-8) + wd*pp)
        ta = acc(p, Xa, Xb, Ya)
        if T_mem is None and ta > 0.9: T_mem = step
        if T_gen is None and ta > 0.9 and acc(p, Va, Vb, Yv) > 0.9: T_gen = step
    va_f = acc(p, Va, Vb, Yv)
    return T_mem, T_gen, va_f

R("=== LAMBDA SCALING LAB (problem 26) ===")
R("task (a+b) mod %d | 50%% data | sweep weight decay | %d steps each" % (P, STEPS))
R("%6s %10s %10s %12s %14s" % ("wd", "T_mem", "T_gen", "final_val", "eta(ineff)"))
results = []
t0 = time.perf_counter()
rule_bits = np.log2(P)
for wd in WDS:
    T_mem, T_gen, va_f = run(wd)
    if T_gen:
        eta = T_gen * BATCH * rule_bits / rule_bits / ntr  # pair-presentations per pair
        eta = T_gen * BATCH / ntr
    else:
        eta = None
    R("%6.1f %10s %10s %12.4f %14s" % (wd, T_mem, T_gen, va_f,
        ("%.0fx" % eta) if eta else "no-gen<%d" % STEPS))
    results.append(dict(wd=wd, T_mem=T_mem, T_gen=T_gen, final_val=va_f, eta=eta))
R("wall: %.0fs" % (time.perf_counter()-t0))
json.dump(results, open("lambda_scaling%s_results.json" % (("_" + AIM_TAG) if AIM_TAG else ""), "w"), indent=1)
R("saved: lab/cpu-native/lambda_scaling_results.json")
