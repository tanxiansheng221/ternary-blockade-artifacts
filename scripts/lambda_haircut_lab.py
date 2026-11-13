# -*- coding: utf-8 -*-
import os as _aim_os
AIM_SEED = int(_aim_os.environ.get("AIM_SEED", "1"))
AIM_TAG = _aim_os.environ.get("AIM_TAG", "")
_OFF = (AIM_SEED - 1) * 2

# EXPERIMENT N (question 22): the lambda-haircut curve, direct measurement.
# Same architecture (emb+MLP, mod-97 addition, 50% train split), five prior
# pressure levels (wd in {0.1, 0.5, 1.0, 2.0, 3.0}), 6000 steps each.
# Metrics per level:
#   T_mem  = first step with train acc >= 0.9
#   T_gen  = first step with val acc >= 0.9
#   haircut width = T_gen - T_mem (steps where train loss says "done" but val ~0)
#   val@T_mem     = haircut depth (prediction: ~0)
#   OOD transfer  = final accuracy on mod-89 pairs (both operands < 89, argmax
#                   over first 89 logits vs (a+b) mod 89)
# Theory under test: lambda redundancy pays for generalization, memory solution
# contributes zero OOD -> transfer rate should be a function of prior level.
import numpy as np, time, json

R = lambda s: print(s, flush=True)
OUT = ("lambda_haircut_results%s.json" % (("_" + AIM_TAG) if AIM_TAG else ""))
P = 97
EMB, HID = 96, 192
lr = 2e-3
STEPS, BATCH, LOG = 6000, 1024, 250

rng0 = np.random.default_rng(0 + _OFF)
pairs = [(a, b) for a in range(P) for b in range(P)]
labels = np.array([(a + b) % P for a, b in pairs])
idx = rng0.permutation(len(pairs))
ntr = len(pairs) // 2
tr, va = idx[:ntr], idx[ntr:]
Xa = np.array([pairs[i][0] for i in tr]); Xb = np.array([pairs[i][1] for i in tr]); Ya = labels[tr]
Va = np.array([pairs[i][0] for i in va]); Vb = np.array([pairs[i][1] for i in va]); Yv = labels[va]

# OOD probe: mod-89 pairs (both operands < 89), read first 89 logits
P89 = 89
pairs89 = [(a, b) for a in range(P89) for b in range(P89) if a < P89 and b < P89]
lab89 = np.array([(a + b) % P89 for a, b in pairs89])
o = np.random.default_rng(5 + _OFF).permutation(len(pairs89))[:4000]
Oa = np.array([pairs89[i][0] for i in o]); Ob = np.array([pairs89[i][1] for i in o]); Oy = lab89[o]

def build(rng):
    E = (rng.standard_normal((P, EMB)) * 0.5).astype(np.float64)
    W1 = (rng.standard_normal((2 * EMB, HID)) / np.sqrt(2 * EMB)).astype(np.float64)
    return [E, W1, np.zeros(HID), (rng.standard_normal((HID, P)) / np.sqrt(HID)).astype(np.float64), np.zeros(P)]

def fwd(p, pa, pb):
    Hx = np.concatenate([p[0][pa], p[0][pb]], axis=1)
    A1 = Hx @ p[1] + p[2]
    Z = np.maximum(A1, 0)
    return Z @ p[3] + p[4], (Hx, A1, Z)

def acc(p, pa, pb, y):
    lg, _ = fwd(p, pa, pb)
    return (lg.argmax(1) == y).mean()

def ood_acc(p):
    lg, _ = fwd(p, Oa, Ob)[:, :P89] if False else (fwd(p, Oa, Ob)[0][:, :P89], None)
    return (lg.argmax(1) == Oy).mean()

def run(wd, seed=0):
    rng = np.random.default_rng(seed)
    p = build(rng)
    m = [np.zeros_like(x) for x in p]; v = [np.zeros_like(x) for x in p]
    t_adam = 0
    T_mem = T_gen = None
    traj = []
    t0 = time.perf_counter()
    for step in range(1, STEPS + 1):
        bi = rng.choice(ntr, size=BATCH, replace=True)
        logits, (Hx, A1, Z) = fwd(p, Xa[bi], Xb[bi])
        y = Ya[bi]; B = len(bi)
        lg = logits - logits.max(1, keepdims=True)
        Psm = np.exp(lg); Psm /= Psm.sum(1, keepdims=True)
        dlog = Psm.copy(); dlog[np.arange(B), y] -= 1; dlog /= B
        dW2 = Z.T @ dlog; db2 = dlog.sum(0)
        dZ = dlog @ p[3].T; dZ[A1 <= 0] = 0
        dW1 = Hx.T @ dZ; db1 = dZ.sum(0)
        dHx = dZ @ p[1].T
        dE = np.zeros_like(p[0])
        np.add.at(dE, Xa[bi], dHx[:, :EMB])
        np.add.at(dE, Xb[bi], dHx[:, EMB:])
        grads = [dE, dW1, db1, dW2, db2]
        t_adam += 1
        c1, c2, eps = 0.9, 0.999, 1e-8
        for i, (pp, g) in enumerate(zip(p, grads)):
            m[i] = c1 * m[i] + (1 - c1) * g
            v[i] = c2 * v[i] + (1 - c2) * g * g
            pp -= lr * ((m[i] / (1 - c1 ** t_adam)) / (np.sqrt(v[i] / (1 - c2 ** t_adam)) + eps) + wd * pp)
        if step % LOG == 0 or step == 1:
            ta = acc(p, Xa, Xb, Ya); va_ = acc(p, Va, Vb, Yv)
            traj.append((step, round(ta, 4), round(va_, 4)))
            if T_mem is None and ta >= 0.9:
                T_mem = step
            if T_gen is None and va_ >= 0.9:
                T_gen = step
    ta = acc(p, Xa, Xb, Ya); va_ = acc(p, Va, Vb, Yv); oo = ood_acc(p)
    val_at_mem = None
    if T_mem is not None:
        # haircut depth: val acc at the logged step closest to T_mem
        cand = [t for t in traj if t[0] <= T_mem]
        val_at_mem = cand[-1][2] if cand else None
    return dict(wd=wd, T_mem=T_mem, T_gen=T_gen,
                haircut=(T_gen - T_mem) if (T_gen and T_mem) else None,
                val_at_mem=val_at_mem, train_final=round(ta, 4), val_final=round(va_, 4),
                ood_transfer=round(oo, 4), traj=traj, wall=round(time.perf_counter() - t0))

R("=== EXPERIMENT N: lambda-haircut curve (question 22) ===")
results = {}
t0 = time.perf_counter()
for wd in (0.1, 0.5, 1.0, 2.0, 3.0):
    r = run(wd)
    results[str(wd)] = r
    R("[wd=%.1f] T_mem=%s T_gen=%s haircut=%s val@mem=%s | final train %.3f val %.3f | OOD %.3f | %.0fs" %
      (wd, r["T_mem"], r["T_gen"], r["haircut"], r["val_at_mem"],
       r["train_final"], r["val_final"], r["ood_transfer"], r["wall"]))
    json.dump(results, open(OUT, "w"), indent=1)
R("total wall %.0fs" % (time.perf_counter() - t0))
R("=== HAIRCUT CURVE ===")
R("%6s %8s %8s %10s %10s %10s" % ("wd", "T_mem", "T_gen", "haircut", "val@mem", "OOD"))
for kk, r in results.items():
    R("%6s %8s %8s %10s %10s %10.3f" % (kk, r["T_mem"], r["T_gen"], r["haircut"],
                                        r["val_at_mem"], r["ood_transfer"]))
R("saved: lab/cpu-native/lambda_haircut_results.json")
