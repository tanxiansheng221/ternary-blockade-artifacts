# -*- coding: utf-8 -*-
# EXPERIMENT Q2 (chapter 72 addendum): distillation vs ternary phase block.
# Exp Q found: ternary core blocks the y-phase in the assembly composite
# (fp enters at ~7000 steps, ternary flat at 1.4% through 16K) while the
# single-task ternary tax was ~0. Blueprint's designated antidote: distill
# fp teacher -> ternary student (soft targets carry the structure so the
# student need not co-crystallize it from sparse one-hot CE).
# Arms:
#   teacher  : fp, retrieval+aux, 12000 steps (expect ~90%+)
#   S_distill: ternary student, loss = 0.5*hard CE + 0.5*soft CE(teacher probs),
#              aux v head hard CE, 8000 steps
# Preregistered: PQ4 S_distill acc_y_fact @8K > 30% (undistilled baseline 1.4%).
import numpy as np, json, time, os, sys

R = lambda s: print(s, flush=True)
NV, D, L, H1, T = 97, 96, 2, 256, 10
V = NV + 1; PAD = 97
NTAB = 65536
B, LR, WD = 128, 2e-3, 1e-4
import os as _aim_os
AIM_SEED = int(_aim_os.environ.get("AIM_SEED", "1"))
AIM_TAG = _aim_os.environ.get("AIM_TAG", "")
_OFF = (AIM_SEED - 1) * 2
OUT = "blueprint_distill%s_results.json" % (("_" + AIM_TAG) if AIM_TAG else "")

dRng = np.random.default_rng(7)
perm = dRng.permutation(NV**3)[:NTAB]
KD = np.stack([perm // (NV*NV), (perm // NV) % NV, perm % NV], 1)
KV = dRng.integers(0, NV, size=NTAB)

QKEYS = ("Wq0", "Wk0", "Wv0", "Wo0", "W10", "W20", "Wq1", "Wk1", "Wv1", "Wo1", "W11", "W21")
def make_params(seed):
    rng = np.random.default_rng(seed)
    sc = 0.8 / np.sqrt(D)
    p = {"E": rng.normal(0, sc, (V, D)), "P": rng.normal(0, sc, (T, D))}
    for l in range(L):
        for n in ("Wq", "Wk", "Wv", "Wo"):
            p["%s%d" % (n, l)] = rng.normal(0, sc, (D, D))
        p["W1%d" % l] = rng.normal(0, sc, (D, H1)); p["b1%d" % l] = np.zeros(H1)
        p["W2%d" % l] = rng.normal(0, sc, (H1, D)); p["b2%d" % l] = np.zeros(D)
    p["Wh"] = rng.normal(0, sc, (D, NV)); p["Wy"] = rng.normal(0, sc, (D, NV))
    return p

def ternize(p):
    q = {}
    for k, w in p.items():
        if k in QKEYS:
            gs = np.abs(w).mean()
            q[k] = np.clip(np.round(w / gs), -1, 1) * gs
        else:
            q[k] = w
    return q

def fwd_caches(p, tok):
    h = p["E"][tok] + p["P"][None, :, :]
    caches = []
    for l in range(L):
        hc = h
        q = hc @ p["Wq%d" % l]; k = hc @ p["Wk%d" % l]; v = hc @ p["Wv%d" % l]
        s = (q @ k.transpose(0, 2, 1)) / np.sqrt(D)
        s = s - s.max(-1, keepdims=True)
        e = np.exp(s); A = e / e.sum(-1, keepdims=True)
        o = A @ v
        ha = hc + o @ p["Wo%d" % l]
        z = ha @ p["W1%d" % l] + p["b1%d" % l]
        r = np.maximum(z, 0.0)
        h = ha + r @ p["W2%d" % l] + p["b2%d" % l]
        caches.append((hc, q, k, v, A, o, ha, z, r))
    return h, caches

def bwd(p, tok, caches, dh):
    g = {"E": np.zeros_like(p["E"]), "P": np.zeros_like(p["P"])}
    for l in reversed(range(L)):
        hc, q, k, v, A, o, ha, z, r = caches[l]
        dr = dh @ p["W2%d" % l].T
        dz = dr * (z > 0)
        dha = dh + dz @ p["W1%d" % l].T
        g["W2%d" % l] = r.reshape(-1, H1).T @ dh.reshape(-1, D)
        g["b2%d" % l] = dh.reshape(-1, D).sum(0)
        g["W1%d" % l] = ha.reshape(-1, D).T @ dz.reshape(-1, H1)
        g["b1%d" % l] = dz.reshape(-1, H1).sum(0)
        do = dha @ p["Wo%d" % l].T
        g["Wo%d" % l] = o.reshape(-1, D).T @ dha.reshape(-1, D)
        dv = A.transpose(0, 2, 1) @ do
        g["Wv%d" % l] = hc.reshape(-1, D).T @ dv.reshape(-1, D)
        dA = do @ v.transpose(0, 2, 1)
        ds = A * (dA - (dA * A).sum(-1, keepdims=True))
        dq = ds @ k / np.sqrt(D)
        dk = ds.transpose(0, 2, 1) @ q / np.sqrt(D)
        g["Wq%d" % l] = hc.reshape(-1, D).T @ dq.reshape(-1, D)
        g["Wk%d" % l] = hc.reshape(-1, D).T @ dk.reshape(-1, D)
        dh = dha + dq @ p["Wq%d" % l].T + dk @ p["Wk%d" % l].T + dv @ p["Wv%d" % l].T
    np.add.at(g["E"], tok.reshape(-1), dh.reshape(-1, D))
    g["P"] = dh.sum(0)
    return g

def make_batch(bRng, n):
    idx = bRng.integers(0, NTAB, size=n)
    kd = KD[idx]; vv = KV[idx]
    c = bRng.integers(0, NV, size=n)
    y = (vv + c) % NV
    tok = np.full((n, T), PAD, dtype=np.int64)
    tok[:, 0:3] = kd; tok[:, 3] = vv
    tok[:, 5:8] = kd; tok[:, 8] = c
    return tok, vv, y

def softmax_xent(logits, tgt, Bn):
    lg = logits - logits.max(1, keepdims=True)
    Z = np.exp(lg); sm = Z / Z.sum(1, keepdims=True)
    nll = float(-np.log(sm[np.arange(Bn), tgt] + 1e-12).mean())
    dl = sm.copy(); dl[np.arange(Bn), tgt] -= 1.0
    dl /= Bn
    return nll, dl, sm

def train(steps, seed, distill_teacher=None, soft_w=0.5, batch_seed=None):
    """distill_teacher=None -> plain fp teacher training (hard CE only)."""
    p = make_params(seed)
    m = {k: np.zeros_like(x) for k, x in p.items()}
    v = {k: np.zeros_like(x) for k, x in p.items()}
    bRng = np.random.default_rng(batch_seed if batch_seed is not None else 4000 + seed)
    traj = []; t0 = time.perf_counter()
    for step in range(1, steps + 1):
        lr = LR * min(1.0, step / 300.0) * (0.1 + 0.45 * (1 + np.cos(np.pi * step / steps)))
        pq = ternize(p) if distill_teacher is not None else p
        tok, vv, y = make_batch(bRng, B)
        h, caches = fwd_caches(pq, tok)
        dh = np.zeros_like(h)
        g = {k: np.zeros_like(x) for k, x in p.items()}
        tot = 0.0
        # aux v head (hard)
        nll, dl, _ = softmax_xent(h[:, 3, :] @ pq["Wh"], vv, B)
        tot += nll
        g["Wh"] = h[:, 3, :].T @ dl
        dh[:, 3, :] += dl @ pq["Wh"].T
        # y head: hard + optional soft from teacher
        lg_y = h[:, 9, :] @ pq["Wy"]
        nll, dl, sm = softmax_xent(lg_y, y, B)
        if distill_teacher is not None:
            ht, _ = fwd_caches(distill_teacher, tok)
            lg_t = ht[:, 9, :] @ distill_teacher["Wy"]
            lg_t = lg_t - lg_t.max(1, keepdims=True)
            Zt = np.exp(lg_t); pt = Zt / Zt.sum(1, keepdims=True)
            tot += soft_w * float(-(pt * np.log(sm + 1e-12)).sum(1).mean())
            dl = (1.0 - soft_w) * dl + soft_w * (sm - pt) / B
        else:
            tot += nll
        g["Wy"] = h[:, 9, :].T @ dl
        dh[:, 9, :] += dl @ pq["Wy"].T
        g1 = bwd(pq, tok, caches, dh)
        for kk in g1: g[kk] = g[kk] + g1[kk]
        for kk in p:
            m[kk] = 0.9 * m[kk] + 0.1 * g[kk]
            v[kk] = 0.999 * v[kk] + 0.001 * g[kk] * g[kk]
            p[kk] -= lr * ((m[kk] / (1 - 0.9 ** step)) /
                           (np.sqrt(v[kk] / (1 - 0.999 ** step)) + 1e-8) + WD * p[kk])
        if step % 1000 == 0:
            ev = evaluate(pq if distill_teacher is not None else p)
            traj.append([step, ev])
            tag = "student" if distill_teacher is not None else "teacher"
            R("  [%s] step %5d %s (%.0fs)" % (tag, step, ev, time.perf_counter() - t0))
    return p, traj

def finetune(p, steps, lr, seed=7):
    """Constant-low-lr continuation: anneal-gated phase needs low lr to crystallize."""
    m = {k: np.zeros_like(x) for k, x in p.items()}
    v = {k: np.zeros_like(x) for k, x in p.items()}
    bRng = np.random.default_rng(5000 + seed)
    traj = []; t0 = time.perf_counter()
    for step in range(1, steps + 1):
        tok, vv, y = make_batch(bRng, B)
        h, caches = fwd_caches(p, tok)
        dh = np.zeros_like(h)
        g = {k: np.zeros_like(x) for k, x in p.items()}
        nll, dl, _ = softmax_xent(h[:, 3, :] @ p["Wh"], vv, B)
        g["Wh"] = h[:, 3, :].T @ dl
        dh[:, 3, :] += dl @ p["Wh"].T
        nll, dl, _ = softmax_xent(h[:, 9, :] @ p["Wy"], y, B)
        g["Wy"] = h[:, 9, :].T @ dl
        dh[:, 9, :] += dl @ p["Wy"].T
        g1 = bwd(p, tok, caches, dh)
        for kk in g1: g[kk] = g[kk] + g1[kk]
        for kk in p:
            m[kk] = 0.9 * m[kk] + 0.1 * g[kk]
            v[kk] = 0.999 * v[kk] + 0.001 * g[kk] * g[kk]
            p[kk] -= lr * ((m[kk] / (1 - 0.9 ** step)) /
                           (np.sqrt(v[kk] / (1 - 0.999 ** step)) + 1e-8) + WD * p[kk])
        if step % 1000 == 0:
            ev = evaluate(p)
            traj.append([step, ev])
            R("  [finetune] step %5d %s (%.0fs)" % (step, ev, time.perf_counter() - t0))
    return p, traj

def evaluate(p, n=2048, chunk=512):
    bRng = np.random.default_rng(999)
    ay = 0.0; av = 0.0; ns = 0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        tok, vv, y = make_batch(bRng, e - s)
        h = fwd_caches(p, tok)[0]
        ay += float((h[:, 9, :] @ p["Wy"]).argmax(1).__eq__(y).mean()) * (e - s)
        av += float((h[:, 3, :] @ p["Wh"]).argmax(1).__eq__(vv).mean()) * (e - s)
        ns += e - s
    return {"acc_y": round(ay / ns, 4), "acc_v": round(av / ns, 4)}

if __name__ == "__main__":
    R("=== EXPERIMENT Q2: distillation vs ternary phase block ===")
    results = {}
    R("--- teacher (fp, 8000 steps exact B_fp recipe) ---")
    tp, ttraj = train(8000, 42 + _OFF, batch_seed=3042 + _OFF)
    results["teacher_8k"] = dict(final=evaluate(tp, n=4096), traj=ttraj)
    R("=== teacher @8K %s ===" % results["teacher_8k"]["final"])
    R("--- teacher finetune (constant lr 2e-4, 4000 steps) ---")
    tp, ftraj = finetune(tp, 4000, lr=2e-4, seed=7 + _OFF)
    results["teacher_ft"] = dict(final=evaluate(tp, n=4096), traj=ftraj)
    R("=== teacher FINAL %s ===" % results["teacher_ft"]["final"])
    json.dump(results, open(OUT, "w"), indent=1)
    R("--- student (ternary, distilled, 8000 steps) ---")
    sp, straj = train(8000, 43 + _OFF, distill_teacher=tp)
    results["S_distill"] = dict(final=evaluate(ternize(sp), n=4096), traj=straj)
    json.dump(results, open(OUT, "w"), indent=1)
    R("=== S_distill FINAL %s ===" % results["S_distill"]["final"])
    R("=== SUMMARY ===")
    R("teacher fp 8K+ft:      acc_y %.3f" % results["teacher_ft"]["final"]["acc_y"])
    R("PQ4 distilled ternary: acc_y %.3f @8K (undistilled baseline 0.014 @16K)" %
      results["S_distill"]["final"]["acc_y"])
