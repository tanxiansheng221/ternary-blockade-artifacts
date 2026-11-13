# -*- coding: utf-8 -*-
# EXPERIMENT Q (chapter 72): blueprint micro-assembly v0.
# Theory under test: two-term capacity law — facts live on disk, structure in
# weights, IN ONE MODEL. Task: y = (V[k] + c) % 97 with V = 65536 random kv
# pairs (433K bits => 2.2 bits/param demanded if forced into 200K-param weights
# > 1-2 bits/param fact density => weight-only storage mathematically impossible).
# Pieces assembled: ternary core (STE, verified ch60-64) + retrieval stub
# (exact hash = disk) + supervision-target prior (aux head, verified ch71) +
# BLAS attention. Diffusion decode / distillation = next increments.
# Arms:
#   B_fp     : retrieval + aux head, fp weights        -> expect ~100%
#   B_tern   : same, ternary matmul weights            -> assembly tax
#   A_fp     : NO retrieval (PAD context), same aux    -> facts-in-weights ceiling
# Key anti-shortcut probe: acc_y with context blanked (nofact) — if high, the
# "retrieval" arm secretly memorized the table in weights.
import numpy as np, json, time, os, sys

R = lambda s: print(s, flush=True)
NV, D, L, H1, T = 97, 96, 2, 256, 10
V = NV + 1; PAD = 97
NTAB = 65536
B, STEPS, LR, WD = 128, 8000, 2e-3, 1e-4
import os as _aim_os
AIM_SEED = int(_aim_os.environ.get("AIM_SEED", "1"))
AIM_TAG = _aim_os.environ.get("AIM_TAG", "")
_OFF = (AIM_SEED - 1) * 2
OUT = "blueprint_assembly%s_results.json" % (("_" + AIM_TAG) if AIM_TAG else "")

# ---- disk (fact table): 65536 random 3-digit keys -> values ----
dRng = np.random.default_rng(7)
perm = dRng.permutation(NV**3)[:NTAB]
KD = np.stack([perm // (NV*NV), (perm // NV) % NV, perm % NV], 1)  # [NTAB,3]
KV = dRng.integers(0, NV, size=NTAB)                                # [NTAB]
R("disk: %d kv pairs = %.0fK bits (need %.2f bits/param in 200K weights)"
  % (NTAB, NTAB * 6.6 / 1000, NTAB * 6.6 / 200000))

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

QKEYS = ("Wq0", "Wk0", "Wv0", "Wo0", "W10", "W20", "Wq1", "Wk1", "Wv1", "Wo1", "W11", "W21")
def ternize(p):
    q = {}
    for k, v in p.items():
        if k in QKEYS:
            gscale = np.abs(v).mean()
            q[k] = np.clip(np.round(v / gscale), -1, 1) * gscale
        else:
            q[k] = v
    return q

def fwd(p, tok):
    h = p["E"][tok] + p["P"][None, :, :]
    for l in range(L):
        hc = h
        q = hc @ p["Wq%d" % l]; k = hc @ p["Wk%d" % l]; v = hc @ p["Wv%d" % l]
        s = (q @ k.transpose(0, 2, 1)) / np.sqrt(D)
        s = s - s.max(-1, keepdims=True)
        e = np.exp(s); A = e / e.sum(-1, keepdims=True)
        ha = hc + (A @ v) @ p["Wo%d" % l]
        z = ha @ p["W1%d" % l] + p["b1%d" % l]
        h = ha + np.maximum(z, 0.0) @ p["W2%d" % l] + p["b2%d" % l]
    return h

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

def head_grad(h, W, pos, tgt, dh, g, Bn):
    lg = h[:, pos, :] @ W
    lg = lg - lg.max(1, keepdims=True)
    Z = np.exp(lg); sm = Z / Z.sum(1, keepdims=True)
    nll = float(-np.log(sm[np.arange(Bn), tgt] + 1e-12).mean())
    dl = sm.copy(); dl[np.arange(Bn), tgt] -= 1.0; dl /= Bn
    g[W] = g[W] + h[:, pos, :].T @ dl  # W is str name; handled below
    return nll, dl

def loss_and_grads(p, tok, v_t, y_t, use_aux):
    Bn = tok.shape[0]
    h, caches = fwd_caches(p, tok)
    dh = np.zeros_like(h); tot = 0.0
    g = {k: np.zeros_like(v) for k, v in p.items()}
    heads = [(3, "Wh", v_t)] if use_aux else []
    heads.append((9, "Wy", y_t))
    for pos, name, tgt in heads:
        lg = h[:, pos, :] @ p[name]
        lg = lg - lg.max(1, keepdims=True)
        Z = np.exp(lg); sm = Z / Z.sum(1, keepdims=True)
        tot += float(-np.log(sm[np.arange(Bn), tgt] + 1e-12).mean())
        dl = sm.copy(); dl[np.arange(Bn), tgt] -= 1.0; dl /= Bn
        g[name] = h[:, pos, :].T @ dl
        dh[:, pos, :] += dl @ p[name].T
    g1 = bwd(p, tok, caches, dh)
    for kk in g1: g[kk] = g[kk] + g1[kk]
    return tot, g

def make_batch(bRng, n, retrieval):
    idx = bRng.integers(0, NTAB, size=n)
    kd = KD[idx]; vv = KV[idx]
    c = bRng.integers(0, NV, size=n)
    y = (vv + c) % NV
    tok = np.full((n, T), PAD, dtype=np.int64)
    if retrieval:
        tok[:, 0:3] = kd; tok[:, 3] = vv
    tok[:, 5:8] = kd; tok[:, 8] = c
    return tok, vv, y

def evaluate(p, retrieval, n=4096, chunk=1024):
    bRng = np.random.default_rng(999)
    out = {}
    for probe, ret in (("fact", retrieval), ("nofact", False)):
        ay = 0.0; av = 0.0; ns = 0
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            tok, vv, y = make_batch(bRng, e - s, ret)
            h = fwd(p, tok)
            ay += float((h[:, 9, :] @ p["Wy"]).argmax(1).__eq__(y).mean()) * (e - s)
            av += float((h[:, 3, :] @ p["Wh"]).argmax(1).__eq__(vv).mean()) * (e - s)
            ns += e - s
        out["acc_y_" + probe] = round(ay / ns, 4)
        out["acc_v_" + probe] = round(av / ns, 4)
    return out

def run(arm, ternary, retrieval, steps=STEPS, seed=42):
    p = make_params(seed)
    m = {k: np.zeros_like(x) for k, x in p.items()}
    v = {k: np.zeros_like(x) for k, x in p.items()}
    bRng = np.random.default_rng(3000 + seed)
    traj = []; t0 = time.perf_counter()
    for step in range(1, steps + 1):
        lr = LR * min(1.0, step / 300.0) * (0.1 + 0.45 * (1 + np.cos(np.pi * step / steps)))
        pq = ternize(p) if ternary else p
        tok, vv, y = make_batch(bRng, B, retrieval)
        _, g = loss_and_grads(pq, tok, vv, y, use_aux=retrieval)
        for kk in p:
            m[kk] = 0.9 * m[kk] + 0.1 * g[kk]
            v[kk] = 0.999 * v[kk] + 0.001 * g[kk] * g[kk]
            p[kk] -= lr * ((m[kk] / (1 - 0.9 ** step)) /
                           (np.sqrt(v[kk] / (1 - 0.999 ** step)) + 1e-8) + WD * p[kk])
        if step % 1000 == 0:
            ev = evaluate(pq, retrieval, n=1024)
            traj.append([step, ev])
            R("  [%s] step %5d %s (%.0fs)" % (arm, step, ev, time.perf_counter() - t0))
    pq = ternize(p) if ternary else p
    final = evaluate(pq, retrieval)
    return dict(final=final, traj=traj)

if __name__ == "__main__":
    R("=== EXPERIMENT Q: blueprint micro-assembly (facts on disk, structure in weights) ===")
    npar = sum(x.size for x in make_params(0).values())
    R("params %d, disk %d pairs" % (npar, NTAB))
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    arms = [("B_fp", False, True, STEPS), ("B_tern", True, True, STEPS),
            ("A_fp", False, False, STEPS), ("B_tern16", True, True, 16000)]
    for arm, ternary, retrieval, st in arms:
        if arm in results:
            R("--- %s already done, skip ---" % arm)
            continue
        R("--- %s (steps=%d) ---" % (arm, st))
        results[arm] = run(arm, ternary, retrieval, steps=st, seed=42 + _OFF)
        json.dump(results, open(OUT, "w"), indent=1)
        R("=== %s FINAL %s ===" % (arm, results[arm]["final"]))
    R("=== SUMMARY ===")
    R("PQ1 retrieval works:      B_fp acc_y_fact %.3f" % results["B_fp"]["final"]["acc_y_fact"])
    R("PQ2 disk>weights:         A_fp acc_y_fact %.3f vs B_fp %.3f" %
      (results["A_fp"]["final"]["acc_y_fact"], results["B_fp"]["final"]["acc_y_fact"]))
    R("PQ3 assembly ternary tax: B_tern %.3f vs B_fp %.3f ; B_tern16 %.3f" %
      (results["B_tern"]["final"]["acc_y_fact"], results["B_fp"]["final"]["acc_y_fact"],
       results["B_tern16"]["final"]["acc_y_fact"]))
    R("anti-shortcut (nofact):   B_fp %.3f B_tern %.3f B_tern16 %.3f" %
      (results["B_fp"]["final"]["acc_y_nofact"], results["B_tern"]["final"]["acc_y_nofact"],
       results["B_tern16"]["final"]["acc_y_nofact"]))
