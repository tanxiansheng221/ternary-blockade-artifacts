# -*- coding: utf-8 -*-
# EXPERIMENT Q4 (chapter 72 addendum 3): does temperature expose dark knowledge
# in a PERFECT teacher? Q3 found distillation fails on linear-ternary composite
# when the teacher is 100% (soft target ~ one-hot, no dark knowledge). Test:
# same teacher, T=4 temperature softening. Preregistered: PQ5 student acc_y
# @8K > 30% (T=1 baseline 1.0%) => hypothesis confirmed.
import numpy as np, json, time, os, sys

R = lambda s: print(s, flush=True)
NV, D, L, H1, T = 97, 96, 2, 256, 10
V = NV + 1; PAD = 97
NTAB = 65536
B, LR, WD = 128, 2e-3, 1e-4
EPS = 1e-6
TEMPS = (1.0, 4.0)
import os as _aim_os
AIM_SEED = int(_aim_os.environ.get("AIM_SEED", "1"))
AIM_TAG = _aim_os.environ.get("AIM_TAG", "")
_OFF = (AIM_SEED - 1) * 2
OUT = "blueprint_tempdist%s_results.json" % (("_" + AIM_TAG) if AIM_TAG else "")
TCKPT = "linear_teacher_8k%s.npz" % (("_" + AIM_TAG) if AIM_TAG else "")

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

def elu1(x):
    return np.where(x > 0, x, np.exp(np.minimum(x, 20.0)))

def elu1_grad(x):
    return np.where(x > 0, 1.0, np.exp(np.minimum(x, 20.0)))

def lin_fwd_caches(p, tok):
    h = p["E"][tok] + p["P"][None, :, :]
    caches = []
    for l in range(L):
        hc = h
        q = hc @ p["Wq%d" % l]; k = hc @ p["Wk%d" % l]; v = hc @ p["Wv%d" % l]
        pq_ = elu1(q); pk = elu1(k)
        Skv = np.einsum("btd,bte->bde", pk, v)
        Sk = pk.sum(1)
        num = np.einsum("btd,bde->bte", pq_, Skv)
        den = np.einsum("btd,bd->bt", pq_, Sk) + EPS
        o = num / den[:, :, None]
        ha = hc + o @ p["Wo%d" % l]
        z = ha @ p["W1%d" % l] + p["b1%d" % l]
        r = np.maximum(z, 0.0)
        h = ha + r @ p["W2%d" % l] + p["b2%d" % l]
        caches.append((hc, q, k, v, pq_, pk, Skv, Sk, num, den, o, ha, z, r))
    return h, caches

def lin_bwd(p, tok, caches, dh):
    g = {"E": np.zeros_like(p["E"]), "P": np.zeros_like(p["P"])}
    for l in reversed(range(L)):
        hc, q, k, v, pq_, pk, Skv, Sk, num, den, o, ha, z, r = caches[l]
        dr = dh @ p["W2%d" % l].T
        dz = dr * (z > 0)
        dha = dh + dz @ p["W1%d" % l].T
        g["W2%d" % l] = r.reshape(-1, H1).T @ dh.reshape(-1, D)
        g["b2%d" % l] = dh.reshape(-1, D).sum(0)
        g["W1%d" % l] = ha.reshape(-1, D).T @ dz.reshape(-1, H1)
        g["b1%d" % l] = dz.reshape(-1, H1).sum(0)
        do = dha @ p["Wo%d" % l].T
        g["Wo%d" % l] = o.reshape(-1, D).T @ dha.reshape(-1, D)
        dnum = do / den[:, :, None]
        dden = -(do * o).sum(-1) / den
        dphiq = np.einsum("bde,bte->btd", Skv, dnum) + dden[:, :, None] * Sk[:, None, :]
        dSkv = np.einsum("btd,bte->bde", pq_, dnum)
        dSk = np.einsum("bt,btd->bd", dden, pq_)
        dphik = np.einsum("bde,bte->btd", dSkv, v) + dSk[:, None, :]
        dv = np.einsum("btd,bde->bte", pk, dSkv)
        dq = dphiq * elu1_grad(q)
        dk = dphik * elu1_grad(k)
        g["Wq%d" % l] = hc.reshape(-1, D).T @ dq.reshape(-1, D)
        g["Wk%d" % l] = hc.reshape(-1, D).T @ dk.reshape(-1, D)
        g["Wv%d" % l] = hc.reshape(-1, D).T @ dv.reshape(-1, D)
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

def evaluate(p, n=2048, chunk=512):
    bRng = np.random.default_rng(999)
    ay = 0.0; av = 0.0; ns = 0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        tok, vv, y = make_batch(bRng, e - s)
        h = lin_fwd_caches(p, tok)[0]
        ay += float((h[:, 9, :] @ p["Wy"]).argmax(1).__eq__(y).mean()) * (e - s)
        av += float((h[:, 3, :] @ p["Wh"]).argmax(1).__eq__(vv).mean()) * (e - s)
        ns += e - s
    return {"acc_y": round(ay / ns, 4), "acc_v": round(av / ns, 4)}

def get_teacher():
    if os.path.exists(TCKPT):
        z = np.load(TCKPT)
        p = {kk: z[kk] for kk in z.files}
        R("teacher loaded from ckpt")
        return p
    p = make_params(42 + _OFF)
    m = {k: np.zeros_like(x) for k, x in p.items()}
    v = {k: np.zeros_like(x) for k, x in p.items()}
    bRng = np.random.default_rng(3042 + _OFF)
    for step in range(1, 8001):
        lr = LR * min(1.0, step / 300.0) * (0.1 + 0.45 * (1 + np.cos(np.pi * step / 8000)))
        tok, vv, y = make_batch(bRng, B)
        h, caches = lin_fwd_caches(p, tok)
        dh = np.zeros_like(h)
        g = {k: np.zeros_like(x) for k, x in p.items()}
        nll, dl, _ = softmax_xent(h[:, 3, :] @ p["Wh"], vv, B)
        g["Wh"] = h[:, 3, :].T @ dl
        dh[:, 3, :] += dl @ p["Wh"].T
        nll, dl, _ = softmax_xent(h[:, 9, :] @ p["Wy"], y, B)
        g["Wy"] = h[:, 9, :].T @ dl
        dh[:, 9, :] += dl @ p["Wy"].T
        g1 = lin_bwd(p, tok, caches, dh)
        for kk in g1: g[kk] = g[kk] + g1[kk]
        for kk in p:
            m[kk] = 0.9 * m[kk] + 0.1 * g[kk]
            v[kk] = 0.999 * v[kk] + 0.001 * g[kk] * g[kk]
            p[kk] -= lr * ((m[kk] / (1 - 0.9 ** step)) /
                           (np.sqrt(v[kk] / (1 - 0.999 ** step)) + 1e-8) + WD * p[kk])
        if step % 2000 == 0:
            R("  [teacher] step %5d %s" % (step, evaluate(p)))
    np.savez(TCKPT, **p)
    R("teacher ckpt saved")
    return p

def student(Ttemp, steps=8000, seed=44, soft_w=0.5):
    tp = get_teacher()
    p = make_params(seed)
    m = {k: np.zeros_like(x) for k, x in p.items()}
    v = {k: np.zeros_like(x) for k, x in p.items()}
    bRng = np.random.default_rng(6000 + seed)
    traj = []; ent_sum = 0.0; ent_n = 0; t0 = time.perf_counter()
    for step in range(1, steps + 1):
        lr = LR * min(1.0, step / 300.0) * (0.1 + 0.45 * (1 + np.cos(np.pi * step / steps)))
        pq = ternize(p)
        tok, vv, y = make_batch(bRng, B)
        h, caches = lin_fwd_caches(pq, tok)
        dh = np.zeros_like(h)
        g = {k: np.zeros_like(x) for k, x in p.items()}
        nll, dl, _ = softmax_xent(h[:, 3, :] @ pq["Wh"], vv, B)
        g["Wh"] = h[:, 3, :].T @ dl
        dh[:, 3, :] += dl @ pq["Wh"].T
        lg_y = h[:, 9, :] @ pq["Wy"]
        nll, dl, sm = softmax_xent(lg_y, y, B)
        ht = lin_fwd_caches(tp, tok)[0]
        lg_t = (ht[:, 9, :] @ tp["Wy"]) / Ttemp
        lg_t = lg_t - lg_t.max(1, keepdims=True)
        Zt = np.exp(lg_t); pt = Zt / Zt.sum(1, keepdims=True)
        if step % 500 == 0:
            pe = -(pt * np.log(pt + 1e-12)).sum(1) / np.log(2)
            ent_sum += float(pe.mean()); ent_n += 1
        smT = np.exp(lg_y / Ttemp - (lg_y / Ttemp).max(1, keepdims=True))
        smT = smT / smT.sum(1, keepdims=True)
        dl = (1.0 - soft_w) * dl + soft_w * (smT - pt) / (Ttemp * B)
        g["Wy"] = h[:, 9, :].T @ dl
        dh[:, 9, :] += dl @ pq["Wy"].T
        g1 = lin_bwd(pq, tok, caches, dh)
        for kk in g1: g[kk] = g[kk] + g1[kk]
        for kk in p:
            m[kk] = 0.9 * m[kk] + 0.1 * g[kk]
            v[kk] = 0.999 * v[kk] + 0.001 * g[kk] * g[kk]
            p[kk] -= lr * ((m[kk] / (1 - 0.9 ** step)) /
                           (np.sqrt(v[kk] / (1 - 0.999 ** step)) + 1e-8) + WD * p[kk])
        if step % 1000 == 0:
            ev = evaluate(pq)
            traj.append([step, ev])
            R("  [student T=%.1f] step %5d %s (%.0fs)" % (Ttemp, step, ev, time.perf_counter() - t0))
    return dict(final=evaluate(pq, n=4096), traj=traj,
                teacher_target_entropy_bits=round(ent_sum / max(1, ent_n), 4))

if __name__ == "__main__":
    R("=== EXPERIMENT Q4: temperature distillation on a PERFECT teacher ===")
    tp = get_teacher()
    R("teacher eval: %s" % evaluate(tp, n=4096))
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for Ttemp in TEMPS:
        key = "student_T%.1f" % Ttemp
        if key in results:
            R("--- %s already done, skip ---" % key)
            continue
        R("--- student T=%.1f ---" % Ttemp)
        results[key] = student(Ttemp, seed=44 + _OFF)
        json.dump(results, open(OUT, "w"), indent=1)
        R("=== %s FINAL %s ===" % (key, results[key]["final"]))
    R("=== SUMMARY ===")
    for Ttemp in TEMPS:
        key = "student_T%.1f" % Ttemp
        r = results[key]
        R("T=%.1f: acc_y %.3f | teacher-target entropy %.3f bits" %
          (Ttemp, r["final"]["acc_y"], r["teacher_target_entropy_bits"]))
