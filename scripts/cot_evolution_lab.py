# -*- coding: utf-8 -*-
# EXPERIMENT P (chapter 71): chain-of-thought vs evolution-of-thought, tiny transformer.
# Task: 2-step chained mod-97 addition: h1=(a+b)%97, y=(h1+c)%97.
# Regimes (identical architecture/seed/data order, only the loss differs):
#   DIRECT : loss on final y only                        (one-pass 3-ary readout)
#   COT    : aux loss on intermediate h1 (+ y)           (scratchpad supervision)
#   EVOLVE : COT + round-2 with TRUE h1 appended as token (trains the token readout)
# Eval (EVOLVE): y2 with SELF-fed argmax h1 vs ORACLE true h1 -> error-propagation tax.
# Preregistered predictions:
#   P1: serial token (y2 self) > one-pass (y1)          -- CoT core claim (test-time depth)
#   P2: y2 oracle > y2 self                             -- error propagation tax
#   P3: COT y1 vs DIRECT y1                             -- aux supervision transfer (measure)
# Grad check (float64 finite diff) must pass before training (house discipline).
import numpy as np, json, time, os, sys

R = lambda s: print(s, flush=True)
V, NV, D, L, H1, T = 98, 97, 96, 2, 256, 5
B, STEPS, LR, WD = 128, 12000, 2e-3, 1e-4
PAD = 97
LOG_EVERY = 1000
import os as _aim_os
AIM_SEED = int(_aim_os.environ.get("AIM_SEED", "1"))
AIM_TAG = _aim_os.environ.get("AIM_TAG", "")
_OFF = (AIM_SEED - 1) * 2
ROOT = os.getcwd()
OUT = os.path.join(ROOT, "cot_evolution%s_results.json" % (("_" + AIM_TAG) if AIM_TAG else ""))

# ---------------- data ----------------
dRng = np.random.default_rng(7)
perm = dRng.permutation(NV**3)
tr_idx = perm[:20000]; va_idx = perm[20000:24096]
def unflatten(idx):
    a = idx // (NV*NV); b = (idx // NV) % NV; c = idx % NV
    return a, b, c
Atr, Btr_, Ctr = unflatten(tr_idx); Ava, Bva, Cva = unflatten(va_idx)
H1tr = (Atr + Btr_) % NV; Ytr = (H1tr + Ctr) % NV
H1va = (Ava + Bva) % NV;  Yva = (H1va + Cva) % NV
R("data: train %d val %d (disjoint triples, 97^3 space)" % (len(tr_idx), len(va_idx)))

# ---------------- model ----------------
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

def fwd(p, tok):
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
        # MLP: h = ha + relu(z)@W2 + b2 ; z = ha@W1 + b1
        dr = dh @ p["W2%d" % l].T
        dz = dr * (z > 0)
        dha = dh + dz @ p["W1%d" % l].T
        g["W2%d" % l] = r.reshape(-1, H1).T @ dh.reshape(-1, D)
        g["b2%d" % l] = dh.reshape(-1, D).sum(0)
        g["W1%d" % l] = ha.reshape(-1, D).T @ dz.reshape(-1, H1)
        g["b1%d" % l] = dz.reshape(-1, H1).sum(0)
        # attn: ha = hc + (A@v)@Wo
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

def ce_head(h, W, pos, tgt, w):
    lg = h[:, pos, :] @ W
    lg = lg - lg.max(1, keepdims=True)
    Z = np.exp(lg); sm = Z / Z.sum(1, keepdims=True)
    nll = float(-np.log(sm[np.arange(len(tgt)), tgt] + 1e-12).mean())
    dl = sm.copy(); dl[np.arange(len(tgt)), tgt] -= 1.0
    dl *= w / len(tgt)
    return nll, dl

def loss_and_grads(p, x, h1, y, mode):
    """x [B,3] ints. Returns (total_weighted_loss, grads, dict of raw CE values)."""
    Bn = x.shape[0]
    tok1 = np.full((Bn, T), PAD, dtype=np.int64); tok1[:, :3] = x
    tot = 0.0; g = {k: np.zeros_like(v) for k, v in p.items()}; ces = {}
    h, caches = fwd(p, tok1)
    dh = np.zeros_like(h)
    heads = []
    if mode in ("cot", "evolve"):
        heads.append((3, "Wh", h1, 1.0, "h1"))
    heads.append((4, "Wy", y, 1.0, "y1"))
    for pos, name, tgt, w, key in heads:
        nll, dl = ce_head(h, p[name], pos, tgt, w)
        tot += w * nll; ces[key] = nll
        g[name] = g[name] + h[:, pos, :].T @ dl
        dh[:, pos, :] += dl @ p[name].T
    g1 = bwd(p, tok1, caches, dh)
    for kk in g1: g[kk] = g[kk] + g1[kk]
    if mode == "evolve":
        tok2 = np.full((Bn, T), PAD, dtype=np.int64)
        tok2[:, :3] = x; tok2[:, 3] = h1
        h2, caches2 = fwd(p, tok2)
        dh2 = np.zeros_like(h2)
        nll2, dl2 = ce_head(h2, p["Wy"], 4, y, 1.0)
        tot += nll2; ces["y2_train"] = nll2
        g["Wy"] = g["Wy"] + h2[:, 4, :].T @ dl2
        dh2[:, 4, :] += dl2 @ p["Wy"].T
        g2 = bwd(p, tok2, caches2, dh2)
        for kk in g2: g[kk] = g[kk] + g2[kk]
    return tot, g, ces

# ---------------- grad check ----------------
def gradcheck():
    rng = np.random.default_rng(0)
    p = make_params(0)
    for kk in p: p[kk] = p[kk].astype(np.float64)
    x = rng.integers(0, NV, size=(3, 3))
    h1 = (x[:, 0] + x[:, 1]) % NV; y = (h1 + x[:, 2]) % NV
    t0, g, _ = loss_and_grads(p, x, h1, y, "evolve")
    worst = 0.0
    for kk, arr in p.items():
        flat = arr.reshape(-1)
        for _ in range(2):
            i = rng.integers(0, flat.size)
            eps = 1e-6
            old = flat[i]
            flat[i] = old + eps; lp, _, _ = loss_and_grads(p, x, h1, y, "evolve")
            flat[i] = old - eps; lm, _, _ = loss_and_grads(p, x, h1, y, "evolve")
            flat[i] = old
            num = (lp - lm) / (2 * eps); ana = g[kk].reshape(-1)[i]
            absd = abs(num - ana)
            err = absd / max(1e-8, abs(num) + abs(ana))
            # noise floor: tiny gradients hit finite-diff precision (~1e-10 abs);
            # a real indexing bug produces abs errors orders of magnitude larger.
            ok = (absd < 1e-6) or (err < 1e-3)
            worst = max(worst, 0.0 if ok else err)
            if not ok:
                R("  gradcheck %s[%d]: num %.8f ana %.8f abs %.2e rel %.2e" % (kk, i, num, ana, absd, err))
    R("grad check %s (worst %.2e)" % ("OK" if worst < 1e-5 else "FAILED", worst))
    return worst < 1e-5

# ---------------- train / eval ----------------
def adam_train(mode, steps=STEPS, seed=42):
    p = make_params(seed)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    bRng = np.random.default_rng(1000 + {"direct": 0, "cot": 1, "evolve": 2}.get(mode, 0))
    traj = []; t0 = time.perf_counter()
    for step in range(1, steps + 1):
        lr = LR * min(1.0, step / 300.0) * (0.1 + 0.45 * (1 + np.cos(np.pi * step / steps)))
        bi = bRng.integers(0, len(tr_idx), size=B)
        x = np.stack([Atr[bi], Btr_[bi], Ctr[bi]], 1)
        _, g, ces = loss_and_grads(p, x, H1tr[bi], Ytr[bi], mode)
        for kk in p:
            m[kk] = 0.9 * m[kk] + 0.1 * g[kk]
            v[kk] = 0.999 * v[kk] + 0.001 * g[kk] * g[kk]
            p[kk] -= lr * ((m[kk] / (1 - 0.9 ** step)) /
                           (np.sqrt(v[kk] / (1 - 0.999 ** step)) + 1e-8) + WD * p[kk])
        if step % LOG_EVERY == 0 or step == 1:
            ev = eval_model(p, mode, n=2048)
            traj.append([step, ev])
            R("  [%s] step %5d trainCE %.3f %s (%.0fs)" % (mode, step,
              sum(ces.values()) / len(ces),
              " ".join("%s %.3f" % (k, vv) for k, vv in ev.items()), time.perf_counter() - t0))
    return p, traj

def eval_model(p, mode, n=4096, chunk=1024):
    ns = min(n, len(va_idx))
    accs = {k: 0 for k in ["h1", "y1", "y2_self", "y2_oracle"]}
    nlls = {k: 0.0 for k in ["h1", "y1", "y2_self", "y2_oracle"]}
    prop = {"n_h1bad": 0, "self_ok_h1bad": 0, "oracle_ok_h1bad": 0}
    for s in range(0, ns, chunk):
        e = min(s + chunk, ns)
        x = np.stack([Ava[s:e], Bva[s:e], Cva[s:e]], 1)
        h1t, yt = H1va[s:e], Yva[s:e]
        Bn = x.shape[0]; idx = np.arange(Bn)
        tok1 = np.full((Bn, T), PAD, dtype=np.int64); tok1[:, :3] = x
        h, _ = fwd(p, tok1)
        if mode in ("cot", "evolve"):
            lg = h[:, 3, :] @ p["Wh"]
            m = lg.max(1, keepdims=True)
            lsm = lg - m - np.log(np.exp(lg - m).sum(1, keepdims=True))
            h1p = lg.argmax(1)
            nlls["h1"] += float(-lsm[idx, h1t].sum())
            accs["h1"] += int((h1p == h1t).sum())
        lgy = h[:, 4, :] @ p["Wy"]
        m = lgy.max(1, keepdims=True)
        lsm = lgy - m - np.log(np.exp(lgy - m).sum(1, keepdims=True))
        y1p = lgy.argmax(1)
        nlls["y1"] += float(-lsm[idx, yt].sum())
        accs["y1"] += int((y1p == yt).sum())
        if mode == "evolve":
            h1ok = h1p == h1t
            for fed, key in ((h1p, "y2_self"), (h1t, "y2_oracle")):
                tok2 = np.full((Bn, T), PAD, dtype=np.int64)
                tok2[:, :3] = x; tok2[:, 3] = fed
                h2, _ = fwd(p, tok2)
                lg2 = h2[:, 4, :] @ p["Wy"]
                m2 = lg2.max(1, keepdims=True)
                lsm2 = lg2 - m2 - np.log(np.exp(lg2 - m2).sum(1, keepdims=True))
                y2p = lg2.argmax(1)
                nlls[key] += float(-lsm2[idx, yt].sum())
                accs[key] += int((y2p == yt).sum())
            nbad = int((~h1ok).sum())
            prop["n_h1bad"] += nbad
            prop["self_ok_h1bad"] += int(((y2p == yt) & ~h1ok).sum())
    out = {}
    for k in ["h1", "y1", "y2_self", "y2_oracle"]:
        if mode == "direct" and k != "y1":
            continue
        if mode == "cot" and k.startswith("y2"):
            continue
        out["acc_" + k] = round(accs[k] / ns, 4)
        out["nll_" + k] = round(nlls[k] / ns / np.log(2), 4)
    if mode == "evolve":
        out["recovery_h1bad_self"] = round(prop["self_ok_h1bad"] / max(1, prop["n_h1bad"]), 4)
    return out

# ---------------- main ----------------
if __name__ == "__main__":
    R("=== EXPERIMENT P: CoT vs Evolution-of-Thought (2-step mod-97 chain) ===")
    R("arch: emb+2x(transformer block) D=%d L=%d H1=%d params~%d" %
      (D, L, H1, sum(x.size for x in make_params(0).values())))
    w = gradcheck()
    if not w:
        R("GRAD CHECK FAILED - ABORT"); sys.exit(2)
    results = {"arch": dict(V=V, D=D, L=L, H1=H1, T=T, steps=STEPS, B=B, lr=LR, wd=WD),
               "task": "h1=(a+b)%97, y=(h1+c)%97, train 20K / val 4096 disjoint"}
    for mode in ("direct", "cot", "evolve"):
        R("--- training %s ---" % mode)
        p, traj = adam_train(mode, seed=42 + _OFF)
        final = eval_model(p, mode)
        results[mode] = dict(final=final, traj=traj)
        R("=== %s FINAL: %s ===" % (mode, final))
    json.dump(results, open(OUT, "w"), indent=1)
    R("saved %s" % OUT)
    R("=== SUMMARY (preregistered checks) ===")
    d, c, e = results["direct"]["final"], results["cot"]["final"], results["evolve"]["final"]
    R("P3 aux-supervision transfer: y1 direct %.3f vs y1 cot %.3f" % (d["acc_y1"], c["acc_y1"]))
    R("P1 CoT core: y2_self %.3f vs y1(evol) %.3f vs y1(cot) %.3f" % (e["acc_y2_self"], e["acc_y1"], c["acc_y1"]))
    R("P2 error tax: y2_oracle %.3f vs y2_self %.3f ; h1 acc %.3f" %
      (e["acc_y2_oracle"], e["acc_y2_self"], e["acc_h1"]))
