# -*- coding: utf-8 -*-
# EXPERIMENT S (chapter 74): natural-text retrieval piece (RAG form).
# Task = exp R infiller (ctx 12 + 4 masked slots) + injection prefix slot:
#   T=28 = [inj 12 | ctx 12 | 4 slots]. Disk = the 8MB train library itself
#   (8MB >> 200K-param capacity => injection supplies memorization-free
#   knowledge). Retrieval stub = 3-gram overlap k-NN over 24-char chunks
#   (step 64), leave-one-out (exclude chunks within 256 chars of query).
# Train mixes injection modes (50% top1 / 25% random / 25% none); eval reads
#   nll/acc/exact4 per mode. Delta(top1 - none) = the retrieval piece's value.
# Arms: fp teacher, ternary raw, ternary distilled (Q2 recipe).
import numpy as np, json, time, os, sys, glob
import os as _aim_os
AIM_SEED = int(_aim_os.environ.get("AIM_SEED", "1"))
AIM_TAG = _aim_os.environ.get("AIM_TAG", "")

R = lambda s: print(s, flush=True)
D, L, H1 = 128, 2, 256
TINJ, TCTX, K = 12, 12, 4
T = TINJ + TCTX + K
B, STEPS, LR, WD = 96, 5000, 2e-3, 1e-4
OUT = "blueprint_rag%s_results.json" % (("_" + AIM_TAG) if AIM_TAG else "")

paths = []
for pat in (r"H:\AimCode\build-tools\*.py", r"H:\AimCode\lab\**\*.md", r"H:\AimCode\lab\**\*.py",
            r"H:\AimCode\AimCode\**\*.py"):
    paths += glob.glob(pat, recursive=True)
paths += glob.glob(os.path.join(os.path.dirname(sys.executable), "Lib", "**", "*.py"), recursive=True)
paths = sorted(set(paths))

# ---------- corpus: procedural ASCII text (Kaggle-safe, no local paths) ----------
_words = ("the model learns to store facts in weights and the curve grows until the "
          "capacity wall appears then extra parameters add nothing new at all "
          "we measure bits per param with gzip compression on a fixed corpus "
          "training uses adam with cosine decay and early stop on validation loss "
          "ternary weights quantize to three levels and pay a small tax on loss "
          "masked prediction spans several tokens in parallel like diffusion ").split()
def _mk_corpus(n_chars, seed):
    r = np.random.default_rng(seed)
    parts = []
    tot = 0
    while tot < n_chars:
        ln = int(r.integers(6, 14))
        s = " ".join(r.choice(_words) for _ in range(ln))
        s = s[0].upper() + s[1:] + ". " + str(int(r.integers(0, 9999))) + "\n"
        parts.append(s); tot += len(s)
    return "".join(parts)
corpus = _mk_corpus(600000 if 'bitnet' in __file__ else 1000000, 0 + _OFF)
chars = sorted(set(corpus)); NV = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
ids = np.array([stoi[c] for c in corpus], dtype=np.int32)
split = int(len(ids) * 0.9)
V = NV + 1; MASK = NV
R("corpus %.2f MB | vocab %d" % (len(corpus) / 1e6, NV))

# ---- retrieval index: 24-char chunks, step 64, 3-gram -> chunk ids ----
CH, STEP = 24, 64
t0 = time.perf_counter()
starts = np.arange(0, split - CH, STEP)
index = {}
for ci, st in enumerate(starts):
    seg = ids[st:st + CH]
    for g in zip(seg[:-2], seg[1:-1], seg[2:]):
        index.setdefault(g, []).append(ci)
starts_list = starts.tolist()
R("index: %d chunks, %d trigram keys (%.0fs)" % (len(starts), len(index), time.perf_counter() - t0))

def retrieve(ctx_ids, qpos, rng, mode):
    """ctx_ids [12]; returns 12 injection ids (MASK-filled if none)."""
    inj = np.full(TINJ, MASK, dtype=np.int64)
    if mode == "none":
        return inj
    if mode == "random":
        ci = rng.integers(0, len(starts_list))
        return ids[starts_list[ci]:starts_list[ci] + TINJ].astype(np.int64)
    grams = set(zip(ctx_ids[:-2], ctx_ids[1:-1], ctx_ids[2:]))
    cand = {}
    for g in grams:
        for ci in index.get(g, ()):
            cand[ci] = cand.get(ci, 0) + 1
    best, best_s = -1, 0
    for ci, s in cand.items():
        if s <= best_s:
            continue
        st = starts_list[ci]
        if abs(st - qpos) < 256:      # leave-one-out
            continue
        best, best_s = ci, s
    if best < 0:
        return inj
    st = starts_list[best]
    return ids[st:st + TINJ].astype(np.int64)

def win_batch(bRng, lo, hi, n):
    s = bRng.integers(lo, hi - TCTX - K, size=n)
    return ids[s[:, None] + np.arange(TCTX + K)[None, :]]

def make_batch(bRng, lo, hi, n, mode_mix=True):
    s = bRng.integers(lo, hi - T - 16, size=n)
    win = ids[s[:, None] + np.arange(TCTX + K)[None, :]]   # [n,16]
    ctx, tgt = win[:, :TCTX], win[:, TCTX:]
    inj = np.zeros((n, TINJ), dtype=np.int64)
    for b in range(n):
        u = bRng.random()
        mode = "top1" if (u < 0.5 or not mode_mix) else ("random" if u < 0.75 else "none")
        inj[b] = retrieve(ctx[b], int(s[b]), bRng, mode)
    nmask = bRng.integers(1, K + 1, size=n)
    msk = np.zeros((n, K), dtype=bool)
    for b in range(n):
        msk[b, bRng.choice(K, size=nmask[b], replace=False)] = True
    tok = np.concatenate([inj, ctx, np.where(msk, MASK, tgt)], 1)
    return tok, tgt, msk

def make_params(seed):
    rng = np.random.default_rng(seed)
    sc = 0.8 / np.sqrt(D)
    p = {"E": rng.normal(0, sc, (V, D)), "P": rng.normal(0, sc, (T, D))}
    for l in range(L):
        for n in ("Wq", "Wk", "Wv", "Wo"):
            p["%s%d" % (n, l)] = rng.normal(0, sc, (D, D))
        p["W1%d" % l] = rng.normal(0, sc, (D, H1)); p["b1%d" % l] = np.zeros(H1)
        p["W2%d" % l] = rng.normal(0, sc, (H1, D)); p["b2%d" % l] = np.zeros(D)
    p["Wout"] = rng.normal(0, sc, (D, NV))
    return p

QKEYS = tuple("%s%d" % (n, l) for l in range(L) for n in ("Wq", "Wk", "Wv", "Wo", "W1", "W2"))
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

def loss_and_grads(p, tok, tgt, msk, teacher=None, soft_w=0.5):
    Bn = tok.shape[0]
    h, caches = fwd_caches(p, tok)
    dh = np.zeros_like(h)
    g = {k: np.zeros_like(x) for k, x in p.items()}
    tot = 0.0
    gW = np.zeros_like(p["Wout"])
    base = TINJ + TCTX
    for j in range(K):
        pos = base + j
        lg = h[:, pos, :] @ p["Wout"]
        lg = lg - lg.max(1, keepdims=True)
        Z = np.exp(lg); sm = Z / Z.sum(1, keepdims=True)
        m = msk[:, j]
        if m.any():
            ti = tgt[m, j]
            nll = float(-np.log(sm[np.arange(Bn)[m], ti] + 1e-12).mean())
            tot += nll
            dl = sm.copy()
            dl[np.arange(Bn)[m], ti] -= 1.0
            dl *= m[:, None] / Bn
            if teacher is not None:
                lg_t = teacher[1][:, pos, :] @ teacher[0]["Wout"]
                lg_t = lg_t - lg_t.max(1, keepdims=True)
                Zt = np.exp(lg_t); pt = Zt / Zt.sum(1, keepdims=True)
                tot += soft_w * float(-(pt * np.log(sm + 1e-12)).sum(1)[m].mean())
                dl = (1.0 - soft_w) * dl + soft_w * (sm - pt) * m[:, None] / Bn
            gW += h[:, pos, :].T @ dl
            dh[:, pos, :] += dl @ p["Wout"].T
    g["Wout"] = gW
    g1 = bwd(p, tok, caches, dh)
    for kk in g1: g[kk] = g[kk] + g1[kk]
    return tot, g

def eval_model(p, n=512):
    bRng = np.random.default_rng(999)
    out = {}
    for mode in ("top1", "random", "none"):
        nll = 0.0; acc = 0.0; ref4 = 0; ns = 0
        for s in range(0, n, 128):
            e = min(s + 128, n)
            w = win_batch(bRng, split, len(ids), e - s)
            Bn = e - s
            ctx, tgt = w[:, :TCTX], w[:, TCTX:]
            inj = np.zeros((Bn, TINJ), dtype=np.int64)
            for b in range(Bn):
                inj[b] = retrieve(ctx[b], -1, bRng, mode)
            msk = np.ones((Bn, K), dtype=bool)
            tok = np.concatenate([inj, ctx, np.full((Bn, K), MASK, dtype=np.int64)], 1)
            h = fwd_caches(p, tok)[0]
            rpreds = np.zeros((Bn, K), dtype=np.int64)
            for j in range(K):
                lg = h[:, TINJ + TCTX + j, :] @ p["Wout"]
                lg = lg - lg.max(1, keepdims=True)
                Z = np.exp(lg); sm = Z / Z.sum(1, keepdims=True)
                nll += float((-np.log(sm[np.arange(Bn), tgt[:, j]] + 1e-12)).sum())
                rpreds[:, j] = lg.argmax(1)
                acc += int((rpreds[:, j] == tgt[:, j]).sum())
            ref4 += int(((rpreds == tgt).all(1)).sum()); ns += Bn
        out[mode] = {"nll_bits": round(nll / (ns * K), 4), "acc": round(acc / (ns * K), 4),
                     "exact4": round(ref4 / ns, 4)}
    out["retrieval_gain_bits"] = round(out["none"]["nll_bits"] - out["top1"]["nll_bits"], 4)
    return out

def train(steps, seed, ternary=False, teacher=None, soft_w=0.5):
    p = make_params(seed)
    m = {k: np.zeros_like(x) for k, x in p.items()}
    v = {k: np.zeros_like(x) for k, x in p.items()}
    bRng = np.random.default_rng(7000 + seed)
    traj = []; t0 = time.perf_counter()
    for step in range(1, steps + 1):
        lr = LR * min(1.0, step / 300.0) * (0.1 + 0.45 * (1 + np.cos(np.pi * step / steps)))
        pq = ternize(p) if (ternary or teacher is not None) else p
        tok, tgt, msk = make_batch(bRng, 0, split - 16, B)
        tea = None
        if teacher is not None:
            th = fwd_caches(teacher, tok)[0]
            tea = (teacher, th)
        _, g = loss_and_grads(pq, tok, tgt, msk, teacher=tea, soft_w=soft_w)
        for kk in p:
            m[kk] = 0.9 * m[kk] + 0.1 * g[kk]
            v[kk] = 0.999 * v[kk] + 0.001 * g[kk] * g[kk]
            p[kk] -= lr * ((m[kk] / (1 - 0.9 ** step)) /
                           (np.sqrt(v[kk] / (1 - 0.999 ** step)) + 1e-8) + WD * p[kk])
        if step % 1000 == 0:
            ev = eval_model(pq if (ternary or teacher is not None) else p)
            traj.append([step, ev])
            R("  step %5d top1 %.3f none %.3f gain %.3f (%.0fs)" %
              (step, ev["top1"]["nll_bits"], ev["none"]["nll_bits"], ev["retrieval_gain_bits"],
               time.perf_counter() - t0))
    pq = ternize(p) if (ternary or teacher is not None) else p
    return pq, traj

if __name__ == "__main__":
    R("=== EXPERIMENT S: natural-text retrieval piece (RAG form) ===")
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if "teacher" not in results:
        R("--- teacher (fp) ---")
        tp, traj = train(STEPS, 42 + (AIM_SEED - 1))
        results["teacher"] = dict(final=eval_model(tp, n=1024), traj=traj)
        np.savez("rag_teacher%s.npz" % (("_" + AIM_TAG) if AIM_TAG else ""), **tp)
        json.dump(results, open(OUT, "w"), indent=1)
        R("=== teacher FINAL %s ===" % results["teacher"]["final"])
    _z = np.load("rag_teacher%s.npz" % (("_" + AIM_TAG) if AIM_TAG else ""))
    tp = {kk: _z[kk] for kk in _z.files}
    if "tern_distill" not in results:
        R("--- tern_distill ---")
        sp, traj = train(STEPS, 43 + (AIM_SEED - 1), ternary=True, teacher=tp)
        results["tern_distill"] = dict(final=eval_model(sp, n=1024), traj=traj)
        json.dump(results, open(OUT, "w"), indent=1)
        R("=== tern_distill FINAL %s ===" % results["tern_distill"]["final"])
    R("=== SUMMARY ===")
    for k in ("teacher", "tern_distill"):
        R("%s: %s" % (k, results[k]["final"]))
