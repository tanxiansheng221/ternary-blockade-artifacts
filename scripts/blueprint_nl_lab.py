# -*- coding: utf-8 -*-
import os as _aim_os
AIM_SEED = int(_aim_os.environ.get("AIM_SEED", "1"))
AIM_TAG = _aim_os.environ.get("AIM_TAG", "")
_OFF = (AIM_SEED - 1) * 2

# EXPERIMENT R (chapter 73): natural-language-domain micro-assembly v0.
# Pieces assembled on 8MB real text (same corpus pipeline as exp O):
#   ternary core (STE) + masked-parallel multi-slot target (K=4 next chars,
#   ch60) + test-time iterative decode (MaskGIT confidence reveal, ch62) +
#   distillation (fp teacher -> ternary student, Q2 recipe).
# Window T=16: context = first 12 chars, 4 target slots at the end. Training
# masks a random subset of the 4 slots (model = conditional infiller). Eval:
# refine loop reveals most-confident slot per round; metrics: NLL bits/char,
# slot acc, exact-4 rate (one-shot vs refine).
# Arms: teacher (fp) / tern_raw / tern_distill.
import numpy as np, json, time, os, sys, glob

R = lambda s: print(s, flush=True)
D, L, H1 = 128, 2, 256
TCTX, K = 12, 4
T = TCTX + K
B, STEPS, LR, WD = 96, 5000, 2e-3, 1e-4
ROOT = r"H:\AimCode\build-tools"
OUT = ("blueprint_nl_results%s.json" % (("_" + AIM_TAG) if AIM_TAG else ""))

# ---- corpus (same pipeline as bitsparam_overnight_lab) ----
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
R("corpus %.2f MB | vocab %d | split %d/%d" % (len(corpus) / 1e6, NV, split, len(ids) - split))

def win_batch(bRng, lo, hi, n):
    s = bRng.integers(lo, hi - T, size=n)
    return ids[s[:, None] + np.arange(T)[None, :]]  # [n,T]

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
    """tok [B,T] with MASK at masked slots; tgt [B,K] true chars; msk [B,K] bool."""
    Bn = tok.shape[0]
    h, caches = fwd_caches(p, tok)
    dh = np.zeros_like(h)
    g = {k: np.zeros_like(x) for k, x in p.items()}
    tot = 0.0
    gW = np.zeros_like(p["Wout"])
    for j in range(K):
        pos = TCTX + j
        lg = h[:, pos, :] @ p["Wout"]                       # [B,NV]
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

def gradcheck():
    rng = np.random.default_rng(0 + _OFF)
    p = make_params(0 + _OFF)
    for kk in p: p[kk] = p[kk].astype(np.float64)
    bRng = np.random.default_rng(1 + _OFF)
    w = win_batch(bRng, 0, 5000, 3)
    ctx, tgt = w[:, :TCTX], w[:, TCTX:]
    msk = np.ones((3, K), dtype=bool)
    tok = w.copy(); tok[:, TCTX:] = MASK
    t0, g = loss_and_grads(p, tok, tgt, msk)
    worst = 0.0
    for kk, arr in p.items():
        flat = arr.reshape(-1)
        for _ in range(2):
            i = rng.integers(0, flat.size)
            eps = 1e-6; old = flat[i]
            flat[i] = old + eps; lp, _ = loss_and_grads(p, tok, tgt, msk)
            flat[i] = old - eps; lm, _ = loss_and_grads(p, tok, tgt, msk)
            flat[i] = old
            num = (lp - lm) / (2 * eps); ana = g[kk].reshape(-1)[i]
            absd = abs(num - ana)
            err = absd / max(1e-8, abs(num) + abs(ana))
            ok = (absd < 1e-6) or (err < 1e-3)
            worst = max(worst, 0.0 if ok else err)
            if not ok:
                R("  gradcheck %s[%d]: num %.8f ana %.8f abs %.2e rel %.2e" % (kk, i, num, ana, absd, err))
    R("grad check %s (worst %.2e)" % ("OK" if worst == 0 else "FAILED", worst))
    return worst == 0

def eval_model(p, n=1024):
    bRng = np.random.default_rng(999 + _OFF)
    w = win_batch(bRng, split, len(ids), n)
    ctx, tgt = w[:, :TCTX], w[:, TCTX:]
    nll = 0.0; acc = 0.0; exact4 = 0; ns = 0
    refine_exact = 0
    for s in range(0, n, 256):
        e = min(s + 256, n)
        wv = w[s:e]
        Bn = e - s
        tok = wv.copy(); tok[:, TCTX:] = MASK
        # one-shot (all masked, single pass)
        h = fwd_caches(p, tok)[0]
        preds = np.zeros((Bn, K), dtype=np.int64)
        conf = np.zeros((Bn, K))
        ok4 = True; lsum = 0.0; acnt = 0
        for j in range(K):
            lg = h[:, TCTX + j, :] @ p["Wout"]
            lg = lg - lg.max(1, keepdims=True)
            Z = np.exp(lg); sm = Z / Z.sum(1, keepdims=True)
            lsum += float((-np.log(sm[np.arange(Bn), tgt[s:e, j]] + 1e-12)).sum())
            pr = lg.argmax(1); preds[:, j] = pr
            conf[:, j] = sm[np.arange(Bn), pr]
            acnt += int((pr == tgt[s:e, j]).sum())
            ok4 &= bool((pr == tgt[s:e, j]).all())
        nll += lsum / K; acc += acnt; exact4 += int(ok4); ns += Bn
        # refine loop (MaskGIT reveal): per sample reveal most-confident slot/round
        tokr = tok.copy()
        rpreds = np.zeros((Bn, K), dtype=np.int64)
        revealed = np.zeros((Bn, K), dtype=bool)
        rok = np.zeros(Bn, dtype=bool)
        for rnd in range(K):
            hr = fwd_caches(p, tokr)[0]
            rconf = np.full((Bn, K), -1.0)
            for j in range(K):
                col = ~revealed[:, j]
                if col.any():
                    lg = hr[col, TCTX + j, :] @ p["Wout"]
                    lg = lg - lg.max(1, keepdims=True)
                    Z = np.exp(lg); sm = Z / Z.sum(1, keepdims=True)
                    pr = lg.argmax(1)
                    rpreds[col, j] = pr
                    rconf[col, j] = sm[np.arange(col.sum()), pr]
            pick = rconf.argmax(1)
            for b in range(Bn):
                j = pick[b]
                revealed[b, j] = True
                tokr[b, TCTX + j] = rpreds[b, j]
            # after all revealed in final round, mark correctness
            if rnd == K - 1:
                rok = (rpreds == tgt[s:e]).all(1)
        refine_exact += int(rok.sum())
    return {"nll_bits": round(nll / ns, 4), "acc": round(acc / (ns * K), 4),
            "exact4_oneshot": round(exact4 / ns, 4), "exact4_refine": round(refine_exact / ns, 4)}

def train(steps, seed, ternary=False, teacher=None, soft_w=0.5):
    p = make_params(seed)
    m = {k: np.zeros_like(x) for k, x in p.items()}
    v = {k: np.zeros_like(x) for k, x in p.items()}
    bRng = np.random.default_rng(7000 + seed)
    traj = []; t0 = time.perf_counter()
    for step in range(1, steps + 1):
        lr = LR * min(1.0, step / 300.0) * (0.1 + 0.45 * (1 + np.cos(np.pi * step / steps)))
        pq = ternize(p) if (ternary or teacher is not None) else p
        w = win_batch(bRng, 0, split, B)
        ctx, tgt = w[:, :TCTX], w[:, TCTX:]
        nmask = bRng.integers(1, K + 1, size=B)
        msk = np.zeros((B, K), dtype=bool)
        for b in range(B):
            cols = bRng.choice(K, size=nmask[b], replace=False)
            msk[b, cols] = True
        tok = w.copy()
        tok[:, TCTX:] = np.where(msk, MASK, tgt)
        tea = None
        if teacher is not None:
            th, _ = fwd_caches(teacher, tok)
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
            R("  step %5d %s (%.0fs)" % (step, ev, time.perf_counter() - t0))
    pq = ternize(p) if (ternary or teacher is not None) else p
    return pq, traj

if __name__ == "__main__":
    R("=== EXPERIMENT R: natural-language-domain micro-assembly ===")
    if not gradcheck():
        R("GRAD CHECK FAILED - ABORT"); sys.exit(2)
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    arms = [("teacher", dict()), ("tern_raw", dict(ternary=True)),
            ("tern_distill", dict(ternary=True, teacher="NEED"))]
    if "teacher" not in results:
        R("--- teacher (fp) ---")
        tp, traj = train(STEPS, 42)
        results["teacher"] = dict(final=eval_model(tp, n=2048), traj=traj)
        np.savez(("nl_teacher.npz"), **tp)
        json.dump(results, open(OUT, "w"), indent=1)
        R("=== teacher FINAL %s ===" % results["teacher"]["final"])
    _z = np.load(("nl_teacher.npz"))
    tp = {kk: _z[kk] for kk in _z.files}
    if "tern_raw" not in results:
        R("--- tern_raw ---")
        sp, traj = train(STEPS, 43, ternary=True)
        results["tern_raw"] = dict(final=eval_model(sp, n=2048), traj=traj)
        json.dump(results, open(OUT, "w"), indent=1)
        R("=== tern_raw FINAL %s ===" % results["tern_raw"]["final"])
    if "tern_distill" not in results:
        R("--- tern_distill ---")
        sp, traj = train(STEPS, 44, ternary=True, teacher=tp)
        results["tern_distill"] = dict(final=eval_model(sp, n=2048), traj=straj if False else traj)
        json.dump(results, open(OUT, "w"), indent=1)
        R("=== tern_distill FINAL %s ===" % results["tern_distill"]["final"])
    R("=== SUMMARY ===")
    for k in ("teacher", "tern_raw", "tern_distill"):
        R("%s: %s" % (k, results[k]["final"]))
