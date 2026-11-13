# -*- coding: utf-8 -*-
import os as _aim_os
AIM_SEED = int(_aim_os.environ.get("AIM_SEED", "1"))
AIM_TAG = _aim_os.environ.get("AIM_TAG", "")
_OFF = (AIM_SEED - 1) * 2

# EXPERIMENT I v2 (problem 20, pair 3): iterative decoding x ternary, PROPER model.
# v1 lesson: sum-pool bag model is a marginal predictor -> zero refinement signal,
# pair-3 tax unmeasurable. v2: bidirectional linear-attention masked model
# (position-aware via per-position queries; S = sum_t phi(k_t) v_t^T, one shot,
# no T-loop) + MaskGIT random-ratio training + iterative confidence decode.
# fp vs ternary (same init): teacher-forced bits/char + decode round curve.
import numpy as np, time, json, glob

R = lambda s: print(s, flush=True)


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
corpus = _mk_corpus(800000, 0)

chars = sorted(set(corpus)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
ids = np.array([stoi[c] for c in corpus], dtype=np.int32)
split = int(len(ids) * 0.9)
# unigram reference
cnt = np.bincount(ids, minlength=V).astype(np.float64)
pu = cnt / cnt.sum()
H1 = -(pu * np.log2(np.maximum(pu, 1e-12))).sum()
R("corpus %d chars | vocab %d | unigram entropy %.3f bits | top1 freq %.3f" % (len(corpus), V, H1, pu.max()))

T = 16
B = 192
D, DH = 64, 128
MASK_ID = V

def windows(lo, hi, n_max):
    n = min(hi - lo - T, n_max)
    return np.stack([ids[lo + i: lo + i + T] for i in range(n)])

Xtr = windows(0, split, 90_000)
Xva = windows(split, len(ids) - T, 4_000)
R("train %d win | val %d win | T=%d" % (len(Xtr), len(Xva), T))

def ternize(W, on):
    if not on: return W
    g = np.mean(np.abs(W)) + 1e-8
    return np.clip(np.round(W / g), -1, 1) * g

def make_params(seed):
    g = np.random.default_rng(seed)
    sc = lambda *s: (g.standard_normal(s) / np.sqrt(s[0]) * 0.5).astype(np.float32)
    return [g.standard_normal((V + 1, D)).astype(np.float32) * 0.05,  # emb (+MASK)
            g.standard_normal((T, D)).astype(np.float32) * 0.02,      # pos
            sc(D, D), sc(D, D), sc(D, D),                             # Wq Wk Wv
            np.zeros(D, np.float32),
            sc(D, DH), np.zeros(DH, np.float32),                      # Wa ba
            sc(DH, V), np.zeros(V, np.float32)]                       # Wb bb

def elu1(x): return np.where(x > 0, x, np.expm1(np.minimum(x, 0))) + 1.0
def elu1_p(x): return np.where(x > 0, 1.0, np.exp(np.minimum(x, 0)))

def fwd_parts(p, m_ids):
    """m_ids: (B,T) ids (MASK slot where masked). returns logits + cache"""
    emb, pos, Wq, Wk, Wv, bq, Wa, ba, Wb, bb = p
    X = emb[m_ids] + pos[None, :, :]                       # (B,T,D)
    qx = X @ Wq; kx = X @ Wk; vx = X @ Wv
    q = elu1(qx); k = elu1(kx); v = vx
    S = np.einsum('btd,bte->bde', k, v)                    # (B,Dk,Dv) bidirectional
    z = k.sum(1)                                           # (B,Dk)
    u = np.einsum('bde,btd->bte', S, q)                    # (B,T,Dv)
    den = np.einsum('bd,btd->bt', z, q) + 1e-6             # (B,T)
    o = u / den[:, :, None]
    h = o + X
    A = h @ Wa + ba
    Hm = np.maximum(A, 0)
    logits = Hm @ Wb + bb
    return logits, (X, qx, kx, vx, q, k, v, S, z, u, den, h, A, Hm)

def loss_and_grads(p, batch, ternary, trng, ratio=None):
    Bn = len(batch)
    if ratio is None:
        ratio = trng.uniform(0.15, 0.85)
    mask = trng.random((Bn, T)) < ratio
    has = mask.sum(1) == 0
    if has.any():
        mask[has, trng.integers(0, T, size=has.sum())] = True
    m_ids = np.full((Bn, T), MASK_ID, dtype=np.int64); m_ids[~mask] = batch[~mask]
    pq = [ternize(x, ternary) if (ternary and i in (2, 3, 4, 8)) else x for i, x in enumerate(p)]
    logits, c = fwd_parts(pq, m_ids)
    lg = logits - logits.max(2, keepdims=True)
    P = np.exp(lg); P /= P.sum(2, keepdims=True)
    n_masked = max(mask.sum(), 1)
    onehot = np.zeros_like(P)
    onehot[np.arange(Bn)[:, None], np.arange(T)[None, :], batch] = 1.0
    dl = (P - onehot) * mask[:, :, None] / n_masked
    X, qx, kx, vx, q, k, v, S, z, u, den, h, A, Hm = c
    g = [np.zeros_like(x) for x in p]
    g[8] = (Hm.reshape(-1, DH).T @ dl.reshape(-1, V)); g[9] = dl.sum((0, 1))
    dHm = dl @ pq[8].T; dHm[A <= 0] = 0
    dh = dHm @ pq[6].T
    g[6] = h.reshape(-1, D).T @ dHm.reshape(-1, DH); g[7] = dHm.sum((0, 1))
    do = dh.copy(); dX = dh.copy()
    dden = -(do * u).sum(2) / (den ** 2)                   # (B,T)
    du = do / den[:, :, None]
    dq = np.einsum('bde,bte->btd', S, du) / den[:, :, None]
    dq += dden[:, :, None] * z[:, None, :]
    dz = np.einsum('bt,btd->bd', dden, q)                  # (B,Dk)
    dS = np.einsum('bte,btd->bde', du, q)                  # (B,Dk,Dv)
    dk = np.einsum('bde,bte->btd', dS, v) + dz[:, None, :] # z = sum_t k_t
    dv = np.einsum('btd,bde->bte', k, dS)                  # dv[t,e] = k[t,:] @ dS
    dqx = dq * elu1_p(qx); dkx = dk * elu1_p(kx); dvx = dv * elu1_p(vx)
    g[2] = np.einsum('btd,bte->de', X, dqx)
    g[3] = np.einsum('btd,bte->de', X, dkx)
    g[4] = np.einsum('btd,bte->de', X, dvx)
    dX += dqx @ pq[2].T + dkx @ pq[3].T + dvx @ pq[4].T    # STE -> master
    g[0] = np.zeros_like(p[0])
    np.add.at(g[0], m_ids.ravel(), dX.reshape(-1, D))
    g[1] = dX.sum(0)
    return g, c, mask

def val_nll(p, ternary, ratio=0.5, n=2048):
    trng = np.random.default_rng(7 + _OFF)
    tot, cnt = 0.0, 0
    for i in range(0, n, 512):
        xb = Xva[i:i + 512]
        _, c, mask = loss_and_grads(p, xb, ternary, trng, ratio=ratio)
        m_ids = np.full((len(xb), T), MASK_ID, dtype=np.int64); m_ids[~mask] = xb[~mask]
        logits, _ = fwd_parts([ternize(x, ternary) if (ternary and i in (2, 3, 4, 8)) else x
                               for i, x in enumerate(p)], m_ids)
        lg = logits - logits.max(2, keepdims=True)
        lse = np.log(np.exp(lg).sum(2))
        nll = lse - np.take_along_axis(lg, xb[:, :, None], 2)[:, :, 0]
        tot += nll[mask].sum(); cnt += mask.sum()
    return tot / cnt / np.log(2)

def iterative_decode(p, ternary, rounds=8, n=2048):
    xb = Xva[:n]
    cur = np.full_like(xb, MASK_ID)
    revealed = np.zeros_like(xb, dtype=bool)
    trng = np.random.default_rng(7 + _OFF)
    accs = []
    for r_i in range(rounds):
        m_ids = np.where(revealed, xb, MASK_ID)
        logits, _ = fwd_parts([ternize(x, ternary) if (ternary and i in (2, 3, 4, 8)) else x
                               for i, x in enumerate(p)], m_ids)
        lg = logits - logits.max(2, keepdims=True)
        P = np.exp(lg); P /= P.sum(2, keepdims=True)
        conf = P.max(2)
        still = ~revealed
        n_reveal = max(1, int(np.ceil(T * (r_i + 1) / rounds)) * len(xb)) - int(revealed.sum())
        score = np.where(still, conf, -1.0)
        order = np.argsort(-score.ravel())
        take = min(n_reveal, int(still.sum()))
        sel = np.zeros(score.size, bool); sel[order[:take]] = True
        pick = sel.reshape(xb.shape) & still
        cur = np.where(pick, np.argmax(P, 2), cur)
        revealed |= pick
        accs.append(round(float((cur == xb)[revealed].mean()), 3))
    return accs, float((cur == xb).mean()), float((cur == xb).all(1).mean())

def run(ternary, steps=1500, lr=1e-3, wd=1e-4):
    p = make_params(42 + _OFF)
    trng = np.random.default_rng(0 + _OFF)
    m = [np.zeros_like(x) for x in p]; v = [np.zeros_like(x) for x in p]
    traj = []
    for step in range(1, steps + 1):
        bi = trng.choice(len(Xtr), size=B, replace=True)
        g, _, _ = loss_and_grads(p, Xtr[bi], ternary, trng)
        for i in range(len(p)):
            m[i] = 0.9 * m[i] + 0.1 * g[i]
            v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i]
            p[i] -= lr * ((m[i] / (1 - 0.9 ** step)) /
                          (np.sqrt(v[i] / (1 - 0.999 ** step)) + 1e-8) + wd * p[i])
        if step % 250 == 0 or step == 1:
            vl = val_nll(p, ternary)
            traj.append((step, round(float(vl), 3)))
            R("  step %5d val(r=0.5) %.3f bits/char" % (step, vl))
    return p, traj, val_nll(p, ternary)

R("=== EXPERIMENT I v2: ITERATIVE-DECODE x TERNARY (bidirectional linear attn) ===")
t0 = time.perf_counter()
R("[I1] full-precision")
p1, tr1, f1 = run(ternary=False)
R("[I2] ternary (STE)")
p2, tr2, f2 = run(ternary=True)
R("wall: %.0fs" % (time.perf_counter() - t0))
acc1, fa1, ex1 = iterative_decode(p1, False)
acc2, fa2, ex2 = iterative_decode(p2, True)
R("[I1] fp  rounds: %s | final %.3f | exact %.3f" % (acc1, fa1, ex1))
R("[I2] tern rounds: %s | final %.3f | exact %.3f" % (acc2, fa2, ex2))
R("=== RESULT: teacher-forced fp %.3f | tern %.3f | tax %.3f bits | decode-acc fp %.3f vs tern %.3f ===" %
  (f1, f2, f2 - f1, fa1, fa2))
json.dump(dict(unigram_entropy=round(H1, 3), full_precision=tr1, ternary=tr2,
               fp_final=f1, tern_final=f2, tax=round(f2 - f1, 3),
               fp_round_accs=acc1, tern_round_accs=acc2,
               fp_final_acc=fa1, tern_final_acc=fa2, fp_exact=ex1, tern_exact=ex2,
               config=dict(T=T, B=B, D=D, DH=DH, steps=1500, rounds=8)),
          open("diffusion_ternary_v2%s_results.json" % (("_" + AIM_TAG) if AIM_TAG else ""), "w"), indent=1)
R("saved: lab/cpu-native/diffusion_ternary_v2_results.json")
