# -*- coding: utf-8 -*-
import os as _aim_os
AIM_SEED = int(_aim_os.environ.get("AIM_SEED", "1"))
AIM_TAG = _aim_os.environ.get("AIM_TAG", "")
_OFF = (AIM_SEED - 1) * 2

# EXPERIMENT G (problem 20): BitNet-ternary x masked-parallel-objective co-training.
# The blueprint's first load-bearing wall: do ternary weights train stably under a
# parallel multi-token (diffusion-cousin) objective?
# Two runs: full-precision masked vs ternary masked -> "ternary tax" in bits/char.
import numpy as np, time, json, glob

R = lambda s: print(s, flush=True)
rng = np.random.default_rng(0 + _OFF)

# corpus: same ASCII source as cmax lab

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
chars = sorted(set(corpus)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
ids = np.array([stoi[c] for c in corpus], dtype=np.int32)
split = int(len(ids)*0.9)
K = 8  # window length = parallel prediction span
def windows(lo, hi, n_max):
    n = min(hi-lo-K, n_max)
    st = lo
    X = np.stack([ids[st+i:st+i+K] for i in range(n)])
    return X
Xtr = windows(0, split, 120_000)
Xva = windows(split, len(ids)-K, 8_000)
R("corpus %d chars | vocab %d | train %d win | val %d win" % (len(corpus), V, len(Xtr), len(Xva)))

D, H = 256, 512
MASK_ID = V  # extra embedding slot for mask token

def build():
    return [(rng.standard_normal((V+1, D))*0.05),                 # emb (+mask)
            (rng.standard_normal((K, D))*0.02),                   # positional
            (rng.standard_normal((D, H))/np.sqrt(D)), np.zeros(H),# hidden
            (rng.standard_normal((H, K*V))/np.sqrt(H))*0.5, np.zeros(K*V)]  # parallel head

def ternize(W, on):
    if not on: return W
    g = np.mean(np.abs(W)) + 1e-8
    return np.clip(np.round(W/g), -1, 1) * g

def fwd(p, x, ternary):
    emb, pos, W1, b1, W2, b2 = p
    W1q = ternize(W1, ternary); W2q = ternize(W2, ternary)
    B, L = x.shape
    m = np.full((B, L), MASK_ID, dtype=np.int64)
    mask = rng.random((B, L)) < 0.3
    has = mask.sum(1) == 0
    if has.any():
        mask[has, rng.integers(0, L, size=has.sum())] = True
    m[~mask] = x[~mask]  # unmasked = true char; masked = MASK_ID
    Hx = emb[m] + pos[None, :, :]
    Hs = Hx.sum(1)
    A1 = Hs@W1q+b1; Z = np.maximum(A1, 0)
    logits = Z@W2q+b2                                                # (B, K*V)
    return logits.reshape(B, K, V), mask, (Hs, A1, Z), (W1q, W2q)

def val_loss(p, ternary, n_batches=8):
    tot = 0.0; cnt = 0
    for i in range(n_batches):
        xb = Xva[i*512:(i+1)*512]
        if len(xb) < 10: break
        lg, mask, _, _ = fwd(p, xb, ternary)
        lg = lg - lg.max(2, keepdims=True)
        lse = np.log(np.exp(lg).sum(2))
        lgy = np.take_along_axis(lg, xb[:, :, None], 2)[:, :, 0]
        nll = lse - lgy  # NLL = logsumexp - target logit
        tot += nll[mask].sum()/np.log(2); cnt += mask.sum()
    return tot/cnt

def run(ternary, steps=1500, lr=2e-3, wd=1e-4):
    p = build()
    m = [np.zeros_like(x) for x in p]; v = [np.zeros_like(x) for x in p]
    traj = []
    for step in range(1, steps+1):
        bi = rng.choice(len(Xtr), size=512, replace=True)
        xb = Xtr[bi]
        logits, mask, (Hs, A1, Z), (W1q, W2q) = fwd(p, xb, ternary)
        B = len(bi); y = xb
        lg = logits - logits.max(2, keepdims=True)
        Ps = np.exp(lg); Ps /= Ps.sum(2, keepdims=True)
        # grad: dlogits = (Ps - onehot(y)) * mask / n_masked
        n_masked = max(mask.sum(), 1)
        onehot = np.zeros_like(Ps)
        onehot[np.arange(B)[:, None], np.arange(K)[None, :], y] = 1.0
        dl = (Ps - onehot) * mask[:, :, None] / n_masked
        dlogits = dl.reshape(B, K*V)
        gW2 = Z.T@dlogits; gb2 = dlogits.sum(0)
        dZ = dlogits@W2q.T; dZ[A1 <= 0] = 0
        gW1 = Hs.T@dZ; gb1 = dZ.sum(0)
        dHs = dZ@W1q.T
        gemb = np.zeros_like(p[0]); gpos = np.zeros_like(p[1])
        # embedding grad via masked ids
        mid = np.full((B, K), MASK_ID, dtype=np.int64); mid[~mask] = xb[~mask]
        for Li in range(K):
            np.add.at(gemb, mid[:, Li], dHs)  # sum-pool: each position sees full dHs
            np.add.at(gpos, [Li], dHs.sum(0)[None, :])
        grads = [gemb, gpos, gW1, gb1, gW2, gb2]
        for i, (pp, g) in enumerate(zip(p, grads)):
            m[i] = 0.9*m[i]+0.1*g; v[i] = 0.999*v[i]+0.001*g*g
            pp -= lr*((m[i]/(1-0.9**step))/(np.sqrt(v[i]/(1-0.999**step))+1e-8) + wd*pp)
        if step % 250 == 0 or step == 1:
            vl = val_loss(p, ternary)
            traj.append((step, round(vl, 3)))
            R("  step %5d val %.3f bits/char" % (step, vl))
    return traj, val_loss(p, ternary)

R("=== EXPERIMENT G: TERNARY x MASKED-OBJECTIVE (problem 20) ===")
t0 = time.perf_counter()
R("[G1] full-precision masked (sanity)")
tr1, f1 = run(ternary=False)
R("[G2] ternary masked (BitNet-style STE)")
tr2, f2 = run(ternary=True)
R("wall: %.0fs" % (time.perf_counter()-t0))
R("=== RESULT: full-prec %.3f bits/char | ternary %.3f bits/char | ternary tax = %.3f bits ===" %
  (f1, f2, f2-f1))
json.dump(dict(full_precision=tr1, ternary=tr2, fp_final=f1, tern_final=f2,
               tax=round(f2-f1, 3)),
          open("bitnet_masked%s_results.json" % (("_" + AIM_TAG) if AIM_TAG else ""), "w"), indent=1)
R("saved: lab/cpu-native/bitnet_masked_results.json")
