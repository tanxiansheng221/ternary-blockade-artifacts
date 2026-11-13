# -*- coding: utf-8 -*-
import os as _aim_os
AIM_SEED = int(_aim_os.environ.get("AIM_SEED", "1"))
AIM_TAG = _aim_os.environ.get("AIM_TAG", "")
_OFF = (AIM_SEED - 1) * 2

# EXPERIMENT H (problem 20, pair 2): linear attention x ternary weights.
# Blueprint wall #2: does 1.58-bit ternary quantization destabilize the
# linear-attention recurrence (S_t = S_{t-1} + phi(k_t) v_t^T) during training?
# Two runs (identical init/seed): fp linear-attn LM vs ternary linear-attn LM
# -> "ternary tax" in bits/char + gradient-norm stability trace.
# v0 scope: single layer, no decay/gate (Katharopoulos-style), phi = elu+1.
import numpy as np, time, json

R = lambda s: print(s, flush=True)

# ---- corpus: same ASCII source as cmax/bitnet labs ----
import glob

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
split = int(len(ids) * 0.9)
R("corpus %d chars | vocab %d" % (len(corpus), V))

T, B, D = 32, 192, 64
F32 = np.float32

def windows(lo, hi, n_max):
    n = min(hi - lo - T, n_max)
    return np.stack([ids[lo + i: lo + i + T + 1] for i in range(n)])  # T+1: input+target

Xtr = windows(0, split, 90_000)
Xva = windows(split, len(ids) - T - 1, 4_000)
R("train %d seqs | val %d seqs | T=%d B=%d D=%d" % (len(Xtr), len(Xva), T, B, D))

def ternize(W, on):
    if not on: return W
    g = np.mean(np.abs(W)) + 1e-8
    return np.clip(np.round(W / g), -1, 1) * g

def build():
    g = np.random.default_rng(42 + _OFF)  # identical init for both runs
    return [g.standard_normal((V, D)).astype(F32) * 0.05,          # emb
            (g.standard_normal((D, D)) / np.sqrt(D) * 0.5).astype(F32),  # Wq
            (g.standard_normal((D, D)) / np.sqrt(D) * 0.5).astype(F32),  # Wk
            (g.standard_normal((D, D)) / np.sqrt(D) * 0.5).astype(F32),  # Wv
            (g.standard_normal((D, V)) / np.sqrt(D) * 0.5).astype(F32),  # head
            np.zeros(V, F32)]                                            # head bias

def elu1(x):   # phi(x) = elu(x)+1 >= 0
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0))) + 1.0

def elu1_p(x): # phi'(x)
    return np.where(x > 0, 1.0, np.exp(np.minimum(x, 0)))

def forward(p, batch, ternary, store=False):
    """batch: (B, T+1) ids. returns mean NLL (bits/char). Linear-attn causal LM."""
    emb, Wq, Wk, Wv, Wh, bh = [ternize(x, ternary) if (ternary and i in (1, 2, 3)) else x
                               for i, x in enumerate(p)]
    x_in = batch[:, :-1]   # (B,T)
    y_in = batch[:, 1:]    # (B,T)
    Bn = len(batch)
    X = emb[x_in]          # (B,T,D)
    S = np.zeros((Bn, D, D), F32)
    z = np.zeros((Bn, D), F32)
    cache = [] if store else None
    nll = 0.0
    for t in range(T):
        xt = X[:, t]                                   # (B,D)
        qx = xt @ Wq; kx = xt @ Wk; vx = xt @ Wv
        q = elu1(qx); k = elu1(kx); v = vx
        S = S + k[:, :, None] * v[:, None, :]          # (B,Dk,Dv)
        z = z + k
        u = np.einsum('bkd,bd->bk', S, q)              # (B,Dv)
        dq_den = np.einsum('bd,bd->b', z, q) + 1e-6    # (B,)
        o = u / dq_den[:, None]
        h = o + xt                                     # residual
        logits = h @ Wh + bh
        lg = logits - logits.max(1, keepdims=True)
        lse = np.log(np.exp(lg).sum(1))
        nll += (lse - lg[np.arange(Bn), y_in[:, t]]).sum()
        if store:
            cache.append((xt, qx, kx, vx, q, k, v, S.copy(), z.copy(), u, dq_den, h, lg))
    return nll / (np.log(2) * Bn * T), cache, (Wq, Wk, Wv, Wh)

def train(ternary, steps=1500, lr=1e-3, wd=1e-4):
    p = build()
    m = [np.zeros_like(x, F32) for x in p]; v = [np.zeros_like(x, F32) for x in p]
    brng = np.random.default_rng(0 + _OFF)
    traj, gmax = [], 0.0
    for step in range(1, steps + 1):
        bi = brng.choice(len(Xtr), size=B, replace=True)
        batch = Xtr[bi]
        nll, cache, (Wq, Wk, Wv, Wh) = forward(p, batch, ternary, store=True)
        Bn = B
        g = [np.zeros_like(x) for x in p]
        dS = np.zeros((Bn, D, D), F32)
        dz = np.zeros((Bn, D), F32)
        for t in reversed(range(T)):
            xt, qx, kx, vx, q, k, vt, St, zt, u, den, h, lg = cache[t]
            yt = batch[:, 1 + t]
            P = np.exp(lg); P /= P.sum(1, keepdims=True)
            dlog = P.copy(); dlog[np.arange(Bn), yt] -= 1.0
            dlog /= (Bn * T)                            # mean over batch*time
            g[4] += h.T @ dlog; g[5] += dlog.sum(0)
            dh = dlog @ Wh.T                            # (B,D)
            do = dh.copy()                              # residual: h = o + x
            dx = dh.copy()
            # o = u/den ; u = St q ; den = z.q + eps
            dden = -(do * u).sum(1) / (den ** 2)        # (B,)
            du = do / den[:, None]                      # (B,Dv)
            dq = np.einsum('bkd,bk->bd', St, du) / den[:, None]
            dq += dden[:, None] * zt
            dz += dden[:, None] * q
            dS += du[:, :, None] * q[:, None, :]        # S += k (x) v path
            dk = np.einsum('bkd,bd->bk', dS, vt)        # dS @ v  (contract Dv)
            dv = np.einsum('bkd,bk->bd', dS, k)         # k^T @ dS (contract Dk)
            # through phi
            dkx = dk * elu1_p(kx); dqx = dq * elu1_p(qx)
            g[2] += xt.T @ dkx; g[1] += xt.T @ dqx; g[3] += xt.T @ dv
            dx += dkx @ Wk.T + dqx @ Wq.T + dv @ Wv.T   # STE: grads hit master W
            np.add.at(g[0], batch[:, t], dx)
            # carry backward through recurrence
            dS = dS.copy()  # S_t grad flows to S_{t-1} unchanged
        gmax = max(gmax, float(np.sqrt(sum(float((gi.astype(np.float64) ** 2).sum()) for gi in g))))
        for i in range(len(p)):
            m[i] = (0.9 * m[i] + 0.1 * g[i]).astype(F32)
            v[i] = (0.999 * v[i] + 0.001 * g[i] * g[i]).astype(F32)
            p[i] = (p[i] - lr * ((m[i] / (1 - 0.9 ** step)) /
                    (np.sqrt(v[i] / (1 - 0.999 ** step)) + 1e-8) + wd * p[i])).astype(F32)
        if step % 250 == 0 or step == 1:
            vl, _, _ = forward(p, Xva[:1024], ternary)
            traj.append((step, round(float(vl), 3)))
            R("  step %5d val %.3f bits/char" % (step, vl))
    vl, _, _ = forward(p, Xva[:2048], ternary)
    return traj, float(vl), gmax

R("=== EXPERIMENT H: LINEAR-ATTN x TERNARY (problem 20, pair 2) ===")
t0 = time.perf_counter()
R("[H1] full-precision linear attention")
tr1, f1, gm1 = train(ternary=False)
R("[H2] ternary linear attention (STE)")
tr2, f2, gm2 = train(ternary=True)
R("wall: %.0fs" % (time.perf_counter() - t0))
R("=== RESULT: full-prec %.3f bits/char | ternary %.3f bits/char | ternary tax = %.3f bits ===" % (f1, f2, f2 - f1))
R("grad-norm max: fp %.1f | ternary %.1f" % (gm1, gm2))
json.dump(dict(full_precision=tr1, ternary=tr2, fp_final=f1, tern_final=f2,
               tax=round(f2 - f1, 3), grad_max_fp=gm1, grad_max_tern=gm2,
               config=dict(T=T, B=B, D=D, steps=1500, lr=1e-3, corpus_chars=len(corpus))),
          open("linear_ternary%s_results.json" % (("_" + AIM_TAG) if AIM_TAG else ""), "w"), indent=1)
R("saved: lab/cpu-native/linear_ternary_results.json")
