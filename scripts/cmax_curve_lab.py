# -*- coding: utf-8 -*-
import os as _aim_os
AIM_SEED = int(_aim_os.environ.get("AIM_SEED", "1"))
AIM_TAG = _aim_os.environ.get("AIM_TAG", "")
_OFF = (AIM_SEED - 1) * 2

# EXPERIMENT E (problem 8): measure the bits/param capacity curve c_max(N) at small scale.
# Fixed corpus (S bits, measured by gzip), scale model size, train each to convergence,
# measure stored bits = (H_unigram - L_val) * corpus_chars. Find the saturation plateau.
import numpy as np, time, json, glob, gzip, io

R = lambda s: print(s, flush=True)
rng = np.random.default_rng(0 + _OFF)

# ---------- corpus: local ASCII text ----------

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
corpus = corpus[:1_000_000]
R("corpus: %d chars" % len(corpus))
zb = len(gzip.compress(corpus.encode()))
S_corpus = zb * 8
R("gzip: %d bytes -> S_corpus ~ %.2e bits" % (zb, S_corpus))

chars = sorted(set(corpus)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
ids = np.array([stoi[c] for c in corpus], dtype=np.int32)
split = int(len(ids) * 0.9)
K = 8
def make_set(lo, hi):
    n = hi - lo - K
    a = ids[lo:hi]
    X = np.stack([a[i:i+K] for i in range(n)])
    Y = a[K:hi]
    return X, Y
Xtr, Ytr = make_set(0, split)
Xva, Yva = make_set(split, len(ids)-K)
R("vocab %d | train %d | val %d samples" % (V, len(Xtr), len(Xva)))

# unigram baseline
cnt = np.zeros(V); np.add.at(cnt, Xtr.ravel(), 1)
pu = (cnt + 0.5) / (cnt.sum() + 0.5*V)
H_uni = -(np.log2(pu[Yva])).mean()
R("unigram baseline: %.3f bits/char on val" % H_uni)

def train_eval(D, H, epochs, lr=2e-3, wd=1e-4):
    emb = (rng.standard_normal((V, D)) * 0.05)
    W1 = rng.standard_normal((D, H))/np.sqrt(D); b1 = np.zeros(H)
    W2 = rng.standard_normal((H, V))/np.sqrt(H); b2 = np.zeros(V)
    p = [emb, W1, b1, W2, b2]
    m = [np.zeros_like(x) for x in p]; v = [np.zeros_like(x) for x in p]
    ntr = len(Xtr)
    step = 0
    for ep in range(epochs):
        perm = rng.permutation(ntr)
        for s in range(0, ntr-BATCH+1, BATCH):
            bi = perm[s:s+BATCH]
            xb, yb = Xtr[bi], Ytr[bi]
            Hx = emb[xb].sum(1); A1 = Hx@W1+b1; Z = np.maximum(A1,0); logits = Z@W2+b2
            B = len(bi)
            lg = logits - logits.max(1, keepdims=True)
            Ps = np.exp(lg); Ps /= Ps.sum(1, keepdims=True)
            dl = Ps.copy(); dl[np.arange(B), yb] -= 1; dl /= B
            gW2 = Z.T@dl; gb2 = dl.sum(0)
            dZ = dl@W2.T; dZ[A1<=0] = 0
            gW1 = Hx.T@dZ; gb1 = dZ.sum(0)
            dHx = dZ@W1.T
            gE = np.zeros_like(emb)
            np.add.at(gE, xb.ravel(), np.repeat(dHx, K, 0))
            grads = [gE, gW1, gb1, gW2, gb2]
            step += 1
            for i, (pp, g) in enumerate(zip(p, grads)):
                m[i] = 0.9*m[i]+0.1*g; v[i] = 0.999*v[i]+0.001*g*g
                pp -= lr*((m[i]/(1-0.9**step))/(np.sqrt(v[i]/(1-0.999**step))+1e-8) + wd*pp)
    Hv = p[0][Xva].sum(1); Zv = np.maximum(Hv@p[1]+p[2],0)
    lv = Zv@p[3]+p[4]
    lse = np.log(np.exp(lv - lv.max(1, keepdims=True)).sum(1)) + lv.max(1)
    nll = (lse - lv[np.arange(len(Yva)), Yva]).mean() / np.log(2)
    n_params = sum(x.size for x in p)
    return nll, n_params

BATCH = 1024
R("%8s %12s %12s %12s" % ("params", "val bits/ch", "learned bits", "bits/param"))
results = []
t0 = time.perf_counter()
for D, H in ((64,64), (128,128), (256,256), (384,384), (512,512), (768,768)):
    nll, npar = train_eval(D, H, epochs=2)
    learned = max(0.0, (H_uni - nll)) * len(corpus)
    bpp = learned / npar
    R("%8d %12.3f %12.1e %12.3f" % (npar, nll, learned, bpp))
    results.append(dict(params=npar, val_bits=nll, learned_bits=learned, bits_per_param=bpp))
R("wall: %.0fs" % (time.perf_counter()-t0))
R("S_corpus(gzip)=%.2e bits | plateau bits/param = c_max estimate" % S_corpus)
json.dump(dict(corpus_chars=len(corpus), S_corpus=S_corpus, H_uni=H_uni, results=results),
          open("cmax_curve%s_results.json" % (("_" + AIM_TAG) if AIM_TAG else ""), "w"), indent=1)
R("saved: lab/cpu-native/cmax_curve_results.json")
