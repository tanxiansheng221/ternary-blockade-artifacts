# -*- coding: utf-8 -*-
import os as _aim_os
AIM_SEED = int(_aim_os.environ.get("AIM_SEED", "1"))
AIM_TAG = _aim_os.environ.get("AIM_TAG", "")
_OFF = (AIM_SEED - 1) * 2

# EXPERIMENT F v3 (problem 33): factual storage density — associative lookup probe.
# Input: position-aware one-hot concat of key chars (each key = unique point in R^K*V).
# Output: last char of the associated city (lenient exact-match).
# Pure associative memory; sweep H -> recall curve -> bits/param at the knee.
import numpy as np, time, json, string

R = lambda s: print(s, flush=True)
rng = np.random.default_rng(3 + _OFF)

CITIES = ("paris tokyo madrid berlin london dubai moscow oslo cairo lima dublin "
          "manila vienna athens bangor geneva helsinki budapest jakarta warsaw "
          "kyoto lagos lisbon luxor malmo nairobi oakland perth quebec riga seoul "
          "taipei utrecht valencia xian yokohama zurich boston chicago denver evans "
          "fresno gdansk havana istanbul kiev luxemb montreal nantes odessa").split()
CITIES = sorted(set(CITIES))[:50]
NV = len(CITIES)
K, key_len = 6, 6
N_FACTS = 10000
keys = set()
while len(keys) < N_FACTS:
    keys.add("".join(rng.choice(list(string.ascii_lowercase + string.digits), key_len)))
keys = sorted(keys)
vals = rng.integers(0, NV, size=N_FACTS)
pairs = [(k, CITIES[v]) for k, v in zip(keys, vals)]
order = rng.permutation(N_FACTS); ntr = int(N_FACTS*0.8)
tr = [pairs[i] for i in order[:ntr]]
va = [pairs[i] for i in order[ntr:]]
S_facts = ntr * np.log2(NV)
alphabet = string.ascii_lowercase + string.digits
Ai = {c: i for i, c in enumerate(alphabet)}
A = len(alphabet)
def onehot(s):
    x = np.zeros(K*A, dtype=np.float32)
    for i, c in enumerate(s[:K]):
        x[i*A + Ai[c]] = 1.0
    return x
trX = np.stack([onehot(k + " ") for k, c in tr]); trY = np.array([Ai[c[-1]] for k, c in tr])
vaX = np.stack([onehot(k + " ") for k, c in va]); vaY = np.array([Ai[c[-1]] for k, c in va])
d_in = K*A

def build(H):
    return [(rng.standard_normal((d_in, H))/np.sqrt(d_in)), np.zeros(H),
            (rng.standard_normal((H, A))/np.sqrt(H)), np.zeros(A)]

def fwd(p, x):
    W1, b1, W2, b2 = p
    A1 = x@W1+b1; Z = np.maximum(A1, 0)
    return Z@W2+b2, (A1, Z)

def acc(p, x, y):
    lg, _ = fwd(p, x)
    return (lg.argmax(1) == y).mean()

def run(H, epochs=150):
    p = build(H)
    m = [np.zeros_like(x) for x in p]; v = [np.zeros_like(x) for x in p]
    step = 0
    for ep in range(epochs):
        perm = rng.permutation(ntr)
        for s in range(0, ntr-BATCH+1, BATCH):
            bi = perm[s:s+BATCH]
            xb, yb = trX[bi], trY[bi]
            W1, W2 = p[0], p[2]
            logits, (A1, Z) = fwd(p, xb)
            B = len(bi)
            lg = logits - logits.max(1, keepdims=True)
            Ps = np.exp(lg); Ps /= Ps.sum(1, keepdims=True)
            dl = Ps.copy(); dl[np.arange(B), yb] -= 1; dl /= B
            gW2 = Z.T@dl; gb2 = dl.sum(0)
            dZ = dl@W2.T; dZ[A1 <= 0] = 0
            gW1 = xb.T@dZ; gb1 = dZ.sum(0)
            grads = [gW1, gb1, gW2, gb2]
            step += 1
            for i, (pp, g) in enumerate(zip(p, grads)):
                m[i] = 0.9*m[i]+0.1*g; v[i] = 0.999*v[i]+0.001*g*g
                pp -= lr*((m[i]/(1-0.9**step))/(np.sqrt(v[i]/(1-0.999**step))+1e-8) + 1e-5*pp)
    return acc(p, trX, trY), acc(p, vaX, vaY), sum(x.size for x in p)

BATCH = 512; lr = 3e-3
R("=== EXPERIMENT F v3: FACTS CAPACITY — associative lookup (problem 33) ===")
R("%d facts | %d train | S=%.0fK bits | chance=%.1f%% | knee pred (2b/param): %dK params" %
  (N_FACTS, ntr, S_facts/1e3, 100/NV, S_facts/2/1e3))
R("%8s %10s %10s %14s" % ("params", "train_rec", "val_rec", "eff bits/param"))
results = []
t0 = time.perf_counter()
for H in (64, 128, 256, 512, 1024, 2048):
    tr_r, va_r, npar = run(H)
    eff = va_r * S_facts / npar
    R("%8d %10.3f %10.3f %14.3f" % (npar, tr_r, va_r, eff))
    results.append(dict(params=npar, train_recall=round(tr_r,4), val_recall=round(va_r,4),
                        bits_per_param=round(eff,3)))
R("wall: %.0fs" % (time.perf_counter()-t0))
json.dump(dict(n_facts=N_FACTS, S_facts=S_facts, results=results),
          open("facts_capacity%s_results.json" % (("_" + AIM_TAG) if AIM_TAG else ""), "w"), indent=1)
R("saved: lab/cpu-native/facts_capacity_results.json")
