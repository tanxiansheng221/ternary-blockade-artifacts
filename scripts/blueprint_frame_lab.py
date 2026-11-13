# -*- coding: utf-8 -*-
# EXPERIMENT T (chapter 75): framework-library retrieval, the real-domain form.
# Composition-domain corpus: real text as filler ('~' filtered out) + 20
# framework templates (key "~~"+6 unique chars + 24-char constant body).
# 400 train docs / 100 eval docs (shared library = the framework's nature).
# Task = infiller: ctx = key(8)+body[0:4] (12), masked slots = body[4:8]
# (answer ON DISK inside the template), inj slot 12 = retrieved template
# body[0:12] by exact key lookup vs random template vs none.
# Preregistered: top1 ~ near-100% (copyable), none/random ~ chance (~2.8%)
# => random-domain "answer on disk" verdict reproduced in real-text clothing.
import numpy as np, json, time, os, sys, glob, string
import os as _aim_os
AIM_SEED = int(_aim_os.environ.get("AIM_SEED", "1"))
AIM_TAG = _aim_os.environ.get("AIM_TAG", "")

R = lambda s: print(s, flush=True)
D, L, H1 = 128, 2, 256
TINJ, TCTX, K = 12, 12, 4
T = TINJ + TCTX + K
B, STEPS, LR, WD = 96, 4000, 2e-3, 1e-4
NTEMPL, NTRAIN, NEVAL = 20000, 2000, 200
OUT = "blueprint_frame%s_results.json" % (("_" + AIM_TAG) if AIM_TAG else "")

# ---- corpus: real text filler + synthetic framework templates ----
paths = []
for pat in (r"H:\AimCode\build-tools\*.py", r"H:\AimCode\lab\**\*.md", r"H:\AimCode\lab\**\*.py",
            r"H:\AimCode\AimCode\**\*.py"):
    paths += glob.glob(pat, recursive=True)
paths += glob.glob(os.path.join(os.path.dirname(sys.executable), "Lib", "**", "*.py"), recursive=True)
paths = sorted(set(paths))
corpus = ""
for f in paths:
    if len(corpus) >= 2_000_000:
        break
    try:
        corpus += open(f, encoding="utf-8", errors="ignore").read() + "\n"
    except Exception:
        pass
if len(corpus) < 100000:
    for _pat in ("/usr/local/lib/python3*/**/*.py", "/usr/lib/python3*/**/*.py"):
        paths += glob.glob(_pat, recursive=True)
    paths = sorted(set(paths))
    for _f in paths:
        if len(corpus) >= 1000000:
            break
        try:
            corpus += open(_f, encoding="utf-8", errors="ignore").read() + "\n"
        except Exception:
            pass
filler = "".join(c for c in corpus if (32 <= ord(c) < 127 or c == "\n") and c != "~")
rng = np.random.default_rng(7)
alpha = np.array(list(string.ascii_uppercase + string.digits))
keys = set()
while len(keys) < NTEMPL:
    keys.add("~~" + "".join(rng.choice(alpha, size=6)))
keys = sorted(keys)
templates = {}   # key -> body (24 chars, string)
for k in keys:
    templates[k] = "".join(rng.choice(alpha, size=24))
R("library: %d templates (key 8 + body 24)" % NTEMPL)

filler_arr = np.frombuffer(filler.encode("ascii"), dtype=np.uint8)

def make_doc(rng):
    parts = []
    order = rng.choice(NTEMPL, size=20, replace=False)
    for i in order:
        g = filler_arr[rng.integers(0, len(filler_arr), size=int(rng.integers(30, 60)))]
        parts.append(g.tobytes().decode("ascii"))
        parts.append(keys[i] + templates[keys[i]])
    g = filler_arr[rng.integers(0, len(filler_arr), size=int(rng.integers(30, 60)))]
    parts.append(g.tobytes().decode("ascii"))
    return "".join(parts)

train_docs = [make_doc(np.random.default_rng(100 + i)) for i in range(NTRAIN)]
eval_docs = [make_doc(np.random.default_rng(500 + i)) for i in range(NEVAL)]
all_text = "".join(train_docs) + "".join(eval_docs)
chars = sorted(set(all_text)); NV = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
V = NV + 1; MASK = NV
KEYLEN = 8

# instance table: (doc_ids, key_start)
def instances(docs):
    out = []
    for d in docs:
        ids = np.array([stoi[c] for c in d], dtype=np.int32)
        pos = 0
        while True:
            j = d.find("~~", pos)
            if j < 0 or j + KEYLEN + 24 > len(d):
                break
            out.append((ids, j))
            pos = j + KEYLEN + 24
    return out

train_ins = instances(train_docs); eval_ins = instances(eval_docs)
R("instances: train %d eval %d | vocab %d" % (len(train_ins), len(eval_ins), NV))

lib = {}   # key string -> body ids (the framework library on "disk")
for k in keys:
    lib[k] = np.array([stoi[c] for c in templates[k][4:16]], dtype=np.int64)  # body[0:12]

def get_batch(ins, bRng, n, mode_mix=True, mode=None):
    tok = np.full((n, T), MASK, dtype=np.int64)
    tgt = np.zeros((n, K), dtype=np.int64)
    modes = []
    for b in range(n):
        d, kp = ins[bRng.integers(0, len(ins))]
        tok[b, TINJ:TINJ + TCTX] = d[kp:kp + TCTX]
        tgt[b] = d[kp + KEYLEN + 4:kp + KEYLEN + 8]
        u = bRng.random()
        m = mode if mode else ("top1" if (u < 0.5 or not mode_mix) else ("random" if u < 0.75 else "none"))
        modes.append(m)
        if m == "top1":
            tok[b, :TINJ] = lib[d[kp:kp + KEYLEN].tobytes().decode("ascii", "ignore").replace("~~", "~~")]
    return tok, tgt, np.ones((n, K), dtype=bool), modes

# fix: lib keyed by key string; build key->body[0:12] properly
lib = {}
for k in keys:
    lib[k] = np.array([stoi[c] for c in templates[k][4:16]], dtype=np.int64)

def get_batch2(ins, bRng, n, mode_mix=True, mode=None):
    tok = np.full((n, T), MASK, dtype=np.int64)
    tgt = np.zeros((n, K), dtype=np.int64)
    for b in range(n):
        d, kp = ins[bRng.integers(0, len(ins))]
        tok[b, TINJ:TINJ + TCTX] = d[kp:kp + TCTX]
        tgt[b] = d[kp + KEYLEN + 4:kp + KEYLEN + 8]
        key = "".join(chars[i] for i in d[kp:kp + KEYLEN])
        u = bRng.random()
        m = mode if mode else ("top1" if (u < 0.5 or not mode_mix) else ("random" if u < 0.75 else "none"))
        if m == "top1":
            tok[b, :TINJ] = lib[key]
        elif m == "random":
            kk = keys[bRng.integers(0, NTEMPL)]
            tok[b, :TINJ] = lib[kk]
    return tok, tgt, np.ones((n, K), dtype=bool)

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

def loss_and_grads(p, tok, tgt, msk):
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
        ti = tgt[:, j]
        nll = float(-np.log(sm[np.arange(Bn), ti] + 1e-12).mean())
        tot += nll
        dl = sm.copy()
        dl[np.arange(Bn), ti] -= 1.0
        dl /= Bn
        gW += h[:, pos, :].T @ dl
        dh[:, pos, :] += dl @ p["Wout"].T
    g["Wout"] = gW
    g1 = bwd(p, tok, caches, dh)
    for kk in g1: g[kk] = g[kk] + g1[kk]
    return tot, g

def eval_model(p):
    bRng = np.random.default_rng(999)
    out = {}
    for mode in ("top1", "random", "none", "fakekey"):
        acc = 0.0; ref4 = 0; n = 0; nll = 0.0
        first_dump = True
        for s in range(0, len(eval_ins), 128):
            ins = eval_ins[s:s + 128]
            Bn = len(ins)
            tok = np.full((Bn, T), MASK, dtype=np.int64)
            tgt = np.zeros((Bn, K), dtype=np.int64)
            for b, (d, kp) in enumerate(ins):
                tok[b, TINJ:TINJ + TCTX] = d[kp:kp + TCTX]
                tgt[b] = d[kp + KEYLEN + 4:kp + KEYLEN + 8]
                if mode == "top1":
                    tok[b, :TINJ] = lib["".join(chars[i] for i in d[kp:kp + KEYLEN])]
                elif mode == "random":
                    tok[b, :TINJ] = lib[keys[bRng.integers(0, NTEMPL)]]
                elif mode == "fakekey":
                    fk = tok[b, TINJ:TINJ + TCTX].copy()   # ctx with key corrupted
                    fk[2] = (fk[2] + 7) % NV
                    tok[b, TINJ:TINJ + TCTX] = fk          # inj stays MASK
            h = fwd_caches(p, tok)[0]
            preds = np.zeros((Bn, K), dtype=np.int64)
            for j in range(K):
                lg = h[:, TINJ + TCTX + j, :] @ p["Wout"]
                lg = lg - lg.max(1, keepdims=True)
                Z = np.exp(lg); sm = Z / Z.sum(1, keepdims=True)
                nll += float((-np.log(sm[np.arange(Bn), tgt[:, j]] + 1e-12)).sum())
                preds[:, j] = lg.argmax(1)
                acc += int((preds[:, j] == tgt[:, j]).sum())
            ref4 += int(((preds == tgt).all(1)).sum()); n += Bn
            if first_dump and mode in ("none", "fakekey"):
                first_dump = False
                for b in range(3):
                    R("    [%s] ctx=%s tgt=%s pred=%s" % (mode,
                      "".join(chars[i] for i in tok[b, TINJ:TINJ + TCTX]),
                      "".join(chars[i] for i in tgt[b]),
                      "".join(chars[i] for i in preds[b])))
        out[mode] = {"nll_bits": round(nll / (n * K), 4), "acc": round(acc / (n * K), 4),
                     "exact4": round(ref4 / n, 4)}
    return out

if __name__ == "__main__":
    R("=== EXPERIMENT T: framework-library retrieval (real-domain form) ===")
    p = make_params(42 + (AIM_SEED - 1))
    m = {k: np.zeros_like(x) for k, x in p.items()}
    v = {k: np.zeros_like(x) for k, x in p.items()}
    bRng = np.random.default_rng(7042 + (AIM_SEED - 1))
    traj = []; t0 = time.perf_counter()
    results = {}
    for step in range(1, STEPS + 1):
        lr = LR * min(1.0, step / 300.0) * (0.1 + 0.45 * (1 + np.cos(np.pi * step / STEPS)))
        tok, tgt, msk = get_batch2(train_ins, bRng, B)
        _, g = loss_and_grads(p, tok, tgt, msk)
        for kk in p:
            m[kk] = 0.9 * m[kk] + 0.1 * g[kk]
            v[kk] = 0.999 * v[kk] + 0.001 * g[kk] * g[kk]
            p[kk] -= lr * ((m[kk] / (1 - 0.9 ** step)) /
                           (np.sqrt(v[kk] / (1 - 0.999 ** step)) + 1e-8) + WD * p[kk])
        if step % 500 == 0:
            ev = eval_model(p)
            traj.append([step, ev])
            R("  step %5d top1 %.3f none %.3f (%.0fs)" %
              (step, ev["top1"]["nll_bits"], ev["none"]["nll_bits"], time.perf_counter() - t0))
    results["fp"] = dict(final=eval_model(p), traj=traj)
    json.dump(results, open(OUT, "w"), indent=1)
    np.savez("frame_params%s.npz" % (("_" + AIM_TAG) if AIM_TAG else ""), **p)
    R("=== FINAL %s ===" % results["fp"]["final"])
