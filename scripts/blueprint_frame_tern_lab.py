# -*- coding: utf-8 -*-
# EXPERIMENT T2 (chapter 76): ternary core x framework retrieval — the last
# untested cell of the six-piece blueprint (2x3 matrix discipline: every cell
# measured). Reuses experiment T's corpus/library/instances EXACTLY (same
# seeds), and its fp arm (frame_params.npz) as distillation teacher.
# Preregistration:
#   - tern_raw (STE hard targets): per 2x3 matrix, BLAS cell tern-raw blocked.
#     Blocked := exact4(top1) < 20% @4000 steps.
#   - tern_distill (T=1.0 soft_w=0.5, per Q4 sharpness law): OPEN cell — unlike
#     Q3's perfect teacher (argmax 100%, no dark knowledge) this teacher has
#     ~0.4 bits/char residual entropy, so real soft structure exists.
#     Rescue := exact4(top1) >= 30% @4000.
# Resume-aware: completed arms are skipped.
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
OUT = "blueprint_frame_tern%s_results.json" % (("_" + AIM_TAG) if AIM_TAG else "")
TCKPT = "frame_params%s.npz" % (("_" + AIM_TAG) if AIM_TAG else "")

# ---- corpus: identical to blueprint_frame_lab.py (same seeds) ----
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
templates = {}
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

lib = {}
for k in keys:
    lib[k] = np.array([stoi[c] for c in templates[k][4:16]], dtype=np.int64)

def get_batch2(ins, bRng, n, mode_mix=True):
    tok = np.full((n, T), MASK, dtype=np.int64)
    tgt = np.zeros((n, K), dtype=np.int64)
    for b in range(n):
        d, kp = ins[bRng.integers(0, len(ins))]
        tok[b, TINJ:TINJ + TCTX] = d[kp:kp + TCTX]
        tgt[b] = d[kp + KEYLEN + 4:kp + KEYLEN + 8]
        key = "".join(chars[i] for i in d[kp:kp + KEYLEN])
        u = bRng.random()
        m = "top1" if (u < 0.5 or not mode_mix) else ("random" if u < 0.75 else "none")
        if m == "top1":
            tok[b, :TINJ] = lib[key]
        elif m == "random":
            kk = keys[bRng.integers(0, NTEMPL)]
            tok[b, :TINJ] = lib[kk]
    return tok, tgt

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

def loss_and_grads(pq, p_ref, tok, tgt, tp=None, soft_w=0.5):
    # pq = params used for fwd/bwd (ternized in ternary arms); p_ref = params
    # that receive gradients (STE: grads computed wrt pq, applied to p_ref).
    Bn = tok.shape[0]
    h, caches = fwd_caches(pq, tok)
    ht = fwd_caches(tp, tok)[0] if tp is not None else None
    dh = np.zeros_like(h)
    g = {k: np.zeros_like(x) for k, x in p_ref.items()}
    tot = 0.0
    gW = np.zeros_like(pq["Wout"])
    base = TINJ + TCTX
    for j in range(K):
        pos = base + j
        lg = h[:, pos, :] @ pq["Wout"]
        lg = lg - lg.max(1, keepdims=True)
        Z = np.exp(lg); sm = Z / Z.sum(1, keepdims=True)
        ti = tgt[:, j]
        nll = float(-np.log(sm[np.arange(Bn), ti] + 1e-12).mean())
        tot += nll
        dl = sm.copy()
        dl[np.arange(Bn), ti] -= 1.0
        if ht is not None:
            lg_t = ht[:, pos, :] @ tp["Wout"]     # T=1.0 sharp teacher (Q4 law)
            lg_t = lg_t - lg_t.max(1, keepdims=True)
            Zt = np.exp(lg_t); pt = Zt / Zt.sum(1, keepdims=True)
            dl = (1.0 - soft_w) * dl + soft_w * (sm - pt)
        dl /= Bn
        gW += h[:, pos, :].T @ dl
        dh[:, pos, :] += dl @ pq["Wout"].T
    g["Wout"] = gW
    g1 = bwd(pq, tok, caches, dh)
    for kk in g1: g[kk] = g[kk] + g1[kk]
    return tot, g

def eval_model(p):
    bRng = np.random.default_rng(999)
    out = {}
    for mode in ("top1", "random", "none", "fakekey"):
        acc = 0.0; ref4 = 0; n = 0; nll = 0.0
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
                    fk = tok[b, TINJ:TINJ + TCTX].copy()
                    fk[2] = (fk[2] + 7) % NV
                    tok[b, TINJ:TINJ + TCTX] = fk
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
        out[mode] = {"nll_bits": round(nll / (n * K), 4), "acc": round(acc / (n * K), 4),
                     "exact4": round(ref4 / n, 4)}
    return out

def train_arm(ternary, teacher, seed):
    p = make_params(seed)
    m = {k: np.zeros_like(x) for k, x in p.items()}
    v = {k: np.zeros_like(x) for k, x in p.items()}
    bRng = np.random.default_rng(7042 + seed)
    traj = []; t0 = time.perf_counter()
    for step in range(1, STEPS + 1):
        lr = LR * min(1.0, step / 300.0) * (0.1 + 0.45 * (1 + np.cos(np.pi * step / STEPS)))
        pq = ternize(p) if ternary else p
        tok, tgt = get_batch2(train_ins, bRng, B)
        _, g = loss_and_grads(pq, p, tok, tgt, tp=teacher)
        for kk in p:
            m[kk] = 0.9 * m[kk] + 0.1 * g[kk]
            v[kk] = 0.999 * v[kk] + 0.001 * g[kk] * g[kk]
            p[kk] -= lr * ((m[kk] / (1 - 0.9 ** step)) /
                           (np.sqrt(v[kk] / (1 - 0.999 ** step)) + 1e-8) + WD * p[kk])
        if step % 500 == 0:
            ev = eval_model(ternize(p) if ternary else p)
            traj.append([step, ev])
            R("  [%s] step %5d top1 %.3f/%.2f none %.3f (%.0fs)" %
              ("tern" if ternary else "fp", step, ev["top1"]["exact4"],
               ev["top1"]["nll_bits"], ev["none"]["nll_bits"], time.perf_counter() - t0))
    pf = ternize(p) if ternary else p
    return dict(final=eval_model(pf), traj=traj, dumps=eval_samples(pf))

def eval_samples(p):
    bRng = np.random.default_rng(999)
    out = {}
    for mode in ("top1", "none", "fakekey"):
        rows = []
        for s in range(0, 3 * 128, 128):
            ins = eval_ins[s:s + 128]
            tok = np.full((len(ins), T), MASK, dtype=np.int64)
            tgt = np.zeros((len(ins), K), dtype=np.int64)
            for b, (d, kp) in enumerate(ins):
                tok[b, TINJ:TINJ + TCTX] = d[kp:kp + TCTX]
                tgt[b] = d[kp + KEYLEN + 4:kp + KEYLEN + 8]
                if mode == "top1":
                    tok[b, :TINJ] = lib["".join(chars[i] for i in d[kp:kp + KEYLEN])]
                elif mode == "fakekey":
                    fk = tok[b, TINJ:TINJ + TCTX].copy()
                    fk[2] = (fk[2] + 7) % NV
                    tok[b, TINJ:TINJ + TCTX] = fk
            h = fwd_caches(p, tok)[0]
            for j in range(K):
                lg = h[:, TINJ + TCTX + j, :] @ p["Wout"]
                preds_j = lg.argmax(1)
                if j == 0:
                    preds = preds_j[:, None]
                else:
                    preds = np.concatenate([preds, preds_j[:, None]], 1)
            for b in range(3):
                rows.append("[%s] ctx=%s tgt=%s pred=%s" % (mode,
                  "".join(chars[i] for i in tok[b, TINJ:TINJ + TCTX]),
                  "".join(chars[i] for i in tgt[b]),
                  "".join(chars[i] for i in preds[b])))
            if len(rows) >= 9: break
        out[mode] = rows[:3]
    return out

if __name__ == "__main__":
    R("=== EXPERIMENT T2: ternary x framework retrieval (last blueprint cell) ===")
    tp = None
    if os.path.exists(TCKPT):
        z = np.load(TCKPT)
        tp = {kk: z[kk] for kk in z.files}
        R("teacher loaded: %s" % {k: v for k, v in eval_model(tp).items() if k == "top1"})
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name, ternary, use_teacher, seed in (
            ("tern_raw", True, False, 43 + (AIM_SEED - 1) * 2),
            ("tern_distill", True, True, 44 + (AIM_SEED - 1) * 2)):
        if name in results:
            R("--- %s already done, skip ---" % name)
            continue
        R("--- arm %s ---" % name)
        results[name] = train_arm(ternary, tp if use_teacher else None, seed)
        json.dump(results, open(OUT, "w"), indent=1)
        R("=== %s FINAL %s ===" % (name, results[name]["final"]))
    R("=== SUMMARY (prereg: raw blocked if top1 exact4<0.20; distill rescue if >=0.30) ===")
    for name in ("tern_raw", "tern_distill"):
        if name in results:
            R("%s: top1 %s | none %s" % (name, results[name]["final"]["top1"],
                                         results[name]["final"]["none"]))
