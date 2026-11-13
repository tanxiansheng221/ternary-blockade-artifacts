# -*- coding: utf-8 -*-
import os as _aim_os
AIM_SEED = int(_aim_os.environ.get("AIM_SEED", "1"))
AIM_TAG = _aim_os.environ.get("AIM_TAG", "")
_OFF = (AIM_SEED - 1) * 2

# LAMBDA LAB SUITE v2: A (transfer/haircut), B (complexity window), C (structural prior)
import numpy as np, time, json

R = lambda s: print(s, flush=True)
rng = np.random.default_rng(0 + _OFF)
P, EMB, HID = 97, 96, 192
BATCH, lr = 1024, 1e-3
FDim = 32  # fourier: 8 harmonics x sin/cos x 2 operands

def make_data(task, p_mod=P):
    pairs = [(a, b) for a in range(p_mod) for b in range(p_mod)]
    if task == 'add': y = np.array([(a + b) % p_mod for a, b in pairs])
    else:             y = np.array([(a * b) % p_mod for a, b in pairs])
    idx = rng.permutation(len(pairs)); ntr = len(pairs)//2
    tr, va = idx[:ntr], idx[ntr:]
    Xa = np.array([pairs[i][0] for i in tr]); Xb = np.array([pairs[i][1] for i in tr]); Y = y[tr]
    Va = np.array([pairs[i][0] for i in va]); Vb = np.array([pairs[i][1] for i in va]); Vy = y[va]
    return (Xa, Xb, Y), (Va, Vb, Vy), ntr

def build(fourier):
    d_in = FDim if fourier else 2*EMB
    if fourier:
        return [(rng.standard_normal((d_in, HID)) / np.sqrt(d_in)), np.zeros(HID),
                (rng.standard_normal((HID, P)) / np.sqrt(HID)), np.zeros(P)]
    return [(rng.standard_normal((P, EMB)) * 0.5),
            (rng.standard_normal((d_in, HID)) / np.sqrt(d_in)), np.zeros(HID),
            (rng.standard_normal((HID, P)) / np.sqrt(HID)), np.zeros(P)]

def featurize(p, a, b, fourier, p_mod):
    if fourier:
        k = np.arange(1, 9)
        f = lambda x: np.concatenate([np.sin(2*np.pi*np.outer(x, k)/p_mod),
                                      np.cos(2*np.pi*np.outer(x, k)/p_mod)], 1)
        return np.concatenate([f(a), f(b)], 1)
    return np.concatenate([p[0][a], p[0][b]], 1)

def forward(p, a, b, fourier, p_mod):
    Hx = featurize(p, a, b, fourier, p_mod)
    if fourier: W1, b1, W2, b2 = p[0], p[1], p[2], p[3]
    else:       W1, b1, W2, b2 = p[1], p[2], p[3], p[4]
    A1 = Hx @ W1 + b1; Z = np.maximum(A1, 0)
    return Z @ W2 + b2, (Hx, A1, Z), (W1, W2)

def acc(p, a, b, y, fourier, p_mod):
    lg, _, _ = forward(p, a, b, fourier, p_mod)
    return (lg.argmax(1) == y).mean()

def run(task, wd, steps, fourier, eval_ood=False, p_ood=89, tag=''):
    (Xa, Xb, Ya), (Va, Vb, Vy), ntr = make_data(task)
    p = build(fourier)
    m = [np.zeros_like(x) for x in p]; v = [np.zeros_like(x) for x in p]
    T_mem = T_gen = None; log = []
    mask89 = (Va < p_ood) & (Vb < p_ood)
    Va89, Vb89 = Va[mask89], Vb[mask89]
    Vy89 = ((Va89 + Vb89) % p_ood) if task == 'add' else ((Va89 * Vb89) % p_ood)
    for step in range(1, steps+1):
        bi = rng.choice(ntr, size=BATCH, replace=True)
        logits, (Hx, A1, Z), (W1, W2) = forward(p, Xa[bi], Xb[bi], fourier, P)
        y = Ya[bi]; B = len(bi)
        lg = logits - logits.max(1, keepdims=True)
        Ps = np.exp(lg); Ps /= Ps.sum(1, keepdims=True)
        dl = Ps.copy(); dl[np.arange(B), y] -= 1; dl /= B
        dW2 = Z.T @ dl; db2 = dl.sum(0)
        dZ = dl @ W2.T; dZ[A1 <= 0] = 0
        dW1 = Hx.T @ dZ; db1 = dZ.sum(0)
        grads = [dW1, db1, dW2, db2]
        if not fourier:
            dHx = dZ @ W1.T
            dE = np.zeros_like(p[0])
            np.add.at(dE, Xa[bi], dHx[:, :EMB]); np.add.at(dE, Xb[bi], dHx[:, EMB:])
            grads = [dE] + grads
        for i, (pp, g) in enumerate(zip(p, grads)):
            m[i] = 0.9*m[i] + 0.1*g; v[i] = 0.999*v[i] + 0.001*g*g
            pp -= lr*((m[i]/(1-0.9**step))/(np.sqrt(v[i]/(1-0.999**step))+1e-8) + wd*pp)
        if step % 50 == 0 or step == 1:
            tr = acc(p, Xa, Xb, Ya, fourier, P)
            val = acc(p, Va, Vb, Vy, fourier, P)
            ood = acc(p, Va89, Vb89, Vy89, fourier, p_ood) if (eval_ood and Va89.size > 10) else None
            log.append(dict(step=step, train=round(tr,4), val=round(val,4),
                            ood89=round(ood,4) if ood is not None else None))
            if T_mem is None and tr > 0.9: T_mem = step
            if T_gen is None and tr > 0.9 and val > 0.9: T_gen = step
    eta = (T_gen * BATCH / ntr) if T_gen else None
    return dict(task=task, wd=wd, fourier=fourier, steps=steps, T_mem=T_mem, T_gen=T_gen,
                eta=round(eta, 1) if eta else None,
                final_train=log[-1]['train'], final_val=log[-1]['val'],
                final_ood89=log[-1]['ood89'], log=log)

R("=== LAMBDA LAB SUITE v2 ==="); t0 = time.perf_counter()
res = {}

R("[A] transfer/haircut: add, wd=1.0, 6000 steps, OOD=mod-89")
res['A'] = run('add', 1.0, 6000, False, eval_ood=True)
r = res['A']
R("  T_mem=%s T_gen=%s eta=%s val=%.3f ood89=%.3f" % (r['T_mem'], r['T_gen'], r['eta'], r['final_val'], r['final_ood89']))
for e in r['log']:
    if e['step'] in (500, 700, 800, 900, 1000, 1200, 1500, 2000, 3000, 4500, 6000):
        R("    step %5d train %.3f val %.3f ood89 %s" % (e['step'], e['train'], e['val'], e['ood89']))

R("[B] complexity: mul (a*b) mod 97, wd sweep, 8000 steps each")
res['B'] = {}
for wd in (1.0, 2.0, 4.0):
    r = run('mul', wd, 8000, False)
    res['B'][str(wd)] = {k: r[k] for k in ('T_mem','T_gen','eta','final_train','final_val')}
    R("  wd=%.1f T_mem=%s T_gen=%s eta=%s train=%.3f val=%.3f" %
      (wd, r['T_mem'], r['T_gen'], r['eta'], r['final_train'], r['final_val']))

R("[C] structural prior: add + Fourier features, wd=1.0, 4000 steps")
res['C'] = run('add', 1.0, 4000, True)
r = res['C']
R("  T_mem=%s T_gen=%s eta=%s train=%.3f val=%.3f" % (r['T_mem'], r['T_gen'], r['eta'], r['final_train'], r['final_val']))
for e in r['log']:
    if e['step'] in (50, 100, 200, 400, 800, 1500, 3000, 4000):
        R("    step %5d train %.3f val %.3f" % (e['step'], e['train'], e['val']))

R("wall: %.0fs" % (time.perf_counter()-t0))
json.dump(res, open("lambda_suite%s_results.json" % (("_" + AIM_TAG) if AIM_TAG else ""), "w"), indent=1)
R("saved: lab/cpu-native/lambda_suite_results.json")
