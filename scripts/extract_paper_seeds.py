# -*- coding: utf-8 -*-
"""论文 B 种子数据总提取 v3 (终版)"""
import glob, io, json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def load(p):
    return json.load(io.open(p, encoding="utf-8"))

def fin(d):
    return d.get("final", d) if isinstance(d, dict) else d

def get_arm(d, key):
    """seed42 有显式臂键; kaggle 种子版自动探测含 final 的臂键"""
    if key and key in d:
        return fin(d[key])
    for k, v in d.items():
        if isinstance(v, dict) and ("final" in v or "top1" in v):
            return fin(v)
    return fin(d)

def m(x):
    return "exact4=%s nll=%s" % (x.get("exact4"), x.get("nll_bits"))

CN = r"H:\AimCode\lab\cpu-native"
KT = r"H:\AimCode\build-tools\kaggle_out_paper_s"
W2 = r"H:\AimCode\build-tools\kaggle_out_w2_"

def files(base42, key42, stem):
    out = [("seed42", base42, key42)]
    for s in (1, 2, 3):
        p = KT + str(s) + "\\" + stem % s
        if os.path.exists(p):
            out.append(("s%d" % s, p, None))
    return out

print("=" * 74)
print("T (frame fp) — top1/none exact4+nll, 4 seeds")
print("=" * 74)
for tag, p, key in files(os.path.join(CN, "blueprint_frame_results.json"), "fp", "blueprint_frame_s%d_results.json"):
    f = get_arm(load(p), key)
    print("%-7s top1 %s | none %s" % (tag, m(f["top1"]), m(f.get("none", {}))))

print()
print("=" * 74)
print("T2 (frame_tern) — tern_raw/tern_distill top1/none, 4 seeds")
print("=" * 74)
for tag, p, key in files(os.path.join(CN, "blueprint_frame_tern_results.json"), None, "blueprint_frame_tern_s%d_results.json"):
    d = load(p)
    for arm in ("tern_raw", "tern_distill"):
        if arm in d:
            f = fin(d[arm])
            print("%-7s %-12s top1 %s | none %s" % (tag, arm, m(f["top1"]), m(f["none"])))
        elif tag != "seed42":
            f = get_arm(d, None)
            print("%-7s %-12s top1 %s | none %s" % (tag, "(arm)", m(f["top1"]), m(f["none"])))
            break

print()
print("=" * 74)
print("S (rag 相似检索) — teacher 三模式 + retrieval_gain, 4 seeds")
print("=" * 74)
for tag, p, key in files(os.path.join(CN, "blueprint_rag_results.json"), "teacher", "blueprint_rag_s%d_results.json"):
    f = get_arm(load(p), key)
    g = f.get("retrieval_gain_bits")
    print("%-7s top1 %s | random %s | none %s | gain=%s" % (
        tag, m(f["top1"]), m(f.get("random", {})), m(f["none"]), g))

print()
print("=" * 74)
print("Q3 (装配 2x3 矩阵) — seed42 / s1 / s2 (acc_y final)")
print("=" * 74)
for tag, p in (("seed42", os.path.join(CN, "blueprint_linear_results.json")),
               ("s1", W2 + "linear\\blueprint_linear_s1_results.json"),
               ("s2", r"H:\AimCode\lab\cpu-native\w3a_harvest\aim-w2-linear-s2\blueprint_linear_s2_results.json")):
    d = load(p)
    for k in ("teacher_8k", "teacher_ft", "tern_raw", "tern_distill", "tern_raw_v2"):
        if k in d:
            print("%-7s %-12s %s" % (tag, k, json.dumps(fin(d[k]).get("final", fin(d[k])), ensure_ascii=False)[:110]))

print()
print("=" * 74)
print("Q1/Q2/Q4 (装配/蒸馏/温度) s1 — seed42 主结果已收, s1 补充")
print("=" * 74)
for tag, p in (("assembly_s1", W2 + "assembly\\blueprint_assembly_s1_results.json"),
               ("distill_s1", W2 + "distill\\blueprint_distill_s1_results.json"),
               ("tempdist_s1", W2 + "tempdist\\blueprint_tempdist_s1_results.json")):
    if os.path.exists(p):
        d = load(p)
        print(tag, "keys:", list(d.keys())[:8])
        print("   ", json.dumps({k: fin(v).get("final", fin(v)) for k, v in d.items() if isinstance(v, dict) and not isinstance(fin(v), dict)}, ensure_ascii=False)[:200] or json.dumps(d, ensure_ascii=False)[:400])

print()
print("=" * 74)
print("R (nl 真域装配) — seed42 / s1 (nll / exact4_refine)")
print("=" * 74)
for tag, p in (("seed42", os.path.join(CN, "blueprint_nl_results.json")),
               ("s1", r"H:\AimCode\lab\cpu-native\w3_wave2\_v3\blueprint_nl_s1_results.json")):
    d = load(p)
    for k in ("teacher", "tern_raw", "tern_distill"):
        f = fin(d[k])
        print("%-7s %-12s nll=%s refine=%s" % (tag, k, f.get("nll_bits"), f.get("exact4_refine")))

print()
print("=" * 74)
print("C (cot_mech) acc_y1 — seed42 vs s1")
print("=" * 74)
for tag, p in (("seed42", r"H:\AimCode\aimstack\lab\results\cot_mech_results.json"),
               ("s1", r"H:\AimCode\aimstack\lab\seeds\w2-cotmech\cot_mech_s1_results.json")):
    d = load(p)
    print(tag, {k: v.get("acc_y1") for k, v in d.items() if isinstance(v, dict)})

print()
print("=" * 74)
print("op18 (lambda 溢价) — seed42 原始 + s1 + s2")
print("=" * 74)
ops = [("s1", r"H:\AimCode\aimstack\lab\seeds\op-seed1\op18_results.json"),
       ("s2", r"H:\AimCode\aimstack\lab\seeds\op-seed2\op18_results.json")]
alt = glob.glob(r"H:\AimCode\lab\cpu-native\**\op18_results.json", recursive=True) + glob.glob(r"H:\AimCode\aimstack\lab\results\op18*.json")
if alt:
    ops.insert(0, ("seed42", alt[0]))
for tag, p in ops:
    if not os.path.exists(p):
        print(tag, "MISSING"); continue
    d = load(p)
    cps = d.get("checkpoints", {})
    if isinstance(cps, dict):
        for lam, c in cps.items():
            print("%-9s lam=%-4s val=%s ood=%s margin=%s (step %s)" % (
                tag, lam, c.get("val"), c.get("ood"), c.get("margin"), c.get("step")))
