"""Run all experiments -> outputs/results.json + outputs/fig_*.png.

The submitted PDF report is a separate deliverable; this script reproduces the
JSON results and every figure used in that report.

Part A  headline benchmark: true one-record-at-a-time streaming (file -> parse -> update)
Part B  parameter sweeps: same algorithms, vectorised add_many on pre-parsed arrays
"""
import gc
import json
import math
import os
import random
import sys
import time
import tracemalloc
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from algorithms import BloomFilter, CountMinSketch, HyperLogLog, ReservoirSampler  # noqa: E402

SEED = 42
DATA = os.path.join(HERE, "data", "ml-1m", "ratings.dat")
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

# headline parameters
H_BLOOM_BITS_PER_ELEM, H_BLOOM_K = 10, 7
H_CMS_W, H_CMS_D = 2048, 5  # eps=e/w~0.0013 (eps*N~1.4k < top counts ~3.4k), delta=e^-d~0.0067
H_HLL_P = 14
H_RES_K = 10000


def log(*a):
    print(*a, flush=True)


def stream():
    """Yield (user, movie, rating) one line at a time; nothing is kept in memory."""
    with open(DATA, "r", encoding="latin-1") as f:
        for line in f:
            u, m, r, _ = line.split("::")
            yield int(u), int(m), int(r)


def pair_key(u, m):
    return u * 4096 + m  # movie ids <= 3952 < 4096


def timed(fn):
    gc.collect()
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def traced(fn):
    """Peak Python-heap bytes allocated while fn runs (structure + temporaries)."""
    gc.collect()
    tracemalloc.start()
    out = fn()
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, cur, peak


def load_arrays():
    import pandas as pd
    df = pd.read_csv(DATA, sep="::", engine="python", header=None, names=["u", "m", "r", "t"])
    return df.u.to_numpy(np.int64), df.m.to_numpy(np.int64), df.r.to_numpy(np.int64)


# =========================================================== Part A: headline
def part_a(U, M, R):
    res = {}
    N = len(U)
    log(f"[A] records = {N:,}")

    # --- read-only baseline (parse cost shared by every algorithm)
    def read_only():
        n = 0
        for _ in stream():
            n += 1
        return n
    _, t_read = timed(read_only)
    res["read_only_sec"] = t_read
    log(f"[A] read+parse only: {t_read:.2f}s")

    rng = np.random.default_rng(SEED)
    exact_pairs = set(pair_key(int(u), int(m)) for u, m in zip(U, M))
    # positives / negatives for the Bloom test
    pos_keys = rng.choice(np.fromiter(exact_pairs, dtype=np.int64), 200_000, replace=False)
    neg = []
    while len(neg) < 200_000:
        u, m = int(rng.integers(1, 6041)), int(rng.integers(1, 3953))
        k = pair_key(u, m)
        if k not in exact_pairs:
            neg.append(k)
    neg_keys = np.array(neg, dtype=np.int64)

    # ---------------- Bloom Filter vs set
    n_unique = len(exact_pairs)
    m_bits = H_BLOOM_BITS_PER_ELEM * n_unique

    def bloom_build():
        bf = BloomFilter(m_bits, H_BLOOM_K, SEED)
        for u, m, _ in stream():
            bf.add(pair_key(u, m))
        return bf
    bf, t_b = timed(bloom_build)
    _, _, peak_b = traced(lambda: bloom_build())
    t0 = time.perf_counter()
    fp = sum(1 for k in neg_keys.tolist() if k in bf)
    t_q = time.perf_counter() - t0
    fn_ = sum(1 for k in pos_keys.tolist() if k not in bf)

    def set_build():
        s = set()
        for u, m, _ in stream():
            s.add(pair_key(u, m))
        return s
    _, t_s = timed(set_build)
    _, cur_s, peak_s = traced(set_build)
    res["bloom"] = dict(
        n_unique=n_unique, m_bits=m_bits, k=H_BLOOM_K, bits_per_elem=H_BLOOM_BITS_PER_ELEM,
        fpr=fp / len(neg_keys), theory_fpr=BloomFilter.theoretical_fpr(n_unique, m_bits, H_BLOOM_K),
        fnr=fn_ / len(pos_keys), fill=bf.fill_ratio(),
        time_stream=t_b, time_net=t_b - t_read, throughput=N / t_b, query_us=t_q / len(neg_keys) * 1e6,
        struct_bytes=bf.nbytes, traced_peak=peak_b,
        exact_time=t_s, exact_peak=peak_s, exact_bytes=cur_s, exact_throughput=N / t_s)
    log("[A] bloom", res["bloom"])

    # ---------------- Count-Min Sketch vs Counter
    exact_movie, exact_user = Counter(M.tolist()), Counter(U.tolist())
    for name, col, exact in (("movie", 1, exact_movie), ("user", 0, exact_user)):
        def cms_build(col=col):
            c = CountMinSketch(H_CMS_W, H_CMS_D, SEED)
            for rec in stream():
                c.add(rec[col])
            return c
        cms, t_c = timed(cms_build)
        _, _, peak_c = traced(cms_build)

        def cnt_build(col=col):
            c = Counter()
            for rec in stream():
                c[rec[col]] += 1
            return c
        _, t_d = timed(cnt_build)
        _, cur_d, peak_d = traced(cnt_build)
        ids = np.array(sorted(exact))
        tru = np.array([exact[i] for i in ids])
        est = np.array([cms.estimate(int(i)) for i in ids])
        err = est - tru
        top = np.argsort(-tru)[:100]
        res[f"cms_{name}"] = dict(
            n_keys=len(ids), width=H_CMS_W, depth=H_CMS_D, eps=math.e / H_CMS_W, delta=math.exp(-H_CMS_D),
            mean_abs_err=float(err.mean()), max_abs_err=int(err.max()), min_err=int(err.min()),
            mean_rel_err=float((err / tru).mean()), top100_mean_rel_err=float((err[top] / tru[top]).mean()),
            bound=float(math.e / H_CMS_W * N), frac_within_bound=float((err <= math.e / H_CMS_W * N).mean()),
            time_stream=t_c, time_net=t_c - t_read, throughput=N / t_c,
            struct_bytes=cms.nbytes, traced_peak=peak_c, exact_time=t_d, exact_peak=peak_d, exact_bytes=cur_d,
            exact_throughput=N / t_d)
        log(f"[A] cms_{name}", res[f"cms_{name}"])

    # ---------------- HyperLogLog vs set
    truth = {"pair": n_unique, "user": len(exact_user), "movie": len(exact_movie)}
    hll_res = {}
    for name in ("pair", "user", "movie"):
        def hll_build(name=name):
            h = HyperLogLog(H_HLL_P, SEED)
            for u, m, _ in stream():
                h.add(pair_key(u, m) if name == "pair" else (u if name == "user" else m))
            return h
        h, t_h = timed(hll_build)
        est = h.estimate()
        hll_res[name] = dict(truth=truth[name], est=est, rel_err=abs(est - truth[name]) / truth[name],
                             time_stream=t_h, time_net=t_h - t_read, throughput=N / t_h)
    _, _, peak_h = traced(lambda: hll_build("pair"))
    res["hll"] = dict(p=H_HLL_P, m=1 << H_HLL_P, theory_std_err=HyperLogLog.std_error(H_HLL_P),
                      struct_bytes=1 << H_HLL_P, traced_peak=peak_h, exact_time=t_s, exact_peak=peak_s,
                      exact_bytes=cur_s, exact_throughput=N / t_s, by_field=hll_res)
    log("[A] hll", res["hll"])

    # ---------------- Reservoir sampling vs full data
    true_hist = np.bincount(R, minlength=6)[1:] / N
    true_mean = R.mean()
    rres = {}
    for algo in ("R", "L"):
        def res_build(algo=algo):
            rs = ReservoirSampler(H_RES_K, SEED, algo)
            for _, _, r in stream():
                rs.add(r)
            return rs
        rs, t_r = timed(res_build)
        _, _, peak_r = traced(res_build)
        smp = np.array(rs.sample)
        hist = np.bincount(smp, minlength=6)[1:] / len(smp)
        rres[algo] = dict(time_stream=t_r, time_net=t_r - t_read, throughput=N / t_r, traced_peak=peak_r,
                          tv=0.5 * float(np.abs(hist - true_hist).sum()), mean_err=float(abs(smp.mean() - true_mean)),
                          sample_hist=hist.tolist())

    def full_store():
        return [r for _, _, r in stream()]
    _, t_f = timed(full_store)
    _, cur_f, peak_f = traced(full_store)
    res["reservoir"] = dict(k=H_RES_K, true_hist=true_hist.tolist(), true_mean=float(true_mean), algos=rres,
                            struct_bytes=H_RES_K * 8, exact_time=t_f, exact_peak=peak_f, exact_bytes=cur_f,
                            exact_throughput=N / t_f)
    log("[A] reservoir", {k: {kk: vv for kk, vv in v.items() if kk != 'sample_hist'} for k, v in rres.items()})
    return res, exact_pairs, pos_keys, neg_keys, exact_movie


# ============================================================ Part B: sweeps
def part_b(U, M, R, exact_pairs, pos_keys, neg_keys, exact_movie):
    out = {}
    N = len(U)
    keys = U.astype(np.int64) * 4096 + M
    n_unique = len(exact_pairs)

    # ---- equivalence check scalar vs vector
    bf1, bf2 = BloomFilter(80000, 5, 1), BloomFilter(80000, 5, 1)
    for k in keys[:5000].tolist():
        bf1.add(k)
    bf2.add_many(keys[:5000])
    h1, h2 = HyperLogLog(10, 1), HyperLogLog(10, 1)
    for k in keys[:5000].tolist():
        h1.add(k)
    h2.add_many(keys[:5000])
    out["equivalence_ok"] = bool(bf1.bits == bf2.bits and h1.reg == h2.reg)
    log("[B] scalar==vectorised:", out["equivalence_ok"])

    # ---- Bloom grid
    ratios = [2, 4, 6, 8, 10, 12, 16]
    ks = [1, 2, 3, 4, 5, 7, 10, 14]
    grid = []
    for r in ratios:
        for k in ks:
            bf = BloomFilter(r * n_unique, k, SEED)
            _, t = timed(lambda: bf.add_many(keys))
            fpr = float(bf.contains_many(neg_keys).mean())
            fnr = float(1 - bf.contains_many(pos_keys).mean())
            grid.append(dict(bits_per_elem=r, k=k, fpr=fpr, theory=BloomFilter.theoretical_fpr(n_unique, r * n_unique, k),
                             fnr=fnr, fill=bf.fill_ratio(), bytes=bf.nbytes, time=t))
        log("[B] bloom ratio", r)
    out["bloom"] = grid

    # ---- CMS grid (key = movie id)
    ids = np.array(sorted(exact_movie))
    tru = np.array([exact_movie[i] for i in ids])
    top = np.argsort(-tru)[:100]
    grid = []
    for w in [16, 32, 64, 128, 272, 512, 1024, 2048, 4096]:
        for d in [1, 2, 3, 5, 8]:
            rel, rel_top, mae = [], [], []
            for s in range(5):
                c = CountMinSketch(w, d, seed=SEED + s)
                c.add_many(M)
                e = c.estimate_many(ids) - tru
                rel.append((e / tru).mean())
                rel_top.append((e[top] / tru[top]).mean())
                mae.append(e.mean())
            _, t = timed(lambda: CountMinSketch(w, d).add_many(M))
            grid.append(dict(width=w, depth=d, mean_rel_err=float(np.mean(rel)), top100_rel_err=float(np.mean(rel_top)),
                             mean_abs_err=float(np.mean(mae)), bytes=w * d * 4, time=t))
    out["cms"] = grid
    out["cms_n_keys"] = len(ids)
    log("[B] cms done")

    # ---- HLL over p (10 seeds)
    fields = {"pair": (keys, n_unique), "user": (U, len(np.unique(U))), "movie": (M, len(exact_movie))}
    rows = []
    for p in range(4, 17):
        for name, (arr, tr) in fields.items():
            errs, ests = [], []
            for s in range(10):
                h = HyperLogLog(p, seed=SEED + s)
                h.add_many(arr)
                ests.append(h.estimate())
                errs.append((ests[-1] - tr) / tr)
            _, t = timed(lambda: HyperLogLog(p).add_many(arr))
            rows.append(dict(p=p, m=1 << p, field=name, truth=tr, mean_abs_rel_err=float(np.mean(np.abs(errs))),
                             rmse=float(np.sqrt(np.mean(np.square(errs)))), mean_signed=float(np.mean(errs)),
                             theory=HyperLogLog.std_error(p), bytes=1 << p, time=t))
    out["hll"] = rows
    log("[B] hll done")

    # ---- Reservoir over k (30 trials) using Algorithm L index skipping
    true_hist = np.bincount(R, minlength=6)[1:] / N
    rrows = []
    for k in [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]:
        tv, me, cov = [], [], []
        t0 = time.perf_counter()
        for s in range(30):
            idx = np.array(ReservoirSampler.sample_indices_L(N, k, random.Random(SEED + s)))
            smp = R[idx]
            h = np.bincount(smp, minlength=6)[1:] / k
            tv.append(0.5 * np.abs(h - true_hist).sum())
            me.append(abs(smp.mean() - R.mean()))
            cov.append(len(np.unique(M[idx])) / len(exact_movie))
        t = (time.perf_counter() - t0) / 30
        rrows.append(dict(k=k, tv=float(np.mean(tv)), mean_err=float(np.mean(me)), movie_coverage=float(np.mean(cov)),
                          bytes=k * 8, time=t))
    out["reservoir"] = rrows

    # ---- uniformity: every position equally likely?  N=10,000, k=100, 2000 trials, deciles
    from scipy.stats import chisquare
    n, k, T = 10000, 100, 2000
    uni = {}
    for algo in ("R", "L"):
        cnt = np.zeros(n)
        for s in range(T):
            rs = ReservoirSampler(k, seed=s, algo=algo)
            for i in range(n):
                rs.add(i)
            cnt[rs.sample] += 1
        dec = cnt.reshape(10, -1).sum(1)
        stat, p = chisquare(dec)
        uni[algo] = dict(deciles=dec.tolist(), expected=T * k / 10, chi2=float(stat), p=float(p))
        log("[B] uniformity", algo, uni[algo]["chi2"], uni[algo]["p"])
    out["uniformity"] = uni
    return out


# =============================================================== figures
def figures(A, B):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for fam in ("Malgun Gothic", "NanumGothic", "AppleGothic", "Noto Sans CJK KR"):
        if fam in avail:
            plt.rcParams["font.family"] = fam
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130

    from matplotlib.ticker import FuncFormatter, NullFormatter

    def save(fig, name):
        # plain-text log tick labels (mathtext would need a glyph for U+2212 that Korean fonts lack)
        for a in fig.axes:
            for axis, sc in ((a.xaxis, a.get_xscale()), (a.yaxis, a.get_yscale())):
                if sc == "log":
                    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                    axis.set_minor_formatter(NullFormatter())
        fig.tight_layout(); fig.savefig(os.path.join(OUT, name)); plt.close(fig)

    # Bloom
    g = B["bloom"]
    ratios = sorted({x["bits_per_elem"] for x in g})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for r in ratios:
        rows = [x for x in g if x["bits_per_elem"] == r]
        ax[0].plot([x["k"] for x in rows], [max(x["fpr"], 1e-6) for x in rows], "o-", label=f"{r} bits/elem")
    ax[0].set_yscale("log"); ax[0].set_xlabel("해시 함수 개수 k"); ax[0].set_ylabel("False Positive Rate (log)")
    ax[0].set_title("Bloom Filter: k 증가가 항상 좋은가?"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    ax[1].scatter([x["theory"] for x in g], [x["fpr"] for x in g], s=14)
    lim = [1e-6, 1]; ax[1].plot(lim, lim, "r--", lw=1)
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlim(1e-5, 1); ax[1].set_ylim(1e-6, 1)
    ax[1].set_xlabel("이론 FPR (1-e^{-kn/m})^k"); ax[1].set_ylabel("실측 FPR"); ax[1].set_title("이론 vs 실측 (0은 표시 생략)")
    ax[1].grid(alpha=.3)
    save(fig, "fig_bloom.png")

    # CMS
    g = B["cms"]
    ws = sorted({x["width"] for x in g}); ds = sorted({x["depth"] for x in g})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for d in ds:
        rows = [x for x in g if x["depth"] == d]
        ax[0].plot([x["width"] for x in rows], [x["mean_rel_err"] for x in rows], "o-", label=f"depth={d}")
        ax[1].plot([x["width"] for x in rows], [x["top100_rel_err"] for x in rows], "o-", label=f"depth={d}")
    for a, t in zip(ax[:2], ["전체 영화 평균 상대오차", "상위 100 영화 평균 상대오차"]):
        a.set_xscale("log", base=2); a.set_yscale("log"); a.set_xlabel("width"); a.set_ylabel("상대오차(과대추정)")
        a.set_title(t); a.legend(fontsize=7); a.grid(alpha=.3)
    for d in ds:
        rows = [x for x in g if x["depth"] == d]
        ax[2].plot([x["bytes"] for x in rows], [x["mean_rel_err"] for x in rows], "o-", label=f"depth={d}")
    ax[2].axvline(A["cms_movie"]["exact_bytes"], color="k", ls="--"); ax[2].text(A["cms_movie"]["exact_bytes"], 1, " dict", fontsize=8)
    ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("sketch 크기 (bytes)"); ax[2].set_ylabel("평균 상대오차")
    ax[2].set_title("메모리-정확도 trade-off"); ax[2].grid(alpha=.3); ax[2].legend(fontsize=7)
    save(fig, "fig_cms.png")

    # HLL
    rows = B["hll"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name in ("pair", "user", "movie"):
        r = [x for x in rows if x["field"] == name]
        ax[0].plot([x["p"] for x in r], [x["rmse"] for x in r], "o-", label=f"{name} (n={r[0]['truth']:,})")
    ps = sorted({x["p"] for x in rows})
    ax[0].plot(ps, [HyperLogLog.std_error(p) for p in ps], "k--", label="이론 1.04/√m")
    ax[0].set_yscale("log"); ax[0].set_xlabel("p (레지스터 2^p개)"); ax[0].set_ylabel("RMSE 상대오차 (10 seeds)")
    ax[0].set_title("HyperLogLog 정확도"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    r = [x for x in rows if x["field"] == "pair"]
    ax[1].plot([x["bytes"] for x in r], [x["rmse"] for x in r], "o-")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("메모리 (bytes)"); ax[1].set_ylabel("RMSE 상대오차")
    ax[1].set_title("메모리-정확도 (고유 (user,movie) 쌍)"); ax[1].grid(alpha=.3)
    save(fig, "fig_hll.png")

    # Reservoir
    rows = B["reservoir"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ks = np.array([x["k"] for x in rows])
    ax[0].loglog(ks, [x["tv"] for x in rows], "o-", label="TV distance")
    ax[0].loglog(ks, [x["mean_err"] for x in rows], "s-", label="|평균 평점 오차|")
    ax[0].loglog(ks, 0.9 / np.sqrt(ks), "k--", label="∝1/√k")
    ax[0].set_xlabel("sample size k"); ax[0].set_title("Reservoir: 표본 크기별 오차 (30 trials)"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    ax[1].semilogx(ks, [x["movie_coverage"] for x in rows], "o-"); ax[1].set_xlabel("k"); ax[1].set_ylabel("영화 커버리지")
    ax[1].set_title("표본이 포함하는 고유 영화 비율"); ax[1].grid(alpha=.3)
    for algo, c in (("R", "C0"), ("L", "C1")):
        u = B["uniformity"][algo]
        ax[2].plot(range(1, 11), u["deciles"], "o-", color=c, label=f"Alg {algo} (p={u['p']:.2f})")
    ax[2].axhline(B["uniformity"]["R"]["expected"], color="k", ls="--", label="기대값")
    ax[2].set_xlabel("스트림 위치 decile"); ax[2].set_ylabel("선택 횟수"); ax[2].set_title("균등성 검정 (N=10,000, k=100, 2000회)")
    ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
    save(fig, "fig_reservoir.png")

    # Headline: memory / throughput
    names = ["Bloom\n(vs set)", "CMS movie\n(vs dict)", "HLL\n(vs set)", "Reservoir\n(vs list)"]
    sk = [A["bloom"]["struct_bytes"], A["cms_movie"]["struct_bytes"], A["hll"]["struct_bytes"], A["reservoir"]["struct_bytes"]]
    ex = [A["bloom"]["exact_bytes"], A["cms_movie"]["exact_bytes"], A["bloom"]["exact_bytes"], A["reservoir"]["exact_bytes"]]
    tp = [A["bloom"]["throughput"], A["cms_movie"]["throughput"], A["hll"]["by_field"]["pair"]["throughput"],
          A["reservoir"]["algos"]["R"]["throughput"]]
    etp = [A["bloom"]["exact_throughput"], A["cms_movie"]["exact_throughput"], A["hll"]["exact_throughput"],
           A["reservoir"]["exact_throughput"]]
    x = np.arange(4)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(x - .2, ex, .4, label="Exact"); ax[0].bar(x + .2, sk, .4, label="Streaming")
    ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels(names, fontsize=8); ax[0].set_ylabel("bytes (log)")
    ax[0].set_title("메모리 사용량"); ax[0].legend(); ax[0].grid(alpha=.3, axis="y")
    ax[1].bar(x - .2, etp, .4, label="Exact"); ax[1].bar(x + .2, tp, .4, label="Streaming")
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, fontsize=8); ax[1].set_ylabel("records/sec")
    ax[1].set_title("처리량 (파일 읽기+파싱 포함)"); ax[1].legend(); ax[1].grid(alpha=.3, axis="y")
    save(fig, "fig_headline.png")


def main():
    U, M, R = load_arrays()
    A, exact_pairs, pos, neg, exact_movie = part_a(U, M, R)
    B = part_b(U, M, R, exact_pairs, pos, neg, exact_movie)
    env = dict(python=sys.version.split()[0], numpy=np.__version__, n_records=len(U), seed=SEED)
    import platform
    env["platform"] = platform.platform(); env["cpu"] = platform.processor()
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(dict(env=env, A=A, B=B), f, indent=1, default=float)
    figures(A, B)
    log("done -> outputs/results.json, outputs/fig_*.png")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--figures-only":  # re-draw from saved results
        d = json.load(open(os.path.join(OUT, "results.json"), encoding="utf-8"))
        figures(d["A"], d["B"])
    else:
        main()
