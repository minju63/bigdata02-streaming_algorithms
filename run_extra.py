"""Additional experiments for Bloom Filter and Count-Min Sketch -> outputs/extra.json + fig_extra1/2.png.

 E1  Count-Min Top-K recall (heavy hitters), movie key and (movie, month) key
 E2  Bloom filter over capacity (more elements than it was designed for)
 E3  Merge check: sketches of shards merged == sketch of the whole stream
 E4  Memory vs stream length (exact structures grow, sketches stay constant)
 E5  Timing repeated 5 times (mean +- std), used for the time tables in the report
 E6  Bloom per-record add() cost vs k, CMS per-record add() cost vs depth
 E0  add() and add_many() produce identical structures
"""
import gc
import json
import os
import sys
import time
import tracemalloc
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from algorithms import BloomFilter, CountMinSketch  # noqa: E402
from run_experiments import DATA, OUT, SEED, stream  # noqa: E402

REPEATS = 5


def log(*a):
    print(*a, flush=True)


def load():
    import pandas as pd
    df = pd.read_csv(DATA, sep="::", engine="python", header=None, names=["u", "m", "r", "t"])
    return df.u.to_numpy(np.int64), df.m.to_numpy(np.int64), df.t.to_numpy(np.int64)


def topk_recall(true_counts, est, k):
    """|top-k by estimate  ∩  true top-k| / k  (ties broken by key order)."""
    a = set(np.argsort(-true_counts, kind="stable")[:k].tolist())
    b = set(np.argsort(-est, kind="stable")[:k].tolist())
    return len(a & b) / k


def per_record_cost(pair, movies):
    """microseconds per add() call, best of 3 over 200,000 records."""
    keys, mv = pair[:200_000].tolist(), movies[:200_000].tolist()
    res = {"bloom_k": [], "cms_depth": []}
    for k in (1, 2, 3, 4, 5, 7, 10, 14):
        best = min(_time(lambda: _fill(BloomFilter(10 * 1_000_209, k, SEED), keys)) for _ in range(3))
        res["bloom_k"].append(dict(k=k, us_per_record=best / len(keys) * 1e6))
    for d in (1, 2, 3, 5, 8):
        best = min(_time(lambda: _fill(CountMinSketch(2048, d, SEED), mv)) for _ in range(3))
        res["cms_depth"].append(dict(depth=d, us_per_record=best / len(mv) * 1e6))
    return res


def _fill(structure, items):
    add = structure.add
    for x in items:
        add(x)


def _time(fn):
    gc.collect()
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def main():
    U, M, T = load()
    N = len(U)
    pair = U * 4096 + M
    month = (T - T.min()) // (30 * 86400)
    movie_month = M * 64 + month  # (movie, month) key: "which movie was popular in which month"
    out = {}
    rng = np.random.default_rng(SEED)

    # ---------------------------------------------------------------- E0 add == add_many
    b1, b2 = BloomFilter(80_000, 5, 1), BloomFilter(80_000, 5, 1)
    for k in pair[:5000].tolist():
        b1.add(k)
    b2.add_many(pair[:5000])
    c1, c2 = CountMinSketch(256, 4, 1), CountMinSketch(256, 4, 1)
    for k in M[:5000].tolist():
        c1.add(k)
    c2.add_many(M[:5000])
    out["equivalence"] = dict(bloom=bool(b1.bits == b2.bits),
                              cms=all(bool((a == b).all()) for a, b in zip(c1._views, c2._views)))
    log("[E0]", out["equivalence"])

    # ---------------------------------------------------------------- E1 Top-K
    log("[E1] top-k")
    e1 = {}
    for name, keys, widths in (("movie", M, [64, 272, 1024, 2048, 4096]), ("movie_month", movie_month, [512, 2048, 8192, 32768])):
        ids, cnt = np.unique(keys, return_counts=True)
        rows = []
        top = np.argsort(-cnt, kind="stable")[:100]
        for w in widths:
            rec = {10: [], 50: [], 100: []}
            rel = []
            for s in range(5):
                c = CountMinSketch(w, 5, seed=SEED + s)
                c.add_many(keys)
                est = c.estimate_many(ids)
                for k in rec:
                    rec[k].append(topk_recall(cnt, est, k))
                rel.append(float(((est[top] - cnt[top]) / cnt[top]).mean()))
            rows.append(dict(width=w, depth=5, bytes=w * 5 * 4, recall10=float(np.mean(rec[10])), recall50=float(np.mean(rec[50])),
                             recall100=float(np.mean(rec[100])), top100_rel_err=float(np.mean(rel))))
        gc.collect()
        tracemalloc.start()
        ctr = Counter(keys.tolist())
        cur, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        e1[name] = dict(n_keys=int(len(ids)), exact_counter_bytes=int(cur), rows=rows)
        del ctr
    out["topk"] = e1

    # ---------------------------------------------------------------- E2 Bloom over capacity
    log("[E2] bloom over capacity")
    exact = set(pair.tolist())
    neg = []
    while len(neg) < 200_000:
        k = int(rng.integers(1, 6041)) * 4096 + int(rng.integers(1, 3953))
        if k not in exact:
            neg.append(k)
    neg = np.array(neg, dtype=np.int64)
    n0, kk = 333_334, 7
    m0 = 10 * n0
    rows = []
    for x in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0):
        q = int(x * n0)
        bf = BloomFilter(m0, kk, SEED)
        bf.add_many(pair[:q])
        rows.append(dict(load=x, inserted=q, fpr=float(bf.contains_many(neg).mean()),
                         theory=BloomFilter.theoretical_fpr(q, m0, kk), fill=bf.fill_ratio()))
    out["overcap"] = dict(design_n=n0, m=m0, k=kk, bytes=(m0 + 7) // 8, rows=rows)

    # ---------------------------------------------------------------- E3 merge
    log("[E3] merge")
    shards = np.array_split(rng.permutation(N), 4)  # 4 random shards
    halves = [np.arange(N // 2), np.arange(N // 2, N)]

    def build(idx):
        bf = BloomFilter(10 * 1_000_209, 7, SEED)
        bf.add_many(pair[idx])
        c = CountMinSketch(2048, 5, SEED)
        c.add_many(M[idx])
        return bf, c

    whole = build(np.arange(N))
    ids = np.unique(M)
    true = np.bincount(M)[ids]
    m_res = {}
    for label, parts in (("앞/뒤 2등분", halves), ("무작위 4분할", shards)):
        sk = [build(p) for p in parts]
        mb, mc = sk[0]
        for other in sk[1:]:
            mb.merge(other[0])
            mc.merge(other[1])
        merged_est = mc.estimate_many(ids)
        m_res[label] = dict(
            bloom_identical=bool(mb.bits == whole[0].bits),
            cms_identical=all(bool((a == b).all()) for a, b in zip(mc._views, whole[1]._views)),
            cms_max_abs_diff=int(np.abs(merged_est - whole[1].estimate_many(ids)).max()),
            cms_mean_abs_err_merged=float((merged_est - true).mean()),
            cms_mean_abs_err_whole=float((whole[1].estimate_many(ids) - true).mean()))
    out["merge"] = m_res

    # ---------------------------------------------------------------- E4 scaling
    log("[E4] scaling")
    rows = []
    bloom_bytes = (10 * 1_000_209 + 7) // 8
    for n in (100_000, 250_000, 500_000, 750_000, N):
        sizes = {}
        for nm, arr in (("set_pair", pair), ("counter_movie", M), ("counter_movie_month", movie_month)):
            gc.collect()
            tracemalloc.start()
            obj = set(arr[:n].tolist()) if nm == "set_pair" else Counter(arr[:n].tolist())
            cur, _ = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            sizes[nm] = int(cur)
            sizes[nm + "_keys"] = len(obj)
            del obj
        rows.append(dict(n=n, **sizes, bloom=bloom_bytes, cms=2048 * 5 * 4))
    out["scaling"] = rows

    # ---------------------------------------------------------------- E5 timing repeats
    log("[E5] timing repeats")

    def run(fn):
        gc.collect()
        t = time.perf_counter()
        fn()
        return time.perf_counter() - t

    def read_only():
        for _ in stream():
            pass

    def bloom():
        b = BloomFilter(10 * 1_000_209, 7, SEED)
        for u, m, _ in stream():
            b.add(u * 4096 + m)

    def cms(col):
        def f():
            c = CountMinSketch(2048, 5, SEED)
            for rec in stream():
                c.add(rec[col])
        return f

    def set_():
        s = set()
        for u, m, _ in stream():
            s.add(u * 4096 + m)

    def counter(col):
        def f():
            c = Counter()
            for rec in stream():
                c[rec[col]] += 1
        return f

    tasks = [("read_only", read_only), ("bloom", bloom), ("exact_set", set_), ("cms_movie", cms(1)), ("exact_counter_movie", counter(1)),
             ("cms_user", cms(0)), ("exact_counter_user", counter(0))]
    ts = {}
    for name, fn in tasks:
        v = [run(fn) for _ in range(REPEATS)]
        ts[name] = dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)), runs=v, throughput=float(N / np.mean(v)))
        log("   ", name, f"{ts[name]['mean']:.2f}s +- {ts[name]['std']:.2f}")
    out["timing"] = dict(repeats=REPEATS, results=ts)
    log("[E6] per-record cost vs k / depth")
    out["per_record"] = per_record_cost(pair, M)

    with open(os.path.join(OUT, "extra.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    figures(out)
    log("done -> outputs/extra.json, outputs/fig_extra*.png")


def figures(o):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import FuncFormatter, NullFormatter
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for fam in ("Malgun Gothic", "NanumGothic", "AppleGothic", "Noto Sans CJK KR"):
        if fam in avail:
            plt.rcParams["font.family"] = fam
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130

    def save(fig, name):
        for a in fig.axes:  # plain log tick labels (mathtext needs a U+2212 glyph that Korean fonts lack)
            for axis, sc in ((a.xaxis, a.get_xscale()), (a.yaxis, a.get_yscale())):
                if sc == "log":
                    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                    axis.set_minor_formatter(NullFormatter())
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, name))
        plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, key, title in ((ax[0], "movie", "CMS Top-K 복원율 (영화 키)"), (ax[1], "movie_month", "CMS Top-K 복원율 (영화×월 키)")):
        r = o["topk"][key]["rows"]
        for k, st in ((10, "o-"), (50, "s-"), (100, "^-")):
            a.plot([x["bytes"] for x in r], [x[f"recall{k}"] for x in r], st, label=f"Top-{k}")
        a.set_xscale("log")
        a.set_xlabel("sketch 크기 (bytes, depth=5)")
        a.set_ylabel("Top-K recall")
        a.set_title(title)
        a.legend(fontsize=8)
        a.grid(alpha=.3)
    r = o["overcap"]["rows"]
    ax[2].plot([x["load"] for x in r], [x["fpr"] for x in r], "o-", label="실측")
    ax[2].plot([x["load"] for x in r], [x["theory"] for x in r], "k--", label="이론")
    ax[2].axvline(1.0, color="r", ls=":", label="설계 용량")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("삽입 원소 수 / 설계 용량")
    ax[2].set_ylabel("FPR (로그)")
    ax[2].set_title("Bloom Filter 용량 초과")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=.3)
    save(fig, "fig_extra1.png")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    r = o["scaling"]
    n = [x["n"] for x in r]
    ax[0].plot(n, [x["set_pair"] for x in r], "o-", label="set (고유 쌍)")
    ax[0].plot(n, [x["counter_movie_month"] for x in r], "s-", label="Counter (영화×월)")
    ax[0].plot(n, [x["counter_movie"] for x in r], "^-", label="Counter (영화)")
    ax[0].plot(n, [x["bloom"] for x in r], "o--", label="Bloom (고정)")
    ax[0].plot(n, [x["cms"] for x in r], "s--", label="CMS 2048x5 (고정)")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("처리한 레코드 수")
    ax[0].set_ylabel("메모리 (bytes, 로그)")
    ax[0].set_title("스트림 길이에 따른 메모리")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=.3)
    t = o["timing"]["results"]
    names = ["read_only", "exact_set", "bloom", "exact_counter_movie", "cms_movie", "exact_counter_user", "cms_user"]
    ax[1].bar(range(len(names)), [t[k]["mean"] for k in names], yerr=[t[k]["std"] for k in names], capsize=3,
              color=["#999", "#1f77b4", "#ff7f0e", "#1f77b4", "#ff7f0e", "#1f77b4", "#ff7f0e"])
    ax[1].set_xticks(range(len(names)))
    ax[1].set_xticklabels(["읽기만", "set", "Bloom", "Counter\n(영화)", "CMS\n(영화)", "Counter\n(사용자)", "CMS\n(사용자)"], fontsize=8)
    ax[1].set_ylabel("전체 시간 (s)")
    ax[1].set_title(f"처리 시간 ({o['timing']['repeats']}회 평균±표준편차, 파일 읽기 포함)")
    ax[1].grid(alpha=.3, axis="y")
    save(fig, "fig_extra2.png")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--per-record-only":  # add only E6 to an existing extra.json
        U_, M_, T_ = load()
        path = os.path.join(OUT, "extra.json")
        saved = json.load(open(path, encoding="utf-8"))
        saved["per_record"] = per_record_cost(U_ * 4096 + M_, M_)
        json.dump(saved, open(path, "w", encoding="utf-8"), indent=1, default=float)
        log("added per_record to extra.json")
    elif len(sys.argv) > 1 and sys.argv[1] == "--figures-only":
        figures(json.load(open(os.path.join(OUT, "extra.json"), encoding="utf-8")))
    else:
        main()
