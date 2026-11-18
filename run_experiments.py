"""Experiment with Bloom Filter and Count-Min Sketch on MovieLens 1M."""

import gc
import json
import math
import os
import platform
import sys
import time
import tracemalloc
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from algorithms import BloomFilter, CountMinSketch  # noqa: E402

SEED = 42
DATA = os.path.join(HERE, "data", "ml-1m", "ratings.dat")
OUT = os.path.join(HERE, "outputs")
RESULTS = os.path.join(OUT, "results.json")
os.makedirs(OUT, exist_ok=True)

BLOOM_BITS_PER_ELEMENT = 10
BLOOM_HASHES = 7
CMS_WIDTH = 2048
CMS_DEPTH = 5


def stream():
    """Yield one (user, movie, rating) record at a time."""
    with open(DATA, "r", encoding="latin-1") as source:
        for line in source:
            user_id, movie_id, rating, _ = line.split("::")
            yield int(user_id), int(movie_id), int(rating)


def pair_key(user_id, movie_id):
    return user_id * 4096 + movie_id


def timed(function):
    gc.collect()
    started = time.perf_counter()
    result = function()
    return result, time.perf_counter() - started


def traced(function):
    gc.collect()
    tracemalloc.start()
    result = function()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, current, peak


def load_arrays():
    import pandas as pd

    frame = pd.read_csv(
        DATA,
        sep="::",
        engine="python",
        header=None,
        names=["user", "movie", "rating", "timestamp"],
    )
    return frame.user.to_numpy(np.int64), frame.movie.to_numpy(np.int64)


def representative_experiment(users, movies):
    n_records = len(users)
    _, read_seconds = timed(lambda: sum(1 for _ in stream()))
    exact_pairs = {pair_key(int(user), int(movie)) for user, movie in zip(users, movies)}
    rng = np.random.default_rng(SEED)
    positive_keys = rng.choice(
        np.fromiter(exact_pairs, dtype=np.int64), 200_000, replace=False
    )
    negative_keys = []
    while len(negative_keys) < 200_000:
        user = int(rng.integers(1, 6041))
        movie = int(rng.integers(1, 3953))
        key = pair_key(user, movie)
        if key not in exact_pairs:
            negative_keys.append(key)
    negative_keys = np.asarray(negative_keys, dtype=np.int64)

    bloom_bits = BLOOM_BITS_PER_ELEMENT * len(exact_pairs)

    def build_bloom():
        bloom = BloomFilter(bloom_bits, BLOOM_HASHES, SEED)
        for user, movie, _ in stream():
            bloom.add(pair_key(user, movie))
        return bloom

    bloom, bloom_seconds = timed(build_bloom)
    _, _, bloom_peak = traced(build_bloom)
    false_positives = sum(int(key in bloom) for key in negative_keys.tolist())
    false_negatives = sum(int(key not in bloom) for key in positive_keys.tolist())

    def build_pair_set():
        result = set()
        for user, movie, _ in stream():
            result.add(pair_key(user, movie))
        return result

    _, exact_set_seconds = timed(build_pair_set)
    _, exact_set_bytes, exact_set_peak = traced(build_pair_set)
    representative = {
        "read_only_sec": read_seconds,
        "bloom": {
            "n_unique": len(exact_pairs),
            "m_bits": bloom_bits,
            "k": BLOOM_HASHES,
            "bits_per_elem": BLOOM_BITS_PER_ELEMENT,
            "fpr": false_positives / len(negative_keys),
            "theory_fpr": BloomFilter.theoretical_fpr(
                len(exact_pairs), bloom_bits, BLOOM_HASHES
            ),
            "fnr": false_negatives / len(positive_keys),
            "fill": bloom.fill_ratio(),
            "time_stream": bloom_seconds,
            "time_net": bloom_seconds - read_seconds,
            "throughput": n_records / bloom_seconds,
            "struct_bytes": bloom.nbytes,
            "traced_peak": bloom_peak,
            "exact_time": exact_set_seconds,
            "exact_bytes": exact_set_bytes,
            "exact_peak": exact_set_peak,
            "exact_throughput": n_records / exact_set_seconds,
        },
    }

    exact_counts = {
        "movie": Counter(movies.tolist()),
        "user": Counter(users.tolist()),
    }
    for name, column_index in (("movie", 1), ("user", 0)):
        truth = exact_counts[name]

        def build_cms(column_index=column_index):
            sketch = CountMinSketch(CMS_WIDTH, CMS_DEPTH, SEED)
            for record in stream():
                sketch.add(record[column_index])
            return sketch

        sketch, cms_seconds = timed(build_cms)
        _, _, cms_peak = traced(build_cms)

        def build_counter(column_index=column_index):
            counter = Counter()
            for record in stream():
                counter[record[column_index]] += 1
            return counter

        _, exact_seconds = timed(build_counter)
        _, exact_bytes, exact_peak = traced(build_counter)
        identifiers = np.asarray(sorted(truth))
        true_counts = np.asarray([truth[key] for key in identifiers])
        estimates = np.asarray([sketch.estimate(int(key)) for key in identifiers])
        errors = estimates - true_counts
        top_100 = np.argsort(-true_counts)[:100]
        bound = math.e / CMS_WIDTH * n_records
        representative[f"cms_{name}"] = {
            "n_keys": len(identifiers),
            "width": CMS_WIDTH,
            "depth": CMS_DEPTH,
            "eps": math.e / CMS_WIDTH,
            "delta": math.exp(-CMS_DEPTH),
            "mean_abs_err": float(errors.mean()),
            "max_abs_err": int(errors.max()),
            "min_err": int(errors.min()),
            "mean_rel_err": float((errors / true_counts).mean()),
            "top100_mean_rel_err": float(
                (errors[top_100] / true_counts[top_100]).mean()
            ),
            "bound": bound,
            "frac_within_bound": float((errors <= bound).mean()),
            "time_stream": cms_seconds,
            "time_net": cms_seconds - read_seconds,
            "throughput": n_records / cms_seconds,
            "struct_bytes": sketch.nbytes,
            "traced_peak": cms_peak,
            "exact_time": exact_seconds,
            "exact_bytes": exact_bytes,
            "exact_peak": exact_peak,
            "exact_throughput": n_records / exact_seconds,
        }
    return representative, exact_pairs, positive_keys, negative_keys, exact_counts["movie"]


def parameter_sweeps(users, movies, exact_pairs, positives, negatives, movie_truth):
    keys = users.astype(np.int64) * 4096 + movies
    bloom_rows = []
    for ratio in [2, 4, 6, 8, 10, 12, 16]:
        for hashes in [1, 2, 3, 4, 5, 7, 10, 14]:
            bloom = BloomFilter(ratio * len(exact_pairs), hashes, SEED)
            bloom.add_many(keys)
            bloom_rows.append(
                {
                    "bits_per_elem": ratio,
                    "k": hashes,
                    "fpr": float(bloom.contains_many(negatives).mean()),
                    "theory": BloomFilter.theoretical_fpr(
                        len(exact_pairs), ratio * len(exact_pairs), hashes
                    ),
                    "fnr": float(1 - bloom.contains_many(positives).mean()),
                    "fill": bloom.fill_ratio(),
                    "bytes": bloom.nbytes,
                }
            )

    movie_ids = np.asarray(sorted(movie_truth))
    true_counts = np.asarray([movie_truth[key] for key in movie_ids])
    top_100 = np.argsort(-true_counts)[:100]
    cms_rows = []
    for width in [16, 32, 64, 128, 272, 512, 1024, 2048, 4096]:
        for depth in [1, 2, 3, 5, 8]:
            relative_errors, top_errors, absolute_errors = [], [], []
            for repeat in range(5):
                sketch = CountMinSketch(width, depth, SEED + repeat)
                sketch.add_many(movies)
                errors = sketch.estimate_many(movie_ids) - true_counts
                relative_errors.append(float((errors / true_counts).mean()))
                top_errors.append(float((errors[top_100] / true_counts[top_100]).mean()))
                absolute_errors.append(float(errors.mean()))
            cms_rows.append(
                {
                    "width": width,
                    "depth": depth,
                    "mean_rel_err": float(np.mean(relative_errors)),
                    "top100_rel_err": float(np.mean(top_errors)),
                    "mean_abs_err": float(np.mean(absolute_errors)),
                    "bytes": width * depth * 4,
                }
            )
    return {"bloom": bloom_rows, "cms": cms_rows, "cms_n_keys": len(movie_ids)}


def plain_log_ticks(fig):
    """Log-axis tick labels as plain numbers (mathtext needs a U+2212 glyph that Korean fonts lack)."""
    from matplotlib.ticker import FuncFormatter, NullFormatter

    for axis_obj in fig.axes:
        for axis, scale in ((axis_obj.xaxis, axis_obj.get_xscale()), (axis_obj.yaxis, axis_obj.get_yscale())):
            if scale == "log":
                axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
                axis.set_minor_formatter(NullFormatter())


def figures(representative, sweeps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for family in ("Malgun Gothic", "NanumGothic", "AppleGothic", "Noto Sans CJK KR"):
        if family in available:
            plt.rcParams["font.family"] = family
            break
    plt.rcParams["axes.unicode_minus"] = False

    bloom_rows = sweeps["bloom"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ratio in sorted({row["bits_per_elem"] for row in bloom_rows}):
        rows = [row for row in bloom_rows if row["bits_per_elem"] == ratio]
        axes[0].plot(
            [row["k"] for row in rows],
            [max(row["fpr"], 1e-6) for row in rows],
            "o-",
            label=f"{ratio} bits/element",
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("해시 함수 개수 k")
    axes[0].set_ylabel("False Positive Rate")
    axes[0].set_title("Bloom Filter 파라미터 비교")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    axes[1].scatter(
        [row["theory"] for row in bloom_rows],
        [max(row["fpr"], 1e-6) for row in bloom_rows],
        s=14,
    )
    axes[1].plot([1e-5, 1], [1e-5, 1], "r--", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("이론 FPR")
    axes[1].set_ylabel("실측 FPR")
    axes[1].set_title("이론값과 실측값 비교")
    axes[1].grid(alpha=0.3)
    plain_log_ticks(fig)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_bloom.png"), dpi=150)
    plt.close(fig)

    cms_rows = sweeps["cms"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    depths = sorted({row["depth"] for row in cms_rows})
    for depth in depths:
        rows = [row for row in cms_rows if row["depth"] == depth]
        for axis, field in zip(axes[:2], ("mean_rel_err", "top100_rel_err")):
            axis.plot(
                [row["width"] for row in rows],
                [row[field] for row in rows],
                "o-",
                label=f"depth={depth}",
            )
        axes[2].plot(
            [row["bytes"] for row in rows],
            [row["mean_rel_err"] for row in rows],
            "o-",
            label=f"depth={depth}",
        )
    for axis, title in zip(axes[:2], ("전체 영화 평균 상대오차", "상위 100개 영화 상대오차")):
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("width")
        axis.set_ylabel("상대오차")
        axis.set_title(title)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Sketch 크기(bytes)")
    axes[2].set_ylabel("평균 상대오차")
    axes[2].set_title("메모리-정확도 trade-off")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=7)
    plain_log_ticks(fig)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_cms.png"), dpi=150)
    plt.close(fig)

    names = ["Bloom Filter", "Count-Min Sketch"]
    streaming_memory = [representative["bloom"]["struct_bytes"], representative["cms_movie"]["struct_bytes"]]
    exact_memory = [representative["bloom"]["exact_bytes"], representative["cms_movie"]["exact_bytes"]]
    streaming_speed = [representative["bloom"]["throughput"], representative["cms_movie"]["throughput"]]
    exact_speed = [representative["bloom"]["exact_throughput"], representative["cms_movie"]["exact_throughput"]]
    x = np.arange(2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(x - 0.18, exact_memory, 0.36, label="Exact")
    axes[0].bar(x + 0.18, streaming_memory, 0.36, label="Streaming")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("bytes")
    axes[0].set_title("메모리 사용량")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(x - 0.18, exact_speed, 0.36, label="Exact")
    axes[1].bar(x + 0.18, streaming_speed, 0.36, label="Streaming")
    axes[1].set_xticks(x, names)
    axes[1].set_ylabel("records/second")
    axes[1].set_title("처리량")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")
    plain_log_ticks(fig)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_headline.png"), dpi=150)
    plt.close(fig)


def main():
    users, movies = load_arrays()
    representative, exact_pairs, positives, negatives, movie_truth = representative_experiment(users, movies)
    sweeps = parameter_sweeps(users, movies, exact_pairs, positives, negatives, movie_truth)
    result = {
        "env": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "n_records": len(users),
            "seed": SEED,
            "platform": platform.platform(),
            "cpu": platform.processor(),
        },
        "A": representative,
        "B": sweeps,
    }
    with open(RESULTS, "w", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2, ensure_ascii=False)
    try:
        figures(representative, sweeps)
        print("done -> outputs/results.json and three figures")
    except ModuleNotFoundError as error:
        if error.name != "matplotlib":
            raise
        print("done -> outputs/results.json (matplotlib unavailable, figures kept unchanged)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--figures-only":
        with open(RESULTS, encoding="utf-8") as source:
            saved = json.load(source)
        figures(saved["A"], saved["B"])
    else:
        main()
