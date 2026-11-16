"""Streaming algorithms implemented from scratch (no sketch libraries).

Bloom Filter, Count-Min Sketch, HyperLogLog, Reservoir Sampling (Algorithm R / L).
Every structure has a per-record `add()` (true streaming, used for the headline
benchmark) and a vectorised `add_many()` (same math, used for the large parameter
sweeps).  Both use the same hash so results are identical (checked in run_experiments.py).
"""
import math
import random
from array import array

import numpy as np

MASK64 = (1 << 64) - 1
_C1, _C2, _C3 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB


def splitmix64(x):
    """64-bit integer mixing function (scalar, Python int)."""
    x = (x + _C1) & MASK64
    x = ((x ^ (x >> 30)) * _C2) & MASK64
    x = ((x ^ (x >> 27)) * _C3) & MASK64
    return x ^ (x >> 31)


def splitmix64_np(x):
    """Same function, vectorised on a uint64 array."""
    with np.errstate(over="ignore"):
        x = x.astype(np.uint64) + np.uint64(_C1)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(_C2)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(_C3)
        return x ^ (x >> np.uint64(31))


# ----------------------------------------------------------------- Bloom filter
class BloomFilter:
    """m-bit array, k hash functions via double hashing: idx_i = (h1 + i*h2) mod m."""

    def __init__(self, m, k, seed=0):
        self.m, self.k, self.seed = int(m), int(k), seed
        self.bits = bytearray((self.m + 7) // 8)
        self._arr = np.frombuffer(self.bits, dtype=np.uint8)  # shares memory
        self._s = splitmix64(seed + 12345)

    @property
    def nbytes(self):
        return len(self.bits)

    def _h(self, key):
        h1 = splitmix64(key ^ self._s)
        h2 = splitmix64(h1) | 1
        return h1, h2

    def add(self, key):
        h1, h2 = self._h(key)
        bits, m = self.bits, self.m
        for i in range(self.k):
            idx = ((h1 + i * h2) & MASK64) % m
            bits[idx >> 3] |= 1 << (idx & 7)

    def __contains__(self, key):
        h1, h2 = self._h(key)
        bits, m = self.bits, self.m
        for i in range(self.k):
            idx = ((h1 + i * h2) & MASK64) % m
            if not bits[idx >> 3] & (1 << (idx & 7)):
                return False
        return True

    def _idx_np(self, keys, i, h1, h2):
        with np.errstate(over="ignore"):
            return (h1 + np.uint64(i) * h2) % np.uint64(self.m)

    def _hash_np(self, keys):
        h1 = splitmix64_np(keys.astype(np.uint64) ^ np.uint64(self._s))
        h2 = splitmix64_np(h1) | np.uint64(1)
        return h1, h2

    def add_many(self, keys):
        h1, h2 = self._hash_np(np.asarray(keys))
        for i in range(self.k):
            idx = self._idx_np(keys, i, h1, h2)
            byte = (idx >> np.uint64(3)).astype(np.int64)
            bit = (idx & np.uint64(7)).astype(np.uint8)
            for b in range(8):
                sel = bit == b
                if sel.any():
                    self._arr[byte[sel]] |= np.uint8(1 << b)

    def contains_many(self, keys):
        h1, h2 = self._hash_np(np.asarray(keys))
        out = np.ones(len(h1), dtype=bool)
        for i in range(self.k):
            idx = self._idx_np(keys, i, h1, h2)
            byte = (idx >> np.uint64(3)).astype(np.int64)
            bit = (idx & np.uint64(7)).astype(np.uint8)
            out &= (self._arr[byte] >> bit) & 1 == 1
        return out

    def fill_ratio(self):
        return float(np.unpackbits(self._arr).sum()) / (8 * len(self.bits))

    @staticmethod
    def theoretical_fpr(n, m, k):
        return (1.0 - math.exp(-k * n / m)) ** k


# -------------------------------------------------------------- Count-Min sketch
class CountMinSketch:
    """depth x width counter matrix (32-bit counters); row j has its own hash;
    estimate = min over rows (never underestimates)."""

    def __init__(self, width, depth, seed=0):
        self.w, self.d = int(width), int(depth)
        self._rows = [array("I", bytes(4 * self.w)) for _ in range(self.d)]
        self._views = [np.frombuffer(r, dtype=np.uint32) for r in self._rows]  # share memory
        self._seeds = [splitmix64(seed * 1000 + j + 1) for j in range(self.d)]
        self.total = 0

    @property
    def nbytes(self):
        return self.d * self.w * 4

    def add(self, key):
        for j in range(self.d):
            self._rows[j][splitmix64(key ^ self._seeds[j]) % self.w] += 1
        self.total += 1

    def estimate(self, key):
        return min(self._rows[j][splitmix64(key ^ self._seeds[j]) % self.w] for j in range(self.d))

    def _hash_row(self, keys, j):
        h = splitmix64_np(keys ^ np.uint64(self._seeds[j])) % np.uint64(self.w)
        return h.astype(np.int64)

    def add_many(self, keys):
        keys = np.asarray(keys).astype(np.uint64)
        for j in range(self.d):
            self._views[j] += np.bincount(self._hash_row(keys, j), minlength=self.w).astype(np.uint32)
        self.total += len(keys)

    def estimate_many(self, keys):
        keys = np.asarray(keys).astype(np.uint64)
        est = None
        for j in range(self.d):
            v = self._views[j][self._hash_row(keys, j)].astype(np.int64)
            est = v if est is None else np.minimum(est, v)
        return est


# -------------------------------------------------------------------- HyperLogLog
class HyperLogLog:
    """2^p registers, each keeps the max rho (position of first 1-bit) of its sub-stream."""

    def __init__(self, p, seed=0):
        self.p, self.m = int(p), 1 << int(p)
        self.reg = bytearray(self.m)  # one byte/register (6 bits would suffice)
        self._s = splitmix64(seed + 777)
        self._maxrho = 64 - self.p + 1
        a = {16: 0.673, 32: 0.697, 64: 0.709}
        self.alpha = a.get(self.m, 0.7213 / (1 + 1.079 / self.m))

    @property
    def nbytes(self):
        return self.m

    def add(self, key):
        h = splitmix64(key ^ self._s)
        idx = h >> (64 - self.p)
        w = (h << self.p) & MASK64
        rho = 64 - w.bit_length() + 1 if w else self._maxrho
        if rho > self._maxrho:
            rho = self._maxrho
        if rho > self.reg[idx]:
            self.reg[idx] = rho

    def add_many(self, keys):
        h = splitmix64_np(np.asarray(keys).astype(np.uint64) ^ np.uint64(self._s))
        idx = (h >> np.uint64(64 - self.p)).astype(np.int64)
        v = (h << np.uint64(self.p)) >> np.uint64(11)  # top 53 bits: exact in float64
        _, e = np.frexp(v.astype(np.float64))  # e == bit_length(v) (0 for v==0)
        rho = np.minimum(53 - e.astype(np.int64) + 1, self._maxrho).astype(np.uint8)
        arr = np.frombuffer(self.reg, dtype=np.uint8)
        np.maximum.at(arr, idx, rho)

    def estimate(self):
        reg = np.frombuffer(self.reg, dtype=np.uint8)
        e = self.alpha * self.m * self.m / np.sum(2.0 ** (-reg.astype(np.float64)))
        zeros = int(np.count_nonzero(reg == 0))
        if e <= 2.5 * self.m and zeros:  # small-range correction (linear counting)
            return self.m * math.log(self.m / zeros)
        return e

    def merge(self, other):
        a, b = np.frombuffer(self.reg, dtype=np.uint8), np.frombuffer(other.reg, dtype=np.uint8)
        np.maximum(a, b, out=a)

    @staticmethod
    def std_error(p):
        return 1.04 / math.sqrt(1 << p)


# ------------------------------------------------------------- Reservoir sampling
class ReservoirSampler:
    """Uniform sample of k items from a stream of unknown length.

    algo='R': Vitter's Algorithm R (one random number per item).
    algo='L': Li's Algorithm L (random numbers only when a replacement happens).
    """

    def __init__(self, k, seed=0, algo="R"):
        self.k, self.algo = int(k), algo
        self.rng = random.Random(seed)
        self.sample, self.n = [], 0
        self._next = None
        self._W = None

    def add(self, item):
        self.n += 1
        if len(self.sample) < self.k:
            self.sample.append(item)
            if self.algo == "L" and len(self.sample) == self.k:
                self._W = math.exp(math.log(self.rng.random()) / self.k)
                self._advance()
            return
        if self.algo == "R":
            j = self.rng.randrange(self.n)
            if j < self.k:
                self.sample[j] = item
        elif self.n == self._next:
            self.sample[self.rng.randrange(self.k)] = item
            self._W *= math.exp(math.log(self.rng.random()) / self.k)
            self._advance()

    def _advance(self):
        self._next = self.n + int(math.log(self.rng.random()) / math.log(1 - self._W)) + 1

    @property
    def nbytes(self):
        return self.k * 8  # k 64-bit slots

    @staticmethod
    def sample_indices_L(N, k, rng):
        """Algorithm L without touching skipped items: O(k(1+log(N/k))) work."""
        idx = list(range(k))
        W = math.exp(math.log(rng.random()) / k)
        i = k - 1
        while True:
            i += int(math.log(rng.random()) / math.log(1 - W)) + 1
            if i >= N:
                return idx
            idx[rng.randrange(k)] = i
            W *= math.exp(math.log(rng.random()) / k)
