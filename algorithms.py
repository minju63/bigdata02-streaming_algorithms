"""Bloom Filter and Count-Min Sketch implemented from their core logic.

No dedicated streaming-algorithm package is used. NumPy is used only for the
vectorised parameter sweep; the representative benchmark calls ``add`` once
per input record.
"""

import math
from array import array

import numpy as np

MASK64 = (1 << 64) - 1
_C1 = 0x9E3779B97F4A7C15
_C2 = 0xBF58476D1CE4E5B9
_C3 = 0x94D049BB133111EB


def splitmix64(x):
    """Mix an integer into a deterministic 64-bit hash value."""
    x = (x + _C1) & MASK64
    x = ((x ^ (x >> 30)) * _C2) & MASK64
    x = ((x ^ (x >> 27)) * _C3) & MASK64
    return x ^ (x >> 31)


def splitmix64_np(x):
    """Vectorised equivalent of :func:`splitmix64`."""
    with np.errstate(over="ignore"):
        x = x.astype(np.uint64) + np.uint64(_C1)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(_C2)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(_C3)
        return x ^ (x >> np.uint64(31))


class BloomFilter:
    """Bloom Filter backed by an m-bit array and k double hashes."""

    def __init__(self, m, k, seed=0):
        self.m = int(m)
        self.k = int(k)
        self.bits = bytearray((self.m + 7) // 8)
        self._arr = np.frombuffer(self.bits, dtype=np.uint8)
        self._seed = splitmix64(seed + 12345)

    @property
    def nbytes(self):
        return len(self.bits)

    def _hashes(self, key):
        h1 = splitmix64(int(key) ^ self._seed)
        h2 = splitmix64(h1) | 1
        return h1, h2

    def add(self, key):
        h1, h2 = self._hashes(key)
        for i in range(self.k):
            idx = ((h1 + i * h2) & MASK64) % self.m
            self.bits[idx >> 3] |= 1 << (idx & 7)

    def __contains__(self, key):
        h1, h2 = self._hashes(key)
        for i in range(self.k):
            idx = ((h1 + i * h2) & MASK64) % self.m
            if not self.bits[idx >> 3] & (1 << (idx & 7)):
                return False
        return True

    def _hash_many(self, keys):
        keys = np.asarray(keys, dtype=np.uint64)
        h1 = splitmix64_np(keys ^ np.uint64(self._seed))
        h2 = splitmix64_np(h1) | np.uint64(1)
        return h1, h2

    def add_many(self, keys):
        h1, h2 = self._hash_many(keys)
        for i in range(self.k):
            with np.errstate(over="ignore"):
                idx = (h1 + np.uint64(i) * h2) % np.uint64(self.m)
            byte = (idx >> np.uint64(3)).astype(np.int64)
            bit = (idx & np.uint64(7)).astype(np.uint8)
            for value in range(8):
                selected = bit == value
                if selected.any():
                    self._arr[byte[selected]] |= np.uint8(1 << value)

    def contains_many(self, keys):
        h1, h2 = self._hash_many(keys)
        result = np.ones(len(h1), dtype=bool)
        for i in range(self.k):
            with np.errstate(over="ignore"):
                idx = (h1 + np.uint64(i) * h2) % np.uint64(self.m)
            byte = (idx >> np.uint64(3)).astype(np.int64)
            bit = (idx & np.uint64(7)).astype(np.uint8)
            result &= ((self._arr[byte] >> bit) & 1) == 1
        return result

    def merge(self, other):
        """Union of two filters built with the same m, k and seed: bitwise OR."""
        assert (self.m, self.k, self._seed) == (other.m, other.k, other._seed)
        np.bitwise_or(self._arr, other._arr, out=self._arr)

    def fill_ratio(self):
        return float(np.unpackbits(self._arr).sum()) / (8 * len(self.bits))

    @staticmethod
    def theoretical_fpr(n, m, k):
        return (1.0 - math.exp(-k * n / m)) ** k


class CountMinSketch:
    """Count-Min Sketch with a depth x width matrix of 32-bit counters."""

    def __init__(self, width, depth, seed=0):
        self.w = int(width)
        self.d = int(depth)
        self._rows = [array("I", bytes(4 * self.w)) for _ in range(self.d)]
        self._views = [np.frombuffer(row, dtype=np.uint32) for row in self._rows]
        self._seeds = [splitmix64(seed * 1000 + row + 1) for row in range(self.d)]
        self.total = 0

    @property
    def nbytes(self):
        return self.d * self.w * 4

    def add(self, key):
        key = int(key)
        for row in range(self.d):
            index = splitmix64(key ^ self._seeds[row]) % self.w
            self._rows[row][index] += 1
        self.total += 1

    def estimate(self, key):
        key = int(key)
        return min(
            self._rows[row][splitmix64(key ^ self._seeds[row]) % self.w]
            for row in range(self.d)
        )

    def merge(self, other):
        """Sketch of the concatenated streams: add the counters (same width, depth, seeds)."""
        assert (self.w, self.d, self._seeds) == (other.w, other.d, other._seeds)
        for row in range(self.d):
            self._views[row] += other._views[row]
        self.total += other.total

    def _hash_row(self, keys, row):
        hashes = splitmix64_np(keys ^ np.uint64(self._seeds[row]))
        return (hashes % np.uint64(self.w)).astype(np.int64)

    def add_many(self, keys):
        keys = np.asarray(keys, dtype=np.uint64)
        for row in range(self.d):
            counts = np.bincount(self._hash_row(keys, row), minlength=self.w)
            self._views[row] += counts.astype(np.uint32)
        self.total += len(keys)

    def estimate_many(self, keys):
        keys = np.asarray(keys, dtype=np.uint64)
        result = None
        for row in range(self.d):
            values = self._views[row][self._hash_row(keys, row)].astype(np.int64)
            result = values if result is None else np.minimum(result, values)
        return result
