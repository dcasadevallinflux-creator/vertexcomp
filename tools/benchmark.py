"""Performance measurements for docs/performance.md.

python tools/benchmark.py [--soak SECONDS]
"""
import argparse
import os
import resource
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from vertexcomp import _vccore as vc  # noqa: E402

RGBA = ["rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"]
W, H = 2048, 1080


def timed(fn, n=3):
    best = 1e9
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000


def rss_mb():
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except OSError:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soak", type=float, default=0)
    args = ap.parse_args()
    g = vc.Graph()
    eng = vc.Engine(g, 2 << 30)
    src = g.create("Noise")
    src.set("format", [W, H])
    src.set("octaves", 1)
    eng.render(src, 1, None, RGBA)  # cached upstream
    print(f"threads: {vc.thread_count()}  frame: {W}x{H} float RGBA")

    gb = g.create("GaussianBlur")
    g.connect(gb, 0, src)
    gb.set("sigma", [20.0, 20.0])
    for m in ("fast (3 box)", "exact"):
        gb.set("method", m)
        ms = timed(lambda: (eng.clear_cache(), eng.render(src, 1, None, RGBA), None) and None or
                   eng.render(gb, 1, (0, 0, W, H), RGBA), 2)
        print(f"GaussianBlur sigma 20 [{m}]: {ms:.0f} ms (includes re-render of a 1-octave noise source)")

    # Isolate filter cost: source cached, only the blur re-renders.
    def blur_only():
        gb.set("sigma", [20.0 + np.random.rand() * 1e-3] * 2)
        eng.render(gb, 1, (0, 0, W, H), RGBA)
    gb.set("method", "fast (3 box)")
    print(f"GaussianBlur sigma 20 [fast], cached input: {timed(blur_only):.0f} ms")

    b = g.create("Constant")
    b.set("format", [W, H])
    b.set("color", [0.2, 0.3, 0.4, 1.0])
    m = g.create("Merge")
    g.connect(m, 0, b)
    g.connect(m, 1, src)

    def merge_only():
        m.set("mix", 1.0 - np.random.rand() * 1e-6)
        eng.render(m, 1, (0, 0, W, H), RGBA)
    print(f"Merge over, cached inputs: {timed(merge_only):.0f} ms")

    prev = src
    chain = []
    for _i in range(10):
        cc = g.create("ColorCorrect")
        g.connect(cc, 0, prev)
        chain.append(cc)
        prev = cc

    def chain_edit():
        chain[0].set("saturation", 1.0 + np.random.rand() * 1e-3)
        eng.render(prev, 1, (0, 0, W, H), RGBA)
    print(f"10 x ColorCorrect chain, edit at the top: {timed(chain_edit):.0f} ms")

    def chain_view_step4():
        chain[0].set("saturation", 1.0 + np.random.rand() * 1e-3)
        eng.fetch(prev, 1, (0, 0, W, H), RGBA, 2)
    print(f"  same, viewer fetch at 1/2 display: {timed(chain_view_step4):.0f} ms")

    # EXR sequence decode
    tmp = tempfile.mkdtemp()
    data = np.random.rand(4, H, W).astype(np.float32)
    for f in range(1, 6):
        vc.write_image(os.path.join(tmp, f"s.{f:04d}.exr"), data, RGBA, (0, 0, W, H), (W, H), "exr", "16 bit half", "zip")
    r = g.create("Read")
    r.set("file", os.path.join(tmp, "s.####.exr"))
    r.set("first", 1)
    r.set("last", 5)
    t0 = time.perf_counter()
    for f in range(1, 6):
        eng.render(r, f, None, RGBA)
    print(f"EXR 2K half ZIP decode: {(time.perf_counter() - t0) / 5 * 1000:.0f} ms/frame")
    t0 = time.perf_counter()
    for f in range(1, 6):
        eng.fetch(r, f, (0, 0, W, H), RGBA, 1)
    print(f"Cached 2K frame to display buffer: {(time.perf_counter() - t0) / 5 * 1000:.1f} ms/frame "
          f"({5 / (time.perf_counter() - t0):.0f} fps ceiling)")

    if args.soak:
        eng.set_cache_limit(512)
        n = g.create("Noise")
        n.set("format", [W, H])
        n.set("octaves", 1)
        n.set("speed", 0.1)
        t0 = time.perf_counter()
        samples = []
        f = 0
        while time.perf_counter() - t0 < args.soak:
            f = f % 200 + 1
            eng.fetch(n, f, (0, 0, W, H), RGBA, 2)
            samples.append((time.perf_counter() - t0, rss_mb()))
        cs = eng.cache_stats()
        print(f"soak {args.soak:.0f}s, {len(samples)} frames, cache limit 512 MB: RSS start {samples[0][1]:.0f} MB, "
              f"max {max(s[1] for s in samples):.0f} MB, end {samples[-1][1]:.0f} MB, cache used {cs['used'] / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
