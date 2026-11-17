# Performance

Measured with `python tools/benchmark.py --soak 120` on the build machine: **2 vCPU** (Intel Xeon @ 2.1 GHz,
virtualised), GCC 13 `-O3`, no SIMD intrinsics beyond what the compiler vectorises. The targets in the
specification assume an 8-core desktop; the work parallelises over rows, so expect roughly 3–4x lower times
there. Run the script on your machine to get your numbers.

| Measurement (2048x1080 float RGBA) | 2 vCPU | Target (8 cores) |
|---|---|---|
| GaussianBlur sigma 20, fast (3-box), input cached | 202 ms | |
| GaussianBlur sigma 20, fast, including re-rendering a noise source | 537 ms | |
| GaussianBlur sigma 20, exact separable (121 taps) incl. source | 843 ms | |
| Merge over, inputs cached | 39 ms | |
| 10 ColorCorrect chain, edit at the top (all 10 re-render) | 348 ms | < 100 ms |
| same, fetched for a zoomed-out viewer (1/2) | 262 ms | |
| EXR 2K half ZIP decode (first hit, background thread) | 94 ms / frame | |
| Cached frame to viewer buffer | 12.4 ms (81 fps ceiling) | 24 fps playback |

Viewer pan, zoom, exposure, gamma and channel changes do not touch the graph: they are shader uniforms on the
texture already on the GPU.

## Memory soak

120 s of continuous rendering of an animated 2K noise (7082 frames) with the cache limited to 512 MB:
RSS start 2215 MB, maximum 2216 MB, end 2216 MB; cache usage stayed at 506 MB. No growth over the run.
(The start value includes earlier parts of the benchmark that use a 2 GB cache.)

## What was optimised and what is left

* Per-pixel channel lookups by name were replaced by row reads (`Image::readRow`); this took Merge from 444 to
  39 ms and the colour chain from 6.4 s to 0.35 s.
* ColorCorrect skips neutral stages (pow() dominates its cost).
* Large gaussian blurs use three running-box passes (O(1) per pixel); the exact kernel is available.
* Not done yet: explicit SIMD kernels (Highway/xsimd), tile-level caching (the cache stores whole ROI renders),
  fusing consecutive per-pixel colour ops.
