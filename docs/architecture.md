# Architecture

```
core/            C++17 library (compiled as an object library into the Python module)
  include/vc/    image.h graph.h expr.h roto.h io.h color.h util.h
  src/           image, graph engine + cache, expressions, roto rasterizer, I/O, colour, thread pool
  src/ops/       node implementations, grouped by category
  third_party/   tinyexr (patched, see decisions.md), stb_image, stb_image_write
bindings/        nanobind module `vertexcomp._vccore`
app/vertexcomp/  Python package: project model, CLI renderer, movie I/O, PySide6 UI
tests/           core/ (C++), py/ (pytest + pytest-qt), data/golden.json
tools/           benchmark, demo generator, node reference generator
```

## Image model

* Planar float32, one plane per channel over the image **bbox** (data window). Pixels outside are 0.
* Coordinates: integer grid, origin top-left of the format, **y down**. Pixel (x, y) has its centre at
  (x + 0.5, y + 0.5). Boxes are half-open.
* Colour is scene-linear in the working space (Linear Rec.709 or ACEScg) and **premultiplied** by `rgba.alpha`.
* Channels are named `layer.channel` (`rgba.red`, `diffuse.green`, `depth.Z`, `N.x`). EXR names map as
  `R/G/B/A -> rgba.*`, `Z -> depth.Z`, `layer.R -> layer.red`. Multipart EXR parts without layer prefixes
  get the part name as layer.
* `Info` per node and frame: format, bbox, channel list, frame range, metadata.

## Evaluation

* Pull model: `Engine::render(node, frame, roi, channels)` returns an image whose bbox contains
  `roi ∩ info.bbox`. Each op asks its inputs only for what it needs (Blur grows the ROI by its kernel radius,
  Transform maps the ROI through the inverse matrix, Merge passes it through). Rendering the visible area of a
  4K frame therefore only computes that area.
* Ops are classes with `info()`, `render()`, optional `inputFrames()` (temporal ops: TimeOffset, FrameHold,
  Retime request other frames) and `hashExtra()` (Read adds file mtime, Roto its shapes, Transform its
  motion-blur samples).
* **Hash invalidation**: a node's hash at a frame = op type and version + all knob values evaluated at that frame
  + hashes of the inputs at the frames it requests. Changing a knob changes only that node's hash and everything
  downstream. Hashes do not depend on node names or ids, so undo/redo (which rebuilds the graph from a snapshot)
  keeps the cache valid.
* **Cache**: LRU of rendered images keyed by hash, with a byte limit (Settings > Performance). An entry is reused
  when its bbox contains the requested ROI and it holds the requested channels. A separate small cache avoids
  decoding a file twice between `info()` and `render()`.
* **Threading**: a work-sharing pool (`ThreadPool::parallelFor`, the caller also works, so nested use cannot
  deadlock). Ops parallelise over rows; results do not depend on the thread count (tested). Every request carries
  a `CancelToken` checked between row chunks; the viewer cancels stale requests.
* **Thread safety**: knob values and connections are guarded by the graph mutex. A render snapshots the node's
  parameters and inputs under the lock and then works without it, so the UI can edit while workers render.
  `Node` derives from `enable_shared_from_this` so the bindings never create shared_ptrs whose deleter needs the GIL.
* **Errors**: an exception in an op marks that node (`node.error`) and propagates as `NodeError("Name: message")`;
  downstream nodes do not overwrite the root cause. The viewer shows the message over a black frame.

## Knobs and animation

Knob metadata (type, components, default, soft range, enum options, group, tooltip) is declared next to the op
and drives the properties panel. Numeric components can hold keyframes (constant, linear, smooth Catmull-Rom,
bezier with slopes) and expressions compiled by a small safe evaluator (`expr.cpp`, no Python `eval`).

## Python side

* `project.py` — the C++ graph plus UI metadata (positions, colours, labels, backdrops, viewer state) and the
  `.vcp` format. Shared by the UI and `vertexcomp-render`.
* `ui/session.py` — selection, time, undo (`QUndoStack`; knob edits are granular and merge while dragging,
  structural edits are snapshots), viewer render worker and playback read-ahead cache.
* `ui/viewer.py` — `QOpenGLWidget` (GL 3.3 core). The core hands over an (H, W, 4) float buffer of the visible
  area (box-averaged when zoomed out); exposure, gain, gamma, channel view, display transform, zebra and pixel
  grid are shader uniforms. View transform math is in Python doubles.
* `ui/nodegraph.py` — custom `QGraphicsScene/View`.
* `ui/properties.py` — widgets generated from knob metadata.
* Workers are plain Python threads that release the GIL inside the core; results reach the UI thread through a
  queue polled by a timer, never through Qt calls from the worker.

## Adding a node

Create `core/src/ops/mynode.cpp`:

```cpp
#include "common.h"
namespace vc::ops {
namespace {
class Brightness : public PixelOp {            // PixelOp gives channels, unpremult, mask and mix
    RowKernel kernel(const Params& p, const Info&) const override {
        float k = float(p.f("amount"));
        return [k](float* r, float* g, float* b, float*, int n, int, int) {
            for (int i = 0; i < n; ++i) { r[i] *= k; g[i] *= k; b[i] *= k; }
        };
    }
};
VC_REGISTER_OP(OpDesc{"Brightness", "Color", "Multiply rgb.", {"", "mask"}, 0,
                      concat(concat(pixelOpKnobs("rgb", false), {K::flt("amount", "amount", 1, 0, 4)}),
                             maskKnobs()),
                      1, maker<Brightness>()});
}  // namespace
}  // namespace vc::ops
```

Rebuild (`pip install .` or the CMake developer build). The node appears in the Tab menu, the Nodes menu and the
properties panel with no UI code. Ops that need neighbourhoods implement `Op` directly and grow the ROI they request
(see `filter.cpp`).

## Dependencies

| Needed for | Used | Instead of (spec) |
|---|---|---|
| EXR (scanline, tiled, multipart, multichannel) | tinyexr (vendored) | OpenImageIO |
| PNG/JPG/TGA/BMP/HDR | stb_image / stb_image_write (vendored), own 16-bit PNG writer | OpenImageIO |
| Movies | FFmpeg binary from `imageio-ffmpeg` (pip) | libav* linked into the core |
| Colour | built-in transfer functions + primaries matrices | OpenColorIO |
| Threads | own pool | TBB |
| Bindings | nanobind | — |

The reasons are in `decisions.md`.
