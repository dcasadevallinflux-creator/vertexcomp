# Vertex Comp

Node-based compositor for VFX work. The pixel engine is a C++17 library (graph evaluation with regions of
interest, hash-based cache, multithreaded ops, roto rasterizer, EXR/PNG/JPG I/O); the interface is Python +
PySide6 (OpenGL viewer, node graph, properties, timeline). Python never loops over pixels.

52 node types: Read/Write (EXR multilayer and multipart, image sequences, MOV/MP4), Grade, ColorCorrect,
Exposure, HueShift, HueCorrect, Saturation, Colorspace, Expression, Merge (24 operations, eight image inputs), MergeLayer, Shuffle,
Copy, ChannelSwitch, Keyer (luma/channel/AOV/saturation/hue/colour distance/screen), Despill, DilateErode,
EdgeBlur, EdgeDetect, MaskBlur, Blur, GaussianBlur, Sharpen, Median, Vignette, LensFlare, Distortion,
Noise and film grain, Transform (motion blur and interactive viewer controls), Crop,
Reformat, Roto (animated Bezier shapes, variable feather), TimeOffset, FrameHold, Retime and more.
See `docs/node-reference.md`.

## Install

### Windows 10/11

Requirements, once:

1. **Python 3.11 or newer** (python.org installer, tick *Add python.exe to PATH*).
2. **Visual Studio 2022 Build Tools** with the *Desktop development with C++* workload
   (https://visualstudio.microsoft.com/visual-cpp-build-tools/). CMake and Ninja come from pip; nothing else is
   needed (no vcpkg).

Then, in the unzipped folder, double-click `install_windows.bat` (or run it from a terminal). It creates a
virtual environment in `.venv`, compiles the core and installs the app (2–5 minutes). Start it with
`run_windows.bat`, or:

```bat
.venv\Scripts\vertexcomp.exe
.venv\Scripts\vertexcomp-render.exe project.vcp --first 1001 --last 1100 --write Write1
```

### Linux / macOS

Needs a C++17 compiler (gcc 9+, clang 10+, Xcode CLT) and Python 3.10+.

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install .            # compiles the core with scikit-build-core + nanobind
vertexcomp               # GUI
vertexcomp-render demo/demo.vcp
```

### Try it with demo material

```sh
python tools/make_demo.py demo        # multilayer EXR sequence + MP4 + a project using the main nodes
vertexcomp demo/demo.vcp
```

## Development build

```sh
pip install nanobind scikit-build-core PySide6 numpy imageio-ffmpeg pytest pytest-qt scipy
cmake -S . -B build -DVC_BUILD_TESTS=ON && cmake --build build -j     # puts _vccore next to app/vertexcomp
build/vc_tests                                                        # C++ tests
python -m pytest tests/py                                             # core, every node, UI (needs a display)
python -m vertexcomp                                                  # run from the source tree (PYTHONPATH=app)
python tools/benchmark.py --soak 60
```

On Linux CI without a display, run the UI tests under `xvfb-run`.

## Using it

* **Node graph** (bottom): `Tab` opens a fuzzy node search (`gau` + Enter = GaussianBlur, connected to the
  selection). Hotkeys: M Merge, B Blur, C ColorCorrect, G Grade, R Read, W Write, T Transform, O Roto,
  H Shuffle, K Keyer, `.` Dot, D disable, N rename, Ctrl+B backdrop, Ctrl+G subgroup selected nodes, L align,
  Ctrl+C/V (as text, pasteable into a text editor and back), Ctrl+D duplicate. `1`–`9` send the selected node to
  a viewer input; Ctrl+click views a node. Drag from an output to empty space to create and connect a node; drag
  an arrow end to re-wire or detach. Right-click a subgroup to collapse or expand it. Right-click a Read:
  *Create Shuffle per layer*.
* **Viewer**: wheel zooms at the cursor (1/64x to 100000x), middle-drag or Alt-drag pans, F fits. Exposure,
  gamma and gain fields scrub with a drag; `[` and `]` step exposure. Layer menu lists every AOV at the viewed
  node; Shift+R/G/B/A/Y/C switch channels. The pixel grid appears at high zoom, numeric values inside pixels at
  extreme zoom. Ctrl+drag samples an area (min/max/mean). Zebra, overlays, ROI and A/B wipe are in the bar.
  Select a Transform node and enable *Transform* in the viewer bar to move, scale, or rotate it with on-image
  handles. Roto already includes its own viewer editing tools.
* **Timeline**: Space play/stop, Left/Right step, Shift+Left/Right jump between keys, Home/End. Real-time
  playback skips frames that are not ready; *every frame* waits. The green bar marks cached frames.
* **Properties**: every animatable knob has a key button (click: set/remove key at this frame; right-click:
  interpolation, expression, curve editor). Numeric fields scrub with a left-drag (Shift x10, Ctrl x0.1).
* **Roto**: select the Roto node; tools appear in its properties. See `docs/node-reference.md`.
* **Settings** (Edit > Settings, Ctrl+,): cache size, threads, read-ahead, working space, viewer options, node
  colours, hotkeys (with conflict check, import/export), theme.
* **Python** dock: `create_node`, `connect`, `set_knob`, `render`, `project`, `engine`, `vc`.

## Layout

```
core/        C++ library        bindings/   nanobind module      app/vertexcomp/   Python app
tests/       C++ and Python     tools/      benchmark, demo       docs/            architecture, nodes,
                                                                                   file format, decisions,
                                                                                   performance
```

## Status

What is not implemented, and why, is listed in `docs/decisions.md` (notably: TIFF/DPX, OCIO configs,
ChannelSplit with several outputs, node groups, C++ plugin loading). Measured performance is in
`docs/performance.md`.

---

### Inicio rápido (español)

1. Instala Python 3.11+ y *Visual Studio 2022 Build Tools* con la carga "Desarrollo para el escritorio con C++".
2. Ejecuta `install_windows.bat` (compila el core e instala la app en `.venv`).
3. `python tools\make_demo.py demo` crea material de prueba (secuencia EXR multicapa + MP4 y un proyecto).
4. Abre con `run_windows.bat demo\demo.vcp`.
