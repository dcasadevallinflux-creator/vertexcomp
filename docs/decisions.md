# Decisions and deviations from the specification

Each entry: what the specification asked, what was done, why.

## Dependencies: vendored readers instead of OpenImageIO / OCIO / FFmpeg libraries

The build must work with `pip install .` on a Windows machine without a package manager. OIIO + OCIO + FFmpeg
through vcpkg take hours to build and gigabytes of disk. This version vendors tinyexr and stb (header-only) and
uses the FFmpeg executable shipped in the `imageio-ffmpeg` wheel.

Consequences:

* EXR: scanline, tiled (level 0), multipart and arbitrary channels are read; writing is single-part scanline with
  NONE/ZIP/ZIPS/PIZ, half or float. Deep EXR, DWA, B44 and mip/rip levels beyond level 0 are not handled.
* **TIFF and DPX are not supported.** Reading them reports a clear error on the Read node.
* Colour management uses built-in transfer functions (sRGB, Rec.709, Gamma 2.2/2.4, Cineon, ACEScct) and
  primaries matrices (Rec.709, AP1, AP0). Working space is Linear Rec.709 or ACEScg. **No OCIO config loading.**
* Movie decode/encode runs FFmpeg as a subprocess and converts frames with NumPy (vectorised). Random access
  restarts the decoder with input seeking, which FFmpeg decodes accurately; a test checks random access against
  sequential decode. There is no persistent index file.

The I/O layer is isolated in `core/src/io.cpp` and `app/vertexcomp/movie.py`, so OIIO/OCIO/libav can replace it
without touching ops.

## tinyexr patch

tinyexr always wrote dataWindow = displayWindow = (0, 0, w-1, h-1). The vendored copy is patched (marked
`vertexcomp patch`) to honour the header's data and display windows and pixel aspect ratio, and to write absolute
chunk y coordinates, so EXR round trips preserve both windows.

## One file per category instead of one file per node

Ops share most of their code (PixelOp base, blur kernels, sampling). Files are grouped by category under
`core/src/ops/`. Adding a node still takes one file and one registration macro.

## Coordinates are y-down

Nuke uses y-up. Here the image origin is top-left, matching file storage and screen space. Transform's positive
rotation is counter-clockwise on screen, as in Nuke.

## Blur `size`

`Blur.size` is the kernel radius in pixels for box/triangle/quadratic and the sigma for gaussian.
`GaussianBlur.sigma` is always sigma; `method = auto` switches to a three-box approximation above sigma 16.
Kernels are symmetric on the pixel grid, so there is no half-pixel shift (tested against SciPy).

## Not implemented in this version

* **ChannelSplit with multiple outputs**: the graph has one output per node. Equivalent: right-click a Read >
  *Create Shuffle per layer*, or double-click a layer in the Layers panel.
* **Nested graph groups** that encapsulate nodes and add group input/output ports. Visual subgroups can be created
  from selected nodes and collapsed without changing graph evaluation.
* **Dopesheet** (the curve editor exists).
* **C++ plugin loading** from shared libraries. Plugin directories run Python startup scripts with the console
  API instead.
* **Proxy mode** as a project switch. The viewer evaluates at reduced size only when zoomed out (box-averaged
  display buffer; the graph itself renders at full resolution).
* Disk cache.
* Packaging into a standalone executable (PyInstaller): the app installs as a Python package.

## PySide6 lazy type loading

PySide6 6.12 creates some enum types lazily during paint events. A garbage collection inside that window
traversed a half-built dict and crashed the interpreter (reproduced, backtrace in shiboken
`createPythonEnum`). `vertexcomp/__init__.py` sets `PYSIDE6_OPTION_LAZY=0` before PySide6 is imported.

## Viewer default channel view

The viewer opens in `rgb` (like Nuke). `rgba` composites over the checkerboard using alpha.
