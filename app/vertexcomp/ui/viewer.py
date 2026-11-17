"""Viewer: OpenGL display of the evaluated image. Display transforms
(exposure, gamma, channel view, colour transform, zebra, pixel grid) run in
the fragment shader, so changing them never re-renders the graph."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLTexture,
    QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from vertexcomp import _vccore as vc
from vertexcomp.ui import theme
from vertexcomp.ui.session import PlaybackCache, ViewerRenderer
from vertexcomp.ui.widgets import IconButton, ScrubField

VERT = """
#version 330 core
layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;
uniform vec2 screen;
out vec2 vUV;
void main() {
    vUV = uv;
    gl_Position = vec4(pos.x / screen.x * 2.0 - 1.0, 1.0 - pos.y / screen.y * 2.0, 0.0, 1.0);
}
"""

FRAG = """
#version 330 core
in vec2 vUV;
out vec4 frag;
uniform sampler2D img;
uniform float exposure, gamma, gain;
uniform int mode;        // 0 rgba 1 rgb 2 r 3 g 4 b 5 a 6 luma
uniform int display;     // 0 linear 1 sRGB 2 Rec.709 3 Gamma 2.2
uniform mat3 toDisplay;
uniform int zebra;
uniform float zebraThreshold;
uniform int bg;          // 0 black 1 grey 2 checker
uniform int grid;
uniform vec2 texSize;
uniform float pixelScreen; // screen pixels per texel
uniform vec4 gridColor;

float enc(float v) {
    if (display == 1) return v <= 0.0031308 ? v * 12.92 : 1.055 * pow(v, 1.0 / 2.4) - 0.055;
    if (display == 2) return v < 0.018 ? v * 4.5 : 1.099 * pow(v, 0.45) - 0.099;
    if (display == 3) return pow(max(v, 0.0), 1.0 / 2.2);
    return v;
}

void main() {
    vec4 s = texture(img, vUV);
    vec3 c;
    float a = s.a;
    if (mode == 0 || mode == 1) c = s.rgb;
    else if (mode == 2) c = vec3(s.r);
    else if (mode == 3) c = vec3(s.g);
    else if (mode == 4) c = vec3(s.b);
    else if (mode == 5) c = vec3(s.a);
    else c = vec3(dot(s.rgb, vec3(0.2126, 0.7152, 0.0722)));
    if (mode != 0) a = 1.0;
    float m = exp2(exposure) * gain;
    vec3 lin = c * m;
    if (mode <= 1 || mode == 6) lin = toDisplay * lin;
    bool clip = zebra == 1 && (lin.r > zebraThreshold || lin.g > zebraThreshold || lin.b > zebraThreshold);
    vec3 d = vec3(enc(lin.r), enc(lin.g), enc(lin.b));
    if (gamma > 0.0 && gamma != 1.0) d = sign(d) * pow(abs(d), vec3(1.0 / gamma));
    vec3 back = vec3(0.0);
    if (bg == 1) back = vec3(0.18);
    if (bg == 2) {
        vec2 q = floor(gl_FragCoord.xy / 12.0);
        back = mod(q.x + q.y, 2.0) < 1.0 ? vec3(0.16) : vec3(0.24);
    }
    vec3 outc = mode == 0 ? d + back * (1.0 - clamp(a, 0.0, 1.0)) : d;
    if (clip) {
        float st = mod(gl_FragCoord.x + gl_FragCoord.y, 10.0);
        if (st < 5.0) outc = vec3(1.0, 0.25, 0.1);
    }
    if (grid == 1) {
        vec2 p = vUV * texSize;
        vec2 f = fract(p) * pixelScreen;
        if (f.x < 1.0 || f.y < 1.0) outc = mix(outc, gridColor.rgb, gridColor.a);
    }
    frag = vec4(outc, 1.0);
}
"""

CHANNEL_MODES = ["rgba", "rgb", "red", "green", "blue", "alpha", "luminance"]
DISPLAYS = ["sRGB", "Rec.709", "Gamma 2.2", "linear"]


def layer_slots(layer: str, chans: list[str]) -> list[str]:
    inl = [c for c in chans if c.rsplit(".", 1)[0] == layer]
    slots = [""] * 4
    names = {"red": 0, "green": 1, "blue": 2, "alpha": 3}
    for c in inl:
        i = names.get(c.rsplit(".", 1)[-1])
        if i is not None:
            slots[i] = c
    rest = [c for c in inl if c not in slots]
    for c in rest:
        for i in range(4):
            if not slots[i] and not (i == 3 and len(inl) <= 3):
                slots[i] = c
                break
    return slots


def layers_of(chans: list[str]) -> list[str]:
    out = []
    for c in chans:
        ly = c.rsplit(".", 1)[0] if "." in c else "other"
        if ly not in out:
            out.append(ly)
    if "rgba" in out:
        out.remove("rgba")
        out.insert(0, "rgba")
    return out


class TexLayer:
    def __init__(self):
        self.tex = None
        self.size = (0, 0)
        self.rect = None   # image-space (x0, y0, x1, y1)
        self.array = None  # last uploaded array (H, W, 4)
        self.roi = None
        self.step = 1
        self.frame = None


class ViewerGL(QOpenGLWidget):
    viewChanged = Signal()
    hover = Signal(float, float)
    areaSample = Signal(object)  # (x0, y0, x1, y1) or None

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(200, 150)
        self.zoom = 0.5
        self.cx, self.cy = 960.0, 540.0
        self.layers = {"A": TexLayer(), "B": TexLayer()}
        self.exposure, self.gamma, self.gain = 0.0, 1.0, 1.0
        self.mode = 1
        self.display = 0
        self.matrix = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        self.zebra = False
        self.bg = {"black": 0, "gray": 1, "checker": 2}.get(settings.get("viewer", "background"), 2)
        self.show_overlays = True
        self.format = (1920, 1080)
        self.bbox = None
        self.error = ""
        self.wipe = None  # x fraction of the widget, or None
        self.user_roi = None
        self.roi_tool = False
        self.tool = None  # RotoTool
        self._pan = None
        self._space = False
        self._sample_start = None
        self.sample_box = None
        self.fitted_once = False
        self._prog = None

    # ------------------------------------------------------------ transforms
    def to_screen(self, x, y):
        return ((x - self.cx) * self.zoom + self.width() / 2.0, (y - self.cy) * self.zoom + self.height() / 2.0)

    def to_image(self, sx, sy):
        return ((sx - self.width() / 2.0) / self.zoom + self.cx, (sy - self.height() / 2.0) / self.zoom + self.cy)

    def visible_rect(self):
        x0, y0 = self.to_image(0, 0)
        x1, y1 = self.to_image(self.width(), self.height())
        return x0, y0, x1, y1

    def set_zoom(self, z, anchor=None):
        z = max(1.0 / 64.0, min(100000.0, z))
        if anchor is not None:
            ix, iy = self.to_image(anchor.x(), anchor.y())
            self.zoom = z
            # keep (ix, iy) under the cursor
            self.cx = ix - (anchor.x() - self.width() / 2.0) / z
            self.cy = iy - (anchor.y() - self.height() / 2.0) / z
        else:
            self.zoom = z
        self.viewChanged.emit()
        self.update()

    def fit(self):
        w, h = self.format
        if w <= 0 or h <= 0:
            return
        self.zoom = max(1.0 / 64.0, min((self.width() - 20) / w, (self.height() - 20) / h))
        self.cx, self.cy = w / 2.0, h / 2.0
        self.viewChanged.emit()
        self.update()

    def one_to_one(self, z=1.0):
        self.zoom = z
        self.viewChanged.emit()
        self.update()

    # ------------------------------------------------------------ GL
    def initializeGL(self):
        self._prog = QOpenGLShaderProgram(self)
        self._prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, VERT)
        self._prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, FRAG)
        if not self._prog.link():
            raise RuntimeError("viewer shader: " + self._prog.log())
        self._vao = QOpenGLVertexArrayObject(self)
        self._vao.create()
        self._vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._vbo.create()
        self._vao.bind()
        self._vbo.bind()
        self._vbo.allocate(16 * 4)
        self._prog.bind()
        self._prog.enableAttributeArray(0)
        self._prog.enableAttributeArray(1)
        self._prog.setAttributeBuffer(0, 0x1406, 0, 2, 16)  # GL_FLOAT
        self._prog.setAttributeBuffer(1, 0x1406, 8, 2, 16)
        self._prog.release()
        self._vbo.release()
        self._vao.release()

    def upload(self, which, arr, roi, step):
        L = self.layers[which]
        L.array, L.roi, L.step = arr, roi, step
        if arr is None or arr.size == 0:
            L.rect = None
            self.update()
            return
        h, w = arr.shape[:2]
        self.makeCurrent()
        if L.tex is None or L.size != (w, h):
            if L.tex is not None:
                L.tex.destroy()
            L.tex = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
            L.tex.setFormat(QOpenGLTexture.TextureFormat.RGBA32F)
            L.tex.setSize(w, h)
            L.tex.setMipLevels(1)
            L.tex.allocateStorage()
            L.tex.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)
            L.size = (w, h)
        L.tex.setData(QOpenGLTexture.PixelFormat.RGBA, QOpenGLTexture.PixelType.Float32,
                      np.ascontiguousarray(arr, np.float32).tobytes())
        self.doneCurrent()
        L.rect = (roi[0], roi[1], roi[0] + w * step, roi[1] + h * step)
        self.update()

    def _draw_layer(self, L, clip=None):
        if L.tex is None or L.rect is None:
            return
        f = self.context().functions()
        x0, y0 = self.to_screen(L.rect[0], L.rect[1])
        x1, y1 = self.to_screen(L.rect[2], L.rect[3])
        dpr = self.devicePixelRatioF()
        verts = np.array([x0, y0, 0, 0, x1, y0, 1, 0, x0, y1, 0, 1, x1, y1, 1, 1], np.float32) * np.array(
            [dpr, dpr, 1, 1] * 4, np.float32)
        pix = self.zoom * L.step
        nearest = pix >= float(self.settings.get("viewer", "nearest_zoom"))
        L.tex.setMinificationFilter(QOpenGLTexture.Filter.Linear)
        L.tex.setMagnificationFilter(QOpenGLTexture.Filter.Nearest if nearest else QOpenGLTexture.Filter.Linear)
        p = self._prog
        p.bind()
        p.setUniformValue("screen", float(self.width() * dpr), float(self.height() * dpr))
        p.setUniformValue1i("img", 0)
        p.setUniformValue1f("exposure", float(self.exposure))
        p.setUniformValue1f("gamma", float(self.gamma))
        p.setUniformValue1f("gain", float(self.gain))
        p.setUniformValue1i("mode", int(self.mode))
        p.setUniformValue1i("display", int(self.display))
        from PySide6.QtGui import QMatrix3x3
        m = QMatrix3x3(self.matrix)
        p.setUniformValue("toDisplay", m.transposed())
        p.setUniformValue1i("zebra", 1 if self.zebra else 0)
        p.setUniformValue1f("zebraThreshold", float(self.settings.get("viewer", "zebra_threshold")))
        p.setUniformValue1i("bg", int(self.bg))
        grid = pix >= float(self.settings.get("viewer", "grid_zoom"))
        p.setUniformValue1i("grid", 1 if grid else 0)
        p.setUniformValue("texSize", float(L.size[0]), float(L.size[1]))
        p.setUniformValue1f("pixelScreen", float(pix * dpr))
        p.setUniformValue("gridColor", 0.35, 0.35, 0.35, 0.8)
        if clip is not None:
            f.glEnable(0x0C11)  # GL_SCISSOR_TEST
            sx0, sx1 = clip
            f.glScissor(int(sx0 * dpr), 0, max(0, int((sx1 - sx0) * dpr)), int(self.height() * dpr))
        f.glActiveTexture(0x84C0)
        L.tex.bind()
        self._vao.bind()
        self._vbo.bind()
        self._vbo.write(0, verts.tobytes(), verts.nbytes)
        f.glDrawArrays(0x0005, 0, 4)  # GL_TRIANGLE_STRIP
        self._vbo.release()
        self._vao.release()
        L.tex.release()
        p.release()
        if clip is not None:
            f.glDisable(0x0C11)

    def paintGL(self):
        # Native GL inside begin/endNativePainting so QPainter's GL state is
        # restored before the overlays are drawn.
        p = QPainter(self)
        p.beginNativePainting()
        f = self.context().functions()
        c = QColor(theme.T["base"])
        f.glClearColor(c.redF() * 0.8, c.greenF() * 0.8, c.blueF() * 0.8, 1.0)
        f.glClear(0x4000)
        if self.wipe is not None and self.layers["B"].rect is not None:
            wx = self.wipe * self.width()
            self._draw_layer(self.layers["A"], (0, wx))
            self._draw_layer(self.layers["B"], (wx, self.width()))
        else:
            self._draw_layer(self.layers["A"])
        p.endNativePainting()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self._paint_overlays(p)
        p.end()

    # ------------------------------------------------------------ overlays
    def _paint_overlays(self, p: QPainter):
        w, h = self.format
        if self.show_overlays:
            x0, y0 = self.to_screen(0, 0)
            x1, y1 = self.to_screen(w, h)
            p.setPen(QPen(QColor(110, 110, 110), 1))
            p.drawRect(QRectF(x0, y0, x1 - x0, y1 - y0))
            p.setPen(QColor(140, 140, 140))
            p.setFont(theme.mono_font(8))
            p.drawText(QPointF(x1 - 80, y1 + 12), f"{w}x{h}")
            if self.bbox and self.bbox != (0, 0, w, h) and self.bbox[2] > self.bbox[0]:
                bx0, by0 = self.to_screen(self.bbox[0], self.bbox[1])
                bx1, by1 = self.to_screen(self.bbox[2], self.bbox[3])
                pen = QPen(QColor(self.settings.get("viewer", "bbox_color")), 1, Qt.PenStyle.DashLine)
                p.setPen(pen)
                p.drawRect(QRectF(bx0, by0, bx1 - bx0, by1 - by0))
        self._paint_values(p)
        if self.user_roi:
            rx0, ry0 = self.to_screen(self.user_roi[0], self.user_roi[1])
            rx1, ry1 = self.to_screen(self.user_roi[2], self.user_roi[3])
            p.setPen(QPen(QColor(theme.T["accent"]), 1, Qt.PenStyle.DashLine))
            p.drawRect(QRectF(rx0, ry0, rx1 - rx0, ry1 - ry0))
        if self.sample_box:
            sx0, sy0 = self.to_screen(self.sample_box[0], self.sample_box[1])
            sx1, sy1 = self.to_screen(self.sample_box[2], self.sample_box[3])
            p.setPen(QPen(QColor(220, 220, 220), 1, Qt.PenStyle.DotLine))
            p.drawRect(QRectF(sx0, sy0, sx1 - sx0, sy1 - sy0))
        if self.wipe is not None:
            wx = self.wipe * self.width()
            p.setPen(QPen(QColor(theme.T["accent"]), 1))
            p.drawLine(QPointF(wx, 0), QPointF(wx, self.height()))
            p.drawText(QPointF(wx - 14, 14), "A")
            p.drawText(QPointF(wx + 6, 14), "B")
        if self.tool is not None:
            self.tool.paint(p)
        if self.error:
            p.setPen(QColor(theme.T["error"]))
            p.setFont(theme.mono_font(9))
            p.drawText(QRectF(10, 10, self.width() - 20, self.height() - 20),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap, self.error)

    def _paint_values(self, p: QPainter):
        L = self.layers["A"]
        if L.array is None or L.rect is None or L.step != 1 or not self.settings.get("viewer", "show_values"):
            return
        if self.zoom < float(self.settings.get("viewer", "values_zoom")):
            return
        vx0, vy0, vx1, vy1 = self.visible_rect()
        x0 = max(int(math.floor(vx0)), L.rect[0])
        y0 = max(int(math.floor(vy0)), L.rect[1])
        x1 = min(int(math.ceil(vx1)), L.rect[2])
        y1 = min(int(math.ceil(vy1)), L.rect[3])
        if (x1 - x0) * (y1 - y0) > 4000:
            return
        size = max(6.0, min(10.0, self.zoom / 9.0))
        p.setFont(theme.mono_font(size))
        names = ["R", "G", "B", "A"]
        colors = [QColor(235, 120, 110), QColor(130, 220, 120), QColor(120, 160, 245), QColor(210, 210, 210)]
        for y in range(y0, y1):
            for x in range(x0, x1):
                v = L.array[y - L.rect[1], x - L.rect[0]]
                sx, sy = self.to_screen(x, y)
                lum = 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]
                for i in range(4):
                    c = QColor(colors[i])
                    if lum * (2 ** self.exposure) > 0.6:
                        c = c.darker(250)
                    p.setPen(c)
                    p.drawText(QPointF(sx + 3, sy + (i + 1) * (size + 3) + 1), f"{names[i]} {v[i]:.4f}")

    # ------------------------------------------------------------ input
    def wheelEvent(self, e):
        d = e.angleDelta().y() / 120.0
        self.set_zoom(self.zoom * (1.18 ** d), e.position())

    def mousePressEvent(self, e):
        self.setFocus()
        if self.tool is not None and self.tool.press(e):
            return
        mid = e.button() == Qt.MouseButton.MiddleButton
        alt = e.button() == Qt.MouseButton.LeftButton and (e.modifiers() & Qt.KeyboardModifier.AltModifier or self._space)
        if mid or alt:
            self._pan = (e.position(), self.cx, self.cy)
            return
        if e.button() == Qt.MouseButton.LeftButton and self.roi_tool:
            ix, iy = self.to_image(e.position().x(), e.position().y())
            self._roi_start = (ix, iy)
            self.user_roi = None
            return
        if e.button() == Qt.MouseButton.LeftButton and self.wipe is not None and abs(e.position().x() - self.wipe * self.width()) < 6:
            self._wipe_drag = True
            return
        if e.button() == Qt.MouseButton.LeftButton and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            ix, iy = self.to_image(e.position().x(), e.position().y())
            self._sample_start = (ix, iy)
            self.sample_box = (ix, iy, ix, iy)

    def mouseMoveEvent(self, e):
        ix, iy = self.to_image(e.position().x(), e.position().y())
        self.hover.emit(ix, iy)
        if self.tool is not None and self.tool.move(e):
            return
        if self._pan is not None:
            p0, cx, cy = self._pan
            d = e.position() - p0
            self.cx = cx - d.x() / self.zoom
            self.cy = cy - d.y() / self.zoom
            self.viewChanged.emit()
            self.update()
            return
        if getattr(self, "_roi_start", None) is not None:
            sx, sy = self._roi_start
            self.user_roi = (int(math.floor(min(sx, ix))), int(math.floor(min(sy, iy))),
                             int(math.ceil(max(sx, ix))), int(math.ceil(max(sy, iy))))
            self.update()
            return
        if getattr(self, "_wipe_drag", False):
            self.wipe = max(0.0, min(1.0, e.position().x() / max(1, self.width())))
            self.update()
            return
        if self._sample_start is not None:
            sx, sy = self._sample_start
            self.sample_box = (min(sx, ix), min(sy, iy), max(sx, ix), max(sy, iy))
            self.areaSample.emit(self.sample_box)
            self.update()

    def mouseReleaseEvent(self, e):
        if self.tool is not None and self.tool.release(e):
            return
        if getattr(self, "_roi_start", None) is not None:
            self._roi_start = None
            if self.user_roi and (self.user_roi[2] - self.user_roi[0] < 2 or self.user_roi[3] - self.user_roi[1] < 2):
                self.user_roi = None
            self.viewChanged.emit()
        self._wipe_drag = False
        self._pan = None
        if self._sample_start is not None:
            self._sample_start = None
            self.areaSample.emit(self.sample_box)

    def mouseDoubleClickEvent(self, e):
        if self.tool is not None and self.tool.double_click(e):
            return

    def keyPressEvent(self, e):
        if self.tool is not None and self.tool.key(e):
            return
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
            self._space = True
        k = e.key()
        if k == Qt.Key.Key_F:
            self.fit()
        elif k == Qt.Key.Key_Escape:
            self.sample_box = None
            self.areaSample.emit(None)
            self.update()
        else:
            super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
            self._space = False
        super().keyReleaseEvent(e)

    def sample_at(self, ix, iy, which="A"):
        L = self.layers[which]
        if L.array is None or L.rect is None:
            return None
        x = int(math.floor((ix - L.rect[0]) / L.step))
        y = int(math.floor((iy - L.rect[1]) / L.step))
        if 0 <= y < L.array.shape[0] and 0 <= x < L.array.shape[1]:
            return L.array[y, x]
        return None

    def sample_area(self, box):
        L = self.layers["A"]
        if L.array is None or L.rect is None or box is None:
            return None
        x0 = int(math.floor((box[0] - L.rect[0]) / L.step))
        y0 = int(math.floor((box[1] - L.rect[1]) / L.step))
        x1 = int(math.ceil((box[2] - L.rect[0]) / L.step))
        y1 = int(math.ceil((box[3] - L.rect[1]) / L.step))
        a = L.array[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
        if a.size == 0:
            return None
        a = a.reshape(-1, 4)
        return a.min(0), a.max(0), a.mean(0)


class ViewerPanel(QWidget):
    """Viewer with its control bar and the transport/timeline below."""
    transformToolToggled = Signal(bool)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.s = session
        st = session.settings
        self.gl = ViewerGL(st)
        self.renderers = {"A": ViewerRenderer(session), "B": ViewerRenderer(session)}
        for k, r in self.renderers.items():
            r.ready.connect(lambda out, k=k: self._on_ready(k, out))
        self.cache = PlaybackCache(session)
        self.channels: list[str] = []
        self.layer = "rgba"
        self.last_info = None
        self.last_ms = 0.0
        self._pending_gen = {"A": 0, "B": 0}

        bar = QFrame()
        bar.setObjectName("viewerBar")
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(4, 2, 4, 2)
        hl.setSpacing(3)

        self.layer_box = QComboBox()
        self.layer_box.setToolTip("Layer / AOV to view")
        self.layer_box.setMinimumWidth(90)
        self.layer_box.addItem("rgba")
        self.layer_box.currentTextChanged.connect(self._layer_changed)
        self.mode_box = QComboBox()
        self.mode_box.addItems(CHANNEL_MODES)
        self.mode_box.setToolTip("Channels shown (Shift+R/G/B/A/Y/C)")
        self.mode_box.setCurrentIndex(1)  # rgb, like Nuke's default
        self.mode_box.currentIndexChanged.connect(self._mode_changed)

        self.exp = ScrubField(float(st.get("color", "exposure")), step=0.05, decimals=2)
        self.exp.setToolTip("Exposure in stops ([ and ] step one stop)")
        self.exp.setFixedWidth(52)
        self.exp.valueEdited.connect(lambda v, f: self._set_display(exposure=v))
        self.gam = ScrubField(float(st.get("color", "gamma")), step=0.01, decimals=2, minimum=0.05)
        self.gam.setToolTip("Display gamma")
        self.gam.setFixedWidth(46)
        self.gam.valueEdited.connect(lambda v, f: self._set_display(gamma=v))
        self.gain = ScrubField(1.0, step=0.01, decimals=3, minimum=0.0)
        self.gain.setToolTip("Gain (multiplier, linear)")
        self.gain.setFixedWidth(46)
        self.gain.valueEdited.connect(lambda v, f: self._set_display(gain=v))
        reset = IconButton("refresh", "Reset exposure, gamma and gain")
        reset.clicked.connect(self.reset_display)

        self.disp_box = QComboBox()
        self.disp_box.addItems(DISPLAYS)
        self.disp_box.setCurrentText(st.get("color", "display") if st.get("color", "display") in DISPLAYS else "sRGB")
        self.disp_box.setToolTip("Display transform")
        self.disp_box.currentIndexChanged.connect(lambda i: self._set_display(display=i))

        self.zebra_btn = IconButton("zebra", "Clip warning: values above the zebra threshold", True)
        self.zebra_btn.toggled.connect(lambda on: self._set_display(zebra=on))
        self.ovl_btn = IconButton("overlay", "Overlays: format, bbox, roto", True)
        self.ovl_btn.setChecked(True)
        self.ovl_btn.toggled.connect(self._overlays)
        self.transform_btn = QPushButton("Transform")
        self.transform_btn.setCheckable(True)
        self.transform_btn.setToolTip("Show viewer controls for the selected Transform node")
        self.transform_btn.toggled.connect(self.transformToolToggled)
        self.roi_btn = IconButton("roi", "Region of interest: drag to limit rendering", True)
        self.roi_btn.toggled.connect(self._roi_toggled)
        self.slot_box = QComboBox()
        self.slot_box.addItems([str(i) for i in range(1, 10)])
        self.slot_box.setToolTip("Viewer input (keys 1-9 in the node graph)")
        self.slot_box.currentIndexChanged.connect(self._slot_changed)
        self.wipe_btn = IconButton("wipe", "Wipe A/B", True)
        self.wipe_btn.toggled.connect(self._wipe_toggled)
        self.wipe_box = QComboBox()
        self.wipe_box.addItems([str(i) for i in range(1, 10)])
        self.wipe_box.setCurrentIndex(1)
        self.wipe_box.setToolTip("Wipe B input")
        self.wipe_box.currentIndexChanged.connect(lambda _: self.request())
        fit = IconButton("fit", "Fit (F)")
        fit.clicked.connect(self.gl.fit)

        def lbl(t):
            l = QLabel(t)
            l.setObjectName("knobLabel")
            return l

        for wdg in (self.layer_box, self.mode_box, lbl("exp"), self.exp, lbl("gamma"), self.gam, lbl("gain"),
                    self.gain, reset, self.disp_box, self.zebra_btn, self.ovl_btn, self.transform_btn, self.roi_btn):
            hl.addWidget(wdg)
        hl.addStretch(1)
        for wdg in (lbl("A"), self.slot_box, self.wipe_btn, lbl("B"), self.wipe_box, fit):
            hl.addWidget(wdg)

        self.info_label = QLabel("")
        self.info_label.setFont(theme.mono_font(8))
        self.info_label.setObjectName("knobLabel")
        self.pixel_label = QLabel("")
        self.pixel_label.setFont(theme.mono_font(8))
        info = QFrame()
        il = QHBoxLayout(info)
        il.setContentsMargins(6, 1, 6, 1)
        il.addWidget(self.pixel_label, 1)
        il.addWidget(self.info_label)

        from vertexcomp.ui.timeline import Transport
        self.transport = Transport(session, self)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(bar)
        lay.addWidget(self.gl, 1)
        lay.addWidget(info)
        lay.addWidget(self.transport)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(0)
        self._debounce.timeout.connect(self._do_request)
        self.gl.viewChanged.connect(self.request)
        self.gl.hover.connect(self._hover)
        self.gl.areaSample.connect(self._area)
        session.frameChanged.connect(lambda f: None if self.transport.playing else self.request())
        session.viewedChanged.connect(self._viewed_changed)
        session.nodeChanged.connect(lambda n: self.request())
        session.graphChanged.connect(self.request)
        self._set_display()
        self._shortcuts()

    def _shortcuts(self):
        hk = self.s.settings.data["hotkeys"]
        pairs = {"view_red": 2, "view_green": 3, "view_blue": 4, "view_alpha": 5, "view_luma": 6, "view_rgb": 0}
        for name, idx in pairs.items():
            sc = QShortcut(QKeySequence(hk.get(name, "")), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(lambda idx=idx: self.mode_box.setCurrentIndex(
                0 if self.mode_box.currentIndex() == idx else idx))
        up = QShortcut(QKeySequence(hk.get("exposure_up", "]")), self)
        up.activated.connect(lambda: self._step_exposure(1))
        dn = QShortcut(QKeySequence(hk.get("exposure_down", "[")), self)
        dn.activated.connect(lambda: self._step_exposure(-1))

    def _step_exposure(self, d):
        self.exp.setValue(round(self.exp.value() + d))
        self._set_display(exposure=self.exp.value())

    # ------------------------------------------------------------ display state
    def reset_display(self):
        self.exp.setValue(0.0)
        self.gam.setValue(1.0)
        self.gain.setValue(1.0)
        self._set_display(exposure=0.0, gamma=1.0, gain=1.0)

    def _set_display(self, **kw):
        g = self.gl
        for k, v in kw.items():
            setattr(g, k, v)
        if "display" not in kw:
            g.display = [1, 2, 3, 0][self.disp_box.currentIndex()]
        else:
            g.display = [1, 2, 3, 0][kw["display"]]
        ws = vc.working_space()
        g.matrix = vc.primaries_matrix(ws, "Linear Rec.709") if ws != "Linear Rec.709" else [1, 0, 0, 0, 1, 0, 0, 0, 1]
        g.update()

    def _overlays(self, on):
        self.gl.show_overlays = on
        self.gl.update()

    def _roi_toggled(self, on):
        self.gl.roi_tool = on
        if not on:
            self.gl.user_roi = None
            self.request()

    def _wipe_toggled(self, on):
        self.gl.wipe = 0.5 if on else None
        self.s.wipe_slot = self.wipe_box.currentIndex() + 1 if on else 0
        self.request()

    def _slot_changed(self, i):
        self.s.active_slot = i + 1
        self.s.viewedChanged.emit()

    def _layer_changed(self, name):
        if name:
            self.layer = name
            self.request()

    def _mode_changed(self, i):
        self.gl.mode = i
        self.gl.update()
        if i in (2, 3, 4, 5) and self.layer != "rgba":
            pass
        self.request()

    def _viewed_changed(self):
        self.slot_box.blockSignals(True)
        self.slot_box.setCurrentIndex(self.s.active_slot - 1)
        self.slot_box.blockSignals(False)
        self.request()

    # ------------------------------------------------------------ requests
    def request(self):
        self._debounce.start()

    def view_params(self):
        """ROI (padded visible area), slots and step for the current view."""
        g = self.gl
        vx0, vy0, vx1, vy1 = g.visible_rect()
        pad_x, pad_y = (vx1 - vx0) * 0.15, (vy1 - vy0) * 0.15
        x0, y0 = int(math.floor(vx0 - pad_x)), int(math.floor(vy0 - pad_y))
        x1, y1 = int(math.ceil(vx1 + pad_x)), int(math.ceil(vy1 + pad_y))
        if self.last_info:
            b = self.last_info["bbox"]
            x0, y0, x1, y1 = max(x0, b[0]), max(y0, b[1]), min(x1, b[2]), min(y1, b[3])
        if g.user_roi:
            r = g.user_roi
            x0, y0, x1, y1 = max(x0, r[0]), max(y0, r[1]), min(x1, r[2]), min(y1, r[3])
        step = 1
        if g.zoom < 1.0 and self.s.settings.get("performance", "interactive_downscale"):
            step = 2 ** int(math.floor(math.log2(1.0 / g.zoom)))
        if x1 <= x0 or y1 <= y0:
            x0 = y0 = x1 = y1 = 0
        # Snap to the step grid so neighbouring requests reuse cache.
        x0, y0 = (x0 // step) * step, (y0 // step) * step
        return (x0, y0, x1, y1), self.slots(), step

    def slots(self):
        if self.layer == "rgba" or not self.channels:
            return ["rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"]
        sl = layer_slots(self.layer, self.channels)
        if sum(1 for x in sl if x) == 1:
            c = next(x for x in sl if x)
            return [c, c, c, ""]
        return sl

    def _do_request(self):
        node = self.s.viewed_node()
        if node is None:
            self.gl.upload("A", None, None, 1)
            self.gl.error = ""
            self.info_label.setText("no viewer input: select a node and press 1")
            return
        roi, slots, step = self.view_params()
        if self.last_info is None or roi[2] <= roi[0]:
            # First request for this node: ask for the format window to learn the bbox.
            fw, fh = self.gl.format
            roi = (0, 0, fw, fh) if roi[2] <= roi[0] else roi
        self.cache.configure(node, roi, slots, step)
        cached = self.cache.get(self.s.frame)
        if cached is not None:
            self.gl.upload("A", cached, roi, step)
            self.gl.layers["A"].frame = self.s.frame
        self._pending_gen["A"] = self.renderers["A"].request(node, self.s.frame, roi, slots, step)
        if self.s.wipe_slot:
            nb = self.s.viewed_node(self.s.wipe_slot)
            if nb is not None:
                self._pending_gen["B"] = self.renderers["B"].request(nb, self.s.frame, roi, slots, step)
            else:
                self.gl.upload("B", None, None, 1)

    def _on_ready(self, which, out):
        if out["gen"] != self._pending_gen[which]:
            return
        g = self.gl
        if which == "A":
            node = out["node"]
            if self.s.viewed_node() is None or node.name != self.s.viewed_node().name:
                return
            g.error = out["error"]
            info = out["info"]
            if info is not None:
                new_node = self.last_info is None or self.last_info.get("_node") != node.name
                prev_bbox = self.last_info["bbox"] if self.last_info else None
                self.last_info = dict(info)
                self.last_info["_node"] = node.name
                g.format = info["format"][:2]
                g.bbox = info["bbox"]
                self._update_layers(info["channels"])
                if not g.fitted_once:
                    g.fitted_once = True
                    g.fit()
                if new_node or prev_bbox != info["bbox"]:
                    roi, _, _ = self.view_params()
                    if roi != out["roi"]:
                        self.request()
            if out["array"] is not None:
                g.upload("A", out["array"], out["roi"], out["step"])
                g.layers["A"].frame = out["frame"]
                if not self.transport.playing:
                    self.cache.put(out["frame"], out["array"])
            elif out["error"]:
                g.upload("A", None, None, 1)
            self.last_ms = out["ms"]
            self._update_info()
            self.s.renderStateChanged.emit()
        else:
            g.upload("B", out["array"], out["roi"], out["step"]) if out["array"] is not None else g.upload("B", None, None, 1)

    def show_cached(self, frame) -> bool:
        """Playback: display a read-ahead frame if available."""
        arr = self.cache.get(frame)
        if arr is None:
            return False
        roi, _, step = self.view_params()
        view = self.cache.view
        if view is not None:
            roi, step = view[1], view[3]
        self.gl.upload("A", arr, roi, step)
        self.gl.layers["A"].frame = frame
        return True

    def _update_layers(self, chans):
        if chans == self.channels:
            return
        self.channels = list(chans)
        layers = layers_of(chans)
        cur = self.layer
        self.layer_box.blockSignals(True)
        self.layer_box.clear()
        self.layer_box.addItems(layers or ["rgba"])
        if cur in layers:
            self.layer_box.setCurrentText(cur)
        else:
            self.layer = layers[0] if layers else "rgba"
        self.layer_box.blockSignals(False)
        self.layer_box.setToolTip("Layers: " + ", ".join(layers))

    def set_layer(self, layer):
        if self.layer_box.findText(layer) >= 0:
            self.layer_box.setCurrentText(layer)

    def _update_info(self):
        i = self.last_info
        if not i:
            return
        fw, fh, par = i["format"]
        b = i["bbox"]
        step = self.gl.layers["A"].step
        self.info_label.setText(
            f"{fw}x{fh}  bbox {b[0]} {b[1]} {b[2]} {b[3]}  {len(i['channels'])} ch  "
            f"zoom {self.gl.zoom:.3g}  {'1/' + str(step) + ' ' if step > 1 else ''}{self.last_ms:.0f} ms")

    def _hover(self, ix, iy):
        x, y = int(math.floor(ix)), int(math.floor(iy))
        v = self.gl.sample_at(ix, iy)
        if v is None:
            self.pixel_label.setText(f"x {x:>5} y {y:>5}")
            return
        sl = self.slots()
        names = [s.rsplit(".", 1)[-1][:1].upper() if s else "-" for s in sl]
        g = self.gl
        m = 2 ** g.exposure * g.gain
        disp = []
        for c in v[:3]:
            d = max(0.0, float(c) * m)
            d = d * 12.92 if d <= 0.0031308 else 1.055 * d ** (1 / 2.4) - 0.055
            disp.append(max(0, min(255, int(round(d * 255)))))
        hexs = "#{:02X}{:02X}{:02X}".format(*disp)
        approx = "~" if g.layers["A"].step > 1 else " "
        vals = "  ".join(f"{n} {float(c): .5f}" for n, c in zip(names, v))
        self.pixel_label.setText(f"x {x:>5} y {y:>5} {approx}{vals}   {hexs}")

    def _area(self, box):
        if box is None:
            self._hover(0, 0)
            return
        r = self.gl.sample_area(box)
        if r is None:
            return
        mn, mx, mean = r
        f = lambda a: " ".join(f"{float(v):.4f}" for v in a)
        self.pixel_label.setText(f"area {int(box[2]-box[0])}x{int(box[3]-box[1])}  min {f(mn)}  max {f(mx)}  mean {f(mean)}")

    def shutdown(self):
        for r in self.renderers.values():
            r.stop()
        self.cache.stop()
