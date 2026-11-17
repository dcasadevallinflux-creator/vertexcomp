"""Secondary docks: Layers, Log, Python console, Profiler, Scopes, Curve editor."""
from __future__ import annotations

import code
import io
import math
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vertexcomp import _vccore as vc
from vertexcomp.ui import theme


class LayersPanel(QTreeWidget):
    """Layers and channels of the viewed stream. Click a layer to view it."""

    def __init__(self, session, viewer_panel):
        super().__init__()
        self.s, self.vp = session, viewer_panel
        self.setHeaderLabels(["layer / channel"])
        self.setRootIsDecorated(True)
        self.itemClicked.connect(self._clicked)
        self.itemDoubleClicked.connect(self._double)
        session.renderStateChanged.connect(self.refresh)
        self._last = None

    def refresh(self):
        chans = list(self.vp.channels)
        if chans == self._last:
            return
        self._last = chans
        self.clear()
        layers = {}
        for c in chans:
            ly, _, short = c.rpartition(".")
            layers.setdefault(ly or "other", []).append(short)
        for ly, comps in layers.items():
            it = QTreeWidgetItem([ly])
            it.setData(0, Qt.ItemDataRole.UserRole, ly)
            for c in comps:
                ch = QTreeWidgetItem([c])
                ch.setData(0, Qt.ItemDataRole.UserRole, f"{ly}.{c}")
                it.addChild(ch)
            self.addTopLevelItem(it)

    def _clicked(self, it, col):
        val = it.data(0, Qt.ItemDataRole.UserRole)
        if it.parent() is None:
            self.vp.set_layer(val)
        else:
            self.vp.set_layer(it.parent().data(0, Qt.ItemDataRole.UserRole))
            comp = val.rsplit(".", 1)[-1]
            idx = {"red": 2, "green": 3, "blue": 4, "alpha": 5}.get(comp)
            if idx is not None:
                self.vp.mode_box.setCurrentIndex(idx)

    def _double(self, it, col):
        """Double-click a layer: Shuffle it out of the viewed node."""
        if it.parent() is not None:
            return
        node = self.s.viewed_node()
        if node is None:
            return
        layer = it.data(0, Qt.ItemDataRole.UserRole)
        from vertexcomp.ui.viewer import layer_slots
        slots = layer_slots(layer, self.vp.channels)
        m = self.s.project.node_meta(node)

        def build():
            sh = self.s.project.create("Shuffle", x=m.x + 120, y=m.y + 60)
            for name, src in zip(("red", "green", "blue", "alpha"), slots):
                sh.set(name, src if src else ("1" if name == "alpha" else "0"))
            self.s.project.graph.connect(sh, 0, node)
            self.s.project.node_meta(sh).label = layer
            return sh
        sh = self.s.structural("shuffle layer", build)
        if sh is not None:
            self.s.select([sh])


class LogPanel(QPlainTextEdit):
    def __init__(self, session):
        super().__init__()
        self.setReadOnly(True)
        self.setFont(theme.mono_font(8))
        self.setMaximumBlockCount(5000)
        session.logMessage.connect(self.add)

    def add(self, level, text):
        col = {"error": theme.T["error"], "warning": "#c9a227"}.get(level, theme.T["text"])
        stamp = time.strftime("%H:%M:%S")
        self.appendHtml(f'<span style="color:{theme.T["dim"]}">{stamp}</span> '
                        f'<span style="color:{col}">{text.replace("<", "&lt;")}</span>')


class ScriptConsole(QWidget):
    """Python console with the project API in scope:
    project, graph, engine, session, vc, create_node, connect, set_knob, render."""

    def __init__(self, session, main_window):
        super().__init__()
        self.s = session
        self.out = QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setFont(theme.mono_font(9))
        self.inp = QLineEdit()
        self.inp.setFont(theme.mono_font(9))
        self.inp.setPlaceholderText(">>> create_node('Blur'); set_knob('Blur1', 'size', [5, 5])")
        self.inp.returnPressed.connect(self.run)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(1)
        lay.addWidget(self.out, 1)
        lay.addWidget(self.inp)
        self.history, self.hpos = [], 0
        self.inp.installEventFilter(self)

        def node(n):
            return session.project.find(n) if isinstance(n, str) else n

        def create_node(t, name="", x=None, y=None):
            n = session.create_node(t, x, y, connect_selected=False)
            if name:
                session.rename(n, name)
                n = session.project.find(name)
            return n

        def connect(dst, slot, src):
            return session.connect(node(dst), slot, node(src) if src is not None else None)

        def set_knob(n, k, v):
            session.set_knob(node(n), k, v)

        def render(write, first=None, last=None):
            from vertexcomp.project import write_node_range
            w = node(write)
            write_node_range(session.project, w, first if first is not None else int(w.get("first")[0]),
                             last if last is not None else int(w.get("last")[0]),
                             log=lambda t: session.log(t))

        self.ns = {"project": session.project, "graph": session.project.graph, "engine": session.project.engine,
                   "session": session, "vc": vc, "np": np, "create_node": create_node, "connect": connect,
                   "set_knob": set_knob, "render": render, "node": node, "window": main_window}
        self.console = code.InteractiveInterpreter(self.ns)

    def eventFilter(self, obj, e):
        from PySide6.QtCore import QEvent
        if e.type() == QEvent.Type.KeyPress and self.history:
            if e.key() == Qt.Key.Key_Up:
                self.hpos = max(0, self.hpos - 1)
                self.inp.setText(self.history[self.hpos])
                return True
            if e.key() == Qt.Key.Key_Down:
                self.hpos = min(len(self.history), self.hpos + 1)
                self.inp.setText(self.history[self.hpos] if self.hpos < len(self.history) else "")
                return True
        return False

    def run(self):
        src = self.inp.text()
        if not src.strip():
            return
        self.history.append(src)
        self.hpos = len(self.history)
        self.inp.clear()
        self.out.appendPlainText(">>> " + src)
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            try:
                try:
                    res = eval(compile(src, "<console>", "eval"), self.ns)
                    if res is not None:
                        print(repr(res))
                except SyntaxError:
                    exec(compile(src, "<console>", "exec"), self.ns)
            except Exception:
                traceback.print_exc()
        text = buf.getvalue().rstrip()
        if text:
            self.out.appendPlainText(text)


class ProfilerPanel(QWidget):
    def __init__(self, session):
        super().__init__()
        self.s = session
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["node", "renders", "total ms (incl. inputs)", "ms / render"])
        self.table.verticalHeader().setVisible(False)
        self.label = QLabel("")
        self.label.setFont(theme.mono_font(8))
        reset = QPushButton("Reset")
        reset.clicked.connect(lambda: (session.project.engine.reset_stats(), self.refresh()))
        clear = QPushButton("Clear cache")
        clear.clicked.connect(lambda: (session.project.engine.clear_cache(), self.refresh()))
        top = QHBoxLayout()
        top.addWidget(self.label, 1)
        top.addWidget(clear)
        top.addWidget(reset)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addLayout(top)
        lay.addWidget(self.table, 1)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(1000)

    def refresh(self):
        if not self.isVisible():
            return
        eng = self.s.project.engine
        st = eng.stats()
        cs = eng.cache_stats()
        self.label.setText(f"cache {cs['used'] / 2**20:.0f} / {cs['limit'] / 2**20:.0f} MB   hits {cs['hits']}   "
                           f"misses {cs['misses']}   threads {vc.thread_count()}")
        rows = sorted(st.items(), key=lambda kv: -kv[1][0])
        self.table.setRowCount(len(rows))
        for i, (name, (ms, n)) in enumerate(rows):
            for j, v in enumerate((name, str(n), f"{ms:.1f}", f"{ms / max(1, n):.2f}")):
                self.table.setItem(i, j, QTableWidgetItem(v))


class ScopesPanel(QWidget):
    """Histogram, waveform and vectorscope of the viewer image (computed in the core)."""

    def __init__(self, session, viewer_panel):
        super().__init__()
        self.s, self.vp = session, viewer_panel
        self.mode = QComboBox()
        self.mode.addItems(["histogram", "waveform", "vectorscope"])
        self.mode.currentIndexChanged.connect(self.update)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        top = QHBoxLayout()
        top.addWidget(self.mode)
        top.addStretch(1)
        lay.addLayout(top)
        lay.addStretch(1)
        self.data = None
        self._img = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.compute)
        self.timer.start(300)
        self._last_id = None

    def compute(self):
        if not self.isVisible():
            return
        L = self.vp.gl.layers["A"]
        if L.array is None or id(L.array) == self._last_id:
            return
        self._last_id = id(L.array)
        arr = L.array
        step = max(1, int(math.sqrt(arr.shape[0] * arr.shape[1] / 250000)))
        sub = np.ascontiguousarray(arr[::step, ::step], np.float32)
        self.data = vc.scopes(sub, 256, float(self.vp.gl.exposure))
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        r = QRectF(6, 30, self.width() - 12, self.height() - 36)
        p.fillRect(r, QColor(theme.T["base"]))
        if self.data is None:
            return
        mode = self.mode.currentText()
        cols = [QColor(220, 80, 70), QColor(90, 200, 90), QColor(90, 130, 230), QColor(200, 200, 200)]
        if mode == "histogram":
            h = self.data["histogram"][:, :, 0]
            mx = max(1.0, float(np.percentile(h[:3], 99.5)))
            for c in range(4):
                path = QPainterPath(QPointF(r.left(), r.bottom()))
                for i in range(256):
                    x = r.left() + i / 255.0 * r.width()
                    y = r.bottom() - min(1.0, h[c, i] / mx) * r.height()
                    path.lineTo(QPointF(x, y))
                col = QColor(cols[c])
                col.setAlpha(200 if c < 3 else 120)
                p.setPen(QPen(col, 1))
                p.drawPath(path)
        elif mode == "waveform":
            w = self.data["waveform"]
            img = np.clip(np.log1p(w) / max(1e-6, np.log1p(w).max()) * 255 * 1.6, 0, 255).astype(np.uint8)
            rgba = np.zeros((256, w.shape[2], 4), np.uint8)
            rgba[..., 0], rgba[..., 1], rgba[..., 2] = img[0], img[1], img[2]
            rgba[..., 3] = 255
            qi = QImage(rgba.data, rgba.shape[1], 256, rgba.shape[1] * 4, QImage.Format.Format_RGBA8888).copy()
            p.drawImage(r, qi)
        else:
            v = self.data["vectorscope"][:, :, 0]
            img = np.clip(np.log1p(v) / max(1e-6, np.log1p(v).max()) * 255, 0, 255).astype(np.uint8)
            rgba = np.dstack([img, img, img, np.full_like(img, 255)])
            side = min(r.width(), r.height())
            sq = QRectF(r.center().x() - side / 2, r.center().y() - side / 2, side, side)
            qi = QImage(np.ascontiguousarray(rgba).data, 256, 256, 1024, QImage.Format.Format_RGBA8888).copy()
            p.drawImage(sq, qi)
            p.setPen(QPen(QColor(theme.T["dim"]), 1))
            p.drawEllipse(sq.center(), side / 2 * 0.9, side / 2 * 0.9)


class CurveEditor(QWidget):
    """Animation curves of a knob: drag keys (Shift: frame only, Ctrl: value only),
    double-click to add, Delete to remove, right-click for interpolation."""

    def __init__(self, session):
        super().__init__()
        self.s = session
        self.target = None  # (node name, knob)
        self.knobs = QListWidget()
        self.knobs.setMaximumWidth(150)
        self.knobs.currentTextChanged.connect(self._pick)
        self.canvas = _CurveCanvas(self)
        split = QSplitter()
        split.addWidget(self.knobs)
        split.addWidget(self.canvas)
        split.setStretchFactor(1, 1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)
        session.selectionChanged.connect(self.reload)
        session.nodeChanged.connect(lambda n: self.canvas.update())
        session.frameChanged.connect(lambda f: self.canvas.update())

    def show_knob(self, node, knob):
        self.s.select([node])
        self.reload()
        items = self.knobs.findItems(knob, Qt.MatchFlag.MatchExactly)
        if items:
            self.knobs.setCurrentItem(items[0])

    def reload(self):
        self.knobs.blockSignals(True)
        self.knobs.clear()
        if self.s.selected:
            n = self.s.selected[-1]
            for k in n.knob_names:
                kd = n.knob_def(k)
                if kd["animatable"] and n.is_animated(k):
                    self.knobs.addItem(k)
        self.knobs.blockSignals(False)
        if self.knobs.count():
            self.knobs.setCurrentRow(0)
        else:
            self.target = None
            self.canvas.fit_done = False
            self.canvas.update()

    def _pick(self, k):
        if self.s.selected and k:
            self.target = (self.s.selected[-1].name, k)
            self.canvas.fit()

    def node(self):
        return self.s.project.find(self.target[0]) if self.target else None


class _CurveCanvas(QWidget):
    COLORS = [QColor(220, 90, 80), QColor(100, 200, 100), QColor(100, 140, 230), QColor(200, 200, 200),
              QColor(200, 160, 80), QColor(160, 110, 200)]

    def __init__(self, editor):
        super().__init__()
        self.ed = editor
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.view = [1.0, -0.1, 100.0, 1.1]  # f0, v0, f1, v1
        self.sel = None  # (comp, index)
        self._drag = None
        self.fit_done = False

    def fit(self):
        n = self.ed.node()
        if n is None:
            return
        keys = n.keys(self.ed.target[1])
        fs = [k[0] for c in keys for k in c]
        vs = [k[1] for c in keys for k in c]
        st = self.ed.s.project.settings
        f0, f1 = min(fs + [st.first]), max(fs + [st.last])
        v0, v1 = (min(vs), max(vs)) if vs else (0, 1)
        if v1 - v0 < 1e-6:
            v0, v1 = v0 - 1, v1 + 1
        pad = (v1 - v0) * 0.1
        self.view = [f0, v0 - pad, f1, v1 + pad]
        self.update()

    def to_screen(self, f, v):
        f0, v0, f1, v1 = self.view
        return (20 + (f - f0) / ((f1 - f0) or 1) * (self.width() - 30),
                self.height() - 15 - (v - v0) / ((v1 - v0) or 1) * (self.height() - 25))

    def to_value(self, x, y):
        f0, v0, f1, v1 = self.view
        return (f0 + (x - 20) / max(1, self.width() - 30) * (f1 - f0),
                v0 + (self.height() - 15 - y) / max(1, self.height() - 25) * (v1 - v0))

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(theme.T["base"]))
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        n = self.ed.node()
        p.setPen(QColor(theme.T["dim"]))
        p.setFont(theme.mono_font(8))
        f0, v0, f1, v1 = self.view
        for i in range(5):
            v = v0 + (v1 - v0) * i / 4
            _, y = self.to_screen(f0, v)
            p.setPen(QPen(QColor(theme.T["bg"]), 1))
            p.drawLine(QPointF(20, y), QPointF(self.width(), y))
            p.setPen(QColor(theme.T["dim"]))
            p.drawText(QPointF(2, y - 2), f"{v:.3g}")
        x, _ = self.to_screen(self.ed.s.frame, 0)
        p.setPen(QPen(QColor(theme.T["accent"]), 1))
        p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        if n is None:
            p.setPen(QColor(theme.T["dim"]))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "select a node with animated knobs")
            return
        knob = self.ed.target[1]
        keys = n.keys(knob)
        for c, comp in enumerate(keys):
            if not comp:
                continue
            col = self.COLORS[c % len(self.COLORS)]
            path = QPainterPath()
            steps = max(2, self.width() // 3)
            for i in range(steps + 1):
                f = f0 + (f1 - f0) * i / steps
                v = n.get(knob, f)[c]
                pt = QPointF(*self.to_screen(f, v))
                path.moveTo(pt) if i == 0 else path.lineTo(pt)
            p.setPen(QPen(col, 1.2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
            for i, k in enumerate(comp):
                sx, sy = self.to_screen(k[0], k[1])
                sel = self.sel == (c, i)
                p.setBrush(QColor(theme.T["accent"]) if sel else col)
                p.setPen(QPen(QColor(theme.T["line"]), 1))
                p.drawRect(QRectF(sx - 3, sy - 3, 6, 6))

    def _hit(self, pos):
        n = self.ed.node()
        if n is None:
            return None
        for c, comp in enumerate(n.keys(self.ed.target[1])):
            for i, k in enumerate(comp):
                sx, sy = self.to_screen(k[0], k[1])
                if abs(sx - pos.x()) < 6 and abs(sy - pos.y()) < 6:
                    return c, i
        return None

    def mousePressEvent(self, e):
        n = self.ed.node()
        if n is None:
            return
        if e.button() == Qt.MouseButton.MiddleButton:
            self._drag = ("pan", e.position(), list(self.view))
            return
        h = self._hit(e.position())
        self.sel = h
        if h and e.button() == Qt.MouseButton.LeftButton:
            self._drag = ("key", h, n.knob_state(self.ed.target[1]), e.position())
        elif h and e.button() == Qt.MouseButton.RightButton:
            from PySide6.QtWidgets import QMenu
            m = QMenu(self)
            for i, name in enumerate(("constant", "linear", "smooth")):
                m.addAction(name, lambda i=i: self._interp(h, i))
            m.addAction("delete", lambda: self._delete(h))
            m.exec(e.globalPosition().toPoint())
        self.update()

    def _interp(self, h, interp):
        n = self.ed.node()
        st = n.knob_state(self.ed.target[1])
        k = list(st["curves"][h[0]][h[1]])
        k[2] = interp
        st["curves"][h[0]][h[1]] = tuple(k)
        self.ed.s.set_knob_state(n, self.ed.target[1], {"curves": st["curves"]})

    def _delete(self, h):
        n = self.ed.node()
        st = n.knob_state(self.ed.target[1])
        del st["curves"][h[0]][h[1]]
        self.ed.s.set_knob_state(n, self.ed.target[1], {"curves": st["curves"]})
        self.sel = None

    def mouseMoveEvent(self, e):
        if not self._drag:
            return
        if self._drag[0] == "pan":
            _, p0, v = self._drag
            df = (e.position().x() - p0.x()) / max(1, self.width() - 30) * (v[2] - v[0])
            dv = (e.position().y() - p0.y()) / max(1, self.height() - 25) * (v[3] - v[1])
            self.view = [v[0] - df, v[1] + dv, v[2] - df, v[3] + dv]
            self.update()
            return
        _, (c, i), st, p0 = self._drag
        n = self.ed.node()
        f, v = self.to_value(e.position().x(), e.position().y())
        k = list(st["curves"][c][i])
        if not e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            k[0] = float(round(f))
        if not e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            k[1] = v
        curves = [list(cc) for cc in st["curves"]]
        curves[c][i] = tuple(k)
        n.set_knob_state(self.ed.target[1], {"curves": curves})
        self.ed.s.node_changed(n)
        self.update()

    def mouseReleaseEvent(self, e):
        if self._drag and self._drag[0] == "key":
            n = self.ed.node()
            after = n.knob_state(self.ed.target[1])
            n.set_knob_state(self.ed.target[1], self._drag[2])
            self.ed.s.set_knob_state(n, self.ed.target[1], after)
        self._drag = None

    def mouseDoubleClickEvent(self, e):
        n = self.ed.node()
        if n is None:
            return
        f, _ = self.to_value(e.position().x(), e.position().y())
        f = float(round(f))
        before = n.knob_state(self.ed.target[1])
        for c, v in enumerate(n.get(self.ed.target[1], f)):
            if before["curves"][c]:
                n.set_key(self.ed.target[1], c, f, v)
        after = n.knob_state(self.ed.target[1])
        n.set_knob_state(self.ed.target[1], before)
        self.ed.s.set_knob_state(n, self.ed.target[1], after)

    def wheelEvent(self, e):
        f, v = self.to_value(e.position().x(), e.position().y())
        s = 0.85 if e.angleDelta().y() > 0 else 1 / 0.85
        f0, v0, f1, v1 = self.view
        if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.view = [f0, v + (v0 - v) * s, f1, v + (v1 - v) * s]
        else:
            self.view = [f + (f0 - f) * s, v0, f + (f1 - f) * s, v1]
        self.update()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.sel:
            self._delete(self.sel)
        elif e.key() == Qt.Key.Key_F:
            self.fit()
