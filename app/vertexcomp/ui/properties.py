"""Properties panel, generated from the core's knob metadata."""
from __future__ import annotations

import itertools
import os

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from vertexcomp import _vccore as vc
from vertexcomp import movie
from vertexcomp.project import OP_TYPES
from vertexcomp.ui import theme
from vertexcomp.ui.widgets import IconButton, NumberWithSlider, ScrubField, Swatch

_drag_ids = itertools.count()
STRING_TYPES = ("string", "file", "channel", "channels", "text", "enum")
COMP_LABELS = {
    "xy": ["x", "y"], "box": ["x", "y", "r", "t"], "color": ["r", "g", "b", "a"],
}


class RangeBar(QWidget):
    """Handles for 2 or 4 ordered values over a soft range."""
    edited = Signal(list, bool)

    def __init__(self, values, lo, hi, parent=None):
        super().__init__(parent)
        self.values, self.lo, self.hi = list(values), lo, hi
        self.setFixedHeight(16)
        self.setMinimumWidth(80)
        self._drag = None

    def setValues(self, v):
        self.values = list(v)
        self.update()

    def _x(self, v):
        return 5 + (v - self.lo) / ((self.hi - self.lo) or 1) * (self.width() - 10)

    def _v(self, x):
        return self.lo + (x - 5) / max(1, self.width() - 10) * (self.hi - self.lo)

    def paintEvent(self, e):
        p = QPainter(self)
        T = theme.T
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        y = self.height() / 2
        p.setPen(QPen(QColor(T["line"]), 2))
        p.drawLine(QPointF(5, y), QPointF(self.width() - 5, y))
        xs = [self._x(v) for v in self.values]
        if len(xs) == 4:
            pts = [QPointF(xs[0], y + 5), QPointF(xs[1], y - 5), QPointF(xs[2], y - 5), QPointF(xs[3], y + 5)]
            p.setPen(QPen(QColor(T["accent"]), 1))
            p.drawPolyline(pts)
        else:
            p.setPen(QPen(QColor(T["accent"]), 2))
            p.drawLine(QPointF(xs[0], y), QPointF(xs[-1], y))
        p.setPen(QPen(QColor(T["line"]), 1))
        p.setBrush(QColor(T["dim"]))
        for x in xs:
            p.drawRect(QRectF(x - 3, y - 6, 6, 12))

    def mousePressEvent(self, e):
        x = e.position().x()
        self._drag = min(range(len(self.values)), key=lambda i: abs(self._x(self.values[i]) - x))

    def mouseMoveEvent(self, e):
        if self._drag is None:
            return
        v = self._v(e.position().x())
        i = self._drag
        lo = self.values[i - 1] if i > 0 else -1e9
        hi = self.values[i + 1] if i + 1 < len(self.values) else 1e9
        self.values[i] = max(lo, min(hi, v))
        self.update()
        self.edited.emit(list(self.values), False)

    def mouseReleaseEvent(self, e):
        if self._drag is not None:
            self._drag = None
            self.edited.emit(list(self.values), True)


class KeyButton(IconButton):
    def __init__(self, parent=None):
        super().__init__("key", "Keyframe: click to set/remove a key at this frame, right-click for more")
        self.state = 0

    def set_state(self, animated, on_key, expr=False):
        T = theme.T
        if expr:
            self.setIcon(theme.icon("curve", "#9a7fd0"))
        elif on_key:
            self.setIcon(theme.icon("key_on", "#6fa0dc"))
        elif animated:
            self.setIcon(theme.icon("key", "#6fa0dc"))
        else:
            self.setIcon(theme.icon("key", T["dim"]))


class KnobRow(QWidget):
    renderRequested = Signal(object)

    def __init__(self, panel, node, kd):
        super().__init__()
        self.panel, self.s, self.node, self.kd = panel, panel.s, node, kd
        self.name, self.t = kd["name"], kd["type"]
        self._merge = None
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lab = QLabel(kd["label"])
        lab.setObjectName("knobLabel")
        lab.setFixedWidth(108)
        lab.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        tip = kd["tooltip"] or ""
        lab.setToolTip(f"{self.name}{': ' + tip if tip else ''}")
        lay.addWidget(lab)
        self.fields = []
        self.key = None
        self.build(lay)
        if kd["animatable"]:
            self.key = KeyButton()
            self.key.clicked.connect(self.toggle_key)
            self.key.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.key.customContextMenuRequested.connect(self.key_menu)
            lay.addWidget(self.key)
        self.refresh()

    # ------------------------------------------------------------ building
    def build(self, lay):
        kd, t = self.kd, self.t
        if t in ("float", "int"):
            w = NumberWithSlider(0.0, kd["min"], kd["max"], integer=(t == "int"))
            w.valueEdited.connect(lambda v, f: self.commit(v, f))
            self.fields = [w]
            lay.addWidget(w, 1)
        elif t == "bool":
            w = QCheckBox()
            w.toggled.connect(lambda on: self.commit(1.0 if on else 0.0, True))
            self.fields = [w]
            lay.addWidget(w)
            lay.addStretch(1)
        elif t == "enum":
            w = QComboBox()
            w.addItems(kd["options"])
            w.currentTextChanged.connect(lambda s: self.commit(s, True))
            self.fields = [w]
            lay.addWidget(w, 1)
        elif t in ("string",):
            w = QLineEdit()
            w.editingFinished.connect(lambda: self.commit(w.text(), True))
            self.fields = [w]
            lay.addWidget(w, 1)
        elif t == "text":
            w = QPlainTextEdit()
            w.setFixedHeight(54)
            self._text_timer = QTimer(self)
            self._text_timer.setSingleShot(True)
            self._text_timer.setInterval(400)
            self._text_timer.timeout.connect(lambda: self.commit(w.toPlainText(), True))
            w.textChanged.connect(self._text_timer.start)
            self.fields = [w]
            lay.addWidget(w, 1)
        elif t == "file":
            w = QLineEdit()
            w.editingFinished.connect(lambda: self.set_file(w.text()))
            b = IconButton("folder", "Browse")
            b.clicked.connect(self.browse)
            self.fields = [w]
            lay.addWidget(w, 1)
            lay.addWidget(b)
        elif t in ("channel", "channels"):
            w = QComboBox()
            w.setEditable(True)
            w.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            w.lineEdit().editingFinished.connect(lambda: self.commit(w.currentText(), True))
            w.activated.connect(lambda i: self.commit(w.currentText(), True))
            w.showPopup = self._wrap_popup(w)
            self.fields = [w]
            lay.addWidget(w, 1)
        elif t in ("color", "xy", "box", "range"):
            n = kd["comps"]
            labels = COMP_LABELS.get(t, [""] * n)
            if t == "color" and n == 6:
                labels = ["R", "Y", "G", "C", "B", "M"]
            if t == "xy" and kd["name"] in ("format", "size", "scale", "sigma"):
                labels = ["w", "h"]
            grid = QHBoxLayout()
            grid.setSpacing(2)
            span = (kd["max"] - kd["min"]) or 1.0
            for c in range(n):
                f = ScrubField(0.0, step=max(1e-4, span / 200.0), decimals=4)
                f.setFixedWidth(52 if n > 2 else 64)
                f.setToolTip(f"{self.name} {labels[c] if c < len(labels) else c}")
                f.valueEdited.connect(lambda v, fin, c=c: self.commit_comp(c, v, fin))
                self.fields.append(f)
                grid.addWidget(f)
            lay.addLayout(grid)
            if t == "color":
                self.swatch = Swatch()
                self.swatch.clicked.connect(self.pick_color)
                self.swatch.setToolTip("Pick a colour (converted to linear)")
                lay.addWidget(self.swatch)
                if n <= 4:
                    self.all = ScrubField(0.0, step=max(1e-4, span / 200.0), decimals=4)
                    self.all.setFixedWidth(44)
                    self.all.setToolTip("Set all components")
                    self.all.valueEdited.connect(self.commit_all)
                    lay.addWidget(self.all)
            if t == "range" and n in (2, 4):
                self.rangebar = RangeBar([0.0] * n, kd["min"], kd["max"])
                self.rangebar.edited.connect(lambda v, fin: self.commit(v, fin))
                lay.addWidget(self.rangebar, 1)
            else:
                lay.addStretch(1)
        elif t == "button":
            b = QPushButton(kd["label"])
            b.clicked.connect(self.button)
            lay.addWidget(b)
            lay.addStretch(1)

    def _wrap_popup(self, combo):
        orig = combo.showPopup

        def show():
            self.fill_channels(combo)
            orig()
        return show

    def fill_channels(self, combo):
        cur = combo.currentText()
        chans = []
        try:
            src = self.node.inputs[0] if self.node.inputs and self.node.inputs[0] is not None else None
            if self.node.type in ("Copy", "Merge") and len(self.node.inputs) > 1 and self.node.inputs[1] is not None \
                    and self.name.startswith(("from", "A", "layer")):
                src = self.node.inputs[1]
            target = src if src is not None else self.node
            chans = self.s.project.engine.info(target, self.s.frame)["channels"]
        except Exception:
            pass
        items = []
        if self.t == "channels":
            items = ["rgba", "rgb", "alpha", "all", "none"]
            for c in chans:
                ly = c.rsplit(".", 1)[0]
                if ly not in items:
                    items.append(ly)
        else:
            items = list(chans)
            if self.node.type == "Shuffle":
                items += ["0", "1", "-"]
                if len(self.node.inputs) > 1 and self.node.inputs[1] is not None:
                    try:
                        items += ["in2:" + c for c in self.s.project.engine.info(self.node.inputs[1], self.s.frame)["channels"]]
                    except Exception:
                        pass
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(items)
        combo.setCurrentText(cur)
        combo.blockSignals(False)

    # ------------------------------------------------------------ values
    def value(self):
        return self.node.get(self.name, float(self.s.frame))

    def refresh(self):
        if self.t == "button":
            return
        v = self.value()
        t = self.t
        for f in self.fields:
            f.blockSignals(True)
        if t in ("float", "int"):
            self.fields[0].setValue(v[0])
        elif t == "bool":
            self.fields[0].setChecked(v[0] != 0)
        elif t == "enum":
            self.fields[0].setCurrentText(v)
        elif t in ("string", "file"):
            if not self.fields[0].hasFocus():
                self.fields[0].setText(v)
        elif t == "text":
            if not self.fields[0].hasFocus():
                self.fields[0].setPlainText(v)
        elif t in ("channel", "channels"):
            if not self.fields[0].lineEdit().hasFocus():
                self.fields[0].setEditText(v)
        else:
            for c, f in enumerate(self.fields):
                f.setValue(v[c])
            if hasattr(self, "swatch"):
                self.swatch.setLinear(v[:3] if len(v) >= 3 else v * 3)
            if hasattr(self, "rangebar") and not self.rangebar._drag:
                self.rangebar.setValues(v)
            if hasattr(self, "all"):
                self.all.setValue(v[0])
        for f in self.fields:
            f.blockSignals(False)
        if self.key is not None:
            anim = self.node.is_animated(self.name)
            on = anim and any(any(abs(k[0] - self.s.frame) < 1e-6 for k in comp) for comp in self.node.keys(self.name))
            expr = any(self.node.exprs(self.name))
            self.key.set_state(anim, on, expr)
            col = "#6fa0dc" if anim else ("#9a7fd0" if expr else "")
            for f in self.fields:
                target = f.field if isinstance(f, NumberWithSlider) else f
                target.setStyleSheet(f"color: {col};" if col else "")

    def _merge_key(self, final):
        if self._merge is None:
            self._merge = f"{self.node.name}.{self.name}.{next(_drag_ids)}"
        key = self._merge
        if final:
            self._merge = None
        return key

    def commit(self, value, final):
        self.s.set_knob(self.node, self.name, value, merge_key=self._merge_key(final))
        if self.t == "enum" or self.t == "bool":
            self.panel.refresh_node(self.node)

    def commit_comp(self, c, v, final):
        vals = list(self.value())
        vals[c] = v
        self.s.set_knob(self.node, self.name, vals, merge_key=self._merge_key(final), comp=c)

    def commit_all(self, v, final):
        n = self.kd["comps"]
        vals = [v] * n
        if self.t == "color" and n == 4 and self.name not in ("value", "minimum", "maximum"):
            vals[3] = self.value()[3]
        self.s.set_knob(self.node, self.name, vals, merge_key=self._merge_key(final))

    def pick_color(self):
        v = self.value()

        def enc(x):
            x = max(0.0, min(1.0, x))
            return x * 12.92 if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055

        def dec(x):
            return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4

        c = QColorDialog.getColor(QColor.fromRgbF(enc(v[0]), enc(v[1]), enc(v[2])), self, "Colour")
        if c.isValid():
            nv = list(v)
            nv[0], nv[1], nv[2] = dec(c.redF()), dec(c.greenF()), dec(c.blueF())
            self.s.set_knob(self.node, self.name, nv)

    def browse(self):
        start = os.path.dirname(self.value()) or self.s.settings.get("paths", "read_dir") or os.path.expanduser("~")
        if self.node.type == "Write":
            path, _ = QFileDialog.getSaveFileName(self, "Output", start)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "File", start,
                                                  "Images and movies (*.exr *.png *.jpg *.jpeg *.tga *.bmp *.hdr *.mov *.mp4 *.mkv *.avi);;All files (*)")
        if path:
            self.set_file(path)

    def set_file(self, path):
        path = path.strip()
        n = self.node
        if n.type == "Read" and path:
            set_read_file(self.s, n, path)
        elif n.type == "Write" and path:
            ext = os.path.splitext(path)[1].lower().lstrip(".")
            self.s.undo.beginMacro("set output")
            self.s.set_knob(n, "file", path)
            if ext in OP_TYPES["Write"]["knobs"][1]["options"]:
                self.s.set_knob(n, "fileType", ext)
            self.s.undo.endMacro()
        else:
            self.s.set_knob(n, self.name, path)
        self.panel.refresh_node(n)

    def button(self):
        if self.name == "reload":
            self.node.revision = self.node.revision + 1
            self.node.clear_error()
            self.s.node_changed(self.node)
        elif self.name == "render":
            self.panel.renderRequested.emit(self.node)

    # ------------------------------------------------------------ animation
    def toggle_key(self):
        on = any(any(abs(k[0] - self.s.frame) < 1e-6 for k in comp) for comp in self.node.keys(self.name))
        self.s.set_key(self.node, self.name, not on)

    def key_menu(self, pos):
        m = QMenu(self)
        m.addAction("Set key", lambda: self.s.set_key(self.node, self.name, True))
        m.addAction("Delete key", lambda: self.s.set_key(self.node, self.name, False))
        m.addAction("Clear animation", self.clear_anim)
        interp = m.addMenu("Interpolation at this key")
        for i, name in enumerate(("constant", "linear", "smooth")):
            interp.addAction(name, lambda i=i: self.set_interp(i))
        m.addSeparator()
        m.addAction("Edit expression", self.edit_expr)
        m.addAction("Curve editor", lambda: self.panel.curveRequested.emit(self.node, self.name))
        m.exec(self.key.mapToGlobal(pos))

    def clear_anim(self):
        st = self.node.knob_state(self.name)
        v = self.value()
        self.s.set_knob_state(self.node, self.name, {"v": v, "curves": [[] for _ in st["curves"]]})

    def set_interp(self, interp):
        st = self.node.knob_state(self.name)
        curves = []
        for comp in st["curves"]:
            curves.append([(k[0], k[1], interp if abs(k[0] - self.s.frame) < 1e-6 else k[2], k[3], k[4]) for k in comp])
        self.s.set_knob_state(self.node, self.name, {"curves": curves})

    def edit_expr(self):
        cur = self.node.exprs(self.name)
        dlg = QDialog(self)
        dlg.setWindowTitle(f"{self.node.name}.{self.name} expression")
        form = QFormLayout(dlg)
        edits = []
        labels = COMP_LABELS.get(self.t, [str(i) for i in range(self.kd["comps"])])
        for c in range(self.kd["comps"]):
            e = QLineEdit(cur[c])
            e.setFont(theme.mono_font(9))
            e.setPlaceholderText("e.g. sin(frame / 10) * 5 + Blur1.size")
            form.addRow(labels[c] if c < len(labels) else str(c), e)
            edits.append(e)
        err = QLabel("")
        err.setStyleSheet(f"color: {theme.T['error']};")
        form.addRow(err)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        form.addRow(bb)

        def ok():
            before = self.node.knob_state(self.name)
            try:
                for c, e in enumerate(edits):
                    self.node.set_expr(self.name, c, e.text().strip())
            except RuntimeError as ex:
                self.node.set_knob_state(self.name, before)
                err.setText(str(ex))
                return
            after = self.node.knob_state(self.name)
            self.node.set_knob_state(self.name, before)
            self.s.set_knob_state(self.node, self.name, after)
            dlg.accept()

        bb.accepted.connect(ok)
        bb.rejected.connect(dlg.reject)
        dlg.exec()


def set_read_file(session, node, path):
    """Point a Read at a file: detect sequences and movie length."""
    s = session
    s.undo.beginMacro("set file")
    try:
        if vc.is_movie(path):
            info = movie.movie_info(path)
            s.set_knob(node, "file", path)
            if info:
                frames = info[0]
                s.set_knob(node, "first", 0)
                s.set_knob(node, "last", max(0, frames - 1))
                s.set_knob(node, "offset", s.project.settings.first)
        else:
            pat, first, last, isseq = vc.detect_sequence(path)
            s.set_knob(node, "file", pat if isseq else path)
            s.set_knob(node, "first", first)
            s.set_knob(node, "last", last)
        s.settings.set("paths", "read_dir", os.path.dirname(path))
    finally:
        s.undo.endMacro()


class NodeEditor(QFrame):
    def __init__(self, panel, node):
        super().__init__()
        self.panel, self.s, self.node = panel, panel.s, node
        self.setFrameShape(QFrame.Shape.NoFrame)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 6)
        lay.setSpacing(2)

        head = QFrame()
        head.setObjectName("nodeHeader")
        hl = QHBoxLayout(head)
        hl.setContentsMargins(4, 2, 4, 2)
        hl.setSpacing(4)
        self.name_edit = QLineEdit(node.name)
        self.name_edit.setFixedWidth(130)
        self.name_edit.editingFinished.connect(lambda: self.s.rename(node, self.name_edit.text().strip()))
        typ = QLabel(node.type)
        typ.setObjectName("knobLabel")
        self.disable = QCheckBox("disable")
        self.disable.setChecked(node.disabled)
        self.disable.toggled.connect(lambda on: self.s.toggle_disable([node]) if on != node.disabled else None)
        close = IconButton("minus", "Close")
        close.clicked.connect(lambda: panel.close_editor(node))
        hl.addWidget(self.name_edit)
        hl.addWidget(typ)
        hl.addStretch(1)
        hl.addWidget(self.disable)
        hl.addWidget(close)
        lay.addWidget(head)
        self.err = QLabel("")
        self.err.setWordWrap(True)
        self.err.setStyleSheet(f"color: {theme.T['error']}; padding: 2px 6px;")
        self.err.setVisible(False)
        lay.addWidget(self.err)

        self.rows: list[KnobRow] = []
        groups: dict[str, list] = {}
        for kd in OP_TYPES[node.type]["knobs"]:
            groups.setdefault(kd["group"], []).append(node.knob_def(kd["name"]))
        body = QVBoxLayout()
        body.setContentsMargins(4, 2, 4, 2)
        body.setSpacing(2)
        for g, kds in groups.items():
            if g:
                h = QLabel(g)
                h.setObjectName("sectionHeader")
                body.addWidget(h)
            for kd in kds:
                r = KnobRow(panel, node, kd)
                self.rows.append(r)
                body.addWidget(r)
        lay.addLayout(body)
        if node.type == "Roto":
            from vertexcomp.ui.roto_tool import RotoPanel
            self.roto = RotoPanel(self.s, node)
            lay.addWidget(self.roto)
        if node.type == "Read":
            self.meta = QLabel("")
            self.meta.setFont(theme.mono_font(8))
            self.meta.setWordWrap(True)
            self.meta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.meta.setStyleSheet("padding: 4px 8px;")
            lay.addWidget(self.meta)
        self.refresh()

    def refresh(self):
        for r in self.rows:
            r.refresh()
        e = self.node.error
        self.err.setText(e)
        self.err.setVisible(bool(e))
        self.disable.blockSignals(True)
        self.disable.setChecked(self.node.disabled)
        self.disable.blockSignals(False)
        if hasattr(self, "meta"):
            try:
                i = self.s.project.engine.info(self.node, self.s.frame)
                fw, fh, par = i["format"]
                lines = [f"format {fw}x{fh}  par {par:g}", f"bbox {i['bbox']}",
                         f"frames {i['first']:g}-{i['last']:g}", "channels " + " ".join(i["channels"])]
                lines += [f"{k} = {v}" for k, v in sorted(i["meta"].items())]
                self.meta.setText("\n".join(lines))
            except Exception as ex:
                self.meta.setText(str(ex))
        if hasattr(self, "roto"):
            self.roto.refresh()


class PropertiesPanel(QScrollArea):
    renderRequested = Signal(object)
    curveRequested = Signal(object, str)
    MAX = 4

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.s = session
        self.setWidgetResizable(True)
        self.inner = QWidget()
        self.lay = QVBoxLayout(self.inner)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(0)
        self.lay.addStretch(1)
        self.setWidget(self.inner)
        self.editors: dict[str, NodeEditor] = {}
        self.order: list[str] = []
        session.selectionChanged.connect(self.sync)
        session.graphChanged.connect(self.rebuild)
        session.nodeChanged.connect(self.refresh_node)
        session.frameChanged.connect(lambda f: self.refresh_all())
        self._pending = set()
        self._t = QTimer(self)
        self._t.setSingleShot(True)
        self._t.setInterval(30)
        self._t.timeout.connect(self._flush)

    def sync(self):
        for n in self.s.selected:
            if n.name not in self.editors:
                self.open(n)
            else:
                # bring to top
                self.order.remove(n.name)
                self.order.insert(0, n.name)
        self._relayout()

    def open(self, node):
        ed = NodeEditor(self, node)
        self.editors[node.name] = ed
        self.order.insert(0, node.name)
        while len(self.order) > self.MAX:
            old = self.order.pop()
            w = self.editors.pop(old)
            w.setParent(None)
            w.deleteLater()

    def _relayout(self):
        for name in self.order:
            self.lay.removeWidget(self.editors[name])
        for i, name in enumerate(self.order):
            self.lay.insertWidget(i, self.editors[name])

    def close_editor(self, node):
        name = node.name
        if name in self.editors:
            w = self.editors.pop(name)
            self.order.remove(name)
            w.setParent(None)
            w.deleteLater()

    def rebuild(self):
        names = list(self.order)
        for w in self.editors.values():
            w.setParent(None)
            w.deleteLater()
        self.editors.clear()
        self.order.clear()
        for name in reversed(names):
            n = self.s.project.find(name)
            if n is not None:
                self.open(n)
        self._relayout()
        self.sync()

    def refresh_node(self, node):
        self._pending.add(node.name)
        self._t.start()

    def _flush(self):
        for name in self._pending:
            ed = self.editors.get(name)
            if ed is not None:
                ed.refresh()
        self._pending.clear()

    def refresh_all(self):
        for ed in self.editors.values():
            ed.refresh()
