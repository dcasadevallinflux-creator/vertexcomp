"""Node graph editor (custom QGraphicsScene/View)."""
from __future__ import annotations

import math
import time

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QColor, QCursor, QFont, QKeySequence, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QFrame,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QInputDialog,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QVBoxLayout,
)

from vertexcomp.project import OP_TYPES
from vertexcomp.ui import theme

NODE_W, NODE_H = 96.0, 22.0
PORT = 5.0


def fuzzy_score(query: str, text: str) -> float:
    q, t = query.lower(), text.lower()
    if not q:
        return 1.0
    if t.startswith(q):
        return 100.0 - len(t) * 0.1
    if q in t:
        return 60.0 - t.index(q) - len(t) * 0.1
    i, score, last = 0, 0.0, -1
    for j, ch in enumerate(t):
        if i < len(q) and ch == q[i]:
            score += 3.0 if last == j - 1 else 1.0
            if j == 0 or not t[j - 1].isalpha() or text[j].isupper():
                score += 2.0
            last = j
            i += 1
    return score if i == len(q) else -1.0


class NodeItem(QGraphicsItem):
    def __init__(self, view, node):
        super().__init__()
        self.view, self.node = view, node
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
                      QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setZValue(2)
        self.setAcceptHoverEvents(True)
        self.edges = []
        self.slot_marks: list[int] = []

    def boundingRect(self):
        return QRectF(-4, -14, NODE_W + 30, NODE_H + 34)

    def body(self):
        return QRectF(0, 0, NODE_W, NODE_H)

    def input_pos(self, i):
        n = len(self.node.input_labels)
        labels = self.node.input_labels
        if n and labels[-1] == "mask" and i == n - 1:
            return QPointF(NODE_W + PORT, NODE_H / 2)  # mask input on the right side
        main = n - (1 if n and labels[-1] == "mask" else 0)
        if main <= 1:
            return QPointF(NODE_W / 2, -PORT)
        return QPointF(NODE_W * (i + 1) / (main + 1), -PORT)

    def output_pos(self):
        return QPointF(NODE_W / 2, NODE_H + PORT)

    def port_at(self, scene_pos):
        local = self.mapFromScene(scene_pos)
        for i in range(len(self.node.input_labels)):
            if QLineF(local, self.input_pos(i)).length() < 9:
                return ("in", i)
        if QLineF(local, self.output_pos()).length() < 9:
            return ("out", 0)
        return None

    def category_color(self):
        meta = self.view.s.project.node_meta(self.node)
        if meta.color:
            return QColor(meta.color)
        cat = OP_TYPES[self.node.type]["category"]
        cols = self.view.s.settings.get("nodegraph", "colors")
        return QColor(cols.get(cat, cols.get("Other", "#5c5c5c")))

    def paint(self, p: QPainter, opt, widget=None):
        T = theme.T
        r = self.body()
        cat = self.category_color()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        body = QColor(cat).darker(170) if not self.node.disabled else QColor(T["alt"])
        p.fillRect(r, body)
        p.fillRect(QRectF(0, 0, NODE_W, 4), cat if not self.node.disabled else QColor(T["dim"]))
        err = bool(self.node.error)
        pen = QPen(QColor(T["accent"]) if self.isSelected() else QColor(T["line"]), 1.5 if self.isSelected() else 1)
        if err:
            pen = QPen(QColor(T["error"]), 1.5)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r)
        p.setPen(QColor(T["text"]) if not self.node.disabled else QColor(T["dim"]))
        f = QFont(theme.UI)
        f.setPointSizeF(float(self.view.s.settings.get("nodegraph", "font_size")))
        p.setFont(f)
        p.drawText(r.adjusted(2, 3, -2, 0), Qt.AlignmentFlag.AlignCenter, self.node.name)
        meta = self.view.s.project.node_meta(self.node)
        if meta.label:
            p.setPen(QColor(T["dim"]))
            f2 = QFont(f)
            f2.setPointSizeF(f.pointSizeF() - 1)
            p.setFont(f2)
            p.drawText(QRectF(-20, NODE_H + 6, NODE_W + 40, 14), Qt.AlignmentFlag.AlignCenter, meta.label)
        if self.node.disabled:
            p.setPen(QPen(QColor(T["error"]).darker(130), 1))
            p.drawLine(r.topLeft(), r.bottomRight())
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # Ports
        labels = self.node.input_labels
        for i, lab in enumerate(labels):
            pt = self.input_pos(i)
            connected = i < len(self.node.inputs) and self.node.inputs[i] is not None
            p.setPen(QPen(QColor(T["dim"]), 1))
            p.setBrush(QColor(T["dim"]) if connected else QColor(T["base"]))
            p.drawEllipse(pt, 3, 3)
            if len(labels) > 1 and lab:
                p.setPen(QColor(T["dim"]))
                p.setFont(theme.mono_font(7))
                if lab == "mask" and i == len(labels) - 1:
                    p.drawText(pt + QPointF(5, 3), "m")
                else:
                    p.drawText(pt + QPointF(-3, -5), lab[:2])
        if self.node.type not in ("Write",) or True:
            p.setPen(QPen(QColor(T["dim"]), 1))
            p.setBrush(QColor(T["base"]))
            p.drawRect(QRectF(NODE_W / 2 - 3, NODE_H, 6, 3))
        # Badges
        x = NODE_W + 3
        if err:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(T["error"]))
            p.drawRect(QRectF(x, 1, 8, 8))
            p.setPen(QColor(T["accent_text"]))
            p.setFont(theme.mono_font(7))
            p.drawText(QRectF(x, 0, 8, 10), Qt.AlignmentFlag.AlignCenter, "!")
            x += 10
        if self.slot_marks:
            p.setPen(QColor(T["accent"]))
            p.setFont(theme.mono_font(8))
            p.drawText(QPointF(x, 9), "".join(str(s) for s in self.slot_marks))
        if self.node.type == "Roto" or self._animated():
            pass

    def _animated(self):
        return False

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.view.s.settings.get("nodegraph", "snap_to_grid"):
            g = float(self.view.s.settings.get("nodegraph", "grid_size"))
            return QPointF(round(value.x() / g) * g, round(value.y() / g) * g)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            for e in self.edges:
                e.update_path()
        return super().itemChange(change, value)

    def hoverEnterEvent(self, e):
        tip = OP_TYPES[self.node.type]["help"]
        if self.node.error:
            tip = f"{self.node.error}\n\n{tip}"
        self.setToolTip(tip)


class EdgeItem(QGraphicsPathItem):
    def __init__(self, view, src_item, dst_item, slot):
        super().__init__()
        self.view, self.src, self.dst, self.slot = view, src_item, dst_item, slot
        self.setZValue(1)
        self.setAcceptHoverEvents(True)
        src_item.edges.append(self)
        dst_item.edges.append(self)
        self.update_path()

    def end_points(self):
        a = self.src.mapToScene(self.src.output_pos())
        b = self.dst.mapToScene(self.dst.input_pos(self.slot))
        return a, b

    def update_path(self):
        a, b = self.end_points()
        self.setPath(edge_path(a, b, self.view.s.settings.get("nodegraph", "edge_style"),
                               mask=self.dst.node.input_labels[self.slot] == "mask" and self.slot == len(self.dst.node.input_labels) - 1))

    def paint(self, p, opt, widget=None):
        T = theme.T
        hi = self.src.isSelected() or self.dst.isSelected() or self.isUnderMouse()
        col = QColor(T["accent"]) if hi else QColor(T["dim"])
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(col, 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(self.path())
        a, b = self.end_points()
        draw_arrow(p, self.path(), b, col)
        lab = self.dst.node.input_labels[self.slot]
        if lab and len(self.dst.node.input_labels) > 1:
            pt = self.path().pointAtPercent(0.75)
            p.setPen(col)
            p.setFont(theme.mono_font(7))
            p.drawText(pt + QPointF(4, 0), lab)

    def shape(self):
        QPainterPath()
        st = self.path()
        stroker_path = QPainterPath(st)
        from PySide6.QtGui import QPainterPathStroker
        ps = QPainterPathStroker()
        ps.setWidth(8)
        return ps.createStroke(stroker_path)


def edge_path(a: QPointF, b: QPointF, style: str, mask=False) -> QPainterPath:
    path = QPainterPath(a)
    if style == "straight":
        path.lineTo(b)
    elif style == "orthogonal":
        if mask:
            path.lineTo(QPointF(a.x(), b.y()))
            path.lineTo(b)
        else:
            my = (a.y() + b.y()) / 2
            path.lineTo(QPointF(a.x(), my))
            path.lineTo(QPointF(b.x(), my))
            path.lineTo(b)
    else:
        dy = max(30.0, abs(b.y() - a.y()) * 0.5)
        if mask:
            path.cubicTo(QPointF(a.x(), a.y() + dy), QPointF(b.x() + 40, b.y()), b)
        else:
            path.cubicTo(QPointF(a.x(), a.y() + dy), QPointF(b.x(), b.y() - dy), b)
    return path


def draw_arrow(p, path, tip, col):
    pct = path.percentAtLength(max(0.0, path.length() - 8))
    base = path.pointAtPercent(pct)
    d = tip - base
    ln = math.hypot(d.x(), d.y()) or 1.0
    ux, uy = d.x() / ln, d.y() / ln
    left = QPointF(tip.x() - ux * 7 - uy * 3.2, tip.y() - uy * 7 + ux * 3.2)
    right = QPointF(tip.x() - ux * 7 + uy * 3.2, tip.y() - uy * 7 - ux * 3.2)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(col)
    p.drawPolygon(QPolygonF([tip, left, right]))


class BackdropItem(QGraphicsItem):
    def __init__(self, view, index):
        super().__init__()
        self.view, self.index = view, index
        self.bd = view.s.project.backdrops[index]
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
                      QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setZValue(-1 if not self.bd.note else 3)
        self._resize = None
        self._carry = []
        self.setPos(self.bd.x, self.bd.y)

    def boundingRect(self):
        return QRectF(0, 0, self.bd.w, 24 if self.bd.group and self.bd.collapsed else self.bd.h)

    def paint(self, p, opt, widget=None):
        T = theme.T
        c = QColor(self.bd.color)
        r = self.boundingRect()
        if self.bd.note:
            p.fillRect(r, c)
        else:
            fill = QColor(c)
            fill.setAlpha(150)
            p.fillRect(r, fill)
            p.fillRect(QRectF(0, 0, r.width(), 18), c.lighter(115))
        p.setPen(QPen(QColor(T["accent"]) if self.isSelected() else c.darker(140), 1))
        p.drawRect(r.adjusted(0.5, 0.5, -0.5, -0.5))
        p.setPen(QColor(T["text"]))
        f = QFont(theme.UI)
        f.setPointSizeF(10 if not self.bd.note else 9)
        p.setFont(f)
        if self.bd.note:
            p.drawText(r.adjusted(6, 4, -6, -4), Qt.TextFlag.TextWordWrap, self.bd.title)
        else:
            prefix = "[+] " if self.bd.group and self.bd.collapsed else "[-] " if self.bd.group else ""
            p.drawText(QRectF(6, 0, r.width() - 12, 18), Qt.AlignmentFlag.AlignVCenter, prefix + self.bd.title)
        p.setPen(QPen(c.darker(150), 1))
        p.drawLine(QPointF(r.right() - 9, r.bottom() - 2), QPointF(r.right() - 2, r.bottom() - 9))

    def mousePressEvent(self, e):
        r = self.boundingRect()
        if e.pos().x() > r.width() - 12 and e.pos().y() > r.height() - 12:
            self._resize = (e.scenePos(), self.bd.w, self.bd.h)
            e.accept()
            return
        if not self.bd.note:
            area = self.sceneBoundingRect()
            if self.bd.group:
                members = set(self.bd.members)
                self._carry = [it for it in self.view.node_items.values() if it.node.name in members]
            else:
                self._carry = [it for it in self.view.node_items.values() if area.contains(it.sceneBoundingRect().center())]
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._resize:
            p0, w0, h0 = self._resize
            d = e.scenePos() - p0
            self.prepareGeometryChange()
            self.bd.w, self.bd.h = max(60, w0 + d.x()), max(30, h0 + d.y())
            self.update()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._resize = None
        self._carry = []
        super().mouseReleaseEvent(e)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            old = QPointF(self.bd.x, self.bd.y)
            d = value - old
            self.bd.x, self.bd.y = value.x(), value.y()
            for it in self._carry:
                if not it.isSelected():
                    it.setPos(it.pos() + d)
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, e):
        t, ok = QInputDialog.getMultiLineText(None, "Backdrop", "Title", self.bd.title) if self.bd.note else \
            QInputDialog.getText(None, "Backdrop", "Title", text=self.bd.title)
        if ok:
            self.bd.title = t
            self.update()


class CreateMenu(QFrame):
    """Tab menu: fuzzy search over node types."""
    chosen = Signal(str)

    def __init__(self, parent):
        super().__init__(parent, Qt.WindowType.Popup)
        self.setFrameShape(QFrame.Shape.Box)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("node")
        self.list = QListWidget()
        self.list.setFixedHeight(220)
        lay.addWidget(self.edit)
        lay.addWidget(self.list)
        self.setFixedWidth(220)
        self.edit.textChanged.connect(self._filter)
        self.edit.returnPressed.connect(self._accept)
        self.list.itemActivated.connect(lambda it: self._accept())
        self.edit.installEventFilter(self)
        self._filter("")

    def eventFilter(self, obj, e):
        from PySide6.QtCore import QEvent
        if e.type() == QEvent.Type.KeyPress and e.key() in (Qt.Key.Key_Down, Qt.Key.Key_Up):
            r = self.list.currentRow() + (1 if e.key() == Qt.Key.Key_Down else -1)
            self.list.setCurrentRow(max(0, min(self.list.count() - 1, r)))
            return True
        if e.type() == QEvent.Type.KeyPress and e.key() == Qt.Key.Key_Escape:
            self.close()
            return True
        return False

    def _filter(self, q):
        self.list.clear()
        scored = []
        for t, o in OP_TYPES.items():
            s = fuzzy_score(q, t)
            if s >= 0:
                scored.append((-s, t, o["category"]))
        for _, t, _cat in sorted(scored)[:40]:
            it = QListWidgetItem(f"{t}")
            it.setData(Qt.ItemDataRole.UserRole, t)
            it.setToolTip(OP_TYPES[t]["help"])
            self.list.addItem(it)
        if self.list.count():
            self.list.setCurrentRow(0)

    def _accept(self):
        it = self.list.currentItem()
        if it:
            self.chosen.emit(it.data(Qt.ItemDataRole.UserRole))
        self.close()

    def popup_at(self, pos):
        self.edit.clear()
        self._filter("")
        self.move(pos)
        self.show()
        self.edit.setFocus()


class NodeGraphView(QGraphicsView):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.s = session
        self.sc = QGraphicsScene(self)
        self.sc.setSceneRect(-100000, -100000, 200000, 200000)
        self.setScene(self.sc)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.node_items: dict[str, NodeItem] = {}
        self.edge_items: list[EdgeItem] = []
        self.backdrop_items: list[BackdropItem] = []
        self._pan = None
        self._wire = None  # (kind, item, slot, temp path item)
        self._move_start = None
        self._move_samples = {}
        self._sync_sel = False
        self.creator = CreateMenu(self)
        self.creator.chosen.connect(self._create_from_menu)
        self._pending_wire = None
        self.sc.selectionChanged.connect(self._scene_selection)
        session.graphChanged.connect(self.rebuild)
        session.layoutChanged.connect(self.relayout)
        session.nodeChanged.connect(self._node_changed)
        session.selectionChanged.connect(self._session_selection)
        session.viewedChanged.connect(self._update_marks)
        session.renderStateChanged.connect(lambda: self.viewport().update())
        self._actions()
        self.rebuild()

    # ------------------------------------------------------------ building
    def rebuild(self):
        self._sync_sel = True
        self.sc.clear()
        self.node_items.clear()
        self.edge_items.clear()
        self.backdrop_items.clear()
        for i, _ in enumerate(self.s.project.backdrops):
            it = BackdropItem(self, i)
            self.sc.addItem(it)
            self.backdrop_items.append(it)
        for n in self.s.project.nodes():
            it = NodeItem(self, n)
            m = self.s.project.node_meta(n)
            it.setPos(m.x, m.y)
            self.sc.addItem(it)
            self.node_items[n.name] = it
        collapsed = {name for bd in self.s.project.backdrops if bd.group and bd.collapsed for name in bd.members}
        for name in collapsed:
            if name in self.node_items:
                self.node_items[name].setVisible(False)
        for n in self.s.project.nodes():
            for slot, src in enumerate(n.inputs):
                if src is not None and src.name in self.node_items:
                    e = EdgeItem(self, self.node_items[src.name], self.node_items[n.name], slot)
                    e.setVisible(self.node_items[src.name].isVisible() and self.node_items[n.name].isVisible())
                    self.sc.addItem(e)
                    self.edge_items.append(e)
        sel = {n.name for n in self.s.selected}
        for name, it in self.node_items.items():
            it.setSelected(name in sel)
        self._sync_sel = False
        self._update_marks()

    def relayout(self):
        for n in self.s.project.nodes():
            it = self.node_items.get(n.name)
            if it:
                m = self.s.project.node_meta(n)
                it.setPos(m.x, m.y)
        for b in self.backdrop_items:
            b.setPos(b.bd.x, b.bd.y)

    def _update_marks(self):
        for it in self.node_items.values():
            it.slot_marks = sorted(k for k, v in self.s.viewer_slots.items() if v == it.node.name)
            it.update()

    def _node_changed(self, node):
        it = self.node_items.get(node.name)
        if it:
            it.update()

    def _scene_selection(self):
        if self._sync_sel:
            return
        names = [it.node.name for it in self.sc.selectedItems() if isinstance(it, NodeItem)]
        self.s.selected = [self.node_items[n].node for n in names]
        self.s.selectionChanged.emit()

    def _session_selection(self):
        self._sync_sel = True
        sel = {n.name for n in self.s.selected}
        for name, it in self.node_items.items():
            it.setSelected(name in sel)
        self._sync_sel = False

    # ------------------------------------------------------------ actions
    def _actions(self):
        def add(seq, fn):
            a = QAction(self)
            a.setShortcut(QKeySequence(seq))
            a.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
            a.triggered.connect(fn)
            self.addAction(a)

        hk = self.s.settings.data["hotkeys"]
        add(QKeySequence.StandardKey.Copy, self.copy)
        add(QKeySequence.StandardKey.Paste, self.paste)
        add(QKeySequence.StandardKey.Cut, self.cut)
        add("Ctrl+D", self.duplicate)
        add("Delete", self.delete_selected)
        add("Backspace", self.delete_selected)
        add(hk.get("disable", "D"), lambda: self.s.toggle_disable(self.s.selected) if self.s.selected else None)
        add(hk.get("fit_view", "F"), self.fit)
        add(hk.get("rename", "N"), self.rename_selected)
        add("Ctrl+B", self.add_backdrop)
        add("Ctrl+G", self.add_subgroup)
        add("L", self.auto_align)
        for i in range(1, 10):
            add(str(i), lambda i=i: self._view_slot(i))
        for t in OP_TYPES:
            if t in hk and hk[t]:
                add(hk[t], lambda t=t: self.create(t))

    def _view_slot(self, i):
        if self.s.selected:
            self.s.view_node(self.s.selected[-1], i)
        else:
            self.s.active_slot = i
            self.s.viewedChanged.emit()

    def create(self, op_type, scene_pos=None):
        if scene_pos is not None:
            n = self.s.create_node(op_type, scene_pos.x(), scene_pos.y(), connect_selected=False)
        else:
            n = self.s.create_node(op_type)
        return n

    def _create_from_menu(self, op_type):
        pos = self.mapToScene(self.mapFromGlobal(QCursor.pos()))
        wire = self._pending_wire
        self._pending_wire = None
        if wire is not None:
            kind, src, slot = wire
            n = self.s.create_node(op_type, pos.x() - NODE_W / 2, pos.y(), connect_selected=False)
            if kind == "out" and n.input_labels:
                self.s.connect(n, 0, src)
            elif kind == "in":
                self.s.connect(src, slot, n)
            return
        if self.s.selected:
            self.s.create_node(op_type)
        else:
            self.s.create_node(op_type, pos.x() - NODE_W / 2, pos.y())

    def copy(self):
        if self.s.selected:
            QApplication.clipboard().setText(self.s.copy_text(self.s.selected))

    def cut(self):
        self.copy()
        self.delete_selected()

    def paste(self):
        pos = self.mapToScene(self.mapFromGlobal(QCursor.pos()))
        self.s.paste_text(QApplication.clipboard().text(), (pos.x(), pos.y()))

    def duplicate(self):
        if not self.s.selected:
            return
        m = self.s.project.node_meta(self.s.selected[0])
        self.s.paste_text(self.s.copy_text(self.s.selected), (m.x + 40, m.y + 40))

    def delete_selected(self):
        bds = [b.index for b in self.backdrop_items if b.isSelected()]
        self.s.delete_nodes(list(self.s.selected), bds)

    def rename_selected(self):
        if not self.s.selected:
            return
        n = self.s.selected[-1]
        name, ok = QInputDialog.getText(self, "Rename", "Name", text=n.name)
        if ok:
            self.s.rename(n, name.strip())

    def add_backdrop(self, note=False):
        from vertexcomp.project import Backdrop
        items = [self.node_items[n.name] for n in self.s.selected if n.name in self.node_items]
        if items:
            r = QRectF()
            for it in items:
                r = r.united(it.sceneBoundingRect())
            r = r.adjusted(-20, -40, 20, 20)
        else:
            c = self.mapToScene(self.viewport().rect().center())
            r = QRectF(c.x() - 150, c.y() - 100, 300, 200) if not note else QRectF(c.x() - 80, c.y() - 40, 160, 80)

        def do():
            self.s.project.backdrops.append(Backdrop("Note" if note else "Backdrop", r.x(), r.y(), r.width(), r.height(),
                                                     "#6b6545" if note else "#4a4a44", note))
        self.s.structural("add backdrop", do)

    def add_subgroup(self):
        members = [n.name for n in self.s.selected]
        if not members:
            self.s.log("select nodes before creating a subgroup", "warning")
            return
        items = [self.node_items[name] for name in members if name in self.node_items]
        r = QRectF()
        for it in items:
            r = r.united(it.sceneBoundingRect())
        r = r.adjusted(-20, -40, 20, 20)
        from vertexcomp.project import Backdrop

        def do():
            self.s.project.backdrops.append(
                Backdrop("Subgroup", r.x(), r.y(), r.width(), r.height(), "#3f5666", False, True, members)
            )

        self.s.structural("group nodes", do)

    def auto_align(self):
        sel = [self.node_items[n.name] for n in self.s.selected if n.name in self.node_items]
        if len(sel) < 2:
            return
        before = {it.node.name: (it.pos().x(), it.pos().y(), "node") for it in sel}
        g = 16.0
        x = round(sum(it.pos().x() for it in sel) / len(sel) / g) * g
        for it in sel:
            it.setPos(x, it.pos().y())
            m = self.s.project.node_meta(it.node)
            m.x, m.y = it.pos().x(), it.pos().y()
        after = {it.node.name: (it.pos().x(), it.pos().y(), "node") for it in sel}
        self.s.move_nodes(before, after)

    def fit(self):
        r = QRectF()
        for it in list(self.node_items.values()) + self.backdrop_items:
            r = r.united(it.sceneBoundingRect())
        if r.isNull():
            return
        self.fitInView(r.adjusted(-40, -40, 40, 40), Qt.AspectRatioMode.KeepAspectRatio)
        if self.transform().m11() > 1.5:
            self.resetTransform()
            self.centerOn(r.center())

    # ------------------------------------------------------------ painting
    def drawBackground(self, p, rect):
        T = theme.T
        p.fillRect(rect, QColor(T["base"]))
        g = float(self.s.settings.get("nodegraph", "grid_size")) * 4
        if self.transform().m11() < 0.3:
            return
        p.setPen(QPen(QColor(T["bg"]), 1))
        x = math.floor(rect.left() / g) * g
        while x < rect.right():
            p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += g
        y = math.floor(rect.top() / g) * g
        while y < rect.bottom():
            p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            y += g

    # ------------------------------------------------------------ input
    def wheelEvent(self, e):
        f = 1.15 ** (e.angleDelta().y() / 120.0)
        s = self.transform().m11() * f
        if 0.1 < s < 4:
            self.scale(f, f)

    def _item_at(self, pos):
        for it in self.items(pos):
            if isinstance(it, NodeItem):
                return it
        return None

    def mousePressEvent(self, e):
        sp = self.mapToScene(e.position().toPoint())
        if e.button() == Qt.MouseButton.MiddleButton or (
                e.button() == Qt.MouseButton.LeftButton and e.modifiers() & Qt.KeyboardModifier.AltModifier):
            self._pan = e.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if e.button() == Qt.MouseButton.LeftButton:
            # Ports first (they extend outside the node body).
            for it in self.node_items.values():
                port = it.port_at(sp)
                if port:
                    kind, slot = port
                    if kind == "in" and it.node.inputs[slot] is not None:
                        # Pick up an existing connection from its input end.
                        src = it.node.inputs[slot]
                        self._start_wire("out", self.node_items[src.name], 0, sp, detach=(it, slot))
                    else:
                        self._start_wire(kind, it, slot, sp)
                    return
            it = self._item_at(e.position().toPoint())
            if it and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.s.view_node(it.node)
                return
            # Click on an edge near its arrow: detach for re-wiring.
            for ed in self.edge_items:
                a, b = ed.end_points()
                if QLineF(sp, b).length() < 10:
                    self._start_wire("out", ed.src, 0, sp, detach=(ed.dst, ed.slot))
                    return
            self._move_start = {name: (i.pos().x(), i.pos().y(), "node") for name, i in self.node_items.items()}
            self._move_start.update({str(b.index): (b.bd.x, b.bd.y, "bd") for b in self.backdrop_items})
            started = time.monotonic()
            self._move_samples = {
                name: [(started, x, y)]
                for name, (x, y, kind) in self._move_start.items() if kind == "node"
            }
        super().mousePressEvent(e)

    def _start_wire(self, kind, item, slot, sp, detach=None):
        path = QGraphicsPathItem()
        path.setPen(QPen(QColor(theme.T["accent"]), 1.2, Qt.PenStyle.DashLine))
        path.setZValue(5)
        self.sc.addItem(path)
        self._wire = (kind, item, slot, path, detach)
        if detach:
            for ed in self.edge_items:
                if ed.dst is detach[0] and ed.slot == detach[1]:
                    ed.hide()
        self._update_wire(sp)

    def _update_wire(self, sp):
        kind, item, slot, path, _ = self._wire
        style = self.s.settings.get("nodegraph", "edge_style")
        if kind == "out":
            path.setPath(edge_path(item.mapToScene(item.output_pos()), sp, style))
        else:
            path.setPath(edge_path(sp, item.mapToScene(item.input_pos(slot)), style))

    def mouseMoveEvent(self, e):
        if self._pan is not None:
            d = e.position() - self._pan
            self._pan = e.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - d.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - d.y()))
            return
        if self._wire is not None:
            self._update_wire(self.mapToScene(e.position().toPoint()))
            return
        super().mouseMoveEvent(e)
        self._record_move_sample()

    def _record_move_sample(self):
        if self._move_start is None:
            return
        now = time.monotonic()
        for name, item in self.node_items.items():
            path = self._move_samples.get(name)
            if path is None:
                continue
            point = (now, item.pos().x(), item.pos().y())
            if (point[1], point[2]) != (path[-1][1], path[-1][2]):
                path.append(point)
                if len(path) > 100:
                    del path[1]

    def _jittered_nodes(self):
        grid = float(self.s.settings.get("nodegraph", "grid_size"))
        result = []
        for name, samples in self._move_samples.items():
            if len(samples) < 4:
                continue
            duration = samples[-1][0] - samples[0][0]
            if duration <= 0 or duration > 0.8:
                continue
            vectors = []
            distance = 0.0
            for a, b in zip(samples, samples[1:]):
                dx, dy = b[1] - a[1], b[2] - a[2]
                length = math.hypot(dx, dy)
                if length < grid * 0.25:
                    continue
                vectors.append((dx / length, dy / length))
                distance += length
            reversals = sum(1 for a, b in zip(vectors, vectors[1:]) if a[0] * b[0] + a[1] * b[1] < -0.5)
            if distance >= grid * 4 and distance / duration >= grid * 5 and reversals >= 2:
                item = self.node_items.get(name)
                if item is not None:
                    result.append(item.node)
        return result

    def mouseReleaseEvent(self, e):
        if self._pan is not None:
            self._pan = None
            self.unsetCursor()
            return
        if self._wire is not None:
            kind, item, slot, path, detach = self._wire
            self._wire = None
            self.sc.removeItem(path)
            sp = self.mapToScene(e.position().toPoint())
            target = self._item_at(e.position().toPoint())
            port = None
            for it in self.node_items.values():
                port = it.port_at(sp)
                if port:
                    target = it
                    break
            if kind == "out":
                src = item.node
                if target is not None and target is not item:
                    if port and port[0] == "in":
                        tslot = port[1]
                    else:
                        free = [i for i, x in enumerate(target.node.inputs) if x is None]
                        tslot = free[0] if free else (0 if target.node.inputs else None)
                    if tslot is not None:
                        if detach and (detach[0] is not target or detach[1] != tslot):
                            before = self.s.snapshot()
                            self.s.project.graph.connect(detach[0].node, detach[1], None)
                            if self.s.project.graph.would_cycle(target.node, src):
                                self.s.restore(before)
                                self.s.log("connection refused: it would create a cycle", "warning")
                                return
                            self.s.project.graph.connect(target.node, tslot, src)
                            from vertexcomp.ui.session import SnapshotCommand
                            self.s.undo.push(SnapshotCommand(self.s, "reconnect", before, self.s.snapshot()))
                            self.s.graphChanged.emit()
                        else:
                            self.s.connect(target.node, tslot, src)
                        return
                elif target is None and detach:
                    self.s.connect(detach[0].node, detach[1], None)
                    return
                elif target is None:
                    self._pending_wire = ("out", src, 0)
                    self.creator.popup_at(e.globalPosition().toPoint())
                    return
            else:
                if target is not None and target is not item:
                    self.s.connect(item.node, slot, target.node)
                    return
                if target is None:
                    self._pending_wire = ("in", item.node, slot)
                    self.creator.popup_at(e.globalPosition().toPoint())
                    return
            self.rebuild()
            return
        super().mouseReleaseEvent(e)
        self._record_move_sample()
        if self._move_start is not None:
            after = {name: (i.pos().x(), i.pos().y(), "node") for name, i in self.node_items.items()}
            after.update({str(b.index): (b.bd.x, b.bd.y, "bd") for b in self.backdrop_items})
            moved_before = {k: v for k, v in self._move_start.items() if after.get(k) != v}
            if moved_before:
                for name, (x, y, kind) in after.items():
                    if kind == "node":
                        n = self.node_items[name].node
                        m = self.s.project.node_meta(n)
                        m.x, m.y = x, y
                self.s.move_nodes(moved_before, {k: after[k] for k in moved_before})
                jittered = self._jittered_nodes()
                if jittered:
                    self.s.disconnect_nodes(jittered)
            self._move_start = None
            self._move_samples = {}

    def mouseDoubleClickEvent(self, e):
        it = self._item_at(e.position().toPoint())
        if it:
            self.s.select([it.node])
            self.s.view_node(it.node) if not self.s.viewer_slots else None
            return
        super().mouseDoubleClickEvent(e)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Tab:
            self.creator.popup_at(QCursor.pos())
            return
        super().keyPressEvent(e)

    def event(self, e):
        from PySide6.QtCore import QEvent
        if e.type() == QEvent.Type.KeyPress and e.key() == Qt.Key.Key_Tab:
            self.keyPressEvent(e)
            return True
        return super().event(e)

    def contextMenuEvent(self, e):
        sp = self.mapToScene(e.pos())
        backdrop = next((it for it in self.items(e.pos()) if isinstance(it, BackdropItem)), None)
        it = self._item_at(e.pos())
        if it and it.node.name not in {n.name for n in self.s.selected}:
            self.s.select([it.node])
        menu = QMenu(self)
        cats = {}
        for t, o in sorted(OP_TYPES.items()):
            cats.setdefault(o["category"], []).append(t)
        for cat in sorted(cats):
            sub = menu.addMenu(cat)
            for t in cats[cat]:
                sub.addAction(t, lambda t=t: self.s.create_node(t, sp.x(), sp.y(), connect_selected=bool(self.s.selected)))
        menu.addSeparator()
        if self.s.selected:
            n = self.s.selected[-1]
            menu.addAction("View", lambda: self.s.view_node(n))
            menu.addAction("Rename", self.rename_selected)
            menu.addAction("Disable / enable", lambda: self.s.toggle_disable(self.s.selected))
            menu.addAction("Node colour", self.pick_color)
            menu.addAction("Label", self.edit_label)
            menu.addAction("Force re-render", self.force_rerender)
            if n.type == "Read":
                menu.addAction("Create Shuffle per layer", lambda: self.shuffle_per_layer(n))
            menu.addSeparator()
            menu.addAction("Copy", self.copy)
            menu.addAction("Duplicate", self.duplicate)
            menu.addAction("Delete", self.delete_selected)
        menu.addAction("Paste", self.paste)
        menu.addSeparator()
        menu.addAction("Backdrop", self.add_backdrop)
        menu.addAction("Sticky note", lambda: self.add_backdrop(note=True))
        if self.s.selected:
            menu.addAction("Group selected nodes", self.add_subgroup)
        if backdrop and backdrop.bd.group:
            menu.addAction("Expand subgroup" if backdrop.bd.collapsed else "Collapse subgroup",
                           lambda bd=backdrop.bd: self._toggle_subgroup(bd))
        menu.addAction("Fit view", self.fit)
        menu.exec(e.globalPos())

    def _toggle_subgroup(self, backdrop):
        self.s.structural("toggle subgroup", lambda: setattr(backdrop, "collapsed", not backdrop.collapsed))

    def pick_color(self):
        if not self.s.selected:
            return
        c = QColorDialog.getColor(parent=self)
        if c.isValid():
            for n in self.s.selected:
                self.s.set_node_meta(n, color=c.name())

    def edit_label(self):
        if not self.s.selected:
            return
        n = self.s.selected[-1]
        t, ok = QInputDialog.getText(self, "Label", "Label", text=self.s.project.node_meta(n).label)
        if ok:
            self.s.set_node_meta(n, label=t)

    def force_rerender(self):
        for n in self.s.selected:
            n.revision = n.revision + 1
            n.clear_error()
            self.s.node_changed(n)

    def shuffle_per_layer(self, read):
        try:
            info = self.s.project.engine.info(read, self.s.frame)
        except RuntimeError as err:
            self.s.log(str(err), "error")
            return
        layers = []
        for c in info["channels"]:
            ly = c.rsplit(".", 1)[0]
            if ly != "rgba" and ly not in layers:
                layers.append(ly)
        if not layers:
            self.s.log(f"{read.name} has no layers besides rgba", "info")
            return
        m = self.s.project.node_meta(read)

        def build():
            from vertexcomp.ui.viewer import layer_slots
            for i, ly in enumerate(layers):
                sh = self.s.project.create("Shuffle", x=m.x + (i - len(layers) / 2) * 120, y=m.y + 90)
                slots = layer_slots(ly, info["channels"])
                for name, src in zip(("red", "green", "blue", "alpha"), slots):
                    sh.set(name, src if src else ("1" if name == "alpha" else "0"))
                if sum(1 for s in slots if s) == 1:
                    c = next(s for s in slots if s)
                    for name in ("red", "green", "blue"):
                        sh.set(name, c)
                self.s.project.graph.connect(sh, 0, read)
                self.s.project.node_meta(sh).label = ly
        self.s.structural("shuffle per layer", build)
