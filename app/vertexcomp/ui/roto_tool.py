"""Roto editing: viewer tools (draw, select, transform points and tangents,
feather) and the shape panel shown in the properties of a Roto node."""
from __future__ import annotations

import itertools
import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)

from vertexcomp import _vccore as vc
from vertexcomp.ui import theme
from vertexcomp.ui.widgets import IconButton, ScrubField

BLENDS = ["over", "add", "subtract", "max", "min", "multiply"]
KAPPA = 0.5522847498
_merge_ids = itertools.count()


def pt_list(pts):
    return [vc.RotoPoint(p.x, p.y, p.in_x, p.in_y, p.out_x, p.out_y, p.feather, p.smooth) for p in pts]


def apply_m(m, x, y):
    return m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]


def inv_m(m):
    det = m[0] * m[4] - m[1] * m[3] or 1e-12
    a, b, d, e = m[4] / det, -m[1] / det, -m[3] / det, m[0] / det
    return [a, b, -(a * m[2] + b * m[5]), d, e, -(d * m[2] + e * m[5])]


def auto_tangents(pts, closed, i):
    n = len(pts)
    prev = pts[(i - 1) % n] if (closed or i > 0) else pts[i]
    nxt = pts[(i + 1) % n] if (closed or i < n - 1) else pts[i]
    tx, ty = (nxt.x - prev.x) / 6.0, (nxt.y - prev.y) / 6.0
    p = pts[i]
    p.in_x, p.in_y, p.out_x, p.out_y = -tx, -ty, tx, ty


class RotoState:
    """Selection shared by the viewer tool and the panel."""

    def __init__(self):
        self.shape = -1
        self.points: set[int] = set()
        self.tool = "select"
        self.clipboard = None


class RotoTool:
    HANDLE = 4.0

    def __init__(self, session, viewer, node_name):
        self.s, self.v, self.node_name = session, viewer, node_name
        self.st = session.roto_state
        self._drag = None
        self._creating = None  # index of the open shape being drawn
        self._merge = None

    # ------------------------------------------------------------ data
    def node(self):
        return self.s.project.find(self.node_name)

    def shapes(self):
        n = self.node()
        return n.roto_shapes() if n is not None else []

    def frame(self):
        return float(self.s.frame)

    def commit(self, shapes, final=True):
        n = self.node()
        if n is None:
            return
        if self._merge is None:
            self._merge = f"roto{next(_merge_ids)}"
        self.s.set_roto(n, shapes, merge_key=self._merge)
        if final:
            self._merge = None
        self.v.update()

    def write_points(self, shape, new_pts):
        """Store edited points honouring auto-key (see docs/node-reference.md, Roto)."""
        f = self.frame()
        keys = {k: pt_list(v) for k, v in shape.keys.items()}
        if not keys or self.s.autokey or any(abs(k - f) < 1e-6 for k in keys):
            match = next((k for k in keys if abs(k - f) < 1e-6), f)
            keys[match] = new_pts
        elif len(keys) == 1:
            keys[next(iter(keys))] = new_pts
        else:
            cur = shape.points_at(f)
            for _k, pts in keys.items():
                for i, p in enumerate(pts):
                    if i < len(cur) and i < len(new_pts):
                        p.x += new_pts[i].x - cur[i].x
                        p.y += new_pts[i].y - cur[i].y
                        p.in_x += new_pts[i].in_x - cur[i].in_x
                        p.in_y += new_pts[i].in_y - cur[i].in_y
                        p.out_x += new_pts[i].out_x - cur[i].out_x
                        p.out_y += new_pts[i].out_y - cur[i].out_y
                        p.feather += new_pts[i].feather - cur[i].feather
        shape.keys = keys

    def screen_points(self, shape):
        m = shape.transform_at(self.frame())
        out = []
        for p in shape.points_at(self.frame()):
            x, y = apply_m(m, p.x, p.y)
            ix, iy = m[0] * p.in_x + m[1] * p.in_y, m[3] * p.in_x + m[4] * p.in_y
            ox, oy = m[0] * p.out_x + m[1] * p.out_y, m[3] * p.out_x + m[4] * p.out_y
            out.append((self.v.to_screen(x, y), self.v.to_screen(x + ix, y + iy), self.v.to_screen(x + ox, y + oy)))
        return out

    def to_shape_space(self, shape, sx, sy):
        ix, iy = self.v.to_image(sx, sy)
        return apply_m(inv_m(shape.transform_at(self.frame())), ix, iy)

    # ------------------------------------------------------------ painting
    def paint(self, p: QPainter):
        if not self.v.show_overlays:
            return
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        shapes = self.shapes()
        f = self.frame()
        for si, sh in enumerate(shapes):
            if not sh.visible or not sh.keys:
                continue
            col = QColor.fromRgbF(*[max(0.0, min(1.0, c)) for c in sh.color[:3]])
            sel_shape = si == self.st.shape
            pts = sh.transformed_at(f)
            poly = vc.flatten_shape(pts, sh.closed)
            if len(poly) >= 2:
                path = QPainterPath(QPointF(*self.v.to_screen(*poly[0])))
                for x, y in poly[1:]:
                    path.lineTo(QPointF(*self.v.to_screen(x, y)))
                if sh.closed:
                    path.closeSubpath()
                p.setPen(QPen(col, 1.5 if sel_shape else 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPath(path)
                fe = sh.feather
                if sel_shape and sh.closed and (abs(fe) > 0.01 or any(abs(q.feather) > 0.01 for q in pts)):
                    self._paint_feather(p, poly, pts, sh, col)
            sp = self.screen_points(sh)
            for i, (pt, hin, hout) in enumerate(sp):
                selected = sel_shape and i in self.st.points
                if selected:
                    p.setPen(QPen(QColor(200, 200, 200), 1))
                    for h in (hin, hout):
                        if abs(h[0] - pt[0]) + abs(h[1] - pt[1]) > 0.5:
                            p.drawLine(QPointF(*pt), QPointF(*h))
                            p.setBrush(QColor(200, 200, 200))
                            p.drawEllipse(QPointF(*h), 2.5, 2.5)
                p.setPen(QPen(col.darker(150) if not selected else QColor(theme.T["accent"]), 1))
                p.setBrush(QColor(theme.T["accent"]) if selected else col)
                r = self.HANDLE
                p.drawRect(QRectF(pt[0] - r / 2 - 0.5, pt[1] - r / 2 - 0.5, r + 1, r + 1))
        # Multi-point gizmo
        box = self._selection_box()
        if box is not None:
            x0, y0, x1, y1 = box
            p.setPen(QPen(QColor(theme.T["accent"]), 1, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(x0, y0, x1 - x0, y1 - y0))
            p.setPen(QPen(QColor(theme.T["accent"]), 1))
            for cx, cy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
                p.drawRect(QRectF(cx - 3, cy - 3, 6, 6))
        if self._drag and self._drag[0] == "box":
            _, a, b = self._drag
            p.setPen(QPen(QColor(220, 220, 220), 1, Qt.PenStyle.DotLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(a, b).normalized())
        if self._drag and self._drag[0] in ("rect", "ellipse"):
            _, a, b = self._drag
            p.setPen(QPen(QColor(theme.T["accent"]), 1))
            r = QRectF(a, b).normalized()
            p.drawRect(r) if self._drag[0] == "rect" else p.drawEllipse(r)

    def _paint_feather(self, p, poly, pts, sh, col):
        n = len(poly)
        area = sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1] for i in range(n))
        sign = -1.0 if area > 0 else 1.0  # y-down: positive shoelace area is clockwise on screen
        avg_pf = sum(q.feather for q in pts) / max(1, len(pts))
        d = sh.feather + avg_pf
        out = []
        for i in range(n):
            a, b = poly[i - 1], poly[(i + 1) % n]
            tx, ty = b[0] - a[0], b[1] - a[1]
            ln = math.hypot(tx, ty) or 1.0
            nx, ny = ty / ln * sign, -tx / ln * sign
            out.append(QPointF(*self.v.to_screen(poly[i][0] - nx * d, poly[i][1] - ny * d)))
        c = QColor(col)
        c.setAlpha(140)
        p.setPen(QPen(c, 1, Qt.PenStyle.DashLine))
        p.drawPolygon(QPolygonF(out))

    def _selection_box(self):
        if self.st.shape < 0 or len(self.st.points) < 2:
            return None
        shapes = self.shapes()
        if self.st.shape >= len(shapes):
            return None
        sp = self.screen_points(shapes[self.st.shape])
        xs = [sp[i][0][0] for i in self.st.points if i < len(sp)]
        ys = [sp[i][0][1] for i in self.st.points if i < len(sp)]
        if len(xs) < 2:
            return None
        return min(xs) - 6, min(ys) - 6, max(xs) + 6, max(ys) + 6

    # ------------------------------------------------------------ hit tests
    def hit(self, pos):
        """-> (kind, shape, point) for point / in / out handles, nearest first."""
        x, y = pos.x(), pos.y()
        shapes = self.shapes()
        best = None
        for si, sh in enumerate(shapes):
            if not sh.visible or sh.locked:
                continue
            for i, (pt, hin, hout) in enumerate(self.screen_points(sh)):
                cands = [("point", pt)]
                if si == self.st.shape and i in self.st.points:
                    cands += [("in", hin), ("out", hout)]
                for kind, (px, py) in cands:
                    d = math.hypot(px - x, py - y)
                    if d < 7 and (best is None or d < best[0]):
                        best = (d, kind, si, i)
        return best[1:] if best else None

    def inside_shape(self, pos):
        ix, iy = self.v.to_image(pos.x(), pos.y())
        for si, sh in reversed(list(enumerate(self.shapes()))):
            if not sh.visible or sh.locked or not sh.closed:
                continue
            poly = vc.flatten_shape(sh.transformed_at(self.frame()), True)
            inside = False
            n = len(poly)
            for i in range(n):
                (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % n]
                if (y0 > iy) != (y1 > iy) and ix < (x1 - x0) * (iy - y0) / ((y1 - y0) or 1e-12) + x0:
                    inside = not inside
            if inside:
                return si
        return -1

    def nearest_segment(self, sh, pos, max_dist=8.0):
        pts = sh.transformed_at(self.frame())
        n = len(pts)
        segs = n if sh.closed else n - 1
        best = None
        for i in range(segs):
            a, b = pts[i], pts[(i + 1) % n]
            for k in range(1, 40):
                t = k / 40.0
                x, y = bez(a, b, t)
                sx, sy = self.v.to_screen(x, y)
                d = math.hypot(sx - pos.x(), sy - pos.y())
                if d < max_dist and (best is None or d < best[0]):
                    best = (d, i, t)
        return best

    # ------------------------------------------------------------ events
    def press(self, e) -> bool:
        if e.button() != Qt.MouseButton.LeftButton or e.modifiers() & Qt.KeyboardModifier.AltModifier:
            return False
        pos = e.position()
        tool = self.st.tool
        shapes = self.shapes()
        if tool == "bezier":
            return self._bezier_press(pos, shapes)
        if tool in ("rect", "ellipse"):
            self._drag = (tool, pos, pos)
            return True
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            for si, sh in enumerate(shapes):
                seg = self.nearest_segment(sh, pos)
                if seg:
                    self.insert_point(si, seg[1], seg[2])
                    return True
        h = self.hit(pos)
        if h:
            kind, si, i = h
            if kind == "point":
                if e.modifiers() & Qt.KeyboardModifier.ShiftModifier and si == self.st.shape:
                    self.st.points ^= {i}
                elif si != self.st.shape or i not in self.st.points:
                    self.st.shape, self.st.points = si, {i}
                self.s.roto_selection_changed()
            mode = "feather" if tool == "feather" and kind == "point" else kind
            self._drag = (mode, pos, si, i, [copy_shape(s) for s in shapes])
            self.v.update()
            return True
        box = self._selection_box()
        if box and box[0] <= pos.x() <= box[2] and box[1] <= pos.y() <= box[3]:
            corner = None
            for ci, (cx, cy) in enumerate(((box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3]))):
                if abs(cx - pos.x()) < 6 and abs(cy - pos.y()) < 6:
                    corner = ci
            self._drag = ("scale" if corner is not None else "move", pos, self.st.shape, corner, [copy_shape(s) for s in shapes], box)
            return True
        si = self.inside_shape(pos)
        if si >= 0:
            self.st.shape = si
            self.st.points = set(range(len(shapes[si].points_at(self.frame()))))
            self.s.roto_selection_changed()
            self._drag = ("move", pos, si, None, [copy_shape(s) for s in shapes], None)
            self.v.update()
            return True
        self.st.points = set()
        self.s.roto_selection_changed()
        self._drag = ("box", pos, pos)
        self.v.update()
        return True

    def move(self, e) -> bool:
        if not self._drag:
            return self.st.tool != "select" and False
        pos = e.position()
        kind = self._drag[0]
        if kind in ("box", "rect", "ellipse"):
            self._drag = (kind, self._drag[1], pos)
            self.v.update()
            return True
        if kind == "bezier_tangent":
            _, si, i = self._drag
            shapes = self.shapes()
            sh = shapes[si]
            pts = pt_list(sh.points_at(self.frame()))
            x, y = self.to_shape_space(sh, pos.x(), pos.y())
            p = pts[i]
            p.out_x, p.out_y = x - p.x, y - p.y
            p.in_x, p.in_y = -p.out_x, -p.out_y
            self.write_points(sh, pts)
            self.commit(shapes, final=False)
            return True
        orig = self._drag[-1] if kind in ("point", "in", "out", "feather") else self._drag[4]
        start = self._drag[1]
        si = self._drag[2]
        shapes = [copy_shape(s) for s in orig]
        sh = shapes[si]
        f = self.frame()
        m = sh.transform_at(f)
        im = inv_m(m)
        dsx, dsy = pos.x() - start.x(), pos.y() - start.y()
        dix, diy = dsx / self.v.zoom, dsy / self.v.zoom
        dx, dy = im[0] * dix + im[1] * diy, im[3] * dix + im[4] * diy
        pts = pt_list(sh.points_at(f))
        if kind == "point":
            for i in self.st.points:
                if i < len(pts):
                    pts[i].x += dx
                    pts[i].y += dy
        elif kind in ("in", "out"):
            i = self._drag[3]
            p = pts[i]
            if kind == "in":
                p.in_x += dx
                p.in_y += dy
            else:
                p.out_x += dx
                p.out_y += dy
            if p.smooth and not (e.modifiers() & Qt.KeyboardModifier.ControlModifier):
                a = (p.in_x, p.in_y) if kind == "in" else (p.out_x, p.out_y)
                other_len = math.hypot(p.out_x, p.out_y) if kind == "in" else math.hypot(p.in_x, p.in_y)
                ln = math.hypot(*a) or 1.0
                ox, oy = -a[0] / ln * other_len, -a[1] / ln * other_len
                if kind == "in":
                    p.out_x, p.out_y = ox, oy
                else:
                    p.in_x, p.in_y = ox, oy
        elif kind == "feather":
            for i in self.st.points:
                if i < len(pts):
                    pts[i].feather = orig[si].points_at(f)[i].feather + dsx / self.v.zoom
        elif kind == "move":
            for i in (self.st.points or range(len(pts))):
                pts[i].x += dx
                pts[i].y += dy
        elif kind == "scale":
            box = self._drag[5]
            corner = self._drag[3]
            ax = box[2] if corner in (0, 2) else box[0]
            ay = box[3] if corner in (0, 1) else box[1]
            cx0 = box[0] if corner in (0, 2) else box[2]
            cy0 = box[1] if corner in (0, 1) else box[3]
            sxf = (pos.x() - ax) / ((cx0 - ax) or 1)
            syf = (pos.y() - ay) / ((cy0 - ay) or 1)
            if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                sxf = syf = max(sxf, syf)
            aix, aiy = apply_m(im, *self.v.to_image(ax, ay))
            for i in self.st.points:
                p = pts[i]
                p.x = aix + (p.x - aix) * sxf
                p.y = aiy + (p.y - aiy) * syf
                p.in_x *= sxf
                p.in_y *= syf
                p.out_x *= sxf
                p.out_y *= syf
        self.write_points(sh, pts)
        self.commit(shapes, final=False)
        return True

    def release(self, e) -> bool:
        if not self._drag:
            return False
        kind = self._drag[0]
        if kind == "box":
            r = QRectF(self._drag[1], self._drag[2]).normalized()
            shapes = self.shapes()
            if self.st.shape < 0 and shapes:
                self.st.shape = 0
            if 0 <= self.st.shape < len(shapes):
                sp = self.screen_points(shapes[self.st.shape])
                self.st.points = {i for i, (pt, _, _) in enumerate(sp) if r.contains(QPointF(*pt))}
            self.s.roto_selection_changed()
        elif kind in ("rect", "ellipse"):
            r = QRectF(self._drag[1], self._drag[2]).normalized()
            if r.width() > 3 and r.height() > 3:
                self.add_primitive(kind, r)
        self._drag = None
        self._merge = None
        self.v.update()
        return True

    def double_click(self, e) -> bool:
        if self._creating is not None:
            self.close_creating()
            return True
        return False

    def key(self, e) -> bool:
        k = e.key()
        shapes = self.shapes()
        if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self._creating is not None:
            self.close_creating()
            return True
        if k == Qt.Key.Key_Escape and self._creating is not None:
            self._creating = None
            self.st.tool = "select"
            self.s.roto_selection_changed()
            return True
        if k in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.st.points and 0 <= self.st.shape < len(shapes):
            self.delete_points()
            return True
        arrows = {Qt.Key.Key_Left: (-1, 0), Qt.Key.Key_Right: (1, 0), Qt.Key.Key_Up: (0, -1), Qt.Key.Key_Down: (0, 1)}
        if k in arrows and self.st.points and 0 <= self.st.shape < len(shapes):
            d = 10.0 if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
            sh = shapes[self.st.shape]
            pts = pt_list(sh.points_at(self.frame()))
            for i in self.st.points:
                pts[i].x += arrows[k][0] * d
                pts[i].y += arrows[k][1] * d
            self.write_points(sh, pts)
            self.commit(shapes)
            return True
        return False

    # ------------------------------------------------------------ operations
    def _bezier_press(self, pos, shapes):
        f = self.frame()
        if self._creating is not None and self._creating < len(shapes):
            sh = shapes[self._creating]
            sp = self.screen_points(sh)
            if len(sp) >= 3 and math.hypot(sp[0][0][0] - pos.x(), sp[0][0][1] - pos.y()) < 8:
                self.close_creating()
                return True
            x, y = self.to_shape_space(sh, pos.x(), pos.y())
            pts = pt_list(sh.points_at(f)) + [vc.RotoPoint(x, y)]
            sh.keys = {k: (pts if abs(k - f) < 1e-6 or len(sh.keys) == 1 else pt_list(v) + [vc.RotoPoint(x, y)])
                       for k, v in sh.keys.items()}
            self.st.points = {len(pts) - 1}
            self.commit(shapes, final=False)
            self._drag = ("bezier_tangent", self._creating, len(pts) - 1)
            return True
        x, y = self.v.to_image(pos.x(), pos.y())
        sh = vc.RotoShape()
        sh.name = f"Bezier{len(shapes) + 1}"
        sh.closed = False
        sh.keys = {f: [vc.RotoPoint(x, y)]}
        shapes.append(sh)
        self._creating = len(shapes) - 1
        self.st.shape, self.st.points = self._creating, {0}
        self.commit(shapes, final=False)
        self._drag = ("bezier_tangent", self._creating, 0)
        self.s.roto_selection_changed()
        return True

    def close_creating(self):
        shapes = self.shapes()
        i = self._creating
        self._creating = None
        if i is None or i >= len(shapes):
            return
        if len(shapes[i].points_at(self.frame())) < 3:
            del shapes[i]
            self.st.shape = -1
        else:
            shapes[i].closed = True
        self.st.tool = "select"
        self.commit(shapes)
        self.s.roto_selection_changed()

    def add_primitive(self, kind, r):
        x0, y0 = self.v.to_image(r.left(), r.top())
        x1, y1 = self.v.to_image(r.right(), r.bottom())
        shapes = self.shapes()
        sh = vc.RotoShape()
        f = self.frame()
        if kind == "rect":
            sh.name = f"Rectangle{len(shapes) + 1}"
            pts = [vc.RotoPoint(x0, y0, smooth=False), vc.RotoPoint(x1, y0, smooth=False),
                   vc.RotoPoint(x1, y1, smooth=False), vc.RotoPoint(x0, y1, smooth=False)]
        else:
            sh.name = f"Ellipse{len(shapes) + 1}"
            cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
            kx, ky = rx * KAPPA, ry * KAPPA
            pts = [vc.RotoPoint(cx, y0, -kx, 0, kx, 0), vc.RotoPoint(x1, cy, 0, -ky, 0, ky),
                   vc.RotoPoint(cx, y1, kx, 0, -kx, 0), vc.RotoPoint(x0, cy, 0, ky, 0, -ky)]
        sh.keys = {f: pts}
        sh.cx, sh.cy = 0, 0
        sh.set_transform("cx", (x0 + x1) / 2)
        sh.set_transform("cy", (y0 + y1) / 2)
        shapes.append(sh)
        self.st.shape, self.st.points = len(shapes) - 1, set()
        self.st.tool = "select"
        self.commit(shapes)
        self.s.roto_selection_changed()

    def insert_point(self, si, seg, t):
        shapes = self.shapes()
        sh = shapes[si]
        keys = {}
        for k, pts in sh.keys.items():
            pts = pt_list(pts)
            n = len(pts)
            a, b = pts[seg], pts[(seg + 1) % n]
            new, a_out, b_in = split_bez(a, b, t)
            a.out_x, a.out_y = a_out
            b.in_x, b.in_y = b_in
            pts.insert(seg + 1, new)
            keys[k] = pts
        sh.keys = keys
        self.st.shape, self.st.points = si, {seg + 1}
        self.commit(shapes)
        self.s.roto_selection_changed()

    def delete_points(self):
        shapes = self.shapes()
        sh = shapes[self.st.shape]
        keep = lambda pts: [p for i, p in enumerate(pt_list(pts)) if i not in self.st.points]
        sh.keys = {k: keep(v) for k, v in sh.keys.items()}
        if len(sh.points_at(self.frame())) < (3 if sh.closed else 1):
            del shapes[self.st.shape]
            self.st.shape = -1
        self.st.points = set()
        self.commit(shapes)
        self.s.roto_selection_changed()

    def set_smooth(self, smooth):
        shapes = self.shapes()
        if not (0 <= self.st.shape < len(shapes)):
            return
        sh = shapes[self.st.shape]
        keys = {}
        for k, v in sh.keys.items():
            pts = pt_list(v)
            for i in self.st.points:
                if i >= len(pts):
                    continue
                pts[i].smooth = smooth
                if smooth:
                    auto_tangents(pts, sh.closed, i)
                else:
                    pts[i].in_x = pts[i].in_y = pts[i].out_x = pts[i].out_y = 0.0
            keys[k] = pts
        sh.keys = keys
        self.commit(shapes)


def bez(a, b, t):
    p0, p1 = (a.x, a.y), (a.x + a.out_x, a.y + a.out_y)
    p2, p3 = (b.x + b.in_x, b.y + b.in_y), (b.x, b.y)
    u = 1 - t
    return (u ** 3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t ** 3 * p3[0],
            u ** 3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t ** 3 * p3[1])


def split_bez(a, b, t):
    """De Casteljau split; returns (new point, a's new out handle, b's new in handle)."""
    p0, p1 = (a.x, a.y), (a.x + a.out_x, a.y + a.out_y)
    p2, p3 = (b.x + b.in_x, b.y + b.in_y), (b.x, b.y)
    lerp = lambda p, q: (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)
    q0, q1, q2 = lerp(p0, p1), lerp(p1, p2), lerp(p2, p3)
    r0, r1 = lerp(q0, q1), lerp(q1, q2)
    s = lerp(r0, r1)
    new = vc.RotoPoint(s[0], s[1], r0[0] - s[0], r0[1] - s[1], r1[0] - s[0], r1[1] - s[1])
    return new, (q0[0] - p0[0], q0[1] - p0[1]), (q2[0] - p3[0], q2[1] - p3[1])


def copy_shape(s):
    c = s.copy()
    c.keys = {k: pt_list(v) for k, v in s.keys.items()}
    return c


class RotoPanel(QFrame):
    """Shape list and per-shape attributes for the Roto node."""

    def __init__(self, session, node):
        super().__init__()
        self.s, self.node_name = session, node.name
        self.st = session.roto_state
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(3)
        h = QLabel("Shapes")
        h.setObjectName("sectionHeader")
        lay.addWidget(h)

        tools = QHBoxLayout()
        tools.setSpacing(2)
        self.tool_buttons = {}
        for name, icon, tip in (("select", "select", "Select / transform (S)"), ("bezier", "bezier",
                                                                       "Bezier: click to add points, drag for tangents, "
                                                                       "click the first point or Enter to close"),
                                ("rect", "rect", "Rectangle"), ("ellipse", "ellipse", "Ellipse"),
                                ("feather", "feather", "Point feather: drag points horizontally")):
            b = IconButton(icon, tip, True)
            b.clicked.connect(lambda _=False, n=name: self.set_tool(n))
            tools.addWidget(b)
            self.tool_buttons[name] = b
        for icon, tip, fn in (("smooth", "Smooth selected points", lambda: self._tool_call("set_smooth", True)),
                              ("cusp", "Cusp selected points", lambda: self._tool_call("set_smooth", False)),
                              ("del_point", "Delete selected points (Delete)", lambda: self._tool_call("delete_points"))):
            b = IconButton(icon, tip)
            b.clicked.connect(fn)
            tools.addWidget(b)
        self.autokey = IconButton("autokey", "Auto-key: edits create a key at the current frame", True)
        self.autokey.setChecked(session.autokey)
        self.autokey.toggled.connect(lambda on: setattr(session, "autokey", on))
        tools.addWidget(self.autokey)
        tools.addStretch(1)
        lay.addLayout(tools)

        self.list = QListWidget()
        self.list.setFixedHeight(96)
        self.list.currentRowChanged.connect(self._row)
        self.list.itemChanged.connect(self._item_changed)
        lay.addWidget(self.list)
        row = QHBoxLayout()
        for icon, tip, fn in (("up", "Move up", lambda: self.reorder(-1)), ("down", "Move down", lambda: self.reorder(1)),
                              ("plus", "Duplicate shape", self.duplicate), ("minus", "Delete shape", self.delete_shape),
                              ("key", "Set shape key at this frame", self.key_here),
                              ("key_on", "Delete shape key at this frame", self.unkey_here)):
            b = IconButton(icon, tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        copyb = QLabel("<a href='c' style='color:#8a8a8a'>copy</a> <a href='p' style='color:#8a8a8a'>paste</a>")
        copyb.setToolTip("Copy the shape's points at this frame / paste them as a key at the current frame")
        copyb.linkActivated.connect(lambda h: self.copy_key() if h == "c" else self.paste_key())
        row.addStretch(1)
        row.addWidget(copyb)
        lay.addLayout(row)

        g = QGridLayout()
        g.setHorizontalSpacing(4)
        g.setVerticalSpacing(2)
        self.name = QLineEdit()
        self.name.editingFinished.connect(lambda: self.edit(name=self.name.text()))
        self.blend = QComboBox()
        self.blend.addItems(BLENDS)
        self.blend.activated.connect(lambda i: self.edit(blend=i))
        self.invert = QCheckBox("invert")
        self.invert.toggled.connect(lambda on: self.edit(invert=on))
        self.falloff = QComboBox()
        self.falloff.addItems(["smooth", "linear"])
        self.falloff.activated.connect(lambda i: self.edit(feather_falloff=i))
        self.fields = {}
        r = 0
        g.addWidget(self._lab("name"), r, 0)
        g.addWidget(self.name, r, 1, 1, 3)
        r += 1
        g.addWidget(self._lab("blend"), r, 0)
        g.addWidget(self.blend, r, 1)
        g.addWidget(self.invert, r, 2, 1, 2)
        r += 1
        for name, label, step in (("opacity", "opacity", 0.01), ("feather", "feather", 0.5), ("blur", "blur", 0.1),
                                  ("motion_blur_samples", "motion blur", 1), ("shutter", "shutter", 0.01)):
            f = ScrubField(0.0, step=step, integer=name == "motion_blur_samples", minimum=0 if name != "feather" else None)
            f.valueEdited.connect(lambda v, fin, n=name: self.edit(**{n: (int(v) if n == "motion_blur_samples" else v)}, final=fin))
            self.fields[name] = f
            g.addWidget(self._lab(label), r, 0)
            g.addWidget(f, r, 1)
            if name == "feather":
                g.addWidget(self.falloff, r, 2)
            r += 1
        self.point_feather = ScrubField(0.0, step=0.5)
        self.point_feather.valueEdited.connect(self.set_point_feather)
        g.addWidget(self._lab("point feather"), r, 0)
        g.addWidget(self.point_feather, r, 1)
        r += 1
        h2 = QLabel("Shape transform")
        h2.setObjectName("sectionHeader")
        g.addWidget(h2, r, 0, 1, 4)
        r += 1
        self.tfields = {}
        for name, label in (("tx", "translate x"), ("ty", "translate y"), ("rot", "rotate"), ("sx", "scale x"),
                            ("sy", "scale y"), ("cx", "center x"), ("cy", "center y")):
            f = ScrubField(0.0, step=0.01 if name in ("sx", "sy") else 0.5)
            f.valueEdited.connect(lambda v, fin, n=name: self.edit_transform(n, v, fin))
            k = IconButton("key", "Key this parameter at the current frame")
            k.clicked.connect(lambda _=False, n=name: self.key_transform(n))
            self.tfields[name] = (f, k)
            g.addWidget(self._lab(label), r, 0)
            g.addWidget(f, r, 1)
            g.addWidget(k, r, 2)
            r += 1
        lay.addLayout(g)
        self._merge = None
        session.frameChanged.connect(lambda f: self.refresh())

    def _lab(self, t):
        l = QLabel(t)
        l.setObjectName("knobLabel")
        l.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return l

    def node(self):
        return self.s.project.find(self.node_name)

    def _tool_call(self, name, *args):
        t = self.s.roto_tool
        if t is not None:
            getattr(t, name)(*args)

    def set_tool(self, name):
        self.st.tool = name
        for n, b in self.tool_buttons.items():
            b.setChecked(n == name)

    def refresh(self):
        n = self.node()
        if n is None:
            return
        shapes = n.roto_shapes()
        for nm, b in self.tool_buttons.items():
            b.setChecked(nm == self.st.tool)
        self.list.blockSignals(True)
        if self.list.count() != len(shapes):
            self.list.clear()
            for s in shapes:
                it = QListWidgetItem(s.name)
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                self.list.addItem(it)
        for i, s in enumerate(shapes):
            it = self.list.item(i)
            it.setText(s.name)
            it.setCheckState(Qt.CheckState.Checked if s.visible else Qt.CheckState.Unchecked)
        if 0 <= self.st.shape < len(shapes):
            self.list.setCurrentRow(self.st.shape)
        self.list.blockSignals(False)
        sh = shapes[self.st.shape] if 0 <= self.st.shape < len(shapes) else None
        for w in [self.name, self.blend, self.invert, self.falloff, self.point_feather] + list(self.fields.values()):
            w.setEnabled(sh is not None)
        if sh is None:
            return
        f = float(self.s.frame)
        if not self.name.hasFocus():
            self.name.setText(sh.name)
        self.blend.setCurrentIndex(sh.blend)
        self.invert.blockSignals(True)
        self.invert.setChecked(sh.invert)
        self.invert.blockSignals(False)
        self.falloff.setCurrentIndex(sh.feather_falloff)
        for k, w in self.fields.items():
            w.setValue(getattr(sh, k))
        pts = sh.points_at(f)
        sel = [pts[i].feather for i in self.st.points if i < len(pts)]
        self.point_feather.setValue(sel[0] if sel else 0.0)
        for name, (fld, key) in self.tfields.items():
            fld.setValue(sh.get_transform(name, f))
            keys = sh.transform_keys(name)
            on = any(abs(k[0] - f) < 1e-6 for k in keys)
            key.setIcon(theme.icon("key_on" if on else "key", "#6fa0dc" if keys else theme.T["dim"]))

    def _shapes_and_current(self):
        n = self.node()
        shapes = n.roto_shapes()
        if not (0 <= self.st.shape < len(shapes)):
            return n, shapes, None
        return n, shapes, shapes[self.st.shape]

    def _commit(self, n, shapes, final=True):
        if self._merge is None:
            self._merge = f"rotopanel{next(_merge_ids)}"
        self.s.set_roto(n, shapes, merge_key=self._merge)
        if final:
            self._merge = None
        self.refresh()

    def edit(self, final=True, **kw):
        n, shapes, sh = self._shapes_and_current()
        if sh is None:
            return
        for k, v in kw.items():
            setattr(sh, k, v)
        self._commit(n, shapes, final)

    def edit_transform(self, name, v, final):
        n, shapes, sh = self._shapes_and_current()
        if sh is None:
            return
        f = float(self.s.frame)
        if sh.transform_keys(name) or (self.s.autokey and name not in ("cx", "cy")):
            sh.set_transform(name, v, f)
        else:
            sh.set_transform(name, v)
        self._commit(n, shapes, final)

    def key_transform(self, name):
        n, shapes, sh = self._shapes_and_current()
        if sh is None:
            return
        f = float(self.s.frame)
        keys = sh.transform_keys(name)
        if any(abs(k[0] - f) < 1e-6 for k in keys):
            sh.set_transform_keys(name, [k for k in keys if abs(k[0] - f) >= 1e-6])
        else:
            sh.set_transform(name, sh.get_transform(name, f), f)
        self._commit(n, shapes)

    def set_point_feather(self, v, final):
        n, shapes, sh = self._shapes_and_current()
        if sh is None or not self.st.points:
            return
        pts = pt_list(sh.points_at(float(self.s.frame)))
        for i in self.st.points:
            if i < len(pts):
                pts[i].feather = v
        tool = self.s.roto_tool
        if tool is not None:
            tool.write_points(sh, pts)
        self._commit(n, shapes, final)

    def _row(self, r):
        self.st.shape = r
        self.st.points = set()
        self.s.roto_selection_changed()

    def _item_changed(self, it):
        n, shapes, _ = self._shapes_and_current()
        i = self.list.row(it)
        if 0 <= i < len(shapes):
            vis = it.checkState() == Qt.CheckState.Checked
            if shapes[i].visible != vis:
                shapes[i].visible = vis
                self._commit(n, shapes)

    def reorder(self, d):
        n, shapes, sh = self._shapes_and_current()
        i = self.st.shape
        j = i + d
        if sh is None or not (0 <= j < len(shapes)):
            return
        shapes[i], shapes[j] = shapes[j], shapes[i]
        self.st.shape = j
        self._commit(n, shapes)

    def duplicate(self):
        n, shapes, sh = self._shapes_and_current()
        if sh is None:
            return
        c = copy_shape(sh)
        c.name = sh.name + "_copy"
        shapes.insert(self.st.shape + 1, c)
        self.st.shape += 1
        self.list.clear()
        self._commit(n, shapes)

    def delete_shape(self):
        n, shapes, sh = self._shapes_and_current()
        if sh is None:
            return
        del shapes[self.st.shape]
        self.st.shape = min(self.st.shape, len(shapes) - 1)
        self.st.points = set()
        self.list.clear()
        self._commit(n, shapes)

    def key_here(self):
        n, shapes, sh = self._shapes_and_current()
        if sh is None:
            return
        f = float(self.s.frame)
        keys = {k: pt_list(v) for k, v in sh.keys.items()}
        keys[f] = pt_list(sh.points_at(f))
        sh.keys = keys
        self._commit(n, shapes)

    def copy_key(self):
        n, shapes, sh = self._shapes_and_current()
        if sh is not None:
            self.st.clipboard = pt_list(sh.points_at(float(self.s.frame)))

    def paste_key(self):
        n, shapes, sh = self._shapes_and_current()
        clip = self.st.clipboard
        if sh is None or not clip:
            return
        if sh.keys and len(next(iter(sh.keys.values()))) != len(clip):
            self.s.log("paste: point counts differ", "warning")
            return
        keys = {k: pt_list(v) for k, v in sh.keys.items()}
        keys[float(self.s.frame)] = pt_list(clip)
        sh.keys = keys
        self._commit(n, shapes)

    def unkey_here(self):
        n, shapes, sh = self._shapes_and_current()
        if sh is None:
            return
        f = float(self.s.frame)
        keys = {k: pt_list(v) for k, v in sh.keys.items() if abs(k - f) >= 1e-6}
        if keys:
            sh.keys = keys
            self._commit(n, shapes)
