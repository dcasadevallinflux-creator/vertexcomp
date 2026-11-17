"""Interactive viewer controls for a selected Transform node."""
from __future__ import annotations

import itertools
import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen

from vertexcomp.ui import theme

_drag_ids = itertools.count()


class TransformTool:
    HANDLE = 7.0

    def __init__(self, session, viewer, node_name):
        self.s, self.v, self.node_name = session, viewer, node_name
        self._drag = None

    def node(self):
        return self.s.project.find(self.node_name)

    def geometry(self):
        node = self.node()
        if node is None:
            return None
        if node.inputs and node.inputs[0] is not None:
            info = self.s.project.engine.info(node.inputs[0], self.s.frame)
            x0, y0, x1, y1 = info["bbox"]
        else:
            w, h = self.s.project.settings.format_w, self.s.project.settings.format_h
            x0, y0, x1, y1 = 0, 0, w, h
        translate = node.get("translate", self.s.frame)
        scale = node.get("scale", self.s.frame)
        center = node.get("center", self.s.frame)
        rotate = math.radians(node.get("rotate", self.s.frame)[0])
        cx, cy = center[0] + translate[0], center[1] + translate[1]
        c, sn = math.cos(rotate), math.sin(rotate)

        def point(x, y):
            dx, dy = (x - center[0]) * scale[0], (y - center[1]) * scale[1]
            return cx + c * dx - sn * dy, cy + sn * dx + c * dy

        corners = [point(x0, y0), point(x1, y0), point(x1, y1), point(x0, y1)]
        screen = [QPointF(*self.v.to_screen(x, y)) for x, y in corners]
        top = (screen[0] + screen[1]) / 2
        center_screen = QPointF(*self.v.to_screen(cx, cy))
        top_vec = top - center_screen
        length = math.hypot(top_vec.x(), top_vec.y()) or 1.0
        rotate_handle = top + QPointF(top_vec.x() / length * 24, top_vec.y() / length * 24)
        box = QRectF(min(p.x() for p in screen), min(p.y() for p in screen),
                     max(p.x() for p in screen) - min(p.x() for p in screen),
                     max(p.y() for p in screen) - min(p.y() for p in screen))
        return node, center_screen, screen, rotate_handle, box

    def paint(self, painter: QPainter):
        if not self.v.show_overlays:
            return
        data = self.geometry()
        if data is None:
            return
        _, center, corners, rotate_handle, _ = data
        color = QColor(theme.T["accent"])
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(color, 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for i in range(4):
            painter.drawLine(corners[i], corners[(i + 1) % 4])
        for point in corners:
            painter.setBrush(QColor(theme.T["base"]))
            painter.drawRect(QRectF(point.x() - self.HANDLE / 2, point.y() - self.HANDLE / 2,
                                    self.HANDLE, self.HANDLE))
        top = (corners[0] + corners[1]) / 2
        painter.drawLine(top, rotate_handle)
        painter.setBrush(color)
        painter.drawEllipse(rotate_handle, self.HANDLE / 2, self.HANDLE / 2)
        painter.drawEllipse(center, 4, 4)
        painter.drawLine(center + QPointF(-10, 0), center + QPointF(10, 0))
        painter.drawLine(center + QPointF(0, -10), center + QPointF(0, 10))

    def press(self, event) -> bool:
        if event.button() != Qt.MouseButton.LeftButton or event.modifiers() & Qt.KeyboardModifier.AltModifier:
            return False
        data = self.geometry()
        if data is None:
            return False
        node, center, corners, rotate_handle, box = data
        pos = event.position()
        if math.hypot(pos.x() - rotate_handle.x(), pos.y() - rotate_handle.y()) <= self.HANDLE:
            mode = "rotate"
        elif any(math.hypot(pos.x() - p.x(), pos.y() - p.y()) <= self.HANDLE for p in corners):
            mode = "scale"
        elif math.hypot(pos.x() - center.x(), pos.y() - center.y()) <= 12 or box.contains(pos):
            mode = "move"
        else:
            return False
        pivot = self.v.to_image(center.x(), center.y())
        start = self.v.to_image(pos.x(), pos.y())
        self._drag = (mode, node, pos, start, pivot, list(node.get("translate", self.s.frame)),
                      list(node.get("scale", self.s.frame)), node.get("rotate", self.s.frame)[0],
                      math.atan2(start[1] - pivot[1], start[0] - pivot[0]),
                      next(_drag_ids))
        return True

    def move(self, event) -> bool:
        if self._drag is None:
            return False
        mode, node, start_screen, start, pivot, translate, scale, rotate, start_angle, drag_id = self._drag
        current = self.v.to_image(event.position().x(), event.position().y())
        key = f"viewer-transform-{node.id}-{drag_id}"
        if mode == "move":
            moved = [translate[0] + current[0] - start[0], translate[1] + current[1] - start[1]]
            self.s.set_knob(node, "translate", moved, merge_key=key)
        elif mode == "rotate":
            angle = math.atan2(current[1] - pivot[1], current[0] - pivot[0])
            value = rotate + math.degrees(angle - start_angle)
            self.s.set_knob(node, "rotate", value, merge_key=key)
        else:
            theta = math.radians(rotate)
            c, sn = math.cos(theta), math.sin(theta)
            def unrotate(point):
                dx, dy = point[0] - pivot[0], point[1] - pivot[1]
                return c * dx + sn * dy, -sn * dx + c * dy

            a, b = unrotate(start), unrotate(current)
            sx = max(0.01, scale[0] * abs(b[0] / (a[0] or 1e-6)))
            sy = max(0.01, scale[1] * abs(b[1] / (a[1] or 1e-6)))
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                ratio = max(sx / max(scale[0], 1e-6), sy / max(scale[1], 1e-6))
                sx, sy = scale[0] * ratio, scale[1] * ratio
            self.s.set_knob(node, "scale", [sx, sy], merge_key=key)
        self.v.update()
        return True

    def release(self, event) -> bool:
        if self._drag is None:
            return False
        self._drag = None
        return True

    def double_click(self, event) -> bool:
        return False

    def key(self, event) -> bool:
        return False
