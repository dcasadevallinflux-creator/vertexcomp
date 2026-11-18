"""Freehand viewer tool that stores brush strokes as Paint node shapes."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen

from vertexcomp import _vccore as vc


class PaintTool:
    def __init__(self, session, viewer, node_name):
        self.s, self.v, self.node_name = session, viewer, node_name
        self._points = []
        self._color = QColor()
        self._size = 1.0
        self._opacity = 1.0

    def _node(self):
        return self.s.project.find(self.node_name)

    def _image_point(self, pos):
        return self.v.to_image(pos.x(), pos.y())

    def press(self, e) -> bool:
        if e.button() != Qt.MouseButton.LeftButton or e.modifiers() & Qt.KeyboardModifier.AltModifier:
            return False
        node = self._node()
        if node is None:
            return False
        color = node.get("color", float(self.s.frame))
        self._color = QColor.fromRgbF(*[max(0.0, min(1.0, c)) for c in color[:3]])
        self._opacity = max(0.0, min(1.0, color[3] if len(color) > 3 else 1.0))
        self._size = max(1.0, float(node.get("size", float(self.s.frame))[0]))
        self._points = [self._image_point(e.position())]
        self.v.update()
        return True

    def move(self, e) -> bool:
        if not self._points:
            return False
        point = self._image_point(e.position())
        if math.hypot(point[0] - self._points[-1][0], point[1] - self._points[-1][1]) >= 0.25:
            self._points.append(point)
        self.v.update()
        return True

    def _shape_points(self):
        points = self._points
        radius = self._size * 0.5
        if len(points) == 1:
            cx, cy = points[0]
            return [(cx + math.cos(i * math.tau / 32) * radius,
                     cy + math.sin(i * math.tau / 32) * radius) for i in range(32)]

        directions = []
        for i in range(len(points)):
            before, after = points[max(0, i - 1)], points[min(len(points) - 1, i + 1)]
            dx, dy = after[0] - before[0], after[1] - before[1]
            length = math.hypot(dx, dy) or 1.0
            directions.append((dx / length, dy / length))

        left, right = [], []
        for (x, y), (dx, dy) in zip(points, directions):
            nx, ny = -dy, dx
            left.append((x + nx * radius, y + ny * radius))
            right.append((x - nx * radius, y - ny * radius))

        end_angle = math.atan2(directions[-1][1], directions[-1][0])
        end_cap = [(points[-1][0] + math.cos(end_angle + math.pi / 2 - i * math.pi / 12) * radius,
                    points[-1][1] + math.sin(end_angle + math.pi / 2 - i * math.pi / 12) * radius)
                   for i in range(1, 13)]
        start_angle = math.atan2(directions[0][1], directions[0][0])
        start_cap = [(points[0][0] + math.cos(start_angle - math.pi / 2 - i * math.pi / 12) * radius,
                      points[0][1] + math.sin(start_angle - math.pi / 2 - i * math.pi / 12) * radius)
                     for i in range(1, 13)]
        return left + end_cap + list(reversed(right)) + start_cap

    def release(self, e) -> bool:
        if not self._points:
            return False
        point = self._image_point(e.position())
        if math.hypot(point[0] - self._points[-1][0], point[1] - self._points[-1][1]) >= 0.25:
            self._points.append(point)
        node = self._node()
        if node is not None:
            stroke = vc.RotoShape()
            stroke.name = f"Stroke {len(node.roto_shapes()) + 1}"
            stroke.color = [self._color.redF(), self._color.greenF(), self._color.blueF()]
            stroke.opacity = self._opacity
            stroke.keys = {float(self.s.frame): [vc.RotoPoint(x, y) for x, y in self._shape_points()]}
            self.s.set_roto(node, node.roto_shapes() + [stroke])
        self._points = []
        self.v.update()
        return True

    def paint(self, painter):
        if not self._points:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(self._color, max(1.0, self._size * self.v.zoom), Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        path = QPainterPath(QPointF(*self.v.to_screen(*self._points[0])))
        for point in self._points[1:]:
            path.lineTo(QPointF(*self.v.to_screen(*point)))
        painter.drawPath(path)

    def key(self, e) -> bool:
        if e.key() == Qt.Key.Key_Escape and self._points:
            self._points = []
            self.v.update()
            return True
        return False

    def double_click(self, _e) -> bool:
        return False
