"""Helpers for generating the Text node's cached, portable glyph bitmap."""
from __future__ import annotations

import base64

from PySide6.QtCore import QByteArray, QBuffer, QIODevice, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter


def _srgb(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * 12.92 if value <= 0.0031308 else 1.055 * value ** (1 / 2.4) - 0.055


def update_text_bitmap(node):
    text = node.get("text")
    font = QFont(node.get("font"))
    font.setPixelSize(max(1, min(1024, int(round(node.get("size")[0])))))
    metrics = QFontMetrics(font)
    lines = text.splitlines() or [""]
    width = max(1, min(8192, max(metrics.horizontalAdvance(line) for line in lines)))
    height = max(1, min(8192, metrics.height() * len(lines)))
    image = QImage(width, height, QImage.Format.Format_RGBA8888)
    image.fill(Qt.GlobalColor.transparent)
    color = node.get("color")
    pen = QColor.fromRgbF(_srgb(color[0]), _srgb(color[1]), _srgb(color[2]), max(0.0, min(1.0, color[3])))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    painter.setFont(font)
    painter.setPen(pen)
    for index, line in enumerate(lines):
        painter.drawText(0, metrics.ascent() + index * metrics.height(), line)
    painter.end()

    data = QByteArray()
    buffer = QBuffer(data)
    if not buffer.open(QIODevice.OpenModeFlag.WriteOnly):
        raise RuntimeError("Could not open in-memory buffer for Text node bitmap")
    if not image.save(buffer, "PNG"):
        buffer.close()
        raise RuntimeError("Could not encode Text node bitmap")
    buffer.close()
    node.set("bitmap", base64.b64encode(bytes(data)).decode("ascii"))
