"""Small precise controls shared by the viewer bar and the properties panel."""
from __future__ import annotations

import math

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QDoubleValidator, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QSizePolicy, QSlider, QToolButton, QWidget

from vertexcomp.ui import theme


def fmt_num(v: float, decimals: int = 3) -> str:
    if v == int(v) and abs(v) < 1e9:
        return str(int(v))
    s = f"{v:.{decimals}f}".rstrip("0").rstrip(".")
    return s if s not in ("-0", "") else "0"


class ScrubField(QLineEdit):
    """Numeric field: type a value, scrub with left-drag (Shift x10, Ctrl x0.1)
    or the mouse wheel. Emits valueEdited(value, final)."""

    valueEdited = Signal(float, bool)

    def __init__(self, value=0.0, step=0.01, decimals=3, integer=False, minimum=None, maximum=None, parent=None):
        super().__init__(parent)
        self.step, self.decimals, self.integer = step, decimals, integer
        self.minimum, self.maximum = minimum, maximum
        self._value = float(value)
        self._press = None
        self._scrubbing = False
        self.setFont(theme.mono_font(9))
        self.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.setFixedHeight(18)
        self.setMinimumWidth(44)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setValidator(QDoubleValidator(self))
        self.editingFinished.connect(self._typed)
        self._show()

    def sizeHint(self):
        return QSize(58, 18)

    def value(self) -> float:
        return self._value

    def setValue(self, v: float):
        self._value = float(v)
        if not self.hasFocus() or self._scrubbing:
            self._show()

    def _clamp(self, v):
        if self.minimum is not None:
            v = max(self.minimum, v)
        if self.maximum is not None:
            v = min(self.maximum, v)
        return round(v) if self.integer else v

    def _show(self):
        self.setText(fmt_num(self._value, self.decimals))
        self.setCursorPosition(0)

    def _typed(self):
        try:
            v = self._clamp(float(self.text()))
        except ValueError:
            self._show()
            return
        if v != self._value:
            self._value = v
            self.valueEdited.emit(v, True)
        self._show()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and not self.hasFocus():
            self._press = e.globalPosition().toPoint()
            self._start = self._value
            self._scrubbing = False
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._press is not None:
            dx = e.globalPosition().toPoint().x() - self._press.x()
            if not self._scrubbing and abs(dx) > 3:
                self._scrubbing = True
                self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
            if self._scrubbing:
                mult = 10.0 if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
                if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    mult = 0.1
                step = 1.0 if self.integer else self.step
                v = self._clamp(self._start + dx * step * mult)
                if v != self._value:
                    self._value = v
                    self._show()
                    self.valueEdited.emit(v, False)
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._press is not None:
            was = self._scrubbing
            self._press = None
            self._scrubbing = False
            self.unsetCursor()
            if was:
                self.valueEdited.emit(self._value, True)
            else:
                self.setFocus()
                self.selectAll()
            return
        super().mouseReleaseEvent(e)

    def wheelEvent(self, e):
        if not self.hasFocus() and not self.underMouse():
            return super().wheelEvent(e)
        d = 1 if e.angleDelta().y() > 0 else -1
        mult = 10.0 if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
        v = self._clamp(self._value + d * (1.0 if self.integer else self.step * 10) * mult)
        self._value = v
        self._show()
        self.valueEdited.emit(v, True)
        e.accept()

    def focusOutEvent(self, e):
        super().focusOutEvent(e)
        self._show()


class FloatSlider(QSlider):
    """Slider over a soft range; values outside stay reachable through the field."""

    moved = Signal(float, bool)

    def __init__(self, lo=0.0, hi=1.0, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.lo, self.hi = lo, hi
        self.setRange(0, 1000)
        self.setFixedHeight(16)
        self._block = False
        self.valueChanged.connect(self._changed)
        self.sliderReleased.connect(lambda: self.moved.emit(self.fvalue(), True))

    def fvalue(self):
        return self.lo + (self.hi - self.lo) * self.value() / 1000.0

    def setFValue(self, v):
        self._block = True
        span = (self.hi - self.lo) or 1.0
        self.setValue(int(round(max(0.0, min(1.0, (v - self.lo) / span)) * 1000)))
        self._block = False

    def _changed(self, _):
        if not self._block:
            self.moved.emit(self.fvalue(), not self.isSliderDown())


class NumberWithSlider(QWidget):
    valueEdited = Signal(float, bool)

    def __init__(self, value, lo, hi, integer=False, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        step = 1.0 if integer else max(1e-4, (hi - lo) / 200.0)
        self.field = ScrubField(value, step=step, integer=integer)
        self.field.setFixedWidth(64)
        self.slider = FloatSlider(lo, hi)
        self.slider.setFValue(value)
        lay.addWidget(self.field)
        lay.addWidget(self.slider, 1)
        self.field.valueEdited.connect(self._from_field)
        self.slider.moved.connect(self._from_slider)
        self.integer = integer

    def _from_field(self, v, final):
        self.slider.setFValue(v)
        self.valueEdited.emit(v, final)

    def _from_slider(self, v, final):
        if self.integer:
            v = float(round(v))
        self.field.setValue(v)
        self.valueEdited.emit(v, final)

    def setValue(self, v):
        self.field.setValue(v)
        if not self.slider.isSliderDown():
            self.slider.setFValue(v)


class Swatch(QToolButton):
    """Colour preview (display-referred approximation of a linear colour)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rgb = (0.0, 0.0, 0.0)
        self.setFixedSize(18, 18)

    def setLinear(self, rgb):
        self.rgb = tuple(rgb[:3]) + (0.0,) * max(0, 3 - len(rgb))
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)

        def enc(v):
            v = max(0.0, min(1.0, v))
            return v * 12.92 if v <= 0.0031308 else 1.055 * math.pow(v, 1 / 2.4) - 0.055

        c = QColor.fromRgbF(enc(self.rgb[0]), enc(self.rgb[1]), enc(self.rgb[2]))
        p.fillRect(self.rect(), c)
        p.setPen(QPen(QColor(theme.T["line"]), 1))
        p.drawRect(QRectF(0.5, 0.5, self.width() - 1, self.height() - 1))


class IconButton(QToolButton):
    def __init__(self, icon_name, tip="", checkable=False, parent=None):
        super().__init__(parent)
        self.icon_name = icon_name
        self.setIcon(theme.icon(icon_name))
        self.setToolTip(tip)
        self.setCheckable(checkable)
        self.setAutoRaise(False)
        self.setFixedSize(20, 20)
        self.setIconSize(QSize(14, 14))
