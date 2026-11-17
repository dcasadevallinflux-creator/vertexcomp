"""Timeline ruler and transport (playback with real-time clock and read-ahead)."""
from __future__ import annotations

import collections
import math
import time

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from vertexcomp.ui import theme
from vertexcomp.ui.widgets import IconButton, ScrubField

FPS_PRESETS = ["23.976", "24", "25", "29.97", "30", "48", "50", "59.94", "60"]


class Ruler(QWidget):
    scrubbed = Signal(int)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.s = session
        self.setMinimumHeight(34)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.cached: set[int] = set()
        self.keys: list[int] = []
        self._drag = None
        self.setMouseTracking(True)

    def sizeHint(self):
        return QSize(600, 34)

    def _range(self):
        st = self.s.project.settings
        return st.first, st.last

    def f2x(self, f):
        a, b = self._range()
        span = max(1, b - a)
        return 8 + (f - a) / span * (self.width() - 16)

    def x2f(self, x):
        a, b = self._range()
        span = max(1, b - a)
        return int(round(a + (x - 8) / max(1, self.width() - 16) * span))

    def paintEvent(self, e):
        p = QPainter(self)
        T = theme.T
        p.fillRect(self.rect(), QColor(T["base"]))
        a, b = self._range()
        h = self.height()
        # Cache bar
        pw = max(1.0, (self.width() - 16) / max(1, b - a + 1))
        for f in self.cached:
            if a <= f <= b:
                p.fillRect(QRectF(self.f2x(f) - pw / 2, h - 4, pw + 0.5, 3), QColor(80, 140, 90))
        # Ticks
        span = max(1, b - a)
        target = max(1, int(span / max(1, self.width() / 60)))
        mag = 10 ** int(math.floor(math.log10(target)))
        step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= target)
        p.setFont(theme.mono_font(8))
        f0 = int(math.ceil(a / step) * step)
        for f in range(f0, b + 1, step):
            x = self.f2x(f)
            p.setPen(QColor(T["dim"]))
            p.drawLine(QPointF(x, 0), QPointF(x, 7))
            p.drawText(QPointF(x + 2, 16), str(f))
        minor = max(1, step // 5)
        p.setPen(QColor(T["line"]).lighter(160))
        for f in range(int(math.ceil(a / minor) * minor), b + 1, minor):
            x = self.f2x(f)
            p.drawLine(QPointF(x, 0), QPointF(x, 3))
        # Keys
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(110, 160, 220))
        for k in self.keys:
            if a <= k <= b:
                x = self.f2x(k)
                p.drawRect(QRectF(x - 1.5, 18, 3, 9))
        # Playhead
        x = self.f2x(self.s.frame)
        p.setPen(QPen(QColor(T["accent"]), 1))
        p.drawLine(QPointF(x, 0), QPointF(x, h))
        p.setBrush(QColor(T["accent"]))
        p.drawRect(QRectF(x - 3, 0, 6, 4))

    def mousePressEvent(self, e):
        self._drag = True
        self.scrubbed.emit(self.x2f(e.position().x()))

    def mouseMoveEvent(self, e):
        if self._drag:
            a, b = self._range()
            self.scrubbed.emit(max(a, min(b, self.x2f(e.position().x()))))

    def mouseReleaseEvent(self, e):
        self._drag = None


class Transport(QFrame):
    """Playback controls. Real-time clocked: the frame shown is derived from
    elapsed wall time, so slow frames are skipped (unless 'every frame')."""

    def __init__(self, session, viewer_panel):
        super().__init__(viewer_panel)
        self.setObjectName("transport")
        self.s = session
        self.vp = viewer_panel
        self.playing = False
        self.direction = 1
        self._t0 = 0.0
        self._f0 = 0
        self._shown = collections.deque(maxlen=48)
        self.dropped = 0

        self.ruler = Ruler(session)
        self.ruler.scrubbed.connect(session.set_frame)

        def btn(icon, tip, fn):
            b = IconButton(icon, tip)
            b.clicked.connect(fn)
            return b

        st = session.project.settings
        self.first = ScrubField(st.first, integer=True)
        self.first.setFixedWidth(50)
        self.first.setToolTip("First frame")
        self.first.valueEdited.connect(lambda v, f: f and session.set_range(int(v), session.project.settings.last))
        self.last = ScrubField(st.last, integer=True)
        self.last.setFixedWidth(50)
        self.last.setToolTip("Last frame")
        self.last.valueEdited.connect(lambda v, f: f and session.set_range(session.project.settings.first, int(v)))
        self.cur = ScrubField(session.frame, integer=True)
        self.cur.setFixedWidth(56)
        self.cur.setToolTip("Current frame")
        self.cur.valueEdited.connect(lambda v, f: session.set_frame(int(v)))
        self.fps = QComboBox()
        self.fps.setEditable(True)
        self.fps.addItems(FPS_PRESETS)
        self.fps.setCurrentText(f"{st.fps:g}")
        self.fps.setFixedWidth(64)
        self.fps.setToolTip("Playback rate (frames per second)")
        self.fps.currentTextChanged.connect(self._fps_changed)
        self.loop = QComboBox()
        self.loop.addItems(["loop", "bounce", "once"])
        self.loop.setToolTip("Playback mode")
        self.every = QComboBox()
        self.every.addItems(["real time", "every frame"])
        self.every.setToolTip("Real time skips frames that are not ready; every frame waits for each")
        if session.settings.get("performance", "play_every_frame"):
            self.every.setCurrentIndex(1)
        self.achieved = QLabel("")
        self.achieved.setFont(theme.mono_font(8))
        self.achieved.setMinimumWidth(110)

        row = QHBoxLayout()
        row.setContentsMargins(4, 2, 4, 2)
        row.setSpacing(3)
        row.addWidget(self.first)
        row.addWidget(btn("to_start", "First frame (Home)", self.to_start))
        row.addWidget(btn("step_back", "Previous frame (Left)", lambda: self.step(-1)))
        row.addWidget(btn("play_back", "Play backwards", lambda: self.play(-1)))
        row.addWidget(btn("stop", "Stop", self.stop))
        row.addWidget(btn("play", "Play (Space)", lambda: self.play(1)))
        row.addWidget(btn("step_fwd", "Next frame (Right)", lambda: self.step(1)))
        row.addWidget(btn("to_end", "Last frame (End)", self.to_end))
        row.addWidget(self.last)
        row.addSpacing(8)
        row.addWidget(self.cur)
        row.addSpacing(8)
        row.addWidget(self.fps)
        row.addWidget(self.loop)
        row.addWidget(self.every)
        row.addWidget(self.achieved)
        row.addStretch(1)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.ruler)
        lay.addLayout(row)

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self.cache_timer = QTimer(self)
        self.cache_timer.timeout.connect(self._refresh_cache_bar)
        self.cache_timer.start(500)

        session.frameChanged.connect(self._frame_changed)
        session.rangeChanged.connect(self._range_changed)
        session.selectionChanged.connect(self._keys_changed)
        session.nodeChanged.connect(lambda n: self._keys_changed())

        hk = session.settings.data["hotkeys"]
        for name, fn in (("play", self.toggle), ("step_back", lambda: self.step(-1)),
                         ("step_forward", lambda: self.step(1)), ("first_frame", self.to_start),
                         ("last_frame", self.to_end), ("prev_key", lambda: self.jump_key(-1)),
                         ("next_key", lambda: self.jump_key(1))):
            sc = QShortcut(QKeySequence(hk.get(name, "")), viewer_panel.window() or viewer_panel)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(fn)

    # ------------------------------------------------------------ helpers
    def fps_value(self):
        try:
            return max(0.1, float(self.fps.currentText()))
        except ValueError:
            return self.s.project.settings.fps

    def _fps_changed(self, t):
        try:
            self.s.project.settings.fps = float(t)
        except ValueError:
            pass
        if self.playing:
            self._t0, self._f0 = time.perf_counter(), self.s.frame

    def _frame_changed(self, f):
        self.cur.setValue(f)
        self.ruler.update()

    def _range_changed(self):
        st = self.s.project.settings
        self.first.setValue(st.first)
        self.last.setValue(st.last)
        self.ruler.update()

    def _keys_changed(self):
        self.ruler.keys = self.s.key_frames()
        self.ruler.update()

    def _refresh_cache_bar(self):
        node = self.s.viewed_node()
        st = self.s.project.settings
        cached = set()
        if node is not None and st.last - st.first < 2000:
            eng = self.s.project.engine
            for f in range(st.first, st.last + 1):
                try:
                    if self.vp.cache.get(f) is not None or eng.is_cached(node, f):
                        cached.add(f)
                except RuntimeError:
                    pass
        if cached != self.ruler.cached:
            self.ruler.cached = cached
            self.ruler.update()

    # ------------------------------------------------------------ actions
    def to_start(self):
        self.s.set_frame(self.s.project.settings.first)

    def to_end(self):
        self.s.set_frame(self.s.project.settings.last)

    def step(self, d):
        self.stop()
        st = self.s.project.settings
        f = self.s.frame + d
        if f > st.last:
            f = st.first
        if f < st.first:
            f = st.last
        self.s.set_frame(f)

    def jump_key(self, d):
        keys = self.s.key_frames()
        f = self.s.frame
        cand = [k for k in keys if (k > f if d > 0 else k < f)]
        if cand:
            self.s.set_frame(min(cand) if d > 0 else max(cand))

    def toggle(self):
        if self.playing:
            self.stop()
        else:
            self.play(self.direction)

    def play(self, direction=1):
        self.direction = direction
        self.playing = True
        self._t0 = time.perf_counter()
        self._f0 = self.s.frame
        self._shown.clear()
        self.dropped = 0
        self.vp.cache.start(self.s.frame, direction)
        self.timer.start(4)

    def stop(self):
        if not self.playing:
            return
        self.playing = False
        self.timer.stop()
        self.vp.cache.stop()
        self.achieved.setText("")
        self.vp.request()

    def _wrap(self, f):
        st = self.s.project.settings
        a, b = st.first, st.last
        mode = self.loop.currentText()
        span = b - a + 1
        if a <= f <= b:
            return f, False
        if mode == "once":
            return max(a, min(b, f)), True
        if mode == "bounce":
            period = max(1, 2 * span - 2)
            k = (f - a) % period
            return (a + k if k < span else a + period - k), False
        return a + (f - a) % span, False

    def _tick(self):
        fps = self.fps_value()
        now = time.perf_counter()
        every = self.every.currentIndex() == 1
        if every:
            target = self.s.frame + self.direction
            if self._shown and now - self._shown[-1] < 1.0 / fps:
                return
        else:
            target = self._f0 + int(math.floor((now - self._t0) * fps)) * self.direction
        f, stop = self._wrap(target)
        mode = self.loop.currentText()
        if mode == "bounce" and not (self.s.project.settings.first <= target <= self.s.project.settings.last) and every:
            self.direction = -self.direction
            f, _ = self._wrap(self.s.frame + self.direction)
        if f == self.s.frame and not every:
            return
        self.vp.cache.advance(f)
        if self.vp.show_cached(f):
            if not every and self._shown and f != self.s.frame + self.direction:
                self.dropped += 1
            self._shown.append(now)
            self.s.frame = f
            self.cur.setValue(f)
            self.ruler.update()
            self.s.frameChanged.emit(f)  # keeps panels in sync; viewer reuses the cached buffer
        if stop:
            self.stop()
        if len(self._shown) >= 2:
            dt = self._shown[-1] - self._shown[0]
            got = (len(self._shown) - 1) / dt if dt > 0 else 0.0
            warn = got < fps * 0.95 and len(self._shown) > 10
            self.achieved.setText(f"{got:5.1f} fps{'  slow' if warn else ''}")
            self.achieved.setStyleSheet(f"color: {theme.T['error'] if warn else theme.T['dim']};")
