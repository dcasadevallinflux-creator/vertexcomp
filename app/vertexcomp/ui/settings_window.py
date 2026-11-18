"""Settings window: searchable section list on the left, live-applied values."""
from __future__ import annotations

import json

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vertexcomp import _vccore as vc
from vertexcomp.settings import DEFAULTS
from vertexcomp.ui import theme

SECTIONS = [
    ("General", "general"), ("Performance", "performance"), ("Color Management", "color"), ("Viewer", "viewer"),
    ("Node Graph", "nodegraph"), ("Hotkeys", "hotkeys"), ("Appearance", "appearance"), ("Paths & Plugins", "paths"),
    ("AI Models", "ai"),
]

CHOICES = {
    ("color", "working_space"): ["Linear Rec.709", "ACEScg"],
    ("color", "display"): ["sRGB", "Rec.709", "Gamma 2.2", "linear"],
    ("viewer", "background"): ["checker", "black", "gray"],
    ("nodegraph", "edge_style"): ["curved", "straight", "orthogonal"],
    ("appearance", "theme"): ["dark", "graphite"],
}
LABELS = {
    "format_w": "default width", "format_h": "default height", "fps": "frame rate", "first_frame": "first frame",
    "last_frame": "last frame", "autosave_minutes": "autosave every (min, 0 = off)", "autosave_backups": "backups kept",
    "undo_depth": "undo levels", "cache_gb": "RAM cache (GB)", "threads": "threads (0 = all cores)",
    "readahead_frames": "playback read-ahead (frames)", "interactive_downscale": "downscale when zoomed out",
    "play_every_frame": "play every frame by default", "working_space": "working space",
    "display": "default display", "exposure": "default exposure", "gamma": "default gamma",
    "background": "background", "grid_zoom": "pixel grid from zoom", "values_zoom": "pixel values from zoom",
    "show_values": "print values at extreme zoom", "zebra_threshold": "zebra threshold",
    "nearest_zoom": "nearest filtering from zoom", "overlay_color": "overlay colour", "bbox_color": "bbox colour",
    "edge_style": "edge style", "snap_to_grid": "snap to grid", "grid_size": "grid size", "font_size": "font size",
    "autoconnect": "auto-connect new nodes", "max_properties": "maximum open property panels",
    "theme": "theme", "ui_scale": "UI scale (restart)",
    "read_dir": "default read directory", "write_dir": "default write directory",
    "plugin_dirs": "plugin directories (one per line)",
}
LIVE_NOTE = {"ui_scale": "applies after restart"}


class SettingsWindow(QWidget):
    changed = Signal(str, str)

    def __init__(self, settings, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Settings")
        self.resize(760, 520)
        self.st = settings
        self.search = QLineEdit()
        self.search.setPlaceholderText("search settings")
        self.search.textChanged.connect(self._filter)
        self.list = QListWidget()
        self.list.setFixedWidth(170)
        self.stack = QStackedWidget()
        self.pages = {}
        self.rows = {}  # (section, key) -> label widget
        for title, sec in SECTIONS:
            self.list.addItem(title)
            page = self._make_page(sec)
            self.pages[sec] = page
            self.stack.addWidget(page)
        self.list.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.list.setCurrentRow(0)
        left = QVBoxLayout()
        left.addWidget(self.search)
        left.addWidget(self.list, 1)
        restore = QPushButton("Restore section defaults")
        restore.clicked.connect(self.restore)
        save = QPushButton("Save")
        save.clicked.connect(self.st.save)
        right = QVBoxLayout()
        right.addWidget(self.stack, 1)
        btns = QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(restore)
        btns.addWidget(save)
        right.addLayout(btns)
        lay = QHBoxLayout(self)
        lay.addLayout(left)
        lay.addLayout(right, 1)

    def _make_page(self, sec):
        if sec == "hotkeys":
            return self._hotkeys_page()
        if sec == "ai":
            from vertexcomp.ui.ai_settings import AISettingsPage
            page = AISettingsPage(self.st, on_change=lambda s, k: self.changed.emit(s, k))
            self.rows.update(page.rows)
            return page
        return self._page(sec)

    def _set(self, sec, key, value):
        self.st.set(sec, key, value)
        self.st.save()
        self.changed.emit(sec, key)

    def _page(self, sec):
        w = QWidget()
        form = QFormLayout(w)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        for key, default in DEFAULTS[sec].items():
            if isinstance(default, dict) or key == "recent_projects":
                continue
            val = self.st.get(sec, key)
            choices = CHOICES.get((sec, key))
            if choices:
                ed = QComboBox()
                ed.addItems(choices)
                ed.setCurrentText(str(val))
                ed.currentTextChanged.connect(lambda v, s=sec, k=key: self._set(s, k, v))
            elif isinstance(default, bool):
                ed = QCheckBox()
                ed.setChecked(bool(val))
                ed.toggled.connect(lambda v, s=sec, k=key: self._set(s, k, v))
            elif isinstance(default, int):
                ed = QSpinBox()
                if (sec, key) == ("nodegraph", "max_properties"):
                    ed.setRange(1, 100)
                else:
                    ed.setRange(0, 100000)
                ed.setValue(int(val))
                ed.valueChanged.connect(lambda v, s=sec, k=key: self._set(s, k, v))
            elif isinstance(default, float):
                ed = QDoubleSpinBox()
                ed.setDecimals(3)
                ed.setRange(-100.0, 100000.0)
                ed.setSingleStep(0.5 if default >= 1 else 0.05)
                ed.setValue(float(val))
                ed.valueChanged.connect(lambda v, s=sec, k=key: self._set(s, k, v))
            elif isinstance(default, list):
                ed = QPlainTextEdit("\n".join(val))
                ed.setFixedHeight(60)
                ed.textChanged.connect(lambda e=ed, s=sec, k=key: self._set(s, k, [x for x in e.toPlainText().splitlines() if x.strip()]))
            elif key.endswith("_color"):
                ed = QPushButton(str(val))
                ed.clicked.connect(lambda _=False, b=ed, s=sec, k=key: self._pick(b, s, k))
            elif key.endswith("_dir"):
                ed = QWidget()
                hl = QHBoxLayout(ed)
                hl.setContentsMargins(0, 0, 0, 0)
                le = QLineEdit(str(val))
                le.editingFinished.connect(lambda e=le, s=sec, k=key: self._set(s, k, e.text()))
                b = QPushButton("...")
                b.setFixedWidth(28)
                b.clicked.connect(lambda _=False, e=le, s=sec, k=key: self._browse(e, s, k))
                hl.addWidget(le, 1)
                hl.addWidget(b)
            else:
                ed = QLineEdit(str(val))
                ed.editingFinished.connect(lambda e=ed, s=sec, k=key: self._set(s, k, e.text()))
            label = QLabel(LABELS.get(key, key))
            label.setObjectName("knobLabel")
            if key in LIVE_NOTE:
                label.setToolTip(LIVE_NOTE[key])
            form.addRow(label, ed)
            self.rows[(sec, key)] = label
        if sec == "nodegraph":
            cols = self.st.get("nodegraph", "colors")
            for cat in sorted(cols):
                b = QPushButton(cols[cat])
                b.setStyleSheet(f"background: {cols[cat]};")
                b.clicked.connect(lambda _=False, c=cat, btn=b: self._pick_cat(btn, c))
                lab = QLabel(f"{cat} colour")
                lab.setObjectName("knobLabel")
                form.addRow(lab, b)
                self.rows[(sec, cat)] = lab
        if sec == "color":
            note = QLabel("Built-in colour management (transfer functions and Rec.709/AP1/AP0 primaries). "
                          "Colourspaces: " + ", ".join(vc.colorspaces()))
            note.setWordWrap(True)
            note.setObjectName("knobLabel")
            form.addRow(note)
        return w

    def _hotkeys_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.hk_filter = QLineEdit()
        self.hk_filter.setPlaceholderText("filter")
        self.hk_table = QTableWidget(0, 2)
        self.hk_table.setHorizontalHeaderLabels(["action", "shortcut"])
        self.hk_table.horizontalHeader().setStretchLastSection(True)
        self.hk_table.verticalHeader().setVisible(False)
        self.hk_conflicts = QLabel("")
        self.hk_conflicts.setStyleSheet(f"color: {theme.T['error']};")
        hk = self.st.get("hotkeys", "Merge") and self.st.data["hotkeys"]
        self.hk_table.setRowCount(len(hk))
        for i, (action, seq) in enumerate(sorted(hk.items())):
            self.hk_table.setItem(i, 0, QTableWidgetItem(action))
            ed = QKeySequenceEdit(QKeySequence(seq))
            ed.editingFinished.connect(lambda a=action, e=ed: self._set_hotkey(a, e.keySequence().toString()))
            self.hk_table.setCellWidget(i, 1, ed)
        self.hk_filter.textChanged.connect(
            lambda t: [self.hk_table.setRowHidden(i, t.lower() not in self.hk_table.item(i, 0).text().lower())
                       for i in range(self.hk_table.rowCount())])
        io = QHBoxLayout()
        exp = QPushButton("Export...")
        exp.clicked.connect(self._export_hotkeys)
        imp = QPushButton("Import...")
        imp.clicked.connect(self._import_hotkeys)
        io.addWidget(self.hk_conflicts, 1)
        io.addWidget(imp)
        io.addWidget(exp)
        lay.addWidget(self.hk_filter)
        lay.addWidget(self.hk_table, 1)
        lay.addLayout(io)
        note = QLabel("Shortcut changes apply after restart.")
        note.setObjectName("knobLabel")
        lay.addWidget(note)
        self._check_conflicts()
        return w

    def _set_hotkey(self, action, seq):
        hk = dict(self.st.data["hotkeys"])
        hk[action] = seq
        self.st.data["hotkeys"] = hk
        self.st.save()
        self._check_conflicts()

    def _check_conflicts(self):
        seen, bad = {}, []
        for a, s in self.st.data["hotkeys"].items():
            if not s:
                continue
            if s in seen:
                bad.append(f"{s}: {seen[s]} / {a}")
            seen[s] = a
        self.hk_conflicts.setText("Conflicts: " + "; ".join(bad) if bad else "")

    def _export_hotkeys(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export hotkeys", "hotkeys.json", "JSON (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.st.data["hotkeys"], f, indent=2)

    def _import_hotkeys(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import hotkeys", "", "JSON (*.json)")
        if path:
            with open(path, encoding="utf-8") as f:
                self.st.data["hotkeys"].update(json.load(f))
            self.st.save()
            self._check_conflicts()

    def _pick(self, btn, sec, key):
        from PySide6.QtGui import QColor
        c = QColorDialog.getColor(QColor(btn.text()), self)
        if c.isValid():
            btn.setText(c.name())
            self._set(sec, key, c.name())

    def _pick_cat(self, btn, cat):
        from PySide6.QtGui import QColor
        c = QColorDialog.getColor(QColor(btn.text()), self)
        if c.isValid():
            cols = dict(self.st.get("nodegraph", "colors"))
            cols[cat] = c.name()
            btn.setText(c.name())
            btn.setStyleSheet(f"background: {c.name()};")
            self._set("nodegraph", "colors", cols)

    def _browse(self, edit, sec, key):
        d = QFileDialog.getExistingDirectory(self, "Directory", edit.text())
        if d:
            edit.setText(d)
            self._set(sec, key, d)

    def restore(self):
        sec = SECTIONS[self.list.currentRow()][1]
        self.st.restore_section(sec)
        self.st.save()
        old = self.pages[sec]
        idx = self.stack.indexOf(old)
        page = self._make_page(sec)
        self.stack.insertWidget(idx, page)
        self.stack.removeWidget(old)
        old.deleteLater()
        self.pages[sec] = page
        self.stack.setCurrentIndex(idx)
        self.changed.emit(sec, "*")

    def _filter(self, text):
        t = text.lower().strip()
        for i, (title, sec) in enumerate(SECTIONS):
            hit = not t or t in title.lower() or any(
                t in (lab.text().lower() + " " + key) for (s, key), lab in self.rows.items() if s == sec)
            self.list.item(i).setHidden(not hit)
            if hit and t:
                for (s, _key), lab in self.rows.items():
                    if s == sec:
                        lab.setStyleSheet(f"color: {theme.T['accent']};" if t in lab.text().lower() else "")
        if t:
            for i in range(self.list.count()):
                if not self.list.item(i).isHidden():
                    self.list.setCurrentRow(i)
                    break
