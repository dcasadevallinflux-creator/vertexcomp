"""Main window: viewer in the centre, node graph below, properties on the right."""
from __future__ import annotations

import glob
import os
import shutil
import threading
import time
import traceback

from PySide6.QtCore import QByteArray, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
)

from vertexcomp import APP_NAME, VERSION
from vertexcomp import _vccore as vc
from vertexcomp.project import OP_TYPES, SchemaError, write_node_range
from vertexcomp.settings import config_dir
from vertexcomp.ui import theme
from vertexcomp.ui.nodegraph import NodeGraphView
from vertexcomp.ui.paint_tool import PaintTool
from vertexcomp.ui.panels import CurveEditor, LayersPanel, LogPanel, ProfilerPanel, ScopesPanel, ScriptConsole
from vertexcomp.ui.properties import PropertiesPanel, set_read_file
from vertexcomp.ui.roto_tool import RotoTool
from vertexcomp.ui.session import Session
from vertexcomp.ui.settings_window import SettingsWindow
from vertexcomp.ui.transform_tool import TransformTool
from vertexcomp.ui.viewer import ViewerPanel


class RenderJob(QObject):
    progress = Signal(int, int, int)
    finished = Signal(str)
    log = Signal(str, str)

    def __init__(self, session, node, first, last):
        super().__init__()
        self.s, self.node, self.first, self.last = session, node, first, last
        self.token = vc.CancelToken()
        self.thread = threading.Thread(target=self._run, name="render", daemon=True)

    def start(self):
        self.thread.start()

    def cancel(self):
        self.token.cancel()

    def _run(self):
        t0 = time.perf_counter()
        try:
            write_node_range(self.s.project, self.node, self.first, self.last,
                             progress=lambda d, t, f: self.progress.emit(d, t, f), cancel=self.token,
                             log=lambda m: self.log.emit("info", m))
            self.finished.emit(f"{self.node.name}: {self.last - self.first + 1} frames in {time.perf_counter() - t0:.1f}s")
        except vc.Cancelled:
            self.finished.emit(f"{self.node.name}: render cancelled")
        except Exception as e:
            self.log.emit("error", f"{self.node.name}: {e}")
            self.finished.emit(f"{self.node.name}: render failed")


class MainWindow(QMainWindow):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.s = Session(settings, self)
        self.setDockOptions(QMainWindow.DockOption.AllowTabbedDocks | QMainWindow.DockOption.AllowNestedDocks |
                            QMainWindow.DockOption.AnimatedDocks)
        self.viewer = ViewerPanel(self.s)
        self.setCentralWidget(self.viewer)
        self.graph = NodeGraphView(self.s)
        self.props = PropertiesPanel(self.s)
        self.layers = LayersPanel(self.s, self.viewer)
        self.logp = LogPanel(self.s)
        self.console = ScriptConsole(self.s, self)
        self.profiler = ProfilerPanel(self.s)
        self.scopes = ScopesPanel(self.s, self.viewer)
        self.curves = CurveEditor(self.s)
        self.docks = {}
        self._dock("Node Graph", self.graph, Qt.DockWidgetArea.BottomDockWidgetArea)
        self._dock("Curve Editor", self.curves, Qt.DockWidgetArea.BottomDockWidgetArea, tab_with="Node Graph")
        self._dock("Log", self.logp, Qt.DockWidgetArea.BottomDockWidgetArea, tab_with="Node Graph")
        self._dock("Python", self.console, Qt.DockWidgetArea.BottomDockWidgetArea, tab_with="Node Graph")
        self._dock("Properties", self.props, Qt.DockWidgetArea.RightDockWidgetArea)
        self._dock("Layers", self.layers, Qt.DockWidgetArea.RightDockWidgetArea, tab_with="Properties")
        self._dock("Scopes", self.scopes, Qt.DockWidgetArea.RightDockWidgetArea, tab_with="Properties")
        self._dock("Profiler", self.profiler, Qt.DockWidgetArea.RightDockWidgetArea, tab_with="Properties")
        self.docks["Node Graph"].raise_()
        self.docks["Properties"].raise_()
        self.resizeDocks([self.docks["Node Graph"]], [300], Qt.Orientation.Vertical)
        self.resizeDocks([self.docks["Properties"]], [380], Qt.Orientation.Horizontal)
        self.settings_win = None
        self.jobs = []
        self.prompt_on_close = True  # tests and the smoke run switch this off

        self.status = QLabel("")
        self.status.setFont(theme.mono_font(8))
        self.render_bar = QProgressBar()
        self.render_bar.setFixedWidth(180)
        self.render_bar.setVisible(False)
        self.render_cancel = QPushButton("Cancel")
        self.render_cancel.setVisible(False)
        self.statusBar().addWidget(self.status, 1)
        self.statusBar().addPermanentWidget(self.render_bar)
        self.statusBar().addPermanentWidget(self.render_cancel)

        from vertexcomp.ai.nodes import controller as ai_controller
        self.ai = ai_controller(self.s)
        self._menus()
        self.s.selectionChanged.connect(self._selection)
        self.viewer.transformToolToggled.connect(lambda _: self._selection())
        self.s.rotoSelectionChanged.connect(self._roto_sel)
        self.s.logMessage.connect(self._status_message)
        self.s.undo.indexChanged.connect(self._undo_index)
        self.s.graphChanged.connect(self._title)
        self.props.renderRequested.connect(self.render_write)
        self.props.curveRequested.connect(self._show_curve)
        self.setAcceptDrops(True)
        self.resize(1600, 1000)
        self._title()
        self._restore_layout()

        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self.autosave)
        self._apply_autosave()
        settings.on_change(self._setting_changed)
        self._run_plugins()

    # ------------------------------------------------------------ layout
    def _dock(self, title, widget, area, tab_with=None):
        d = QDockWidget(title, self)
        d.setObjectName(title.replace(" ", ""))
        d.setWidget(widget)
        d.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable |
                      QDockWidget.DockWidgetFeature.DockWidgetClosable)
        if tab_with:
            self.tabifyDockWidget(self.docks[tab_with], d)
        else:
            self.addDockWidget(area, d)
        self.docks[title] = d
        return d

    def _layout_path(self):
        return config_dir() / "layout.bin"

    def save_layout(self):
        self._layout_path().write_bytes(bytes(self.saveState().data()) + b"||" + bytes(self.saveGeometry().data()))

    def _restore_layout(self):
        try:
            state, geo = self._layout_path().read_bytes().split(b"||", 1)
            self.restoreGeometry(QByteArray(geo))
            self.restoreState(QByteArray(state))
        except (OSError, ValueError):
            pass

    def reset_layout(self):
        try:
            self._layout_path().unlink()
        except OSError:
            pass
        self.s.log("layout reset; restart to apply", "warning")

    # ------------------------------------------------------------ menus
    def _menus(self):
        mb = self.menuBar()

        def act(menu, text, fn, seq=None, tip=""):
            a = QAction(text, self)
            if seq:
                a.setShortcut(QKeySequence(seq))
            a.setToolTip(tip)
            a.triggered.connect(fn)
            menu.addAction(a)
            return a

        f = mb.addMenu("File")
        act(f, "New", self.new_project, QKeySequence.StandardKey.New)
        act(f, "Open...", self.open_dialog, QKeySequence.StandardKey.Open)
        self.recent_menu = f.addMenu("Open Recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        act(f, "Save", self.save, QKeySequence.StandardKey.Save)
        act(f, "Save As...", self.save_as, "Ctrl+Shift+S")
        act(f, "Save As (relative paths)...", lambda: self.save_as(relative=True))
        f.addSeparator()
        act(f, "Import Read...", self.import_read)
        act(f, "Reconnect Media...", self.reconnect_media)
        f.addSeparator()
        act(f, "Quit", self.close, QKeySequence.StandardKey.Quit)

        e = mb.addMenu("Edit")
        u = self.s.undo.createUndoAction(self, "Undo")
        u.setShortcut(QKeySequence.StandardKey.Undo)
        r = self.s.undo.createRedoAction(self, "Redo")
        r.setShortcuts([QKeySequence.StandardKey.Redo, QKeySequence("Ctrl+Y")])
        e.addAction(u)
        e.addAction(r)
        e.addSeparator()
        act(e, "Copy", self.graph.copy)
        act(e, "Paste", self.graph.paste)
        act(e, "Duplicate", self.graph.duplicate)
        act(e, "Delete", self.graph.delete_selected)
        e.addSeparator()
        act(e, "Project Settings...", self.project_settings)
        act(e, "Settings...", self.open_settings, "Ctrl+,")

        v = mb.addMenu("View")
        act(v, "Fit Viewer", self.viewer.gl.fit)
        act(v, "Viewer 100%", lambda: self.viewer.gl.one_to_one(1.0), "Ctrl+1")
        act(v, "Viewer 200%", lambda: self.viewer.gl.one_to_one(2.0), "Ctrl+2")
        act(v, "Fit Node Graph", self.graph.fit)
        v.addSeparator()
        act(v, "Save Layout", self.save_layout)
        act(v, "Reset Layout", self.reset_layout)

        nm = mb.addMenu("Nodes")
        cats = {}
        for t, o in sorted(OP_TYPES.items()):
            cats.setdefault(o["category"], []).append(t)
        for cat in sorted(cats):
            sub = nm.addMenu(cat)
            for t in cats[cat]:
                sub.addAction(t, lambda t=t: self.s.create_node(t))
        nm.addSeparator()
        act(nm, "Backdrop", self.graph.add_backdrop)
        act(nm, "Sticky Note", lambda: self.graph.add_backdrop(note=True))

        rm = mb.addMenu("Render")
        act(rm, "Render Selected Write", self._render_selected, "F7")
        act(rm, "Render All Writes", self._render_all, "Shift+F7")

        am = mb.addMenu("AI")
        act(am, "Generate Selected", self._ai_generate_selected, "Ctrl+Return",
            "Run Generate on the selected AI nodes / add a version on selected Versions nodes")
        am.addSeparator()
        act(am, "Models && Settings...", self.open_ai_settings)
        act(am, "Unload Models from RAM", self._ai_unload)
        act(am, "Clear Caches", self._ai_clear)

        w = mb.addMenu("Window")
        for _title, d in self.docks.items():
            w.addAction(d.toggleViewAction())

        h = mb.addMenu("Help")
        act(h, "Keyboard Shortcuts", self.show_shortcuts)
        act(h, "About", lambda: QMessageBox.about(
            self, APP_NAME, f"{APP_NAME} {VERSION}\nC++ core: {len(OP_TYPES)} node types, "
                            f"{vc.thread_count()} threads\nPySide6 UI"))

    def _fill_recent(self):
        self.recent_menu.clear()
        for p in self.settings.get("paths", "recent_projects"):
            self.recent_menu.addAction(p, lambda p=p: self.open(p))

    # ------------------------------------------------------------ project
    def _status_message(self, level, text):
        if level != "info":
            self.status.setText(text)

    def _undo_index(self, _index):
        self._title()

    def _title(self):
        name = os.path.basename(self.s.project.path) if self.s.project.path else "untitled"
        mod = "*" if self.s.dirty else ""
        self.setWindowTitle(f"{name}{mod} - {APP_NAME}")

    def _confirm_discard(self) -> bool:
        if not self.prompt_on_close:
            return True
        if not self.s.dirty or not self.s.project.nodes():
            return True
        r = QMessageBox.question(self, APP_NAME, "Save changes to the current project?",
                                 QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard |
                                 QMessageBox.StandardButton.Cancel)
        if r == QMessageBox.StandardButton.Save:
            return self.save()
        return r == QMessageBox.StandardButton.Discard

    def new_project(self):
        if not self._confirm_discard():
            return
        self.s.project.clear()
        self.s.apply_project_settings()
        self.s.viewer_slots.clear()
        self.s.undo.clear()
        self.s.dirty = False
        self.s.selected = []
        self.s.graphChanged.emit()
        self.s.rangeChanged.emit()
        self.s.viewedChanged.emit()
        self._title()

    def open_dialog(self):
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "Vertex Comp project (*.vcp);;All files (*)")
        if path:
            self.open(path)

    def open(self, path, force=False):
        p = self.s.project
        try:
            p.load(path, allow_newer=force)
        except SchemaError as e:
            r = QMessageBox.warning(self, APP_NAME, f"{e}.\nOpen anyway? Unknown data will be ignored.",
                                    QMessageBox.StandardButton.Open | QMessageBox.StandardButton.Cancel)
            if r == QMessageBox.StandardButton.Open:
                return self.open(path, force=True)
            return False
        except (OSError, ValueError) as e:
            QMessageBox.critical(self, APP_NAME, f"Cannot open {path}:\n{e}")
            return False
        st = p.settings
        vw = p.viewer or {}
        self.s.viewer_slots = {int(k): v for k, v in vw.get("slots", {}).items()}
        self.s.active_slot = int(vw.get("active", 1))
        self.s.frame = int(vw.get("frame", st.first))
        self.s.undo.clear()
        self.s.dirty = False
        self.s.selected = []
        self.viewer.transport.fps.setCurrentText(f"{st.fps:g}")
        self.viewer.gl.fitted_once = False
        self.viewer.last_info = None
        self.s.graphChanged.emit()
        self.s.rangeChanged.emit()
        self.s.viewedChanged.emit()
        self.s.frameChanged.emit(self.s.frame)
        self.graph.fit()
        self.settings.add_recent(path)
        self.settings.save()
        self._title()
        missing = [n.name for n in p.nodes() if n.type == "Read" and n.get("file") and not self._media_ok(n)]
        if missing:
            self.s.log(f"offline media in {', '.join(missing)}; use File > Reconnect Media", "warning")
        self.s.log(f"opened {path}")
        return True

    def _media_ok(self, n):
        f = n.get("file")
        if vc.is_pattern(f):
            return os.path.exists(vc.frame_path(f, int(n.get("first")[0])))
        return os.path.exists(f)

    def _store_viewer_state(self):
        self.s.project.viewer = {"slots": {str(k): v for k, v in self.s.viewer_slots.items()},
                                 "active": self.s.active_slot, "frame": self.s.frame,
                                 "exposure": self.viewer.gl.exposure, "gamma": self.viewer.gl.gamma}

    def save(self) -> bool:
        if not self.s.project.path:
            return self.save_as()
        self._store_viewer_state()
        try:
            self.s.project.save(self.s.project.path)
        except OSError as e:
            QMessageBox.critical(self, APP_NAME, f"Cannot save:\n{e}")
            return False
        self.s.dirty = False
        self.settings.add_recent(self.s.project.path)
        self.settings.save()
        self._title()
        self.s.log(f"saved {self.s.project.path}")
        return True

    def save_as(self, relative=False) -> bool:
        path, _ = QFileDialog.getSaveFileName(self, "Save project", self.s.project.path or "untitled.vcp",
                                              "Vertex Comp project (*.vcp)")
        if not path:
            return False
        if not path.endswith(".vcp"):
            path += ".vcp"
        self._store_viewer_state()
        self.s.project.save(path, relative=relative)
        self.s.project.path = path
        return self.save()

    def project_settings(self):
        st = self.s.project.settings
        text, ok = QInputDialog.getText(self, "Project settings", "format (w x h), fps, first, last",
                                        text=f"{st.format_w}x{st.format_h}, {st.fps:g}, {st.first}, {st.last}")
        if not ok:
            return
        try:
            fmt, fps, first, last = [x.strip() for x in text.split(",")]
            w, h = fmt.lower().split("x")
            st.format_w, st.format_h, st.fps = int(w), int(h), float(fps)
            self.s.set_range(int(first), int(last))
            self.viewer.transport.fps.setCurrentText(f"{st.fps:g}")
        except ValueError:
            self.s.log("project settings: expected e.g. 1920x1080, 24, 1, 100", "warning")

    def import_read(self):
        d = self.settings.get("paths", "read_dir") or os.path.expanduser("~")
        paths, _ = QFileDialog.getOpenFileNames(self, "Read", d)
        for p in paths:
            self._make_read(p)

    def _make_read(self, path, x=None, y=None):
        n = self.s.create_node("Read", x, y, connect_selected=False)
        set_read_file(self.s, n, path)
        self.s.select([n])
        self.s.view_node(n)
        return n

    def reconnect_media(self):
        old, ok = QInputDialog.getText(self, "Reconnect media", "Replace path prefix:")
        if not ok or not old:
            return
        new, ok = QInputDialog.getText(self, "Reconnect media", f"Replace '{old}' with:")
        if not ok:
            return
        n = self.s.structural("reconnect media", lambda: self.s.project.remap_paths(old, new))
        self.s.log(f"reconnected {n} Read nodes")

    # ------------------------------------------------------------ autosave / recovery
    def _autosave_dir(self):
        d = config_dir() / "autosave"
        d.mkdir(exist_ok=True)
        return d

    def _apply_autosave(self):
        mins = float(self.settings.get("general", "autosave_minutes"))
        if mins > 0:
            self.autosave_timer.start(int(mins * 60000))
        else:
            self.autosave_timer.stop()

    def autosave(self):
        if not self.s.dirty or not self.s.project.nodes():
            return
        base = os.path.splitext(os.path.basename(self.s.project.path or "untitled.vcp"))[0]
        d = self._autosave_dir()
        keep = int(self.settings.get("general", "autosave_backups"))
        target = d / f"{base}.autosave.vcp"
        for i in range(keep - 1, 0, -1):
            src = d / (f"{base}.autosave.vcp" if i == 1 else f"{base}.autosave{i - 1}.vcp")
            dst = d / f"{base}.autosave{i}.vcp"
            if src.exists():
                shutil.copyfile(src, dst)
        self._store_viewer_state()
        path_keep = self.s.project.path
        try:
            self.s.project.save(str(target))
        finally:
            self.s.project.path = path_keep
        (d / "last.txt").write_text(str(target), encoding="utf-8")

    def check_recovery(self):
        lock = config_dir() / "session.lock"
        crashed = lock.exists()
        lock.write_text(str(os.getpid()), encoding="utf-8")
        last = self._autosave_dir() / "last.txt"
        if crashed and last.exists():
            path = last.read_text(encoding="utf-8").strip()
            if os.path.exists(path):
                r = QMessageBox.question(self, APP_NAME, f"The last session did not exit cleanly.\nRecover {path}?")
                if r == QMessageBox.StandardButton.Yes:
                    self.open(path)
                    self.s.project.path = ""
                    self.s.dirty = True
                    self._title()

    # ------------------------------------------------------------ selection / roto
    def _selection(self):
        last = self.s.selected[-1] if self.s.selected else None
        if last is not None and last.type == "Roto":
            if self.s.roto_tool is None or self.s.roto_tool.node_name != last.name:
                self.s.roto_state.shape = -1
                self.s.roto_state.points = set()
                self.s.roto_tool = RotoTool(self.s, self.viewer.gl, last.name)
            tool = self.s.roto_tool
        elif last is not None and last.type == "Paint":
            tool = PaintTool(self.s, self.viewer.gl, last.name)
        elif last is not None and last.type == "Transform" and self.viewer.transform_btn.isChecked():
            tool = TransformTool(self.s, self.viewer.gl, last.name)
        else:
            self.s.roto_tool = None
            tool = None
        self.viewer.gl.tool = tool
        self.viewer.gl.update()

    def _roto_sel(self):
        self.viewer.gl.update()
        for ed in self.props.editors.values():
            if hasattr(ed, "roto"):
                ed.roto.refresh()

    def _show_curve(self, node, knob):
        self.docks["Curve Editor"].show()
        self.docks["Curve Editor"].raise_()
        self.curves.show_knob(node, knob)

    # ------------------------------------------------------------ render
    def render_write(self, node, first=None, last=None):
        first = int(node.get("first")[0]) if first is None else first
        last = int(node.get("last")[0]) if last is None else last
        job = RenderJob(self.s, node, first, last)
        job.progress.connect(self._render_progress)
        job.log.connect(lambda lvl, t: self.s.log(t, lvl))
        job.finished.connect(lambda msg, j=job: self._render_done(j, msg))
        self.render_cancel.clicked.connect(job.cancel)
        self.render_bar.setRange(0, last - first + 1)
        self.render_bar.setValue(0)
        self.render_bar.setFormat(f"{node.name} %v/%m")
        self.render_bar.setVisible(True)
        self.render_cancel.setVisible(True)
        self.jobs.append(job)
        self.s.log(f"{node.name}: rendering {first}-{last}")
        job.start()

    def _render_progress(self, done, total, frame):
        self.render_bar.setValue(done)

    def _render_done(self, job, msg):
        self.s.log(msg)
        self.status.setText(msg)
        if job in self.jobs:
            self.jobs.remove(job)
        try:
            self.render_cancel.clicked.disconnect(job.cancel)
        except (RuntimeError, TypeError):
            pass
        if not self.jobs:
            self.render_bar.setVisible(False)
            self.render_cancel.setVisible(False)

    def _render_selected(self):
        ws = [n for n in self.s.selected if n.type == "Write"]
        if not ws:
            self.s.log("select a Write node to render", "warning")
        for w in ws:
            self.render_write(w)

    def _render_all(self):
        for n in self.s.project.nodes():
            if n.type == "Write" and not n.disabled:
                self.render_write(n)

    # ------------------------------------------------------------ AI
    def _ai_generate_selected(self):
        from vertexcomp.ai import GEN_TYPES
        from vertexcomp.ai.nodes import add_version, generate
        for n in list(self.s.selected):
            if n.type in GEN_TYPES:
                generate(self.s, n)
            elif n.type == "Versions":
                add_version(self.s, n)

    def open_ai_settings(self):
        self.open_settings()
        from vertexcomp.ui.settings_window import SECTIONS
        self.settings_win.list.setCurrentRow([s for _, s in SECTIONS].index("ai"))

    def _ai_unload(self):
        self.ai.mgr.unload()
        self.s.log("AI models unloaded from RAM/VRAM")

    def _ai_clear(self):
        self.s.log(self.ai.mgr.clear_caches())

    # ------------------------------------------------------------ settings
    def open_settings(self):
        if self.settings_win is None:
            self.settings_win = SettingsWindow(self.settings, self)
        self.settings_win.show()
        self.settings_win.raise_()

    def _setting_changed(self, sec, key, value):
        if sec == "performance" and key == "cache_gb":
            self.s.project.engine.set_cache_limit(int(float(value) * 1024))
        elif sec == "performance" and key == "threads":
            vc.set_threads(int(value))
        elif sec == "general" and key == "autosave_minutes":
            self._apply_autosave()
        elif sec == "general" and key == "undo_depth":
            self.s.undo.setUndoLimit(int(value)) if self.s.undo.count() == 0 else None
        elif sec == "color" and key == "working_space":
            vc.set_working_space(value)
            self.s.project.settings.working_space = value
            self.s.project.engine.clear_cache()
            self.viewer._set_display()
            self.viewer.request()
        elif sec == "appearance" and key == "theme":
            theme.apply(QApplication.instance(), value, float(self.settings.get("appearance", "font_size")))
        elif sec == "viewer" and key == "background":
            self.viewer.gl.bg = {"black": 0, "gray": 1, "checker": 2}.get(value, 2)
        elif sec == "nodegraph":
            self.graph.rebuild()
        self.viewer.gl.update()

    def show_shortcuts(self):
        hk = self.settings.data["hotkeys"]
        lines = [f"{a:<16} {k}" for a, k in sorted(hk.items())]
        lines += ["", "Node graph: Tab create, 1-9 view in slot, Ctrl+click view, drag from output to empty space",
                  "  to create and connect, drag an arrow end to re-wire, Ctrl+B backdrop, L align",
                  "Viewer: wheel zoom, middle/Alt drag pan, F fit, Ctrl+drag area sample, [ ] exposure",
                  "Roto: Ctrl+click curve inserts point, Delete removes, arrows nudge (Shift x10)"]
        m = QMessageBox(self)
        m.setWindowTitle("Keyboard shortcuts")
        m.setText("<pre>" + "\n".join(lines) + "</pre>")
        m.exec()

    def _run_plugins(self):
        for d in self.settings.get("paths", "plugin_dirs"):
            for f in sorted(glob.glob(os.path.join(d, "*.py"))):
                try:
                    with open(f, encoding="utf-8") as fh:
                        exec(compile(fh.read(), f, "exec"), dict(self.console.ns))
                    self.s.log(f"plugin {f}")
                except Exception:
                    self.s.log(f"plugin {f} failed:\n{traceback.format_exc()}", "error")

    # ------------------------------------------------------------ drag & drop, close
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if p.endswith(".vcp"):
                if self._confirm_discard():
                    self.open(p)
                return
            self._make_read(p)

    def closeEvent(self, e):
        if not self._confirm_discard():
            e.ignore()
            return
        for j in self.jobs:
            j.cancel()
        self.ai.mgr.stop(timeout=3.0)
        self.save_layout()
        self.settings.save()
        self.viewer.shutdown()
        try:
            (config_dir() / "session.lock").unlink()
        except OSError:
            pass
        e.accept()
