"""--smoke-test: drives the real UI through a typical session and saves a screenshot."""
from __future__ import annotations

import os
import tempfile
import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from vertexcomp import _vccore as vc


def _wait(cond, timeout=20.0):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        QApplication.processEvents()
        time.sleep(0.01)
    return cond()


def run_smoke(win, png):
    s = win.s
    ok = True
    try:
        s.create_node("CheckerBoard", 0, 0, connect_selected=False)
        n = s.create_node("Noise", 0, 70)
        n.set("mode", "add")
        g = s.create_node("Grade", 0, 140)
        s.set_knob(g, "gain", [1.4, 1.1, 0.8, 1.0])
        ro = s.create_node("Roto", 200, 70, connect_selected=False)
        sh = vc.RotoShape()
        sh.keys = {1.0: [vc.RotoPoint(500, 300, -100, 0, 100, 0), vc.RotoPoint(1400, 300), vc.RotoPoint(1200, 800),
                         vc.RotoPoint(600, 820)]}
        sh.feather = 25
        s.set_roto(ro, [sh])
        m = s.create_node("Merge", 0, 220, connect_selected=False)
        s.connect(m, 0, g)
        s.connect(m, 1, ro)
        s.view_node(m, 1)
        s.select([ro])
        win.graph.fit()
        ok &= _wait(lambda: win.viewer.gl.layers["A"].array is not None)
        win.viewer.gl.fit()
        _wait(lambda: False, 1.0)
        s.undo.undo()
        s.undo.redo()
        win.viewer.transport.play(1)
        _wait(lambda: False, 1.5)
        win.viewer.transport.stop()
        win.open_settings()
        win.settings_win.close()
        path = os.path.join(tempfile.gettempdir(), "vc_smoke.vcp")
        s.project.path = path
        ok &= win.save()
        ok &= win.open(path)
        ok &= len(s.project.nodes()) == 5
        _wait(lambda: win.viewer.gl.layers["A"].array is not None)
        _wait(lambda: False, 1.0)
        win.grab().save(png)
        print(f"smoke test {'passed' if ok else 'FAILED'}; screenshot {png}")
    except Exception:
        import traceback
        traceback.print_exc()
        ok = False
    s.dirty = False
    win.prompt_on_close = False
    QTimer.singleShot(0, lambda: QApplication.instance().exit(0 if ok else 1))
