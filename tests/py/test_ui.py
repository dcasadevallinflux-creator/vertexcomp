"""UI smoke tests (pytest-qt). Need a display with OpenGL 3.3 (on Linux CI: xvfb)."""
import os
import sys

import numpy as np
import pytest

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    pytest.skip("no display (run under xvfb-run)", allow_module_level=True)

pytest.importorskip("pytestqt")

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtCore import QEvent, QPointF  # noqa: E402
from PySide6.QtGui import QMouseEvent, QSurfaceFormat  # noqa: E402
from vertexcomp import _vccore as vc  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def gl_format():
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)


@pytest.fixture
def win(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "cfg"))
    from PySide6.QtWidgets import QApplication
    from vertexcomp.settings import Settings
    from vertexcomp.ui import theme
    from vertexcomp.ui.mainwindow import MainWindow
    theme.apply(QApplication.instance())
    w = MainWindow(Settings(tmp_path / "settings.json"))
    w.prompt_on_close = False
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    yield w
    w.s.dirty = False
    w.viewer.shutdown()


def wait_image(qtbot, w, timeout=10000):
    qtbot.waitUntil(lambda: w.viewer.gl.layers["A"].array is not None, timeout=timeout)


def test_build_view_undo(win, qtbot):
    s = win.s
    c = s.create_node("CheckerBoard", 0, 0, connect_selected=False)
    g = s.create_node("Grade")
    assert g.inputs[0].name == c.name
    s.view_node(g, 1)
    wait_image(qtbot, win)
    s.set_knob(g, "gain", [2.0, 2.0, 2.0, 1.0])
    assert g.get("gain")[0] == 2.0
    s.undo.undo()
    assert s.project.find("Grade1").get("gain")[0] == 1.0
    s.undo.undo()  # create Grade
    assert s.project.find("Grade1") is None
    s.undo.redo()
    assert s.project.find("Grade1") is not None


def test_viewer_zoom_values_and_channels(win, qtbot):
    s = win.s
    c = s.create_node("Constant", connect_selected=False)
    s.set_knob(c, "color", [0.25, 0.5, 0.75, 1.0])
    s.set_knob(c, "format", [64, 32])
    s.view_node(c, 1)
    wait_image(qtbot, win)
    gl = win.viewer.gl
    gl.cx, gl.cy = 10.5, 10.5
    gl.set_zoom(120.0)
    win.viewer.request()
    qtbot.waitUntil(lambda: gl.layers["A"].step == 1 and gl.layers["A"].array is not None)
    v = gl.sample_at(10.5, 10.5)
    assert v is not None and abs(v[1] - 0.5) < 1e-6
    gl.set_zoom(50000.0)
    assert gl.zoom == 50000.0
    win.viewer.mode_box.setCurrentIndex(5)  # alpha
    assert gl.mode == 5
    img = gl.grabFramebuffer()
    assert img.width() > 0


def test_aov_layers_listed(win, qtbot, tmp_path):
    path = tmp_path / "aov.exr"
    chans = ["rgba.red", "rgba.green", "rgba.blue", "rgba.alpha", "diffuse.red", "diffuse.green", "diffuse.blue",
             "depth.Z"]
    vc.write_image(str(path), np.random.rand(len(chans), 16, 16).astype(np.float32), chans, (0, 0, 16, 16), (16, 16))
    win._make_read(str(path))
    wait_image(qtbot, win)
    qtbot.waitUntil(lambda: win.viewer.layer_box.count() >= 3)
    items = [win.viewer.layer_box.itemText(i) for i in range(win.viewer.layer_box.count())]
    assert {"rgba", "diffuse", "depth"} <= set(items)
    win.viewer.set_layer("depth")
    assert win.viewer.slots()[0] == "depth.Z"


def test_save_load_roundtrip(win, qtbot, tmp_path):
    s = win.s
    s.create_node("Noise", connect_selected=False)
    b = s.create_node("Blur")
    s.set_key(b, "size", True)
    ro = s.create_node("Roto", connect_selected=False)
    sh = vc.RotoShape()
    sh.keys = {1.0: [vc.RotoPoint(0, 0), vc.RotoPoint(10, 0), vc.RotoPoint(5, 8)]}
    s.set_roto(ro, [sh])
    path = str(tmp_path / "p.vcp")
    s.project.path = path
    assert win.save()
    before = s.project.to_dict()
    assert win.open(path)
    assert s.project.to_dict()["nodes"] == before["nodes"]


def test_roto_draw_tool(win, qtbot):
    s = win.s
    s.create_node("Constant", connect_selected=False)
    ro = s.create_node("Roto")
    s.view_node(ro, 1)
    s.select([ro])
    wait_image(qtbot, win)
    gl = win.viewer.gl
    assert gl.tool is not None
    s.roto_state.tool = "bezier"
    pts = [QPoint(200, 150), QPoint(400, 150), QPoint(300, 320)]
    for p in pts:
        qtbot.mouseClick(gl, Qt.MouseButton.LeftButton, pos=p)
    qtbot.mouseClick(gl, Qt.MouseButton.LeftButton, pos=pts[0])  # close on first point
    shapes = ro.roto_shapes()
    assert len(shapes) == 1 and shapes[0].closed and len(shapes[0].points_at(1.0)) == 3
    s.undo.undo()


def test_transform_viewer_gizmo_moves_node(win):
    node = win.s.create_node("Transform", connect_selected=False)
    win.s.set_knob(node, "center", [960, 540])
    win.s.select([node])
    win.viewer.transform_btn.setChecked(True)
    tool = win.viewer.gl.tool
    assert tool is not None

    gl = win.viewer.gl
    x, y = gl.to_screen(960, 540)
    start = QPointF(x, y)
    end = QPointF(x + 20, y + 10)
    press = QMouseEvent(QEvent.Type.MouseButtonPress, start, start, Qt.MouseButton.LeftButton,
                        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    move = QMouseEvent(QEvent.Type.MouseMove, end, end, Qt.MouseButton.NoButton,
                       Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    release = QMouseEvent(QEvent.Type.MouseButtonRelease, end, end, Qt.MouseButton.LeftButton,
                          Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    assert tool.press(press)
    assert tool.move(move)
    assert tool.release(release)
    assert node.get("translate", win.s.frame) == [40, 20]


def test_visual_subgroup_collapses_and_persists(win):
    a = win.s.create_node("Constant", connect_selected=False)
    b = win.s.create_node("Grade", connect_selected=False)
    win.s.select([a, b])
    win.graph.add_subgroup()
    group = win.s.project.backdrops[-1]
    assert group.group and set(group.members) == {a.name, b.name}

    win.graph._toggle_subgroup(group)
    assert group.collapsed
    assert not win.graph.node_items[a.name].isVisible()
    assert not win.graph.node_items[b.name].isVisible()

    backdrop = win.s.project.to_dict()["backdrops"][-1]
    assert backdrop["group"] and backdrop["collapsed"]
    assert set(backdrop["members"]) == {a.name, b.name}


def test_tab_creator_and_graph_hotkeys(win, qtbot):
    g = win.graph
    g.setFocus()
    g.creator.popup_at(g.mapToGlobal(QPoint(50, 50)))
    qtbot.keyClicks(g.creator.edit, "gau")
    assert g.creator.list.currentItem().data(Qt.ItemDataRole.UserRole) == "GaussianBlur"
    qtbot.keyClick(g.creator.edit, Qt.Key.Key_Return)
    assert any(n.type == "GaussianBlur" for n in win.s.project.nodes())


def test_settings_window_and_render(win, qtbot, tmp_path):
    win.open_settings()
    qtbot.waitExposed(win.settings_win)
    win.settings_win.search.setText("zebra")
    win.settings_win.close()
    s = win.s
    r = s.create_node("Ramp", connect_selected=False)
    s.set_knob(r, "format", [32, 16])
    w = s.create_node("Write")
    s.set_knob(w, "file", str(tmp_path / "out.####.exr"))
    s.set_knob(w, "first", 1)
    s.set_knob(w, "last", 3)
    win.render_write(w)
    qtbot.waitUntil(lambda: not win.jobs, timeout=20000)
    assert all((tmp_path / f"out.000{i}.exr").exists() for i in (1, 2, 3))


def test_playback_uses_cache(win, qtbot):
    s = win.s
    c = s.create_node("Noise", connect_selected=False)
    s.set_knob(c, "format", [128, 64])
    s.view_node(c, 1)
    s.set_range(1, 10)
    wait_image(qtbot, win)
    win.viewer.transport.play(1)
    qtbot.wait(1500)
    win.viewer.transport.stop()
    assert s.frame != 1 or len(win.viewer.cache.frames) > 1
