"""Application entry point."""
from __future__ import annotations

import argparse
import os
import sys
import traceback


def main(argv=None) -> int:
    # PySide6 6.12 lazily creates enum types inside paint events; a garbage
    # collection during that window traverses a half-built dict and crashes.
    # Eager type loading avoids it (see docs/decisions.md).
    os.environ.setdefault("PYSIDE6_OPTION_LAZY", "0")
    ap = argparse.ArgumentParser(prog="vertexcomp")
    ap.add_argument("project", nargs="?", help=".vcp project or an image/movie to open in a Read")
    ap.add_argument("--smoke-test", metavar="PNG", help="build a demo graph, render, save a screenshot and exit")
    args, qt_args = ap.parse_known_args(argv)

    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QSurfaceFormat
    from PySide6.QtWidgets import QApplication

    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(fmt)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    app = QApplication([sys.argv[0]] + qt_args)
    from vertexcomp import APP_NAME, movie
    from vertexcomp import _vccore as vc
    from vertexcomp.settings import Settings, config_dir
    from vertexcomp.ui import theme

    app.setApplicationName(APP_NAME)
    settings = Settings()
    theme.apply(app, settings.get("appearance", "theme"), float(settings.get("appearance", "font_size")),
                float(settings.get("appearance", "ui_scale")))
    vc.set_threads(int(settings.get("performance", "threads")))
    if not movie.install(vc):
        print("imageio-ffmpeg not found: movie files are disabled", file=sys.stderr)

    from vertexcomp.ui.mainwindow import MainWindow

    win = MainWindow(settings)

    def excepthook(t, v, tb):
        text = "".join(traceback.format_exception(t, v, tb))
        try:
            with open(config_dir() / "error.log", "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except OSError:
            pass
        win.s.log(text.strip().splitlines()[-1], "error")
        print(text, file=sys.stderr)

    sys.excepthook = excepthook
    win.show()

    if args.smoke_test:
        from vertexcomp.smoke import run_smoke
        QTimer.singleShot(200, lambda: run_smoke(win, args.smoke_test))
    else:
        QTimer.singleShot(0, win.check_recovery)
        if args.project:
            if args.project.endswith(".vcp"):
                QTimer.singleShot(0, lambda: win.open(os.path.abspath(args.project)))
            else:
                QTimer.singleShot(0, lambda: win._make_read(os.path.abspath(args.project)))
    return app.exec()
