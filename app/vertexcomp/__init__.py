"""Vertex Comp: node-based compositor. C++ core (_vccore) + PySide6 UI."""

APP_NAME = "Vertex Comp"
APP_ID = "vertexcomp"
VERSION = "0.1.0"
PROJECT_SCHEMA = 2

import os as _os  # noqa: E402

# Must be set before PySide6 is imported anywhere (see main.py).
_os.environ.setdefault("PYSIDE6_OPTION_LAZY", "0")
