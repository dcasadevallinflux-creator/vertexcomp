"""Neutral dark theme, fonts and monochrome icons."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer

RES = Path(__file__).resolve().parent.parent / "resources"

THEMES = {
    "dark": {
        "bg": "#2b2b2b", "base": "#1f1f1f", "alt": "#262626", "panel": "#323232", "line": "#191919",
        "text": "#c8c8c8", "dim": "#8a8a8a", "accent": "#d18b2c", "accent_text": "#111111", "error": "#c8463c",
        "field": "#1b1b1b", "button": "#3a3a3a", "hover": "#444444",
    },
    "graphite": {
        "bg": "#333333", "base": "#272727", "alt": "#2d2d2d", "panel": "#3a3a3a", "line": "#202020",
        "text": "#d2d2d2", "dim": "#929292", "accent": "#5d9bd5", "accent_text": "#0d0d0d", "error": "#cc5040",
        "field": "#222222", "button": "#424242", "hover": "#4c4c4c",
    },
}

T = dict(THEMES["dark"])
MONO = "monospace"
UI = "sans-serif"


def load_fonts():
    global MONO, UI
    fams = []
    for f in (RES / "fonts").glob("*.ttf"):
        i = QFontDatabase.addApplicationFont(str(f))
        if i >= 0:
            fams += QFontDatabase.applicationFontFamilies(i)
    mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    MONO = mono.family()
    for cand in ("Inter", "Source Sans 3", "Segoe UI", "Noto Sans", "DejaVu Sans"):
        if cand in fams or cand in QFontDatabase.families():
            UI = cand
            break


def mono_font(size=9) -> QFont:
    f = QFont(MONO)
    f.setStyleHint(QFont.StyleHint.Monospace)
    f.setPointSizeF(size)
    return f


def apply(app, name="dark", font_size=9, scale=1.0):
    T.clear()
    T.update(THEMES.get(name, THEMES["dark"]))
    load_fonts()
    f = QFont(UI)
    f.setPointSizeF(font_size * scale)
    app.setFont(f)
    pal = QPalette()
    c = QColor
    pal.setColor(QPalette.ColorRole.Window, c(T["bg"]))
    pal.setColor(QPalette.ColorRole.WindowText, c(T["text"]))
    pal.setColor(QPalette.ColorRole.Base, c(T["field"]))
    pal.setColor(QPalette.ColorRole.AlternateBase, c(T["alt"]))
    pal.setColor(QPalette.ColorRole.Text, c(T["text"]))
    pal.setColor(QPalette.ColorRole.Button, c(T["button"]))
    pal.setColor(QPalette.ColorRole.ButtonText, c(T["text"]))
    pal.setColor(QPalette.ColorRole.Highlight, c(T["accent"]))
    pal.setColor(QPalette.ColorRole.HighlightedText, c(T["accent_text"]))
    pal.setColor(QPalette.ColorRole.ToolTipBase, c(T["base"]))
    pal.setColor(QPalette.ColorRole.ToolTipText, c(T["text"]))
    pal.setColor(QPalette.ColorRole.PlaceholderText, c(T["dim"]))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, c(T["dim"]))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, c(T["dim"]))
    app.setPalette(pal)
    app.setStyleSheet(qss(font_size * scale))
    icon.cache_clear()


def _arrow_path() -> str:
    """Small down arrow for combo boxes, written next to the user config so the
    stylesheet can reference it by path."""
    from vertexcomp.settings import config_dir
    p = config_dir() / f"arrow_{T['dim'].lstrip('#')}.svg"
    if not p.exists():
        p.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="7" height="4" viewBox="0 0 7 4">'
                     f'<path d="M0 0h7L3.5 4z" fill="{T["dim"]}"/></svg>', encoding="utf-8")
    return p.as_posix()


def qss(fs: float) -> str:
    return f"""
* {{ outline: 0; }}
QMainWindow, QDialog {{ background: {T['bg']}; }}
QWidget {{ color: {T['text']}; }}
QToolTip {{ background: {T['base']}; color: {T['text']}; border: 1px solid {T['line']}; padding: 2px 4px; }}
QMenuBar {{ background: {T['bg']}; border-bottom: 1px solid {T['line']}; }}
QMenuBar::item {{ padding: 3px 8px; background: transparent; }}
QMenuBar::item:selected, QMenu::item:selected {{ background: {T['accent']}; color: {T['accent_text']}; }}
QMenu {{ background: {T['alt']}; border: 1px solid {T['line']}; padding: 2px 0; }}
QMenu::item {{ padding: 3px 18px 3px 18px; }}
QMenu::separator {{ height: 1px; background: {T['line']}; margin: 2px 0; }}
QDockWidget {{ titlebar-close-icon: none; }}
QDockWidget::title {{ background: {T['panel']}; padding: 2px 6px; border-bottom: 1px solid {T['line']}; text-align: left; }}
QMainWindow::separator {{ background: {T['line']}; width: 1px; height: 1px; }}
QTabBar::tab {{ background: {T['bg']}; padding: 3px 10px; border: 1px solid {T['line']}; border-bottom: none; margin-right: 1px; border-radius: 0; }}
QTabBar::tab:selected {{ background: {T['panel']}; }}
QTabWidget::pane {{ border: 1px solid {T['line']}; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit, QListWidget, QTreeWidget, QTableWidget {{
    background: {T['field']}; border: 1px solid {T['line']}; border-radius: 1px; padding: 1px 3px;
    selection-background-color: {T['accent']}; selection-color: {T['accent_text']}; }}
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus {{ border: 1px solid {T['accent']}; }}
QComboBox {{ padding: 1px 4px; min-height: 16px; }}
QComboBox::drop-down {{ border: none; width: 12px; }}
QComboBox::down-arrow {{ image: url({_arrow_path()}); width: 7px; height: 4px; margin-right: 4px; }}
QComboBox QAbstractItemView {{ background: {T['alt']}; border: 1px solid {T['line']}; }}
QPushButton, QToolButton {{ background: {T['button']}; border: 1px solid {T['line']}; border-radius: 1px; padding: 2px 8px; }}
QToolButton {{ padding: 1px; }}
QPushButton:hover, QToolButton:hover {{ background: {T['hover']}; }}
QPushButton:pressed, QToolButton:pressed, QToolButton:checked, QPushButton:checked {{ background: {T['accent']}; color: {T['accent_text']}; }}
QPushButton:disabled {{ color: {T['dim']}; }}
QCheckBox::indicator {{ width: 11px; height: 11px; border: 1px solid {T['line']}; background: {T['field']}; }}
QCheckBox::indicator:checked {{ background: {T['accent']}; }}
QSlider::groove:horizontal {{ height: 2px; background: {T['line']}; }}
QSlider::handle:horizontal {{ width: 7px; margin: -5px 0; background: {T['dim']}; border: 1px solid {T['line']}; }}
QSlider::handle:horizontal:hover {{ background: {T['text']}; }}
QScrollBar:vertical {{ background: {T['bg']}; width: 9px; margin: 0; }}
QScrollBar:horizontal {{ background: {T['bg']}; height: 9px; margin: 0; }}
QScrollBar::handle {{ background: {T['button']}; min-height: 20px; min-width: 20px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QHeaderView::section {{ background: {T['panel']}; border: none; border-right: 1px solid {T['line']}; padding: 2px 4px; }}
QStatusBar {{ background: {T['bg']}; border-top: 1px solid {T['line']}; }}
QProgressBar {{ background: {T['field']}; border: 1px solid {T['line']}; text-align: center; height: 12px; }}
QProgressBar::chunk {{ background: {T['accent']}; }}
QGroupBox {{ border: 1px solid {T['line']}; margin-top: 8px; padding-top: 4px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 4px; color: {T['dim']}; }}
#knobLabel {{ color: {T['dim']}; }}
#sectionHeader {{ background: {T['panel']}; border-top: 1px solid {T['line']}; border-bottom: 1px solid {T['line']}; padding: 1px 4px; color: {T['dim']}; }}
#nodeHeader {{ background: {T['panel']}; border-bottom: 1px solid {T['line']}; }}
#viewerBar, #transport {{ background: {T['bg']}; }}
"""


# Monochrome single-weight glyphs on a 16x16 grid.
_ICONS = {
    "play": '<path d="M5 3.5v9l7-4.5z" fill="C" stroke="none"/>',
    "play_back": '<path d="M11 3.5v9l-7-4.5z" fill="C" stroke="none"/>',
    "stop": '<rect x="4.5" y="4.5" width="7" height="7" fill="C" stroke="none"/>',
    "step_fwd": '<path d="M4 4v8l5.5-4z" fill="C" stroke="none"/><path d="M11.5 4v8"/>',
    "step_back": '<path d="M12 4v8l-5.5-4z" fill="C" stroke="none"/><path d="M4.5 4v8"/>',
    "to_start": '<path d="M4 3.5v9M13 4v8l-7-4z"/>',
    "to_end": '<path d="M12 3.5v9M3 4v8l7-4z"/>',
    "loop": '<path d="M3.5 9V7.5A2.5 2.5 0 0 1 6 5h7M10.5 2.5L13 5l-2.5 2.5M12.5 7v1.5A2.5 2.5 0 0 1 10 11H3M5.5 13.5L3 11l2.5-2.5"/>',
    "key": '<path d="M8 3l5 5-5 5-5-5z"/>',
    "key_on": '<path d="M8 3l5 5-5 5-5-5z" fill="C"/>',
    "fit": '<path d="M3 6V3h3M10 3h3v3M13 10v3h-3M6 13H3v-3"/>',
    "select": '<path d="M4 2.5v10l3-3 2 4.5 1.6-.7-2-4.4H13z"/>',
    "bezier": '<path d="M3 12C4 6 11 10 13 4"/><circle cx="3" cy="12" r="1.3"/><circle cx="13" cy="4" r="1.3"/><path d="M8 7.7l3-2.6"/>',
    "rect": '<rect x="3" y="4" width="10" height="8"/>',
    "ellipse": '<ellipse cx="8" cy="8" rx="5.5" ry="4"/>',
    "add_point": '<path d="M3 11c3-6 7-6 10 0"/><path d="M8 3v4M6 5h4"/>',
    "del_point": '<path d="M3 11c3-6 7-6 10 0"/><path d="M6 5h4"/>',
    "smooth": '<path d="M2.5 11c2-5 9-5 11 0"/><path d="M3 5.5h10"/><circle cx="8" cy="5.5" r="1.3" fill="C"/>',
    "cusp": '<path d="M2.5 12L8 5l5.5 7"/><circle cx="8" cy="5" r="1.3" fill="C"/>',
    "feather": '<rect x="4" y="5" width="8" height="6"/><rect x="2" y="3" width="12" height="10" stroke-dasharray="1.5 1.5"/>',
    "zebra": '<path d="M3 13L13 3M3 9l6-6M7 13l6-6"/>',
    "roi": '<rect x="2.5" y="2.5" width="11" height="11" stroke-dasharray="2 1.5"/><rect x="5.5" y="5.5" width="5" height="5"/>',
    "grid": '<path d="M2.5 6h11M2.5 10h11M6 2.5v11M10 2.5v11"/>',
    "overlay": '<path d="M3 11c2-6 8-6 10 0z"/><circle cx="8" cy="9" r="1"/>',
    "wipe": '<rect x="2.5" y="3.5" width="11" height="9"/><path d="M8 2v12"/>',
    "settings": '<circle cx="8" cy="8" r="2"/><path d="M8 2v2M8 12v2M2 8h2M12 8h2M3.8 3.8l1.4 1.4M10.8 10.8l1.4 1.4M3.8 12.2l1.4-1.4M10.8 5.2l1.4-1.4"/>',
    "refresh": '<path d="M12.5 8A4.5 4.5 0 1 1 11 4.6M11.5 2v3h-3"/>',
    "autokey": '<path d="M8 3l5 5-5 5-5-5z"/><path d="M6.5 9.5L8 6.5l1.5 3M7 8.6h2"/>',
    "visible": '<path d="M1.5 8s2.5-4 6.5-4 6.5 4 6.5 4-2.5 4-6.5 4-6.5-4-6.5-4z"/><circle cx="8" cy="8" r="1.6"/>',
    "lock": '<rect x="4" y="7.5" width="8" height="6"/><path d="M5.5 7.5V5.5a2.5 2.5 0 0 1 5 0v2"/>',
    "plus": '<path d="M8 3.5v9M3.5 8h9"/>',
    "minus": '<path d="M3.5 8h9"/>',
    "up": '<path d="M4 10l4-4 4 4"/>',
    "down": '<path d="M4 6l4 4 4-4"/>',
    "folder": '<path d="M2.5 4.5h4l1 1.5h6v6.5h-11z"/>',
    "curve": '<path d="M2.5 13C6 13 6 3 9.5 3S13.5 8 13.5 8"/><path d="M2.5 2.5v11h11" stroke-opacity="0.5"/>',
}


@lru_cache(maxsize=512)
def icon(name: str, color: str = "", size: int = 16) -> QIcon:
    col = color or T["text"]
    body = _ICONS.get(name, "").replace('fill="C"', f'fill="{col}"')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="{size}" height="{size}" '
           f'fill="none" stroke="{col}" stroke-width="1.15" stroke-linecap="square" stroke-linejoin="miter">'
           f'{body}</svg>')
    r = QSvgRenderer(QByteArray(svg.encode()))
    ic = QIcon()
    for scale in (1, 2):
        pm = QPixmap(size * scale, size * scale)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r.render(p, QRectF(0, 0, size * scale, size * scale))
        p.end()
        pm.setDevicePixelRatio(scale)
        ic.addPixmap(pm)
    return ic


def export_svgs(directory: Path):
    """Write the icon set as SVG files (resources/icons) for use outside Qt."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, body in _ICONS.items():
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="none" stroke="#c8c8c8" '
               'stroke-width="1.15" stroke-linecap="square">' + body.replace('fill="C"', 'fill="#c8c8c8"') + "</svg>")
        (directory / f"{name}.svg").write_text(svg, encoding="utf-8")
