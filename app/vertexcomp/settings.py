"""Persistent user settings (JSON in the user config directory)."""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

from vertexcomp import APP_ID

DEFAULTS: dict[str, dict] = {
    "general": {
        "format_w": 1920,
        "format_h": 1080,
        "fps": 24.0,
        "first_frame": 1,
        "last_frame": 100,
        "autosave_minutes": 2,
        "autosave_backups": 5,
        "undo_depth": 200,
    },
    "performance": {
        "cache_gb": 4.0,
        "threads": 0,
        "readahead_frames": 24,
        "interactive_downscale": True,
        "play_every_frame": False,
    },
    "color": {
        "working_space": "Linear Rec.709",
        "display": "sRGB",
        "exposure": 0.0,
        "gamma": 1.0,
    },
    "viewer": {
        "background": "checker",
        "grid_zoom": 12.0,
        "values_zoom": 70.0,
        "show_values": True,
        "zebra_threshold": 1.0,
        "nearest_zoom": 2.0,
        "overlay_color": "#e8a33c",
        "bbox_color": "#7a7a7a",
    },
    "nodegraph": {
        "edge_style": "curved",
        "snap_to_grid": False,
        "grid_size": 16,
        "font_size": 9,
        "autoconnect": True,
        "max_properties": 4,
        "colors": {
            "Image": "#5f6f7f",
            "Color": "#7c6d4f",
            "Merge": "#5a6a8a",
            "Channel": "#6b5a7a",
            "Keyer": "#4f7a62",
            "Filter": "#6e5560",
            "Transform": "#6a6a4e",
            "Time": "#5a7480",
            "Draw": "#7a6450",
            "AI": "#7a5a86",
            "Other": "#5c5c5c",
        },
    },
    "hotkeys": {
        "create_menu": "Tab",
        "Merge": "M",
        "Blur": "B",
        "ColorCorrect": "C",
        "Grade": "G",
        "Read": "R",
        "Write": "W",
        "Transform": "T",
        "Roto": "O",
        "Shuffle": "H",
        "Keyer": "K",
        "Dot": ".",
        "disable": "D",
        "rename": "N",
        "fit_view": "F",
        "play": "Space",
        "step_back": "Left",
        "step_forward": "Right",
        "prev_key": "Shift+Left",
        "next_key": "Shift+Right",
        "first_frame": "Home",
        "last_frame": "End",
        "exposure_up": "]",
        "exposure_down": "[",
        "view_red": "Shift+R",
        "view_green": "Shift+G",
        "view_blue": "Shift+B",
        "view_alpha": "Shift+A",
        "view_luma": "Shift+Y",
        "view_rgb": "Shift+C",
    },
    "appearance": {
        "theme": "dark",
        "ui_scale": 1.0,
        "font_size": 9,
    },
    "ai": {
        "python": "",
        "models_dir": "",
        "memory_mode": "low vram (model offload)",
        "precision": "fp16",
        "vae_tiling": True,
        "idle_unload_minutes": 0,
        "hf_token": "",
    },
    "paths": {
        "read_dir": "",
        "write_dir": "",
        "plugin_dirs": [],
        "recent_projects": [],
    },
}


def config_dir() -> Path:
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        d = base / "VertexComp"
    elif sys.platform == "darwin":
        d = Path.home() / "Library" / "Application Support" / "VertexComp"
    else:
        d = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / APP_ID
    d.mkdir(parents=True, exist_ok=True)
    return d


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class Settings:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else config_dir() / "settings.json"
        self.data = copy.deepcopy(DEFAULTS)
        self._listeners = []
        self.load()

    def load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                self.data = _merge(DEFAULTS, json.load(f))
        except (OSError, ValueError):
            self.data = copy.deepcopy(DEFAULTS)

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)

    def get(self, section: str, key: str):
        return self.data.get(section, {}).get(key, DEFAULTS.get(section, {}).get(key))

    def set(self, section: str, key: str, value):
        self.data.setdefault(section, {})[key] = value
        for fn in list(self._listeners):
            fn(section, key, value)

    def restore_section(self, section: str):
        self.data[section] = copy.deepcopy(DEFAULTS[section])
        for key, value in self.data[section].items():
            for fn in list(self._listeners):
                fn(section, key, value)

    def on_change(self, fn):
        self._listeners.append(fn)

    def add_recent(self, path: str):
        rec = [p for p in self.get("paths", "recent_projects") if p != path]
        rec.insert(0, path)
        self.set("paths", "recent_projects", rec[:10])
