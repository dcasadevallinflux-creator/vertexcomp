"""Project model and the .vcp file format (versioned JSON).

The C++ graph holds nodes, knobs and connections; this module adds what only
the UI cares about (positions, colours, labels, backdrops, viewer state) and
(de)serialises everything. See docs/file-format.md.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from vertexcomp import PROJECT_SCHEMA, VERSION
from vertexcomp import _vccore as vc

OP_TYPES = {o["type"]: o for o in vc.op_types()}


def knob_defaults(op_type: str) -> dict:
    return {k["name"]: k for k in OP_TYPES[op_type]["knobs"]}


@dataclass
class NodeMeta:
    x: float = 0.0
    y: float = 0.0
    color: str = ""  # empty: category colour
    label: str = ""


@dataclass
class Backdrop:
    title: str = "Backdrop"
    x: float = 0.0
    y: float = 0.0
    w: float = 300.0
    h: float = 200.0
    color: str = "#4a4a44"
    note: bool = False  # StickyNote: text only, no grouping
    group: bool = False
    members: list[str] = field(default_factory=list)
    collapsed: bool = False


@dataclass
class ProjectSettings:
    first: int = 1
    last: int = 100
    fps: float = 24.0
    format_w: int = 1920
    format_h: int = 1080
    working_space: str = "Linear Rec.709"


class SchemaError(RuntimeError):
    pass


def _is_default_state(state: dict, kdef: dict) -> bool:
    if any(state["exprs"]) or any(state["curves"]):
        return False
    t = kdef["type"]
    if t in ("string", "file", "channel", "channels", "text", "enum"):
        d = kdef["sdefault"] if (kdef["sdefault"] or t != "enum") else (kdef["options"][0] if kdef["options"] else "")
        return state["s"] == d
    if t == "button":
        return True
    d = list(kdef["default"]) + [kdef["default"][-1] if kdef["default"] else 0.0] * (kdef["comps"] - len(kdef["default"]))
    return all(abs(a - b) < 1e-12 for a, b in zip(state["v"], d))


def shape_to_dict(s) -> dict:
    return {
        "name": s.name, "visible": s.visible, "locked": s.locked, "closed": s.closed, "invert": s.invert,
        "color": list(s.color), "blend": s.blend, "opacity": s.opacity, "feather": s.feather,
        "feather_falloff": s.feather_falloff, "blur": s.blur, "motion_blur_samples": s.motion_blur_samples,
        "shutter": s.shutter,
        "keys": [[f, [[p.x, p.y, p.in_x, p.in_y, p.out_x, p.out_y, p.feather, p.smooth] for p in pts]]
                 for f, pts in sorted(s.keys.items())],
        "transform": {n: {"value": s.get_transform(n, 0.0) if not s.transform_keys(n) else None,
                          "keys": s.transform_keys(n)} for n in ("tx", "ty", "rot", "sx", "sy", "cx", "cy")},
    }


def shape_from_dict(d: dict):
    s = vc.RotoShape()
    for k in ("name", "visible", "locked", "closed", "invert", "blend", "opacity", "feather", "feather_falloff", "blur",
              "motion_blur_samples", "shutter"):
        if k in d:
            setattr(s, k, d[k])
    s.color = list(d.get("color", [1.0, 0.6, 0.1]))
    keys = {}
    for f, pts in d.get("keys", []):
        keys[float(f)] = [vc.RotoPoint(*p[:6], feather=p[6] if len(p) > 6 else 0.0, smooth=bool(p[7]) if len(p) > 7 else True)
                          for p in pts]
    s.keys = keys
    for n, t in d.get("transform", {}).items():
        if t.get("keys"):
            s.set_transform_keys(n, [tuple(k) for k in t["keys"]])
        elif t.get("value") is not None:
            s.set_transform(n, float(t["value"]))
    return s


class Project:
    def __init__(self, cache_bytes: int = 4 << 30):
        self.graph = vc.Graph()
        self.engine = vc.Engine(self.graph, cache_bytes)
        self.meta: dict[int, NodeMeta] = {}
        self.backdrops: list[Backdrop] = []
        self.settings = ProjectSettings()
        self.viewer: dict = {}
        self.path: str = ""

    # ------------------------------------------------------------ nodes
    def create(self, op_type: str, name: str = "", x: float = 0.0, y: float = 0.0):
        n = self.graph.create(op_type, name)
        self.meta[n.id] = NodeMeta(x, y)
        fmt = [self.settings.format_w, self.settings.format_h]
        if "format" in n.knob_names and op_type not in ("Reformat",):
            n.set("format", fmt)
        if op_type == "Write":
            n.set("first", self.settings.first)
            n.set("last", self.settings.last)
        if op_type == "Transform":
            n.set("center", [fmt[0] / 2, fmt[1] / 2])
        if op_type == "Crop":
            n.set("box", [0, 0, fmt[0], fmt[1]])
        if op_type == "Ramp":
            n.set("p0", [0, fmt[1] / 2])
            n.set("p1", [fmt[0], fmt[1] / 2])
        return n

    def remove(self, node):
        self.graph.remove(node)
        self.meta.pop(node.id, None)

    def node_meta(self, node) -> NodeMeta:
        return self.meta.setdefault(node.id, NodeMeta())

    def nodes(self):
        return self.graph.nodes()

    def find(self, name):
        return self.graph.find(name)

    # ------------------------------------------------------------ serialisation
    def node_to_dict(self, n) -> dict:
        m = self.node_meta(n)
        defs = knob_defaults(n.type)
        knobs = {}
        for k in n.knob_names:
            if defs[k]["type"] == "button":
                continue
            st = n.knob_state(k)
            if _is_default_state(st, defs[k]):
                continue
            entry = {}
            if defs[k]["type"] in ("string", "file", "channel", "channels", "text", "enum"):
                entry["value"] = st["s"]
            else:
                entry["value"] = st["v"]
                if any(st["curves"]):
                    entry["curves"] = st["curves"]
                if any(st["exprs"]):
                    entry["exprs"] = st["exprs"]
            knobs[k] = entry
        d = {
            "name": n.name, "type": n.type, "x": m.x, "y": m.y, "knobs": knobs,
            "inputs": [i.name if i is not None else None for i in n.inputs],
        }
        if m.color:
            d["color"] = m.color
        if m.label:
            d["label"] = m.label
        if n.disabled:
            d["disabled"] = True
        if n.type == "Roto":
            d["roto"] = [shape_to_dict(s) for s in n.roto_shapes()]
        return d

    def apply_node_dict(self, n, d: dict, base_dir: str = ""):
        defs = knob_defaults(n.type)
        for k, entry in d.get("knobs", {}).items():
            if k not in defs:
                continue
            t = defs[k]["type"]
            if t in ("string", "file", "channel", "channels", "text", "enum"):
                val = entry["value"]
                if t == "file" and val and base_dir and not os.path.isabs(val) and not val.startswith("["):
                    val = os.path.normpath(os.path.join(base_dir, val))
                n.set_knob_state(k, {"s": val})
            else:
                n.set_knob_state(k, {"v": entry.get("value", []), "curves": entry.get("curves", []),
                                     "exprs": entry.get("exprs", [])})
        n.disabled = bool(d.get("disabled", False))
        m = self.node_meta(n)
        m.x, m.y = float(d.get("x", 0)), float(d.get("y", 0))
        m.color = d.get("color", "")
        m.label = d.get("label", "")
        if n.type == "Roto" and "roto" in d:
            n.set_roto_shapes([shape_from_dict(s) for s in d["roto"]])

    def to_dict(self, relative_to: str = "") -> dict:
        nodes = [self.node_to_dict(n) for n in self.nodes()]
        if relative_to:
            base = os.path.dirname(os.path.abspath(relative_to))
            for nd in nodes:
                for k, e in nd["knobs"].items():
                    if k == "file" and isinstance(e.get("value"), str) and os.path.isabs(e["value"]):
                        try:
                            e["value"] = os.path.relpath(e["value"], base).replace("\\", "/")
                        except ValueError:
                            pass  # different drive on Windows
        return {
            "schema": PROJECT_SCHEMA,
            "app_version": VERSION,
            "settings": vars(self.settings),
            "nodes": nodes,
            "backdrops": [vars(b) for b in self.backdrops],
            "viewer": self.viewer,
        }

    def paste(self, data: dict, offset=(0.0, 0.0), base_dir: str = ""):
        """Create nodes from a (partial) project dict. Returns created nodes.
        Names that clash are renamed; links between pasted nodes are kept."""
        created = {}
        for nd in data.get("nodes", []):
            if nd.get("type") not in OP_TYPES:
                continue
            n = self.graph.create(nd["type"], nd["name"])
            self.meta[n.id] = NodeMeta()
            self.apply_node_dict(n, nd, base_dir)
            m = self.node_meta(n)
            m.x += offset[0]
            m.y += offset[1]
            created[nd["name"]] = n
        for nd in data.get("nodes", []):
            n = created.get(nd["name"])
            if n is None:
                continue
            for slot, src in enumerate(nd.get("inputs", [])):
                if src is None or slot >= len(n.inputs):
                    continue
                s = created.get(src)
                if s is None and data.get("keep_external_inputs"):
                    s = self.graph.find(src)
                if s is not None:
                    try:
                        self.graph.connect(n, slot, s)
                    except RuntimeError:
                        pass
        return list(created.values())

    def save(self, path: str, relative: bool = False):
        d = self.to_dict(path if relative else "")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)
        os.replace(tmp, path)
        self.path = path

    @staticmethod
    def migrate(d: dict) -> dict:
        schema = d.get("schema", 0)
        if schema > PROJECT_SCHEMA:
            raise SchemaError(
                f"project was saved by a newer version (schema {schema}, this build reads {PROJECT_SCHEMA})")
        if schema < 1:
            schema = 1
        if schema < 2:
            for node in d.get("nodes", []):
                if node.get("type") == "Merge":
                    inputs = list(node.get("inputs", []))
                    if len(inputs) == 3:
                        mask = inputs.pop()
                        inputs.extend([None] * (8 - len(inputs)))
                        inputs.append(mask)
                        node["inputs"] = inputs
            schema = 2
        d["schema"] = PROJECT_SCHEMA
        return d

    def load(self, path: str, allow_newer: bool = False):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        try:
            d = self.migrate(d)
        except SchemaError:
            if not allow_newer:
                raise
        self.clear()
        st = d.get("settings", {})
        for k, v in st.items():
            if hasattr(self.settings, k):
                setattr(self.settings, k, type(getattr(self.settings, k))(v))
        try:
            vc.set_working_space(self.settings.working_space)
        except RuntimeError:
            pass
        d["keep_external_inputs"] = False
        self.paste(d, base_dir=os.path.dirname(os.path.abspath(path)))
        self.backdrops = [Backdrop(**{k: v for k, v in b.items() if k in Backdrop.__dataclass_fields__})
                          for b in d.get("backdrops", [])]
        self.viewer = d.get("viewer", {})
        self.path = path

    def clear(self):
        self.graph.clear()
        self.meta.clear()
        self.backdrops.clear()
        self.viewer = {}
        self.engine.clear_cache()
        self.path = ""

    # ------------------------------------------------------------ reconnect media
    def remap_paths(self, old_root: str, new_root: str) -> int:
        count = 0
        for n in self.nodes():
            if n.type != "Read":
                continue
            f = n.get("file")
            if f.replace("\\", "/").startswith(old_root.replace("\\", "/")):
                n.set("file", new_root + f[len(old_root):])
                n.revision += 1
                count += 1
        return count


def write_node_range(project: Project, node, first: int, last: int, progress=None, cancel=None, log=None):
    """Render a Write node over a frame range (images or movie). Shared by the
    UI render job and the command line renderer."""
    from vertexcomp import movie

    ftype = node.get("fileType")
    path = node.get("file")
    if not path:
        raise RuntimeError(f"{node.name}: no output file")
    total = last - first + 1
    if ftype in ("mov", "mp4"):
        writer = None
        try:
            for i, f in enumerate(range(first, last + 1)):
                if cancel is not None and cancel.cancelled:
                    raise vc.Cancelled()
                arr, chans = project.engine.prepare_write(node, f, cancel)
                if writer is None:
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    h, w = arr.shape[:2]
                    writer = movie.MovieWriter(path, w, h, project.settings.fps, node.get("codec"),
                                               int(node.get("quality")[0]))
                rgb = arr[..., :3] if arr.shape[2] >= 3 else arr[..., :1].repeat(3, axis=2)
                writer.write(rgb)
                if progress:
                    progress(i + 1, total, f)
        finally:
            if writer is not None:
                writer.close()
        if log:
            log(f"{node.name}: wrote {path}")
        return
    for i, f in enumerate(range(first, last + 1)):
        wrote = project.engine.write_frame(node, f, cancel)
        if log:
            log(f"{node.name}: frame {f} {'written' if wrote else 'skipped (exists)'}")
        if progress:
            progress(i + 1, total, f)
