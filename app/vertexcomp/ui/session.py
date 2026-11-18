"""Application state shared by all panels: project, selection, time, undo,
plus the background render services for the viewer and playback."""
from __future__ import annotations

import collections
import json
import threading
import time
import traceback

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QUndoCommand, QUndoStack

from vertexcomp import _vccore as vc
from vertexcomp.ai import GEN_TYPES
from vertexcomp.project import Project, shape_from_dict, shape_to_dict


# ---------------------------------------------------------------- undo commands
class SnapshotCommand(QUndoCommand):
    """Structural edits (create, delete, connect, paste): whole-graph snapshots.
    The render cache survives because it is keyed by content hashes."""

    def __init__(self, session, text, before, after, selection_after=None):
        super().__init__(text)
        self.s, self.before, self.after = session, before, after
        self.sel_before = [n.name for n in session.selected]
        self.sel_after = selection_after if selection_after is not None else self.sel_before
        self.first = True

    def redo(self):
        if self.first:
            self.first = False
            return
        self.s.restore(self.after, self.sel_after)

    def undo(self):
        self.s.restore(self.before, self.sel_before)


class KnobCommand(QUndoCommand):
    def __init__(self, session, node, knob, before, after, merge_key=None):
        super().__init__(f"{node.name}.{knob}")
        self.s, self.node, self.knob = session, node.name, knob
        self.before, self.after = before, after
        self.merge_key = merge_key
        self.first = True

    def id(self):
        return 1001 if self.merge_key else -1

    def mergeWith(self, other):
        if not isinstance(other, KnobCommand) or other.merge_key is None or other.merge_key != self.merge_key:
            return False
        self.after = other.after
        return True

    def _apply(self, state):
        n = self.s.project.find(self.node)
        if n is not None:
            n.set_knob_state(self.knob, state)
            if n.type == "Text" and self.knob in ("text", "font", "size", "color", "position"):
                from vertexcomp.ui.text_node import update_text_bitmap
                update_text_bitmap(n)
            self.s.node_changed(n)

    def redo(self):
        if self.first:
            self.first = False
            return
        self._apply(self.after)

    def undo(self):
        self._apply(self.before)


class MoveCommand(QUndoCommand):
    def __init__(self, session, before, after):
        super().__init__("move")
        self.s, self.before, self.after = session, before, after
        self.first = True

    def _apply(self, pos):
        for name, (x, y, kind) in pos.items():
            if kind == "node":
                n = self.s.project.find(name)
                if n is not None:
                    m = self.s.project.node_meta(n)
                    m.x, m.y = x, y
            else:
                idx = int(name)
                if idx < len(self.s.project.backdrops):
                    self.s.project.backdrops[idx].x, self.s.project.backdrops[idx].y = x, y
        self.s.layoutChanged.emit()

    def redo(self):
        if self.first:
            self.first = False
            return
        self._apply(self.after)

    def undo(self):
        self._apply(self.before)


class RotoCommand(QUndoCommand):
    def __init__(self, session, node, before, after, merge_key=None):
        super().__init__(f"{node.name} shapes")
        self.s, self.node, self.before, self.after, self.merge_key = session, node.name, before, after, merge_key
        self.first = True

    def id(self):
        return 1002 if self.merge_key else -1

    def mergeWith(self, other):
        if not isinstance(other, RotoCommand) or other.merge_key is None or other.merge_key != self.merge_key:
            return False
        self.after = other.after
        return True

    def _apply(self, shapes):
        n = self.s.project.find(self.node)
        if n is not None:
            n.set_roto_shapes([shape_from_dict(d) for d in shapes])
            self.s.node_changed(n)

    def redo(self):
        if self.first:
            self.first = False
            return
        self._apply(self.after)

    def undo(self):
        self._apply(self.before)


# ---------------------------------------------------------------- session
class Session(QObject):
    graphChanged = Signal()           # nodes added/removed/reconnected: rebuild views
    layoutChanged = Signal()          # positions only
    nodeChanged = Signal(object)      # knob values of a node changed
    selectionChanged = Signal()
    frameChanged = Signal(int)
    viewedChanged = Signal()
    rangeChanged = Signal()
    logMessage = Signal(str, str)     # level, text
    renderStateChanged = Signal()
    rotoSelectionChanged = Signal()

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        cache = int(float(settings.get("performance", "cache_gb")) * (1 << 30))
        self.project = Project(cache)
        self.undo = QUndoStack(self)
        self.undo.setUndoLimit(int(settings.get("general", "undo_depth")))
        self.selected: list = []
        self.frame = 1
        self.viewer_slots: dict[int, str] = {}
        self.active_slot = 1
        self.wipe_slot = 0  # 0 = off
        self.autokey = False
        from vertexcomp.ui.roto_tool import RotoState
        self.roto_state = RotoState()
        self.roto_tool = None
        self.dirty = False
        self.undo.indexChanged.connect(self._mark_dirty)
        self.apply_project_settings()

    def _mark_dirty(self, *_):
        self.dirty = True

    def log(self, text, level="info"):
        self.logMessage.emit(level, text)

    # ------------------------------------------------------------ settings
    def apply_project_settings(self):
        s = self.project.settings
        g = self.settings
        s.fps = float(g.get("general", "fps"))
        s.first = int(g.get("general", "first_frame"))
        s.last = int(g.get("general", "last_frame"))
        s.format_w = int(g.get("general", "format_w"))
        s.format_h = int(g.get("general", "format_h"))
        s.working_space = g.get("color", "working_space")
        try:
            vc.set_working_space(s.working_space)
        except RuntimeError:
            pass
        self.frame = s.first

    # ------------------------------------------------------------ snapshots
    def snapshot(self) -> dict:
        d = self.project.to_dict(include_blobs=False)  # blobs persist in project.blobs
        d.pop("viewer", None)
        return d

    def restore(self, d: dict, selection_names=None):
        p = self.project
        p.graph.clear()
        p.meta.clear()
        p.paste(d)
        from vertexcomp.project import Backdrop
        p.backdrops = [Backdrop(**b) for b in d.get("backdrops", [])]
        self.selected = [n for n in (p.find(x) for x in (selection_names or [])) if n is not None]
        self.graphChanged.emit()
        self.selectionChanged.emit()
        self.viewedChanged.emit()

    def structural(self, text, fn, select=None):
        """Run fn() (which edits the graph) as one undoable step."""
        before = self.snapshot()
        result = fn()
        after = self.snapshot()
        sel = [n.name for n in (select if select is not None else self.selected)]
        if select is not None:
            self.selected = list(select)
        self.undo.push(SnapshotCommand(self, text, before, after, sel))
        self.graphChanged.emit()
        self.selectionChanged.emit()
        self.viewedChanged.emit()
        return result

    # ------------------------------------------------------------ graph edits
    def create_node(self, op_type, x=None, y=None, connect_selected=True):
        sel = self.selected[-1] if (self.selected and connect_selected) else None
        if x is None:
            if sel is not None:
                m = self.project.node_meta(sel)
                x, y = m.x, m.y + 60
            else:
                x, y = 0.0, 0.0

        def make():
            n = self.project.create(op_type, x=x, y=y)
            if sel is not None and n.input_labels and self.settings.get("nodegraph", "autoconnect"):
                try:
                    self.project.graph.connect(n, 0, sel)
                except RuntimeError:
                    pass
            if op_type == "Text":
                from vertexcomp.ui.text_node import update_text_bitmap
                update_text_bitmap(n)
            return n

        n = self.structural(f"create {op_type}", make)
        self.selected = [n]
        self.selectionChanged.emit()
        if op_type in ("Read",) or not self.viewer_slots:
            self.view_node(n, self.active_slot)
        return n

    def delete_nodes(self, nodes, backdrops=()):
        if not nodes and not backdrops:
            return

        def rm():
            # Keep the chain connected through deleted single-input nodes.
            for n in nodes:
                up = n.inputs[0] if n.inputs else None
                for other in self.project.nodes():
                    if other in nodes:
                        continue
                    for i, src in enumerate(other.inputs):
                        if src is not None and src.name == n.name:
                            try:
                                self.project.graph.connect(other, i, up if up is not None and up not in nodes else None)
                            except RuntimeError:
                                self.project.graph.connect(other, i, None)
                self.project.remove(n)
            removed = {n.name for n in nodes}
            for backdrop in self.project.backdrops:
                if backdrop.group:
                    backdrop.members = [name for name in backdrop.members if name not in removed]
            for idx in sorted(backdrops, reverse=True):
                del self.project.backdrops[idx]

        self.structural("delete", rm, select=[])

    def connect(self, dst, slot, src):
        if src is not None and self.project.graph.would_cycle(dst, src):
            self.log("connection refused: it would create a cycle", "warning")
            return False
        self.structural("connect", lambda: self.project.graph.connect(dst, slot, src))
        return True

    def rename(self, node, name):
        if not name or name == node.name:
            return
        if self.project.find(name) is not None:
            self.log(f"name '{name}' is taken", "warning")
            return
        old = node.name

        def do():
            node.name = name
            for backdrop in self.project.backdrops:
                if backdrop.group:
                    backdrop.members = [name if member == old else member for member in backdrop.members]
            for slot, n in list(self.viewer_slots.items()):
                if n == old:
                    self.viewer_slots[slot] = name

        self.structural("rename", do, select=[node])

    def toggle_disable(self, nodes):
        def do():
            on = not all(n.disabled for n in nodes)
            for n in nodes:
                n.disabled = on

        self.structural("disable", do)

    def set_node_meta(self, node, **kw):
        def do():
            m = self.project.node_meta(node)
            for k, v in kw.items():
                setattr(m, k, v)
        self.structural("node appearance", do)

    def paste_text(self, text, at=(40.0, 40.0)):
        try:
            data = json.loads(text)
        except ValueError:
            self.log("clipboard does not contain Vertex Comp nodes", "warning")
            return []
        if "nodes" not in data:
            return []
        xs = [n.get("x", 0) for n in data["nodes"]] or [0]
        ys = [n.get("y", 0) for n in data["nodes"]] or [0]
        off = (at[0] - min(xs), at[1] - min(ys))
        holder = []

        def do():
            holder.extend(self.project.paste(data, offset=off))

        self.structural("paste", do)
        self.selected = holder
        self.selectionChanged.emit()
        return holder

    def copy_text(self, nodes) -> str:
        names = {n.name for n in nodes}
        out = []
        for n in nodes:
            d = self.project.node_to_dict(n)
            d["inputs"] = [i if i in names else None for i in d["inputs"]]
            out.append(d)
        return json.dumps({"vertexcomp": 1, "nodes": out}, indent=1)

    def move_nodes(self, before, after):
        self.undo.push(MoveCommand(self, before, after))

    def disconnect_nodes(self, nodes):
        names = {node.name for node in nodes}
        if not names:
            return
        if not any(
            (node.name in names and any(source is not None for source in node.inputs)) or
            any(source is not None and source.name in names for source in node.inputs)
            for node in self.project.nodes()
        ):
            return

        def do():
            for node in self.project.nodes():
                for slot, source in enumerate(node.inputs):
                    if node.name in names or (source is not None and source.name in names):
                        self.project.graph.connect(node, slot, None)

        self.structural("disconnect node connections", do)

    # ------------------------------------------------------------ knobs
    def set_knob(self, node, knob, value, merge_key=None, comp=None, presets=True):
        """Set a knob from the UI. Animated knobs receive a key at the current
        frame (all components, or `comp`), static knobs change their value."""
        if presets and node.type in GEN_TYPES and knob in ("accel", "model"):
            from vertexcomp.ai.nodes import set_with_preset
            return set_with_preset(self, node, knob, value)
        before = node.knob_state(knob)
        kd = node.knob_def(knob)
        if kd["animatable"] and node.is_animated(knob):
            vals = value if isinstance(value, (list, tuple)) else [value] * kd["comps"]
            cur = node.get(knob, self.frame)
            for c in range(kd["comps"]):
                if comp is not None and c != comp:
                    continue
                if abs(vals[c] - cur[c]) > 1e-12 or not self._has_key(node, knob, c):
                    node.set_key(knob, c, float(self.frame), float(vals[c]))
        else:
            node.set(knob, value)
        if node.type == "Text" and knob in ("text", "font", "size", "color", "position"):
            from vertexcomp.ui.text_node import update_text_bitmap
            update_text_bitmap(node)
        after = node.knob_state(knob)
        if before != after:
            self.undo.push(KnobCommand(self, node, knob, before, after, merge_key))
            self.node_changed(node)

    def _has_key(self, node, knob, comp):
        return any(abs(k[0] - self.frame) < 1e-6 for k in node.keys(knob)[comp])

    def set_knob_state(self, node, knob, state, text=None):
        before = node.knob_state(knob)
        node.set_knob_state(knob, state)
        after = node.knob_state(knob)
        if before != after:
            self.undo.push(KnobCommand(self, node, knob, before, after))
            self.node_changed(node)

    def set_key(self, node, knob, on: bool):
        before = node.knob_state(knob)
        if on:
            cur = node.get(knob, self.frame)
            for c, v in enumerate(cur):
                node.set_key(knob, c, float(self.frame), v)
        else:
            node.remove_key(knob, -1, float(self.frame))
        self.undo.push(KnobCommand(self, node, knob, before, node.knob_state(knob)))
        self.node_changed(node)

    def set_roto(self, node, shapes, merge_key=None):
        if node.type not in ("Roto", "Paint"):
            raise ValueError(f"{node.name} does not support shape data")
        before = [shape_to_dict(s) for s in node.roto_shapes()]
        node.set_roto_shapes(shapes)
        after = [shape_to_dict(s) for s in shapes]
        self.undo.push(RotoCommand(self, node, before, after, merge_key))
        self.node_changed(node)

    def roto_selection_changed(self):
        self.rotoSelectionChanged.emit()

    def node_changed(self, node):
        self.nodeChanged.emit(node)

    # ------------------------------------------------------------ selection, viewing, time
    def select(self, nodes):
        self.selected = list(nodes)
        self.selectionChanged.emit()

    def view_node(self, node, slot=None):
        slot = slot or self.active_slot
        self.viewer_slots[slot] = node.name
        self.active_slot = slot
        self.viewedChanged.emit()

    def viewed_node(self, slot=None):
        name = self.viewer_slots.get(slot or self.active_slot)
        return self.project.find(name) if name else None

    def set_frame(self, f):
        f = int(f)
        if f != self.frame:
            self.frame = f
            self.frameChanged.emit(f)

    def set_range(self, first, last):
        s = self.project.settings
        s.first, s.last = int(first), int(max(first, last))
        self.rangeChanged.emit()

    def key_frames(self):
        """Frames with keys on the selected nodes (timeline ticks, key stepping)."""
        out = set()
        for n in self.selected:
            for k in n.knob_names:
                kd = n.knob_def(k)
                if not kd["animatable"]:
                    continue
                for comp in n.keys(k):
                    out.update(int(round(f[0])) for f in comp)
            if n.type == "Roto":
                for s in n.roto_shapes():
                    out.update(int(round(f)) for f in s.keys)
        return sorted(out)


# ---------------------------------------------------------------- render services
class ViewRequest:
    __slots__ = ("node", "frame", "roi", "slots", "step", "gen", "token")

    def __init__(self, node, frame, roi, slots, step, gen):
        self.node, self.frame, self.roi, self.slots, self.step, self.gen = node, frame, roi, slots, step, gen
        self.token = vc.CancelToken()


class ViewerRenderer(QObject):
    """Renders the viewer's request on a worker thread; only the newest
    request matters, older ones are cancelled. Results are handed to the UI
    thread through a queue polled by a timer (no Qt calls from the worker)."""
    ready = Signal(object)  # dict

    def __init__(self, session):
        super().__init__()
        self.s = session
        self._results = collections.deque()
        self._poll = QTimer(self)
        self._poll.timeout.connect(self._deliver)
        self._poll.start(8)
        self._pending = None
        self._current = None
        self._cv = threading.Condition()
        self._gen = 0
        self._quit = False
        self._t = threading.Thread(target=self._loop, name="viewer-render", daemon=True)
        self._t.start()

    def request(self, node, frame, roi, slots, step):
        with self._cv:
            self._gen += 1
            req = ViewRequest(node, frame, roi, tuple(slots), step, self._gen)
            if self._current is not None:
                self._current.token.cancel()
            self._pending = req
            self._cv.notify()
            return req.gen

    def stop(self):
        with self._cv:
            self._quit = True
            if self._current is not None:
                self._current.token.cancel()
            self._cv.notify()

    def _loop(self):
        eng = self.s.project.engine
        while True:
            with self._cv:
                while self._pending is None and not self._quit:
                    self._cv.wait()
                if self._quit:
                    return
                req, self._pending = self._pending, None
                self._current = req
            out = {"gen": req.gen, "frame": req.frame, "roi": req.roi, "step": req.step, "slots": req.slots,
                   "node": req.node, "array": None, "info": None, "error": "", "ms": 0.0}
            t0 = time.perf_counter()
            try:
                out["info"] = eng.info(req.node, req.frame)
                out["array"] = eng.fetch(req.node, req.frame, req.roi, list(req.slots), req.step, req.token)
            except vc.Cancelled:
                continue
            except Exception as e:  # node errors are shown in the viewer, never raised
                out["error"] = str(e)
                if not isinstance(e, RuntimeError):
                    out["error"] += "\n" + traceback.format_exc()
            out["ms"] = (time.perf_counter() - t0) * 1000.0
            with self._cv:
                self._current = None
            self._results.append(out)

    def _deliver(self):
        while self._results:
            self.ready.emit(self._results.popleft())


class PlaybackCache:
    """Display buffers for upcoming frames, filled by a background thread
    while playing (read-ahead). Entries are keyed by the node hash so edits
    invalidate them implicitly."""

    def __init__(self, session):
        self.s = session
        self.frames: "collections.OrderedDict[tuple, np.ndarray]" = collections.OrderedDict()
        self.lock = threading.Lock()
        self.view = None  # (node, roi, slots, step)
        self.target = 1
        self.direction = 1
        self.active = False
        self.token = vc.CancelToken()
        self._cv = threading.Condition()
        self._t = threading.Thread(target=self._loop, name="readahead", daemon=True)
        self._t.start()

    def configure(self, node, roi, slots, step):
        view = (node.name if node else None, tuple(roi), tuple(slots), step)
        with self.lock:
            if self.view is None or self.view[1:] != view[1:] or (self.view[0] != view[0]):
                self.frames.clear()
            self.view = view
        with self._cv:
            self._cv.notify()

    def key(self, frame):
        if self.view is None or self.view[0] is None:
            return None
        node = self.s.project.find(self.view[0])
        if node is None:
            return None
        try:
            h = self.s.project.engine.hash(node, frame)
        except RuntimeError:
            return None
        return (frame, h) + self.view[1:]

    def get(self, frame):
        k = self.key(frame)
        with self.lock:
            return self.frames.get(k) if k else None

    def put(self, frame, arr):
        k = self.key(frame)
        if k is None:
            return
        cap = max(4, int(self.s.settings.get("performance", "readahead_frames")) * 3)
        with self.lock:
            self.frames[k] = arr
            while len(self.frames) > cap:
                self.frames.popitem(last=False)

    def start(self, frame, direction):
        self.target, self.direction, self.active = frame, direction, True
        self.token = vc.CancelToken()
        with self._cv:
            self._cv.notify()

    def advance(self, frame):
        self.target = frame
        with self._cv:
            self._cv.notify()

    def stop(self):
        self.active = False
        self.token.cancel()

    def _next_missing(self):
        s = self.s.project.settings
        n = int(self.s.settings.get("performance", "readahead_frames"))
        f = self.target
        span = s.last - s.first + 1
        for i in range(n):
            ff = f + i * self.direction
            ff = s.first + (ff - s.first) % span
            if self.get(ff) is None:
                return ff
        return None

    def _loop(self):
        while True:
            with self._cv:
                self._cv.wait(timeout=0.05)
            while self.active and self.view and self.view[0]:
                f = self._next_missing()
                if f is None:
                    break
                node = self.s.project.find(self.view[0])
                if node is None:
                    break
                try:
                    arr = self.s.project.engine.fetch(node, f, self.view[1], list(self.view[2]), self.view[3], self.token)
                    self.put(f, arr)
                except vc.Cancelled:
                    break
                except Exception:
                    # Errors surface through the normal viewer path.
                    self.put(f, None)
                    break
