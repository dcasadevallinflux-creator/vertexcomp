"""UI-side logic for the AI nodes: Generate / Cancel / Clear, speed presets,
Versions (add, switch, rename, delete) and the panels shown under their knobs.
"""
from __future__ import annotations

import json
import os
import random
import time

import numpy as np
from PySide6.QtCore import QObject, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from vertexcomp.ai import GEN_TYPES
from vertexcomp.ai.blobs import chw_to_hwc4, linear_to_png, linear_to_srgb8, referenced_ids
from vertexcomp.ai.catalog import ACCEL, BASE_MODELS, MODELS, required_models
from vertexcomp.ai.client import get_manager

RGBA = ["rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"]
SIGNATURE_KEYS = ("prompt", "negative", "model", "accel", "loraScale", "checkpoint", "seed", "steps", "cfg", "sampler",
                  "spacing", "size", "denoise", "varSeed", "varStrength")


# ---------------------------------------------------------------- controller
class Job:
    def __init__(self, rid, frame):
        self.rid, self.frame = rid, frame
        self.step, self.steps = 0, 0
        self.status = "starting AI worker..."
        self.t0 = time.perf_counter()


class AIController(QObject):
    """One per session: pumps the AI manager on the UI thread and tracks
    generation jobs by node name."""
    jobChanged = Signal(str)

    def __init__(self, session):
        super().__init__(session)
        self.s = session
        self.mgr = get_manager(session.settings)
        self.jobs: dict[str, Job] = {}
        self.errors: dict[str, str] = {}
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.mgr.pump)
        self._timer.start(40)
        self._idle = QTimer(self)
        self._idle.timeout.connect(self.mgr.check_idle)
        self._idle.start(30000)
        self.mgr.cache_clear_hooks.append(self._clear_app_caches)

    def _clear_app_caches(self):
        eng = self.s.project.engine
        eng.clear_cache()
        keep = referenced_in_session(self.s)
        dropped = self.s.project.blobs.prune(keep)
        _THUMBS.clear()
        if dropped:
            self.s.log(f"removed {dropped} unused stored images")
        self.s.viewedChanged.emit()

    def finish(self, name, error=""):
        self.jobs.pop(name, None)
        if error:
            self.errors[name] = error
        else:
            self.errors.pop(name, None)
        self.jobChanged.emit(name)


def controller(session) -> AIController:
    c = getattr(session, "_ai_controller", None)
    if c is None:
        c = AIController(session)
        session._ai_controller = c
    return c


# ---------------------------------------------------------------- presets
def set_with_preset(session, node, knob, value):
    """Changing the speed option (or the base model) also applies the
    recommended steps / guidance / sampler, as one undo step."""
    session.undo.beginMacro(f"{node.name}.{knob}")
    try:
        session.set_knob(node, knob, value, presets=False)
        accel = node.get("accel")
        p = ACCEL.get(accel)
        if p is not None and knob == "accel":
            session.set_knob(node, "steps", p.steps, presets=False)
            session.set_knob(node, "cfg", p.cfg, presets=False)
            session.set_knob(node, "sampler", p.sampler, presets=False)
    finally:
        session.undo.endMacro()


# ---------------------------------------------------------------- generation
def node_params(node, frame) -> dict:
    out = {}
    for k in SIGNATURE_KEYS:
        v = node.get(k, float(frame))
        if isinstance(v, list):
            ints = node.knob_def(k)["type"] in ("int", "bool") or k == "size"
            v = [int(round(x)) if ints else round(x, 6) for x in v]
            v = v[0] if len(v) == 1 else v
        out[k] = v
    out["init"] = node.inputs[0] is not None
    return out


def gen_info(node) -> dict:
    try:
        return json.loads(node.get("genInfo") or "{}")
    except ValueError:
        return {}


def is_stale(node, frame) -> bool:
    info = gen_info(node)
    if not info or not node.get("image"):
        return False
    cur = node_params(node, frame)
    return any(info.get(k) != cur.get(k) for k in SIGNATURE_KEYS + ("init",))


def missing_models(mgr, node) -> list[str]:
    model, accel = node.get("model"), node.get("accel")
    need = required_models(node.type, model, accel)
    if model == "custom checkpoint" and not mgr.installed("sdxl-base"):
        need = need + ["sdxl-config"] if "sdxl-config" in MODELS else need
    return [m for m in need if not mgr.installed(m)]


def _render_full(session, src, frame) -> np.ndarray:
    eng = session.project.engine
    info = eng.info(src, float(frame))
    w, h = info["format"][0], info["format"][1]
    arr, chans, bbox = eng.render(src, float(frame), (0, 0, w, h), RGBA)
    full = np.zeros((h, w, 4), np.float32)
    if arr.size:
        x0, y0, x1, y1 = bbox
        full[y0:y1, x0:x1] = chw_to_hwc4(arr, list(chans))
    return full


def build_request(session, node, frame) -> dict:
    mgr = get_manager(session.settings)
    st = session.settings
    model, accel = node.get("model"), node.get("accel")
    preset = ACCEL.get(accel)
    if model == "custom checkpoint":
        ck = node.get("checkpoint")
        if not ck or not os.path.isfile(ck):
            raise RuntimeError("model is 'custom checkpoint' but the checkpoint file is not set or missing")
        cfg_dir = None
        for mid in ("sdxl-base", "sdxl-config"):
            if mid in MODELS and mgr.installed(mid):
                cfg_dir = str(mgr.model_dir(mid))
                break
        base = {"file": ck, "config": cfg_dir}
    else:
        base = {"dir": str(mgr.model_dir(BASE_MODELS[model])), "variant": "fp16"}
    lora = None
    if preset is not None and preset.lora:
        lora = {"path": str(mgr.model_dir(preset.lora) / MODELS[preset.lora].main_file),
                "scale": float(node.get("loraScale")[0])}
    spacing = node.get("spacing")
    if spacing == "auto":
        spacing = preset.spacing if preset else None
    size = node.get("size")
    req = {
        "base": base, "lora": lora, "vae_dir": str(mgr.model_dir("sdxl-vae-fp16-fix")),
        "prompt": node.get("prompt"), "negative": node.get("negative"),
        "seed": int(node.get("seed")[0]), "steps": int(node.get("steps")[0]), "cfg": float(node.get("cfg")[0]),
        "sampler": node.get("sampler"), "spacing": spacing,
        "width": int(size[0]), "height": int(size[1]), "denoise": float(node.get("denoise")[0]),
        "var_seed": int(node.get("varSeed")[0]), "var_strength": float(node.get("varStrength")[0]),
        "memory_mode": st.get("ai", "memory_mode"), "precision": st.get("ai", "precision"),
        "vae_tiling": bool(st.get("ai", "vae_tiling")),
        "out_path": str(mgr.temp_dir() / f"gen_{node.name}_{int(time.time() * 1000)}.png"),
    }
    if node.inputs[0] is not None:
        full = _render_full(session, node.inputs[0], frame)
        path = mgr.temp_dir() / f"init_{node.name}_{int(time.time() * 1000)}.png"
        path.write_bytes(linear_to_png(full))
        req["init_image"] = str(path)
    return req


def _ask_download(session, node, missing) -> bool:
    mgr = get_manager(session.settings)
    total = sum(MODELS[m].size_mb for m in missing)
    names = "\n".join(f"  {MODELS[m].label} ({MODELS[m].size_mb / 1024:.1f} GB) - {MODELS[m].license}" for m in missing)
    r = QMessageBox.question(QApplication.activeWindow(), "Models needed",
                             f"{node.name} needs models that are not downloaded yet:\n{names}\n\n"
                             f"Download now ({total / 1024:.1f} GB) into {mgr.models_dir()}?\n"
                             "Generation starts automatically when the download completes.")
    if r != QMessageBox.StandardButton.Yes:
        return False
    name = node.name
    pending = set(missing)

    def done(mid, err):
        pending.discard(mid)
        if err:
            session.log(f"download of {MODELS[mid].label} failed: {err}", "error")
            controller(session).finish(name, f"download failed: {err}")
            return
        session.log(f"downloaded {MODELS[mid].label}")
        if not pending:
            n = session.project.find(name)
            if n is not None:
                generate(session, n)

    for m in missing:
        mgr.download(m, on_done=done)
    c = controller(session)
    c.errors.pop(name, None)
    c.jobChanged.emit(name)
    return True


def generate(session, node):
    c = controller(session)
    mgr = c.mgr
    if node.name in c.jobs:
        return
    if not node.get("prompt").strip() and node.inputs[0] is None:
        c.finish(node.name, "write a prompt first")
        return
    missing = missing_models(mgr, node)
    if missing:
        _ask_download(session, node, missing)
        return
    frame = session.frame
    if node.get("randomSeed")[0]:
        session.set_knob(node, "seed", random.randrange(0, 2**31 - 1))
    try:
        req = build_request(session, node, frame)
    except Exception as e:
        c.finish(node.name, str(e))
        session.log(f"{node.name}: {e}", "error")
        return
    params = node_params(node, frame)
    name = node.name

    def on_result(msg):
        n = session.project.find(name)
        job = c.jobs.get(name)
        try:
            with open(msg["path"], "rb") as f:
                data = f.read()
            os.remove(msg["path"])
        except OSError as e:
            c.finish(name, f"result missing: {e}")
            return
        if n is None:
            c.finish(name, "node was deleted")
            return
        bid = session.project.blobs.add_png(data)
        info = dict(params, frame=frame, seconds=msg.get("seconds"), peak_vram_gb=msg.get("peak_vram_gb"),
                    time=time.strftime("%Y-%m-%d %H:%M:%S"), width=msg.get("width"), height=msg.get("height"))
        session.undo.beginMacro(f"{name} generate")
        try:
            session.set_knob(n, "image", bid)
            session.set_knob(n, "genFrame", int(job.frame if job else frame))
            session.set_knob(n, "genInfo", json.dumps(info))
        finally:
            session.undo.endMacro()
        if req.get("init_image"):
            try:
                os.remove(req["init_image"])
            except OSError:
                pass
        c.finish(name)
        show_in_viewer(session, n)
        extra = f", peak VRAM {msg['peak_vram_gb']} GB" if msg.get("peak_vram_gb") else ""
        session.log(f"{name}: generated {msg.get('width')}x{msg.get('height')} in {msg.get('seconds')} s{extra}")

    def on_error(text):
        c.finish(name, text)
        if not text.startswith("cancelled"):
            session.log(f"{name}: {text}", "error")

    def on_progress(step, steps):
        j = c.jobs.get(name)
        if j:
            j.step, j.steps = step, steps
            j.status = f"step {step}/{steps}"
            c.jobChanged.emit(name)

    def on_status(text):
        j = c.jobs.get(name)
        if j:
            j.status = text
            c.jobChanged.emit(name)

    try:
        rid = mgr.request("generate", on_result=on_result, on_error=on_error, on_progress=on_progress,
                          on_status=on_status, **req)
    except Exception as e:
        c.finish(name, str(e))
        return
    c.jobs[name] = Job(rid, frame)
    c.errors.pop(name, None)
    c.jobChanged.emit(name)


def show_in_viewer(session, node):
    """View a freshly generated image: put the node in the active viewer slot
    and fit it once the new format arrives."""
    if session.settings.get("ai", "view_after_generate") is False:
        return
    session._ai_fit_next = node.name
    if session.viewed_node() is not node:
        session.view_node(node)
    else:
        session.viewedChanged.emit()


def cancel(session, node):
    c = controller(session)
    j = c.jobs.get(node.name)
    if j is not None:
        c.mgr.cancel(j.rid)
        j.status = "cancelling..."
        c.jobChanged.emit(node.name)


def clear_image(session, node):
    session.undo.beginMacro(f"{node.name} clear image")
    try:
        session.set_knob(node, "image", "")
        session.set_knob(node, "genInfo", "")
    finally:
        session.undo.endMacro()


# ---------------------------------------------------------------- Versions
def versions_of(node) -> list[dict]:
    try:
        v = json.loads(node.get("versions") or "[]")
        return v if isinstance(v, list) else []
    except ValueError:
        return []


def add_version(session, node):
    src = node.inputs[0]
    if src is None:
        session.log(f"{node.name}: connect an input to store a version", "warning")
        return None
    frame = session.frame
    eng = session.project.engine
    try:
        info = eng.info(src, float(frame))
    except RuntimeError as e:
        session.log(f"{node.name}: {e}", "error")
        return None
    bid = info["meta"].get("vc/blob", "")
    if not bid or bid not in session.project.blobs.items:
        x0, y0, x1, y1 = info["bbox"]
        if x1 <= x0 or y1 <= y0:
            session.log(f"{node.name}: the input is empty at frame {frame}", "warning")
            return None
        try:
            arr, chans, bbox = eng.render(src, float(frame), (x0, y0, x1, y1), RGBA)
        except RuntimeError as e:
            session.log(f"{node.name}: {e}", "error")
            return None
        bid = session.project.blobs.add_array(chw_to_hwc4(arr, list(chans)), bbox[0], bbox[1])
    vers = versions_of(node)
    num = max([int(v.get("num", i + 1)) for i, v in enumerate(vers)] + [0]) + 1
    vers.append({"id": bid, "num": num, "label": f"v{num:03d}", "frame": int(frame),
                 "time": time.strftime("%Y-%m-%d %H:%M"), "source": src.name})
    session.undo.beginMacro(f"{node.name} add version")
    try:
        session.set_knob(node, "versions", json.dumps(vers))
        _select(session, node, len(vers), vers)
    finally:
        session.undo.endMacro()
    session.log(f"{node.name}: stored {vers[-1]['label']} (frame {frame})")
    return len(vers)


def _select(session, node, index, vers):
    index = max(0, min(index, len(vers)))
    session.set_knob(node, "current", index)
    if index > 0:
        v = vers[index - 1]
        session.set_knob(node, "image", v["id"])
        session.set_knob(node, "genFrame", int(v.get("frame", session.frame)))
    else:
        session.set_knob(node, "image", "")


def set_version(session, node, index):
    vers = versions_of(node)
    if int(node.get("current")[0]) == index:
        return
    session.undo.beginMacro(f"{node.name} version {index}")
    try:
        _select(session, node, index, vers)
    finally:
        session.undo.endMacro()


def delete_version(session, node, index):
    vers = versions_of(node)
    if not 1 <= index <= len(vers):
        return
    cur = int(node.get("current")[0])
    del vers[index - 1]
    session.undo.beginMacro(f"{node.name} delete version")
    try:
        session.set_knob(node, "versions", json.dumps(vers))
        if cur == index:
            _select(session, node, min(index, len(vers)), vers)
        elif cur > index:
            _select(session, node, cur - 1, vers)
    finally:
        session.undo.endMacro()


def rename_version(session, node, index, label):
    vers = versions_of(node)
    if 1 <= index <= len(vers) and label:
        vers[index - 1]["label"] = label
        session.set_knob(node, "versions", json.dumps(vers))


def referenced_in_session(session) -> set[str]:
    """Blob ids used by the graph or reachable through undo (kept on prune)."""
    ids = referenced_ids(session.project.to_dict(include_blobs=False)["nodes"])
    for i in range(session.undo.count()):
        cmd = session.undo.command(i)
        stack = [cmd]
        while stack:
            c = stack.pop()
            for attr in ("before", "after"):
                v = getattr(c, attr, None)
                if isinstance(v, dict) and "nodes" in v:
                    ids |= referenced_ids(v["nodes"])
                elif isinstance(v, dict) and isinstance(v.get("s"), str):
                    s = v["s"]
                    if getattr(c, "knob", "") == "image" and s:
                        ids.add(s)
                    elif getattr(c, "knob", "") == "versions" and s:
                        ids |= referenced_ids([{"knobs": {"versions": {"value": s}}}])
            stack.extend(c.child(j) for j in range(c.childCount()))
    return ids


# ---------------------------------------------------------------- buttons
def handle_button(session, node, name) -> bool:
    if node.type in GEN_TYPES and name == "generate":
        generate(session, node)
    elif node.type in GEN_TYPES and name == "clearImage":
        clear_image(session, node)
    elif node.type == "Versions" and name == "addVersion":
        add_version(session, node)
    else:
        return False
    return True


# ---------------------------------------------------------------- panels
_THUMBS: dict[str, QIcon] = {}


def thumbnail(session, bid, size=56) -> QIcon:
    ic = _THUMBS.get(bid)
    if ic is not None:
        return ic
    arr = session.project.blobs.array(bid)
    if arr is None:
        return QIcon()
    h, w = arr.shape[:2]
    step = max(1, int(max(h, w) / size))
    small = arr[::step, ::step]
    a = np.clip(small[..., 3:4], 0, 1)
    rgb = small[..., :3] + (1 - a) * 0.18  # over mid grey
    u8 = np.ascontiguousarray(linear_to_srgb8(rgb))
    hh, ww = u8.shape[:2]
    img = QImage(u8.data, ww, hh, ww * 3, QImage.Format.Format_RGB888).copy()
    ic = QIcon(QPixmap.fromImage(img))
    _THUMBS[bid] = ic
    return ic


class GenPanel(QWidget):
    def __init__(self, session, node):
        super().__init__()
        self.s, self.node = session, node
        self.c = controller(session)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(3)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setObjectName("knobLabel")
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row = QHBoxLayout()
        self.bar = QProgressBar()
        self.bar.setTextVisible(True)
        self.cancel = QPushButton("Cancel")
        self.cancel.clicked.connect(lambda: cancel(self.s, self.node))
        row.addWidget(self.bar, 1)
        row.addWidget(self.cancel)
        self.dl = QPushButton("Download missing models")
        self.dl.clicked.connect(self._download)
        lay.addWidget(self.status)
        lay.addLayout(row)
        lay.addWidget(self.dl)
        self.c.jobChanged.connect(self._job)
        self.c.mgr.listeners.append(self._mgr_event)
        self.destroyed.connect(lambda *_: self._detach())
        self._poll = QTimer(self)
        self._poll.timeout.connect(self.refresh)
        self.refresh()

    def _detach(self):
        try:
            self.c.mgr.listeners.remove(self._mgr_event)
        except ValueError:
            pass

    def _mgr_event(self, kind, payload):
        if kind in ("download", "models"):
            self.refresh()

    def _job(self, name):
        if name == self.node.name:
            self.refresh()

    def _download(self):
        miss = missing_models(self.c.mgr, self.node)
        for m in miss:
            self.c.mgr.download(m)
        self._poll.start(500)
        self.refresh()

    def refresh(self):
        n, mgr = self.node, self.c.mgr
        job = self.c.jobs.get(n.name)
        self.bar.setVisible(job is not None)
        self.cancel.setVisible(job is not None)
        miss = missing_models(mgr, n)
        downloading = [m for m in miss if m in mgr.downloads and not mgr.downloads[m].get("finished")]
        if job is not None:
            self.bar.setMaximum(max(1, job.steps or 1))
            self.bar.setValue(job.step)
            self.bar.setFormat(f"{job.status}  {time.perf_counter() - job.t0:.0f}s")
            self.status.setText("Generating...")
            return
        if downloading:
            done = sum(mgr.downloads[m]["done"] for m in downloading)
            total = sum(mgr.downloads[m]["total"] for m in downloading) or 1
            self.bar.setVisible(True)
            self.bar.setMaximum(1000)
            self.bar.setValue(int(1000 * done / total))
            self.bar.setFormat(f"downloading {done / 2**30:.2f} / {total / 2**30:.2f} GB")
            if not self._poll.isActive():
                self._poll.start(500)
        elif self._poll.isActive():
            self._poll.stop()
        self.dl.setVisible(bool(miss) and not downloading)
        if miss and not downloading:
            self.dl.setText("Download missing models: " + ", ".join(MODELS[m].label for m in miss))
        lines = []
        err = self.c.errors.get(n.name)
        if err:
            lines.append(f"<span style='color:#d06a5f'>{err.splitlines()[0][:300]}</span>")
        info = gen_info(n)
        if n.get("image") and info:
            lines.append(f"Cached image {info.get('width')}x{info.get('height')}, seed {info.get('seed')}, "
                         f"{info.get('steps')} steps, {info.get('seconds')} s (frame {int(n.get('genFrame')[0])})")
            if is_stale(n, self.s.frame):
                lines.append("<span style='color:#e8a33c'>Parameters changed: showing the cached image until "
                             "you press Generate.</span>")
        elif n.get("image"):
            lines.append("Cached image.")
        else:
            lines.append("No image yet. Write a prompt and press Generate.")
        self.status.setText("<br>".join(lines))


class VersionsPanel(QWidget):
    def __init__(self, session, node):
        super().__init__()
        self.s, self.node = session, node
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(3)
        self.list = QListWidget()
        self.list.setIconSize(QSize(56, 40))
        self.list.setMinimumHeight(120)
        self.list.currentRowChanged.connect(self._picked)
        self.list.itemDoubleClicked.connect(lambda _: self._rename())
        row = QHBoxLayout()
        ren = QPushButton("Rename")
        ren.clicked.connect(self._rename)
        rem = QPushButton("Delete")
        rem.clicked.connect(self._delete)
        row.addStretch(1)
        row.addWidget(ren)
        row.addWidget(rem)
        lay.addWidget(self.list)
        lay.addLayout(row)
        self._sig = None
        self.refresh()

    def refresh(self):
        vers = versions_of(self.node)
        cur = int(self.node.get("current")[0])
        sig = (json.dumps(vers), cur)
        if sig == self._sig:
            return
        self._sig = sig
        self.list.blockSignals(True)
        self.list.clear()
        live = QListWidgetItem("live input")
        self.list.addItem(live)
        for v in vers:
            text = f"{v.get('label')}   frame {v.get('frame')}   {v.get('time', '')}"
            it = QListWidgetItem(thumbnail(self.s, v["id"]), text)
            it.setToolTip(f"from {v.get('source', '')}")
            self.list.addItem(it)
        self.list.setCurrentRow(min(cur, len(vers)))
        self.list.blockSignals(False)

    def _picked(self, row):
        if row >= 0:
            set_version(self.s, self.node, row)

    def _rename(self):
        row = self.list.currentRow()
        vers = versions_of(self.node)
        if 1 <= row <= len(vers):
            label, ok = QInputDialog.getText(self, "Rename version", "Label", text=vers[row - 1].get("label", ""))
            if ok:
                rename_version(self.s, self.node, row, label.strip())

    def _delete(self):
        row = self.list.currentRow()
        if row >= 1:
            delete_version(self.s, self.node, row)


def make_panel(session, node):
    if node.type in GEN_TYPES:
        return GenPanel(session, node)
    if node.type == "Versions":
        return VersionsPanel(session, node)
    return None
