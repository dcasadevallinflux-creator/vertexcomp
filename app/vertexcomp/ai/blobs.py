"""Images stored inside the project (AI generations, Versions snapshots).

Each blob is kept encoded (PNG for 8-bit generations, compressed float16 NPZ
for snapshots of the comp) under a content id, and its decoded pixels are
registered with the core (`vc.set_image_blob`). Nodes reference blobs by id
through a string knob, so undo snapshots stay small; the bytes are written
into the .vcp under "blobs" on save.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json

import numpy as np

from vertexcomp import _vccore as vc

_SRGB_TO_LIN = np.array([c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                         for c in (np.arange(256) / 255.0)], dtype=np.float32)


def srgb_to_linear(u8: np.ndarray) -> np.ndarray:
    return _SRGB_TO_LIN[u8]


def linear_to_srgb8(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    s = np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return np.clip(np.round(s * 255.0), 0, 255).astype(np.uint8)


def png_to_linear(data: bytes) -> np.ndarray:
    """PNG bytes -> (H, W, 4) premultiplied linear float32."""
    from PySide6.QtGui import QImage

    img = QImage.fromData(data)
    if img.isNull():
        raise ValueError("could not decode PNG")
    img = img.convertToFormat(QImage.Format.Format_RGBA8888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    raw = np.frombuffer(img.constBits(), dtype=np.uint8, count=bpl * h).reshape(h, bpl)[:, :w * 4]
    u8 = raw.reshape(h, w, 4)
    out = np.empty((h, w, 4), np.float32)
    out[..., :3] = srgb_to_linear(u8[..., :3])
    out[..., 3] = u8[..., 3] / 255.0
    out[..., :3] *= out[..., 3:4]
    return out


def linear_to_png(arr: np.ndarray) -> bytes:
    """(H, W, 4) premultiplied linear -> 8-bit sRGB PNG (straight alpha)."""
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice
    from PySide6.QtGui import QImage

    a = np.clip(arr[..., 3:4], 0.0, 1.0)
    rgb = np.where(a > 1e-6, arr[..., :3] / np.maximum(a, 1e-6), 0.0)
    u8 = np.empty(arr.shape[:2] + (4,), np.uint8)
    u8[..., :3] = linear_to_srgb8(rgb)
    u8[..., 3] = np.clip(np.round(a[..., 0] * 255), 0, 255).astype(np.uint8)
    u8 = np.ascontiguousarray(u8)
    h, w = u8.shape[:2]
    img = QImage(u8.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    buf.close()
    return bytes(ba)


def chw_to_hwc4(arr: np.ndarray, chans: list[str]) -> np.ndarray:
    """Engine render output (C, H, W) -> (H, W, 4) rgba; missing alpha = 1."""
    _, h, w = arr.shape
    out = np.zeros((h, w, 4), np.float32)
    names = ["rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"]
    for i, n in enumerate(names):
        if n in chans:
            out[..., i] = arr[chans.index(n)]
        elif i == 3:
            out[..., 3] = 1.0
    return out


class BlobStore:
    def __init__(self):
        self.items: dict[str, dict] = {}   # id -> {"kind", "data" (bytes), "x0", "y0"}

    # ------------------------------------------------------------ adding
    def _register(self, bid: str):
        it = self.items[bid]
        if it["kind"] == "png":
            arr = png_to_linear(it["data"])
        else:
            with np.load(io.BytesIO(it["data"]), allow_pickle=False) as z:
                arr = z["rgba"].astype(np.float32)
        vc.set_image_blob(bid, np.ascontiguousarray(arr), int(it.get("x0", 0)), int(it.get("y0", 0)))

    def _add(self, kind: str, data: bytes, x0: int = 0, y0: int = 0) -> str:
        bid = hashlib.sha1(kind.encode() + data + f"{x0},{y0}".encode()).hexdigest()[:20]
        if bid not in self.items:
            self.items[bid] = {"kind": kind, "data": data, "x0": x0, "y0": y0}
        if not vc.has_image_blob(bid):
            self._register(bid)
        return bid

    def add_png(self, data: bytes) -> str:
        return self._add("png", data)

    def add_array(self, arr: np.ndarray, x0: int = 0, y0: int = 0) -> str:
        """(H, W, 4) premultiplied linear float -> half-float NPZ blob."""
        buf = io.BytesIO()
        np.savez_compressed(buf, rgba=np.ascontiguousarray(arr, dtype=np.float16))
        return self._add("npz", buf.getvalue(), x0, y0)

    def array(self, bid: str) -> np.ndarray | None:
        it = self.items.get(bid)
        if it is None:
            return None
        if it["kind"] == "png":
            return png_to_linear(it["data"])
        with np.load(io.BytesIO(it["data"]), allow_pickle=False) as z:
            return z["rgba"].astype(np.float32)

    def ensure_registered(self):
        for bid in self.items:
            if not vc.has_image_blob(bid):
                self._register(bid)

    # ------------------------------------------------------------ project I/O
    def to_dict(self, ids) -> dict:
        out = {}
        for bid in sorted(ids):
            it = self.items.get(bid)
            if it is not None:
                out[bid] = {"kind": it["kind"], "x0": it.get("x0", 0), "y0": it.get("y0", 0),
                            "data": base64.b64encode(it["data"]).decode("ascii")}
        return out

    def load_dict(self, d: dict):
        for bid, e in (d or {}).items():
            if bid in self.items:
                continue
            try:
                self.items[bid] = {"kind": e["kind"], "data": base64.b64decode(e["data"]),
                                   "x0": int(e.get("x0", 0)), "y0": int(e.get("y0", 0))}
                self._register(bid)
            except Exception:
                self.items.pop(bid, None)

    def clear(self):
        for bid in self.items:
            vc.remove_image_blob(bid)
        self.items.clear()

    def prune(self, keep) -> int:
        """Drop blobs not in `keep` (cache clean-up). Returns the count."""
        drop = [b for b in self.items if b not in keep]
        for b in drop:
            vc.remove_image_blob(b)
            del self.items[b]
        return len(drop)

    def nbytes(self) -> int:
        return sum(len(it["data"]) for it in self.items.values())


def referenced_ids(node_dicts) -> set[str]:
    """Blob ids used by serialised nodes (the `image` knob and Versions lists)."""
    ids = set()
    for nd in node_dicts:
        knobs = nd.get("knobs", {})
        img = knobs.get("image", {}).get("value")
        if isinstance(img, str) and img:
            ids.add(img)
        vers = knobs.get("versions", {}).get("value")
        if isinstance(vers, str) and vers:
            try:
                for v in json.loads(vers):
                    if v.get("id"):
                        ids.add(v["id"])
            except ValueError:
                pass
    return ids
