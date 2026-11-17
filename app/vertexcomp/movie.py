"""Movie decode/encode through the FFmpeg binary shipped with imageio-ffmpeg.

Decoding keeps one reader per file and reads forward; a seek restarts the
reader with an input-side `-ss`, which FFmpeg decodes frame-accurately. The
first open probes frame count and rate, so random access does not depend on
container index quality. Pixels are converted to float32 with NumPy only.
"""
from __future__ import annotations

import re
import subprocess
import threading

import numpy as np

try:
    import imageio_ffmpeg

    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    FFMPEG_VERSION = imageio_ffmpeg.get_ffmpeg_version()
except Exception:  # pragma: no cover - optional dependency
    FFMPEG = None
    FFMPEG_VERSION = ""

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def available() -> bool:
    return FFMPEG is not None


def _frame_sync_args():
    match = re.match(r"(\d+)\.(\d+)", FFMPEG_VERSION)
    if match and tuple(map(int, match.groups())) < (5, 1):
        return ["-vsync", "0"]
    return ["-fps_mode", "passthrough"]


def _probe(path: str):
    cmd = [FFMPEG, "-hide_banner", "-i", path, "-map", "0:v:0", "-c", "copy", "-f", "null", "-"]
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", creationflags=_NO_WINDOW)
    err = p.stderr
    m = re.search(r"Stream #\S+.*Video:.*?(\d{2,5})x(\d{2,5})", err)
    if not m:
        return None
    w, h = int(m.group(1)), int(m.group(2))
    fps = 24.0
    fm = re.search(r"(\d+(?:\.\d+)?) fps", err)
    if fm:
        fps = float(fm.group(1))
    frames = 0
    for fm in re.finditer(r"frame=\s*(\d+)", err):
        frames = int(fm.group(1))
    if frames == 0:
        dm = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", err)
        if dm:
            secs = int(dm.group(1)) * 3600 + int(dm.group(2)) * 60 + float(dm.group(3))
            frames = int(round(secs * fps))
    return frames, w, h, fps


class _Reader:
    def __init__(self, path, w, h, fps):
        self.path, self.w, self.h, self.fps = path, w, h, fps
        self.proc = None
        self.next = 0

    def _start(self, frame):
        self.close()
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error"]
        if frame > 0:
            cmd += ["-ss", f"{frame / self.fps:.6f}"]
        cmd += ["-i", self.path, "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb48le"] + _frame_sync_args() + ["-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=_NO_WINDOW)
        self.next = frame

    def read(self, frame):
        if self.proc is None or frame < self.next or frame > self.next + 48:
            self._start(frame)
        size = self.w * self.h * 6
        while True:
            buf = self.proc.stdout.read(size)
            if len(buf) < size:
                self.close()
                return None
            idx = self.next
            self.next += 1
            if idx == frame:
                arr = np.frombuffer(buf, dtype="<u2").reshape(self.h, self.w, 3)
                return (arr.astype(np.float32) / 65535.0).copy()

    def close(self):
        if self.proc:
            try:
                self.proc.kill()
                self.proc.wait(timeout=2)
            except Exception:
                pass
        self.proc = None


class MovieProvider:
    def __init__(self):
        self._info = {}
        self._readers = {}
        self._lock = threading.Lock()

    def info(self, path):
        with self._lock:
            if path not in self._info:
                self._info[path] = _probe(path)
            return self._info[path]

    def frame(self, path, index):
        inf = self.info(path)
        if inf is None:
            return None
        frames, w, h, fps = inf
        index = max(0, min(frames - 1, index)) if frames else max(0, index)
        with self._lock:
            r = self._readers.get(path)
            if r is None:
                r = self._readers[path] = _Reader(path, w, h, fps)
            return r.read(index)


_provider = None


def install(core):
    """Register the decoder with the C++ core (no-op without FFmpeg)."""
    global _provider
    if not available():
        return False
    _provider = MovieProvider()
    core.set_movie_provider(_provider.info, _provider.frame)
    return True


def movie_info(path):
    if _provider is None:
        return None
    return _provider.info(path)


CODECS = {
    "h264": ["-c:v", "libx264", "-pix_fmt", "yuv420p"],
    "prores": ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"],
    "mjpeg": ["-c:v", "mjpeg", "-pix_fmt", "yuvj444p"],
}


class MovieWriter:
    def __init__(self, path, w, h, fps, codec="h264", quality=90):
        if not available():
            raise RuntimeError("movie writing needs imageio-ffmpeg")
        args = list(CODECS.get(codec, CODECS["h264"]))
        if codec == "h264":
            crf = int(round(51 - quality / 100 * 41))
            args += ["-crf", str(crf), "-preset", "veryfast"]
        elif codec == "mjpeg":
            args += ["-q:v", str(max(2, int(round(31 - quality / 100 * 29))))]
        w2, h2 = w - (w % 2), h - (h % 2)
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le",
               "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-vf", f"crop={w2}:{h2}:0:0"] + args + [path]
        self.w, self.h = w, h
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NO_WINDOW)

    def write(self, rgb: np.ndarray):
        a = np.clip(rgb[..., :3], 0.0, 1.0)
        self.proc.stdin.write((a * 65535.0 + 0.5).astype("<u2").tobytes())

    def close(self):
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace")
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err.strip()}")
