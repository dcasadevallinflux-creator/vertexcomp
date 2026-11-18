"""AI manager: model downloads, the worker process and its requests.

No Qt here. Background threads only append to an event queue; the UI calls
`pump()` from a timer so every callback runs on the UI thread (the same rule
the rest of the app follows for render workers).
"""
from __future__ import annotations

import collections
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from vertexcomp.ai.catalog import MODELS, ModelSpec

HF = "https://huggingface.co"
WORKER = Path(__file__).with_name("worker.py")


# ---------------------------------------------------------------- paths
def default_models_dir() -> Path:
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "VertexComp" / "models"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "VertexComp" / "models"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "vertexcomp" / "models"


def candidate_pythons() -> list[str]:
    """Interpreters that may have torch + diffusers, best first."""
    out = []
    env = os.environ.get("VERTEXCOMP_AI_PYTHON")
    if env:
        out.append(env)
    # install_ai_windows.bat creates .venv-ai next to the app's .venv.
    for root in {Path(sys.prefix).parent, Path(__file__).resolve().parents[3]}:
        for rel in (".venv-ai/Scripts/python.exe", ".venv-ai/bin/python"):
            p = root / rel
            if p.exists():
                out.append(str(p))
    out.append(sys.executable)
    return out


class AIManager:
    def __init__(self, settings):
        self.settings = settings
        self.events: "collections.deque[tuple]" = collections.deque()
        self.callbacks: dict[int, dict] = {}
        self.downloads: dict[str, dict] = {}   # model id -> {"done", "total", "error", "cancel"}
        self.proc: subprocess.Popen | None = None
        self.env_info: dict = {}
        self.stderr_tail: "collections.deque[str]" = collections.deque(maxlen=60)
        self.listeners = []                    # fn(kind, payload) for UI status (pumped)
        self.cache_clear_hooks = []            # fn() run by clear_caches (engine cache, ...)
        self._next = 1
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._fatal = ""
        self._last_use = time.time()
        self._temp = Path(tempfile.gettempdir()) / "vertexcomp-ai"

    # ------------------------------------------------------------ settings
    def models_dir(self) -> Path:
        d = self.settings.get("ai", "models_dir") or ""
        return Path(d) if d else default_models_dir()

    def python(self) -> str:
        p = self.settings.get("ai", "python") or ""
        return p if p else candidate_pythons()[0]

    def temp_dir(self) -> Path:
        self._temp.mkdir(parents=True, exist_ok=True)
        return self._temp

    # ------------------------------------------------------------ models on disk
    def model_dir(self, mid: str) -> Path:
        return self.models_dir() / mid

    def installed(self, mid: str) -> bool:
        spec = MODELS[mid]
        d = self.model_dir(mid)
        return all((d / f).is_file() for f in spec.files) and (d / ".complete").exists()

    def model_size_on_disk(self, mid: str) -> int:
        d = self.model_dir(mid)
        if not d.exists():
            return 0
        return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())

    def delete_model(self, mid: str):
        shutil.rmtree(self.model_dir(mid), ignore_errors=True)
        self._emit("models", mid)

    # ------------------------------------------------------------ downloads
    def download(self, mid: str, on_done=None):
        """Download a model and its dependencies in a background thread."""
        todo = [d for d in MODELS[mid].requires if not self.installed(d)] + [mid]
        for m in todo:
            if m in self.downloads and not self.downloads[m].get("finished"):
                continue
            st = {"done": 0, "total": MODELS[m].size_mb << 20, "error": "", "cancel": False, "finished": False,
                  "on_done": on_done if m == mid else None}
            self.downloads[m] = st
            threading.Thread(target=self._download, args=(MODELS[m], st), name=f"download-{m}", daemon=True).start()

    def cancel_download(self, mid: str):
        if mid in self.downloads:
            self.downloads[mid]["cancel"] = True

    def _request(self, url, headers=None):
        h = {"User-Agent": "vertexcomp"}
        tok = self.settings.get("ai", "hf_token") or os.environ.get("HF_TOKEN", "")
        if tok:
            h["Authorization"] = f"Bearer {tok}"
        h.update(headers or {})
        return urllib.request.Request(url, headers=h)

    def _file_list(self, spec: ModelSpec) -> list[tuple[str, int]]:
        """(path, size) for the files to fetch. Sizes come from the Hub API when
        reachable (exact progress), otherwise 0."""
        sizes = {}
        try:
            with urllib.request.urlopen(self._request(f"{HF}/api/models/{spec.repo}/tree/main?recursive=true"),
                                        timeout=20) as r:
                for e in json.load(r):
                    if e.get("type") == "file":
                        sizes[e["path"]] = int((e.get("lfs") or {}).get("size") or e.get("size") or 0)
        except (OSError, ValueError):
            pass
        return [(f, sizes.get(f, 0)) for f in spec.files]

    def _download(self, spec: ModelSpec, st: dict):
        dest = self.model_dir(spec.id)
        try:
            dest.mkdir(parents=True, exist_ok=True)
            files = self._file_list(spec)
            known = sum(s for _, s in files)
            if known:
                st["total"] = known
            done_before = 0
            for rel, size in files:
                target = dest / rel
                if target.is_file() and (not size or target.stat().st_size == size):
                    done_before += target.stat().st_size
                    st["done"] = done_before
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                part = target.with_name(target.name + ".part")
                have = part.stat().st_size if part.exists() else 0
                headers = {"Range": f"bytes={have}-"} if have else {}
                url = f"{HF}/{spec.repo}/resolve/main/{rel}?download=true"
                try:
                    r = urllib.request.urlopen(self._request(url, headers), timeout=60)
                except urllib.error.HTTPError as e:
                    if e.code == 416:  # already complete
                        part.replace(target)
                        continue
                    if e.code in (401, 403):
                        raise RuntimeError(f"{spec.repo}: access denied ({e.code}). Accept the model licence on "
                                           f"huggingface.co and set a token in Settings > AI Models.") from e
                    raise RuntimeError(f"{spec.repo}/{rel}: HTTP {e.code}") from e
                mode = "ab" if have and r.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                with r, open(part, mode) as f:
                    while True:
                        if st["cancel"]:
                            raise InterruptedError("cancelled")
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        st["done"] = done_before + have
                part.replace(target)
                done_before += have
            (dest / ".complete").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            st["done"] = st["total"] = max(done_before, 1)
        except InterruptedError:
            st["error"] = "cancelled"
        except Exception as e:
            st["error"] = str(e)
        st["finished"] = True
        self.events.append(("download", spec.id))

    # ------------------------------------------------------------ worker process
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self):
        if self.running():
            return
        self._ready.clear()
        self._fatal = ""
        py = self.python()
        env = dict(os.environ)
        env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1",
                    "HF_HUB_DISABLE_TELEMETRY": "1", "DIFFUSERS_VERBOSITY": "error",
                    "TRANSFORMERS_VERBOSITY": "error",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        env.pop("PYTHONPATH", None)
        flags = subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0
        self.proc = subprocess.Popen([py, "-u", str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, env=env, cwd=str(self.temp_dir()),
                                     creationflags=flags, text=True, encoding="utf-8", errors="replace", bufsize=1)
        threading.Thread(target=self._read_out, args=(self.proc,), daemon=True).start()
        threading.Thread(target=self._read_err, args=(self.proc,), daemon=True).start()
        self._emit("worker", "starting")

    def _read_out(self, proc):
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("event") == "ready":
                self.env_info = msg
                self._ready.set()
            elif msg.get("event") == "fatal":
                self._fatal = msg.get("message", "worker failed")
                self._ready.set()
            self.events.append(("worker", msg))
        code = proc.wait()
        self._ready.set()
        self.events.append(("exit", {"code": code, "proc": proc}))

    def _read_err(self, proc):
        for line in proc.stderr:
            line = line.rstrip()
            if line:
                self.stderr_tail.append(line)

    def _send(self, msg: dict):
        if not self.running():
            self.start()
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except OSError as e:
            raise RuntimeError(f"AI worker is not running: {e}") from e

    def request(self, cmd: str, on_result=None, on_error=None, on_progress=None, on_status=None, **payload) -> int:
        with self._lock:
            rid = self._next
            self._next += 1
        self.callbacks[rid] = {"result": on_result, "error": on_error, "progress": on_progress, "status": on_status}
        self._last_use = time.time()
        self._send({"id": rid, "cmd": cmd, **payload})
        return rid

    def cancel(self, rid: int):
        if self.running():
            self._send({"cmd": "cancel", "target": rid})

    def stop(self, timeout: float = 5.0):
        """Terminate the worker: frees all RAM and VRAM it holds."""
        p = self.proc
        if p is None:
            return
        if p.poll() is None:
            try:
                p.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                p.stdin.flush()
                p.wait(timeout)
            except (OSError, subprocess.TimeoutExpired):
                pass
            if p.poll() is None:
                p.kill()
        self.proc = None
        self._emit("worker", "stopped")

    # ------------------------------------------------------------ high level
    def unload(self):
        """Unload models from RAM and VRAM by ending the worker process (it
        restarts on the next Generate)."""
        pending = list(self.callbacks)
        self.stop()
        for rid in pending:
            cb = self.callbacks.pop(rid, {})
            if cb.get("error"):
                cb["error"]("cancelled: models unloaded")

    def clear_caches(self) -> str:
        n = 0
        for f in list(self.temp_dir().glob("*")):
            try:
                f.unlink()
                n += 1
            except OSError:
                pass
        for fn in list(self.cache_clear_hooks):
            try:
                fn()
            except Exception:
                pass
        if self.running():
            try:
                self.request("clear")
            except RuntimeError:
                pass
        self._emit("cache", n)
        return f"caches cleared ({n} temporary files removed)"

    def check_idle(self):
        minutes = float(self.settings.get("ai", "idle_unload_minutes") or 0)
        if minutes > 0 and self.running() and not self.callbacks and time.time() - self._last_use > minutes * 60:
            self.stop()

    def check_environment(self, python: str | None = None, timeout: float = 120) -> dict:
        """Run the configured interpreter once to report torch/diffusers/GPU."""
        py = python or self.python()
        code = ("import json,sys\nout={'python':sys.version.split()[0]}\n"
                "try:\n import torch;out['torch']=torch.__version__;out['cuda']=torch.cuda.is_available()\n"
                " out['gpu']=torch.cuda.get_device_name(0) if out['cuda'] else ''\n"
                " out['vram_gb']=round(torch.cuda.get_device_properties(0).total_memory/2**30,1) if out['cuda'] else 0\n"
                "except Exception as e: out['torch_error']=str(e)\n"
                "try:\n import diffusers;out['diffusers']=diffusers.__version__\n"
                "except Exception as e: out['diffusers_error']=str(e)\n"
                "try:\n import peft;out['peft']=peft.__version__\nexcept Exception: out['peft']=''\n"
                "print(json.dumps(out))")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0
        try:
            r = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=timeout, creationflags=flags)
            return json.loads(r.stdout.strip().splitlines()[-1])
        except Exception as e:
            return {"error": f"{py}: {e}"}

    # ------------------------------------------------------------ UI-thread dispatch
    def _emit(self, kind, payload):
        self.events.append(("ui", (kind, payload)))

    def pump(self):
        while self.events:
            kind, payload = self.events.popleft()
            if kind == "worker":
                self._dispatch(payload)
            elif kind == "exit":
                if payload["proc"] is self.proc or self.proc is None:
                    tail = "\n".join(list(self.stderr_tail)[-8:])
                    for rid in list(self.callbacks):
                        cb = self.callbacks.pop(rid)
                        if cb.get("error"):
                            msg = self._fatal or f"AI worker exited (code {payload['code']})"
                            cb["error"](msg + ("\n" + tail if tail and not self._fatal else ""))
                    if payload["proc"] is self.proc:
                        self.proc = None
                for fn in list(self.listeners):
                    fn("worker", "stopped")
            elif kind == "download":
                st = self.downloads.get(payload, {})
                cb = st.pop("on_done", None)
                if cb:
                    cb(payload, st.get("error", ""))
                for fn in list(self.listeners):
                    fn("download", payload)
            elif kind == "ui":
                for fn in list(self.listeners):
                    fn(*payload)

    def _dispatch(self, msg: dict):
        ev = msg.get("event")
        if ev in ("ready", "fatal"):
            for fn in list(self.listeners):
                fn("worker", msg)
            if ev == "fatal":
                for rid in list(self.callbacks):
                    cb = self.callbacks.pop(rid)
                    if cb.get("error"):
                        cb["error"](msg.get("message", "AI worker failed"))
            return
        rid = msg.get("id")
        cb = self.callbacks.get(rid)
        if cb is None:
            return
        if ev == "progress" and cb.get("progress"):
            cb["progress"](msg.get("step", 0), msg.get("steps", 1))
        elif ev == "status" and cb.get("status"):
            cb["status"](msg.get("message", ""))
        elif ev == "result":
            self.callbacks.pop(rid, None)
            self._last_use = time.time()
            if cb.get("result"):
                cb["result"](msg)
        elif ev in ("error", "cancelled"):
            self.callbacks.pop(rid, None)
            if cb.get("error"):
                cb["error"]("cancelled" if ev == "cancelled" else msg.get("message", "error"))


_MANAGER: AIManager | None = None


def get_manager(settings=None) -> AIManager:
    global _MANAGER
    if _MANAGER is None:
        if settings is None:
            from vertexcomp.settings import Settings
            settings = Settings()
        _MANAGER = AIManager(settings)
    return _MANAGER
