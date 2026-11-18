"""Vertex Comp AI worker: runs SDXL pipelines in a separate process.

Started by vertexcomp.ai.client with the interpreter configured under
Settings > AI Models (it needs torch + diffusers; the app itself does not).
This file is standalone: it imports nothing from vertexcomp, so it also runs
from a different virtual environment (e.g. Python 3.12 with CUDA torch while
the app runs on a newer Python).

Protocol: one JSON object per line. Requests on stdin:
  {"id": 1, "cmd": "generate", ...}    {"id": 2, "cmd": "cancel", "target": 1}
  {"id": 3, "cmd": "unload"}           {"id": 4, "cmd": "clear"}
  {"id": 5, "cmd": "status"}           {"id": 6, "cmd": "quit"}
Events on stdout (the real stdout is reserved for the protocol; anything
libraries print goes to stderr):
  {"id": 1, "event": "progress", "step": 3, "steps": 8}
  {"id": 1, "event": "result", ...}    {"id": 1, "event": "error", "message": "..."}
"""
from __future__ import annotations

import gc
import json
import os
import queue
import sys
import threading
import time
import traceback

# Reserve the real stdout for protocol messages; route stray prints to stderr.
_PROTO = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
sys.stdout = sys.stderr
_SEND_LOCK = threading.Lock()


def send(msg: dict):
    with _SEND_LOCK:
        _PROTO.write(json.dumps(msg) + "\n")
        _PROTO.flush()


class Cancelled(Exception):
    pass


# ---------------------------------------------------------------- torch setup
torch = None
diffusers = None


def _imports():
    global torch, diffusers
    if torch is None:
        import diffusers as _diffusers  # noqa: PLC0415
        import torch as _torch  # noqa: PLC0415
        torch, diffusers = _torch, _diffusers
        try:
            diffusers.utils.logging.disable_progress_bar()
        except Exception:
            pass


def dtype_kw(dtype) -> dict:
    """`dtype=` on new diffusers (where `torch_dtype=` is deprecated), `torch_dtype=` on older ones."""
    try:
        from diffusers.utils import deprecation_utils  # noqa: PLC0415
        if hasattr(deprecation_utils, "_resolve_dtype"):
            return {"dtype": dtype}
    except ImportError:
        pass
    return {"torch_dtype": dtype}


def device_name() -> str:
    _imports()
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def env_report() -> dict:
    _imports()
    out = {"python": sys.version.split()[0], "torch": torch.__version__, "diffusers": diffusers.__version__,
           "device": device_name(), "cuda": torch.version.cuda or ""}
    try:
        import peft  # noqa: F401, PLC0415
        out["peft"] = peft.__version__
    except Exception:
        out["peft"] = ""
    if out["device"] == "cuda":
        p = torch.cuda.get_device_properties(0)
        out["gpu"] = p.name
        out["vram_gb"] = round(p.total_memory / 2**30, 1)
    return out


def memory_report() -> dict:
    out = {}
    if torch is not None and torch.cuda.is_available():
        out["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / 2**30, 2)
        out["vram_reserved_gb"] = round(torch.cuda.memory_reserved() / 2**30, 2)
    return out


# ---------------------------------------------------------------- pipelines
class Pipelines:
    """Keeps at most one base pipeline (plus its img2img twin sharing the same
    components) and the currently loaded speed LoRA."""

    def __init__(self):
        self.key = None
        self.t2i = None
        self.i2i = None
        self.active = None  # which of t2i / i2i currently owns the offload hooks
        self.lora = None    # (path, scale)
        self.mode = None

    def unload(self):
        self.t2i = self.i2i = self.active = None
        self.key = self.lora = self.mode = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    # -------------------------------------------------------------- loading
    def _dtype(self, req):
        if device_name() == "cpu":
            return torch.float32
        return torch.bfloat16 if req.get("precision") == "bf16" else torch.float16

    def ensure(self, req):
        base = req["base"]
        key = (base.get("dir"), base.get("file"), base.get("config"), req.get("vae_dir"), req.get("precision"),
               req.get("memory_mode"))
        if key != self.key:
            self.unload()
            send({"id": req["id"], "event": "status", "message": "loading model..."})
            self._load(req)
            self.key = key
        self._set_lora(req)

    def _load(self, req):
        from diffusers import AutoencoderKL, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline  # noqa: PLC0415

        dtype = self._dtype(req)
        base = req["base"]
        kw = dtype_kw(dtype)
        if req.get("vae_dir"):
            kw["vae"] = AutoencoderKL.from_pretrained(req["vae_dir"], **dtype_kw(dtype))
        if base.get("file"):
            if base.get("config"):
                kw["config"] = base["config"]
            pipe = StableDiffusionXLPipeline.from_single_file(base["file"], **kw)
        else:
            # Only the fp16 weights are downloaded; on CPU they are upcast below.
            variant = base.get("variant") or None
            pipe = StableDiffusionXLPipeline.from_pretrained(base["dir"], variant=variant, use_safetensors=True, **kw)
            if dtype == torch.float32:
                pipe = pipe.to(dtype=torch.float32)
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        self.t2i = pipe
        self.i2i = StableDiffusionXLImg2ImgPipeline(**pipe.components)
        try:
            self.i2i.set_progress_bar_config(disable=True)
        except Exception:
            pass
        self.mode = req.get("memory_mode", "low vram (model offload)")
        for p in (self.t2i, self.i2i):
            if req.get("vae_tiling", True):
                try:
                    p.vae.enable_tiling()
                    p.vae.enable_slicing()
                except Exception:
                    pass
        self.active = None

    def _place(self, pipe):
        """Move `pipe` onto the device according to the memory mode. The two
        pipelines share modules, so offload hooks are re-attached on switch."""
        if self.active is pipe:
            return
        dev = device_name()
        mode = self.mode or ""
        if dev == "cuda" and mode.startswith("minimal"):
            pipe.enable_sequential_cpu_offload()
        elif dev == "cuda" and mode.startswith("low"):
            pipe.enable_model_cpu_offload()
        elif dev == "cuda" and mode.startswith("auto"):
            free, total = torch.cuda.mem_get_info()
            if total >= 15 * 2**30 and free >= 11 * 2**30:
                pipe.to("cuda")
            else:
                pipe.enable_model_cpu_offload()
        else:
            pipe.to(dev)
        self.active = pipe

    def _set_lora(self, req):
        lora = req.get("lora")
        want = (lora["path"], float(lora.get("scale", 1.0))) if lora else None
        if want == self.lora:
            return
        pipe = self.t2i
        if self.lora is not None:
            try:
                pipe.unload_lora_weights()
            except Exception:
                traceback.print_exc()
        self.lora = None
        if want is None:
            return
        send({"id": req["id"], "event": "status", "message": "loading speed LoRA..."})
        folder, name = os.path.split(lora["path"])
        pipe.load_lora_weights(folder, weight_name=name, adapter_name="speed")
        try:
            pipe.set_adapters(["speed"], adapter_weights=[want[1]])
        except Exception:
            # Older diffusers without PEFT adapters: fuse instead.
            pipe.fuse_lora(lora_scale=want[1])
        self.lora = want


PIPES = Pipelines()


def make_scheduler(name: str, spacing: str | None, config):
    from diffusers import (  # noqa: PLC0415
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        LCMScheduler,
        TCDScheduler,
        UniPCMultistepScheduler,
    )

    kw = {}
    if spacing:
        kw["timestep_spacing"] = spacing
    if name == "euler":
        return EulerDiscreteScheduler.from_config(config, **kw)
    if name == "euler_a":
        return EulerAncestralDiscreteScheduler.from_config(config, **kw)
    if name == "dpm++_2m":
        return DPMSolverMultistepScheduler.from_config(config, **kw)
    if name == "dpm++_2m_karras":
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True, **kw)
    if name == "dpm++_sde_karras":
        return DPMSolverMultistepScheduler.from_config(config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True, **kw)
    if name == "ddim":
        return DDIMScheduler.from_config(config, **kw)
    if name == "lcm":
        return LCMScheduler.from_config(config, **kw)
    if name == "tcd":
        return TCDScheduler.from_config(config, **kw)
    if name == "unipc":
        return UniPCMultistepScheduler.from_config(config, **kw)
    raise ValueError(f"unknown sampler '{name}'")


def slerp(t: float, a, b):
    af, bf = a.flatten(), b.flatten()
    dot = (af / af.norm() * (bf / bf.norm())).sum().clamp(-1, 1)
    omega = torch.acos(dot)
    if omega.abs() < 1e-4:
        return (1 - t) * a + t * b
    so = torch.sin(omega)
    return (torch.sin((1 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b


# ---------------------------------------------------------------- generation
CANCEL: set = set()


def generate(req: dict):
    _imports()
    t0 = time.perf_counter()
    PIPES.ensure(req)
    width = max(64, int(req["width"]) // 8 * 8)
    height = max(64, int(req["height"]) // 8 * 8)
    steps = max(1, int(req["steps"]))
    cfg = float(req["cfg"])
    seed = int(req["seed"]) & 0x7FFFFFFF
    init = req.get("init_image")
    pipe = PIPES.i2i if init else PIPES.t2i
    pipe.scheduler = make_scheduler(req["sampler"], req.get("spacing"), PIPES.t2i.scheduler.config)
    if pipe is PIPES.i2i:
        PIPES.t2i.scheduler = pipe.scheduler
    PIPES._place(pipe)

    rid = req["id"]
    total = [steps]

    def on_step(p, i, t, kwargs):
        if rid in CANCEL:
            p._interrupt = True
        send({"id": rid, "event": "progress", "step": i + 1, "steps": total[0]})
        return kwargs

    gen = torch.Generator("cpu").manual_seed(seed)
    args = dict(prompt=req.get("prompt", ""), num_inference_steps=steps, guidance_scale=cfg,
                generator=gen, callback_on_step_end=on_step, output_type="pil")
    if cfg > 1.0 and req.get("negative"):
        args["negative_prompt"] = req["negative"]
    if init:
        from PIL import Image  # noqa: PLC0415
        img = Image.open(init).convert("RGB").resize((width, height), Image.LANCZOS)
        strength = min(1.0, max(0.01, float(req.get("denoise", 0.6))))
        # Distilled models need at least one effective step.
        if steps * strength < 1.0:
            strength = min(1.0, 1.0 / steps + 1e-3)
        total[0] = max(1, int(steps * strength))
        args.update(image=img, strength=strength)
    else:
        args.update(width=width, height=height)
        shape = (1, 4, height // 8, width // 8)
        lat = torch.randn(shape, generator=torch.Generator("cpu").manual_seed(seed), dtype=torch.float32)
        vs = float(req.get("var_strength", 0.0))
        if vs > 0:
            other = torch.randn(shape, generator=torch.Generator("cpu").manual_seed(int(req.get("var_seed", 1)) & 0x7FFFFFFF),
                                dtype=torch.float32)
            lat = slerp(vs, lat, other)
        args["latents"] = lat.to(dtype=pipe.unet.dtype)
    send({"id": rid, "event": "status", "message": "sampling..."})
    if device_name() == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        out = pipe(**args)
    if rid in CANCEL:
        raise Cancelled()
    image = out.images[0]
    os.makedirs(os.path.dirname(req["out_path"]), exist_ok=True)
    image.save(req["out_path"])
    res = {"id": rid, "event": "result", "path": req["out_path"], "seed": seed, "width": image.width,
           "height": image.height, "seconds": round(time.perf_counter() - t0, 2)}
    if device_name() == "cuda":
        res["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    send(res)


# ---------------------------------------------------------------- main loop
def gen_thread(q: "queue.Queue[dict]"):
    while True:
        req = q.get()
        if req is None:
            return
        rid = req.get("id")
        cmd = req.get("cmd")
        try:
            if cmd == "generate":
                if rid in CANCEL:
                    raise Cancelled()
                generate(req)
            elif cmd == "unload":
                PIPES.unload()
                send({"id": rid, "event": "result", **memory_report()})
            elif cmd == "clear":
                gc.collect()
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
                send({"id": rid, "event": "result", **memory_report()})
            elif cmd == "status":
                send({"id": rid, "event": "result", **env_report(), **memory_report(),
                      "loaded": bool(PIPES.t2i), "lora": PIPES.lora[0] if PIPES.lora else ""})
            else:
                send({"id": rid, "event": "error", "message": f"unknown command {cmd}"})
        except Cancelled:
            send({"id": rid, "event": "cancelled"})
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if torch is not None and isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())):
                msg += " - try memory mode 'minimal (sequential offload)' or a smaller size"
                PIPES.unload()
            traceback.print_exc()
            send({"id": rid, "event": "error", "message": msg})
        finally:
            CANCEL.discard(rid)


def main():
    q: "queue.Queue[dict]" = queue.Queue()
    t = threading.Thread(target=gen_thread, args=(q,), daemon=True)
    t.start()
    try:
        _imports()
        send({"id": 0, "event": "ready", **env_report()})
    except Exception as e:
        send({"id": 0, "event": "fatal", "message": f"{type(e).__name__}: {e}. Install torch and diffusers in the "
                                                     f"AI Python environment (see install_ai_windows.bat)."})
        return 2
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        cmd = req.get("cmd")
        if cmd == "quit":
            break
        if cmd == "cancel":
            CANCEL.add(req.get("target"))
            continue
        q.put(req)
    q.put(None)
    PIPES.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
