"""AI nodes: blob-backed evaluation, project round trip, Versions, model
downloads (against a local HTTP server) and the Generate flow end to end with
a fake worker process that speaks the real protocol."""
import http.server
import json
import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from vertexcomp import _vccore as vc

RGBA = ["rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"]


def test_blob_ops_evaluate_and_hash():
    g = vc.Graph()
    eng = vc.Engine(g, 64 << 20)
    n = g.create("SDXLLightning")
    assert eng.info(n, 1)["bbox"] == (0, 0, 0, 0)  # nothing generated yet
    a = np.random.default_rng(1).random((16, 24, 4), dtype=np.float32)
    vc.set_image_blob("t-gen", a)
    n.set("image", "t-gen")
    info = eng.info(n, 1)
    assert info["format"][:2] == (24, 16) and info["bbox"] == (0, 0, 24, 16)
    arr, chans, _ = eng.render(n, 1.0, (0, 0, 24, 16), RGBA)
    np.testing.assert_allclose(arr[0], a[..., 0])
    np.testing.assert_allclose(arr[3], a[..., 3])
    n.set("show", "generated frame only")
    n.set("genFrame", 7)
    assert eng.info(n, 6)["bbox"] == (0, 0, 0, 0) and eng.info(n, 7)["bbox"] == (0, 0, 24, 16)
    n.set("show", "frame range")
    n.set("rangeFirst", 3)
    n.set("rangeLast", 5)
    assert [eng.info(n, f)["bbox"][2] for f in range(2, 7)] == [0, 24, 24, 24, 0]
    h = eng.hash(n, 4)
    vc.set_image_blob("t-gen", a * 0.5)  # re-registering invalidates
    assert eng.hash(n, 4) != h
    vc.remove_image_blob("t-gen")


def test_versions_passthrough_and_stored():
    g = vc.Graph()
    eng = vc.Engine(g, 64 << 20)
    c = g.create("Constant")
    c.set("format", [8, 6])
    c.set("color", [0.25, 0.5, 0.75, 1.0])
    v = g.create("Versions")
    g.connect(v, 0, c)
    arr, _, _ = eng.render(v, 1.0, (0, 0, 8, 6), RGBA)
    assert np.allclose(arr[1], 0.5)
    vc.set_image_blob("t-ver", np.full((6, 8, 4), 0.1, np.float32))
    v.set("image", "t-ver")
    assert np.allclose(eng.render(v, 1.0, (0, 0, 8, 6), RGBA)[0][1], 0.5)  # current = 0: live
    v.set("current", 1)
    assert np.allclose(eng.render(v, 1.0, (0, 0, 8, 6), RGBA)[0][1], 0.1)
    assert eng.info(v, 1)["format"][:2] == (8, 6)
    vc.remove_image_blob("t-ver")


def test_png_and_npz_blobs_roundtrip():
    from vertexcomp.ai.blobs import BlobStore, linear_to_png, png_to_linear
    rng = np.random.default_rng(3)
    a = rng.random((10, 12, 4), dtype=np.float32)
    a[..., :3] *= a[..., 3:4]
    back = png_to_linear(linear_to_png(a))
    assert back.shape == a.shape
    assert np.abs(back - a).max() < 0.02
    st = BlobStore()
    bid = st.add_array(a, 5, 7)
    assert vc.has_image_blob(bid)
    d = st.to_dict({bid})
    st.clear()
    assert not vc.has_image_blob(bid)
    st.load_dict(d)
    np.testing.assert_allclose(st.array(bid), a, atol=1e-3)
    st.clear()


def test_project_save_load_keeps_blobs(tmp_path):
    from vertexcomp.project import Project
    p = Project(64 << 20)
    gen = p.create("SDXLGeneration")
    img = np.zeros((4, 4, 4), np.float32)
    img[..., 1] = img[..., 3] = 1.0
    bid = p.blobs.add_array(img)
    gen.set("image", bid)
    path = str(tmp_path / "ai.vcp")
    p.save(path)
    saved = json.loads(Path(path).read_text())
    assert bid in saved["blobs"]
    assert "blobs" not in p.to_dict(include_blobs=False)
    q = Project(64 << 20)
    vc.clear_image_blobs()
    q.load(path)
    n = q.find(gen.name)
    arr, _, _ = q.engine.render(n, 1.0, (0, 0, 4, 4), RGBA)
    assert np.allclose(arr[1], 1.0)


# ---------------------------------------------------------------- downloads
class _Handler(http.server.BaseHTTPRequestHandler):
    files: dict = {}

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        if path.startswith("/api/models/"):
            repo = path[len("/api/models/"):].split("/tree/")[0]
            body = json.dumps([{"type": "file", "path": k.split("/resolve/main/")[1], "size": len(v)}
                               for k, v in self.files.items() if k.startswith(f"/{repo}/")]).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)
            return
        data = self.files.get(path)
        if data is None:
            self.send_response(404)
            self.end_headers()
            return
        start = 0
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            self.send_response(206)
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(data) - start))
        self.end_headers()
        self.wfile.write(data[start:])


def test_model_download_with_resume(tmp_path, monkeypatch):
    from vertexcomp.ai import client
    from vertexcomp.ai.catalog import MODELS
    from vertexcomp.settings import Settings

    payload = os.urandom(300_000)
    spec = MODELS["lightning-4step"]
    _Handler.files = {f"/{spec.repo}/resolve/main/{spec.files[0]}": payload}
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(client, "HF", f"http://127.0.0.1:{srv.server_address[1]}")
    st = Settings(tmp_path / "s.json")
    st.set("ai", "models_dir", str(tmp_path / "models"))
    mgr = client.AIManager(st)
    # A partial earlier download is resumed with a Range request.
    part = tmp_path / "models" / "lightning-4step" / (spec.files[0] + ".part")
    part.parent.mkdir(parents=True)
    part.write_bytes(payload[:1000])
    done = []
    mgr.download("lightning-4step", on_done=lambda m, e: done.append((m, e)))
    for _ in range(200):
        mgr.pump()
        if done:
            break
        threading.Event().wait(0.02)
    srv.shutdown()
    assert done == [("lightning-4step", "")]
    assert mgr.installed("lightning-4step")
    assert (tmp_path / "models" / "lightning-4step" / spec.files[0]).read_bytes() == payload


# ---------------------------------------------------------------- generate flow
FAKE_WORKER = r'''
import json, sys, struct, zlib
def png(w, h, rgb):
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
out = sys.stdout
out.write(json.dumps({"id": 0, "event": "ready", "device": "fake"}) + "\n"); out.flush()
for line in sys.stdin:
    r = json.loads(line)
    if r.get("cmd") == "quit":
        break
    if r.get("cmd") == "generate":
        with open(sys.argv[1], "w") as f:
            json.dump(r, f)
        for i in range(r["steps"]):
            out.write(json.dumps({"id": r["id"], "event": "progress", "step": i + 1, "steps": r["steps"]}) + "\n")
        open(r["out_path"], "wb").write(png(r["width"], r["height"], (255, 128, 0)))
        out.write(json.dumps({"id": r["id"], "event": "result", "path": r["out_path"], "seed": r["seed"],
                              "width": r["width"], "height": r["height"], "seconds": 0.1}) + "\n")
        out.flush()
    else:
        out.write(json.dumps({"id": r.get("id"), "event": "result"}) + "\n"); out.flush()
'''


@pytest.fixture
def session(qtbot, tmp_path, monkeypatch):
    pytest.importorskip("pytestqt")
    from vertexcomp.ai import client
    from vertexcomp.settings import Settings
    from vertexcomp.ui.session import Session

    monkeypatch.setattr(client, "_MANAGER", None)
    st = Settings(tmp_path / "s.json")
    st.set("ai", "models_dir", str(tmp_path / "models"))
    st.set("ai", "python", sys.executable)
    s = Session(st)
    s.project.blobs.clear()
    yield s
    client.get_manager().stop(1)


def _install_fake_models(mgr, ids):
    from vertexcomp.ai.catalog import MODELS
    for mid in ids:
        d = mgr.model_dir(mid)
        for f in MODELS[mid].files:
            (d / f).parent.mkdir(parents=True, exist_ok=True)
            (d / f).write_bytes(b"x")
        (d / ".complete").write_text("test")


def test_generate_end_to_end_with_fake_worker(session, qtbot, tmp_path, monkeypatch):
    from vertexcomp.ai import client
    from vertexcomp.ai.nodes import controller, generate, is_stale, missing_models

    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER)
    req_dump = tmp_path / "req.json"
    monkeypatch.setattr(client, "WORKER", script)
    real_popen = client.subprocess.Popen

    def popen(cmd, **kw):
        return real_popen(cmd + [str(req_dump)], **kw)

    monkeypatch.setattr(client.subprocess, "Popen", popen)
    s = session
    n = s.create_node("SDXLGeneration")
    s.set_knob(n, "prompt", "a red fox, flat colours")
    s.set_knob(n, "size", [64, 48])
    c = controller(s)
    assert set(missing_models(c.mgr, n)) == {"sdxl-base", "sdxl-vae-fp16-fix", "lightning-8step"}
    _install_fake_models(c.mgr, ["sdxl-base", "sdxl-vae-fp16-fix", "lightning-8step"])
    s.set_frame(12)
    generate(s, n)
    qtbot.waitUntil(lambda: bool(n.get("image")) and n.name not in c.jobs, timeout=15000)
    req = json.loads(req_dump.read_text())
    assert req["steps"] == 8 and req["sampler"] == "euler" and req["spacing"] == "trailing"
    assert req["lora"]["path"].endswith("sdxl_lightning_8step_lora.safetensors")
    assert req["width"] == 64 and req["memory_mode"].startswith("low vram")
    assert int(n.get("genFrame")[0]) == 12
    arr, chans, _ = s.project.engine.render(n, 12.0, (0, 0, 64, 48), RGBA)
    assert abs(arr[0].mean() - 1.0) < 1e-3 and abs(arr[1].mean() - 0.2158605) < 1e-3  # sRGB 128 -> linear
    # Editing parameters keeps the cached image (no regeneration) and flags it stale.
    h = s.project.engine.hash(n, 12.0)
    s.set_knob(n, "prompt", "a blue fox")
    assert is_stale(n, 12) and n.get("image")
    assert s.project.engine.render(n, 12.0, (0, 0, 64, 48), RGBA)[0][0].mean() > 0.99
    assert s.project.engine.hash(n, 12.0) != h  # knob edit, same image
    # Undo restores the previous (empty) state in one step per action.
    s.undo.undo()
    s.undo.undo()
    assert not n.get("image")
    s.undo.redo()
    assert n.get("image")


def test_speed_preset_applies_recommended_sampling(session):
    s = session
    n = s.create_node("SDXLGeneration")
    s.set_knob(n, "accel", "none (full steps)")
    assert int(n.get("steps")[0]) == 28 and n.get("sampler") == "dpm++_2m_karras" and n.get("cfg")[0] > 5
    s.set_knob(n, "accel", "DMD2 4-step")
    assert int(n.get("steps")[0]) == 4 and n.get("sampler") == "lcm"
    s.undo.undo()  # one undo reverts knob + preset
    assert n.get("accel") == "none (full steps)" and int(n.get("steps")[0]) == 28


def test_versions_add_switch_delete_and_save(session, tmp_path):
    from vertexcomp.ai.nodes import add_version, delete_version, set_version, versions_of
    s = session
    c = s.create_node("Constant")
    c.set("format", [6, 4])
    s.set_knob(c, "color", [1.0, 0.0, 0.0, 1.0])
    v = s.create_node("Versions")
    s.project.graph.connect(v, 0, c)
    assert add_version(s, v) == 1
    s.set_knob(c, "color", [0.0, 1.0, 0.0, 1.0])
    assert add_version(s, v) == 2
    eng = s.project.engine
    red = lambda: eng.render(v, 1.0, (0, 0, 6, 4), RGBA)[0][0].mean()  # noqa: E731
    assert red() == 0.0
    set_version(s, v, 1)
    assert red() == 1.0
    set_version(s, v, 0)  # live input (green)
    assert red() == 0.0
    set_version(s, v, 1)
    path = str(tmp_path / "v.vcp")
    s.project.save(path)
    s.project.load(path)
    v2 = s.project.find(v.name)
    assert len(versions_of(v2)) == 2 and eng.render(v2, 1.0, (0, 0, 6, 4), RGBA)[0][0].mean() == 1.0
    delete_version(s, v2, 1)
    assert len(versions_of(v2)) == 1 and int(v2.get("current")[0]) == 1
    assert eng.render(v2, 1.0, (0, 0, 6, 4), RGBA)[0][0].mean() == 0.0  # remaining version is green


def test_version_of_generation_reuses_blob(session):
    from vertexcomp.ai.nodes import add_version, versions_of
    s = session
    g = s.create_node("SDXLTurbo")
    bid = s.project.blobs.add_array(np.ones((4, 4, 4), np.float32))
    s.set_knob(g, "image", bid)
    v = s.create_node("Versions")
    s.project.graph.connect(v, 0, g)
    add_version(s, v)
    assert versions_of(v)[0]["id"] == bid and len(s.project.blobs.items) == 1


def test_clear_caches_prunes_unused_blobs(session):
    from vertexcomp.ai.nodes import controller
    s = session
    g = s.create_node("SDXLTurbo")
    keep = s.project.blobs.add_array(np.ones((2, 2, 4), np.float32))
    drop = s.project.blobs.add_array(np.zeros((2, 2, 4), np.float32))
    s.set_knob(g, "image", keep)
    controller(s).mgr.clear_caches()
    assert keep in s.project.blobs.items and drop not in s.project.blobs.items
