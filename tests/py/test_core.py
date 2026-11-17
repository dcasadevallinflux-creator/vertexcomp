"""Core numerics against NumPy/SciPy references, I/O round trips, graph behaviour."""
import os

import numpy as np
import pytest
from vertexcomp import _vccore as vc

RGBA = ["rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"]


@pytest.fixture
def g():
    graph = vc.Graph()
    eng = vc.Engine(graph, 512 << 20)
    return graph, eng


def write_exr(path, arr, chans, w, h, bbox=None, data_type="32 bit float", meta=None):
    bbox = bbox or (0, 0, w, h)
    vc.write_image(str(path), np.ascontiguousarray(arr, np.float32), chans, bbox, (w, h), "exr", data_type, "zip", 90,
                   meta or {})


def read_node(graph, path, first=1, last=1):
    r = graph.create("Read")
    r.set("file", str(path))
    r.set("first", first)
    r.set("last", last)
    return r


def full(eng, node, frame=1, chans=RGBA):
    arr, present, bbox = eng.render(node, frame, None, chans)
    return arr, present, bbox


def test_ops_registered():
    types = {o["type"] for o in vc.op_types()}
    for t in ["Read", "Write", "Merge", "Blur", "GaussianBlur", "Grade", "ColorCorrect", "Keyer", "Roto", "Shuffle",
              "Copy", "MergeLayer", "EdgeBlur", "EdgeDetect", "MaskBlur", "DilateErode", "Noise", "Transform",
              "Distortion", "Vignette", "LensFlare", "TimeOffset"]:
        assert t in types


def test_schema_one_merge_inputs_migrate_to_last_mask_slot():
    from vertexcomp.project import Project

    data = {"schema": 1, "nodes": [{"type": "Merge", "inputs": ["B", "A", "mask"]}]}
    migrated = Project.migrate(data)
    assert migrated["schema"] == 2
    assert migrated["nodes"][0]["inputs"] == ["B", "A", None, None, None, None, None, None, "mask"]


def test_constant_and_merge_over(g, tmp_path):
    graph, eng = g
    rng = np.random.default_rng(1)
    w, h = 64, 48
    a = rng.random((4, h, w), dtype=np.float32)
    a[:3] *= a[3]
    b = rng.random((4, h, w), dtype=np.float32)
    b[:3] *= b[3]
    write_exr(tmp_path / "a.exr", a, RGBA, w, h)
    write_exr(tmp_path / "b.exr", b, RGBA, w, h)
    ra, rb = read_node(graph, tmp_path / "a.exr"), read_node(graph, tmp_path / "b.exr")
    m = graph.create("Merge")
    graph.connect(m, 0, rb)
    graph.connect(m, 1, ra)
    out, present, _ = full(eng, m)
    ref = a + b * (1 - a[3])
    assert present == RGBA
    np.testing.assert_allclose(out, ref, atol=1e-6)
    for op, f in [("plus", lambda A, B: A + B), ("multiply", lambda A, B: A * B), ("max", np.maximum),
                  ("difference", lambda A, B: np.abs(A - B))]:
        m.set("operation", op)
        out, _, _ = full(eng, m)
        np.testing.assert_allclose(out, f(a, b), atol=1e-6, err_msg=op)


def test_merge_multiple_foregrounds(g, tmp_path):
    graph, eng = g
    w, h = 4, 3
    images = []
    for i, value in enumerate((0.1, 0.2, 0.3)):
        image = np.full((4, h, w), value, np.float32)
        image[3] = 1.0
        path = tmp_path / f"merge-{i}.exr"
        write_exr(path, image, RGBA, w, h)
        images.append(read_node(graph, path))
    merge = graph.create("Merge")
    for slot, image in enumerate(images):
        graph.connect(merge, slot, image)
    out, present, _ = full(eng, merge)
    assert present == RGBA
    np.testing.assert_allclose(out[:3], 0.3, atol=1e-6)
    np.testing.assert_allclose(out[3], 1.0, atol=1e-6)


def test_new_image_nodes(g, tmp_path):
    graph, eng = g
    w, h = 32, 24
    image = np.zeros((4, h, w), np.float32)
    image[0, :, w // 2:] = 1.0
    image[1] = 0.25
    image[2] = 0.5
    image[3] = 1.0
    path = tmp_path / "looks.exr"
    write_exr(path, image, RGBA, w, h)
    source = read_node(graph, path)
    source.set("raw", True)

    distortion = graph.create("Distortion")
    graph.connect(distortion, 0, source)
    distorted, _, _ = full(eng, distortion)
    np.testing.assert_allclose(distorted, image, atol=1e-6)

    edge = graph.create("EdgeDetect")
    graph.connect(edge, 0, source)
    edge.set("channel", "rgba.red")
    edge.set("width", 1)
    thin, _, _ = eng.render(edge, 1, None, ["edge"])
    edge.set("width", 7)
    thick, _, _ = eng.render(edge, 1, None, ["edge"])
    assert np.count_nonzero(thick) > np.count_nonzero(thin)

    vignette = graph.create("Vignette")
    graph.connect(vignette, 0, source)
    vignette.set("center", [0.5, 0.5])
    vignette.set("radius", 0.1)
    vignette.set("softness", 0.2)
    vignette.set("amount", 1.0)
    vignette.set("color", [0.0, 0.0, 0.0])
    result, _, _ = full(eng, vignette)
    assert result[1, h // 2, w // 2] < result[1, 0, 0]

    flare_source = np.zeros((4, h, w), np.float32)
    flare_source[0, h // 2, w // 2] = 1.0
    flare_source[1, h // 2, w // 2] = 1.0
    flare_source[2, h // 2, w // 2] = 1.0
    flare_source[3] = 1.0
    flare_path = tmp_path / "flare.exr"
    write_exr(flare_path, flare_source, RGBA, w, h)
    flare_read = read_node(graph, flare_path)
    flare_read.set("raw", True)
    flare = graph.create("LensFlare")
    graph.connect(flare, 0, flare_read)
    flare.set("position", [0.5, 0.5])
    flare.set("threshold", 0.5)
    glow, _, _ = full(eng, flare)
    assert glow[0, h // 2, w // 2 + 2] > 0.0


def test_gaussian_matches_scipy(g, tmp_path):
    from scipy.ndimage import gaussian_filter
    graph, eng = g
    w, h = 80, 60
    img = np.zeros((4, h, w), np.float32)
    img[:, 20:40, 30:50] = 1.0
    img[0, 5, 5] = 4.0
    write_exr(tmp_path / "s.exr", img, RGBA, w, h)
    r = read_node(graph, tmp_path / "s.exr")
    b = graph.create("GaussianBlur")
    b.set("sigma", [3.0, 3.0])
    b.set("method", "exact")
    b.set("quality", 4.0)
    graph.connect(b, 0, r)
    out, _, bbox = eng.render(b, 1, (0, 0, w, h), RGBA)
    ref = gaussian_filter(img[0].astype(np.float64), 3.0, mode="constant", truncate=4.0)
    np.testing.assert_allclose(out[0], ref, atol=1e-5)
    # Centre of mass unchanged: no half-pixel shift.
    ys, xs = np.mgrid[0:h, 0:w]
    sq = out[1]
    assert abs((sq * xs).sum() / sq.sum() - 39.5) < 1e-3
    assert abs((sq * ys).sum() / sq.sum() - 29.5) < 1e-3
    # bbox grows by ceil(quality * sigma)
    info = eng.info(b, 1)
    assert info["bbox"] == (-12, -12, w + 12, h + 12)
    # Fast path approximates within a few percent.
    b.set("method", "fast (3 box)")
    fast, _, _ = eng.render(b, 1, (0, 0, w, h), RGBA)
    assert np.abs(fast[0] - ref).max() < 0.03


def test_tiled_equals_full(g):
    graph, eng = g
    n = graph.create("Noise")
    n.set("format", [96, 64])
    blur = graph.create("Blur")
    blur.set("size", [4.0, 2.0])
    graph.connect(blur, 0, n)
    t = graph.create("Transform")
    t.set("rotate", 17.0)
    t.set("center", [48.0, 32.0])
    graph.connect(t, 0, blur)
    fullimg, _, _ = eng.render(t, 1, (0, 0, 96, 64), RGBA)
    eng.clear_cache()
    tiles = np.zeros_like(fullimg)
    for y in range(0, 64, 16):
        for x in range(0, 96, 32):
            part, _, bb = eng.render(t, 1, (x, y, x + 32, y + 16), RGBA)
            tiles[:, y:y + 16, x:x + 32] = part
    np.testing.assert_array_equal(fullimg, tiles)


def test_thread_count_determinism(g):
    graph, eng = g
    n = graph.create("Noise")
    n.set("format", [64, 64])
    n.set("mode", "generate")
    gr = graph.create("Noise")
    gr.set("mode", "grain")
    graph.connect(gr, 0, n)
    vc.set_threads(1)
    a, _, _ = full(eng, gr)
    eng.clear_cache()
    vc.set_threads(0)
    b, _, _ = full(eng, gr)
    np.testing.assert_array_equal(a, b)


def test_exr_roundtrip_multilayer(tmp_path):
    w, h = 40, 30
    chans = RGBA + ["depth.Z", "diffuse.red", "diffuse.green", "diffuse.blue", "N.x", "N.y", "N.z"]
    rng = np.random.default_rng(3)
    data = (rng.standard_normal((len(chans), 20, 25)) * 10).astype(np.float32)
    bbox = (5, 4, 30, 24)
    write_exr(tmp_path / "ml.exr", data, chans, w, h, bbox, meta={"exr/comment": "hello"})
    arr, rch, rbbox, fmt, meta = vc.read_image(str(tmp_path / "ml.exr"))
    assert rbbox == bbox and fmt == (w, h)
    assert sorted(rch) == sorted(chans)
    for i, c in enumerate(chans):
        np.testing.assert_array_equal(arr[rch.index(c)], data[i])
    assert meta.get("exr/comment") == "hello"


def test_png16_and_sequence(tmp_path, g):
    graph, eng = g
    w, h = 16, 8
    for f in (1001, 1002, 1004):
        img = np.full((4, h, w), f / 2000.0, np.float32)
        img[3] = 1
        vc.write_image(str(tmp_path / f"seq.{f}.png"), img, RGBA, (0, 0, w, h), (w, h), "png", "16 bit")
    pat, first, last, isseq = vc.detect_sequence(str(tmp_path / "seq.1002.png"))
    assert isseq and pat.endswith("seq.####.png") and (first, last) == (1001, 1004)
    arr, ch, _, _, _ = vc.read_image(str(tmp_path / "seq.1001.png"))
    np.testing.assert_allclose(arr[0], 1001 / 2000.0, atol=1 / 65535)
    r = read_node(graph, pat, first, last)
    r.set("colorspace", "Linear Rec.709")
    r.set("missing", "nearest")
    out, _, _ = full(eng, r, 1003)
    assert abs(out[0, 0, 0] - 1002 / 2000.0) < 1e-4 or abs(out[0, 0, 0] - 1004 / 2000.0) < 1e-4
    r.set("missing", "error")
    with pytest.raises(RuntimeError):
        full(eng, r, 1003)
    assert "missing frame" in r.error
    r.set("missing", "black")
    out, _, _ = full(eng, r, 1003)
    assert out.size == 0 or np.all(out == 0)


def test_cycle_and_invalidation(g):
    graph, eng = g
    c = graph.create("Constant")
    c.set("format", [8, 8])
    gr = graph.create("Grade")
    graph.connect(gr, 0, c)
    with pytest.raises(RuntimeError):
        graph.connect(c if c.input_labels else gr, 0, gr)
    h1 = eng.hash(gr, 1)
    gr.set("gain", [2, 2, 2, 1])
    assert eng.hash(gr, 1) != h1
    hc = eng.hash(c, 1)
    gr.set("gain", [3, 3, 3, 1])
    assert eng.hash(c, 1) == hc  # upstream untouched


def test_animation_and_expression(g):
    graph, eng = g
    t = graph.create("Transform")
    t.set_key("rotate", 0, 1, 0.0, 1)
    t.set_key("rotate", 0, 11, 100.0, 1)
    assert abs(t.get("rotate", 6)[0] - 50.0) < 1e-9
    t.set_expr("scale", 0, "1 + rotate / 100")
    assert abs(t.get("scale", 11)[0] - 2.0) < 1e-9
    with pytest.raises(RuntimeError):
        t.set_expr("scale", 1, "1 +")
    assert vc.eval_expression("clamp(x*2, 0, 1) + (x > 0.2 ? 10 : 0)", {"x": 0.3}) == pytest.approx(10.6)


def test_time_offset(g, tmp_path):
    graph, eng = g
    for f in range(1, 6):
        vc.write_image(str(tmp_path / f"t.{f:04d}.exr"), np.full((4, 2, 2), f, np.float32), RGBA, (0, 0, 2, 2), (2, 2))
    r = read_node(graph, tmp_path / "t.####.exr", 1, 5)
    to = graph.create("TimeOffset")
    to.set("offset", 2)
    graph.connect(to, 0, r)
    out, _, _ = full(eng, to, 4)
    assert out[0, 0, 0] == 2
    fh = graph.create("FrameHold")
    fh.set("firstFrame", 3)
    graph.connect(fh, 0, r)
    assert full(eng, fh, 5)[0][0, 0, 0] == 3


def test_roto_coverage(g):
    graph, eng = g
    ro = graph.create("Roto")
    ro.set("format", [100, 100])
    s = vc.RotoShape()
    pts = [vc.RotoPoint(20, 20), vc.RotoPoint(80, 20), vc.RotoPoint(80, 80), vc.RotoPoint(20, 80)]
    s.keys = {1.0: pts}
    ro.set_roto_shapes([s])
    out, ch, _ = eng.render(ro, 1, (0, 0, 100, 100), ["rgba.alpha"])
    a = out[0]
    assert a.sum() == pytest.approx(3600, abs=1e-3)  # exact area
    assert a[50, 50] == 1 and a[10, 10] == 0
    s.feather = 10
    ro.set_roto_shapes([s])
    out, _, _ = eng.render(ro, 1, (0, 0, 100, 100), ["rgba.alpha"])
    assert 0 < out[0, 50, 85] < 1 and out[0, 50, 50] == 1
    # Animation: second key moves the square.
    s.feather = 0
    s.keys = {1.0: pts, 11.0: [vc.RotoPoint(p.x + 10, p.y) for p in pts]}
    ro.set_roto_shapes([s])
    out, _, _ = eng.render(ro, 6, (0, 0, 100, 100), ["rgba.alpha"])
    assert out[0, 50, 22] == 0 and out[0, 50, 84] == 1


def test_keyer_channel_range(g, tmp_path):
    graph, eng = g
    w, h = 10, 1
    z = np.linspace(0, 1, w, dtype=np.float32)[None, None, :].repeat(1, 1)
    data = np.concatenate([np.zeros((4, h, w), np.float32), z.reshape(1, h, w)])
    write_exr(tmp_path / "z.exr", data, RGBA + ["depth.Z"], w, h)
    r = read_node(graph, tmp_path / "z.exr")
    k = graph.create("Keyer")
    k.set("operation", "channel")
    k.set("source", "depth.Z")
    k.set("range", [0.2, 0.4, 0.6, 0.8])
    graph.connect(k, 0, r)
    out, ch, _ = eng.render(k, 1, (0, 0, w, h), ["rgba.alpha"])
    zz = z.ravel()
    ref = np.clip(np.minimum((zz - 0.2) / 0.2, (0.8 - zz) / 0.2), 0, 1)
    np.testing.assert_allclose(out[0, 0], ref, atol=1e-5)


def test_shuffle_mergelayer_copy(g, tmp_path):
    graph, eng = g
    w, h = 4, 4
    chans = RGBA + ["diffuse.red", "diffuse.green", "diffuse.blue", "specular.red", "specular.green", "specular.blue"]
    data = np.zeros((len(chans), h, w), np.float32)
    data[4:7] = 0.25
    data[7:10] = 0.5
    data[3] = 1
    write_exr(tmp_path / "aov.exr", data, chans, w, h)
    r = read_node(graph, tmp_path / "aov.exr")
    ml = graph.create("MergeLayer")
    ml.set("layer2", "")
    graph.connect(ml, 0, r)
    out, _, _ = full(eng, ml)
    np.testing.assert_allclose(out[:3], 0.75)
    sh = graph.create("Shuffle")
    sh.set("red", "specular.red")
    sh.set("alpha", "1")
    graph.connect(sh, 0, r)
    out, _, _ = full(eng, sh)
    assert out[0, 0, 0] == 0.5 and out[3, 0, 0] == 1
    cp = graph.create("Copy")
    cp.set("from0", "diffuse.red")
    cp.set("to0", "mask.a")
    graph.connect(cp, 0, r)
    graph.connect(cp, 1, r)
    assert "mask.a" in eng.info(cp, 1)["channels"]


def test_error_does_not_crash(g):
    graph, eng = g
    r = graph.create("Read")
    r.set("file", "/nonexistent/file.exr")
    gr = graph.create("Grade")
    graph.connect(gr, 0, r)
    with pytest.raises(RuntimeError) as e:
        full(eng, gr)
    assert "Read1" in str(e.value)
    assert r.error and not gr.error


def test_cancel(g):
    graph, eng = g
    n = graph.create("Noise")
    n.set("format", [512, 512])
    tok = vc.CancelToken()
    tok.cancel()
    with pytest.raises(vc.Cancelled):
        eng.render(n, 1, None, RGBA, tok)


def test_write_node(g, tmp_path):
    graph, eng = g
    c = graph.create("Ramp")
    c.set("format", [32, 16])
    wn = graph.create("Write")
    graph.connect(wn, 0, c)
    wn.set("file", str(tmp_path / "out" / "r.####.exr"))
    assert eng.write_frame(wn, 7)
    arr, ch, bbox, fmt, _ = vc.read_image(str(tmp_path / "out" / "r.0007.exr"))
    assert fmt == (32, 16)
    src, _, _ = full(eng, c, 7)
    np.testing.assert_allclose(arr, src, atol=2e-3)  # half float
    wn.set("fileType", "png")
    wn.set("file", str(tmp_path / "out" / "r.####.png"))
    assert eng.write_frame(wn, 7)
    assert os.path.exists(tmp_path / "out" / "r.0007.png")
