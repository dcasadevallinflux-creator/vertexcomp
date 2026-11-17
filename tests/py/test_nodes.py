"""Every node type renders on a standard input; results are compared with
stored golden statistics (block means, tolerant to libm differences).

Regenerate after an intentional change:  VC_UPDATE_GOLDEN=1 pytest tests/py/test_nodes.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from vertexcomp import _vccore as vc

GOLDEN = Path(__file__).resolve().parent.parent / "data" / "golden.json"
UPDATE = os.environ.get("VC_UPDATE_GOLDEN") == "1"
RGBA = ["rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"]
W, H = 64, 48

# Knob tweaks so each node does something visible.
TWEAKS = {
    "Grade": {"gain": [1.5, 1.2, 0.8, 1.0], "gamma": [1.2, 1.2, 1.2, 1.0]},
    "ColorCorrect": {"saturation": 0.5, "contrast": 1.3},
    "Exposure": {"stops": [1.0, 1.0, 1.0]},
    "HueShift": {"hue": 45.0},
    "HueCorrect": {"sat": [0.5, 1, 1.5, 1, 1, 1]},
    "Saturation": {"saturation": 0.3},
    "Multiply": {"value": [0.5, 0.6, 0.7, 1.0]},
    "Add": {"value": [0.1, 0.0, -0.1, 0.0]},
    "Gamma": {"value": [2.0, 2.0, 2.0, 1.0]},
    "Clamp": {"maximum": [0.5, 0.5, 0.5, 1.0]},
    "Expression": {"expr0": "r * 0.5 + x / width * 0.1", "expr3": "1"},
    "Colorspace": {"out": "sRGB"},
    "Blur": {"size": [3.0, 2.0]},
    "MaskBlur": {"size": [2.0, 2.0]},
    "GaussianBlur": {"sigma": [2.0, 2.0]},
    "Sharpen": {"amount": 0.8},
    "Median": {"size": 2},
    "DilateErode": {"size": [2.0, 2.0]},
    "EdgeBlur": {"size": 2.0},
    "Keyer": {"operation": "luminance", "range": [0.2, 0.4, 0.8, 1.0]},
    "Despill": {"amount": 1.0},
    "Transform": {"rotate": 10.0, "scale": [0.8, 0.8], "center": [32.0, 24.0]},
    "Crop": {"box": [8, 8, 40, 30], "softness": 2.0},
    "Reformat": {"type": "scale", "scale": 0.5},
    "Shuffle": {"red": "rgba.blue", "blue": "rgba.red"},
    "Copy": {"from0": "rgba.red", "to0": "rgba.alpha"},
    "Remove": {"operation": "keep", "channels": "rgb"},
    "AddChannels": {"channels": "mask.a", "value": 0.5},
    "ChannelSwitch": {"which": 2},
    "MergeLayer": {"layer0": "diffuse", "layer1": "specular", "layer2": ""},
    "Noise": {"mode": "grain"},
    "Rectangle": {"area": [10, 10, 40, 30], "softness": 3.0},
    "Ramp": {"p0": [0.0, 0.0], "p1": [64.0, 48.0]},
    "TimeOffset": {"offset": 1},
    "FrameHold": {"firstFrame": 1},
    "Retime": {"speed": 0.5},
    "Merge": {"operation": "screen"},
    "Switch": {"which": 1},
    "Premult": {},
    "Unpremult": {},
}

SKIP = {"Read", "Write"}  # covered by I/O tests


def block_stats(arr):
    c, h, w = arr.shape
    by, bx = 4, 4
    out = []
    for ch in range(c):
        for j in range(by):
            for i in range(bx):
                blk = arr[ch, j * h // by:(j + 1) * h // by, i * w // bx:(i + 1) * w // bx]
                out.append(float(blk.mean()) if blk.size else 0.0)
    return out


def source(graph):
    n = graph.create("Noise")
    n.set("format", [W, H])
    n.set("size", [12.0, 12.0])
    n.set("color", [1.0, 0.8, 0.6, 0.9])
    n.set("mono", False)
    sh = graph.create("Shuffle")
    sh.set("out", "diffuse")
    graph.connect(sh, 0, n)
    sh2 = graph.create("Shuffle")
    sh2.set("out", "specular")
    sh2.set("red", "rgba.green")
    graph.connect(sh2, 0, sh)
    return sh2


@pytest.fixture(scope="module")
def golden():
    data = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
    yield data
    if UPDATE:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(data, indent=0, sort_keys=True))


@pytest.mark.parametrize("op", sorted(o["type"] for o in vc.op_types() if o["type"] not in SKIP))
def test_node_renders(op, golden):
    g = vc.Graph()
    eng = vc.Engine(g, 256 << 20)
    src = source(g)
    n = g.create(op)
    if "format" in n.knob_names:
        n.set("format", [W, H])
    for k, v in TWEAKS.get(op, {}).items():
        n.set(k, v)
    if n.input_labels:
        g.connect(n, 0, src)
        if len(n.input_labels) > 1 and n.input_labels[1] in ("A", "in2", "1"):
            r = g.create("Rectangle")
            r.set("format", [W, H])
            r.set("area", [16, 12, 48, 36])
            r.set("color", [0.2, 0.5, 0.9, 1.0])
            g.connect(n, 1, r)
    if op == "Roto":
        s = vc.RotoShape()
        s.keys = {1.0: [vc.RotoPoint(10, 10, 0, 0, 10, 0), vc.RotoPoint(50, 12), vc.RotoPoint(40, 40)]}
        s.feather = 4.0
        n.set_roto_shapes([s])
    arr, chans, bbox = eng.render(n, 2.0, (0, 0, W, H), RGBA)
    assert np.isfinite(arr).all(), op
    assert arr.shape[0] > 0, op
    stats = block_stats(arr)
    if UPDATE or op not in golden:
        golden[op] = stats
        if not UPDATE:
            pytest.skip("golden entry created; run again to compare")
    np.testing.assert_allclose(stats, golden[op], atol=2e-3, err_msg=op)
