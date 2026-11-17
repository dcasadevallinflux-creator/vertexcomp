"""Create demo media and a project that exercises the main pipeline:

  multilayer EXR sequence (rgba, diffuse, specular, depth.Z, N) and an MP4
  -> MergeLayer / Shuffle / ChannelSwitch -> ColorCorrect / Exposure / HueShift
  -> Keyer on depth.Z -> animated Roto mask -> EdgeBlur / MaskBlur / GaussianBlur
  -> Noise (grain) -> Merge -> Write EXR sequence + Write MP4

python tools/make_demo.py [output_dir]      (default: ./demo)
then: vertexcomp demo/demo.vcp   or   vertexcomp-render demo/demo.vcp
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from vertexcomp import _vccore as vc  # noqa: E402
from vertexcomp import movie  # noqa: E402
from vertexcomp.project import Project  # noqa: E402

W, H, FRAMES = 960, 540, 24


def render_layers(f):
    """A lit sphere over a ground plane, as AOVs."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy, r = W * (0.35 + 0.3 * f / FRAMES), H * 0.5, H * 0.28
    dx, dy = (x - cx) / r, (y - cy) / r
    d2 = dx * dx + dy * dy
    inside = d2 < 1.0
    nz = np.sqrt(np.clip(1 - d2, 0, 1))
    light = np.array([-0.5, -0.6, 0.62])
    light /= np.linalg.norm(light)
    ndl = np.clip(dx * light[0] + dy * light[1] + nz * light[2], 0, 1)
    spec = np.clip(ndl, 0, 1) ** 40 * 4.0
    ground = (y > H * 0.62).astype(np.float32)
    diffuse = np.zeros((3, H, W), np.float32)
    albedo = np.array([0.8, 0.25, 0.15])[:, None, None]
    diffuse += albedo * ndl * inside
    diffuse += np.array([0.18, 0.2, 0.22])[:, None, None] * ground * (1 - inside)
    specular = np.repeat((spec * inside)[None], 3, 0).astype(np.float32)
    rgb = diffuse + specular
    alpha = np.clip(inside + ground, 0, 1).astype(np.float32)
    depth = np.where(inside, 5.0 - nz, np.where(ground > 0, 2.0 + 30.0 * (H - y) / H, 1e4)).astype(np.float32)
    n = np.stack([dx * inside, dy * inside, nz * inside]).astype(np.float32)
    chans = ["rgba.red", "rgba.green", "rgba.blue", "rgba.alpha", "diffuse.red", "diffuse.green", "diffuse.blue",
             "specular.red", "specular.green", "specular.blue", "depth.Z", "N.x", "N.y", "N.z"]
    arr = np.concatenate([rgb * alpha, alpha[None], diffuse, specular, depth[None], n]).astype(np.float32)
    return chans, arr


def main():
    out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "demo")
    os.makedirs(os.path.join(out, "render"), exist_ok=True)
    seq = os.path.join(out, "plate", "sphere.####.exr")
    os.makedirs(os.path.dirname(seq), exist_ok=True)
    for f in range(1, FRAMES + 1):
        chans, arr = render_layers(f)
        vc.write_image(vc.frame_path(seq, f), arr, chans, (0, 0, W, H), (W, H), "exr", "16 bit half", "zip", 90,
                       {"exr/comment": f"demo frame {f}"})
    mov = os.path.join(out, "plate", "background.mp4")
    if movie.available():
        wr = movie.MovieWriter(mov, W, H, 24.0, "h264", 90)
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        for f in range(FRAMES):
            r = 0.5 + 0.5 * np.sin(x / 60.0 + f * 0.3)
            g = 0.5 + 0.5 * np.sin(y / 45.0 - f * 0.2)
            b = np.full_like(x, 0.6)
            wr.write(np.dstack([r, g, b]) ** 2.2 * 0.6)
        wr.close()
    movie.install(vc)

    p = Project()
    p.settings.format_w, p.settings.format_h = W, H
    p.settings.first, p.settings.last = 1, FRAMES
    c = p.graph.connect

    rd = p.create("Read", x=0, y=0)
    rd.set("file", seq)
    rd.set("first", 1)
    rd.set("last", FRAMES)
    ml = p.create("MergeLayer", x=0, y=70)
    ml.set("layer2", "")
    c(ml, 0, rd)
    sh = p.create("Shuffle", x=180, y=70)
    sh.set("out", "rgba")
    sh.set("red", "N.x")
    sh.set("green", "N.y")
    sh.set("blue", "N.z")
    c(sh, 0, rd)
    cs = p.create("ChannelSwitch", x=360, y=70)
    cs.set("candidates", "rgba.red, diffuse.red, specular.red, depth.Z")
    cs.set_key("which", 0, 1.0, 0.0, 0)
    cs.set_key("which", 0, 12.0, 3.0, 0)
    c(cs, 0, rd)
    cc = p.create("ColorCorrect", x=0, y=140)
    cc.set("saturation", 1.2)
    cc.set("highlights_gain", 0.8)
    c(cc, 0, ml)
    ex = p.create("Exposure", x=0, y=200)
    ex.set("stops", [0.3, 0.3, 0.3])
    c(ex, 0, cc)
    hs = p.create("HueShift", x=0, y=260)
    hs.set("hue", 8.0)
    c(hs, 0, ex)
    key = p.create("Keyer", x=0, y=320)
    key.set("operation", "channel")
    key.set("source", "depth.Z")
    key.set("range", [0.0, 0.0, 6.0, 12.0])
    key.set("output", "mask.a")
    c(key, 0, hs)
    ro = p.create("Roto", x=200, y=320)
    s = vc.RotoShape()
    s.name = "garbage"
    k = 0.5522847498
    def ellipse(cx, cy, rx, ry):
        return [vc.RotoPoint(cx, cy - ry, -rx * k, 0, rx * k, 0), vc.RotoPoint(cx + rx, cy, 0, -ry * k, 0, ry * k),
                vc.RotoPoint(cx, cy + ry, rx * k, 0, -rx * k, 0), vc.RotoPoint(cx - rx, cy, 0, ry * k, 0, -ry * k)]
    s.keys = {1.0: ellipse(W * 0.38, H * 0.5, 190, 170), float(FRAMES): ellipse(W * 0.62, H * 0.5, 190, 170)}
    s.feather = 30.0
    ro.set_roto_shapes([s])
    ro.set("format", [W, H])
    mk = p.create("Merge", x=0, y=380)
    mk.set("operation", "mask")
    mk.set("Achannels", "alpha")
    mk.set("output", "alpha")
    c(mk, 0, key)
    c(mk, 1, ro)
    eb = p.create("EdgeBlur", x=0, y=440)
    eb.set("size", 3.0)
    c(eb, 0, mk)
    mb = p.create("MaskBlur", x=0, y=500)
    mb.set("size", [1.5, 1.5])
    c(mb, 0, eb)
    gb = p.create("GaussianBlur", x=200, y=500)
    gb.set("sigma", [6.0, 6.0])
    gb.set("channels", "specular")
    c(gb, 0, mb)
    gr = p.create("Noise", x=0, y=560)
    gr.set("mode", "grain")
    c(gr, 0, gb)
    bgr = p.create("Read", x=-220, y=500)
    if os.path.exists(mov):
        bgr.set("file", mov)
        bgr.set("first", 0)
        bgr.set("last", FRAMES - 1)
        bgr.set("offset", 1)
    mg = p.create("Merge", x=-60, y=640)
    c(mg, 0, bgr if os.path.exists(mov) else None)
    c(mg, 1, gr)
    w1 = p.create("Write", x=-60, y=720)
    w1.set("file", os.path.join(out, "render", "comp.####.exr"))
    w1.set("channels", "all")
    w1.set("first", 1)
    w1.set("last", FRAMES)
    c(w1, 0, mg)
    w2 = p.create("Write", x=120, y=720)
    w2.set("file", os.path.join(out, "render", "comp.mp4"))
    w2.set("fileType", "mp4")
    w2.set("first", 1)
    w2.set("last", FRAMES)
    c(w2, 0, mg)
    p.viewer = {"slots": {"1": mg.name, "2": rd.name}, "active": 1, "frame": 1}
    path = os.path.join(out, "demo.vcp")
    p.save(path, relative=True)
    print(f"demo project: {path}")


if __name__ == "__main__":
    main()
