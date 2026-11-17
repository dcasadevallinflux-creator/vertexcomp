// Transform, Crop, Reformat.
#include "common.h"

namespace vc::ops {
namespace {

const std::vector<std::string> kFilters{"impulse", "bilinear", "cubic", "lanczos3"};

struct Affine {
    double a = 1, b = 0, c = 0, d = 0, e = 1, f = 0;  // x' = a x + b y + c ; y' = d x + e y + f
    Affine inverse() const {
        double det = a * e - b * d;
        if (std::fabs(det) < 1e-12) det = 1e-12;
        Affine r;
        r.a = e / det;
        r.b = -b / det;
        r.d = -d / det;
        r.e = a / det;
        r.c = -(r.a * c + r.b * f);
        r.f = -(r.d * c + r.e * f);
        return r;
    }
    void apply(double x, double y, double& ox, double& oy) const {
        ox = a * x + b * y + c;
        oy = d * x + e * y + f;
    }
    Box mapBox(const Box& bx, int pad) const {
        if (bx.empty()) return bx;
        double xs[4] = {double(bx.x0), double(bx.x1), double(bx.x0), double(bx.x1)};
        double ys[4] = {double(bx.y0), double(bx.y0), double(bx.y1), double(bx.y1)};
        double x0 = 1e18, y0 = 1e18, x1 = -1e18, y1 = -1e18;
        for (int i = 0; i < 4; ++i) {
            double X, Y;
            apply(xs[i], ys[i], X, Y);
            x0 = std::min(x0, X); x1 = std::max(x1, X);
            y0 = std::min(y0, Y); y1 = std::max(y1, Y);
        }
        auto cl = [](double v) { return std::max(-1e7, std::min(1e7, v)); };
        return Box{int(std::floor(cl(x0))) - pad, int(std::floor(cl(y0))) - pad, int(std::ceil(cl(x1))) + pad,
                   int(std::ceil(cl(y1))) + pad};
    }
    double minScale() const {
        double sx = std::hypot(a, d), sy = std::hypot(b, e);
        return std::min(sx, sy);
    }
};

Affine transformMatrix(const Params& p) {
    double tx = p.f("translate", 0), ty = p.f("translate", 1);
    double r = -p.f("rotate") * kPi / 180.0;  // y-down: negate so positive turns counter-clockwise on screen
    double sx = p.f("scale", 0), sy = p.f("scale", 1);
    double kx = p.f("skew", 0), ky = p.f("skew", 1);
    double cx = p.f("center", 0), cy = p.f("center", 1);
    double cr = std::cos(r), sr = std::sin(r);
    // M = T(c + t) R K S T(-c)
    double m00 = sx, m01 = kx * sy, m10 = ky * sx, m11 = sy;   // K S
    double a = cr * m00 - sr * m10, b = cr * m01 - sr * m11;
    double d = sr * m00 + cr * m10, e = sr * m01 + cr * m11;
    Affine m;
    m.a = a; m.b = b; m.d = d; m.e = e;
    m.c = cx + tx - (a * cx + b * cy);
    m.f = cy + ty - (d * cx + e * cy);
    if (p.b("invert")) m = m.inverse();
    return m;
}

// Resample `src` through inverse matrices (one per motion sample) into out.
void resample(Eval& ev, const Image& src, Image& out, const ChannelList& chans, const std::vector<Affine>& inv,
              int filter, int ss) {
    const Box& roi = out.bbox();
    for (auto& c : chans) {
        const float* sp = src.plane(c);
        float* op = out.plane(c);
        if (!op) continue;
        if (!sp) {
            std::fill(op, op + size_t(roi.w()) * roi.h(), 0.0f);
            continue;
        }
        ev.parallelRows(roi, [&](int y) {
            float* row = op + size_t(y - roi.y0) * roi.w();
            for (int x = roi.x0; x < roi.x1; ++x) {
                double acc = 0;
                int n = 0;
                for (auto& m : inv)
                    for (int j = 0; j < ss; ++j)
                        for (int i = 0; i < ss; ++i) {
                            double px = x + (i + 0.5) / ss, py = y + (j + 0.5) / ss, sx, sy;
                            m.apply(px, py, sx, sy);
                            acc += sample(sp, src.bbox(), sx, sy, filter);
                            n++;
                        }
                row[x - roi.x0] = float(acc / n);
            }
        });
    }
}

class Transform : public Op {
public:
    void hashExtra(Hasher& h, const Node& n, double frame) const override {
        Params p = n.params(frame);
        int samples = std::max(1, p.i("motionblur"));
        if (samples <= 1) return;
        for (int k = 0; k < samples; ++k) {
            double f = frame + p.f("shutter") * (double(k) / (samples - 1) - 0.5);
            Affine m = transformMatrix(n.params(f));
            h.add(m.a).add(m.b).add(m.c).add(m.d).add(m.e).add(m.f);
        }
    }
    std::vector<Affine> matrices(Eval& ev, const Params& p) const {
        int samples = std::max(1, p.i("motionblur"));
        std::vector<Affine> ms;
        if (samples <= 1) return {transformMatrix(p)};
        for (int k = 0; k < samples; ++k) {
            double f = ev.frame + p.f("shutter") * (double(k) / (samples - 1) - 0.5);
            ms.push_back(transformMatrix(ev.paramsAt(f)));
        }
        return ms;
    }
    Info info(Eval& ev, const Params& p) const override {
        if (!ev.connected(0)) return emptyInfo();
        Info i = ev.inputInfo(0);
        int pad = filterRadius(int(std::find(kFilters.begin(), kFilters.end(), p.s("filter")) - kFilters.begin())) + 1;
        Box b;
        for (auto& m : matrices(ev, p)) b = b.unite(m.mapBox(i.bbox, pad));
        i.bbox = b.intersect(i.format.box().grow(i.format.w * 4, i.format.h * 4));
        return i;
    }
    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        int filter = int(std::find(kFilters.begin(), kFilters.end(), p.s("filter")) - kFilters.begin());
        auto ms = matrices(ev, p);
        std::vector<Affine> inv;
        Box need;
        double minScale = 1e9;
        for (auto& m : ms) {
            inv.push_back(m.inverse());
            need = need.unite(inv.back().mapBox(roi, filterRadius(filter) + 2));
            minScale = std::min(minScale, m.minScale());
        }
        Info inf = ev.inputInfo(0);
        need = need.intersect(inf.bbox);
        ImagePtr in = ev.input(0, need, chans);
        auto out = std::make_shared<Image>(roi, chans);
        int ss = minScale < 1.0 ? std::min(6, int(std::ceil(1.0 / std::max(1e-3, minScale)))) : 1;
        if (ms.size() > 1) ss = 1;
        resample(ev, *in, *out, chans, inv, filter, ss);
        return out;
    }
};
VC_REGISTER_OP(OpDesc{"Transform", "Transform",
                      "Translate, rotate, scale and skew around a centre, with filtering and motion blur.", {""}, 0,
                      {K::xy("translate", "translate", 0, 0), K::flt("rotate", "rotate", 0, -180, 180, "degrees"),
                       K::wh("scale", "scale", 1, 1, 0, 4), K::xy("skew", "skew", 0, 0),
                       K::xy("center", "center", 960, 540), K::enm("filter", "filter", kFilters, "cubic"),
                       K::boolean("invert", "invert", false),
                       K::integer("motionblur", "motion blur samples", 1, 1, 32, "1 = off"),
                       K::flt("shutter", "shutter", 0.5, 0, 2, "frames")},
                      1, maker<Transform>()});

// ------------------------------------------------------------ Crop
class Crop : public Op {
public:
    static Box cropBox(const Params& p) {
        return Box{int(std::floor(p.f("box", 0))), int(std::floor(p.f("box", 1))), int(std::ceil(p.f("box", 2))),
                   int(std::ceil(p.f("box", 3)))};
    }
    Info info(Eval& ev, const Params& p) const override {
        if (!ev.connected(0)) return emptyInfo();
        Info i = ev.inputInfo(0);
        Box cb = cropBox(p);
        i.bbox = i.bbox.intersect(cb);
        if (p.b("reformat")) {
            i.format.w = std::max(1, cb.w());
            i.format.h = std::max(1, cb.h());
            i.bbox = Box{i.bbox.x0 - cb.x0, i.bbox.y0 - cb.y0, i.bbox.x1 - cb.x0, i.bbox.y1 - cb.y0};
            if (i.bbox.empty()) i.bbox = Box{};
        }
        return i;
    }
    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        Box cb = cropBox(p);
        int ox = p.b("reformat") ? cb.x0 : 0, oy = p.b("reformat") ? cb.y0 : 0;
        Box src{roi.x0 + ox, roi.y0 + oy, roi.x1 + ox, roi.y1 + oy};
        ImagePtr in = ev.input(0, src, chans);
        auto out = std::make_shared<Image>(roi, chans);
        double soft = std::max(0.0, p.f("softness"));
        double x0 = p.f("box", 0), y0 = p.f("box", 1), x1 = p.f("box", 2), y1 = p.f("box", 3);
        auto cov = [soft](double lo, double hi, double c) {
            double w = soft + 1.0;
            return std::min(1.0, std::max(0.0, (c - lo) / w + 0.5)) * std::min(1.0, std::max(0.0, (hi - c) / w + 0.5));
        };
        for (auto& c : chans)
            ev.parallelRows(roi, [&](int y) {
                float* o = out->row(c, y);
                in->readRow(c, y + oy, roi.x0 + ox, roi.w(), o);
                double cy = cov(y0, y1, y + oy + 0.5);
                for (int x = roi.x0; x < roi.x1; ++x)
                    o[x - roi.x0] = float(o[x - roi.x0] * cov(x0, x1, x + ox + 0.5) * cy);
            });
        return out;
    }
};
VC_REGISTER_OP(OpDesc{"Crop", "Transform", "Cut the image to a box; optionally make the box the new format.", {""}, 0,
                      {K::box("box", "box", {0, 0, 1920, 1080}), K::flt("softness", "softness", 0, 0, 100),
                       K::boolean("reformat", "reformat", false)},
                      1, maker<Crop>()});

// ------------------------------------------------------------ Reformat
class Reformat : public Op {
public:
    static Affine matrix(const Info& in, const Params& p, Format& outFmt) {
        double sx, sy;
        outFmt = in.format;
        if (p.s("type") == "scale") {
            sx = sy = std::max(1e-3, p.f("scale"));
            outFmt.w = std::max(1, int(std::lround(in.format.w * sx)));
            outFmt.h = std::max(1, int(std::lround(in.format.h * sy)));
        } else {
            outFmt.w = std::max(1, int(p.f("format", 0)));
            outFmt.h = std::max(1, int(p.f("format", 1)));
            double rx = double(outFmt.w) / in.format.w, ry = double(outFmt.h) / in.format.h;
            std::string rt = p.s("resize");
            if (rt == "none") sx = sy = 1;
            else if (rt == "width") sx = sy = rx;
            else if (rt == "height") sx = sy = ry;
            else if (rt == "fill") sx = sy = std::max(rx, ry);
            else if (rt == "distort") { sx = rx; sy = ry; }
            else sx = sy = std::min(rx, ry);  // fit
        }
        Affine m;
        m.a = sx;
        m.e = sy;
        if (p.b("center")) {
            m.c = (outFmt.w - in.format.w * sx) * 0.5;
            m.f = (outFmt.h - in.format.h * sy) * 0.5;
        }
        return m;
    }
    Info info(Eval& ev, const Params& p) const override {
        if (!ev.connected(0)) return emptyInfo();
        Info i = ev.inputInfo(0);
        Format f;
        Affine m = matrix(i, p, f);
        int filter = int(std::find(kFilters.begin(), kFilters.end(), p.s("filter")) - kFilters.begin());
        i.bbox = m.mapBox(i.bbox, filterRadius(filter));
        if (p.b("crop")) i.bbox = i.bbox.intersect(f.box());
        i.format = f;
        return i;
    }
    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        Info inf = ev.inputInfo(0);
        Format f;
        Affine m = matrix(inf, p, f);
        int filter = int(std::find(kFilters.begin(), kFilters.end(), p.s("filter")) - kFilters.begin());
        Affine inv = m.inverse();
        ImagePtr in = ev.input(0, inv.mapBox(roi, filterRadius(filter) + 2).intersect(inf.bbox), chans);
        auto out = std::make_shared<Image>(roi, chans);
        double ms = m.minScale();
        int ss = ms < 1.0 ? std::min(8, int(std::ceil(1.0 / std::max(1e-3, ms)))) : 1;
        resample(ev, *in, *out, chans, {inv}, filter, ss);
        return out;
    }
};
VC_REGISTER_OP(OpDesc{"Reformat", "Transform", "Resize to another format or by a scale factor.", {""}, 0,
                      {K::enm("type", "type", {"to format", "scale"}), K::wh("format", "format", 1920, 1080, 1, 8192),
                       K::flt("scale", "scale", 0.5, 0.05, 4),
                       K::enm("resize", "resize type", {"fit", "fill", "width", "height", "distort", "none"}),
                       K::boolean("center", "center", true), K::enm("filter", "filter", kFilters, "cubic"),
                       K::boolean("crop", "black outside", true)},
                      1, maker<Reformat>()});

}  // namespace
}  // namespace vc::ops
