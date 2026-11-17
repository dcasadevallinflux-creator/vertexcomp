// Blur, GaussianBlur, MaskBlur, Sharpen, Median, DilateErode, EdgeBlur.
#include "common.h"

namespace vc::ops {
namespace {

const std::vector<std::string> kBlurFilters{"box", "triangle", "quadratic", "gaussian"};

struct BlurSpec {
    double sx, sy, quality;
    int filter;  // 0..3, 4 = box3 gaussian approximation
};

// Shared implementation for all blur-like nodes: blur selected channels,
// grow bbox by the kernel radius, mask/mix.
class BlurBase : public Op {
public:
    virtual BlurSpec spec(const Params& p) const = 0;

    Info info(Eval& ev, const Params& p) const override {
        if (!ev.connected(0)) return emptyInfo();
        Info i = ev.inputInfo(0);
        BlurSpec s = spec(p);
        int rx = blurRadius(s.sx, s.filter, s.quality), ry = blurRadius(s.sy, s.filter, s.quality);
        i.bbox = i.bbox.grow(rx, ry);
        if (p.b("crop")) i.bbox = i.bbox.intersect(i.format.box());
        return i;
    }
    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        Info inf = ev.inputInfo(0);
        BlurSpec s = spec(p);
        int rx = blurRadius(s.sx, s.filter, s.quality), ry = blurRadius(s.sy, s.filter, s.quality);
        ChannelList sel = intersect(resolveChannels(p.s("channels"), inf.channels), chans);
        ImagePtr in = ev.input(0, roi.grow(rx, ry), chans);
        auto out = copyOf(*in, roi, chans);
        for (auto& c : sel) blurPlane(*in, c, roi, out->plane(c), s.sx, s.sy, s.filter, s.quality, ev.cancel);
        applyMaskMix(ev, p, 1, *in, *out, sel);
        return out;
    }
};

class Blur : public BlurBase {
    BlurSpec spec(const Params& p) const override {
        int f = int(std::find(kBlurFilters.begin(), kBlurFilters.end(), p.s("filter")) - kBlurFilters.begin());
        return {std::max(0.0, p.f("size", 0)), std::max(0.0, p.f("size", 1)), std::max(1.0, p.f("quality")), f};
    }
};
std::vector<KnobDef> blurKnobs(const std::string& chans) {
    return concat({K::channels("channels", "channels", chans),
                   K::wh("size", "size", 3, 3, 0, 100, "Radius in pixels; sigma for the gaussian filter"),
                   K::enm("filter", "filter", kBlurFilters, "gaussian"),
                   K::flt("quality", "quality", 3, 1, 5, "Gaussian kernel truncation in sigmas"),
                   K::boolean("crop", "crop to format", false)},
                  maskKnobs());
}
VC_REGISTER_OP(OpDesc{"Blur", "Filter", "Separable box/triangle/quadratic/gaussian blur.", {"", "mask"}, 0,
                      blurKnobs("rgba"), 1, maker<Blur>()});
VC_REGISTER_OP(OpDesc{"MaskBlur", "Keyer", "Blur restricted to the matte channel.", {"", "mask"}, 0,
                      blurKnobs("alpha"), 1, maker<Blur>()});

class GaussianBlur : public BlurBase {
    BlurSpec spec(const Params& p) const override {
        double sx = std::max(0.0, p.f("sigma", 0)), sy = std::max(0.0, p.f("sigma", 1));
        std::string m = p.s("method");
        bool fast = m == "fast (3 box)" || (m == "auto" && std::max(sx, sy) > 16.0);
        return {sx, sy, std::max(1.0, p.f("quality")), fast ? 4 : 3};
    }
};
VC_REGISTER_OP(OpDesc{"GaussianBlur", "Filter",
                      "Gaussian blur by sigma. 'auto' switches to a three-box approximation above sigma 16.",
                      {"", "mask"}, 0,
                      concat({K::channels("channels", "channels", "rgba"),
                              K::wh("sigma", "sigma", 2, 2, 0, 100, "Standard deviation in pixels"),
                              K::enm("method", "method", {"auto", "exact", "fast (3 box)"}),
                              K::flt("quality", "quality", 3, 1, 5, "Exact kernel truncation in sigmas"),
                              K::boolean("crop", "crop to format", false)},
                             maskKnobs()),
                      1, maker<GaussianBlur>()});

// ------------------------------------------------------------ Sharpen
class Sharpen : public Op {
public:
    Info info(Eval& ev, const Params&) const override { return ev.connected(0) ? ev.inputInfo(0) : emptyInfo(); }
    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        Info inf = ev.inputInfo(0);
        double s = std::max(0.0, p.f("size")), amt = p.f("amount");
        int r = blurRadius(s, 3, 3);
        ChannelList sel = intersect(resolveChannels(p.s("channels"), inf.channels), chans);
        ImagePtr in = ev.input(0, roi.grow(r, r), chans);
        auto out = copyOf(*in, roi, chans);
        std::vector<float> bl(size_t(roi.w()) * roi.h());
        for (auto& c : sel) {
            blurPlane(*in, c, roi, bl.data(), s, s, 3, 3, ev.cancel);
            float* o = out->plane(c);
            for (size_t i = 0; i < bl.size(); ++i) o[i] = float(o[i] + amt * (o[i] - bl[i]));
        }
        applyMaskMix(ev, p, 1, *in, *out, sel);
        return out;
    }
};
VC_REGISTER_OP(OpDesc{"Sharpen", "Filter", "Unsharp mask: in + amount * (in - gaussian(in)).", {"", "mask"}, 0,
                      concat({K::channels("channels", "channels", "rgb"), K::flt("amount", "amount", 1, 0, 4),
                              K::flt("size", "size", 2, 0, 20, "Gaussian sigma in pixels")},
                             maskKnobs()),
                      1, maker<Sharpen>()});

// ------------------------------------------------------------ Median
class Median : public Op {
public:
    Info info(Eval& ev, const Params&) const override { return ev.connected(0) ? ev.inputInfo(0) : emptyInfo(); }
    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        Info inf = ev.inputInfo(0);
        int r = std::max(0, std::min(10, p.i("size")));
        ChannelList sel = intersect(resolveChannels(p.s("channels"), inf.channels), chans);
        ImagePtr in = ev.input(0, roi.grow(r, r), chans);
        auto out = copyOf(*in, roi, chans);
        Box pb = roi.grow(r, r);
        auto padded = copyOf(*in, pb, sel);
        for (auto& c : sel) {
            const float* pp = padded->plane(c);
            const int pw = pb.w();
            ev.parallelRows(roi, [&](int y) {
                std::vector<float> win;
                win.reserve(size_t(2 * r + 1) * (2 * r + 1));
                float* o = out->row(c, y);
                for (int x = roi.x0; x < roi.x1; ++x) {
                    win.clear();
                    for (int j = -r; j <= r; ++j) {
                        const float* row = pp + size_t(y + j - pb.y0) * pw + (x - pb.x0);
                        for (int i = -r; i <= r; ++i) win.push_back(row[i]);
                    }
                    std::nth_element(win.begin(), win.begin() + win.size() / 2, win.end());
                    o[x - roi.x0] = win[win.size() / 2];
                }
            });
        }
        applyMaskMix(ev, p, 1, *in, *out, sel);
        return out;
    }
};
VC_REGISTER_OP(OpDesc{"Median", "Filter", "Median of a square window; removes speckle while keeping edges.",
                      {"", "mask"}, 0,
                      concat({K::channels("channels", "channels", "rgba"), K::integer("size", "radius", 1, 0, 10)},
                             maskKnobs()),
                      1, maker<Median>()});

// ------------------------------------------------------------ DilateErode
class DilateErode : public Op {
public:
    Info info(Eval& ev, const Params& p) const override {
        if (!ev.connected(0)) return emptyInfo();
        Info i = ev.inputInfo(0);
        int gx = int(std::ceil(std::max(0.0, p.f("size", 0)))) + blurRadius(p.f("blur"), 3, 3);
        int gy = int(std::ceil(std::max(0.0, p.f("size", 1)))) + blurRadius(p.f("blur"), 3, 3);
        i.bbox = i.bbox.grow(gx, gy);
        return i;
    }
    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        Info inf = ev.inputInfo(0);
        double sx = p.f("size", 0), sy = p.f("size", 1), soft = std::max(0.0, p.f("blur"));
        int br = blurRadius(soft, 3, 3);
        int gx = int(std::ceil(std::fabs(sx))) + 1, gy = int(std::ceil(std::fabs(sy))) + 1;
        ChannelList sel = intersect(resolveChannels(p.s("channels"), inf.channels), chans);
        Box mid = roi.grow(br, br);
        ImagePtr in = ev.input(0, mid.grow(gx, gy), chans);
        auto out = copyOf(*in, roi, chans);
        for (auto& c : sel) {
            auto tmp = std::make_shared<Image>(mid, ChannelList{c});
            auto src = copyOf(*in, mid.grow(gx, gy), {c});
            morphPlane(src->plane(c), src->bbox(), tmp->plane(c), mid, sx, sy, ev.cancel);
            if (soft > 0) {
                blurPlane(*tmp, c, roi, out->plane(c), soft, soft, 3, 3, ev.cancel);
            } else {
                auto cropped = copyOf(*tmp, roi, {c});
                std::memcpy(out->plane(c), cropped->plane(c), sizeof(float) * size_t(roi.w()) * roi.h());
            }
        }
        applyMaskMix(ev, p, 1, *in, *out, sel);
        return out;
    }
};
VC_REGISTER_OP(OpDesc{"DilateErode", "Keyer",
                      "Grow (positive size) or shrink (negative) with sub-pixel radius, then optional blur.",
                      {"", "mask"}, 0,
                      concat({K::channels("channels", "channels", "alpha"),
                              K::wh("size", "size", -1, -1, -20, 20, "Pixels; negative erodes"),
                              K::flt("blur", "blur", 0, 0, 10, "Gaussian sigma applied after")},
                             maskKnobs()),
                      1, maker<DilateErode>()});

// ------------------------------------------------------------ EdgeBlur
class EdgeBlur : public Op {
public:
    Info info(Eval& ev, const Params&) const override { return ev.connected(0) ? ev.inputInfo(0) : emptyInfo(); }
    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        Info inf = ev.inputInfo(0);
        double size = std::max(0.0, p.f("size")), width = std::max(0.0, p.f("edgeWidth"));
        double soft = std::max(0.0, p.f("edgeSoftness"));
        std::string matte = p.s("matte");
        int rb = blurRadius(size, 3, 3), rs = blurRadius(soft, 3, 3), rw = int(std::ceil(width)) + 1;
        ChannelList sel = intersect(resolveChannels(p.s("channels"), inf.channels), chans);
        ChannelList need = unite(chans, intersect(ChannelList{matte}, inf.channels));
        Box bandBox = roi.grow(rs, rs);
        ImagePtr in = ev.input(0, roi.grow(rb, rb).unite(bandBox.grow(rw, rw)), need);
        auto out = copyOf(*in, roi, chans);

        // Edge band: dilate(matte) - erode(matte), softened.
        auto m = copyOf(*in, bandBox.grow(rw, rw), {matte});
        auto band = std::make_shared<Image>(bandBox, ChannelList{"b"});
        std::vector<float> dil(size_t(bandBox.w()) * bandBox.h()), ero(dil.size());
        const float* mp = m->has(matte) ? m->plane(matte) : nullptr;
        std::vector<float> zeros;
        if (!mp) {
            zeros.assign(size_t(m->bbox().w()) * m->bbox().h(), 0.0f);
            mp = zeros.data();
        }
        morphPlane(mp, m->bbox(), dil.data(), bandBox, width, width, ev.cancel);
        morphPlane(mp, m->bbox(), ero.data(), bandBox, -width, -width, ev.cancel);
        float* bp = band->plane("b");
        for (size_t i = 0; i < dil.size(); ++i) bp[i] = clamp01(dil[i] - ero[i]);
        std::vector<float> e(size_t(roi.w()) * roi.h());
        blurPlane(*band, "b", roi, e.data(), soft, soft, 3, 3, ev.cancel);

        std::vector<float> bl(e.size());
        for (auto& c : sel) {
            blurPlane(*in, c, roi, bl.data(), size, size, 3, 3, ev.cancel);
            float* o = out->plane(c);
            float k = float(p.f("amount"));
            for (size_t i = 0; i < bl.size(); ++i) o[i] = o[i] + (bl[i] - o[i]) * clamp01(e[i] * k);
        }
        applyMaskMix(ev, p, 1, *in, *out, sel);
        return out;
    }
};
VC_REGISTER_OP(OpDesc{"EdgeBlur", "Keyer", "Blur only inside a band around the matte edge.", {"", "mask"}, 0,
                      concat({K::channels("channels", "channels", "rgba"),
                              K::channel("matte", "edge matte", "rgba.alpha"),
                              K::flt("size", "size", 3, 0, 50, "Blur sigma in pixels"),
                              K::flt("edgeWidth", "edge width", 2, 0, 20, "Band half-width in pixels"),
                              K::flt("edgeSoftness", "edge softness", 1, 0, 20, "Band blur sigma"),
                              K::flt("amount", "edge multiplier", 1, 0, 4)},
                             maskKnobs()),
                      1, maker<EdgeBlur>()});

}  // namespace
}  // namespace vc::ops
