// Keyer and Despill.
#include "common.h"

namespace vc::ops {
namespace {

const std::vector<std::string> kKeyModes{"luminance", "channel", "saturation", "hue", "color distance", "screen"};

// Trapezoid: 0 below a, ramps to 1 at b, 1 until c, ramps to 0 at d.
// With d <= c the upper side stays at 1 (open-ended range).
inline double trapezoid(double v, double a, double b, double c, double d) {
    if (v < a) return 0.0;
    if (v < b) return b > a ? (v - a) / (b - a) : 1.0;
    if (v <= c) return 1.0;
    if (d <= c) return 1.0;
    if (v < d) return 1.0 - (v - c) / (d - c);
    return 0.0;
}

inline double smooth01(double t) {
    t = std::min(1.0, std::max(0.0, t));
    return t * t * (3 - 2 * t);
}

class Keyer : public Op {
public:
    Info info(Eval& ev, const Params& p) const override {
        if (!ev.connected(0)) return emptyInfo();
        Info i = ev.inputInfo(0);
        i.channels = unite(i.channels, {p.s("output")});
        return i;
    }

    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        Info inf = ev.inputInfo(0);
        std::string mode = p.s("operation"), outCh = p.s("output"), srcCh = p.s("source");
        double grow = p.f("shrinkGrow"), soft = std::max(0.0, p.f("softness"));
        int rb = blurRadius(soft, 3, 3), rg = int(std::ceil(std::fabs(grow))) + 1;
        Box keyBox = roi.grow(rb, rb).grow(rg, rg);
        ChannelList need = unite(chans, {"rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"});
        if (mode == "channel") need.push_back(srcCh);
        need.push_back(outCh);
        ImagePtr in = ev.input(0, keyBox, intersect(sortedChannels(need), inf.channels));
        auto out = copyOf(*in, roi, chans);

        double a = p.f("range", 0), b = p.f("range", 1), c = p.f("range", 2), d = p.f("range", 3);
        double kr = p.f("keyColor", 0), kg = p.f("keyColor", 1), kb = p.f("keyColor", 2);
        double tol = p.f("tolerance"), cs = std::max(1e-6, p.f("colorSoftness"));
        double sr = p.f("screenColor", 0), sg = p.f("screenColor", 1), sb = p.f("screenColor", 2);
        double bal = p.f("balance"), gain = p.f("screenGain");
        bool inv = p.b("invert");
        double clo = p.f("clip", 0), chi = p.f("clip", 1);
        // Dominant screen channel decides which difference to take.
        int dom = sg >= sr && sg >= sb ? 1 : (sb >= sr ? 2 : 0);
        auto screenDiff = [&](double R, double G, double B) {
            double ch[3] = {R, G, B};
            int o1 = (dom + 1) % 3, o2 = (dom + 2) % 3;
            return ch[dom] - (ch[o1] * bal + ch[o2] * (1 - bal));
        };
        double screenRef = std::max(1e-6, screenDiff(sr, sg, sb));

        auto key = std::make_shared<Image>(keyBox, ChannelList{"k"});
        float* kp = key->plane("k");
        const bool unp = hasChannel(inf.channels, "rgba.alpha") && p.b("unpremult");
        ev.parallelRows(keyBox, [&](int y) {
            int n = keyBox.w();
            std::vector<float> rows(size_t(n) * 5);
            in->readRow("rgba.red", y, keyBox.x0, n, rows.data());
            in->readRow("rgba.green", y, keyBox.x0, n, rows.data() + n);
            in->readRow("rgba.blue", y, keyBox.x0, n, rows.data() + 2 * n);
            in->readRow("rgba.alpha", y, keyBox.x0, n, rows.data() + 3 * n);
            in->readRow(mode == "channel" ? in->index(srcCh) : -1, y, keyBox.x0, n, rows.data() + 4 * n);
            for (int x = keyBox.x0; x < keyBox.x1; ++x) {
                int xi = x - keyBox.x0;
                double al = rows[3 * n + xi];
                double R = rows[xi], G = rows[n + xi], B = rows[2 * n + xi];
                if (al > 1e-6 && unp) {
                    R /= al; G /= al; B /= al;
                }
                double m;
                if (mode == "luminance") {
                    m = trapezoid(0.2126 * R + 0.7152 * G + 0.0722 * B, a, b, c, d);
                } else if (mode == "channel") {
                    m = trapezoid(rows[4 * n + xi], a, b, c, d);
                } else if (mode == "saturation") {
                    double mx = std::max({R, G, B}), mn = std::min({R, G, B});
                    m = trapezoid(mx > 1e-9 ? (mx - mn) / mx : 0.0, a, b, c, d);
                } else if (mode == "hue") {
                    double mx = std::max({R, G, B}), mn = std::min({R, G, B}), dd = mx - mn, h = 0;
                    if (dd > 1e-9) {
                        if (mx == R) h = std::fmod((G - B) / dd + 6.0, 6.0);
                        else if (mx == G) h = (B - R) / dd + 2.0;
                        else h = (R - G) / dd + 4.0;
                    }
                    m = trapezoid(h / 6.0, a, b, c, d);
                } else if (mode == "color distance") {
                    double dist = std::sqrt((R - kr) * (R - kr) + (G - kg) * (G - kg) + (B - kb) * (B - kb));
                    m = 1.0 - smooth01((dist - tol) / cs);
                } else {
                    // Screen key: foreground alpha from how far the pixel is from the screen.
                    m = 1.0 - std::min(1.0, std::max(0.0, screenDiff(R, G, B) / screenRef * gain));
                }
                if (chi > clo) m = (m - clo) / (chi - clo);
                m = std::min(1.0, std::max(0.0, m));
                if (inv) m = 1.0 - m;
                kp[size_t(y - keyBox.y0) * keyBox.w() + (x - keyBox.x0)] = float(m);
            }
        });

        // Edge controls: shrink/grow then soften.
        Box mid = roi.grow(rb, rb);
        auto edged = std::make_shared<Image>(mid, ChannelList{"k"});
        if (std::fabs(grow) > 1e-6) {
            morphPlane(kp, keyBox, edged->plane("k"), mid, grow, grow, ev.cancel);
        } else {
            auto c2 = copyOf(*key, mid, {"k"});
            edged = c2;
        }
        std::vector<float> matte(size_t(roi.w()) * roi.h());
        blurPlane(*edged, "k", roi, matte.data(), soft, soft, 3, 3, ev.cancel);

        std::string comb = p.s("combine");
        bool premult = p.b("premultiply") && outCh == "rgba.alpha";
        bool despill = p.b("despill") && mode == "screen";
        ev.parallelRows(roi, [&](int y) {
            float* o = out->row(outCh, y);
            int n = roi.w();
            std::vector<float> olds(n), o1r(n), o2r(n);
            const char* rgbn[3] = {"rgba.red", "rgba.green", "rgba.blue"};
            in->readRow(outCh, y, roi.x0, n, olds.data());
            in->readRow(rgbn[(dom + 1) % 3], y, roi.x0, n, o1r.data());
            in->readRow(rgbn[(dom + 2) % 3], y, roi.x0, n, o2r.data());
            for (int x = roi.x0; x < roi.x1; ++x) {
                size_t i = size_t(y - roi.y0) * roi.w() + (x - roi.x0);
                float m = matte[i], old = olds[x - roi.x0], v;
                if (comb == "multiply") v = old * m;
                else if (comb == "max") v = std::max(old, m);
                else if (comb == "min") v = std::min(old, m);
                else v = m;
                if (o) o[x - roi.x0] = v;
                if (despill) {
                    const char* rgb[3] = {"rgba.red", "rgba.green", "rgba.blue"};
                    float* sp = out->row(rgb[dom], y);
                    float o1 = o1r[x - roi.x0], o2 = o2r[x - roi.x0];
                    if (sp) sp[x - roi.x0] = std::min(sp[x - roi.x0], 0.5f * (o1 + o2));
                }
                if (premult)
                    for (auto ch : {"rgba.red", "rgba.green", "rgba.blue"})
                        if (float* r = out->row(ch, y)) r[x - roi.x0] *= v;
            }
        });
        return out;
    }
};
VC_REGISTER_OP(OpDesc{"Keyer", "Keyer",
                      "Pull a matte from luminance, any channel/AOV, saturation, hue, distance to a colour, or a "
                      "green/blue screen difference. Range is a trapezoid: 0 below A, 1 from B to C, 0 above D.",
                      {""}, 0,
                      {K::enm("operation", "operation", kKeyModes),
                       K::channel("source", "source channel", "rgba.red", "Used by the channel mode (e.g. depth.Z)"),
                       K::range("range", "range A B C D", {0.0, 0.5, 1.0, 1.0}, 0, 1),
                       K::range("clip", "clip black/white", {0.0, 1.0}, 0, 1),
                       K::boolean("invert", "invert", false),
                       K::boolean("unpremult", "unpremult input", true),
                       K::group(K::color("keyColor", "key colour", {0.1, 0.6, 0.15}), "Colour"),
                       K::group(K::flt("tolerance", "tolerance", 0.1, 0, 1), "Colour"),
                       K::group(K::flt("colorSoftness", "softness", 0.1, 0, 1), "Colour"),
                       K::group(K::color("screenColor", "screen colour", {0.1, 0.6, 0.15}), "Screen"),
                       K::group(K::flt("balance", "balance", 0.5, 0, 1, "Weight of the first other channel"), "Screen"),
                       K::group(K::flt("screenGain", "screen gain", 1, 0, 4), "Screen"),
                       K::group(K::boolean("despill", "despill", true), "Screen"),
                       K::group(K::flt("shrinkGrow", "shrink/grow", 0, -10, 10, "Pixels"), "Edge"),
                       K::group(K::flt("softness", "edge softness", 0, 0, 10, "Gaussian sigma"), "Edge"),
                       K::channel("output", "output", "rgba.alpha"),
                       K::enm("combine", "combine with existing", {"replace", "multiply", "max", "min"}),
                       K::boolean("premultiply", "premultiply rgb", false)},
                      1, maker<Keyer>()});

// ------------------------------------------------------------ Despill
class Despill : public PixelOp {
    RowKernel kernel(const Params& p, const Info&) const override {
        int dom = p.s("screen") == "blue" ? 2 : 1;
        std::string lim = p.s("limit");
        double amt = p.f("amount");
        bool keepLum = p.b("restoreLuminance");
        return [=](float* r, float* g, float* b, float* a, int n, int, int) {
            float* ch[3] = {r, g, b};
            int o1 = (dom + 1) % 3, o2 = (dom + 2) % 3;
            for (int i = 0; i < n; ++i) {
                float x1 = ch[o1][i], x2 = ch[o2][i], l;
                if (lim == "max") l = std::max(x1, x2);
                else if (lim == "min") l = std::min(x1, x2);
                else l = 0.5f * (x1 + x2);
                float spill = std::max(0.0f, ch[dom][i] - l) * float(amt);
                float before = luma709(r[i], g[i], b[i]);
                ch[dom][i] -= spill;
                if (keepLum) {
                    float d = before - luma709(r[i], g[i], b[i]);
                    r[i] += d; g[i] += d; b[i] += d;
                }
                (void)a;
            }
        };
    }
};
VC_REGISTER_OP(OpDesc{"Despill", "Keyer", "Suppress green or blue spill by limiting it to the other channels.",
                      {"", "mask"}, 0,
                      concat(concat(pixelOpKnobs("rgb", true),
                                    {K::enm("screen", "screen", {"green", "blue"}),
                                     K::enm("limit", "limit", {"average", "max", "min"}),
                                     K::flt("amount", "amount", 1, 0, 1),
                                     K::boolean("restoreLuminance", "restore luminance", false)}),
                             maskKnobs()),
                      1, maker<Despill>()});

}  // namespace
}  // namespace vc::ops
