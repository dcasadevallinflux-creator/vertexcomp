// Colour ops. All are PixelOps: channel/layer selection, optional
// unpremultiply, mask input and mix come from the base class.
#include "vc/color.h"

#include "common.h"
#include "vc/expr.h"

namespace vc::ops {
namespace {

const std::vector<std::string> kPixelInputs{"", "mask"};

std::vector<KnobDef> withCommon(std::vector<KnobDef> own, const std::string& chans, bool unp) {
    return concat(concat(pixelOpKnobs(chans, unp), own), maskKnobs());
}

// Applies f(value, component) to r, g, b, a.
template <class F>
void perChannel(float* r, float* g, float* b, float* a, int n, F f) {
    float* ch[4] = {r, g, b, a};
    for (int c = 0; c < 4; ++c)
        for (int i = 0; i < n; ++i) ch[c][i] = f(ch[c][i], c);
}

inline float safePow(float x, double e) {
    if (e == 1.0) return x;
    return x > 0 ? float(std::pow(double(x), e)) : x;
}

// ------------------------------------------------------------ Grade
class Grade : public PixelOp {
    RowKernel kernel(const Params& p, const Info&) const override {
        double A[4], B[4], G[4];
        for (int c = 0; c < 4; ++c) {
            double bp = p.f("blackpoint", c), wp = p.f("whitepoint", c), lift = p.f("lift", c);
            double gain = p.f("gain", c), mul = p.f("multiply", c), off = p.f("offset", c);
            double d = wp - bp;
            A[c] = mul * (gain - lift) / (std::fabs(d) < 1e-9 ? 1e-9 : d);
            B[c] = off + lift - A[c] * bp;
            double g = p.f("gamma", c);
            G[c] = g <= 0 ? 1e9 : 1.0 / g;
        }
        bool bc = p.b("blackClamp"), wc = p.b("whiteClamp");
        return [=](float* r, float* g, float* b, float* a, int n, int, int) {
            perChannel(r, g, b, a, n, [&](float v, int c) {
                float o = safePow(float(A[c] * v + B[c]), G[c]);
                if (bc) o = std::max(0.0f, o);
                if (wc) o = std::min(1.0f, o);
                return o;
            });
        };
    }
};
VC_REGISTER_OP(OpDesc{"Grade", "Color", "Black/white point, lift, gain, multiply, offset and gamma.", kPixelInputs, 0,
                      withCommon({K::color("blackpoint", "blackpoint", {0, 0, 0, 0}, -1, 1),
                                  K::color("whitepoint", "whitepoint", {1, 1, 1, 1}, 0, 4),
                                  K::color("lift", "lift", {0, 0, 0, 0}, -1, 1),
                                  K::color("gain", "gain", {1, 1, 1, 1}, 0, 4),
                                  K::color("multiply", "multiply", {1, 1, 1, 1}, 0, 4),
                                  K::color("offset", "offset", {0, 0, 0, 0}, -1, 1),
                                  K::color("gamma", "gamma", {1, 1, 1, 1}, 0.2, 5),
                                  K::boolean("blackClamp", "black clamp", true),
                                  K::boolean("whiteClamp", "white clamp", false)},
                                 "rgb", false),
                      1, maker<Grade>()});

// ------------------------------------------------------------ ColorCorrect
struct CCParams {
    double sat, con, gam, gain, off;
    bool identity() const { return sat == 1 && con == 1 && gam == 1 && gain == 1 && off == 0; }
};

// Each stage is skipped when neutral; pow() dominates the cost otherwise.
inline void ccApply(const CCParams& k, double pivot, double& r, double& g, double& b) {
    if (k.sat != 1) {
        double l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        r = l + (r - l) * k.sat;
        g = l + (g - l) * k.sat;
        b = l + (b - l) * k.sat;
    }
    if (k.con != 1) {
        auto con = [&](double v) { return v > 0 ? pivot * std::pow(v / pivot, k.con) : v; };
        r = con(r); g = con(g); b = con(b);
    }
    if (k.gam != 1) {
        double e = 1.0 / std::max(1e-4, k.gam);
        auto gam = [&](double v) { return v > 0 ? std::pow(v, e) : v; };
        r = gam(r); g = gam(g); b = gam(b);
    }
    r = r * k.gain + k.off;
    g = g * k.gain + k.off;
    b = b * k.gain + k.off;
}

class ColorCorrect : public PixelOp {
    RowKernel kernel(const Params& p, const Info&) const override {
        auto get = [&](const std::string& pre) {
            return CCParams{p.f(pre + "saturation"), p.f(pre + "contrast"), p.f(pre + "gamma"), p.f(pre + "gain"),
                            p.f(pre + "offset")};
        };
        CCParams m = get(""), s = get("shadows_"), mid = get("midtones_"), h = get("highlights_");
        double pivot = std::max(1e-4, p.f("pivot"));
        double s0 = p.f("shadowRange", 0), s1 = p.f("shadowRange", 1);
        double h0 = p.f("highlightRange", 0), h1 = p.f("highlightRange", 1);
        bool clampOut = p.b("clamp");
        const bool ranges = !(s.identity() && mid.identity() && h.identity());
        return [=](float* r, float* g, float* b, float*, int n, int, int) {
            auto smooth = [](double e0, double e1, double x) {
                if (e1 <= e0) return x >= e1 ? 1.0 : 0.0;
                double t = std::min(1.0, std::max(0.0, (x - e0) / (e1 - e0)));
                return t * t * (3 - 2 * t);
            };
            for (int i = 0; i < n; ++i) {
                double R = r[i], G = g[i], B = b[i];
                if (!ranges) {
                    ccApply(m, pivot, R, G, B);
                    if (clampOut) {
                        R = std::min(1.0, std::max(0.0, R));
                        G = std::min(1.0, std::max(0.0, G));
                        B = std::min(1.0, std::max(0.0, B));
                    }
                    r[i] = float(R); g[i] = float(G); b[i] = float(B);
                    continue;
                }
                double l = 0.2126 * R + 0.7152 * G + 0.0722 * B;
                double ws = 1.0 - smooth(s0, s1, l), wh = smooth(h0, h1, l);
                double wm = std::max(0.0, 1.0 - ws - wh);
                double sr = R, sg = G, sb = B, mr = R, mg = G, mb = B, hr = R, hg = G, hb = B;
                ccApply(s, pivot, sr, sg, sb);
                ccApply(mid, pivot, mr, mg, mb);
                ccApply(h, pivot, hr, hg, hb);
                double tot = ws + wm + wh;
                if (tot <= 0) tot = 1;
                R = (ws * sr + wm * mr + wh * hr) / tot;
                G = (ws * sg + wm * mg + wh * hg) / tot;
                B = (ws * sb + wm * mb + wh * hb) / tot;
                ccApply(m, pivot, R, G, B);
                if (clampOut) {
                    R = std::min(1.0, std::max(0.0, R));
                    G = std::min(1.0, std::max(0.0, G));
                    B = std::min(1.0, std::max(0.0, B));
                }
                r[i] = float(R);
                g[i] = float(G);
                b[i] = float(B);
            }
        };
    }
};

std::vector<KnobDef> ccKnobs() {
    std::vector<KnobDef> k;
    for (std::string grp : {"", "shadows_", "midtones_", "highlights_"}) {
        std::string label = grp.empty() ? "Master" : std::string(1, char(std::toupper(grp[0]))) + grp.substr(1, grp.size() - 2);
        k.push_back(K::group(K::flt(grp + "saturation", "saturation", 1, 0, 4), label));
        k.push_back(K::group(K::flt(grp + "contrast", "contrast", 1, 0, 4), label));
        k.push_back(K::group(K::flt(grp + "gamma", "gamma", 1, 0.2, 5), label));
        k.push_back(K::group(K::flt(grp + "gain", "gain", 1, 0, 4), label));
        k.push_back(K::group(K::flt(grp + "offset", "offset", 0, -1, 1), label));
    }
    k.push_back(K::group(K::flt("pivot", "contrast pivot", 0.18, 0.01, 1, "Value left unchanged by contrast"), "Ranges"));
    k.push_back(K::group(K::range("shadowRange", "shadows", {0.0, 0.09}, 0, 1, "Luminance where shadows fade out"), "Ranges"));
    k.push_back(K::group(K::range("highlightRange", "highlights", {0.5, 1.0}, 0, 2, "Luminance where highlights fade in"), "Ranges"));
    k.push_back(K::boolean("clamp", "clamp 0-1", false));
    return k;
}
VC_REGISTER_OP(OpDesc{"ColorCorrect", "Color",
                      "Saturation, contrast (around a pivot), gamma, gain and offset for master, shadows, midtones and highlights.",
                      kPixelInputs, 0, withCommon(ccKnobs(), "rgb", false), 1, maker<ColorCorrect>()});

// ------------------------------------------------------------ Exposure
class Exposure : public PixelOp {
    RowKernel kernel(const Params& p, const Info&) const override {
        double m[3], bk[3];
        for (int c = 0; c < 3; ++c) {
            m[c] = std::pow(2.0, p.f("stops", c));
            bk[c] = p.f("black", c);
        }
        return [=](float* r, float* g, float* b, float*, int n, int, int) {
            float* ch[3] = {r, g, b};
            for (int c = 0; c < 3; ++c)
                for (int i = 0; i < n; ++i) ch[c][i] = float((ch[c][i] - bk[c]) * m[c] + bk[c]);
        };
    }
};
VC_REGISTER_OP(OpDesc{"Exposure", "Color", "Exposure in stops around a black point, in linear light.", kPixelInputs, 0,
                      withCommon({K::color("stops", "stops", {0, 0, 0}, -4, 4, "EV"),
                                  K::color("black", "black point", {0, 0, 0}, -0.1, 0.1)},
                                 "rgb", false),
                      1, maker<Exposure>()});

// ------------------------------------------------------------ hue helpers
void hueRotationMatrix(double deg, double m[9]) {
    double a = deg * kPi / 180.0, c = std::cos(a), s = std::sin(a), k = 1.0 / 3.0, q = std::sqrt(k);
    m[0] = c + (1 - c) * k;     m[1] = k * (1 - c) - q * s; m[2] = k * (1 - c) + q * s;
    m[3] = k * (1 - c) + q * s; m[4] = c + k * (1 - c);     m[5] = k * (1 - c) - q * s;
    m[6] = k * (1 - c) - q * s; m[7] = k * (1 - c) + q * s; m[8] = c + k * (1 - c);
}

class HueShift : public PixelOp {
    RowKernel kernel(const Params& p, const Info&) const override {
        double m[9];
        hueRotationMatrix(p.f("hue"), m);
        double sat = p.f("saturation"), bri = p.f("brightness");
        return [=](float* r, float* g, float* b, float*, int n, int, int) {
            for (int i = 0; i < n; ++i) {
                double R = r[i], G = g[i], B = b[i];
                double nr = m[0] * R + m[1] * G + m[2] * B;
                double ng = m[3] * R + m[4] * G + m[5] * B;
                double nb = m[6] * R + m[7] * G + m[8] * B;
                double l = 0.2126 * nr + 0.7152 * ng + 0.0722 * nb;
                r[i] = float((l + (nr - l) * sat) * bri);
                g[i] = float((l + (ng - l) * sat) * bri);
                b[i] = float((l + (nb - l) * sat) * bri);
            }
        };
    }
};
VC_REGISTER_OP(OpDesc{"HueShift", "Color", "Rotate hue around the grey axis; scale saturation and brightness.",
                      kPixelInputs, 0,
                      withCommon({K::flt("hue", "hue rotation", 0, -180, 180, "degrees"),
                                  K::flt("saturation", "saturation", 1, 0, 4),
                                  K::flt("brightness", "brightness", 1, 0, 4)},
                                 "rgb", true),
                      1, maker<HueShift>()});

// HueCorrect: curves sampled at the six primaries/secondaries
// (red, yellow, green, cyan, blue, magenta), interpolated periodically.
double periodic6(const double v[6], double h) {  // h in [0, 6)
    int i = int(std::floor(h)) % 6;
    double t = h - std::floor(h);
    double p0 = v[(i + 5) % 6], p1 = v[i], p2 = v[(i + 1) % 6], p3 = v[(i + 2) % 6];
    double t2 = t * t, t3 = t2 * t;
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
}

class HueCorrect : public PixelOp {
    RowKernel kernel(const Params& p, const Info&) const override {
        double sat[6], lum[6], hue[6];
        for (int i = 0; i < 6; ++i) {
            sat[i] = p.f("sat", i);
            lum[i] = p.f("lum", i);
            hue[i] = p.f("hueShift", i);
        }
        return [=](float* r, float* g, float* b, float*, int n, int, int) {
            for (int i = 0; i < n; ++i) {
                double R = r[i], G = g[i], B = b[i];
                double mx = std::max({R, G, B}), mn = std::min({R, G, B}), d = mx - mn;
                if (d <= 1e-9) continue;
                double h;
                if (mx == R) h = std::fmod((G - B) / d + 6.0, 6.0);
                else if (mx == G) h = (B - R) / d + 2.0;
                else h = (R - G) / d + 4.0;
                double s = periodic6(sat, h), l = periodic6(lum, h), hs = periodic6(hue, h);
                double m[9];
                hueRotationMatrix(hs, m);
                double nr = m[0] * R + m[1] * G + m[2] * B;
                double ng = m[3] * R + m[4] * G + m[5] * B;
                double nb = m[6] * R + m[7] * G + m[8] * B;
                double y = 0.2126 * nr + 0.7152 * ng + 0.0722 * nb;
                // Fade the correction in with chroma so neutrals stay neutral.
                double w = std::min(1.0, d / std::max(1e-6, mx));
                double s2 = 1.0 + (s - 1.0) * w, l2 = 1.0 + (l - 1.0) * w;
                r[i] = float((y + (nr - y) * s2) * l2);
                g[i] = float((y + (ng - y) * s2) * l2);
                b[i] = float((y + (nb - y) * s2) * l2);
            }
        };
    }
};
KnobDef six(const std::string& n, const std::string& l, double d, double lo, double hi, const std::string& tip) {
    KnobDef k = K::color(n, l, {d, d, d, d, d, d}, lo, hi, tip);
    return k;
}
VC_REGISTER_OP(OpDesc{"HueCorrect", "Color",
                      "Hue-selective saturation, luminance and hue shift. Six control points at R Y G C B M.",
                      kPixelInputs, 0,
                      withCommon({six("sat", "saturation", 1, 0, 2, "Saturation multiplier per hue: R Y G C B M"),
                                  six("lum", "luminance", 1, 0, 2, "Luminance multiplier per hue: R Y G C B M"),
                                  six("hueShift", "hue shift", 0, -60, 60, "Degrees per hue: R Y G C B M")},
                                 "rgb", true),
                      1, maker<HueCorrect>()});

// ------------------------------------------------------------ Saturation
class Saturation : public PixelOp {
    RowKernel kernel(const Params& p, const Info&) const override {
        double s = p.f("saturation");
        std::string mode = p.s("luminanceMath");
        return [=](float* r, float* g, float* b, float*, int n, int, int) {
            for (int i = 0; i < n; ++i) {
                double l;
                if (mode == "Rec 601") l = 0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i];
                else if (mode == "average") l = (r[i] + g[i] + b[i]) / 3.0;
                else if (mode == "maximum") l = std::max({r[i], g[i], b[i]});
                else l = 0.2126 * r[i] + 0.7152 * g[i] + 0.0722 * b[i];
                r[i] = float(l + (r[i] - l) * s);
                g[i] = float(l + (g[i] - l) * s);
                b[i] = float(l + (b[i] - l) * s);
            }
        };
    }
};
VC_REGISTER_OP(OpDesc{"Saturation", "Color", "Saturation around luminance.", kPixelInputs, 0,
                      withCommon({K::flt("saturation", "saturation", 1, 0, 4),
                                  K::enm("luminanceMath", "luminance math", {"Rec 709", "Rec 601", "average", "maximum"})},
                                 "rgb", true),
                      1, maker<Saturation>()});

// ------------------------------------------------------------ simple math
template <int Kind>
class MathOp : public PixelOp {
    RowKernel kernel(const Params& p, const Info&) const override {
        double v[4], lo[4], hi[4];
        for (int c = 0; c < 4; ++c) {
            v[c] = p.f("value", c);
            lo[c] = p.f("minimum", c);
            hi[c] = p.f("maximum", c);
        }
        bool doMin = p.b("clampMin"), doMax = p.b("clampMax");
        return [=](float* r, float* g, float* b, float* a, int n, int, int) {
            perChannel(r, g, b, a, n, [&](float x, int c) -> float {
                switch (Kind) {
                    case 0: return float(x * v[c]);
                    case 1: return float(x + v[c]);
                    case 2: return v[c] <= 0 ? 0.0f : safePow(x, 1.0 / v[c]);
                    case 3: return 1.0f - x;
                    default: {
                        float o = x;
                        if (doMin) o = std::max(o, float(lo[c]));
                        if (doMax) o = std::min(o, float(hi[c]));
                        return o;
                    }
                }
            });
        };
    }
    bool usesUnpremult() const override { return true; }
};
VC_REGISTER_OP(OpDesc{"Multiply", "Color", "Multiply channels by a value.", kPixelInputs, 0,
                      withCommon({K::color("value", "value", {1, 1, 1, 1}, 0, 4)}, "rgba", false), 1, maker<MathOp<0>>()});
VC_REGISTER_OP(OpDesc{"Add", "Color", "Add a value to channels.", kPixelInputs, 0,
                      withCommon({K::color("value", "value", {0, 0, 0, 0}, -1, 1)}, "rgb", false), 1, maker<MathOp<1>>()});
VC_REGISTER_OP(OpDesc{"Gamma", "Color", "out = in ^ (1 / value) for positive values.", kPixelInputs, 0,
                      withCommon({K::color("value", "value", {1, 1, 1, 1}, 0.2, 5)}, "rgb", false), 1, maker<MathOp<2>>()});
VC_REGISTER_OP(OpDesc{"Invert", "Color", "out = 1 - in.", kPixelInputs, 0, withCommon({}, "rgb", false), 1,
                      maker<MathOp<3>>()});
VC_REGISTER_OP(OpDesc{"Clamp", "Color", "Clamp channels to a range.", kPixelInputs, 0,
                      withCommon({K::color("minimum", "minimum", {0, 0, 0, 0}, -1, 1),
                                  K::boolean("clampMin", "clamp minimum", true),
                                  K::color("maximum", "maximum", {1, 1, 1, 1}, 0, 4),
                                  K::boolean("clampMax", "clamp maximum", true)},
                                 "rgba", false),
                      1, maker<MathOp<4>>()});

// ------------------------------------------------------------ Premult / Unpremult
template <bool Mult>
class PremultOp : public PixelOp {
    RowKernel kernel(const Params&, const Info&) const override {
        return [](float* r, float* g, float* b, float* a, int n, int, int) {
            for (int i = 0; i < n; ++i) {
                float al = a[i];
                if (Mult) {
                    r[i] *= al; g[i] *= al; b[i] *= al;
                } else if (al > 1e-6f) {
                    r[i] /= al; g[i] /= al; b[i] /= al;
                }
            }
        };
    }
    bool usesUnpremult() const override { return false; }
};
VC_REGISTER_OP(OpDesc{"Premult", "Merge", "Multiply colour by alpha.", kPixelInputs, 0,
                      concat({K::channels("channels", "channels", "rgb")}, maskKnobs()), 1, maker<PremultOp<true>>()});
VC_REGISTER_OP(OpDesc{"Unpremult", "Merge", "Divide colour by alpha (pixels with alpha <= 1e-6 are left alone).",
                      kPixelInputs, 0, concat({K::channels("channels", "channels", "rgb")}, maskKnobs()), 1,
                      maker<PremultOp<false>>()});

// ------------------------------------------------------------ Colorspace
class Colorspace : public PixelOp {
    RowKernel kernel(const Params& p, const Info&) const override {
        std::string in = p.s("in"), out = p.s("out");
        if (in == "working") in = workingSpace();
        if (out == "working") out = workingSpace();
        auto xf = std::make_shared<ColorTransform>(in, out);
        return [xf](float* r, float* g, float* b, float*, int n, int, int) {
            for (int i = 0; i < n; ++i) xf->apply(r[i], g[i], b[i]);
        };
    }
};
std::vector<std::string> csOptions() {
    auto v = colorSpaceNames();
    v.insert(v.begin(), "working");
    return v;
}
VC_REGISTER_OP(OpDesc{"Colorspace", "Color", "Convert between colour spaces (transfer function and primaries).",
                      kPixelInputs, 0,
                      withCommon({K::enm("in", "in", csOptions(), "working"), K::enm("out", "out", csOptions(), "sRGB")},
                                 "rgb", true),
                      1, maker<Colorspace>()});

// ------------------------------------------------------------ Expression
class Expression : public PixelOp {
    RowKernel kernel(const Params& p, const Info& info) const override {
        static const char* vars[] = {"r", "g", "b", "a", "x", "y", "frame", "width", "height"};
        auto resolve = [&](const std::string& n) -> int {
            for (int i = 0; i < 9; ++i)
                if (n == vars[i]) return i;
            return -1;
        };
        auto exprs = std::make_shared<std::vector<Expr>>();
        std::vector<bool> has;
        for (int c = 0; c < 4; ++c) {
            std::string src = p.s("expr" + std::to_string(c));
            if (src.empty()) {
                exprs->emplace_back();
                has.push_back(false);
            } else {
                exprs->emplace_back(src, resolve);
                has.push_back(true);
            }
        }
        double frame = p.f("__frame"), w = info.format.w, h = info.format.h;
        return [exprs, has, frame, w, h](float* r, float* g, float* b, float* a, int n, int x0, int y) {
            double v[9];
            float res[4];
            for (int i = 0; i < n; ++i) {
                v[0] = r[i]; v[1] = g[i]; v[2] = b[i]; v[3] = a[i];
                v[4] = x0 + i + 0.5; v[5] = y + 0.5; v[6] = frame; v[7] = w; v[8] = h;
                for (int c = 0; c < 4; ++c) res[c] = has[c] ? float((*exprs)[c].eval(v)) : float(v[c]);
                r[i] = res[0]; g[i] = res[1]; b[i] = res[2]; a[i] = res[3];
            }
        };
    }
};
VC_REGISTER_OP(OpDesc{"Expression", "Color",
                      "Per-channel math. Variables: r g b a x y frame width height. Empty keeps the channel.",
                      kPixelInputs, 0,
                      withCommon({K::str("expr0", "= red", ""), K::str("expr1", "= green", ""),
                                  K::str("expr2", "= blue", ""), K::str("expr3", "= alpha", "")},
                                 "rgba", false),
                      1, maker<Expression>()});

}  // namespace
}  // namespace vc::ops
