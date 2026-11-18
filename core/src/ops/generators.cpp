// Constant, Ramp, CheckerBoard, Rectangle, Noise (including film grain).
#include "common.h"
#include "../../third_party/stb_image.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace vc::ops {
namespace {

const ChannelList kRGBA{"rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"};

std::vector<unsigned char> decodeBase64(const std::string& value) {
    static const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<unsigned char> out;
    uint32_t buffer = 0;
    int bits = -8;
    for (unsigned char ch : value) {
        if (ch == '=') break;
        const size_t index = alphabet.find(char(ch));
        if (index == std::string::npos) continue;
        buffer = (buffer << 6) | uint32_t(index);
        bits += 6;
        if (bits >= 0) {
            out.push_back(static_cast<unsigned char>((buffer >> bits) & 0xff));
            bits -= 8;
        }
    }
    return out;
}

float srgbToLinear(float value) {
    return value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
}

Info generatorInfo(const Params& p) {
    Info i;
    i.format.w = std::max(1, int(p.f("format", 0)));
    i.format.h = std::max(1, int(p.f("format", 1)));
    i.bbox = i.format.box();
    i.channels = kRGBA;
    return i;
}

KnobDef formatKnob() { return K::wh("format", "format", 1920, 1080, 1, 8192, "Output resolution in pixels"); }

std::vector<KnobDef> textKnobs() {
    auto size = K::flt("size", "size", 64, 1, 1024);
    auto color = K::color("color", "color", {1, 1, 1, 1});
    auto position = K::xy("position", "position", 100, 100);
    size.animatable = color.animatable = position.animatable = false;
    return {K::text("text", "text", "Text"), K::str("font", "font", "Arial"), size, color, position,
            formatKnob(), K::str("bitmap", "bitmap", "", "Generated text bitmap")};
}

// ------------------------------------------------------------ Constant
class Constant : public Op {
public:
    Info info(Eval&, const Params& p) const override { return generatorInfo(p); }
    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        auto out = std::make_shared<Image>(roi, chans);
        for (auto& c : chans) {
            int idx = int(std::find(kRGBA.begin(), kRGBA.end(), c) - kRGBA.begin());
            float v = float(p.f("color", idx));
            float* d = out->plane(c);
            std::fill(d, d + size_t(roi.w()) * roi.h(), v);
        }
        (void)ev;
        return out;
    }
};
VC_REGISTER_OP(OpDesc{"Constant", "Image", "Solid colour over a format.", {}, 0,
                      {K::color("color", "color", {0, 0, 0, 0}, 0, 1, "Premultiplied RGBA"), formatKnob()}, 1,
                      maker<Constant>()});

class Text : public Op {
public:
    Info info(Eval& ev, const Params& p) const override {
        Info i;
        if (ev.connected(0)) i = ev.inputInfo(0);
        else {
            i.format.w = std::max(1, int(p.f("format", 0)));
            i.format.h = std::max(1, int(p.f("format", 1)));
            i.bbox = Box{};
        }
        i.bbox = i.bbox.unite(i.format.box());
        i.channels = unite(i.channels, kRGBA);
        return i;
    }

    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        ImagePtr in = ev.input(0, roi, chans);
        auto out = copyOf(*in, roi, chans);
        const std::string& encoded = p.s("bitmap");
        if (encoded.empty()) return out;

        const auto bytes = decodeBase64(encoded);
        if (bytes.empty()) throw std::runtime_error("Text node bitmap is not valid base64 PNG data");
        int width = 0, height = 0, components = 0;
        unsigned char* pixels = stbi_load_from_memory(bytes.data(), int(bytes.size()), &width, &height, &components, 4);
        if (!pixels) throw std::runtime_error("Text node bitmap could not be decoded");

        const int ox = int(std::lround(p.f("position", 0)));
        const int oy = int(std::lround(p.f("position", 1)));
        const Box overlap = Box{ox, oy, ox + width, oy + height}.intersect(roi);
        for (int y = overlap.y0; y < overlap.y1; ++y) {
            for (int x = overlap.x0; x < overlap.x1; ++x) {
                const size_t index = (size_t(y - oy) * size_t(width) + size_t(x - ox)) * 4;
                const float alpha = float(pixels[index + 3]) / 255.0f;
                const size_t outIndex = size_t(y - roi.y0) * size_t(roi.w()) + size_t(x - roi.x0);
                const unsigned char values[3] = {pixels[index], pixels[index + 1], pixels[index + 2]};
                const char* names[3] = {"rgba.red", "rgba.green", "rgba.blue"};
                for (int c = 0; c < 3; ++c) {
                    if (float* plane = out->plane(names[c])) {
                        const float color = srgbToLinear(float(values[c]) / 255.0f);
                        plane[outIndex] = color * alpha + plane[outIndex] * (1.0f - alpha);
                    }
                }
                if (float* plane = out->plane("rgba.alpha"))
                    plane[outIndex] = alpha + plane[outIndex] * (1.0f - alpha);
            }
        }
        stbi_image_free(pixels);
        return out;
    }
};

VC_REGISTER_OP(OpDesc{"Text", "Draw", "Render text over the input image.", {"bg"}, 0,
                      textKnobs(),
                      1, maker<Text>()});

// ------------------------------------------------------------ Ramp
class Ramp : public Op {
public:
    Info info(Eval&, const Params& p) const override { return generatorInfo(p); }
    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        auto out = std::make_shared<Image>(roi, chans);
        double x0 = p.f("p0", 0), y0 = p.f("p0", 1), x1 = p.f("p1", 0), y1 = p.f("p1", 1);
        std::string type = p.s("type");
        double dx = x1 - x0, dy = y1 - y0, len2 = dx * dx + dy * dy, len = std::sqrt(len2);
        ev.parallelRows(roi, [&](int y) {
            for (int x = roi.x0; x < roi.x1; ++x) {
                double cx = x + 0.5, cy = y + 0.5, t;
                if (type == "radial") t = len > 0 ? std::hypot(cx - x0, cy - y0) / len : 0;
                else t = len2 > 0 ? ((cx - x0) * dx + (cy - y0) * dy) / len2 : 0;
                t = std::min(1.0, std::max(0.0, t));
                if (type == "smooth" || type == "radial") t = t * t * (3 - 2 * t);
                for (size_t c = 0; c < chans.size(); ++c) {
                    int idx = int(std::find(kRGBA.begin(), kRGBA.end(), chans[c]) - kRGBA.begin());
                    double a = p.f("color0", idx), b = p.f("color1", idx);
                    out->row(chans[c], y)[x - roi.x0] = float(a + (b - a) * t);
                }
            }
        });
        return out;
    }
};
VC_REGISTER_OP(OpDesc{"Ramp", "Image", "Linear or radial gradient between two points.", {}, 0,
                      {K::xy("p0", "point 0", 0, 540), K::xy("p1", "point 1", 1920, 540),
                       K::color("color0", "color 0", {0, 0, 0, 1}), K::color("color1", "color 1", {1, 1, 1, 1}),
                       K::enm("type", "type", {"linear", "smooth", "radial"}), formatKnob()},
                      1, maker<Ramp>()});

// ------------------------------------------------------------ CheckerBoard
class CheckerBoard : public Op {
public:
    Info info(Eval&, const Params& p) const override { return generatorInfo(p); }
    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        auto out = std::make_shared<Image>(roi, chans);
        int s = std::max(1, p.i("boxsize"));
        bool lines = p.b("centerlines");
        Info inf = generatorInfo(p);
        ev.parallelRows(roi, [&](int y) {
            for (int x = roi.x0; x < roi.x1; ++x) {
                bool odd = (((x / s) + (y / s)) & 1) != 0;
                bool line = lines && (x == inf.format.w / 2 || y == inf.format.h / 2);
                for (auto& c : chans) {
                    int idx = int(std::find(kRGBA.begin(), kRGBA.end(), c) - kRGBA.begin());
                    double v = line ? p.f("linecolor", idx) : odd ? p.f("color1", idx) : p.f("color0", idx);
                    out->row(c, y)[x - roi.x0] = float(v);
                }
            }
        });
        return out;
    }
};
VC_REGISTER_OP(OpDesc{"CheckerBoard", "Image", "Checker pattern for checking filtering and alignment.", {}, 0,
                      {K::integer("boxsize", "box size", 64, 1, 512), K::color("color0", "color 0", {0.1, 0.1, 0.1, 1}),
                       K::color("color1", "color 1", {0.5, 0.5, 0.5, 1}), K::boolean("centerlines", "centre lines", true),
                       K::color("linecolor", "line color", {1, 1, 0, 1}), formatKnob()},
                      1, maker<CheckerBoard>()});

// ------------------------------------------------------------ Rectangle
class Rectangle : public Op {
public:
    Info info(Eval& ev, const Params& p) const override {
        Info i = ev.connected(0) ? ev.inputInfo(0) : generatorInfo(p);
        if (!ev.connected(0)) i.bbox = Box{};
        Box area = areaBox(p).grow(int(std::ceil(p.f("softness"))) + 1, int(std::ceil(p.f("softness"))) + 1);
        i.bbox = i.bbox.unite(area.intersect(Box{-100000, -100000, 100000, 100000}));
        i.channels = unite(i.channels, kRGBA);
        return i;
    }
    static Box areaBox(const Params& p) {
        double x0 = p.f("area", 0), y0 = p.f("area", 1), x1 = p.f("area", 2), y1 = p.f("area", 3);
        return Box{int(std::floor(std::min(x0, x1))), int(std::floor(std::min(y0, y1))),
                   int(std::ceil(std::max(x0, x1))), int(std::ceil(std::max(y0, y1)))};
    }
    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        ImagePtr bg = ev.input(0, roi, chans);
        auto out = copyOf(*bg, roi, chans);
        double x0 = std::min(p.f("area", 0), p.f("area", 2)), x1 = std::max(p.f("area", 0), p.f("area", 2));
        double y0 = std::min(p.f("area", 1), p.f("area", 3)), y1 = std::max(p.f("area", 1), p.f("area", 3));
        double soft = std::max(0.0, p.f("softness"));
        auto cover = [soft](double lo, double hi, double c) {
            // Coverage of the pixel [c-0.5, c+0.5] by [lo, hi], softened by `soft`.
            double w = soft + 1.0;
            double a = std::min(1.0, std::max(0.0, (c - lo) / w + 0.5));
            double b = std::min(1.0, std::max(0.0, (hi - c) / w + 0.5));
            return a * b;
        };
        ev.parallelRows(roi, [&](int y) {
            double cy = cover(y0, y1, y + 0.5);
            for (int x = roi.x0; x < roi.x1; ++x) {
                double m = cy * cover(x0, x1, x + 0.5);
                if (m <= 0) continue;
                float a = float(p.f("color", 3) * m);
                for (auto& c : chans) {
                    int idx = int(std::find(kRGBA.begin(), kRGBA.end(), c) - kRGBA.begin());
                    if (idx >= 4) continue;
                    float* d = out->row(c, y) + (x - roi.x0);
                    *d = float(p.f("color", idx) * m) + *d * (1.0f - a);
                }
            }
        });
        return out;
    }
};
VC_REGISTER_OP(OpDesc{"Rectangle", "Image", "Soft-edged rectangle, composited over the optional input.", {"bg"}, 0,
                      {K::box("area", "area", {480, 270, 1440, 810}), K::flt("softness", "softness", 0, 0, 100, "pixels"),
                       K::color("color", "color", {1, 1, 1, 1}), formatKnob()},
                      1, maker<Rectangle>()});

// ------------------------------------------------------------ Noise
inline float fade(float t) { return t * t * t * (t * (t * 6 - 15) + 10); }

float gradDot(uint32_t h, float x, float y, float z) {
    switch (h % 12) {
        case 0: return x + y;   case 1: return -x + y;  case 2: return x - y;   case 3: return -x - y;
        case 4: return x + z;   case 5: return -x + z;  case 6: return x - z;   case 7: return -x - z;
        case 8: return y + z;   case 9: return -y + z;  case 10: return y - z;  default: return -y - z;
    }
}

float perlin3(float x, float y, float z, uint32_t seed) {
    int ix = int(std::floor(x)), iy = int(std::floor(y)), iz = int(std::floor(z));
    float fx = x - ix, fy = y - iy, fz = z - iz;
    float u = fade(fx), v = fade(fy), w = fade(fz);
    auto g = [&](int dx, int dy, int dz) {
        return gradDot(hash32(ix + dx, iy + dy, iz + dz, seed), fx - dx, fy - dy, fz - dz);
    };
    float x00 = g(0, 0, 0) + u * (g(1, 0, 0) - g(0, 0, 0));
    float x10 = g(0, 1, 0) + u * (g(1, 1, 0) - g(0, 1, 0));
    float x01 = g(0, 0, 1) + u * (g(1, 0, 1) - g(0, 0, 1));
    float x11 = g(0, 1, 1) + u * (g(1, 1, 1) - g(0, 1, 1));
    float y0 = x00 + v * (x10 - x00), y1 = x01 + v * (x11 - x01);
    return (y0 + w * (y1 - y0)) * 0.7f;
}

float value3(float x, float y, float z, uint32_t seed) {
    int ix = int(std::floor(x)), iy = int(std::floor(y)), iz = int(std::floor(z));
    float u = fade(x - ix), v = fade(y - iy), w = fade(z - iz);
    auto h = [&](int dx, int dy, int dz) { return hashFloat(hash32(ix + dx, iy + dy, iz + dz, seed)) * 2.0f - 1.0f; };
    float x00 = h(0, 0, 0) + u * (h(1, 0, 0) - h(0, 0, 0));
    float x10 = h(0, 1, 0) + u * (h(1, 1, 0) - h(0, 1, 0));
    float x01 = h(0, 0, 1) + u * (h(1, 0, 1) - h(0, 0, 1));
    float x11 = h(0, 1, 1) + u * (h(1, 1, 1) - h(0, 1, 1));
    float y0 = x00 + v * (x10 - x00), y1 = x01 + v * (x11 - x01);
    return y0 + w * (y1 - y0);
}

class Noise : public Op {
public:
    Info info(Eval& ev, const Params& p) const override {
        if (ev.connected(0)) {
            Info i = ev.inputInfo(0);
            if (p.s("mode") == "generate") {
                i.bbox = i.format.box();
                i.channels = unite(i.channels, kRGBA);
            }
            return i;
        }
        return generatorInfo(p);
    }
    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        std::string mode = p.s("mode");
        ImagePtr in = ev.input(0, roi, unite(chans, {"rgba.red", "rgba.green", "rgba.blue"}));
        auto out = copyOf(*in, roi, chans);
        double frame = p.f("__frame");
        uint32_t seed = uint32_t(p.i("seed"));
        double sx = std::max(1e-3, p.f("size", 0)), sy = std::max(1e-3, p.f("size", 1));
        double z = p.f("zoffset") + frame * p.f("speed");
        int oct = std::max(1, std::min(10, p.i("octaves")));
        double lac = p.f("lacunarity"), gain = p.f("gain");
        bool perlin = p.s("type") == "perlin";
        double amp = p.f("amplitude"), off = p.f("offset");
        auto fbm = [&](double x, double y, double zz, uint32_t s) {
            double sum = 0, a = 1, norm = 0, f = 1;
            for (int o = 0; o < oct; ++o) {
                float n = perlin ? perlin3(float(x * f), float(y * f), float(zz), s + o * 1013u)
                                 : value3(float(x * f), float(y * f), float(zz), s + o * 1013u);
                sum += a * n;
                norm += a;
                a *= gain;
                f *= lac;
            }
            return norm > 0 ? sum / norm : 0.0;
        };
        const char* rgb[3] = {"rgba.red", "rgba.green", "rgba.blue"};
        if (mode == "grain") {
            double gsize = std::max(0.25, p.f("grainSize"));
            double irr = p.f("irregularity");
            double resp = p.f("response");
            uint32_t fseed = hash32(uint32_t(int(std::floor(frame))) * 7919u ^ seed);
            ev.parallelRows(roi, [&](int y) {
                int n = roi.w();
                std::vector<float> rr(n), gg(n), bb(n);
                in->readRow(rgb[0], y, roi.x0, n, rr.data());
                in->readRow(rgb[1], y, roi.x0, n, gg.data());
                in->readRow(rgb[2], y, roi.x0, n, bb.data());
                for (int x = roi.x0; x < roi.x1; ++x) {
                    float r = rr[x - roi.x0], g = gg[x - roi.x0], b = bb[x - roi.x0];
                    float lum = std::max(0.0f, luma709(r, g, b));
                    // Grain is strongest in the mid-tones; `response` shifts it towards highlights.
                    double weight = std::pow(std::min(1.0f, lum), std::max(0.0, resp)) * (1.0 - 0.5 * std::min(1.0f, lum));
                    for (int c = 0; c < 3; ++c) {
                        if (!hasChannel(chans, rgb[c])) continue;
                        uint32_t cs = fseed + uint32_t(c) * 104729u;
                        double coherent = value3(float((x + 0.5) / gsize), float((y + 0.5) / gsize), 0.0f, cs);
                        double white = hashFloat(hash32(x, y, c, cs)) * 2.0 - 1.0;
                        double n = coherent * (1 - irr) + white * irr;
                        float& d = out->row(rgb[c], y)[x - roi.x0];
                        d = float(d + p.f("intensity", c) * weight * n);
                    }
                }
            });
            return out;
        }
        ev.parallelRows(roi, [&](int y) {
            for (int x = roi.x0; x < roi.x1; ++x) {
                double nx = (x + 0.5) / sx, ny = (y + 0.5) / sy;
                double base = fbm(nx, ny, z, seed);
                for (size_t ci = 0; ci < kRGBA.size(); ++ci) {
                    const std::string& c = kRGBA[ci];
                    if (!hasChannel(chans, c)) continue;
                    double n = p.b("mono") || ci == 3 ? base : fbm(nx, ny, z, seed + uint32_t(ci) * 7777u);
                    double v = (n * 0.5 + 0.5) * amp + off;
                    v *= p.f("color", int(ci));
                    float& d = out->row(c, y)[x - roi.x0];
                    if (mode == "add") d = float(d + (v - 0.5 * amp * p.f("color", int(ci))));
                    else d = float(v);
                }
            }
        });
        return out;
    }
};
VC_REGISTER_OP(OpDesc{"Noise", "Image",
                      "Procedural fBm noise (generate or add) and film grain. Deterministic per seed, frame and pixel.",
                      {"bg"}, 0,
                      {K::enm("mode", "mode", {"generate", "add", "grain"}),
                       K::enm("type", "type", {"perlin", "value"}),
                       K::wh("size", "size", 120, 120, 1, 1000, "Feature size in pixels"),
                       K::flt("zoffset", "z", 0, 0, 10), K::flt("speed", "speed", 0.05, 0, 1, "z change per frame"),
                       K::integer("octaves", "octaves", 4, 1, 10), K::flt("lacunarity", "lacunarity", 2, 1, 4),
                       K::flt("gain", "gain", 0.5, 0, 1), K::flt("amplitude", "amplitude", 1, 0, 2),
                       K::flt("offset", "offset", 0, -1, 1), K::color("color", "color", {1, 1, 1, 1}),
                       K::boolean("mono", "monochrome", true), K::integer("seed", "seed", 0, 0, 1000),
                       K::group(K::color("intensity", "intensity", {0.05, 0.05, 0.05}, 0, 0.3), "Grain"),
                       K::group(K::flt("grainSize", "grain size", 1.5, 0.25, 8, "pixels"), "Grain"),
                       K::group(K::flt("irregularity", "irregularity", 0.4, 0, 1), "Grain"),
                       K::group(K::flt("response", "response", 0.5, 0, 2, "Luminance dependency"), "Grain"),
                       formatKnob()},
                      1, maker<Noise>()});

}  // namespace
}  // namespace vc::ops
