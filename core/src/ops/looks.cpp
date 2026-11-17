// Vignette, lens flare, radial lens distortion and channel-based edge drawing.
#include "common.h"

namespace vc::ops {
namespace {

Info passthroughInfo(Eval& ev) {
    return ev.connected(0) ? ev.inputInfo(0) : emptyInfo();
}

int imageWidth(const Info& info) { return std::max(1, info.format.w); }
int imageHeight(const Info& info) { return std::max(1, info.format.h); }

std::pair<double, double> pixelCenter(const Params& p, const Info& info, const std::string& knob) {
    return {p.f(knob, 0) * imageWidth(info), p.f(knob, 1) * imageHeight(info)};
}

class Vignette : public Op {
public:
    Info info(Eval& ev, const Params&) const override { return passthroughInfo(ev); }

    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        Info inf = ev.inputInfo(0);
        ImagePtr in = ev.input(0, roi, chans);
        auto out = copyOf(*in, roi, chans);
        const auto center = pixelCenter(p, inf, "center");
        const double aspect = std::max(0.01, p.f("aspect"));
        const double radius = std::max(0.0, p.f("radius"));
        const double softness = std::max(1e-6, p.f("softness"));
        const double amount = clamp01(float(p.f("amount")));
        const double falloff = std::max(0.1, p.f("falloff"));
        const auto selected = intersect(resolveChannels(p.s("channels"), inf.channels), chans);
        const auto color = p.v("color");
        ev.parallelRows(roi, [&](int y) {
            for (int x = roi.x0; x < roi.x1; ++x) {
                double dx = (x + 0.5 - center.first) / imageWidth(inf) * 2.0 * aspect;
                double dy = (y + 0.5 - center.second) / imageHeight(inf) * 2.0 / aspect;
                double t = clamp01(float((std::hypot(dx, dy) - radius) / softness));
                float k = float(amount * std::pow(t, falloff));
                size_t i = size_t(y - roi.y0) * roi.w() + x - roi.x0;
                for (const auto& channel : selected) {
                    int c = channel == "rgba.red" ? 0 : channel == "rgba.green" ? 1 :
                            channel == "rgba.blue" ? 2 : -1;
                    if (c < 0 || !out->plane(channel)) continue;
                    float target = c < int(color.size()) ? float(color[c]) : 0.0f;
                    out->plane(channel)[i] += (target - out->plane(channel)[i]) * k;
                }
            }
        });
        applyMaskMix(ev, p, 1, *in, *out, selected);
        return out;
    }
};

VC_REGISTER_OP(OpDesc{"Vignette", "Color", "Darken or tint the image edges with a soft elliptical falloff.",
                      {"", "mask"}, 0,
                      concat({K::xy("center", "center (normalized)", 0.5, 0.5),
                              K::flt("radius", "radius", 0.55, 0, 1.5),
                              K::flt("softness", "softness", 0.35, 0.001, 1.5),
                              K::flt("amount", "amount", 0.8, 0, 1),
                              K::flt("aspect", "aspect", 1.0, 0.1, 4),
                              K::flt("falloff", "falloff", 1.5, 0.1, 8),
                              K::color("color", "edge color", {0, 0, 0}),
                              K::channels("channels", "channels", "rgb")},
                             maskKnobs()),
                      1, maker<Vignette>()});

class LensFlare : public Op {
public:
    Info info(Eval& ev, const Params&) const override { return passthroughInfo(ev); }

    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        Info inf = ev.inputInfo(0);
        ImagePtr in = ev.input(0, roi, chans);
        auto out = copyOf(*in, roi, chans);
        const double w = imageWidth(inf), h = imageHeight(inf);
        const double scale = std::max(1.0, std::min(w, h));
        const auto source = pixelCenter(p, inf, "position");
        const double sx = source.first / scale, sy = source.second / scale;
        const double cx = w * 0.5 / scale, cy = h * 0.5 / scale;
        const double size = std::max(0.005, p.f("size"));
        const double intensity = std::max(0.0, p.f("intensity"));
        const double threshold = p.f("threshold");
        float sourceEnergy = 1.0f;
        const int px = int(std::floor(source.first)), py = int(std::floor(source.second));
        Box probe{px, py, px + 1, py + 1};
        if (px >= inf.bbox.x0 && px < inf.bbox.x1 && py >= inf.bbox.y0 && py < inf.bbox.y1) {
            ImagePtr point = ev.input(0, probe, {"rgba.red", "rgba.green", "rgba.blue"});
            float rgb[3]{};
            point->readRow("rgba.red", py, px, 1, &rgb[0]);
            point->readRow("rgba.green", py, px, 1, &rgb[1]);
            point->readRow("rgba.blue", py, px, 1, &rgb[2]);
            float peak = std::max({rgb[0], rgb[1], rgb[2]});
            sourceEnergy = clamp01((peak - float(threshold)) / std::max(0.001f, 1.0f - float(threshold)));
        }
        const auto color = p.v("color");
        int rays = std::max(0, std::min(16, p.i("rays")));
        int ghosts = std::max(0, std::min(8, p.i("ghosts")));
        const double angle = p.f("angle") * kPi / 180.0;
        const double lineX = cx - sx, lineY = cy - sy;
        const double lineLen = std::hypot(lineX, lineY);
        const double dirX = lineLen > 1e-8 ? lineX / lineLen : 1.0;
        const double dirY = lineLen > 1e-8 ? lineY / lineLen : 0.0;
        ev.parallelRows(roi, [&](int y) {
            for (int x = roi.x0; x < roi.x1; ++x) {
                const double nx = (x + 0.5) / scale, ny = (y + 0.5) / scale;
                const double dx = nx - sx, dy = ny - sy;
                const double r = std::hypot(dx, dy);
                double flare = std::exp(-(r * r) / (size * size * 0.05)) * 1.8;
                flare += std::exp(-(r * r) / (size * size * 1.2)) * 0.4;

                const double lx = nx - sx, ly = ny - sy;
                const double along = lx * dirX + ly * dirY;
                const double across = std::abs(lx * dirY - ly * dirX);
                flare += std::exp(-across * across / (size * size * 0.0008)) *
                         std::exp(-along * along / (size * size * 0.35)) * 0.55;
                for (int i = 0; i < ghosts; ++i) {
                    double t = 0.35 + 0.2 * i;
                    double gx = sx + (cx - sx) * t, gy = sy + (cy - sy) * t;
                    double gr = std::hypot(nx - gx, ny - gy);
                    double radius = size * (0.025 + 0.012 * (i % 3));
                    flare += std::exp(-gr * gr / (radius * radius)) * 0.32;
                }
                for (int i = 0; i < rays; ++i) {
                    double a = angle + kPi * i / rays;
                    double vx = std::cos(a), vy = std::sin(a);
                    double q = dx * vx + dy * vy;
                    double d = std::abs(dx * vy - dy * vx);
                    if (q > 0) flare += std::exp(-d * d / (size * size * 0.00015)) *
                                              std::exp(-q / (size * 0.45)) * 0.25;
                }
                float k = float(flare * intensity * sourceEnergy);
                size_t i = size_t(y - roi.y0) * roi.w() + x - roi.x0;
                for (int c = 0; c < 3; ++c) {
                    const char* channel = c == 0 ? "rgba.red" : c == 1 ? "rgba.green" : "rgba.blue";
                    if (float* plane = out->plane(channel))
                        plane[i] += k * (c < int(color.size()) ? float(color[c]) : 1.0f);
                }
            }
        });
        applyMaskMix(ev, p, 1, *in, *out, {"rgba.red", "rgba.green", "rgba.blue"});
        return out;
    }
};

VC_REGISTER_OP(OpDesc{"LensFlare", "Filter", "Generate a highlight-driven lens flare with rays and optical ghosts.",
                      {"", "mask"}, 0,
                      concat({K::xy("position", "source (normalized)", 0.5, 0.5),
                              K::flt("intensity", "intensity", 1.0, 0, 10),
                              K::flt("threshold", "source threshold", 0.8, 0, 1),
                              K::flt("size", "size", 0.6, 0.01, 2),
                              K::integer("rays", "rays", 6, 0, 16),
                              K::integer("ghosts", "ghosts", 4, 0, 8),
                              K::flt("angle", "ray angle", 0, -180, 180),
                              K::color("color", "color", {1, 0.85, 0.6})},
                             maskKnobs()),
                      1, maker<LensFlare>()});

class Distortion : public Op {
public:
    Info info(Eval& ev, const Params&) const override { return passthroughInfo(ev); }

    static std::pair<double, double> sourcePoint(const Params& p, const Info& inf, double x, double y) {
        const double w = imageWidth(inf), h = imageHeight(inf);
        const double scale = std::max(1.0, std::min(w, h));
        auto center = pixelCenter(p, inf, "center");
        double nx = (x - center.first) / scale, ny = (y - center.second) / scale;
        double r2 = nx * nx + ny * ny;
        double factor = 1.0 + p.f("k1") * r2 + p.f("k2") * r2 * r2;
        return {center.first + nx * factor * scale, center.second + ny * factor * scale};
    }

    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        Info inf = ev.inputInfo(0);
        auto selected = intersect(resolveChannels(p.s("channels"), inf.channels), chans);
        if (selected.empty()) return ev.input(0, roi, chans);

        Box need{roi.x1, roi.y1, roi.x0, roi.y0};
        for (int j = 0; j <= 16; ++j)
            for (int i = 0; i <= 16; ++i) {
                double x = roi.x0 + roi.w() * i / 16.0;
                double y = roi.y0 + roi.h() * j / 16.0;
                auto q = sourcePoint(p, inf, x, y);
                need.x0 = std::min(need.x0, int(std::floor(q.first)));
                need.y0 = std::min(need.y0, int(std::floor(q.second)));
                need.x1 = std::max(need.x1, int(std::ceil(q.first)) + 1);
                need.y1 = std::max(need.y1, int(std::ceil(q.second)) + 1);
            }
        need = need.grow(filterRadius(p.s("filter") == "nearest" ? FilterImpulse :
                                     p.s("filter") == "bilinear" ? FilterBilinear :
                                     p.s("filter") == "lanczos3" ? FilterLanczos : FilterCubic),
                         filterRadius(p.s("filter") == "nearest" ? FilterImpulse :
                                     p.s("filter") == "bilinear" ? FilterBilinear :
                                     p.s("filter") == "lanczos3" ? FilterLanczos : FilterCubic));
        ImagePtr in = ev.input(0, need.unite(roi), chans);
        auto out = copyOf(*in, roi, chans);
        int filter = p.s("filter") == "nearest" ? FilterImpulse :
                     p.s("filter") == "bilinear" ? FilterBilinear :
                     p.s("filter") == "lanczos3" ? FilterLanczos : FilterCubic;
        for (const auto& channel : selected) {
            const float* src = in->plane(channel);
            float* dst = out->plane(channel);
            if (!src || !dst) continue;
            ev.parallelRows(roi, [&](int y) {
                for (int x = roi.x0; x < roi.x1; ++x) {
                    auto q = sourcePoint(p, inf, x + 0.5, y + 0.5);
                    dst[size_t(y - roi.y0) * roi.w() + x - roi.x0] = sample(src, in->bbox(), q.first, q.second, filter);
                }
            });
        }
        applyMaskMix(ev, p, 1, *in, *out, selected);
        return out;
    }
};

VC_REGISTER_OP(OpDesc{"Distortion", "Transform", "Apply radial barrel or pincushion lens distortion.",
                      {"", "mask"}, 0,
                      concat({K::xy("center", "center (normalized)", 0.5, 0.5),
                              K::flt("k1", "radial k1", 0, -1, 1),
                              K::flt("k2", "radial k2", 0, -1, 1),
                              K::enm("filter", "filter", {"nearest", "bilinear", "cubic", "lanczos3"}, "cubic"),
                              K::channels("channels", "channels", "rgba")},
                             maskKnobs()),
                      1, maker<Distortion>()});

float channelAt(const Image& img, const std::string& channel, int x, int y) {
    const Box& bbox = img.bbox();
    if (x < bbox.x0 || x >= bbox.x1 || y < bbox.y0 || y >= bbox.y1) return 0.0f;
    const float* plane = img.plane(channel);
    if (!plane) return 0.0f;
    return plane[size_t(y - img.bbox().y0) * img.bbox().w() + x - img.bbox().x0];
}

class EdgeDetect : public Op {
public:
    Info info(Eval& ev, const Params& p) const override {
        Info inf = passthroughInfo(ev);
        if (inf.channels.empty()) return inf;
        inf.channels = unite(inf.channels, {"rgba.alpha", "rgba.red", "rgba.green", "rgba.blue", p.s("output")});
        return inf;
    }

    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        Info inf = ev.inputInfo(0);
        int width = std::max(1, std::min(128, p.i("width")));
        int left = -(width - 1) / 2, right = width / 2;
        std::string inputChannel = p.s("channel");
        std::string outputChannel = p.s("output");
        Box need = roi.grow(std::max(std::abs(left), std::abs(right)) + 1,
                            std::max(std::abs(left), std::abs(right)) + 1);
        ChannelList needed = chans;
        needed.push_back(inputChannel);
        ImagePtr in = ev.input(0, need, sortedChannels(needed));
        auto out = copyOf(*in, roi, chans);
        std::vector<float> edge(size_t(roi.w()) * roi.h());
        const double threshold = std::max(0.0, p.f("threshold"));
        ev.parallelRows(roi, [&](int y) {
            for (int x = roi.x0; x < roi.x1; ++x) {
                float strongest = 0.0f;
                for (int oy = left; oy <= right; ++oy)
                    for (int ox = left; ox <= right; ++ox) {
                        int px = x + ox, py = y + oy;
                        float gx = -channelAt(*in, inputChannel, px - 1, py - 1) +
                                   channelAt(*in, inputChannel, px + 1, py - 1) -
                                   2 * channelAt(*in, inputChannel, px - 1, py) +
                                   2 * channelAt(*in, inputChannel, px + 1, py) -
                                   channelAt(*in, inputChannel, px - 1, py + 1) +
                                   channelAt(*in, inputChannel, px + 1, py + 1);
                        float gy = -channelAt(*in, inputChannel, px - 1, py - 1) -
                                   2 * channelAt(*in, inputChannel, px, py - 1) -
                                   channelAt(*in, inputChannel, px + 1, py - 1) +
                                   channelAt(*in, inputChannel, px - 1, py + 1) +
                                   2 * channelAt(*in, inputChannel, px, py + 1) +
                                   channelAt(*in, inputChannel, px + 1, py + 1);
                        strongest = std::max(strongest, float(std::hypot(gx, gy) / 4.0));
                    }
                float alpha = clamp01(float((strongest - threshold) / std::max(1e-6, 1.0 - threshold)));
                edge[size_t(y - roi.y0) * roi.w() + x - roi.x0] = alpha * clamp01(float(p.f("opacity")));
            }
        });
        std::vector<std::string> changed;
        auto addChanged = [&](const std::string& channel) {
            if (std::find(changed.begin(), changed.end(), channel) == changed.end()) changed.push_back(channel);
        };
        if (float* plane = out->plane(outputChannel)) {
            std::copy(edge.begin(), edge.end(), plane);
            addChanged(outputChannel);
        }
        const auto color = p.v("color");
        const char* rgb[3] = {"rgba.red", "rgba.green", "rgba.blue"};
        for (int c = 0; c < 3; ++c)
            if (float* plane = out->plane(rgb[c])) {
                float col = c < int(color.size()) ? float(color[c]) : 1.0f;
                for (size_t i = 0; i < edge.size(); ++i) plane[i] = col * edge[i] + plane[i] * (1.0f - edge[i]);
                addChanged(rgb[c]);
            }
        if (float* alpha = out->plane("rgba.alpha")) {
            for (size_t i = 0; i < edge.size(); ++i) alpha[i] = edge[i] + alpha[i] * (1.0f - edge[i]);
            addChanged("rgba.alpha");
        }
        applyMaskMix(ev, p, 1, *in, *out, changed);
        return out;
    }
};

VC_REGISTER_OP(OpDesc{"EdgeDetect", "Filter", "Detect edges in a selected channel and draw them with a pixel-width stroke.",
                      {"", "mask"}, 0,
                      concat({K::channel("channel", "edge channel", "rgba.alpha"),
                              K::flt("threshold", "threshold", 0.1, 0, 1),
                              K::integer("width", "line width (pixels)", 1, 1, 128),
                              K::channel("output", "edge matte", "edge"),
                              K::flt("opacity", "opacity", 1, 0, 1),
                              K::color("color", "line color", {1, 1, 1})},
                             maskKnobs()),
                      1, maker<EdgeDetect>()});

}  // namespace
}  // namespace vc::ops
