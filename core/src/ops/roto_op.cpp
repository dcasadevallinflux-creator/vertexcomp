// Roto: rasterizes the node's shape list into a matte channel.
#include "common.h"
#include "vc/roto.h"

namespace vc::ops {
namespace {

class Roto : public Op {
public:
    void hashExtra(Hasher& h, const Node& n, double frame) const override {
        if (n.roto) n.roto->hash(h, frame);
    }
    Info info(Eval& ev, const Params& p) const override {
        Info i;
        if (ev.connected(0)) {
            i = ev.inputInfo(0);
        } else {
            i.format.w = std::max(1, int(p.f("format", 0)));
            i.format.h = std::max(1, int(p.f("format", 1)));
            i.bbox = Box{};
        }
        i.bbox = i.bbox.unite(i.format.box());
        ChannelList add{p.s("output")};
        if (p.b("rgb")) add = unite(add, {"rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"});
        i.channels = unite(i.channels, add);
        return i;
    }
    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        std::shared_ptr<RotoData> data;
        {
            std::lock_guard<std::recursive_mutex> lk(ev.engine.graph().mutex());
            data = ev.node->roto;
        }
        ImagePtr in = ev.input(0, roi, chans);
        auto out = copyOf(*in, roi, chans);
        std::vector<float> m(size_t(roi.w()) * roi.h());
        if (data) rasterizeRoto(*data, ev.frame, roi, m.data(), ev.cancel);
        std::string outCh = p.s("output"), comb = p.s("combine");
        if (float* o = out->plane(outCh)) {
            for (size_t i = 0; i < m.size(); ++i) {
                if (comb == "over") o[i] = m[i] + o[i] * (1.0f - m[i]);
                else if (comb == "multiply") o[i] *= m[i];
                else if (comb == "max") o[i] = std::max(o[i], m[i]);
                else o[i] = m[i];
            }
        }
        if (p.b("rgb")) {
            const char* rgb[3] = {"rgba.red", "rgba.green", "rgba.blue"};
            for (int c = 0; c < 3; ++c) {
                float* o = out->plane(rgb[c]);
                if (!o) continue;
                float col = float(p.f("color", c));
                for (size_t i = 0; i < m.size(); ++i) o[i] = col * m[i] + o[i] * (1.0f - m[i]);
            }
            if (outCh != "rgba.alpha")
                if (float* a = out->plane("rgba.alpha"))
                    for (size_t i = 0; i < m.size(); ++i) a[i] = m[i] + a[i] * (1.0f - m[i]);
        }
        return out;
    }
};
VC_REGISTER_OP(OpDesc{"Roto", "Draw",
                      "Animated Bezier shapes with variable feather, blend modes and motion blur, rasterized to a "
                      "matte. Edit shapes in the viewer.",
                      {"bg"}, 0,
                      {K::channel("output", "output", "rgba.alpha"),
                       K::enm("combine", "combine with input", {"replace", "over", "max", "multiply"}),
                       K::boolean("rgb", "also write colour", false, "Composite the colour below over rgb"),
                       K::color("color", "colour", {1, 1, 1}),
                       K::wh("format", "format", 1920, 1080, 1, 8192, "Used when nothing is connected")},
                      1, maker<Roto>()});

class Paint : public Op {
public:
    void hashExtra(Hasher& h, const Node& n, double frame) const override {
        if (n.roto) n.roto->hash(h, frame);
    }

    Info info(Eval& ev, const Params& p) const override {
        Info i;
        if (ev.connected(0)) i = ev.inputInfo(0);
        else {
            i.format.w = std::max(1, int(p.f("format", 0)));
            i.format.h = std::max(1, int(p.f("format", 1)));
            i.bbox = Box{};
        }
        i.bbox = i.bbox.unite(i.format.box());
        i.channels = unite(i.channels, {"rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"});
        return i;
    }

    ImagePtr render(Eval& ev, const Params&, const Box& roi, const ChannelList& chans) const override {
        std::shared_ptr<RotoData> data;
        {
            std::lock_guard<std::recursive_mutex> lk(ev.engine.graph().mutex());
            data = ev.node->roto;
        }
        ImagePtr in = ev.input(0, roi, chans);
        auto out = copyOf(*in, roi, chans);
        if (!data) return out;

        const char* rgb[] = {"rgba.red", "rgba.green", "rgba.blue"};
        const size_t count = size_t(roi.w()) * roi.h();
        std::vector<float> coverage(count);
        for (const auto& shape : data->shapes) {
            if (!shape.visible) continue;
            RotoData one;
            one.shapes.push_back(shape);
            std::fill(coverage.begin(), coverage.end(), 0.0f);
            rasterizeRoto(one, ev.frame, roi, coverage.data(), ev.cancel);
            for (size_t c = 0; c < 3; ++c) {
                float* plane = out->plane(rgb[c]);
                if (!plane) continue;
                const float color = float(c < shape.color.size() ? shape.color[c] : 1.0);
                for (size_t i = 0; i < count; ++i) {
                    const float alpha = coverage[i];
                    plane[i] = color * alpha + plane[i] * (1.0f - alpha);
                }
            }
            if (float* plane = out->plane("rgba.alpha")) {
                for (size_t i = 0; i < count; ++i) {
                    const float alpha = coverage[i];
                    plane[i] = alpha + plane[i] * (1.0f - alpha);
                }
            }
        }
        return out;
    }
};

VC_REGISTER_OP(OpDesc{"Paint", "Draw", "Paint freehand colour strokes over an image.", {"bg"}, 0,
                      {K::flt("size", "brush size", 20, 1, 512), K::color("color", "brush colour", {1, 0, 0, 1}),
                       K::wh("format", "format", 1920, 1080, 1, 8192, "Used when nothing is connected")},
                      1, maker<Paint>()});

}  // namespace
}  // namespace vc::ops
