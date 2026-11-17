// Merge, MergeLayer, Switch, Dot, NoOp and the time ops.
#include "common.h"

namespace vc::ops {
namespace {

// ------------------------------------------------------------ Merge
const std::vector<std::string> kMergeOps{"over",      "under",       "plus",        "multiply", "screen",
                                         "overlay",   "soft-light",  "hard-light",  "color-dodge",
                                         "color-burn", "minus",      "from",        "difference", "min",
                                         "max",       "atop",        "in",          "out",       "mask",
                                         "stencil",   "divide",      "average",     "copy",      "hypot"};

// A, B: channel values; a, b: the inputs' alphas.
inline float mergeValue(int op, float A, float B, float a, float b) {
    switch (op) {
        case 0: return A + B * (1.0f - a);
        case 1: return A * (1.0f - b) + B;
        case 2: return A + B;
        case 3: return A * B;
        case 4: return A + B - A * B;
        case 5: return B <= 0.5f ? 2 * A * B : 1 - 2 * (1 - A) * (1 - B);
        case 6: return (1 - 2 * A) * B * B + 2 * A * B;
        case 7: return A <= 0.5f ? 2 * A * B : 1 - 2 * (1 - A) * (1 - B);
        case 8: return A >= 1.0f ? std::max(A, B) : B / (1.0f - A);
        case 9: return A <= 0.0f ? 0.0f : 1.0f - (1.0f - B) / A;
        case 10: return A - B;
        case 11: return B - A;
        case 12: return std::fabs(A - B);
        case 13: return std::min(A, B);
        case 14: return std::max(A, B);
        case 15: return A * b + B * (1.0f - a);
        case 16: return A * b;
        case 17: return A * (1.0f - b);
        case 18: return B * a;
        case 19: return B * (1.0f - a);
        case 20: return B != 0.0f ? A / B : 0.0f;
        case 21: return 0.5f * (A + B);
        case 22: return A;
        default: return std::sqrt(A * A + B * B);
    }
}

// Photographic modes combine alpha as a union rather than with their colour formula.
inline bool unionAlpha(int op) { return op >= 4 && op <= 9; }

std::vector<std::string> slotsFor(const std::string& sel, const ChannelList& avail) {
    if (sel == "rgb") return {"rgba.red", "rgba.green", "rgba.blue", ""};
    if (sel == "alpha") return {"", "", "", "rgba.alpha"};
    if (sel == "rgba") return {"rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"};
    if (sel == "none") return {"", "", "", ""};
    auto s = layerSlots(sel, avail);
    // A layer that does not exist yet is created with standard names.
    bool any = false;
    for (auto& c : s) any |= !c.empty();
    if (!any) return {sel + ".red", sel + ".green", sel + ".blue", sel + ".alpha"};
    return s;
}

class Merge : public Op {
public:
    Info info(Eval& ev, const Params& p) const override {
        int base = -1;
        Info out;
        for (int i = 0; i < 8; ++i)
            if (ev.connected(i)) {
                if (base < 0) {
                    base = i;
                    out = ev.inputInfo(i);
                } else {
                    out.bbox = out.bbox.unite(ev.inputInfo(i).bbox);
                }
            }
        if (base < 0) return emptyInfo();
        std::string mode = p.s("bbox");
        Info B = ev.inputInfo(0), A = ev.inputInfo(1);
        if (mode == "intersection" && ev.connected(0) && ev.connected(1)) out.bbox = A.bbox.intersect(B.bbox);
        else if (mode == "A" && ev.connected(1)) out.bbox = A.bbox;
        else if (mode == "B" && ev.connected(0)) out.bbox = B.bbox;
        ChannelList outCh = out.channels;
        for (int input = 0; input < 8; ++input) {
            if (!ev.connected(input)) continue;
            Info source = ev.inputInfo(input);
            std::string knob = input == 0 ? "Bchannels" : input == 1 ? "Achannels" :
                               std::string(1, char('A' + input)) + "channels";
            auto ss = slotsFor(p.s(knob), source.channels);
            auto os = slotsFor(p.s("output"), source.channels);
            for (int i = 0; i < 4; ++i)
                if (!os[i].empty() && !ss[i].empty() && hasChannel(source.channels, ss[i])) outCh.push_back(os[i]);
        }
        out.channels = sortedChannels(outCh);
        return out;
    }
    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        int base = ev.connected(0) ? 0 : -1;
        for (int i = 1; base < 0 && i < 8; ++i)
            if (ev.connected(i)) base = i;
        if (base < 0) return std::make_shared<Image>(roi, chans);
        int op = int(std::find(kMergeOps.begin(), kMergeOps.end(), p.s("operation")) - kMergeOps.begin());
        Info baseInfo = ev.inputInfo(base);
        ChannelList baseNeed = chans;
        baseNeed.push_back("rgba.alpha");
        ImagePtr first = ev.input(base, roi, intersect(sortedChannels(baseNeed), baseInfo.channels));
        auto out = copyOf(*first, roi, chans);
        ChannelList changed;
        ChannelList available;
        for (int input = 0; input < 8; ++input)
            if (ev.connected(input)) available = unite(available, ev.inputInfo(input).channels);
        auto os = slotsFor(p.s("output"), available);
        for (int input = 0; input < 8; ++input) {
            if (input == base || !ev.connected(input)) continue;
            Info sourceInfo = ev.inputInfo(input);
            std::string knob = input == 0 ? "Bchannels" : input == 1 ? "Achannels" :
                               std::string(1, char('A' + input)) + "channels";
            auto ss = slotsFor(p.s(knob), sourceInfo.channels);
            ChannelList need = chans;
            need.push_back("rgba.alpha");
            for (const auto& channel : ss)
                if (!channel.empty()) need.push_back(channel);
            ImagePtr src = ev.input(input, roi, intersect(sortedChannels(need), sourceInfo.channels));
            for (int c = 0; c < 4; ++c) {
                if (os[c].empty() || ss[c].empty() || !hasChannel(chans, os[c])) continue;
                if (std::find(changed.begin(), changed.end(), os[c]) == changed.end()) changed.push_back(os[c]);
                const bool uni = c == 3 && unionAlpha(op);
                ev.parallelRows(roi, [&](int y) {
                    int n = roi.w();
                    std::vector<float> buf(size_t(n) * 4);
                    float *aa = buf.data(), *ba = aa + n, *av = ba + n, *bv = av + n;
                    src->readRow("rgba.alpha", y, roi.x0, n, aa);
                    out->readRow("rgba.alpha", y, roi.x0, n, ba);
                    src->readRow(ss[c], y, roi.x0, n, av);
                    out->readRow(os[c], y, roi.x0, n, bv);
                    float* dst = out->row(os[c], y);
                    if (uni) {
                        for (int x = 0; x < n; ++x) dst[x] = aa[x] + ba[x] - aa[x] * ba[x];
                    } else if (op == 0) {
                        for (int x = 0; x < n; ++x) dst[x] = av[x] + bv[x] * (1.0f - aa[x]);
                    } else {
                        for (int x = 0; x < n; ++x) dst[x] = mergeValue(op, av[x], bv[x], aa[x], ba[x]);
                    }
                });
            }
        }
        applyMaskMix(ev, p, 8, *first, *out, changed);
        return out;
    }
};
VC_REGISTER_OP(OpDesc{"Merge", "Merge",
                      "Composite up to eight inputs with premultiplied maths. over: A + B(1 - a).",
                      {"B", "A", "C", "D", "E", "F", "G", "H", "mask"}, 0,
                      concat({K::enm("operation", "operation", kMergeOps, "over"),
                              K::channels("Achannels", "A channels", "rgba"),
                              K::channels("Bchannels", "B channels", "rgba"),
                              K::channels("Cchannels", "C channels", "rgba"),
                              K::channels("Dchannels", "D channels", "rgba"),
                              K::channels("Echannels", "E channels", "rgba"),
                              K::channels("Fchannels", "F channels", "rgba"),
                              K::channels("Gchannels", "G channels", "rgba"),
                              K::channels("Hchannels", "H channels", "rgba"),
                              K::channels("output", "output", "rgba"),
                              K::enm("bbox", "set bbox to", {"union", "intersection", "A", "B"})},
                             maskKnobs()),
                      1, maker<Merge>()});

// ------------------------------------------------------------ MergeLayer
const int kLayerRows = 6;
class MergeLayer : public Op {
public:
    Info info(Eval& ev, const Params& p) const override {
        if (!ev.connected(0)) return emptyInfo();
        Info i = ev.inputInfo(0);
        auto ts = slotsFor(p.s("target"), i.channels);
        for (int s = 0; s < 3; ++s) i.channels.push_back(ts[s]);
        i.channels = sortedChannels(i.channels);
        return i;
    }
    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        Info inf = ev.inputInfo(0);
        auto ts = slotsFor(p.s("target"), inf.channels);
        struct Row {
            std::vector<std::string> slots;
            std::string op;
            double gain;
        };
        std::vector<Row> rows;
        ChannelList need = chans;
        for (int r = 0; r < kLayerRows; ++r) {
            std::string layer = p.s("layer" + std::to_string(r)), op = p.s("op" + std::to_string(r));
            if (layer.empty() || op == "off") continue;
            Row row{layerSlots(layer, inf.channels), op, p.f("gain" + std::to_string(r))};
            for (auto& s : row.slots)
                if (!s.empty()) need.push_back(s);
            rows.push_back(row);
        }
        for (int s = 0; s < 3; ++s) need.push_back(ts[s]);
        ImagePtr in = ev.input(0, roi, intersect(sortedChannels(need), inf.channels));
        auto out = copyOf(*in, roi, chans);
        bool fromBlack = p.s("start") == "black";
        ev.parallelRows(roi, [&](int y) {
            int n = roi.w();
            std::vector<float> tgt(size_t(n) * 3), lay(size_t(n) * 4 * rows.size());
            for (int s = 0; s < 3; ++s) in->readRow(ts[s], y, roi.x0, n, tgt.data() + size_t(s) * n);
            for (size_t r = 0; r < rows.size(); ++r)
                for (int s = 0; s < 4; ++s)
                    in->readRow(rows[r].slots[s].empty() ? -1 : in->index(rows[r].slots[s]), y, roi.x0, n,
                                lay.data() + (r * 4 + s) * n);
            for (int x = roi.x0; x < roi.x1; ++x) {
                int xi = x - roi.x0;
                float acc[3];
                for (int s = 0; s < 3; ++s) acc[s] = fromBlack ? 0.0f : tgt[size_t(s) * n + xi];
                for (size_t r = 0; r < rows.size(); ++r) {
                    const auto& row = rows[r];
                    float la = lay[(r * 4 + 3) * n + xi];
                    for (int s = 0; s < 3; ++s) {
                        float v = float(lay[(r * 4 + s) * n + xi] * row.gain);
                        if (row.op == "plus") acc[s] += v;
                        else if (row.op == "multiply") acc[s] *= v;
                        else if (row.op == "over") acc[s] = v + acc[s] * (1.0f - la);
                        else if (row.op == "minus") acc[s] -= v;
                        else if (row.op == "max") acc[s] = std::max(acc[s], v);
                        else if (row.op == "replace") acc[s] = v;
                    }
                }
                for (int s = 0; s < 3; ++s)
                    if (float* o = out->row(ts[s], y)) o[x - roi.x0] = acc[s];
            }
        });
        return out;
    }
};
std::vector<KnobDef> mergeLayerKnobs() {
    std::vector<KnobDef> k{K::channels("target", "target layer", "rgba"),
                           K::enm("start", "start from", {"black", "target"})};
    const char* defaults[kLayerRows] = {"diffuse", "specular", "reflection", "", "", ""};
    for (int r = 0; r < kLayerRows; ++r) {
        std::string g = "Layer " + std::to_string(r + 1);
        k.push_back(K::group(K::channels("layer" + std::to_string(r), "layer", defaults[r]), g));
        k.push_back(K::group(K::enm("op" + std::to_string(r), "operation",
                                    {"plus", "multiply", "over", "minus", "max", "replace", "off"}),
                             g));
        k.push_back(K::group(K::flt("gain" + std::to_string(r), "gain", 1, 0, 4), g));
    }
    return k;
}
VC_REGISTER_OP(OpDesc{"MergeLayer", "Merge",
                      "Rebuild a layer from AOVs of the same stream, e.g. diffuse + specular + reflection into rgba.",
                      {""}, 1, mergeLayerKnobs(), 1, maker<MergeLayer>()});

// ------------------------------------------------------------ pass-through family
class PassThrough : public Op {
public:
    Info info(Eval& ev, const Params&) const override {
        return ev.connected(0) ? ev.inputInfo(0) : emptyInfo();
    }
    ImagePtr render(Eval& ev, const Params&, const Box& roi, const ChannelList& chans) const override {
        return ev.input(0, roi, chans);
    }
};
VC_REGISTER_OP(OpDesc{"Dot", "Other", "Reroute point.", {""}, 0, {}, 1, maker<PassThrough>()});
VC_REGISTER_OP(OpDesc{"NoOp", "Other", "Does nothing; holds user knobs and labels.", {""}, 0,
                      {K::text("note", "note", "")}, 1, maker<PassThrough>()});

class Switch : public Op {
public:
    static int pick(Eval& ev, const Params& p) {
        int n = int(ev.inputs.size());
        int w = std::max(0, std::min(n - 1, p.i("which")));
        return w;
    }
    Info info(Eval& ev, const Params& p) const override {
        int w = pick(ev, p);
        return ev.connected(w) ? ev.inputInfo(w) : emptyInfo();
    }
    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        return ev.input(pick(ev, p), roi, chans);
    }
};
VC_REGISTER_OP(OpDesc{"Switch", "Merge", "Pass one of the inputs, selected by index (animatable).",
                      {"0", "1", "2", "3", "4"}, 0, {K::integer("which", "which", 0, 0, 4)}, 1, maker<Switch>()});

// ------------------------------------------------------------ time
class TimeOffset : public Op {
public:
    std::vector<double> inputFrames(int, double frame, const Params& p) const override {
        return {map(frame, p)};
    }
    static double map(double f, const Params& p) {
        return p.b("reverse") ? -f + p.f("offset") : f - p.f("offset");
    }
    Info info(Eval& ev, const Params& p) const override {
        if (!ev.connected(0)) return emptyInfo();
        Info i = ev.inputInfo(0, map(ev.frame, p));
        i.first += p.f("offset");
        i.last += p.f("offset");
        return i;
    }
    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        return ev.input(0, roi, chans, map(ev.frame, p));
    }
};
VC_REGISTER_OP(OpDesc{"TimeOffset", "Time", "Shift the input in time: out(f) = in(f - offset).", {""}, 0,
                      {K::integer("offset", "time offset", 0, -100, 100, "frames"), K::boolean("reverse", "reverse input", false)},
                      1, maker<TimeOffset>()});

class FrameHold : public Op {
public:
    static double map(double f, const Params& p) {
        double first = p.f("firstFrame"), inc = p.f("increment");
        if (inc <= 0) return first;
        return first + std::floor((f - first) / inc) * inc;
    }
    std::vector<double> inputFrames(int, double frame, const Params& p) const override { return {map(frame, p)}; }
    Info info(Eval& ev, const Params& p) const override {
        return ev.connected(0) ? ev.inputInfo(0, map(ev.frame, p)) : emptyInfo();
    }
    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        return ev.input(0, roi, chans, map(ev.frame, p));
    }
};
VC_REGISTER_OP(OpDesc{"FrameHold", "Time", "Hold a frame; with increment > 0, hold every N frames.", {""}, 0,
                      {K::integer("firstFrame", "first frame", 1, 0, 2000), K::integer("increment", "increment", 0, 0, 24)},
                      1, maker<FrameHold>()});

class Retime : public Op {
public:
    static double map(double f, const Params& p) {
        return p.f("inputStart") + (f - p.f("outputStart")) * p.f("speed");
    }
    std::vector<double> inputFrames(int, double frame, const Params& p) const override {
        double s = map(frame, p);
        if (p.s("filter") == "nearest") return {std::floor(s + 0.5)};
        return {std::floor(s), std::floor(s) + 1};
    }
    Info info(Eval& ev, const Params& p) const override {
        if (!ev.connected(0)) return emptyInfo();
        Info i = ev.inputInfo(0, std::floor(map(ev.frame, p)));
        double sp = std::fabs(p.f("speed")) < 1e-6 ? 1.0 : p.f("speed");
        i.first = p.f("outputStart") + (i.first - p.f("inputStart")) / sp;
        i.last = p.f("outputStart") + (i.last - p.f("inputStart")) / sp;
        if (i.first > i.last) std::swap(i.first, i.last);
        return i;
    }
    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        double s = map(ev.frame, p);
        if (p.s("filter") == "nearest" || std::fabs(s - std::floor(s)) < 1e-6)
            return ev.input(0, roi, chans, std::floor(s + 0.5));
        double f0 = std::floor(s), t = s - f0;
        ImagePtr a = ev.input(0, roi, chans, f0), b = ev.input(0, roi, chans, f0 + 1);
        auto out = std::make_shared<Image>(roi, chans);
        for (auto& c : chans)
            ev.parallelRows(roi, [&](int y) {
                int n = roi.w();
                std::vector<float> rb(n);
                float* o = out->row(c, y);
                a->readRow(c, y, roi.x0, n, o);
                b->readRow(c, y, roi.x0, n, rb.data());
                for (int x = 0; x < n; ++x) o[x] = float(o[x] * (1 - t) + rb[x] * t);
            });
        return out;
    }
};
VC_REGISTER_OP(OpDesc{"Retime", "Time", "Constant speed change with nearest or blended frames.", {""}, 0,
                      {K::flt("speed", "speed", 1, 0.1, 4), K::integer("inputStart", "input start", 1, 0, 2000),
                       K::integer("outputStart", "output start", 1, 0, 2000), K::enm("filter", "filter", {"blend", "nearest"})},
                      1, maker<Retime>()});

}  // namespace
}  // namespace vc::ops
