// Channel and layer ops: Shuffle, Copy, Remove, AddChannels, ChannelSwitch.
#include <sstream>

#include "common.h"

namespace vc::ops {
namespace {

std::vector<std::string> outSlots(const std::string& layer, const ChannelList& avail) {
    if (layer == "rgba") return {"rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"};
    auto s = layerSlots(layer, avail);
    bool any = false;
    for (auto& c : s) any |= !c.empty();
    if (!any) return {layer + ".red", layer + ".green", layer + ".blue", layer + ".alpha"};
    return s;
}

// Source spec: "0", "1", "<channel>" (input 1) or "in2:<channel>".
struct Src {
    int input = 0;
    std::string ch;
    float constant = -1;
};
Src parseSrc(const std::string& s) {
    Src r;
    if (s.empty() || s == "0" || s == "black") { r.constant = 0; return r; }
    if (s == "1" || s == "white") { r.constant = 1; return r; }
    if (s.rfind("in2:", 0) == 0) {
        r.input = 1;
        r.ch = s.substr(4);
    } else {
        r.ch = s;
    }
    return r;
}

// ------------------------------------------------------------ Shuffle
class Shuffle : public Op {
public:
    static constexpr const char* kSlots[4] = {"red", "green", "blue", "alpha"};
    Info info(Eval& ev, const Params& p) const override {
        if (!ev.connected(0)) return emptyInfo();
        Info i = ev.inputInfo(0);
        Info j = ev.inputInfo(1);
        if (ev.connected(1)) i.bbox = i.bbox.unite(j.bbox);
        auto os = outSlots(p.s("out"), i.channels);
        ChannelList c = i.channels;
        for (int s = 0; s < 4; ++s)
            if (!os[s].empty() && p.s(kSlots[s]) != "-") c.push_back(os[s]);
        i.channels = sortedChannels(c);
        return i;
    }
    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        Info i0 = ev.inputInfo(0), i1 = ev.inputInfo(1);
        auto os = outSlots(p.s("out"), i0.channels);
        Src src[4];
        ChannelList n0 = chans, n1;
        for (int s = 0; s < 4; ++s) {
            src[s] = parseSrc(p.s(kSlots[s]));
            if (src[s].constant < 0) (src[s].input == 0 ? n0 : n1).push_back(src[s].ch);
        }
        ImagePtr a = ev.input(0, roi, intersect(sortedChannels(n0), i0.channels));
        ImagePtr b = ev.input(1, roi, intersect(sortedChannels(n1), i1.channels));
        auto out = copyOf(*a, roi, chans);
        for (int s = 0; s < 4; ++s) {
            if (os[s].empty() || !hasChannel(chans, os[s]) || p.s(kSlots[s]) == "-") continue;
            const Image& from = src[s].input == 0 ? *a : *b;
            ev.parallelRows(roi, [&](int y) {
                float* o = out->row(os[s], y);
                if (src[s].constant >= 0) std::fill(o, o + roi.w(), src[s].constant);
                else from.readRow(src[s].ch, y, roi.x0, roi.w(), o);
            });
        }
        return out;
    }
};
VC_REGISTER_OP(OpDesc{"Shuffle", "Channel",
                      "Route any channel to each slot of the output layer. Sources: a channel name, 0, 1, "
                      "in2:<channel> for the second input, or - to leave the slot untouched.",
                      {"in", "in2"}, 0,
                      {K::channels("out", "output layer", "rgba"), K::channel("red", "red from", "rgba.red"),
                       K::channel("green", "green from", "rgba.green"), K::channel("blue", "blue from", "rgba.blue"),
                       K::channel("alpha", "alpha from", "rgba.alpha")},
                      1, maker<Shuffle>()});

// ------------------------------------------------------------ Copy
class Copy : public Op {
public:
    Info info(Eval& ev, const Params& p) const override {
        if (!ev.connected(0)) return emptyInfo();
        Info B = ev.inputInfo(0), A = ev.inputInfo(1);
        ChannelList c = B.channels;
        for (int r = 0; r < 4; ++r) {
            std::string from = p.s("from" + std::to_string(r)), to = p.s("to" + std::to_string(r));
            if (!from.empty() && !to.empty()) c.push_back(to);
        }
        for (auto& ch : resolveChannels(p.s("layer"), A.channels)) c.push_back(ch);
        B.channels = sortedChannels(c);
        if (p.s("bbox") == "union") B.bbox = B.bbox.unite(A.bbox);
        else if (p.s("bbox") == "A") B.bbox = A.bbox;
        return B;
    }
    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        Info Ai = ev.inputInfo(1);
        std::vector<std::pair<std::string, std::string>> map;
        for (int r = 0; r < 4; ++r) {
            std::string from = p.s("from" + std::to_string(r)), to = p.s("to" + std::to_string(r));
            if (!from.empty() && !to.empty()) map.push_back({from, to});
        }
        for (auto& ch : resolveChannels(p.s("layer"), Ai.channels)) map.push_back({ch, ch});
        ChannelList needA;
        for (auto& m : map) needA.push_back(m.first);
        ImagePtr A = ev.input(1, roi, intersect(sortedChannels(needA), Ai.channels));
        ImagePtr B = ev.input(0, roi, chans);
        auto out = copyOf(*B, roi, chans);
        ChannelList changed;
        for (auto& m : map) {
            if (!hasChannel(chans, m.second)) continue;
            changed.push_back(m.second);
            ev.parallelRows(roi, [&](int y) {
                A->readRow(m.first, y, roi.x0, roi.w(), out->row(m.second, y));
            });
        }
        applyMaskMix(ev, p, 2, *B, *out, changed);
        return out;
    }
};
VC_REGISTER_OP(OpDesc{"Copy", "Channel", "Copy channels from A into B, optionally renaming; or a whole layer.",
                      {"B", "A", "mask"}, 0,
                      concat({K::channel("from0", "copy", "rgba.alpha"), K::channel("to0", "to", "rgba.alpha"),
                              K::channel("from1", "copy", ""), K::channel("to1", "to", ""),
                              K::channel("from2", "copy", ""), K::channel("to2", "to", ""),
                              K::channel("from3", "copy", ""), K::channel("to3", "to", ""),
                              K::channels("layer", "layer copy", "none", "Copy every channel of these layers"),
                              K::enm("bbox", "set bbox to", {"B", "union", "A"})},
                             maskKnobs()),
                      1, maker<Copy>()});

// ------------------------------------------------------------ Remove / Keep
class Remove : public Op {
public:
    Info info(Eval& ev, const Params& p) const override {
        if (!ev.connected(0)) return emptyInfo();
        Info i = ev.inputInfo(0);
        ChannelList sel = resolveChannels(p.s("channels"), i.channels);
        ChannelList out;
        bool keep = p.s("operation") == "keep";
        for (auto& c : i.channels)
            if (hasChannel(sel, c) == keep) out.push_back(c);
        i.channels = out;
        return i;
    }
    ImagePtr render(Eval& ev, const Params&, const Box& roi, const ChannelList& chans) const override {
        return copyOf(*ev.input(0, roi, chans), roi, chans);
    }
};
VC_REGISTER_OP(OpDesc{"Remove", "Channel", "Remove or keep only the listed channels/layers.", {""}, 0,
                      {K::enm("operation", "operation", {"remove", "keep"}),
                       K::channels("channels", "channels", "none", "Comma separated: rgba, depth, diffuse.red, ...")},
                      1, maker<Remove>()});

// ------------------------------------------------------------ AddChannels
ChannelList listChannels(const std::string& spec) {
    ChannelList out;
    std::stringstream ss(spec);
    std::string t;
    while (std::getline(ss, t, ',')) {
        t.erase(0, t.find_first_not_of(" "));
        t.erase(t.find_last_not_of(" ") + 1);
        if (t.empty()) continue;
        if (t == "rgba") {
            for (auto c : {"rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"}) out.push_back(c);
        } else if (t.find('.') == std::string::npos) {
            for (auto c : {".red", ".green", ".blue", ".alpha"}) out.push_back(t + c);
        } else {
            out.push_back(t);
        }
    }
    return out;
}

class AddChannels : public Op {
public:
    Info info(Eval& ev, const Params& p) const override {
        Info i = ev.connected(0) ? ev.inputInfo(0) : emptyInfo();
        i.channels = unite(i.channels, listChannels(p.s("channels")));
        return i;
    }
    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        ImagePtr in = ev.input(0, roi, chans);
        auto out = copyOf(*in, roi, chans);
        float v = float(p.f("value"));
        for (auto& c : listChannels(p.s("channels"))) {
            if (!hasChannel(chans, c) || in->has(c)) continue;
            float* d = out->plane(c);
            std::fill(d, d + size_t(roi.w()) * roi.h(), v);
        }
        return out;
    }
};
VC_REGISTER_OP(OpDesc{"AddChannels", "Channel", "Create channels that do not exist yet, filled with a value.", {""}, 0,
                      {K::channels("channels", "channels", "mask.a"), K::flt("value", "value", 0, 0, 1)}, 1,
                      maker<AddChannels>()});

// ------------------------------------------------------------ ChannelSwitch
class ChannelSwitch : public Op {
public:
    Info info(Eval& ev, const Params& p) const override {
        if (!ev.connected(0)) return emptyInfo();
        Info i = ev.inputInfo(0);
        i.channels = unite(i.channels, outSlots(p.s("out"), i.channels));
        return i;
    }
    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        Info i = ev.inputInfo(0);
        ChannelList cands = listChannels(p.s("candidates"));
        if (cands.empty()) return copyOf(*ev.input(0, roi, chans), roi, chans);
        int w = std::max(0, std::min(int(cands.size()) - 1, p.i("which")));
        std::string src = cands[w];
        ImagePtr in = ev.input(0, roi, intersect(unite(chans, {src}), i.channels));
        auto out = copyOf(*in, roi, chans);
        auto os = outSlots(p.s("out"), i.channels);
        bool rgbOnly = p.s("mode") == "to rgb";
        for (int s = 0; s < 4; ++s) {
            if (os[s].empty() || !hasChannel(chans, os[s]) || (rgbOnly && s == 3)) continue;
            ev.parallelRows(roi, [&](int y) {
                in->readRow(src, y, roi.x0, roi.w(), out->row(os[s], y));
            });
        }
        return out;
    }
};
VC_REGISTER_OP(OpDesc{"ChannelSwitch", "Channel",
                      "Show one channel of a list in the output layer; 'which' is animatable.", {""}, 0,
                      {K::str("candidates", "channels", "rgba.red, rgba.green, rgba.blue, rgba.alpha",
                              "Comma separated channel names"),
                       K::integer("which", "which", 0, 0, 8), K::channels("out", "output layer", "rgba"),
                       K::enm("mode", "mode", {"to rgb", "to rgba"})},
                      1, maker<ChannelSwitch>()});

}  // namespace
}  // namespace vc::ops
