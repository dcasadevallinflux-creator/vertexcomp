#pragma once
// Shared pieces for op implementations: knob builders, mask/mix handling,
// the per-pixel colour op base, and small image helpers.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>

#include "vc/graph.h"
#include "vc/image.h"

namespace vc::ops {

// ------------------------------------------------------------ knob builders
namespace K {
inline KnobDef flt(const std::string& n, const std::string& l, double d, double lo = 0, double hi = 1,
                   const std::string& tip = "") {
    KnobDef k;
    k.name = n; k.label = l; k.type = KnobType::Float; k.def = {d}; k.smin = lo; k.smax = hi; k.tooltip = tip;
    return k;
}
inline KnobDef integer(const std::string& n, const std::string& l, int d, int lo = 0, int hi = 10,
                       const std::string& tip = "") {
    KnobDef k = flt(n, l, d, lo, hi, tip);
    k.type = KnobType::Int;
    return k;
}
inline KnobDef boolean(const std::string& n, const std::string& l, bool d, const std::string& tip = "") {
    KnobDef k = flt(n, l, d ? 1 : 0, 0, 1, tip);
    k.type = KnobType::Bool;
    return k;
}
inline KnobDef enm(const std::string& n, const std::string& l, std::vector<std::string> opts,
                   const std::string& d = "", const std::string& tip = "") {
    KnobDef k;
    k.name = n; k.label = l; k.type = KnobType::Enum; k.options = std::move(opts);
    k.sdef = d.empty() ? k.options.front() : d; k.tooltip = tip; k.animatable = false;
    return k;
}
inline KnobDef str(const std::string& n, const std::string& l, const std::string& d = "", const std::string& tip = "") {
    KnobDef k;
    k.name = n; k.label = l; k.type = KnobType::String; k.sdef = d; k.tooltip = tip; k.animatable = false;
    return k;
}
inline KnobDef text(const std::string& n, const std::string& l, const std::string& d = "", const std::string& tip = "") {
    KnobDef k = str(n, l, d, tip);
    k.type = KnobType::Text;
    return k;
}
inline KnobDef file(const std::string& n, const std::string& l, const std::string& tip = "") {
    KnobDef k = str(n, l, "", tip);
    k.type = KnobType::File;
    return k;
}
inline KnobDef color(const std::string& n, const std::string& l, std::vector<double> d, double lo = 0, double hi = 1,
                     const std::string& tip = "") {
    KnobDef k;
    k.name = n; k.label = l; k.type = KnobType::Color; k.comps = int(d.size()); k.def = d;
    k.smin = lo; k.smax = hi; k.tooltip = tip;
    return k;
}
inline KnobDef xy(const std::string& n, const std::string& l, double x, double y, const std::string& tip = "") {
    KnobDef k;
    k.name = n; k.label = l; k.type = KnobType::XY; k.comps = 2; k.def = {x, y}; k.smin = -1000; k.smax = 1000;
    k.tooltip = tip;
    return k;
}
inline KnobDef wh(const std::string& n, const std::string& l, double w, double h, double lo = 0, double hi = 100,
                  const std::string& tip = "") {
    KnobDef k = xy(n, l, w, h, tip);
    k.smin = lo; k.smax = hi;
    return k;
}
inline KnobDef box(const std::string& n, const std::string& l, std::vector<double> d, const std::string& tip = "") {
    KnobDef k;
    k.name = n; k.label = l; k.type = KnobType::Box; k.comps = 4; k.def = d; k.smin = 0; k.smax = 2048;
    k.tooltip = tip;
    return k;
}
inline KnobDef range(const std::string& n, const std::string& l, std::vector<double> d, double lo = 0, double hi = 1,
                     const std::string& tip = "") {
    KnobDef k;
    k.name = n; k.label = l; k.type = KnobType::Range; k.comps = int(d.size()); k.def = d; k.smin = lo; k.smax = hi;
    k.tooltip = tip;
    return k;
}
inline KnobDef channel(const std::string& n, const std::string& l, const std::string& d, const std::string& tip = "") {
    KnobDef k = str(n, l, d, tip);
    k.type = KnobType::Channel;
    return k;
}
inline KnobDef channels(const std::string& n, const std::string& l, const std::string& d, const std::string& tip = "") {
    KnobDef k = str(n, l, d, tip);
    k.type = KnobType::Channels;
    return k;
}
inline KnobDef button(const std::string& n, const std::string& l, const std::string& tip = "") {
    KnobDef k;
    k.name = n; k.label = l; k.type = KnobType::Button; k.tooltip = tip; k.animatable = false;
    return k;
}
inline KnobDef group(KnobDef k, const std::string& g) {
    k.group = g;
    return k;
}
}  // namespace K

inline std::vector<KnobDef> maskKnobs() {
    return {K::group(K::channel("maskChannel", "mask channel", "rgba.alpha", "Channel of the mask input"), "Mask"),
            K::group(K::boolean("invertMask", "invert mask", false), "Mask"),
            K::group(K::flt("mix", "mix", 1.0, 0, 1, "Blend between input (0) and result (1)"), "Mask")};
}

template <class T>
std::function<std::unique_ptr<Op>()> maker() {
    return [] { return std::unique_ptr<Op>(new T()); };
}

inline std::vector<KnobDef> concat(std::vector<KnobDef> a, const std::vector<KnobDef>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

// ------------------------------------------------------------ image helpers

inline float clamp01(float v) { return std::min(1.0f, std::max(0.0f, v)); }
inline float luma709(float r, float g, float b) { return 0.2126f * r + 0.7152f * g + 0.0722f * b; }

// New image over `box` holding `chans` copied from `src` (zeros outside src).
inline MutImagePtr copyOf(const Image& src, const Box& box, const ChannelList& chans) {
    auto out = std::make_shared<Image>(box, chans);
    Box ov = box.intersect(src.bbox());
    for (auto& c : chans) {
        const float* s = src.plane(c);
        float* d = out->plane(c);
        if (!s || ov.empty()) continue;
        for (int y = ov.y0; y < ov.y1; ++y)
            std::memcpy(d + size_t(y - box.y0) * box.w() + (ov.x0 - box.x0),
                        s + size_t(y - src.bbox().y0) * src.bbox().w() + (ov.x0 - src.bbox().x0),
                        sizeof(float) * ov.w());
    }
    out->meta = src.meta;
    return out;
}

// Row pointer helper that tolerates rows outside the image (returns null).
inline const float* rowAt(const Image& img, const std::string& ch, int y, int x0) {
    const float* p = img.plane(ch);
    if (!p || y < img.bbox().y0 || y >= img.bbox().y1) return nullptr;
    return p + size_t(y - img.bbox().y0) * img.bbox().w() + (x0 - img.bbox().x0);
}

inline Info emptyInfo() {
    Info i;
    i.bbox = Box{};
    return i;
}

// Blend out towards orig: out = orig + (out - orig) * mix * mask. The mask is
// read from input slot `maskSlot` if connected. `chans` are the channels that
// were modified.
void applyMaskMix(Eval& ev, const Params& p, int maskSlot, const Image& orig, Image& out, const ChannelList& chans);

// Per-pixel colour ops: operate on up to four channels of each selected
// layer. Optional unpremultiply by rgba.alpha around the kernel.
using RowKernel = std::function<void(float* r, float* g, float* b, float* a, int n, int x0, int y)>;

class PixelOp : public Op {
public:
    Info info(Eval& ev, const Params& p) const override;
    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override;

protected:
    virtual RowKernel kernel(const Params& p, const Info& info) const = 0;
    virtual bool usesUnpremult() const { return true; }
};

// Knobs common to PixelOps (prepend op specific knobs).
std::vector<KnobDef> pixelOpKnobs(const std::string& defChannels, bool defUnpremult);

// Separable Gaussian/box helpers on single planes; buffers cover `box`.
// Reads outside the source bbox are zero.
void blurPlane(const Image& src, const std::string& ch, const Box& outBox, float* out, double sx, double sy,
               int filter, double quality, const CancelToken& cancel);
int blurRadius(double sigma, int filter, double quality);

// Erode/dilate with fractional radius (positive = dilate).
void morphPlane(const float* in, const Box& inBox, float* out, const Box& outBox, double rx, double ry,
                const CancelToken& cancel);

// Sampling helpers for geometric ops.
enum Filter { FilterImpulse = 0, FilterBilinear, FilterCubic, FilterLanczos };
int filterRadius(int f);
float sample(const float* plane, const Box& b, double x, double y, int filter);

}  // namespace vc::ops
