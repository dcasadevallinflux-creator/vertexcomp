#pragma once
// Image model.
//
// Coordinates: integer pixel grid, origin at the top-left of the format,
// y grows downwards. A pixel (x, y) covers [x, x+1) x [y, y+1); its centre
// is at (x + 0.5, y + 0.5). Boxes are half-open.
//
// Storage: planar float32, one plane per channel, each plane covering the
// image bbox row-major. Pixels outside the bbox are zero in every channel.
//
// Colour: linear, premultiplied by rgba.alpha. Ops that need straight colour
// unpremultiply locally (see ops/common.h).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace vc {

struct Box {
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;

    int w() const { return std::max(0, x1 - x0); }
    int h() const { return std::max(0, y1 - y0); }
    bool empty() const { return x1 <= x0 || y1 <= y0; }
    bool contains(const Box& o) const {
        return o.empty() || (o.x0 >= x0 && o.y0 >= y0 && o.x1 <= x1 && o.y1 <= y1);
    }
    Box intersect(const Box& o) const {
        Box r{std::max(x0, o.x0), std::max(y0, o.y0), std::min(x1, o.x1), std::min(y1, o.y1)};
        if (r.empty()) return Box{};
        return r;
    }
    Box unite(const Box& o) const {
        if (empty()) return o;
        if (o.empty()) return *this;
        return {std::min(x0, o.x0), std::min(y0, o.y0), std::max(x1, o.x1), std::max(y1, o.y1)};
    }
    Box grow(int dx, int dy) const {
        if (empty()) return *this;
        return {x0 - dx, y0 - dy, x1 + dx, y1 + dy};
    }
    bool operator==(const Box& o) const {
        return x0 == o.x0 && y0 == o.y0 && x1 == o.x1 && y1 == o.y1;
    }
    bool operator!=(const Box& o) const { return !(*this == o); }
};

struct Format {
    int w = 1920, h = 1080;
    double par = 1.0;
    Box box() const { return {0, 0, w, h}; }
};

using ChannelList = std::vector<std::string>;

// "layer.channel" naming. Bare names are placed in the "other" layer except
// for the EXR conventions R/G/B/A (-> rgba.*) and Z (-> depth.Z).
std::string layerOf(const std::string& ch);
std::string shortOf(const std::string& ch);
std::string canonicalChannel(const std::string& name);

// rgba first (red, green, blue, alpha), then other layers alphabetically,
// channels inside a layer in red/green/blue/alpha/x/y/z/... order.
ChannelList sortedChannels(ChannelList chans);
bool hasChannel(const ChannelList& l, const std::string& c);
ChannelList unite(const ChannelList& a, const ChannelList& b);
ChannelList intersect(const ChannelList& a, const ChannelList& b);
std::vector<std::string> layersOf(const ChannelList& l);

// Resolve a channel selector against the channels available.
//   "rgba" "rgb" "alpha" "red" "green" "blue" "all" "none"
//   a layer name ("diffuse"), a full channel name ("depth.Z"),
//   or a comma separated combination of the above.
ChannelList resolveChannels(const std::string& spec, const ChannelList& available);

// The (up to) four channels of a layer mapped to RGBA slots; missing slots
// are empty strings. For "rgba" this is {red, green, blue, alpha}.
std::vector<std::string> layerSlots(const std::string& layer, const ChannelList& available);

class Image {
public:
    Image() = default;
    Image(const Box& bbox, const ChannelList& chans);

    const Box& bbox() const { return bbox_; }
    const ChannelList& channels() const { return channels_; }
    int width() const { return bbox_.w(); }
    int height() const { return bbox_.h(); }

    int index(const std::string& ch) const;
    bool has(const std::string& ch) const { return index(ch) >= 0; }
    float* plane(const std::string& ch);
    const float* plane(const std::string& ch) const;
    float* plane(int i) { return planes_[i].data(); }
    const float* plane(int i) const { return planes_[i].data(); }
    float* row(const std::string& ch, int y);
    const float* row(const std::string& ch, int y) const;

    // Copy n values of channel `ch` on row y starting at x0 into dst; pixels
    // outside the bbox (or a missing channel) read as zero. Use this in loops
    // instead of at(), which looks the channel up by name on every call.
    void readRow(const std::string& ch, int y, int x0, int n, float* dst) const;
    void readRow(int chIndex, int y, int x0, int n, float* dst) const;

    float at(const std::string& ch, int x, int y) const;

    void addChannel(const std::string& ch);
    void removeChannel(const std::string& ch);
    void renameChannel(const std::string& from, const std::string& to);

    // Copy channel `srcCh` of `src` into channel `dstCh` of this image over
    // the bbox intersection; area of this image outside src bbox is zeroed.
    void copyChannel(const Image& src, const std::string& srcCh, const std::string& dstCh);

    size_t bytes() const;

    std::map<std::string, std::string> meta;

private:
    Box bbox_;
    ChannelList channels_;
    std::vector<std::vector<float>> planes_;
};

using ImagePtr = std::shared_ptr<const Image>;
using MutImagePtr = std::shared_ptr<Image>;

}  // namespace vc
