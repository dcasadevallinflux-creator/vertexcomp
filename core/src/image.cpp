#include "vc/image.h"

#include <cstring>
#include <set>
#include <sstream>

namespace vc {

std::string layerOf(const std::string& ch) {
    auto p = ch.rfind('.');
    return p == std::string::npos ? std::string("other") : ch.substr(0, p);
}

std::string shortOf(const std::string& ch) {
    auto p = ch.rfind('.');
    return p == std::string::npos ? ch : ch.substr(p + 1);
}

static std::string longComponent(const std::string& s) {
    if (s == "R" || s == "r") return "red";
    if (s == "G" || s == "g") return "green";
    if (s == "B" || s == "b") return "blue";
    if (s == "A" || s == "a") return "alpha";
    return s;
}

std::string canonicalChannel(const std::string& name) {
    if (name == "R" || name == "G" || name == "B" || name == "A") return "rgba." + longComponent(name);
    if (name == "Z") return "depth.Z";
    auto p = name.rfind('.');
    if (p == std::string::npos) return "other." + name;
    return name.substr(0, p) + "." + longComponent(name.substr(p + 1));
}

static int componentRank(const std::string& s) {
    static const char* order[] = {"red", "green", "blue", "alpha", "x", "y", "z", "X", "Y", "Z", "u", "v", "w"};
    for (int i = 0; i < 13; ++i)
        if (s == order[i]) return i;
    return 100;
}

ChannelList sortedChannels(ChannelList chans) {
    std::sort(chans.begin(), chans.end(), [](const std::string& a, const std::string& b) {
        std::string la = layerOf(a), lb = layerOf(b);
        if (la != lb) {
            if (la == "rgba") return true;
            if (lb == "rgba") return false;
            return la < lb;
        }
        int ra = componentRank(shortOf(a)), rb = componentRank(shortOf(b));
        if (ra != rb) return ra < rb;
        return a < b;
    });
    chans.erase(std::unique(chans.begin(), chans.end()), chans.end());
    return chans;
}

bool hasChannel(const ChannelList& l, const std::string& c) {
    return std::find(l.begin(), l.end(), c) != l.end();
}

ChannelList unite(const ChannelList& a, const ChannelList& b) {
    ChannelList r = a;
    for (auto& c : b)
        if (!hasChannel(r, c)) r.push_back(c);
    return sortedChannels(r);
}

ChannelList intersect(const ChannelList& a, const ChannelList& b) {
    ChannelList r;
    for (auto& c : a)
        if (hasChannel(b, c)) r.push_back(c);
    return r;
}

std::vector<std::string> layersOf(const ChannelList& l) {
    std::vector<std::string> out;
    for (auto& c : sortedChannels(l)) {
        auto ly = layerOf(c);
        if (std::find(out.begin(), out.end(), ly) == out.end()) out.push_back(ly);
    }
    return out;
}

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t"), b = s.find_last_not_of(" \t");
    if (a == std::string::npos) return {};
    return s.substr(a, b - a + 1);
}

ChannelList resolveChannels(const std::string& spec, const ChannelList& available) {
    ChannelList out;
    auto add = [&](const std::string& c) {
        if (hasChannel(available, c) && !hasChannel(out, c)) out.push_back(c);
    };
    std::stringstream ss(spec);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        tok = trim(tok);
        if (tok.empty() || tok == "none") continue;
        if (tok == "all") {
            for (auto& c : available) add(c);
        } else if (tok == "rgba") {
            for (auto c : {"rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"}) add(c);
        } else if (tok == "rgb") {
            for (auto c : {"rgba.red", "rgba.green", "rgba.blue"}) add(c);
        } else if (tok == "red" || tok == "green" || tok == "blue" || tok == "alpha") {
            add("rgba." + tok);
        } else if (tok.find('.') != std::string::npos) {
            add(tok);
        } else {
            for (auto& c : available)
                if (layerOf(c) == tok) add(c);
        }
    }
    return sortedChannels(out);
}

std::vector<std::string> layerSlots(const std::string& layer, const ChannelList& available) {
    std::vector<std::string> slots(4);
    ChannelList inLayer;
    for (auto& c : sortedChannels(available))
        if (layerOf(c) == layer) inLayer.push_back(c);
    for (auto& c : inLayer) {
        auto s = shortOf(c);
        if (s == "red") slots[0] = c;
        else if (s == "green") slots[1] = c;
        else if (s == "blue") slots[2] = c;
        else if (s == "alpha") slots[3] = c;
    }
    // Layers named with x/y/z or arbitrary components fill the free slots in order.
    for (auto& c : inLayer) {
        if (std::find(slots.begin(), slots.end(), c) != slots.end()) continue;
        for (int i = 0; i < 4; ++i) {
            if (slots[i].empty() && !(i == 3 && inLayer.size() <= 3)) {
                slots[i] = c;
                break;
            }
        }
    }
    return slots;
}

Image::Image(const Box& bbox, const ChannelList& chans) : bbox_(bbox), channels_(chans) {
    size_t n = size_t(bbox_.w()) * size_t(bbox_.h());
    planes_.resize(channels_.size());
    for (auto& p : planes_) p.assign(n, 0.0f);
}

int Image::index(const std::string& ch) const {
    for (size_t i = 0; i < channels_.size(); ++i)
        if (channels_[i] == ch) return int(i);
    return -1;
}

float* Image::plane(const std::string& ch) {
    int i = index(ch);
    return i < 0 ? nullptr : planes_[i].data();
}

const float* Image::plane(const std::string& ch) const {
    int i = index(ch);
    return i < 0 ? nullptr : planes_[i].data();
}

float* Image::row(const std::string& ch, int y) {
    float* p = plane(ch);
    if (!p || y < bbox_.y0 || y >= bbox_.y1) return nullptr;
    return p + size_t(y - bbox_.y0) * bbox_.w();
}

const float* Image::row(const std::string& ch, int y) const {
    const float* p = plane(ch);
    if (!p || y < bbox_.y0 || y >= bbox_.y1) return nullptr;
    return p + size_t(y - bbox_.y0) * bbox_.w();
}

float Image::at(const std::string& ch, int x, int y) const {
    if (x < bbox_.x0 || x >= bbox_.x1 || y < bbox_.y0 || y >= bbox_.y1) return 0.0f;
    const float* p = plane(ch);
    if (!p) return 0.0f;
    return p[size_t(y - bbox_.y0) * bbox_.w() + (x - bbox_.x0)];
}

void Image::addChannel(const std::string& ch) {
    if (has(ch)) return;
    channels_.push_back(ch);
    planes_.emplace_back(size_t(bbox_.w()) * size_t(bbox_.h()), 0.0f);
}

void Image::removeChannel(const std::string& ch) {
    int i = index(ch);
    if (i < 0) return;
    channels_.erase(channels_.begin() + i);
    planes_.erase(planes_.begin() + i);
}

void Image::renameChannel(const std::string& from, const std::string& to) {
    int i = index(from);
    if (i < 0 || from == to) return;
    removeChannel(to);
    i = index(from);
    channels_[i] = to;
}

void Image::copyChannel(const Image& src, const std::string& srcCh, const std::string& dstCh) {
    addChannel(dstCh);
    float* d = plane(dstCh);
    const float* s = src.plane(srcCh);
    std::fill(d, d + size_t(bbox_.w()) * bbox_.h(), 0.0f);
    if (!s) return;
    Box ov = bbox_.intersect(src.bbox());
    if (ov.empty()) return;
    for (int y = ov.y0; y < ov.y1; ++y) {
        const float* sr = s + size_t(y - src.bbox().y0) * src.bbox().w() + (ov.x0 - src.bbox().x0);
        float* dr = d + size_t(y - bbox_.y0) * bbox_.w() + (ov.x0 - bbox_.x0);
        std::memcpy(dr, sr, sizeof(float) * ov.w());
    }
}

void Image::readRow(const std::string& ch, int y, int x0, int n, float* dst) const {
    readRow(index(ch), y, x0, n, dst);
}

void Image::readRow(int ci, int y, int x0, int n, float* dst) const {
    std::fill(dst, dst + n, 0.0f);
    if (ci < 0 || y < bbox_.y0 || y >= bbox_.y1) return;
    int a = std::max(x0, bbox_.x0), b = std::min(x0 + n, bbox_.x1);
    if (b <= a) return;
    const float* src = planes_[ci].data() + size_t(y - bbox_.y0) * bbox_.w() + (a - bbox_.x0);
    std::memcpy(dst + (a - x0), src, sizeof(float) * (b - a));
}

size_t Image::bytes() const {
    return sizeof(float) * size_t(bbox_.w()) * size_t(bbox_.h()) * channels_.size() + sizeof(Image);
}

}  // namespace vc
