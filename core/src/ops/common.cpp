#include "common.h"

namespace vc::ops {

void applyMaskMix(Eval& ev, const Params& p, int maskSlot, const Image& orig, Image& out, const ChannelList& chans) {
    double mix = p.num.count("mix") ? p.f("mix") : 1.0;
    bool masked = maskSlot >= 0 && ev.connected(maskSlot);
    if (mix >= 1.0 && !masked) return;
    const Box& b = out.bbox();
    ImagePtr mask;
    std::string mch = p.s("maskChannel");
    bool inv = p.b("invertMask");
    if (masked) mask = ev.input(maskSlot, b, {mch});
    ev.parallelRows(b, [&](int y) {
        int n = b.w();
        std::vector<float> m(n, float(mix)), ov(n);
        if (masked) {
            mask->readRow(mch, y, b.x0, n, ov.data());
            for (int i = 0; i < n; ++i) m[i] *= inv ? 1.0f - ov[i] : ov[i];
        }
        for (auto& c : chans) {
            float* o = out.row(c, y);
            if (!o) continue;
            orig.readRow(c, y, b.x0, n, ov.data());
            for (int i = 0; i < n; ++i) o[i] = ov[i] + (o[i] - ov[i]) * m[i];
        }
    });
}

std::vector<KnobDef> pixelOpKnobs(const std::string& defChannels, bool defUnpremult) {
    return {K::channels("channels", "channels", defChannels, "Channels or layer to process"),
            K::boolean("unpremult", "(un)premult by alpha", defUnpremult,
                       "Divide by rgba.alpha before the operation and multiply after")};
}

Info PixelOp::info(Eval& ev, const Params& p) const {
    (void)p;
    if (!ev.connected(0)) return emptyInfo();
    return ev.inputInfo(0);
}

ImagePtr PixelOp::render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const {
    Info in = ev.inputInfo(0);
    ChannelList sel = resolveChannels(p.s("channels"), in.channels);
    bool unp = usesUnpremult() && p.b("unpremult") && hasChannel(in.channels, "rgba.alpha");
    ChannelList need = chans;
    std::vector<std::string> layers;
    for (auto& c : sel) {
        auto l = layerOf(c);
        if (std::find(layers.begin(), layers.end(), l) == layers.end()) layers.push_back(l);
    }
    for (auto& l : layers)
        for (auto& s : layerSlots(l, in.channels))
            if (!s.empty()) need.push_back(s);
    if (hasChannel(in.channels, "rgba.alpha")) need.push_back("rgba.alpha");
    need = sortedChannels(intersect(need, in.channels));
    ImagePtr src = ev.input(0, roi, need);
    auto out = copyOf(*src, roi, chans);
    RowKernel k = kernel(p, in);
    ChannelList changed;
    for (auto& layer : layers) {
        auto slots = layerSlots(layer, in.channels);
        bool active[4];
        bool any = false;
        for (int i = 0; i < 4; ++i) {
            active[i] = !slots[i].empty() && hasChannel(sel, slots[i]) && hasChannel(chans, slots[i]);
            any |= active[i];
            if (active[i]) changed.push_back(slots[i]);
        }
        if (!any) continue;
        std::string alphaCh = !slots[3].empty() ? slots[3] : "rgba.alpha";
        ev.parallelRows(roi, [&](int y) {
            int n = roi.w();
            std::vector<float> buf(size_t(n) * 5, 0.0f);
            float* ch[4] = {buf.data(), buf.data() + n, buf.data() + 2 * n, buf.data() + 3 * n};
            float* pa = buf.data() + 4 * n;
            for (int i = 0; i < 4; ++i) {
                std::string name = i == 3 ? alphaCh : slots[i];
                if (name.empty()) continue;
                src->readRow(name, y, roi.x0, n, ch[i]);
            }
            if (unp) {
                src->readRow("rgba.alpha", y, roi.x0, n, pa);
                for (int x = 0; x < n; ++x) {
                    if (pa[x] > 1e-6f)
                        for (int i = 0; i < 3; ++i) ch[i][x] /= pa[x];
                }
            }
            k(ch[0], ch[1], ch[2], ch[3], n, roi.x0, y);
            if (unp) {
                for (int x = 0; x < n; ++x)
                    if (pa[x] > 1e-6f)
                        for (int i = 0; i < 3; ++i) ch[i][x] *= pa[x];
            }
            for (int i = 0; i < 4; ++i) {
                if (!active[i]) continue;
                float* o = out->row(slots[i], y);
                std::memcpy(o, ch[i], sizeof(float) * n);
            }
        });
    }
    applyMaskMix(ev, p, 1, *src, *out, changed);
    return out;
}

// ------------------------------------------------------------ blur

namespace {

std::vector<float> kernel1d(double s, int filter, double quality) {
    if (s <= 0.0) return {1.0f};
    std::vector<double> w;
    int r;
    if (filter == 3) {
        r = int(std::ceil(quality * s));
        for (int i = -r; i <= r; ++i) w.push_back(std::exp(-0.5 * i * i / (s * s)));
    } else {
        r = int(std::ceil(s));
        for (int i = -r; i <= r; ++i) {
            double a = std::abs(i);
            double v;
            if (filter == 0) v = a <= std::floor(s) ? 1.0 : s - std::floor(s);
            else if (filter == 1) v = std::max(0.0, 1.0 - a / (s + 1.0));
            else v = std::max(0.0, 1.0 - (a / (s + 1.0)) * (a / (s + 1.0)));
            w.push_back(v);
        }
    }
    double sum = 0;
    for (double v : w) sum += v;
    std::vector<float> k(w.size());
    for (size_t i = 0; i < w.size(); ++i) k[i] = float(w[i] / sum);
    return k;
}

std::vector<int> box3Widths(double sigma) {
    double wIdeal = std::sqrt(12.0 * sigma * sigma / 3.0 + 1.0);
    int wl = int(std::floor(wIdeal));
    if (wl % 2 == 0) wl--;
    int wu = wl + 2;
    double mIdeal = (12.0 * sigma * sigma - 3.0 * wl * wl - 12.0 * wl - 9.0) / (-4.0 * wl - 4.0);
    int m = int(std::lround(mIdeal));
    return {0 < m ? wl : wu, 1 < m ? wl : wu, 2 < m ? wl : wu};
}

// In-place running-box blur of a 1D array (zeros beyond the ends).
void boxPass(std::vector<double>& a, int r, std::vector<double>& tmp) {
    if (r <= 0) return;
    size_t L = a.size();
    tmp.assign(L + 1, 0.0);
    for (size_t i = 0; i < L; ++i) tmp[i + 1] = tmp[i] + a[i];
    double inv = 1.0 / (2 * r + 1);
    for (size_t i = 0; i < L; ++i) {
        long lo = long(i) - r, hi = long(i) + r + 1;
        lo = std::max(0L, lo);
        hi = std::min(long(L), hi);
        a[i] = (tmp[hi] - tmp[lo]) * inv;
    }
}

}  // namespace

int blurRadius(double s, int filter, double quality) {
    if (s <= 0) return 0;
    if (filter == 4) {  // box3 approximation of a gaussian
        int r = 0;
        for (int w : box3Widths(s)) r += (w - 1) / 2;
        return r;
    }
    if (filter == 3) return int(std::ceil(quality * s));
    return int(std::ceil(s));
}

void blurPlane(const Image& src, const std::string& ch, const Box& outBox, float* out, double sx, double sy,
               int filter, double quality, const CancelToken& cancel) {
    int rx = blurRadius(sx, filter, quality), ry = blurRadius(sy, filter, quality);
    Box mid{outBox.x0, outBox.y0 - ry, outBox.x1, outBox.y1 + ry};
    std::vector<float> tmp(size_t(mid.w()) * mid.h(), 0.0f);
    const int W = outBox.w();
    auto& pool = ThreadPool::global();

    if (filter == 4) {
        auto wx = box3Widths(sx), wy = box3Widths(sy);
        pool.parallelFor(mid.y0, mid.y1, [&](int y0, int y1) {
            std::vector<double> row, t;
            std::vector<float> fr(size_t(W + 2 * rx));
            int ci = src.index(ch);
            for (int y = y0; y < y1; ++y) {
                row.assign(size_t(W + 2 * rx), 0.0);
                src.readRow(ci, y, outBox.x0 - rx, W + 2 * rx, fr.data());
                for (int i = 0; i < W + 2 * rx; ++i) row[i] = fr[i];
                if (sx > 0)
                    for (int w : wx) boxPass(row, (w - 1) / 2, t);
                float* d = tmp.data() + size_t(y - mid.y0) * W;
                for (int x = 0; x < W; ++x) d[x] = float(row[x + rx]);
            }
        }, &cancel);
        pool.parallelFor(0, W, [&](int x0, int x1) {
            std::vector<double> col, t;
            for (int x = x0; x < x1; ++x) {
                col.assign(size_t(mid.h()), 0.0);
                for (int y = 0; y < mid.h(); ++y) col[y] = tmp[size_t(y) * W + x];
                if (sy > 0)
                    for (int w : wy) boxPass(col, (w - 1) / 2, t);
                for (int y = 0; y < outBox.h(); ++y) out[size_t(y) * W + x] = float(col[y + ry]);
            }
        }, &cancel);
        return;
    }

    auto kx = kernel1d(sx, filter, quality), ky = kernel1d(sy, filter, quality);
    const float* sp = src.plane(ch);
    const Box sb = src.bbox();
    pool.parallelFor(mid.y0, mid.y1, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) {
            float* d = tmp.data() + size_t(y - mid.y0) * W;
            if (!sp || y < sb.y0 || y >= sb.y1) continue;
            const float* srow = sp + size_t(y - sb.y0) * sb.w();
            for (int x = outBox.x0; x < outBox.x1; ++x) {
                int a = std::max(-rx, sb.x0 - x), b = std::min(rx, sb.x1 - 1 - x);
                double acc = 0;
                for (int i = a; i <= b; ++i) acc += double(kx[i + rx]) * srow[x + i - sb.x0];
                d[x - outBox.x0] = float(acc);
            }
        }
    }, &cancel);
    pool.parallelFor(outBox.y0, outBox.y1, [&](int y0, int y1) {
        std::vector<double> acc(W);
        for (int y = y0; y < y1; ++y) {
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int i = -ry; i <= ry; ++i) {
                const float* t = tmp.data() + size_t(y + i - mid.y0) * W;
                double kv = ky[i + ry];
                for (int x = 0; x < W; ++x) acc[x] += kv * t[x];
            }
            float* o = out + size_t(y - outBox.y0) * W;
            for (int x = 0; x < W; ++x) o[x] = float(acc[x]);
        }
    }, &cancel);
}

// ------------------------------------------------------------ morphology

void morphPlane(const float* in, const Box& inBox, float* out, const Box& outBox, double rx, double ry,
                const CancelToken& cancel) {
    auto get = [&](int x, int y) -> float {
        if (x < inBox.x0 || x >= inBox.x1 || y < inBox.y0 || y >= inBox.y1) return 0.0f;
        return in[size_t(y - inBox.y0) * inBox.w() + (x - inBox.x0)];
    };
    bool dilate = rx > 0 || ry > 0;
    double ax = std::fabs(rx), ay = std::fabs(ry);
    int fx = int(std::floor(ax)), fy = int(std::floor(ay));
    double tx = ax - fx, ty = ay - fy;
    int px = fx + (tx > 0 ? 1 : 0), py = fy + (ty > 0 ? 1 : 0);
    Box mid{outBox.x0, outBox.y0 - py, outBox.x1, outBox.y1 + py};
    std::vector<float> tmp(size_t(mid.w()) * mid.h());
    auto pick = [dilate](float a, float b) { return dilate ? std::max(a, b) : std::min(a, b); };
    ThreadPool::global().parallelFor(mid.y0, mid.y1, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = mid.x0; x < mid.x1; ++x) {
                float v = get(x, y);
                for (int i = 1; i <= fx; ++i) v = pick(v, pick(get(x - i, y), get(x + i, y)));
                if (tx > 0) {
                    float e = pick(v, pick(get(x - fx - 1, y), get(x + fx + 1, y)));
                    v = float(v + (e - v) * tx);
                }
                tmp[size_t(y - mid.y0) * mid.w() + (x - mid.x0)] = v;
            }
    }, &cancel);
    auto getT = [&](int x, int y) { return tmp[size_t(y - mid.y0) * mid.w() + (x - mid.x0)]; };
    ThreadPool::global().parallelFor(outBox.y0, outBox.y1, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = outBox.x0; x < outBox.x1; ++x) {
                float v = getT(x, y);
                for (int i = 1; i <= fy; ++i) v = pick(v, pick(getT(x, y - i), getT(x, y + i)));
                if (ty > 0) {
                    float e = pick(v, pick(getT(x, y - fy - 1), getT(x, y + fy + 1)));
                    v = float(v + (e - v) * ty);
                }
                out[size_t(y - outBox.y0) * outBox.w() + (x - outBox.x0)] = v;
            }
    }, &cancel);
}

// ------------------------------------------------------------ sampling

int filterRadius(int f) {
    switch (f) {
        case FilterImpulse: return 0;
        case FilterBilinear: return 1;
        case FilterCubic: return 2;
        default: return 3;
    }
}

namespace {
inline float px(const float* p, const Box& b, int x, int y) {
    if (x < b.x0 || x >= b.x1 || y < b.y0 || y >= b.y1) return 0.0f;
    return p[size_t(y - b.y0) * b.w() + (x - b.x0)];
}
inline double cubicW(double t) {
    t = std::fabs(t);
    const double a = -0.5;
    if (t < 1) return (a + 2) * t * t * t - (a + 3) * t * t + 1;
    if (t < 2) return a * t * t * t - 5 * a * t * t + 8 * a * t - 4 * a;
    return 0;
}
inline double lanczosW(double t) {
    t = std::fabs(t);
    if (t < 1e-9) return 1;
    if (t >= 3) return 0;
    double pt = kPi * t;
    return 3 * std::sin(pt) * std::sin(pt / 3) / (pt * pt);
}
}  // namespace

float sample(const float* p, const Box& b, double x, double y, int filter) {
    if (!p) return 0.0f;
    if (filter == FilterImpulse) return px(p, b, int(std::floor(x)), int(std::floor(y)));
    double fx = x - 0.5, fy = y - 0.5;
    int ix = int(std::floor(fx)), iy = int(std::floor(fy));
    double tx = fx - ix, ty = fy - iy;
    if (filter == FilterBilinear) {
        double a = px(p, b, ix, iy), c = px(p, b, ix + 1, iy);
        double d = px(p, b, ix, iy + 1), e = px(p, b, ix + 1, iy + 1);
        return float((a * (1 - tx) + c * tx) * (1 - ty) + (d * (1 - tx) + e * tx) * ty);
    }
    int r = filter == FilterCubic ? 2 : 3;
    double wx[6], wy[6], sum = 0, acc = 0;
    for (int i = 0; i < 2 * r; ++i) {
        double ox = i - (r - 1) - tx, oy = i - (r - 1) - ty;
        wx[i] = filter == FilterCubic ? cubicW(ox) : lanczosW(ox);
        wy[i] = filter == FilterCubic ? cubicW(oy) : lanczosW(oy);
    }
    for (int j = 0; j < 2 * r; ++j)
        for (int i = 0; i < 2 * r; ++i) {
            double w = wx[i] * wy[j];
            acc += w * px(p, b, ix - (r - 1) + i, iy - (r - 1) + j);
            sum += w;
        }
    return float(sum != 0 ? acc / sum : 0);
}

}  // namespace vc::ops
