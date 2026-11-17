#include "vc/roto.h"

#include <cmath>

namespace vc {

namespace {

double curveOr(const Curve& c, double v, double f) { return c.animated() ? c.eval(f) : v; }

RotoPoint lerpPoint(const RotoPoint& a, const RotoPoint& b, double t) {
    RotoPoint r = a;
    r.x = a.x + (b.x - a.x) * t;
    r.y = a.y + (b.y - a.y) * t;
    r.inX = a.inX + (b.inX - a.inX) * t;
    r.inY = a.inY + (b.inY - a.inY) * t;
    r.outX = a.outX + (b.outX - a.outX) * t;
    r.outY = a.outY + (b.outY - a.outY) * t;
    r.feather = a.feather + (b.feather - a.feather) * t;
    return r;
}

struct Seg {
    double x0, y0, x1, y1;
    double f0, f1;  // feather at the ends
    double minx, maxx, miny, maxy;
};

void gaussianBlurPlane(std::vector<float>& buf, int w, int h, double sigma, const CancelToken& cancel) {
    if (sigma <= 0.0) return;
    int r = int(std::ceil(3.0 * sigma));
    std::vector<float> k(2 * r + 1);
    double sum = 0;
    for (int i = -r; i <= r; ++i) sum += (k[i + r] = float(std::exp(-0.5 * i * i / (sigma * sigma))));
    for (auto& v : k) v = float(v / sum);
    std::vector<float> tmp(buf.size());
    ThreadPool::global().parallelFor(0, h, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < w; ++x) {
                double acc = 0;
                for (int i = -r; i <= r; ++i) {
                    int xx = x + i;
                    if (xx >= 0 && xx < w) acc += k[i + r] * buf[size_t(y) * w + xx];
                }
                tmp[size_t(y) * w + x] = float(acc);
            }
    }, &cancel);
    ThreadPool::global().parallelFor(0, h, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < w; ++x) {
                double acc = 0;
                for (int i = -r; i <= r; ++i) {
                    int yy = y + i;
                    if (yy >= 0 && yy < h) acc += k[i + r] * tmp[size_t(yy) * w + x];
                }
                buf[size_t(y) * w + x] = float(acc);
            }
    }, &cancel);
}

// Coverage of one flattened closed outline into buf (w*h over box).
void rasterizePoly(const std::vector<std::pair<double, double>>& poly, const std::vector<double>& feathers,
                   double baseFeather, int falloff, const Box& box, std::vector<float>& buf,
                   const CancelToken& cancel) {
    size_t n = poly.size();
    if (n < 3) return;
    std::vector<Seg> segs;
    segs.reserve(n);
    double maxF = 0;
    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        Seg s{poly[i].first, poly[i].second, poly[j].first, poly[j].second,
              baseFeather + feathers[i], baseFeather + feathers[j], 0, 0, 0, 0};
        if (s.x0 == s.x1 && s.y0 == s.y1) continue;
        s.minx = std::min(s.x0, s.x1);
        s.maxx = std::max(s.x0, s.x1);
        s.miny = std::min(s.y0, s.y1);
        s.maxy = std::max(s.y0, s.y1);
        maxF = std::max({maxF, std::fabs(s.f0), std::fabs(s.f1)});
        segs.push_back(s);
    }
    const double R = maxF + 1.5;
    const int w = box.w();

    ThreadPool::global().parallelFor(box.y0, box.y1, [&](int ya, int yb) {
        std::vector<std::pair<double, int>> cross;
        std::vector<const Seg*> near;
        for (int y = ya; y < yb; ++y) {
            double yc = y + 0.5;
            cross.clear();
            near.clear();
            for (auto& s : segs) {
                if ((s.y0 <= yc && yc < s.y1) || (s.y1 <= yc && yc < s.y0)) {
                    double t = (yc - s.y0) / (s.y1 - s.y0);
                    cross.push_back({s.x0 + t * (s.x1 - s.x0), s.y1 > s.y0 ? 1 : -1});
                }
                if (yc >= s.miny - R && yc <= s.maxy + R) near.push_back(&s);
            }
            std::sort(cross.begin(), cross.end());
            float* row = buf.data() + size_t(y - box.y0) * w;
            size_t ci = 0;
            int wind = 0;
            for (int x = box.x0; x < box.x1; ++x) {
                double xc = x + 0.5;
                while (ci < cross.size() && cross[ci].first <= xc) wind += cross[ci++].second;
                bool inside = wind != 0;
                double best = 1e30, bestF = baseFeather;
                for (const Seg* s : near) {
                    if (xc < s->minx - R || xc > s->maxx + R) continue;
                    double dx = s->x1 - s->x0, dy = s->y1 - s->y0;
                    double t = ((xc - s->x0) * dx + (yc - s->y0) * dy) / (dx * dx + dy * dy);
                    t = std::min(1.0, std::max(0.0, t));
                    double px = s->x0 + t * dx - xc, py = s->y0 + t * dy - yc;
                    double d2 = px * px + py * py;
                    if (d2 < best) {
                        best = d2;
                        bestF = s->f0 + (s->f1 - s->f0) * t;
                    }
                }
                float a;
                if (best > R * R) {
                    a = inside ? 1.0f : 0.0f;
                } else {
                    double d = std::sqrt(best) * (inside ? -1.0 : 1.0);
                    double f = bestF;
                    double width = std::fabs(f) + 1.0;
                    double t = f >= 0 ? (d + 0.5) / width : (d + 0.5 + std::fabs(f)) / width;
                    t = std::min(1.0, std::max(0.0, t));
                    double smooth = t * t * (3 - 2 * t);
                    double blend = falloff == 1 ? 0.0 : std::min(1.0, std::fabs(f));
                    a = float(1.0 - (t + (smooth - t) * blend));
                }
                row[x - box.x0] = a;
            }
        }
    }, &cancel);
}

}  // namespace

std::vector<RotoPoint> RotoShape::pointsAt(double f) const {
    if (keys.empty()) return {};
    if (f <= keys.begin()->first) return keys.begin()->second;
    if (f >= keys.rbegin()->first) return keys.rbegin()->second;
    auto hi = keys.upper_bound(f);
    auto lo = std::prev(hi);
    double t = (f - lo->first) / (hi->first - lo->first);
    const auto& a = lo->second;
    const auto& b = hi->second;
    std::vector<RotoPoint> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = i < b.size() ? lerpPoint(a[i], b[i], t) : a[i];
    return out;
}

void RotoShape::transformAt(double f, double m[6]) const {
    double tX = curveOr(tx, txv, f), tY = curveOr(ty, tyv, f);
    double r = curveOr(rot, rotv, f) * kPi / 180.0;
    double sX = curveOr(sx, sxv, f), sY = curveOr(sy, syv, f);
    double cX = curveOr(cx, cxv, f), cY = curveOr(cy, cyv, f);
    double c = std::cos(r), s = std::sin(r);
    // p' = C + T + R S (p - C)
    m[0] = c * sX;
    m[1] = -s * sY;
    m[3] = s * sX;
    m[4] = c * sY;
    m[2] = cX + tX - (m[0] * cX + m[1] * cY);
    m[5] = cY + tY - (m[3] * cX + m[4] * cY);
}

std::vector<RotoPoint> RotoShape::transformedAt(double f) const {
    auto pts = pointsAt(f);
    double m[6];
    transformAt(f, m);
    for (auto& p : pts) {
        double x = p.x, y = p.y;
        p.x = m[0] * x + m[1] * y + m[2];
        p.y = m[3] * x + m[4] * y + m[5];
        double ix = p.inX, iy = p.inY, ox = p.outX, oy = p.outY;
        p.inX = m[0] * ix + m[1] * iy;
        p.inY = m[3] * ix + m[4] * iy;
        p.outX = m[0] * ox + m[1] * oy;
        p.outY = m[3] * ox + m[4] * oy;
    }
    return pts;
}

std::vector<std::pair<double, double>> flattenShape(const std::vector<RotoPoint>& pts, bool closed,
                                                    std::vector<double>* featherOut) {
    std::vector<std::pair<double, double>> out;
    size_t n = pts.size();
    if (n == 0) return out;
    size_t segs = closed ? n : n - 1;
    for (size_t i = 0; i < segs; ++i) {
        const RotoPoint& a = pts[i];
        const RotoPoint& b = pts[(i + 1) % n];
        double p0x = a.x, p0y = a.y;
        double p1x = a.x + a.outX, p1y = a.y + a.outY;
        double p2x = b.x + b.inX, p2y = b.y + b.inY;
        double p3x = b.x, p3y = b.y;
        double L = std::hypot(p1x - p0x, p1y - p0y) + std::hypot(p2x - p1x, p2y - p1y) + std::hypot(p3x - p2x, p3y - p2y);
        bool straight = a.outX == 0 && a.outY == 0 && b.inX == 0 && b.inY == 0;
        int steps = straight ? 1 : std::max(2, std::min(512, int(std::ceil(L / 1.5))));
        for (int k = 0; k < steps; ++k) {
            double t = double(k) / steps, u = 1 - t;
            double x = u * u * u * p0x + 3 * u * u * t * p1x + 3 * u * t * t * p2x + t * t * t * p3x;
            double y = u * u * u * p0y + 3 * u * u * t * p1y + 3 * u * t * t * p2y + t * t * t * p3y;
            out.push_back({x, y});
            if (featherOut) featherOut->push_back(a.feather + (b.feather - a.feather) * t);
        }
    }
    if (!closed) {
        out.push_back({pts.back().x, pts.back().y});
        if (featherOut) featherOut->push_back(pts.back().feather);
    }
    return out;
}

void RotoData::hash(Hasher& h, double frame) const {
    for (auto& s : shapes) {
        h.add(s.name).add(int(s.visible)).add(int(s.closed)).add(int(s.invert)).add(int(s.blend));
        h.add(s.opacity).add(s.feather).add(s.featherFalloff).add(s.blur).add(s.motionBlurSamples).add(s.shutter);
        int samples = std::max(1, s.motionBlurSamples);
        for (int k = 0; k < samples; ++k) {
            double f = samples == 1 ? frame : frame + s.shutter * (double(k) / (samples - 1) - 0.5);
            for (auto& p : s.transformedAt(f)) h.add(p.x).add(p.y).add(p.inX).add(p.inY).add(p.outX).add(p.outY).add(p.feather);
        }
    }
}

void rasterizeRoto(const RotoData& data, double frame, const Box& box, float* out, const CancelToken& cancel) {
    const int w = box.w(), h = box.h();
    std::fill(out, out + size_t(w) * h, 0.0f);
    if (box.empty()) return;
    for (const RotoShape& s : data.shapes) {
        if (!s.visible || !s.closed || s.keys.empty()) continue;
        int pad = int(std::ceil(3.0 * s.blur));
        Box work = box.grow(pad, pad);
        std::vector<float> cov(size_t(work.w()) * work.h(), 0.0f), tmp;
        int samples = std::max(1, s.motionBlurSamples);
        for (int k = 0; k < samples; ++k) {
            double f = samples == 1 ? frame : frame + s.shutter * (double(k) / (samples - 1) - 0.5);
            std::vector<double> feathers;
            auto poly = flattenShape(s.transformedAt(f), true, &feathers);
            tmp.assign(cov.size(), 0.0f);
            rasterizePoly(poly, feathers, s.feather, s.featherFalloff, work, tmp, cancel);
            for (size_t i = 0; i < cov.size(); ++i) cov[i] += tmp[i] / samples;
        }
        gaussianBlurPlane(cov, work.w(), work.h(), s.blur, cancel);
        for (int y = 0; y < h; ++y) {
            const float* src = cov.data() + size_t(y + pad) * work.w() + pad;
            float* dst = out + size_t(y) * w;
            for (int x = 0; x < w; ++x) {
                float a = src[x];
                if (s.invert) a = 1.0f - a;
                a *= float(s.opacity);
                float& d = dst[x];
                switch (s.blend) {
                    case RotoBlend::Over: d = a + d * (1.0f - a); break;
                    case RotoBlend::Add: d = d + a; break;
                    case RotoBlend::Subtract: d = std::max(0.0f, d - a); break;
                    case RotoBlend::Max: d = std::max(d, a); break;
                    case RotoBlend::Min: d = std::min(d, a); break;
                    case RotoBlend::Multiply: d = d * a; break;
                }
            }
        }
    }
}

}  // namespace vc
