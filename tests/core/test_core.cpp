// Core unit tests without external frameworks: each TEST registers a function;
// main() runs them all and returns non-zero on failure.
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "vc/expr.h"
#include "vc/graph.h"
#include "vc/io.h"
#include "vc/roto.h"

using namespace vc;

namespace {
struct Case {
    const char* name;
    std::function<void()> fn;
};
std::vector<Case>& cases() {
    static std::vector<Case> c;
    return c;
}
struct Reg {
    Reg(const char* n, std::function<void()> f) { cases().push_back({n, std::move(f)}); }
};
int failures = 0;
#define TEST(name) static void name(); static Reg reg_##name(#name, name); static void name()
#define CHECK(...)                                                              \
    do {                                                                         \
        if (!(__VA_ARGS__)) {                                                         \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #__VA_ARGS__);        \
            failures++;                                                          \
        }                                                                        \
    } while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs(double(a) - double(b)) <= (tol))

const ChannelList kRGBA{"rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"};
}  // namespace

TEST(box_math) {
    Box a{0, 0, 10, 10}, b{5, 5, 20, 20};
    CHECK((a.intersect(b) == Box{5, 5, 10, 10}));
    CHECK((a.unite(b) == Box{0, 0, 20, 20}));
    CHECK(Box{3, 3, 3, 9}.empty());
}

TEST(channel_names) {
    CHECK(canonicalChannel("R") == "rgba.red");
    CHECK(canonicalChannel("diffuse.G") == "diffuse.green");
    CHECK(canonicalChannel("Z") == "depth.Z");
    auto r = resolveChannels("rgb,depth", {"rgba.red", "rgba.green", "rgba.blue", "rgba.alpha", "depth.Z"});
    CHECK(r.size() == 4);
}

TEST(expression) {
    double vars[1] = {0.25};
    Expr e("clamp(x * 4, 0, 1) + (x > 0.2 ? 2 : 0) + 2^3", [](const std::string& n) { return n == "x" ? 0 : -1; });
    CHECK_NEAR(e.eval(vars), 1 + 2 + 8, 1e-12);
    bool threw = false;
    try {
        Expr bad("1 +", nullptr);
    } catch (const Error&) {
        threw = true;
    }
    CHECK(threw);
}

TEST(curve_interpolation) {
    Curve c;
    c.set({0, 0, InterpLinear});
    c.set({10, 10, InterpLinear});
    CHECK_NEAR(c.eval(5), 5, 1e-12);
    c.keys[0].interp = InterpConstant;
    CHECK_NEAR(c.eval(5), 0, 1e-12);
    c.keys[0].interp = InterpSmooth;
    CHECK_NEAR(c.eval(5), 5, 1e-9);  // two keys, flat ends: symmetric
}

TEST(merge_over_and_hash_invalidation) {
    Graph g;
    Engine eng(g, size_t(256) << 20);
    auto b = g.create("Constant");
    b->knobs["color"].v = {0.2, 0.4, 0.6, 1.0};
    b->knobs["format"].v = {8, 8};
    auto a = g.create("Constant");
    a->knobs["color"].v = {0.5, 0.0, 0.0, 0.5};
    a->knobs["format"].v = {8, 8};
    auto m = g.create("Merge");
    g.connect(m, 0, b);
    g.connect(m, 1, a);
    auto img = eng.render(m, 1, Box{0, 0, 8, 8}, kRGBA);
    CHECK_NEAR(img->at("rgba.red", 3, 3), 0.5 + 0.2 * 0.5, 1e-6);
    CHECK_NEAR(img->at("rgba.alpha", 3, 3), 1.0, 1e-6);
    uint64_t hb = eng.hash(b, 1), hm = eng.hash(m, 1);
    a->knobs["color"].v = {0.1, 0.0, 0.0, 0.5};
    CHECK(eng.hash(b, 1) == hb);
    CHECK(eng.hash(m, 1) != hm);
}

TEST(roi_tiles_match_full) {
    Graph g;
    Engine eng(g, size_t(256) << 20);
    auto n = g.create("Noise");
    n->knobs["format"].v = {64, 48};
    auto bl = g.create("Blur");
    bl->knobs["size"].v = {3, 5};
    g.connect(bl, 0, n);
    auto full = eng.render(bl, 1, Box{0, 0, 64, 48}, kRGBA);
    eng.cache().clear();
    for (int y = 0; y < 48; y += 16)
        for (int x = 0; x < 64; x += 16) {
            auto t = eng.render(bl, 1, Box{x, y, x + 16, y + 16}, kRGBA);
            for (int j = y; j < y + 16; ++j)
                for (int i = x; i < x + 16; ++i)
                    for (auto& c : kRGBA) CHECK(t->at(c, i, j) == full->at(c, i, j));
        }
}

TEST(gaussian_normalized_and_bbox) {
    Graph g;
    Engine eng(g, size_t(256) << 20);
    auto c = g.create("Constant");
    c->knobs["color"].v = {1, 1, 1, 1};
    c->knobs["format"].v = {32, 32};
    auto gb = g.create("GaussianBlur");
    gb->knobs["sigma"].v = {2, 2};
    gb->knobs["method"].s = "exact";
    g.connect(gb, 0, c);
    Info i = eng.info(gb, 1);
    CHECK((i.bbox == Box{-6, -6, 38, 38}));
    auto img = eng.render(gb, 1, Box{0, 0, 32, 32}, kRGBA);
    CHECK_NEAR(img->at("rgba.red", 16, 16), 1.0, 1e-5);  // interior of a constant stays 1
    double sum = 0;
    auto all = eng.render(gb, 1, i.bbox, {"rgba.red"});
    for (int y = i.bbox.y0; y < i.bbox.y1; ++y)
        for (int x = i.bbox.x0; x < i.bbox.x1; ++x) sum += all->at("rgba.red", x, y);
    CHECK_NEAR(sum, 32.0 * 32.0, 1e-2);  // energy preserved
}

TEST(cycle_rejected) {
    Graph g;
    auto a = g.create("Grade"), b = g.create("Grade");
    g.connect(b, 0, a);
    bool threw = false;
    try {
        g.connect(a, 0, b);
    } catch (const Error&) {
        threw = true;
    }
    CHECK(threw);
}

TEST(roto_area) {
    RotoData d;
    RotoShape s;
    s.keys[1] = {RotoPoint{10, 10}, RotoPoint{30, 10}, RotoPoint{30, 20}, RotoPoint{10, 20}};
    d.shapes.push_back(s);
    std::vector<float> buf(40 * 30);
    CancelToken t;
    rasterizeRoto(d, 1, Box{0, 0, 40, 30}, buf.data(), t);
    double sum = 0;
    for (float v : buf) sum += v;
    CHECK_NEAR(sum, 200.0, 1e-3);
}

TEST(sequence_patterns) {
    CHECK(framePath("a/b.####.exr", 12) == "a/b.0012.exr");
    CHECK(framePath("b.%04d.exr", 7) == "b.0007.exr");
    CHECK(framePath("b.$F3.exr", 7) == "b.007.exr");
    CHECK(isPattern("x.@@.png"));
}

int main() {
    for (auto& c : cases()) {
        int before = failures;
        c.fn();
        std::printf("%s %s\n", failures == before ? "ok  " : "FAIL", c.name);
    }
    std::printf("%d failure(s)\n", failures);
    return failures ? 1 : 0;
}
