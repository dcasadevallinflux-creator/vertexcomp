// Python bindings for the core. Pixel buffers cross as NumPy arrays that own
// their memory (no per-pixel Python work). Long calls release the GIL.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <cstring>

#include "../core/src/ops/blobs.h"
#include "../core/src/ops/io_ops.h"
#include "vc/color.h"
#include "vc/expr.h"
#include "vc/graph.h"
#include "vc/io.h"
#include "vc/roto.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace vc;

namespace {

using Array3 = nb::ndarray<nb::numpy, float, nb::ndim<3>>;

Array3 makeArray(std::vector<float>&& data, size_t d0, size_t d1, size_t d2) {
    auto* buf = new std::vector<float>(std::move(data));
    nb::capsule owner(buf, [](void* p) noexcept { delete static_cast<std::vector<float>*>(p); });
    size_t shape[3] = {d0, d1, d2};
    return Array3(buf->data(), 3, shape, owner);
}

const char* knobTypeName(KnobType t) {
    switch (t) {
        case KnobType::Float: return "float";
        case KnobType::Int: return "int";
        case KnobType::Bool: return "bool";
        case KnobType::Enum: return "enum";
        case KnobType::String: return "string";
        case KnobType::File: return "file";
        case KnobType::Color: return "color";
        case KnobType::XY: return "xy";
        case KnobType::Box: return "box";
        case KnobType::Channel: return "channel";
        case KnobType::Channels: return "channels";
        case KnobType::Button: return "button";
        case KnobType::Range: return "range";
        case KnobType::Text: return "text";
    }
    return "?";
}

bool isStringType(KnobType t) {
    return t == KnobType::String || t == KnobType::File || t == KnobType::Channel || t == KnobType::Channels ||
           t == KnobType::Text || t == KnobType::Enum;
}

nb::dict knobDict(const KnobDef& k) {
    nb::dict d;
    d["name"] = k.name;
    d["label"] = k.label;
    d["type"] = knobTypeName(k.type);
    d["comps"] = k.comps;
    d["default"] = k.def;
    d["sdefault"] = k.sdef;
    d["min"] = k.smin;
    d["max"] = k.smax;
    d["options"] = k.options;
    d["tooltip"] = k.tooltip;
    d["group"] = k.group;
    d["animatable"] = k.animatable && !isStringType(k.type) && k.type != KnobType::Button;
    return d;
}

nb::dict infoDict(const Info& i) {
    nb::dict d;
    d["format"] = std::make_tuple(i.format.w, i.format.h, i.format.par);
    d["bbox"] = std::make_tuple(i.bbox.x0, i.bbox.y0, i.bbox.x1, i.bbox.y1);
    d["channels"] = i.channels;
    d["first"] = i.first;
    d["last"] = i.last;
    d["meta"] = i.meta;
    return d;
}

Box toBox(const std::tuple<int, int, int, int>& t) {
    return Box{std::get<0>(t), std::get<1>(t), std::get<2>(t), std::get<3>(t)};
}

std::recursive_mutex& gm(Node& n) { return n.graph()->mutex(); }

KnobValue& knob(Node& n, const std::string& k) {
    auto it = n.knobs.find(k);
    if (it == n.knobs.end()) throw nb::key_error(("no knob '" + k + "' on " + n.name).c_str());
    return it->second;
}

Curve* rotoCurve(RotoShape& s, const std::string& name, double** staticValue) {
    struct E {
        const char* n;
        Curve RotoShape::*c;
        double RotoShape::*v;
    };
    static const E table[] = {{"tx", &RotoShape::tx, &RotoShape::txv},  {"ty", &RotoShape::ty, &RotoShape::tyv},
                              {"rot", &RotoShape::rot, &RotoShape::rotv}, {"sx", &RotoShape::sx, &RotoShape::sxv},
                              {"sy", &RotoShape::sy, &RotoShape::syv},  {"cx", &RotoShape::cx, &RotoShape::cxv},
                              {"cy", &RotoShape::cy, &RotoShape::cyv}};
    for (auto& e : table)
        if (name == e.n) {
            if (staticValue) *staticValue = &(s.*e.v);
            return &(s.*e.c);
        }
    throw nb::key_error(("no transform parameter '" + name + "'").c_str());
}

struct MovieCallbacks {
    nb::object info, frame;
};
MovieCallbacks* gMovie = nullptr;  // intentionally leaked: callbacks may run during shutdown

}  // namespace

NB_MODULE(_vccore, m) {
    m.doc() = "Vertex Comp core";
    // Objects owned by Qt-side Python wrappers can outlive the interpreter teardown.
    nb::set_leak_warnings(false);
    nb::exception<Cancelled>(m, "Cancelled");

    nb::class_<CancelToken>(m, "CancelToken")
        .def(nb::init<>())
        .def("cancel", &CancelToken::cancel)
        .def_prop_ro("cancelled", &CancelToken::cancelled);

    // ---------------------------------------------------------------- roto
    nb::class_<RotoPoint>(m, "RotoPoint")
        .def(nb::init<>())
        .def("__init__",
             [](RotoPoint* p, double x, double y, double inX, double inY, double outX, double outY, double feather,
                bool smooth) {
                 new (p) RotoPoint();
                 p->x = x; p->y = y; p->inX = inX; p->inY = inY; p->outX = outX; p->outY = outY;
                 p->feather = feather; p->smooth = smooth;
             },
             "x"_a, "y"_a, "in_x"_a = 0.0, "in_y"_a = 0.0, "out_x"_a = 0.0, "out_y"_a = 0.0, "feather"_a = 0.0,
             "smooth"_a = true)
        .def_rw("x", &RotoPoint::x)
        .def_rw("y", &RotoPoint::y)
        .def_rw("in_x", &RotoPoint::inX)
        .def_rw("in_y", &RotoPoint::inY)
        .def_rw("out_x", &RotoPoint::outX)
        .def_rw("out_y", &RotoPoint::outY)
        .def_rw("feather", &RotoPoint::feather)
        .def_rw("smooth", &RotoPoint::smooth)
        .def("copy", [](const RotoPoint& p) { return p; });

    nb::class_<RotoShape>(m, "RotoShape")
        .def(nb::init<>())
        .def_rw("name", &RotoShape::name)
        .def_rw("visible", &RotoShape::visible)
        .def_rw("locked", &RotoShape::locked)
        .def_rw("closed", &RotoShape::closed)
        .def_rw("invert", &RotoShape::invert)
        .def_rw("color", &RotoShape::color)
        .def_prop_rw("blend", [](const RotoShape& s) { return int(s.blend); },
                     [](RotoShape& s, int v) { s.blend = RotoBlend(std::max(0, std::min(5, v))); })
        .def_rw("opacity", &RotoShape::opacity)
        .def_rw("feather", &RotoShape::feather)
        .def_rw("feather_falloff", &RotoShape::featherFalloff)
        .def_rw("blur", &RotoShape::blur)
        .def_rw("motion_blur_samples", &RotoShape::motionBlurSamples)
        .def_rw("shutter", &RotoShape::shutter)
        .def_rw("keys", &RotoShape::keys)
        .def("points_at", &RotoShape::pointsAt)
        .def("transformed_at", &RotoShape::transformedAt)
        .def("transform_at", [](const RotoShape& s, double f) {
            double mm[6];
            s.transformAt(f, mm);
            return std::vector<double>(mm, mm + 6);
        })
        .def("get_transform", [](RotoShape& s, const std::string& n, double f) {
            double* v = nullptr;
            Curve* c = rotoCurve(s, n, &v);
            return c->animated() ? c->eval(f) : *v;
        })
        .def("set_transform", [](RotoShape& s, const std::string& n, double value, std::optional<double> frame) {
            double* v = nullptr;
            Curve* c = rotoCurve(s, n, &v);
            if (frame) {
                Key k;
                k.frame = *frame;
                k.value = value;
                k.interp = InterpSmooth;
                c->set(k);
            } else if (c->animated()) {
                for (auto& k : c->keys) k.value = value;
                *v = value;
            } else {
                *v = value;
            }
        }, "name"_a, "value"_a, "frame"_a = nb::none())
        .def("transform_keys", [](RotoShape& s, const std::string& n) {
            std::vector<std::pair<double, double>> out;
            for (auto& k : rotoCurve(s, n, nullptr)->keys) out.push_back({k.frame, k.value});
            return out;
        })
        .def("set_transform_keys", [](RotoShape& s, const std::string& n, const std::vector<std::pair<double, double>>& ks) {
            Curve* c = rotoCurve(s, n, nullptr);
            c->keys.clear();
            for (auto& kv : ks) {
                Key k;
                k.frame = kv.first;
                k.value = kv.second;
                c->set(k);
            }
        })
        .def("copy", [](const RotoShape& s) { return s; });

    m.def("flatten_shape", [](const std::vector<RotoPoint>& pts, bool closed) { return flattenShape(pts, closed); });

    // ---------------------------------------------------------------- node
    nb::class_<Node>(m, "Node")
        .def_prop_ro("id", &Node::id)
        .def_prop_ro("type", [](Node& n) { return n.type(); })
        .def_prop_rw("name", [](Node& n) {
                std::lock_guard<std::recursive_mutex> lk(gm(n));
                return n.name;
            },
            [](Node& n, const std::string& v) {
                std::lock_guard<std::recursive_mutex> lk(gm(n));
                if (v.empty()) throw Error("empty name");
                auto other = n.graph()->find(v);
                if (other && other.get() != &n) throw Error("name '" + v + "' is taken");
                n.name = v;
            })
        .def_prop_rw("disabled", [](Node& n) { return n.disabled; },
                     [](Node& n, bool v) {
                         std::lock_guard<std::recursive_mutex> lk(gm(n));
                         n.disabled = v;
                     })
        .def_prop_rw("revision", [](Node& n) { return n.revision; },
                     [](Node& n, int v) {
                         std::lock_guard<std::recursive_mutex> lk(gm(n));
                         n.revision = v;
                     })
        .def_prop_ro("error", [](Node& n) {
            std::lock_guard<std::recursive_mutex> lk(gm(n));
            return n.error;
        })
        .def("clear_error", [](Node& n) {
            std::lock_guard<std::recursive_mutex> lk(gm(n));
            n.error.clear();
        })
        .def_prop_ro("inputs", [](Node& n) {
            std::lock_guard<std::recursive_mutex> lk(gm(n));
            return n.inputs;
        })
        .def_prop_ro("input_labels", [](Node& n) { return n.desc().inputs; })
        .def_prop_ro("knob_names", [](Node& n) {
            std::vector<std::string> v;
            for (auto& k : n.desc().knobs) v.push_back(k.name);
            return v;
        })
        .def("knob_def", [](Node& n, const std::string& k) {
            const KnobDef* d = n.knobDef(k);
            if (!d) throw nb::key_error(k.c_str());
            return knobDict(*d);
        })
        .def("get", [](Node& n, const std::string& k, double frame) -> nb::object {
            std::lock_guard<std::recursive_mutex> lk(gm(n));
            const KnobDef* d = n.knobDef(k);
            if (!d) throw nb::key_error(k.c_str());
            if (isStringType(d->type)) return nb::cast(knob(n, k).s);
            std::vector<double> v(knob(n, k).v.size());
            for (size_t c = 0; c < v.size(); ++c) v[c] = n.valueAt(k, int(c), frame);
            return nb::cast(v);
        }, "knob"_a, "frame"_a = 0.0)
        .def("set", [](Node& n, const std::string& k, nb::handle value) {
            std::lock_guard<std::recursive_mutex> lk(gm(n));
            const KnobDef* d = n.knobDef(k);
            if (!d) throw nb::key_error(k.c_str());
            KnobValue& kv = knob(n, k);
            if (isStringType(d->type)) {
                kv.s = nb::cast<std::string>(value);
                return;
            }
            if (d->type == KnobType::Button) return;
            std::vector<double> v;
            if (nb::isinstance<nb::float_>(value) || nb::isinstance<nb::int_>(value) || nb::isinstance<nb::bool_>(value))
                v.assign(kv.v.size(), nb::cast<double>(value));
            else
                v = nb::cast<std::vector<double>>(value);
            v.resize(kv.v.size(), v.empty() ? 0.0 : v.back());
            kv.v = v;
        })
        .def("set_key", [](Node& n, const std::string& k, int comp, double frame, double value, int interp,
                           double inSlope, double outSlope) {
            std::lock_guard<std::recursive_mutex> lk(gm(n));
            KnobValue& kv = knob(n, k);
            if (comp < 0 || comp >= int(kv.curves.size())) throw nb::index_error("component");
            Key key;
            key.frame = frame;
            key.value = value;
            key.interp = interp;
            key.inSlope = inSlope;
            key.outSlope = outSlope;
            kv.curves[comp].set(key);
        }, "knob"_a, "comp"_a, "frame"_a, "value"_a, "interp"_a = 2, "in_slope"_a = 0.0, "out_slope"_a = 0.0)
        .def("remove_key", [](Node& n, const std::string& k, int comp, double frame) {
            std::lock_guard<std::recursive_mutex> lk(gm(n));
            KnobValue& kv = knob(n, k);
            if (comp < 0) {
                for (auto& c : kv.curves) c.remove(frame);
            } else if (comp < int(kv.curves.size())) {
                kv.curves[comp].remove(frame);
            }
        })
        .def("clear_animation", [](Node& n, const std::string& k) {
            std::lock_guard<std::recursive_mutex> lk(gm(n));
            for (auto& c : knob(n, k).curves) c.keys.clear();
        })
        .def("keys", [](Node& n, const std::string& k) {
            std::lock_guard<std::recursive_mutex> lk(gm(n));
            std::vector<std::vector<std::tuple<double, double, int, double, double>>> out;
            for (auto& c : knob(n, k).curves) {
                std::vector<std::tuple<double, double, int, double, double>> ks;
                for (auto& key : c.keys) ks.emplace_back(key.frame, key.value, key.interp, key.inSlope, key.outSlope);
                out.push_back(ks);
            }
            return out;
        })
        .def("is_animated", [](Node& n, const std::string& k) {
            std::lock_guard<std::recursive_mutex> lk(gm(n));
            for (auto& c : knob(n, k).curves)
                if (c.animated()) return true;
            return false;
        })
        .def("set_expr", [](Node& n, const std::string& k, int comp, const std::string& e) {
            std::lock_guard<std::recursive_mutex> lk(gm(n));
            KnobValue& kv = knob(n, k);
            if (comp < 0 || comp >= int(kv.exprs.size())) throw nb::index_error("component");
            if (!e.empty()) {
                // Validate syntax and references now so the UI can report errors.
                Expr test(e, [&](const std::string& name) -> int {
                    if (name == "frame" || name == "t") return 0;
                    auto dot = name.find('.');
                    std::string head = dot == std::string::npos ? name : name.substr(0, dot);
                    if (n.knobs.count(head) || n.graph()->find(head)) return 0;
                    return -1;
                });
            }
            kv.exprs[comp] = e;
        })
        .def("exprs", [](Node& n, const std::string& k) {
            std::lock_guard<std::recursive_mutex> lk(gm(n));
            return knob(n, k).exprs;
        })
        .def("knob_state", [](Node& n, const std::string& k) {
            // Full knob state (static value, animation, expressions) for undo and project files.
            std::lock_guard<std::recursive_mutex> lk(gm(n));
            const KnobValue& kv = knob(n, k);
            nb::dict d;
            d["v"] = kv.v;
            d["s"] = kv.s;
            d["exprs"] = kv.exprs;
            std::vector<std::vector<std::tuple<double, double, int, double, double>>> curves;
            for (auto& c : kv.curves) {
                std::vector<std::tuple<double, double, int, double, double>> ks;
                for (auto& key : c.keys) ks.emplace_back(key.frame, key.value, key.interp, key.inSlope, key.outSlope);
                curves.push_back(ks);
            }
            d["curves"] = curves;
            return d;
        })
        .def("set_knob_state", [](Node& n, const std::string& k, nb::dict d) {
            std::lock_guard<std::recursive_mutex> lk(gm(n));
            KnobValue& kv = knob(n, k);
            if (d.contains("s")) kv.s = nb::cast<std::string>(d["s"]);
            if (d.contains("v")) {
                auto v = nb::cast<std::vector<double>>(d["v"]);
                if (!kv.v.empty()) {
                    v.resize(kv.v.size(), v.empty() ? 0.0 : v.back());
                    kv.v = v;
                }
            }
            if (d.contains("exprs")) {
                auto e = nb::cast<std::vector<std::string>>(d["exprs"]);
                e.resize(kv.exprs.size());
                kv.exprs = e;
            }
            if (d.contains("curves")) {
                auto cs = nb::cast<std::vector<std::vector<std::tuple<double, double, int, double, double>>>>(d["curves"]);
                for (size_t c = 0; c < kv.curves.size(); ++c) {
                    kv.curves[c].keys.clear();
                    if (c >= cs.size()) continue;
                    for (auto& t : cs[c]) {
                        Key key;
                        key.frame = std::get<0>(t);
                        key.value = std::get<1>(t);
                        key.interp = std::get<2>(t);
                        key.inSlope = std::get<3>(t);
                        key.outSlope = std::get<4>(t);
                        kv.curves[c].set(key);
                    }
                }
            }
        })
        .def("roto_shapes", [](Node& n) {
            std::lock_guard<std::recursive_mutex> lk(gm(n));
            if (!n.roto) throw Error(n.name + " does not store shape data");
            return n.roto->shapes;
        })
        .def("set_roto_shapes", [](Node& n, const std::vector<RotoShape>& shapes) {
            auto d = std::make_shared<RotoData>();
            d->shapes = shapes;
            std::lock_guard<std::recursive_mutex> lk(gm(n));
            if (!n.roto) throw Error(n.name + " does not store shape data");
            n.roto = d;
        });

    // ---------------------------------------------------------------- graph
    nb::class_<Graph>(m, "Graph")
        .def(nb::init<>())
        .def("create", &Graph::create, "type"_a, "name"_a = "")
        .def("remove", &Graph::remove)
        .def("find", &Graph::find)
        .def("find_id", &Graph::findId)
        .def("nodes", &Graph::nodes)
        .def("connect", [](Graph& g, const NodePtr& dst, int slot, std::optional<NodePtr> src) {
            g.connect(dst, slot, src ? *src : nullptr);
        }, "dst"_a, "slot"_a, "src"_a.none())
        .def("would_cycle", &Graph::wouldCycle)
        .def("unique_name", &Graph::uniqueName)
        .def("clear", &Graph::clear);

    // ---------------------------------------------------------------- engine
    nb::class_<Engine>(m, "Engine")
        .def(nb::init<Graph&, size_t>(), "graph"_a, "cache_bytes"_a = size_t(4) << 30, nb::keep_alive<1, 2>())
        .def("info", [](Engine& e, const NodePtr& n, double frame) {
            Info i;
            {
                nb::gil_scoped_release r;
                i = e.info(n, frame);
            }
            return infoDict(i);
        })
        .def("hash", &Engine::hash)
        .def("is_cached", &Engine::isCached)
        .def("render", [](Engine& e, const NodePtr& n, double frame, std::optional<std::tuple<int, int, int, int>> roi,
                          std::optional<std::vector<std::string>> chans, CancelToken* cancel) {
            ImagePtr img;
            Box box;
            ChannelList cl;
            {
                nb::gil_scoped_release r;
                Info inf = e.info(n, frame, cancel);
                box = roi ? toBox(*roi) : inf.bbox;
                cl = chans ? *chans : inf.channels;
                img = e.render(n, frame, box, cl, cancel);
            }
            box = box.intersect(img->bbox().empty() ? Box{} : img->bbox()).empty() ? Box{} : box.intersect(img->bbox());
            ChannelList present = intersect(cl, img->channels());
            std::vector<float> data(present.size() * size_t(box.w()) * box.h());
            for (size_t c = 0; c < present.size(); ++c) {
                int ci = img->index(present[c]);
                for (int y = box.y0; y < box.y1; ++y)
                    img->readRow(ci, y, box.x0, box.w(), data.data() + (c * box.h() + (y - box.y0)) * box.w());
            }
            return std::make_tuple(makeArray(std::move(data), present.size(), size_t(box.h()), size_t(box.w())),
                                   present, std::make_tuple(box.x0, box.y0, box.x1, box.y1));
        }, "node"_a, "frame"_a, "roi"_a = nb::none(), "channels"_a = nb::none(), "cancel"_a = nullptr)
        .def("fetch", [](Engine& e, const NodePtr& n, double frame, std::tuple<int, int, int, int> roiT,
                         std::vector<std::string> slots, int step, CancelToken* cancel) {
            // Display fetch: four slots (names or "") over roi, sub-sampled by
            // averaging step x step blocks. Returns (H, W, 4).
            step = std::max(1, step);
            Box roi = toBox(roiT);
            slots.resize(4);
            int W = std::max(0, (roi.w() + step - 1) / step), H = std::max(0, (roi.h() + step - 1) / step);
            std::vector<float> data(size_t(W) * H * 4, 0.0f);
            {
                nb::gil_scoped_release r;
                ChannelList want;
                for (auto& s : slots)
                    if (!s.empty()) want.push_back(s);
                ImagePtr img = e.render(n, frame, roi, sortedChannels(want), cancel);
                for (int s = 0; s < 4; ++s) {
                    if (slots[s].empty() || !img->has(slots[s])) continue;
                    const float* p = img->plane(slots[s]);
                    const Box& ib = img->bbox();
                    if (step == 1) {
                        int ci = img->index(slots[s]);
                        ThreadPool::global().parallelFor(0, H, [&](int y0, int y1) {
                            std::vector<float> row(W);
                            for (int oy = y0; oy < y1; ++oy) {
                                img->readRow(ci, roi.y0 + oy, roi.x0, W, row.data());
                                float* d = data.data() + size_t(oy) * W * 4 + s;
                                for (int x = 0; x < W; ++x) d[size_t(x) * 4] = row[x];
                            }
                        });
                        continue;
                    }
                    ThreadPool::global().parallelFor(0, H, [&](int y0, int y1) {
                        for (int oy = y0; oy < y1; ++oy)
                            for (int ox = 0; ox < W; ++ox) {
                                double acc = 0;
                                int cnt = 0;
                                for (int j = 0; j < step; ++j) {
                                    int y = roi.y0 + oy * step + j;
                                    if (y >= roi.y1) break;
                                    for (int i = 0; i < step; ++i) {
                                        int x = roi.x0 + ox * step + i;
                                        if (x >= roi.x1) break;
                                        cnt++;
                                        if (x < ib.x0 || x >= ib.x1 || y < ib.y0 || y >= ib.y1) continue;
                                        acc += p[size_t(y - ib.y0) * ib.w() + (x - ib.x0)];
                                    }
                                }
                                data[(size_t(oy) * W + ox) * 4 + s] = cnt ? float(acc / cnt) : 0.0f;
                            }
                    });
                }
            }
            return makeArray(std::move(data), size_t(H), size_t(W), 4);
        }, "node"_a, "frame"_a, "roi"_a, "slots"_a, "step"_a = 1, "cancel"_a = nullptr)
        .def("sample", [](Engine& e, const NodePtr& n, double frame, int x, int y, std::vector<std::string> chans) {
            nb::gil_scoped_release r;
            ImagePtr img = e.render(n, frame, Box{x, y, x + 1, y + 1}, chans);
            std::vector<float> v;
            for (auto& c : chans) v.push_back(img->at(c, x, y));
            return v;
        })
        .def("set_cache_limit", [](Engine& e, size_t mb) { e.cache().setLimit(mb << 20); })
        .def("clear_cache", [](Engine& e) { e.cache().clear(); })
        .def("cache_stats", [](Engine& e) {
            nb::dict d;
            d["used"] = e.cache().used();
            d["limit"] = e.cache().limit();
            d["hits"] = e.cache().hits;
            d["misses"] = e.cache().misses;
            return d;
        })
        .def("stats", [](Engine& e) {
            nb::dict d;
            for (auto& kv : e.stats()) d[nb::str(kv.first.c_str())] = std::make_tuple(kv.second.ms, kv.second.renders);
            return d;
        })
        .def("reset_stats", &Engine::resetStats)
        .def("write_frame", [](Engine& e, const NodePtr& n, double frame, CancelToken* cancel) {
            nb::gil_scoped_release r;
            return ops::writeFrame(e, n, frame, cancel);
        }, "node"_a, "frame"_a, "cancel"_a = nullptr)
        .def("prepare_write", [](Engine& e, const NodePtr& n, double frame, CancelToken* cancel) {
            MutImagePtr img;
            ChannelList chans;
            Format fmt;
            {
                nb::gil_scoped_release r;
                img = ops::prepareWriteImage(e, n, frame, chans, fmt, cancel);
            }
            // Full format window, interleaved (H, W, C).
            int W = fmt.w, H = fmt.h, C = int(chans.size());
            std::vector<float> data(size_t(W) * H * C, 0.0f);
            {
                nb::gil_scoped_release r;
                ThreadPool::global().parallelFor(0, H, [&](int y0, int y1) {
                    std::vector<float> row(W);
                    for (int y = y0; y < y1; ++y)
                        for (int c = 0; c < C; ++c) {
                            img->readRow(chans[c], y, 0, W, row.data());
                            float* d = data.data() + size_t(y) * W * C + c;
                            for (int x = 0; x < W; ++x) d[size_t(x) * C] = row[x];
                        }
                });
            }
            return std::make_tuple(makeArray(std::move(data), size_t(H), size_t(W), size_t(C)), chans);
        }, "node"_a, "frame"_a, "cancel"_a = nullptr);

    // ---------------------------------------------------------------- registry / misc
    m.def("op_types", [] {
        nb::list out;
        for (auto* d : Registry::get().all()) {
            nb::dict o;
            o["type"] = d->type;
            o["category"] = d->category;
            o["help"] = d->help;
            o["inputs"] = d->inputs;
            o["required"] = d->required;
            nb::list ks;
            for (auto& k : d->knobs) ks.append(knobDict(k));
            o["knobs"] = ks;
            out.append(o);
        }
        return out;
    });
    m.def("set_threads", [](int n) { ThreadPool::global().resize(n); });
    m.def("thread_count", [] { return ThreadPool::global().size(); });
    m.def("detect_sequence", [](const std::string& p) {
        SeqInfo s = detectSequence(p);
        return std::make_tuple(s.pattern, s.first, s.last, s.sequence);
    });
    m.def("frame_path", &framePath);
    m.def("is_pattern", &isPattern);
    m.def("is_movie", &isMovie);
    m.def("colorspaces", &colorSpaceNames);
    m.def("set_working_space", &setWorkingSpace);
    m.def("working_space", &workingSpace);
    m.def("primaries_matrix", [](const std::string& from, const std::string& to) {
        Mat3 mm = primariesMatrix(colorSpace(from).primaries, colorSpace(to).primaries);
        return std::vector<double>(mm.begin(), mm.end());
    });
    m.def("convert_color", [](const std::string& from, const std::string& to, std::vector<float> rgb) {
        ColorTransform xf(from, to);
        rgb.resize(3);
        xf.apply(rgb[0], rgb[1], rgb[2]);
        return rgb;
    });
    m.def("validate_expression", [](const std::string& e, std::vector<std::string> vars) -> std::string {
        try {
            Expr x(e, [&](const std::string& n) -> int {
                for (size_t i = 0; i < vars.size(); ++i)
                    if (vars[i] == n) return int(i);
                return -1;
            });
            return "";
        } catch (const Error& err) {
            return err.what();
        }
    });
    m.def("eval_expression", [](const std::string& e, std::map<std::string, double> vars) {
        std::vector<double> vals;
        Expr x(e, [&](const std::string& n) -> int {
            auto it = vars.find(n);
            if (it == vars.end()) return -1;
            vals.push_back(it->second);
            return int(vals.size()) - 1;
        });
        return x.eval(vals.data());
    });

    m.def("read_image", [](const std::string& path) {
        FileImage fi;
        {
            nb::gil_scoped_release r;
            fi = readImage(path);
        }
        const Image& img = *fi.image;
        Box b = img.bbox();
        ChannelList ch = img.channels();
        std::vector<float> data(ch.size() * size_t(b.w()) * b.h());
        for (size_t c = 0; c < ch.size(); ++c)
            std::memcpy(data.data() + c * size_t(b.w()) * b.h(), img.plane(int(c)), sizeof(float) * size_t(b.w()) * b.h());
        return std::make_tuple(makeArray(std::move(data), ch.size(), size_t(b.h()), size_t(b.w())), ch,
                               std::make_tuple(b.x0, b.y0, b.x1, b.y1), std::make_tuple(fi.format.w, fi.format.h),
                               img.meta);
    });
    m.def("write_image", [](const std::string& path, nb::ndarray<float, nb::ndim<3>, nb::c_contig> arr,
                            std::vector<std::string> chans, std::tuple<int, int, int, int> bbox,
                            std::tuple<int, int> format, std::string fileType, std::string dataType,
                            std::string compression, int quality, std::map<std::string, std::string> meta) {
        Box b = toBox(bbox);
        if (arr.shape(0) != chans.size() || int(arr.shape(1)) != b.h() || int(arr.shape(2)) != b.w())
            throw Error("array shape must be (channels, bbox height, bbox width)");
        Image img(b, chans);
        size_t n = size_t(b.w()) * b.h();
        for (size_t c = 0; c < chans.size(); ++c) std::memcpy(img.plane(int(c)), arr.data() + c * n, sizeof(float) * n);
        img.meta = meta;
        Format f;
        f.w = std::get<0>(format);
        f.h = std::get<1>(format);
        WriteOptions o;
        o.fileType = fileType;
        o.dataType = dataType;
        o.compression = compression;
        o.quality = quality;
        nb::gil_scoped_release r;
        writeImage(path, img, f, chans, o);
    }, "path"_a, "array"_a, "channels"_a, "bbox"_a, "format"_a, "file_type"_a = "exr",
       "data_type"_a = "32 bit float", "compression"_a = "zip", "quality"_a = 90,
       "meta"_a = std::map<std::string, std::string>{});

    m.def("scopes", [](nb::ndarray<float, nb::ndim<3>, nb::c_contig> img, int width, float exposure) {
        // Histogram (4 x 256, log2 range -8..+4 stops mapped linearly on display value),
        // waveform (3 x 256 x width) and vectorscope (256 x 256) of a display buffer (H, W, 4).
        int H = int(img.shape(0)), W = int(img.shape(1)), C = int(img.shape(2));
        width = std::max(16, width);
        std::vector<float> hist(4 * 256, 0.0f), wave(size_t(3) * 256 * width, 0.0f), vec(256 * 256, 0.0f);
        const float* d = img.data();
        float m = std::pow(2.0f, exposure);
        auto enc = [](float v) {
            v = std::max(0.0f, v);
            return v <= 0.0031308f ? v * 12.92f : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
        };
        {
            nb::gil_scoped_release r;
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    const float* px = d + (size_t(y) * W + x) * C;
                    float e[3];
                    for (int c = 0; c < 3 && c < C; ++c) e[c] = enc(px[c] * m);
                    int col = std::min(width - 1, x * width / std::max(1, W));
                    for (int c = 0; c < 3 && c < C; ++c) {
                        int b = std::min(255, std::max(0, int(e[c] * 255.0f)));
                        hist[c * 256 + b] += 1;
                        wave[(size_t(c) * 256 + (255 - b)) * width + col] += 1;
                    }
                    float lum = 0.2126f * e[0] + 0.7152f * e[1] + 0.0722f * e[2];
                    hist[3 * 256 + std::min(255, std::max(0, int(lum * 255.0f)))] += 1;
                    float cb = (e[2] - lum) / 1.8556f, cr = (e[0] - lum) / 1.5748f;
                    int vx = std::min(255, std::max(0, int(128 + cb * 255))), vy = std::min(255, std::max(0, int(128 - cr * 255)));
                    vec[vy * 256 + vx] += 1;
                }
        }
        nb::dict out;
        out["histogram"] = makeArray(std::move(hist), 4, 256, 1);
        out["waveform"] = makeArray(std::move(wave), 3, 256, size_t(width));
        out["vectorscope"] = makeArray(std::move(vec), 256, 256, 1);
        return out;
    }, "image"_a, "width"_a = 256, "exposure"_a = 0.0f);

    // ---------------------------------------------------------------- image blobs (AI / Versions)
    m.def("set_image_blob", [](const std::string& id, nb::ndarray<float, nb::ndim<3>, nb::c_contig> arr, int x0, int y0) {
        // (H, W, C) premultiplied linear float; C = 1..4 mapped to rgba.
        int H = int(arr.shape(0)), W = int(arr.shape(1)), C = int(arr.shape(2));
        if (C < 1 || C > 4) throw nb::value_error("image blob needs 1-4 channels");
        ChannelList ch{"rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"};
        if (C < 4) ch.resize(std::max(3, C));
        auto img = std::make_shared<Image>(Box{x0, y0, x0 + W, y0 + H}, ch);
        const float* src = arr.data();
        {
            nb::gil_scoped_release r;
            for (int c = 0; c < int(ch.size()); ++c) {
                float* d = img->plane(c);
                const int sc = std::min(c, C - 1);
                for (size_t i = 0; i < size_t(W) * H; ++i) d[i] = src[i * C + sc];
            }
        }
        ops::setImageBlob(id, img);
    }, "id"_a, "array"_a, "x0"_a = 0, "y0"_a = 0);
    m.def("has_image_blob", [](const std::string& id) { return ops::imageBlob(id) != nullptr; });
    m.def("remove_image_blob", &ops::removeImageBlob);
    m.def("clear_image_blobs", &ops::clearImageBlobs);
    m.def("image_blob_ids", &ops::imageBlobIds);
    m.def("image_blob_bytes", &ops::imageBlobBytes);

    m.def("set_movie_provider", [](nb::object info, nb::object frame) {
        if (!gMovie) gMovie = new MovieCallbacks();
        gMovie->info = info;
        gMovie->frame = frame;
        if (info.is_none()) {
            setMovieProvider(nullptr, nullptr);
            return;
        }
        setMovieProvider(
            [](const std::string& path, MovieInfo& out) -> bool {
                nb::gil_scoped_acquire g;
                nb::object r = gMovie->info(path);
                if (r.is_none()) return false;
                auto t = nb::cast<std::tuple<int, int, int, double>>(r);
                out.frames = std::get<0>(t);
                out.width = std::get<1>(t);
                out.height = std::get<2>(t);
                out.fps = std::get<3>(t);
                return true;
            },
            [](const std::string& path, int f) -> MutImagePtr {
                nb::gil_scoped_acquire g;
                nb::object r = gMovie->frame(path, f);
                if (r.is_none()) return nullptr;
                auto a = nb::cast<nb::ndarray<float, nb::ndim<3>, nb::c_contig>>(r);
                int H = int(a.shape(0)), W = int(a.shape(1)), C = int(a.shape(2));
                ChannelList ch{"rgba.red", "rgba.green", "rgba.blue"};
                if (C >= 4) ch.push_back("rgba.alpha");
                auto img = std::make_shared<Image>(Box{0, 0, W, H}, ch);
                const float* src = a.data();
                for (int c = 0; c < int(ch.size()); ++c) {
                    float* d = img->plane(c);
                    for (size_t i = 0; i < size_t(W) * H; ++i) d[i] = src[i * C + std::min(c, C - 1)];
                }
                return img;
            });
    });
}
