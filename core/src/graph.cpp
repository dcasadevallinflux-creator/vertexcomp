#include "vc/graph.h"

#include <chrono>
#include <cmath>
#include <set>

#include "vc/roto.h"

namespace vc {

// ---------------------------------------------------------------- curves

double Curve::eval(double f) const {
    if (keys.empty()) return 0.0;
    if (f <= keys.front().frame) return keys.front().value;
    if (f >= keys.back().frame) return keys.back().value;
    size_t i = 0;
    while (i + 1 < keys.size() && keys[i + 1].frame <= f) ++i;
    const Key& a = keys[i];
    const Key& b = keys[i + 1];
    double span = b.frame - a.frame;
    double t = (f - a.frame) / span;
    switch (a.interp) {
        case InterpConstant: return a.value;
        case InterpLinear: return a.value + (b.value - a.value) * t;
        default: break;
    }
    double m0, m1;  // slopes in value per frame
    if (a.interp == InterpBezier) {
        m0 = a.outSlope;
        m1 = b.inSlope;
    } else {
        // Catmull-Rom style slopes from neighbours; flat at curve ends.
        auto slopeAt = [&](size_t k) {
            if (k == 0 || k + 1 >= keys.size()) return 0.0;
            return (keys[k + 1].value - keys[k - 1].value) / (keys[k + 1].frame - keys[k - 1].frame);
        };
        m0 = slopeAt(i);
        m1 = slopeAt(i + 1);
    }
    double t2 = t * t, t3 = t2 * t;
    double h00 = 2 * t3 - 3 * t2 + 1, h10 = t3 - 2 * t2 + t, h01 = -2 * t3 + 3 * t2, h11 = t3 - t2;
    return h00 * a.value + h10 * span * m0 + h01 * b.value + h11 * span * m1;
}

void Curve::set(const Key& k) {
    for (auto& e : keys) {
        if (std::fabs(e.frame - k.frame) < 1e-6) {
            e = k;
            return;
        }
    }
    keys.push_back(k);
    std::sort(keys.begin(), keys.end(), [](const Key& a, const Key& b) { return a.frame < b.frame; });
}

void Curve::remove(double frame) {
    keys.erase(std::remove_if(keys.begin(), keys.end(), [&](const Key& k) { return std::fabs(k.frame - frame) < 1e-6; }),
               keys.end());
}

// ---------------------------------------------------------------- params

double Params::f(const std::string& k, int comp) const {
    auto it = num.find(k);
    if (it == num.end() || comp >= int(it->second.size())) return 0.0;
    return it->second[comp];
}

const std::string& Params::s(const std::string& k) const {
    static const std::string empty;
    auto it = str.find(k);
    return it == str.end() ? empty : it->second;
}

std::vector<double> Params::v(const std::string& k) const {
    auto it = num.find(k);
    return it == num.end() ? std::vector<double>{} : it->second;
}

// ---------------------------------------------------------------- registry

Registry& Registry::get() {
    static Registry r;
    return r;
}

void Registry::add(OpDesc d) {
    std::string t = d.type;
    ops_[t] = std::move(d);
}

const OpDesc* Registry::find(const std::string& type) const {
    auto it = ops_.find(type);
    return it == ops_.end() ? nullptr : &it->second;
}

std::vector<const OpDesc*> Registry::all() const {
    std::vector<const OpDesc*> v;
    for (auto& kv : ops_) v.push_back(&kv.second);
    return v;
}

// ---------------------------------------------------------------- node

static bool isStringKnob(KnobType t) {
    return t == KnobType::String || t == KnobType::File || t == KnobType::Channel || t == KnobType::Channels ||
           t == KnobType::Text || t == KnobType::Enum;
}

Node::Node(Graph* g, const OpDesc* desc, uint64_t id) : graph_(g), desc_(desc), id_(id), op_(desc->make()) {
    inputs.resize(desc->inputs.size());
    for (auto& kd : desc->knobs) {
        KnobValue kv;
        if (kd.type == KnobType::Enum) {
            kv.s = kd.sdef.empty() && !kd.options.empty() ? kd.options.front() : kd.sdef;
        } else if (isStringKnob(kd.type)) {
            kv.s = kd.sdef;
        } else if (kd.type != KnobType::Button) {
            kv.v = kd.def;
            kv.v.resize(kd.comps, kd.def.empty() ? 0.0 : kd.def.back());
            kv.curves.resize(kd.comps);
            kv.exprs.resize(kd.comps);
        }
        knobs[kd.name] = kv;
    }
    if (desc->type == "Roto" || desc->type == "Paint") roto = std::make_shared<RotoData>();
}

const KnobDef* Node::knobDef(const std::string& k) const {
    for (auto& kd : desc_->knobs)
        if (kd.name == k) return &kd;
    return nullptr;
}

double Node::valueAt(const std::string& k, int comp, double frame, int depth) const {
    auto it = knobs.find(k);
    if (it == knobs.end() || comp >= int(it->second.v.size())) return 0.0;
    const KnobValue& kv = it->second;
    if (comp < int(kv.exprs.size()) && !kv.exprs[comp].empty() && depth < 8) {
        std::vector<double> vals;
        auto resolve = [&](const std::string& name) -> int {
            double val;
            if (name == "frame" || name == "t") {
                val = frame;
            } else {
                // knob, knob.N, Node.knob or Node.knob.N (N = 0..3 or x/y/r/g/b/a)
                std::vector<std::string> parts;
                size_t b = 0, p;
                while ((p = name.find('.', b)) != std::string::npos) {
                    parts.push_back(name.substr(b, p - b));
                    b = p + 1;
                }
                parts.push_back(name.substr(b));
                auto compIndex = [](const std::string& s) -> int {
                    if (s == "x" || s == "r" || s == "0") return 0;
                    if (s == "y" || s == "g" || s == "1") return 1;
                    if (s == "b" || s == "2") return 2;
                    if (s == "a" || s == "3") return 3;
                    return -1;
                };
                const Node* target = this;
                std::string knob;
                int c = 0;
                if (parts.size() == 1) {
                    knob = parts[0];
                } else if (parts.size() == 2 && compIndex(parts[1]) >= 0 && knobs.count(parts[0])) {
                    knob = parts[0];
                    c = compIndex(parts[1]);
                } else {
                    auto other = graph_->find(parts[0]);
                    if (!other) return -1;
                    target = other.get();
                    knob = parts[1];
                    if (parts.size() > 2) c = std::max(0, compIndex(parts[2]));
                }
                if (!target->knobs.count(knob)) return -1;
                val = target->valueAt(knob, c, frame, depth + 1);
            }
            vals.push_back(val);
            return int(vals.size()) - 1;
        };
        try {
            Expr e(kv.exprs[comp], resolve);
            return e.eval(vals.data());
        } catch (const Error&) {
            // Fall through to the static/animated value; the UI reports the
            // error when the expression is entered.
        }
    }
    if (comp < int(kv.curves.size()) && kv.curves[comp].animated()) return kv.curves[comp].eval(frame);
    return kv.v[comp];
}

Params Node::params(double frame) const {
    Params p;
    for (auto& kd : desc_->knobs) {
        auto& kv = knobs.at(kd.name);
        if (kd.type == KnobType::Button) continue;
        if (isStringKnob(kd.type)) {
            p.str[kd.name] = kv.s;
        } else {
            std::vector<double> v(kv.v.size());
            for (size_t c = 0; c < v.size(); ++c) v[c] = valueAt(kd.name, int(c), frame);
            p.num[kd.name] = std::move(v);
        }
    }
    p.num["__frame"] = {frame};
    return p;
}

// ---------------------------------------------------------------- graph

std::string Graph::uniqueName(const std::string& base) const {
    std::lock_guard<std::recursive_mutex> lk(mtx_);
    for (int i = 1;; ++i) {
        std::string n = base + std::to_string(i);
        if (!find(n)) return n;
    }
}

NodePtr Graph::create(const std::string& type, const std::string& name) {
    const OpDesc* d = Registry::get().find(type);
    if (!d) throw Error("unknown node type '" + type + "'");
    std::lock_guard<std::recursive_mutex> lk(mtx_);
    auto n = std::make_shared<Node>(this, d, nextId_++);
    n->name = name.empty() || find(name) ? uniqueName(type) : name;
    nodes_.push_back(n);
    return n;
}

void Graph::remove(const NodePtr& n) {
    std::lock_guard<std::recursive_mutex> lk(mtx_);
    for (auto& o : nodes_)
        for (auto& in : o->inputs)
            if (in == n) in.reset();
    nodes_.erase(std::remove(nodes_.begin(), nodes_.end(), n), nodes_.end());
}

NodePtr Graph::find(const std::string& name) const {
    std::lock_guard<std::recursive_mutex> lk(mtx_);
    for (auto& n : nodes_)
        if (n->name == name) return n;
    return nullptr;
}

NodePtr Graph::findId(uint64_t id) const {
    std::lock_guard<std::recursive_mutex> lk(mtx_);
    for (auto& n : nodes_)
        if (n->id() == id) return n;
    return nullptr;
}

std::vector<NodePtr> Graph::nodes() const {
    std::lock_guard<std::recursive_mutex> lk(mtx_);
    return nodes_;
}

bool Graph::wouldCycle(const NodePtr& dst, const NodePtr& src) const {
    if (!src) return false;
    std::set<const Node*> seen;
    std::vector<const Node*> stack{src.get()};
    while (!stack.empty()) {
        const Node* n = stack.back();
        stack.pop_back();
        if (n == dst.get()) return true;
        if (!seen.insert(n).second) continue;
        for (auto& in : n->inputs)
            if (in) stack.push_back(in.get());
    }
    return false;
}

void Graph::connect(const NodePtr& dst, int slot, const NodePtr& src) {
    std::lock_guard<std::recursive_mutex> lk(mtx_);
    if (!dst || slot < 0 || slot >= int(dst->inputs.size())) throw Error("bad input slot");
    if (wouldCycle(dst, src)) throw Error("connection would create a cycle");
    dst->inputs[slot] = src;
}

void Graph::clear() {
    std::lock_guard<std::recursive_mutex> lk(mtx_);
    for (auto& n : nodes_)
        for (auto& in : n->inputs) in.reset();
    nodes_.clear();
}

// ---------------------------------------------------------------- cache

ImagePtr Cache::find(uint64_t hash, const Box& roi, const ChannelList& chans) {
    std::lock_guard<std::mutex> lk(mtx_);
    auto range = index_.equal_range(hash);
    for (auto it = range.first; it != range.second; ++it) {
        const ImagePtr& img = it->second->img;
        if (!img->bbox().contains(roi)) continue;
        bool ok = true;
        for (auto& c : chans)
            if (!img->has(c)) { ok = false; break; }
        if (!ok) continue;
        lru_.splice(lru_.begin(), lru_, it->second);
        hits++;
        return img;
    }
    misses++;
    return nullptr;
}

bool Cache::contains(uint64_t hash) {
    std::lock_guard<std::mutex> lk(mtx_);
    return index_.count(hash) > 0;
}

void Cache::insert(uint64_t hash, ImagePtr img) {
    std::lock_guard<std::mutex> lk(mtx_);
    size_t b = img->bytes();
    if (b > limit_) return;
    lru_.push_front({hash, img});
    index_.emplace(hash, lru_.begin());
    used_ += b;
    evict();
}

void Cache::evict() {
    while (used_ > limit_ && !lru_.empty()) {
        auto it = std::prev(lru_.end());
        auto range = index_.equal_range(it->hash);
        for (auto r = range.first; r != range.second; ++r) {
            if (r->second == it) {
                index_.erase(r);
                break;
            }
        }
        used_ -= it->img->bytes();
        lru_.erase(it);
    }
}

void Cache::setLimit(size_t bytes) {
    std::lock_guard<std::mutex> lk(mtx_);
    limit_ = bytes;
    evict();
}

void Cache::clear() {
    std::lock_guard<std::mutex> lk(mtx_);
    lru_.clear();
    index_.clear();
    used_ = 0;
}

// ---------------------------------------------------------------- eval

Info Eval::inputInfo(int i, double f) const {
    if (i >= int(inputs.size()) || !inputs[i]) {
        Info inf;
        inf.bbox = Box{};
        return inf;
    }
    return engine.infoImpl(inputs[i], f, cancel, 1);
}

bool Eval::connected(int i) const { return i < int(inputs.size()) && inputs[i] != nullptr; }

ImagePtr Eval::input(int i, const Box& roi, const ChannelList& chans, double f) const {
    if (!connected(i)) return std::make_shared<Image>(Box{}, ChannelList{});
    return engine.renderImpl(inputs[i], f, roi, chans, cancel, 1);
}

Params Eval::paramsAt(double f) const {
    std::lock_guard<std::recursive_mutex> lk(engine.graph().mutex());
    return node->params(f);
}

void Eval::parallelRows(const Box& b, const std::function<void(int)>& fn) const {
    ThreadPool::global().parallelFor(
        b.y0, b.y1,
        [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y) fn(y);
        },
        &cancel);
}

// ---------------------------------------------------------------- engine

Engine::Engine(Graph& g, size_t cacheBytes) : graph_(g), cache_(cacheBytes) {}

uint64_t Engine::hashLocked(const NodePtr& n, double frame, int depth) {
    Hasher h;
    if (!n || depth > 512) return h.add(int64_t(-1)).value();
    if (n->disabled) {
        if (!n->inputs.empty() && n->inputs[0]) return hashLocked(n->inputs[0], frame, depth + 1);
        return h.add(std::string("disabled-empty")).value();
    }
    h.add(n->type()).add(n->desc().version).add(n->revision);
    Params p = n->params(frame);
    for (auto& kv : p.num) {
        if (kv.first == "__frame") continue;
        h.add(kv.first);
        for (double d : kv.second) h.add(d);
    }
    for (auto& kv : p.str) h.add(kv.first).add(kv.second);
    n->op().hashExtra(h, *n, frame);
    for (size_t i = 0; i < n->inputs.size(); ++i) {
        h.add(int(i));
        if (!n->inputs[i]) {
            h.add(int64_t(0));
            continue;
        }
        for (double f : n->op().inputFrames(int(i), frame, p)) h.add(hashLocked(n->inputs[i], f, depth + 1));
    }
    return h.value();
}

uint64_t Engine::hash(const NodePtr& n, double frame) {
    std::lock_guard<std::recursive_mutex> lk(graph_.mutex());
    return hashLocked(n, frame, 0);
}

bool Engine::isCached(const NodePtr& n, double frame) { return cache_.contains(hash(n, frame)); }

Info Engine::info(const NodePtr& n, double frame, const CancelToken* cancel) {
    CancelToken none;
    return infoImpl(n, frame, cancel ? *cancel : none, 0);
}

Info Engine::infoImpl(const NodePtr& n, double frame, const CancelToken& cancel, int depth) {
    if (depth > 512) throw Error("graph too deep");
    Params p;
    std::vector<NodePtr> inputs;
    uint64_t h;
    bool disabled;
    {
        std::lock_guard<std::recursive_mutex> lk(graph_.mutex());
        h = hashLocked(n, frame, 0);
        disabled = n->disabled;
        inputs = n->inputs;
        if (!disabled) p = n->params(frame);
    }
    {
        std::lock_guard<std::mutex> lk(infoMtx_);
        auto it = infoCache_.find(h);
        if (it != infoCache_.end()) return it->second;
    }
    Info inf;
    if (disabled) {
        if (!inputs.empty() && inputs[0]) inf = infoImpl(inputs[0], frame, cancel, depth + 1);
    } else {
        Eval ev{*this, cancel, frame, n, inputs};
        try {
            inf = n->op().info(ev, p);
        } catch (const Cancelled&) {
            throw;
        } catch (const NodeError&) {
            throw;
        } catch (const std::exception& e) {
            std::lock_guard<std::recursive_mutex> lk(graph_.mutex());
            n->error = e.what();
            throw NodeError(n->name + ": " + e.what());
        }
    }
    inf.channels = sortedChannels(inf.channels);
    std::lock_guard<std::mutex> lk(infoMtx_);
    if (infoCache_.size() > 20000) infoCache_.clear();
    infoCache_[h] = inf;
    return inf;
}

ImagePtr Engine::render(const NodePtr& n, double frame, const Box& roi, const ChannelList& chans,
                        const CancelToken* cancel) {
    CancelToken none;
    return renderImpl(n, frame, roi, chans, cancel ? *cancel : none, 0);
}

ImagePtr Engine::renderImpl(const NodePtr& n, double frame, const Box& roiIn, const ChannelList& chansIn,
                            const CancelToken& cancel, int depth) {
    cancel.check();
    if (depth > 512) throw Error("graph too deep");
    Params p;
    std::vector<NodePtr> inputs;
    uint64_t h;
    bool disabled;
    {
        std::lock_guard<std::recursive_mutex> lk(graph_.mutex());
        h = hashLocked(n, frame, 0);
        disabled = n->disabled;
        inputs = n->inputs;
        if (!disabled) p = n->params(frame);
    }
    if (disabled) {
        if (!inputs.empty() && inputs[0]) return renderImpl(inputs[0], frame, roiIn, chansIn, cancel, depth + 1);
        return std::make_shared<Image>(Box{}, ChannelList{});
    }
    Info inf = infoImpl(n, frame, cancel, depth);
    Box roi = roiIn.intersect(inf.bbox);
    ChannelList chans = sortedChannels(intersect(chansIn, inf.channels));
    if (roi.empty() || chans.empty()) return std::make_shared<Image>(Box{}, chans);

    if (auto hit = cache_.find(h, roi, chans)) return hit;

    Eval ev{*this, cancel, frame, n, inputs};
    auto t0 = std::chrono::steady_clock::now();
    ImagePtr out;
    try {
        out = n->op().render(ev, p, roi, chans);
    } catch (const Cancelled&) {
        throw;
    } catch (const NodeError&) {
        throw;
    } catch (const std::exception& e) {
        {
            std::lock_guard<std::recursive_mutex> lk(graph_.mutex());
            n->error = e.what();
        }
        throw NodeError(n->name + ": " + e.what());
    }
    auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    {
        std::lock_guard<std::recursive_mutex> lk(graph_.mutex());
        n->error.clear();
    }
    {
        std::lock_guard<std::mutex> lk(statsMtx_);
        auto& s = stats_[n->name];
        s.ms += ms;
        s.renders++;
    }
    cache_.insert(h, out);
    return out;
}

std::map<std::string, NodeStats> Engine::stats() {
    std::lock_guard<std::mutex> lk(statsMtx_);
    return stats_;
}

void Engine::resetStats() {
    std::lock_guard<std::mutex> lk(statsMtx_);
    stats_.clear();
}

}  // namespace vc
