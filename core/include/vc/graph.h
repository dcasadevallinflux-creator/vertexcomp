#pragma once
// Node graph and pull-based evaluation.
//
// Evaluation contract:
//   Engine::info(node, frame)  -> output format, bbox, channels, frame range.
//   Engine::render(node, frame, roi, chans) -> an image whose bbox contains
//     roi ∩ info.bbox and whose channels contain chans ∩ info.channels.
//     The returned image may be larger than asked (cache reuse); consumers
//     always address pixels in absolute coordinates.
//
// Hashing: a node's hash at a frame covers its type/version, every knob
// value evaluated at that frame, its disabled flag and the hashes of the
// inputs at the frames the op asks for. Changing a knob therefore only
// invalidates the node and everything downstream of it.
//
// Thread safety: knob values and connections are guarded by the graph
// mutex; renders take a snapshot of what they need under that lock, so the
// UI thread can edit while renders run on worker threads.

#include <cmath>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "vc/expr.h"
#include "vc/image.h"
#include "vc/util.h"

namespace vc {

enum class KnobType { Float, Int, Bool, Enum, String, File, Color, XY, Box, Channel, Channels, Button, Range, Text };

struct KnobDef {
    std::string name, label, tooltip, group;
    KnobType type = KnobType::Float;
    int comps = 1;
    std::vector<double> def;
    std::string sdef;
    double smin = 0.0, smax = 1.0;  // soft slider range
    std::vector<std::string> options;
    bool animatable = true;
};

enum Interp { InterpConstant = 0, InterpLinear = 1, InterpSmooth = 2, InterpBezier = 3 };

struct Key {
    double frame = 0, value = 0;
    int interp = InterpSmooth;          // interpolation leaving this key
    double inSlope = 0, outSlope = 0;  // value per frame, used by InterpBezier
};

struct Curve {
    std::vector<Key> keys;  // sorted by frame
    bool animated() const { return !keys.empty(); }
    double eval(double frame) const;
    void set(const Key& k);
    void remove(double frame);
};

struct KnobValue {
    std::vector<double> v;
    std::vector<Curve> curves;      // per component, empty when static
    std::vector<std::string> exprs; // per component, empty string when none
    std::string s;
};

class Graph;
class Engine;
class Node;
using NodePtr = std::shared_ptr<Node>;

struct Info {
    Format format;
    Box bbox;
    ChannelList channels;
    double first = 1, last = 1;
    std::map<std::string, std::string> meta;
};

// Read-only view of a node's knobs at one frame; built under the graph lock.
class Params {
public:
    double f(const std::string& k, int comp = 0) const;
    int i(const std::string& k) const { return int(std::lround(f(k))); }
    bool b(const std::string& k) const { return f(k) != 0.0; }
    const std::string& s(const std::string& k) const;
    std::vector<double> v(const std::string& k) const;

    std::map<std::string, std::vector<double>> num;
    std::map<std::string, std::string> str;
};

struct Eval;

class Op {
public:
    virtual ~Op() = default;
    // Frames requested from input i to produce `frame`.
    virtual std::vector<double> inputFrames(int input, double frame, const Params& p) const {
        (void)input; (void)p;
        return {frame};
    }
    virtual Info info(Eval& ev, const Params& p) const = 0;
    virtual ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const = 0;
    // Extra data that participates in the hash (e.g. roto shapes, file mtime).
    virtual void hashExtra(Hasher& h, const Node& n, double frame) const { (void)h; (void)n; (void)frame; }
};

struct OpDesc {
    std::string type, category, help;
    std::vector<std::string> inputs;  // labels
    int required = 0;                 // inputs that must be connected
    std::vector<KnobDef> knobs;
    int version = 1;
    std::function<std::unique_ptr<Op>()> make;
};

class Registry {
public:
    static Registry& get();
    void add(OpDesc d);
    const OpDesc* find(const std::string& type) const;
    std::vector<const OpDesc*> all() const;

private:
    std::map<std::string, OpDesc> ops_;
};

struct Registrar {
    explicit Registrar(OpDesc d) { Registry::get().add(std::move(d)); }
};

#define VC_CAT2(a, b) a##b
#define VC_CAT(a, b) VC_CAT2(a, b)
#define VC_REGISTER_OP(...) static ::vc::Registrar VC_CAT(vc_registrar_, __LINE__)(__VA_ARGS__)

struct RotoData;

// enable_shared_from_this lets the Python bindings hand back the original
// shared_ptr instead of one whose deleter must take the GIL.
class Node : public std::enable_shared_from_this<Node> {
public:
    Node(Graph* g, const OpDesc* desc, uint64_t id);

    uint64_t id() const { return id_; }
    const OpDesc& desc() const { return *desc_; }
    const std::string& type() const { return desc_->type; }
    const Op& op() const { return *op_; }
    Graph* graph() const { return graph_; }

    std::string name;
    bool disabled = false;
    std::string error;  // last evaluation error, empty when OK

    std::vector<NodePtr> inputs;  // size == desc().inputs.size(); null when unconnected
    std::map<std::string, KnobValue> knobs;
    std::shared_ptr<RotoData> roto;  // Roto node only
    int revision = 0;                // bumped by "reload"-style actions

    const KnobDef* knobDef(const std::string& k) const;
    // Value of one component at a frame (animation and expressions applied).
    // Caller must hold the graph lock.
    double valueAt(const std::string& k, int comp, double frame, int depth = 0) const;
    Params params(double frame) const;

private:
    Graph* graph_;
    const OpDesc* desc_;
    uint64_t id_;
    std::unique_ptr<Op> op_;
};

class Graph {
public:
    NodePtr create(const std::string& type, const std::string& name = "");
    void remove(const NodePtr& n);
    NodePtr find(const std::string& name) const;
    NodePtr findId(uint64_t id) const;
    std::vector<NodePtr> nodes() const;
    // Throws vc::Error on a cycle or bad slot.
    void connect(const NodePtr& dst, int slot, const NodePtr& src);
    bool wouldCycle(const NodePtr& dst, const NodePtr& src) const;
    std::string uniqueName(const std::string& base) const;
    void clear();

    std::recursive_mutex& mutex() const { return mtx_; }

private:
    std::vector<NodePtr> nodes_;
    uint64_t nextId_ = 1;
    mutable std::recursive_mutex mtx_;
};

// LRU cache of rendered images keyed by node hash. Several entries can exist
// per hash (different ROIs / channel sets).
class Cache {
public:
    explicit Cache(size_t limitBytes) : limit_(limitBytes) {}
    ImagePtr find(uint64_t hash, const Box& roi, const ChannelList& chans);
    void insert(uint64_t hash, ImagePtr img);
    bool contains(uint64_t hash);
    void setLimit(size_t bytes);
    size_t limit() const { return limit_; }
    size_t used() const { return used_; }
    void clear();
    uint64_t hits = 0, misses = 0;

private:
    struct Entry {
        uint64_t hash;
        ImagePtr img;
    };
    void evict();
    std::list<Entry> lru_;  // front = most recent
    std::unordered_multimap<uint64_t, std::list<Entry>::iterator> index_;
    size_t limit_, used_ = 0;
    std::mutex mtx_;
};

// Per-request evaluation context. Not shared between threads.
struct Eval {
    Engine& engine;
    const CancelToken& cancel;
    double frame;
    NodePtr node;  // node being evaluated in this frame of the stack

    Info inputInfo(int i, double frame) const;
    Info inputInfo(int i) const { return inputInfo(i, frame); }
    bool connected(int i) const;
    ImagePtr input(int i, const Box& roi, const ChannelList& chans, double frame) const;
    ImagePtr input(int i, const Box& roi, const ChannelList& chans) const { return input(i, roi, chans, frame); }
    void parallelRows(const Box& b, const std::function<void(int y)>& fn) const;
    void check() const { cancel.check(); }
    // Knob values of the node being evaluated at another frame (motion blur).
    Params paramsAt(double f) const;

    std::vector<NodePtr> inputs;  // snapshot taken under the graph lock
};

struct NodeStats {
    double ms = 0;
    uint64_t renders = 0;
};

class Engine {
public:
    explicit Engine(Graph& g, size_t cacheBytes = size_t(4) << 30);

    Info info(const NodePtr& n, double frame, const CancelToken* cancel = nullptr);
    ImagePtr render(const NodePtr& n, double frame, const Box& roi, const ChannelList& chans,
                    const CancelToken* cancel = nullptr);
    uint64_t hash(const NodePtr& n, double frame);
    bool isCached(const NodePtr& n, double frame);

    Cache& cache() { return cache_; }
    Graph& graph() { return graph_; }
    std::map<std::string, NodeStats> stats();
    void resetStats();

private:
    friend struct Eval;
    uint64_t hashLocked(const NodePtr& n, double frame, int depth);
    Info infoImpl(const NodePtr& n, double frame, const CancelToken& cancel, int depth);
    ImagePtr renderImpl(const NodePtr& n, double frame, const Box& roi, const ChannelList& chans,
                        const CancelToken& cancel, int depth);

    Graph& graph_;
    Cache cache_;
    std::mutex infoMtx_;
    std::unordered_map<uint64_t, Info> infoCache_;
    std::mutex statsMtx_;
    std::map<std::string, NodeStats> stats_;
};

}  // namespace vc
