// Read and Write. File decoding lives in io.cpp; colour conversion to and
// from the working space happens here.
#include <list>
#include <mutex>

#include "common.h"
#include "io_ops.h"
#include "vc/color.h"
#include "vc/io.h"

namespace vc::ops {

namespace {

const std::vector<std::string> kBeforeAfter{"hold", "loop", "bounce", "black", "error"};

std::vector<std::string> csWithAuto() {
    auto v = colorSpaceNames();
    v.insert(v.begin(), "auto");
    return v;
}

// Small cache of decoded and converted files, so info() and render() decode once.
struct FileCache {
    struct Entry {
        std::string key;
        std::shared_ptr<const FileImage> img;
    };
    std::list<Entry> lru;
    std::mutex mtx;
    size_t capacity = 6;

    std::shared_ptr<const FileImage> get(const std::string& key) {
        std::lock_guard<std::mutex> lk(mtx);
        for (auto it = lru.begin(); it != lru.end(); ++it)
            if (it->key == key) {
                lru.splice(lru.begin(), lru, it);
                return it->img;
            }
        return nullptr;
    }
    void put(const std::string& key, std::shared_ptr<const FileImage> img) {
        std::lock_guard<std::mutex> lk(mtx);
        lru.push_front({key, img});
        while (lru.size() > capacity) lru.pop_back();
    }
    static FileCache& get() {
        static FileCache c;
        return c;
    }
};

std::string lowerExt(const std::string& path) {
    auto p = path.find_last_of('.');
    std::string e = p == std::string::npos ? "" : path.substr(p);
    for (auto& c : e) c = char(std::tolower(static_cast<unsigned char>(c)));
    return e;
}

// Maps an output frame to a file frame; returns false for "black".
bool mapFrame(const Params& p, double frame, int& fileFrame) {
    int first = p.i("first"), last = p.i("last");
    int f = int(std::floor(frame + 0.5)) - p.i("offset");
    int len = std::max(1, last - first + 1);
    auto apply = [&](const std::string& mode, int edge) -> bool {
        if (mode == "hold") { fileFrame = edge; return true; }
        if (mode == "black") return false;
        if (mode == "error") throw Error("frame " + std::to_string(f) + " is outside the range " + std::to_string(first) + "-" + std::to_string(last));
        int k = f - first;
        if (mode == "loop") {
            fileFrame = first + ((k % len) + len) % len;
            return true;
        }
        int period = std::max(1, 2 * len - 2);
        int m = ((k % period) + period) % period;
        fileFrame = first + (m < len ? m : period - m);
        return true;
    };
    if (f < first) return apply(p.s("before"), first);
    if (f > last) return apply(p.s("after"), last);
    fileFrame = f;
    return true;
}

std::shared_ptr<const FileImage> loadConverted(const Params& p, const std::string& path, int fileFrame) {
    bool movie = isMovie(path);
    std::string key = path + "|" + std::to_string(movie ? fileFrame : 0) + "|" + std::to_string(fileMTime(path)) + "|" +
                      p.s("colorspace") + "|" + p.s("alpha") + "|" + std::to_string(p.b("raw")) + "|" + workingSpace();
    if (auto hit = FileCache::get().get(key)) return hit;

    auto fi = std::make_shared<FileImage>();
    if (movie) {
        MovieInfo mi;
        if (!movieInfo(path, mi)) throw Error("cannot open movie '" + path + "'");
        fi->image = movieFrame(path, fileFrame);
        if (!fi->image) throw Error("cannot decode frame " + std::to_string(fileFrame) + " of '" + path + "'");
        fi->format.w = mi.width;
        fi->format.h = mi.height;
        fi->hasAlpha = fi->image->has("rgba.alpha");
    } else {
        *fi = readImage(path);
    }
    Image& img = *fi->image;
    std::string cs = p.s("colorspace");
    if (cs == "auto") cs = fi->floatData ? workingSpace() : (movie ? "Rec.709" : "sRGB");
    std::string am = p.s("alpha");
    bool straight = am == "straight" || (am == "auto" && !fi->floatData);
    bool raw = p.b("raw");
    float* r = img.plane("rgba.red");
    float* g = img.plane("rgba.green");
    float* b = img.plane("rgba.blue");
    float* a = img.plane("rgba.alpha");
    size_t n = size_t(img.width()) * img.height();
    if (!raw && r && g && b) {
        ColorTransform xf(cs, workingSpace());
        if (!xf.identity()) {
            ThreadPool::global().parallelFor(0, int(n), [&](int i0, int i1) {
                for (int i = i0; i < i1; ++i) {
                    float al = a ? a[i] : 1.0f;
                    bool un = a && !straight && al > 1e-6f;
                    if (un) { r[i] /= al; g[i] /= al; b[i] /= al; }
                    xf.apply(r[i], g[i], b[i]);
                    if (un) { r[i] *= al; g[i] *= al; b[i] *= al; }
                }
            });
        }
    }
    if (!raw && a && straight && r && g && b)
        for (size_t i = 0; i < n; ++i) { r[i] *= a[i]; g[i] *= a[i]; b[i] *= a[i]; }
    img.meta["input/filename"] = path;
    img.meta["input/colorspace"] = raw ? "raw" : cs;
    FileCache::get().put(key, fi);
    return fi;
}

class Read : public Op {
public:
    void hashExtra(Hasher& h, const Node& n, double frame) const override {
        Params p = n.params(frame);
        int ff;
        try {
            if (!mapFrame(p, frame, ff)) {
                h.add(std::string("black"));
                return;
            }
        } catch (const Error&) {
            h.add(std::string("error"));
            return;
        }
        std::string path = isPattern(p.s("file")) ? framePath(p.s("file"), ff) : p.s("file");
        h.add(path).add(fileMTime(path)).add(ff).add(workingSpace());
    }

    // Path of the file actually used for `frame`, or "" for black.
    std::string resolve(const Params& p, double frame, int& ff) const {
        const std::string& file = p.s("file");
        if (file.empty()) throw Error("no file");
        if (!mapFrame(p, frame, ff)) return "";
        if (!isPattern(file)) return file;
        std::string path = framePath(file, ff);
        if (fileExists(path)) return path;
        std::string missing = p.s("missing");
        if (missing == "black") return "";
        if (missing == "nearest") {
            for (int d = 1; d < 10000; ++d) {
                for (int s : {-1, 1}) {
                    int f2 = ff + s * d;
                    if (f2 < p.i("first") || f2 > p.i("last")) continue;
                    std::string p2 = framePath(file, f2);
                    if (fileExists(p2)) { ff = f2; return p2; }
                }
                if (ff - d < p.i("first") && ff + d > p.i("last")) break;
            }
        }
        throw Error("missing frame " + std::to_string(ff) + ": " + path);
    }

    Info info(Eval& ev, const Params& p) const override {
        int ff = 0;
        std::string path = resolve(p, ev.frame, ff);
        Info i;
        i.first = p.f("first") + p.f("offset");
        i.last = p.f("last") + p.f("offset");
        if (path.empty()) {
            // Black frame: keep the format of the first frame we can find.
            Params q = p;
            std::string probe = isPattern(p.s("file")) ? framePath(p.s("file"), p.i("first")) : p.s("file");
            if (fileExists(probe) || isMovie(probe)) {
                try {
                    auto fi = loadConverted(q, probe, p.i("first"));
                    i.format = fi->format;
                    i.channels = fi->image->channels();
                } catch (const Error&) {
                }
            }
            if (i.channels.empty()) i.channels = {"rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"};
            i.bbox = Box{};
            return i;
        }
        auto fi = loadConverted(p, path, ff);
        i.format = fi->format;
        i.bbox = fi->image->bbox();
        i.channels = fi->image->channels();
        i.meta = fi->image->meta;
        return i;
    }

    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        int ff = 0;
        std::string path = resolve(p, ev.frame, ff);
        if (path.empty()) return std::make_shared<Image>(roi, chans);
        auto fi = loadConverted(p, path, ff);
        // Share the decoded image: the engine only needs a superset.
        return std::shared_ptr<const Image>(fi, fi->image.get());
    }
};

}  // namespace

VC_REGISTER_OP(OpDesc{"Read", "Image",
                      "Read an image, image sequence (name.####.exr, %04d, $F4) or movie. EXR layers and parts "
                      "become channels.",
                      {}, 0,
                      {K::file("file", "file"), K::integer("first", "first", 1, 0, 2000),
                       K::integer("last", "last", 1, 0, 2000),
                       K::integer("offset", "frame offset", 0, -1000, 1000, "Output frame = file frame + offset"),
                       K::enm("before", "before", kBeforeAfter, "hold"), K::enm("after", "after", kBeforeAfter, "hold"),
                       K::enm("missing", "missing frames", {"error", "black", "nearest"}),
                       K::enm("colorspace", "colorspace", csWithAuto(), "auto"),
                       K::enm("alpha", "alpha", {"auto", "premultiplied", "straight"}, "auto",
                              "auto: EXR premultiplied, integer formats straight"),
                       K::boolean("raw", "raw data", false, "No colour or alpha conversion"),
                       K::button("reload", "reload")},
                      1, maker<Read>()});

// ------------------------------------------------------------ Write

namespace {
class Write : public Op {
public:
    Info info(Eval& ev, const Params&) const override { return ev.connected(0) ? ev.inputInfo(0) : emptyInfo(); }
    ImagePtr render(Eval& ev, const Params&, const Box& roi, const ChannelList& chans) const override {
        return ev.input(0, roi, chans);
    }
};
}  // namespace

VC_REGISTER_OP(OpDesc{"Write", "Image", "Render the input to an image sequence or a movie.", {""}, 1,
                      {K::file("file", "file", "name.####.exr or name.mov"),
                       K::enm("fileType", "file type", {"exr", "png", "jpg", "tga", "bmp", "hdr", "mov", "mp4"}),
                       K::enm("dataType", "data type", {"16 bit half", "32 bit float", "8 bit", "16 bit"}),
                       K::enm("compression", "compression", {"zip", "zips", "piz", "none"}),
                       K::integer("quality", "quality", 90, 1, 100, "JPEG / movie quality"),
                       K::enm("codec", "codec", {"h264", "prores", "mjpeg"}),
                       K::channels("channels", "channels", "rgba", "rgba, rgb, all or layer names"),
                       K::enm("colorspace", "colorspace", csWithAuto(), "auto"),
                       K::boolean("premultiplied", "premultiplied", false,
                                  "Integer formats: write premultiplied colour (off: unpremultiply)"),
                       K::integer("first", "first", 1, 0, 2000), K::integer("last", "last", 1, 0, 2000),
                       K::boolean("skipExisting", "skip existing frames", false),
                       K::button("render", "Render")},
                      1, maker<Write>()});

MutImagePtr prepareWriteImage(Engine& engine, const NodePtr& node, double frame, ChannelList& chansOut,
                              Format& fmtOut, const CancelToken* cancel) {
    Params p;
    {
        std::lock_guard<std::recursive_mutex> lk(engine.graph().mutex());
        p = node->params(frame);
    }
    Info inf = engine.info(node, frame, cancel);
    std::string type = p.s("fileType");
    bool exr = type == "exr";
    ChannelList chans = resolveChannels(p.s("channels"), inf.channels);
    if (!exr) chans = intersect(ChannelList{"rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"}, inf.channels);
    if (chans.empty()) throw Error(node->name + ": nothing to write (no matching channels)");
    Box box = exr ? inf.bbox : inf.format.box();
    if (box.empty()) box = Box{0, 0, 1, 1};
    ImagePtr src = engine.render(node, frame, box, chans, cancel);
    auto img = copyOf(*src, box, chans);
    img->meta = src->meta;

    std::string cs = p.s("colorspace");
    if (cs == "auto") cs = exr || type == "hdr" ? workingSpace() : (type == "mov" || type == "mp4") ? "Rec.709" : "sRGB";
    ColorTransform xf(workingSpace(), cs);
    bool unpremult = !exr && !p.b("premultiplied");
    float* r = img->plane("rgba.red");
    float* g = img->plane("rgba.green");
    float* b = img->plane("rgba.blue");
    float* a = img->plane("rgba.alpha");
    size_t n = size_t(box.w()) * box.h();
    if (r && g && b) {
        ThreadPool::global().parallelFor(0, int(n), [&](int i0, int i1) {
            for (int i = i0; i < i1; ++i) {
                float al = a ? a[i] : 1.0f;
                bool un = a && al > 1e-6f && (unpremult || !xf.identity());
                if (un) { r[i] /= al; g[i] /= al; b[i] /= al; }
                xf.apply(r[i], g[i], b[i]);
                if (un && !unpremult) { r[i] *= al; g[i] *= al; b[i] *= al; }
            }
        });
    }
    chansOut = chans;
    fmtOut = inf.format;
    return img;
}

bool writeFrame(Engine& engine, const NodePtr& node, double frame, const CancelToken* cancel) {
    Params p;
    {
        std::lock_guard<std::recursive_mutex> lk(engine.graph().mutex());
        p = node->params(frame);
    }
    std::string file = p.s("file");
    if (file.empty()) throw Error(node->name + ": no output file");
    std::string type = p.s("fileType");
    if (type == "mov" || type == "mp4") throw Error("movies are written by the host encoder");
    std::string path = framePath(file, int(std::floor(frame + 0.5)));
    if (p.b("skipExisting") && fileExists(path)) return false;
    ChannelList chans;
    Format fmt;
    auto img = prepareWriteImage(engine, node, frame, chans, fmt, cancel);
    WriteOptions opt;
    opt.fileType = type;
    opt.dataType = p.s("dataType");
    if (type == "exr" && opt.dataType.find("bit half") == std::string::npos &&
        opt.dataType.find("float") == std::string::npos)
        opt.dataType = "16 bit half";
    opt.compression = p.s("compression");
    opt.quality = p.i("quality");
    writeImage(path, *img, fmt, chans, opt);
    return true;
}

}  // namespace vc::ops
