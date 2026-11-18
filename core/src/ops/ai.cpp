// AI image nodes (SDXLGeneration, SDXLLightning, LCMTurbo, SDXLTurbo) and Versions.
//
// The pixel engine never runs a diffusion model. Generation happens in the
// Python app (a separate worker process with torch/diffusers); the result is
// registered in the image blob store and the node only references it through
// the hidden `image` knob. Evaluating these nodes is therefore as cheap as a
// Constant: the cached generation is shown until the user presses Generate
// again. Versions works the same way with snapshots of its input.
#include "blobs.h"
#include "common.h"

#include <atomic>
#include <mutex>
#include <unordered_map>

namespace vc::ops {

// ------------------------------------------------------------ blob store
namespace {
struct BlobEntry {
    ImagePtr img;
    uint64_t serial = 0;
};
std::mutex gBlobMtx;
std::unordered_map<std::string, BlobEntry> gBlobs;
std::atomic<uint64_t> gBlobSerial{0};
}  // namespace

void setImageBlob(const std::string& id, ImagePtr img) {
    std::lock_guard<std::mutex> l(gBlobMtx);
    gBlobs[id] = BlobEntry{std::move(img), ++gBlobSerial};
}

ImagePtr imageBlob(const std::string& id) {
    if (id.empty()) return nullptr;
    std::lock_guard<std::mutex> l(gBlobMtx);
    auto it = gBlobs.find(id);
    return it == gBlobs.end() ? nullptr : it->second.img;
}

uint64_t imageBlobSerial(const std::string& id) {
    if (id.empty()) return 0;
    std::lock_guard<std::mutex> l(gBlobMtx);
    auto it = gBlobs.find(id);
    return it == gBlobs.end() ? 0 : it->second.serial;
}

bool removeImageBlob(const std::string& id) {
    std::lock_guard<std::mutex> l(gBlobMtx);
    return gBlobs.erase(id) > 0;
}

void clearImageBlobs() {
    std::lock_guard<std::mutex> l(gBlobMtx);
    gBlobs.clear();
}

std::vector<std::string> imageBlobIds() {
    std::lock_guard<std::mutex> l(gBlobMtx);
    std::vector<std::string> out;
    out.reserve(gBlobs.size());
    for (auto& [k, v] : gBlobs) out.push_back(k);
    return out;
}

size_t imageBlobBytes() {
    std::lock_guard<std::mutex> l(gBlobMtx);
    size_t n = 0;
    for (auto& [k, v] : gBlobs)
        if (v.img) n += v.img->bytes();
    return n;
}

namespace {

const ChannelList kRGBA{"rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"};

// "all frames" / "frame range" / "generated frame only"
bool frameVisible(const Params& p, double frame) {
    const std::string& show = p.s("show");
    const double f = std::floor(frame + 1e-6);
    if (show == "frame range") {
        double a = p.f("rangeFirst"), b = p.f("rangeLast");
        return f >= std::min(a, b) && f <= std::max(a, b);
    }
    if (show == "generated frame only") return std::abs(f - std::floor(p.f("genFrame") + 1e-6)) < 0.5;
    return true;
}

std::vector<KnobDef> outputKnobs(const std::string& frameLabel) {
    auto first = K::integer("rangeFirst", "range first", 1, 0, 1000, "First frame of 'frame range'");
    auto last = K::integer("rangeLast", "range last", 100, 0, 1000, "Last frame of 'frame range'");
    auto gen = K::integer("genFrame", frameLabel, 1, 0, 1000,
                          "Frame the image belongs to ('generated frame only' shows it there)");
    first.animatable = last.animatable = gen.animatable = false;
    return {K::group(K::enm("show", "show", {"all frames", "frame range", "generated frame only"}, "all frames",
                            "Frames on which the cached image is output; elsewhere the node is empty"),
                     "Output"),
            K::group(first, "Output"), K::group(last, "Output"), K::group(gen, "Output")};
}

// Shared evaluation for every node whose output is a blob: the stored image
// on visible frames, the input (when connected) while nothing is stored,
// and an empty image otherwise.
class BlobOp : public Op {
public:
    explicit BlobOp(bool passInput) : passInput_(passInput) {}

    void hashExtra(Hasher& h, const Node& n, double frame) const override {
        auto it = n.knobs.find("image");
        h.add(it == n.knobs.end() ? uint64_t(0) : imageBlobSerial(it->second.s));
        // Knob hashes do not depend on the frame; visibility does.
        h.add(frameVisible(n.params(frame), frame) ? 1 : 0);
    }

    Info info(Eval& ev, const Params& p) const override {
        ImagePtr img = active(p);
        if (!img) {
            if (passInput_ && ev.connected(0)) return ev.inputInfo(0);
            Info i;
            if (p.num.count("size")) {
                i.format.w = std::max(1, int(p.f("size", 0)));
                i.format.h = std::max(1, int(p.f("size", 1)));
            } else if (ev.connected(0)) {
                i.format = ev.inputInfo(0).format;
            }
            i.bbox = Box{};
            i.channels = kRGBA;
            return i;
        }
        Info i;
        if (ev.connected(0) && keepInputFormat()) i = ev.inputInfo(0);
        else {
            i.format.w = std::max(1, img->bbox().x1);
            i.format.h = std::max(1, img->bbox().y1);
        }
        i.meta.clear();
        i.channels = sortedChannels(img->channels());
        i.bbox = frameVisible(p, ev.frame) ? img->bbox() : Box{};
        i.meta["vc/blob"] = p.s("image");
        return i;
    }

    ImagePtr render(Eval& ev, const Params& p, const Box& roi, const ChannelList& chans) const override {
        ImagePtr img = active(p);
        if (!img) {
            if (passInput_ && ev.connected(0)) return ev.input(0, roi, chans);
            return std::make_shared<Image>(roi, chans);
        }
        if (!frameVisible(p, ev.frame)) return std::make_shared<Image>(roi, chans);
        return copyOf(*img, roi, chans);
    }

protected:
    virtual ImagePtr active(const Params& p) const { return imageBlob(p.s("image")); }
    virtual bool keepInputFormat() const { return false; }
    bool passInput_;
};

// ------------------------------------------------------------ SDXL family
struct GenDefaults {
    std::vector<std::string> models;
    std::vector<std::string> accel;
    std::string accelDefault;
    int w, h, steps;
    double cfg;
    std::string sampler;
    double denoise;
    std::string accelTip;
};

const std::vector<std::string> kSamplers{"euler",     "euler_a", "dpm++_2m", "dpm++_2m_karras", "dpm++_sde_karras",
                                         "ddim",      "lcm",     "tcd",      "unipc"};

std::vector<KnobDef> genKnobs(const GenDefaults& d) {
    auto seed = K::integer("seed", "seed", 0, 0, 99999, "Noise seed (0 - 2147483647)");
    auto steps = K::integer("steps", "steps", d.steps, 1, 50, "Denoising steps");
    auto cfg = K::flt("cfg", "guidance (CFG)", d.cfg, 0, 12, "Classifier-free guidance; <= 1 disables it (fastest)");
    auto denoise = K::flt("denoise", "denoise", d.denoise, 0, 1,
                          "img2img strength when the init input is connected (1 = ignore the init image)");
    auto size = K::wh("size", "size", d.w, d.h, 256, 2048, "Output resolution (rounded to multiples of 8)");
    auto loraScale = K::flt("loraScale", "LoRA strength", 1.0, 0, 1.5, "Weight of the speed LoRA");
    auto varSeed = K::integer("varSeed", "variation seed", 1, 0, 99999, "Second noise mixed in by 'variation'");
    auto varStrength = K::flt("varStrength", "variation", 0, 0, 1,
                              "Blend (spherical) between the seed noise and the variation seed noise");
    for (KnobDef* k : {&seed, &steps, &cfg, &denoise, &size, &loraScale, &varSeed, &varStrength}) k->animatable = false;
    std::vector<KnobDef> ks{
        K::text("prompt", "prompt", "", "What to generate"),
        K::text("negative", "negative", "", "What to avoid (used when guidance > 1)"),
        K::button("generate", "Generate", "Run the model and cache the result; the cached image is kept until the next Generate"),
        K::group(K::enm("model", "model", d.models, d.models.front(), "Base checkpoint"), "Model"),
        K::group(K::enm("accel", "speed", d.accel, d.accelDefault, d.accelTip), "Model"),
        K::group(loraScale, "Model"),
        K::group(K::file("checkpoint", "checkpoint", "SDXL .safetensors file used when model = custom checkpoint"),
                 "Model"),
        K::group(seed, "Sampling"),
        K::group(K::boolean("randomSeed", "random seed", false, "Pick a new random seed on every Generate"),
                 "Sampling"),
        K::group(steps, "Sampling"),
        K::group(cfg, "Sampling"),
        K::group(K::enm("sampler", "sampler", kSamplers, d.sampler), "Sampling"),
        K::group(K::enm("spacing", "timesteps", {"auto", "trailing", "leading", "linspace"}, "auto",
                        "Timestep spacing; auto uses what the speed LoRA/checkpoint was trained with"),
                 "Sampling"),
        K::group(size, "Sampling"),
        K::group(denoise, "Sampling"),
        K::group(varSeed, "Noise"),
        K::group(varStrength, "Noise"),
    };
    ks = concat(ks, outputKnobs("generated frame"));
    ks.push_back(K::group(K::button("clearImage", "Clear cached image"), "Output"));
    ks.push_back(K::str("image", "image", "", "Cached generation (internal id)"));
    ks.push_back(K::str("genInfo", "genInfo", "", "Parameters of the cached generation (internal)"));
    return ks;
}

class Generate : public BlobOp {
public:
    Generate() : BlobOp(false) {}
};

const std::vector<std::string> kBaseModels{"SDXL base 1.0", "custom checkpoint"};

VC_REGISTER_OP(OpDesc{
    "SDXLGeneration", "AI",
    "SDXL text-to-image / image-to-image with an optional step-distilled speed LoRA (Lightning, Hyper-SD, DMD2, LCM). "
    "Press Generate; the image is cached in the project until the next Generate.",
    {"init"}, 0,
    genKnobs(GenDefaults{kBaseModels,
                         {"none (full steps)", "Lightning 2-step", "Lightning 4-step", "Lightning 8-step",
                          "Hyper-SD 8-step", "DMD2 4-step", "LCM-LoRA"},
                         "Lightning 8-step", 1024, 1024, 8, 1.0, "euler", 0.6,
                         "Step-distilled LoRA: few steps, no CFG. 'none' needs ~25-30 steps and guidance 5-7"}),
    1, maker<Generate>()});

VC_REGISTER_OP(OpDesc{
    "SDXLLightning", "AI",
    "SDXL-Lightning (ByteDance): 2/4/8-step distilled LoRA on SDXL, Euler with trailing timesteps, no CFG.",
    {"init"}, 0,
    genKnobs(GenDefaults{kBaseModels, {"2-step", "4-step", "8-step"}, "4-step", 1024, 1024, 4, 1.0, "euler", 0.6,
                         "Lightning LoRA; set steps to match"}),
    1, maker<Generate>()});

VC_REGISTER_OP(OpDesc{
    "LCMTurbo", "AI",
    "Latent Consistency (LCM-LoRA SDXL) with the LCM scheduler: 4-8 steps, guidance 1-2.",
    {"init"}, 0,
    genKnobs(GenDefaults{kBaseModels, {"LCM-LoRA", "DMD2 4-step"}, "LCM-LoRA", 1024, 1024, 6, 1.5, "lcm", 0.6,
                         "Consistency-distilled LoRA used with the LCM scheduler"}),
    1, maker<Generate>()});

VC_REGISTER_OP(OpDesc{
    "SDXLTurbo", "AI",
    "SDXL-Turbo (adversarial distillation): 1-4 steps at 512x512, no CFG. Full checkpoint, no LoRA.",
    {"init"}, 0,
    genKnobs(GenDefaults{{"SDXL-Turbo", "custom checkpoint"}, {"distilled checkpoint"}, "distilled checkpoint", 512,
                         512, 2, 0.0, "euler_a", 0.5, "SDXL-Turbo is itself step-distilled"}),
    1, maker<Generate>()});

// ------------------------------------------------------------ Versions
class Versions : public BlobOp {
public:
    Versions() : BlobOp(true) {}

protected:
    ImagePtr active(const Params& p) const override {
        if (p.i("current") <= 0) return nullptr;
        return imageBlob(p.s("image"));
    }
    bool keepInputFormat() const override { return true; }
};

std::vector<KnobDef> versionKnobs() {
    auto current = K::integer("current", "version", 0, 0, 20, "Version shown; 0 = live input");
    current.animatable = false;
    std::vector<KnobDef> ks{K::button("addVersion", "Add new version",
                                      "Store the input at the current frame as a new version (kept inside the project)"),
                            current};
    ks = concat(ks, outputKnobs("stored frame"));
    ks.push_back(K::str("image", "image", "", "Image id of the current version (internal)"));
    ks.push_back(K::str("versions", "versions", "", "Version list (internal JSON)"));
    return ks;
}

VC_REGISTER_OP(OpDesc{"Versions", "AI",
                      "Stores snapshots of its input (or of a generation) inside the project. Add new version saves "
                      "the current frame; pick a version to output it from the cache, version 0 passes the input "
                      "through.",
                      {""}, 0, versionKnobs(), 1, maker<Versions>()});

}  // namespace
}  // namespace vc::ops
