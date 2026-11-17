#include "vc/io.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <regex>
#include <set>

#include "vc/util.h"

extern "C" {
#include "../third_party/stb_image.h"
#include "../third_party/stb_image_write.h"
}

#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1
#define TINYEXR_USE_OPENMP 0
#define TINYEXR_USE_THREAD 1
#define TINYEXR_IMPLEMENTATION
#include "../third_party/tinyexr.h"

namespace fs = std::filesystem;

namespace vc {

namespace {

std::string lower(std::string s) {
    for (auto& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string extOf(const std::string& path) { return lower(fs::u8path(path).extension().u8string()); }

struct ExrFree {
    std::vector<EXRHeader*> headers;
    std::vector<EXRImage> images;
    EXRHeader** multi = nullptr;
    EXRHeader single;
    bool singleInit = false;
    ~ExrFree() {
        for (auto& im : images) FreeEXRImage(&im);
        if (multi) {
            for (auto* h : headers) {
                FreeEXRHeader(h);
                free(h);
            }
            free(multi);
        } else if (singleInit) {
            FreeEXRHeader(&single);
        }
    }
};

[[noreturn]] void exrFail(const std::string& path, const char* err) {
    std::string msg = "cannot read EXR '" + path + "'";
    if (err) {
        msg += ": ";
        msg += err;
        FreeEXRErrorMessage(err);
    }
    throw Error(msg);
}

FileImage readExr(const std::string& path) {
    EXRVersion ver;
    if (ParseEXRVersionFromFile(&ver, path.c_str()) != TINYEXR_SUCCESS) exrFail(path, nullptr);
    const char* err = nullptr;
    ExrFree hold;
    if (ver.multipart) {
        int n = 0;
        if (ParseEXRMultipartHeaderFromFile(&hold.multi, &n, &ver, path.c_str(), &err) != TINYEXR_SUCCESS)
            exrFail(path, err);
        hold.headers.assign(hold.multi, hold.multi + n);
    } else {
        InitEXRHeader(&hold.single);
        if (ParseEXRHeaderFromFile(&hold.single, &ver, path.c_str(), &err) != TINYEXR_SUCCESS) exrFail(path, err);
        hold.singleInit = true;
        hold.headers.push_back(&hold.single);
    }
    for (auto* h : hold.headers)
        for (int c = 0; c < h->num_channels; ++c) h->requested_pixel_types[c] = TINYEXR_PIXELTYPE_FLOAT;
    hold.images.resize(hold.headers.size());
    for (auto& im : hold.images) InitEXRImage(&im);
    if (ver.multipart) {
        if (LoadEXRMultipartImageFromFile(hold.images.data(), const_cast<const EXRHeader**>(hold.multi),
                                          unsigned(hold.headers.size()), path.c_str(), &err) != TINYEXR_SUCCESS)
            exrFail(path, err);
    } else {
        if (LoadEXRImageFromFile(&hold.images[0], &hold.single, path.c_str(), &err) != TINYEXR_SUCCESS)
            exrFail(path, err);
    }

    FileImage fi;
    const EXRHeader* h0 = hold.headers[0];
    fi.format.w = h0->display_window.max_x - h0->display_window.min_x + 1;
    fi.format.h = h0->display_window.max_y - h0->display_window.min_y + 1;
    fi.format.par = h0->pixel_aspect_ratio > 0 ? h0->pixel_aspect_ratio : 1.0;
    const int ox = h0->display_window.min_x, oy = h0->display_window.min_y;
    fi.floatData = true;

    Box bbox;
    ChannelList names;
    std::vector<std::vector<std::string>> partNames(hold.headers.size());
    for (size_t p = 0; p < hold.headers.size(); ++p) {
        const EXRHeader* h = hold.headers[p];
        Box b{h->data_window.min_x - ox, h->data_window.min_y - oy, h->data_window.max_x - ox + 1,
              h->data_window.max_y - oy + 1};
        bbox = bbox.unite(b);
        std::string part = h->name;
        for (int c = 0; c < h->num_channels; ++c) {
            std::string raw = h->channels[c].name;
            if (hold.headers.size() > 1 && p > 0 && raw.find('.') == std::string::npos && !part.empty())
                raw = part + "." + raw;
            std::string n = canonicalChannel(raw);
            while (hasChannel(names, n)) n += "_";
            names.push_back(n);
            partNames[p].push_back(n);
        }
    }
    auto img = std::make_shared<Image>(bbox, sortedChannels(names));
    for (size_t p = 0; p < hold.headers.size(); ++p) {
        const EXRHeader* h = hold.headers[p];
        const EXRImage& im = hold.images[p];
        int dx = h->data_window.min_x - ox, dy = h->data_window.min_y - oy;
        int w = h->data_window.max_x - h->data_window.min_x + 1;
        for (int c = 0; c < h->num_channels; ++c) {
            float* dst = img->plane(partNames[p][c]);
            if (h->tiled && im.tiles) {
                for (int t = 0; t < im.num_tiles; ++t) {
                    const EXRTile& tile = im.tiles[t];
                    if (tile.level_x != 0 || tile.level_y != 0) continue;
                    const float* src = reinterpret_cast<const float*>(tile.images[c]);
                    int tx0 = tile.offset_x * h->tile_size_x, ty0 = tile.offset_y * h->tile_size_y;
                    for (int y = 0; y < tile.height; ++y) {
                        int ay = dy + ty0 + y;
                        float* row = dst + size_t(ay - bbox.y0) * bbox.w() + (dx + tx0 - bbox.x0);
                        std::memcpy(row, src + size_t(y) * h->tile_size_x, sizeof(float) * tile.width);
                    }
                }
            } else if (im.images) {
                const float* src = reinterpret_cast<const float*>(im.images[c]);
                int hgt = h->data_window.max_y - h->data_window.min_y + 1;
                for (int y = 0; y < hgt; ++y) {
                    float* row = dst + size_t(dy + y - bbox.y0) * bbox.w() + (dx - bbox.x0);
                    std::memcpy(row, src + size_t(y) * w, sizeof(float) * w);
                }
            }
        }
        for (int a = 0; a < h->num_custom_attributes; ++a) {
            const EXRAttribute& at = h->custom_attributes[a];
            if (std::strcmp(at.type, "string") == 0)
                img->meta[std::string("exr/") + at.name] = std::string(reinterpret_cast<const char*>(at.value), at.size);
        }
    }
    static const char* comps[] = {"none", "rle", "zips", "zip", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab"};
    if (h0->compression_type >= 0 && h0->compression_type < 10) img->meta["exr/compression"] = comps[h0->compression_type];
    img->meta["exr/parts"] = std::to_string(hold.headers.size());
    fi.hasAlpha = img->has("rgba.alpha");
    fi.image = img;
    return fi;
}

FileImage readStb(const std::string& path) {
    int w = 0, h = 0, n = 0;
    FileImage fi;
    ChannelList chans;
    std::vector<float> data;
    if (stbi_is_hdr(path.c_str())) {
        float* p = stbi_loadf(path.c_str(), &w, &h, &n, 0);
        if (!p) throw Error("cannot read '" + path + "': " + (stbi_failure_reason() ? stbi_failure_reason() : "unknown"));
        data.assign(p, p + size_t(w) * h * n);
        stbi_image_free(p);
        fi.floatData = true;
    } else if (stbi_is_16_bit(path.c_str())) {
        stbi_us* p = stbi_load_16(path.c_str(), &w, &h, &n, 0);
        if (!p) throw Error("cannot read '" + path + "': " + (stbi_failure_reason() ? stbi_failure_reason() : "unknown"));
        data.resize(size_t(w) * h * n);
        for (size_t i = 0; i < data.size(); ++i) data[i] = p[i] / 65535.0f;
        stbi_image_free(p);
    } else {
        stbi_uc* p = stbi_load(path.c_str(), &w, &h, &n, 0);
        if (!p) throw Error("cannot read '" + path + "': " + (stbi_failure_reason() ? stbi_failure_reason() : "unknown"));
        data.resize(size_t(w) * h * n);
        for (size_t i = 0; i < data.size(); ++i) data[i] = p[i] / 255.0f;
        stbi_image_free(p);
    }
    bool alpha = n == 2 || n == 4;
    chans = {"rgba.red", "rgba.green", "rgba.blue"};
    if (alpha) chans.push_back("rgba.alpha");
    auto img = std::make_shared<Image>(Box{0, 0, w, h}, chans);
    float* r = img->plane("rgba.red");
    float* g = img->plane("rgba.green");
    float* b = img->plane("rgba.blue");
    float* a = alpha ? img->plane("rgba.alpha") : nullptr;
    size_t px = size_t(w) * h;
    for (size_t i = 0; i < px; ++i) {
        const float* s = &data[i * n];
        if (n <= 2) {
            r[i] = g[i] = b[i] = s[0];
        } else {
            r[i] = s[0];
            g[i] = s[1];
            b[i] = s[2];
        }
        if (a) a[i] = s[n - 1];
    }
    fi.format.w = w;
    fi.format.h = h;
    fi.hasAlpha = alpha;
    fi.image = img;
    return fi;
}

uint32_t crc32(const unsigned char* d, size_t n, uint32_t crc = 0) {
    static uint32_t table[256];
    static std::once_flag once;
    std::call_once(once, [] {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
    });
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) crc = table[(crc ^ d[i]) & 0xff] ^ (crc >> 8);
    return ~crc;
}

void writePng16(const std::string& path, int w, int h, int comps, const std::vector<uint16_t>& px) {
    std::vector<unsigned char> raw;
    raw.reserve(size_t(h) * (1 + size_t(w) * comps * 2));
    for (int y = 0; y < h; ++y) {
        raw.push_back(0);
        for (int i = 0; i < w * comps; ++i) {
            uint16_t v = px[size_t(y) * w * comps + i];
            raw.push_back(uint8_t(v >> 8));
            raw.push_back(uint8_t(v & 0xff));
        }
    }
    int zlen = 0;
    unsigned char* z = stbi_zlib_compress(raw.data(), int(raw.size()), &zlen, 6);
    if (!z) throw Error("png16: compression failed");
    std::FILE* f = nullptr;
#ifdef _WIN32
    f = _wfopen(fs::u8path(path).wstring().c_str(), L"wb");
#else
    f = std::fopen(path.c_str(), "wb");
#endif
    if (!f) {
        free(z);
        throw Error("cannot open '" + path + "' for writing");
    }
    auto be32 = [](uint32_t v, unsigned char* o) {
        o[0] = uint8_t(v >> 24); o[1] = uint8_t(v >> 16); o[2] = uint8_t(v >> 8); o[3] = uint8_t(v);
    };
    auto chunk = [&](const char* type, const unsigned char* data, uint32_t len) {
        unsigned char hdr[8];
        be32(len, hdr);
        std::memcpy(hdr + 4, type, 4);
        std::fwrite(hdr, 1, 8, f);
        if (len) std::fwrite(data, 1, len, f);
        uint32_t c = crc32(hdr + 4, 4);
        c = crc32(data, len, c);
        unsigned char cb[4];
        be32(c, cb);
        std::fwrite(cb, 1, 4, f);
    };
    static const unsigned char sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    std::fwrite(sig, 1, 8, f);
    unsigned char ihdr[13];
    be32(uint32_t(w), ihdr);
    be32(uint32_t(h), ihdr + 4);
    ihdr[8] = 16;
    ihdr[9] = comps == 4 ? 6 : 2;
    ihdr[10] = ihdr[11] = ihdr[12] = 0;
    chunk("IHDR", ihdr, 13);
    chunk("IDAT", z, uint32_t(zlen));
    chunk("IEND", nullptr, 0);
    std::fclose(f);
    free(z);
}

std::string exrName(const std::string& ch) {
    std::string layer = layerOf(ch), c = shortOf(ch);
    if (c == "red") c = "R";
    else if (c == "green") c = "G";
    else if (c == "blue") c = "B";
    else if (c == "alpha") c = "A";
    if (layer == "rgba") return c;
    if (layer == "other") return c;
    return layer + "." + c;
}

void writeExr(const std::string& path, const Image& img, const Format& fmt, const ChannelList& chansIn,
              const WriteOptions& opt) {
    ChannelList chans = chansIn;
    std::sort(chans.begin(), chans.end(), [](const std::string& a, const std::string& b) {
        return std::strcmp(exrName(a).c_str(), exrName(b).c_str()) < 0;
    });
    Box b = img.bbox();
    if (b.empty()) b = Box{0, 0, 1, 1};
    int w = b.w(), h = b.h();
    std::vector<std::vector<float>> planes(chans.size());
    std::vector<unsigned char*> ptrs(chans.size());
    for (size_t i = 0; i < chans.size(); ++i) {
        planes[i].assign(size_t(w) * h, 0.0f);
        const float* src = img.plane(chans[i]);
        if (src && !img.bbox().empty()) std::memcpy(planes[i].data(), src, sizeof(float) * planes[i].size());
        ptrs[i] = reinterpret_cast<unsigned char*>(planes[i].data());
    }
    EXRHeader hdr;
    InitEXRHeader(&hdr);
    EXRImage im;
    InitEXRImage(&im);
    im.num_channels = int(chans.size());
    im.images = ptrs.data();
    im.width = w;
    im.height = h;
    std::vector<EXRChannelInfo> infos(chans.size());
    std::vector<int> ptypes(chans.size(), TINYEXR_PIXELTYPE_FLOAT), rtypes(chans.size());
    bool half = opt.dataType.find("half") != std::string::npos || opt.dataType.find("16") != std::string::npos;
    for (size_t i = 0; i < chans.size(); ++i) {
        std::memset(&infos[i], 0, sizeof(EXRChannelInfo));
        std::string n = exrName(chans[i]);
        std::strncpy(infos[i].name, n.c_str(), 255);
        rtypes[i] = half ? TINYEXR_PIXELTYPE_HALF : TINYEXR_PIXELTYPE_FLOAT;
    }
    hdr.num_channels = int(chans.size());
    hdr.channels = infos.data();
    hdr.pixel_types = ptypes.data();
    hdr.requested_pixel_types = rtypes.data();
    hdr.data_window = {b.x0, b.y0, b.x1 - 1, b.y1 - 1};
    hdr.display_window = {0, 0, fmt.w - 1, fmt.h - 1};
    hdr.pixel_aspect_ratio = float(fmt.par);
    std::string c = lower(opt.compression);
    hdr.compression_type = c == "none" ? TINYEXR_COMPRESSIONTYPE_NONE
                           : c == "zips" ? TINYEXR_COMPRESSIONTYPE_ZIPS
                           : c == "piz"  ? TINYEXR_COMPRESSIONTYPE_PIZ
                                         : TINYEXR_COMPRESSIONTYPE_ZIP;
    // String metadata passes through.
    std::vector<EXRAttribute> attrs;
    std::vector<std::string> values;
    for (auto& kv : img.meta) {
        if (kv.first.rfind("exr/", 0) != 0) continue;
        std::string name = kv.first.substr(4);
        if (name == "compression" || name == "parts" || name.empty() || name.size() > 250) continue;
        values.push_back(kv.second);
    }
    size_t vi = 0;
    for (auto& kv : img.meta) {
        if (kv.first.rfind("exr/", 0) != 0) continue;
        std::string name = kv.first.substr(4);
        if (name == "compression" || name == "parts" || name.empty() || name.size() > 250) continue;
        EXRAttribute a;
        std::memset(&a, 0, sizeof a);
        std::strncpy(a.name, name.c_str(), 255);
        std::strncpy(a.type, "string", 255);
        a.value = reinterpret_cast<unsigned char*>(const_cast<char*>(values[vi].data()));
        a.size = int(values[vi].size());
        ++vi;
        attrs.push_back(a);
    }
    hdr.num_custom_attributes = int(attrs.size());
    hdr.custom_attributes = attrs.empty() ? nullptr : attrs.data();
    const char* err = nullptr;
    int ret = SaveEXRImageToFile(&im, &hdr, path.c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        std::string msg = "cannot write EXR '" + path + "'";
        if (err) {
            msg += ": ";
            msg += err;
            FreeEXRErrorMessage(err);
        }
        throw Error(msg);
    }
}

std::mutex& movieMtx() {
    static std::mutex m;
    return m;
}
MovieInfoFn& movieInfoFn() {
    static MovieInfoFn f;
    return f;
}
MovieFrameFn& movieFrameFn() {
    static MovieFrameFn f;
    return f;
}

}  // namespace

FileImage readImage(const std::string& path) {
    if (!fileExists(path)) throw Error("file not found: " + path);
    std::string ext = extOf(path);
    if (ext == ".exr") return readExr(path);
    if (ext == ".tif" || ext == ".tiff" || ext == ".dpx")
        throw Error(ext + " is not supported by the built-in readers (see docs/decisions.md)");
    return readStb(path);
}

void writeImage(const std::string& path, const Image& img, const Format& fmt, const ChannelList& chans,
                const WriteOptions& opt) {
    fs::path p = fs::u8path(path);
    if (p.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(p.parent_path(), ec);
    }
    std::string t = lower(opt.fileType);
    if (t == "exr") return writeExr(path, img, fmt, chans, opt);

    // Integer and RGBE formats write the full format window.
    int w = fmt.w, h = fmt.h;
    bool alpha = hasChannel(chans, "rgba.alpha") && t != "jpg" && t != "hdr";
    int comps = alpha ? 4 : 3;
    const char* names[4] = {"rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"};
    auto value = [&](int c, int x, int y) { return img.at(names[c], x, y); };
    if (t == "hdr") {
        std::vector<float> buf(size_t(w) * h * 3);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                for (int c = 0; c < 3; ++c) buf[(size_t(y) * w + x) * 3 + c] = std::max(0.0f, value(c, x, y));
        if (!stbi_write_hdr(path.c_str(), w, h, 3, buf.data())) throw Error("cannot write '" + path + "'");
        return;
    }
    bool sixteen = t == "png" && opt.dataType.find("16") != std::string::npos;
    if (sixteen) {
        std::vector<uint16_t> buf(size_t(w) * h * comps);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                for (int c = 0; c < comps; ++c) {
                    float v = std::min(1.0f, std::max(0.0f, value(c, x, y)));
                    buf[(size_t(y) * w + x) * comps + c] = uint16_t(std::lround(v * 65535.0f));
                }
        writePng16(path, w, h, comps, buf);
        return;
    }
    std::vector<unsigned char> buf(size_t(w) * h * comps);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < comps; ++c) {
                float v = std::min(1.0f, std::max(0.0f, value(c, x, y)));
                buf[(size_t(y) * w + x) * comps + c] = static_cast<unsigned char>(std::lround(v * 255.0f));
            }
    int ok = 0;
    if (t == "png") ok = stbi_write_png(path.c_str(), w, h, comps, buf.data(), w * comps);
    else if (t == "jpg" || t == "jpeg") ok = stbi_write_jpg(path.c_str(), w, h, comps, buf.data(), opt.quality);
    else if (t == "tga") ok = stbi_write_tga(path.c_str(), w, h, comps, buf.data());
    else if (t == "bmp") ok = stbi_write_bmp(path.c_str(), w, h, comps, buf.data());
    else throw Error("unsupported output type '" + t + "'");
    if (!ok) throw Error("cannot write '" + path + "'");
}

// ---------------------------------------------------------------- sequences

namespace {
// Returns {prefix, padding, suffix} if the pattern has a frame token.
bool splitPattern(const std::string& s, std::string& pre, int& pad, std::string& suf) {
    static const std::regex hashes("(#+|@+)");
    static const std::regex printf_("%0?(\\d*)d");
    static const std::regex houdini("\\$F(\\d*)");
    std::smatch m;
    std::string name = s;
    if (std::regex_search(name, m, printf_)) {
        pre = m.prefix();
        suf = m.suffix();
        pad = m[1].length() ? std::stoi(m[1]) : 1;
        return true;
    }
    if (std::regex_search(name, m, houdini)) {
        pre = m.prefix();
        suf = m.suffix();
        pad = m[1].length() ? std::stoi(m[1]) : 1;
        return true;
    }
    // Only consider hashes in the file name, not directories.
    auto slash = s.find_last_of("/\\");
    std::string dir = slash == std::string::npos ? "" : s.substr(0, slash + 1);
    std::string file = slash == std::string::npos ? s : s.substr(slash + 1);
    if (std::regex_search(file, m, hashes)) {
        pre = dir + std::string(m.prefix());
        suf = m.suffix();
        pad = int(m[1].length());
        return true;
    }
    return false;
}
}  // namespace

bool isPattern(const std::string& path) {
    std::string a, c;
    int b;
    return splitPattern(path, a, b, c);
}

std::string framePath(const std::string& pattern, int frame) {
    std::string pre, suf;
    int pad;
    if (!splitPattern(pattern, pre, pad, suf)) return pattern;
    std::string n = std::to_string(std::abs(frame));
    while (int(n.size()) < pad) n = "0" + n;
    if (frame < 0) n = "-" + n;
    return pre + n + suf;
}

bool fileExists(const std::string& path) {
    std::error_code ec;
    return fs::is_regular_file(fs::u8path(path), ec);
}

int64_t fileMTime(const std::string& path) {
    std::error_code ec;
    auto t = fs::last_write_time(fs::u8path(path), ec);
    if (ec) return 0;
    return int64_t(t.time_since_epoch().count());
}

SeqInfo detectSequence(const std::string& path) {
    SeqInfo si;
    std::string pre, suf;
    int pad = 0;
    fs::path p = fs::u8path(path);
    std::string dir = p.has_parent_path() ? p.parent_path().u8string() : ".";
    if (!splitPattern(path, pre, pad, suf)) {
        // Concrete file: look for the last run of digits in the file name.
        std::string file = p.filename().u8string();
        std::smatch m;
        static const std::regex digits("(\\d+)(?!.*\\d)");
        if (!std::regex_search(file, m, digits)) {
            si.pattern = path;
            return si;
        }
        std::string dpre = p.has_parent_path() ? p.parent_path().u8string() + "/" : "";
        pre = dpre + std::string(m.prefix());
        suf = m.suffix();
        pad = int(m[1].length());
    }
    std::string filePre = fs::u8path(pre + "0").filename().u8string();
    filePre.pop_back();
    std::set<int> frames;
    std::error_code ec;
    for (auto& e : fs::directory_iterator(fs::u8path(dir), ec)) {
        std::string f = e.path().filename().u8string();
        if (f.size() <= filePre.size() + suf.size()) continue;
        if (f.compare(0, filePre.size(), filePre) != 0) continue;
        if (f.compare(f.size() - suf.size(), suf.size(), suf) != 0) continue;
        std::string mid = f.substr(filePre.size(), f.size() - filePre.size() - suf.size());
        if (mid.empty() || !std::all_of(mid.begin(), mid.end(), [](char c) { return std::isdigit(static_cast<unsigned char>(c)); }))
            continue;
        if (int(mid.size()) != pad && !(mid.size() > size_t(pad) && mid[0] != '0')) continue;
        frames.insert(std::stoi(mid));
    }
    if (frames.size() <= 1 && !isPattern(path)) {
        si.pattern = path;
        if (!frames.empty()) si.first = si.last = *frames.begin();
        return si;
    }
    si.pattern = pre + std::string(size_t(std::max(1, pad)), '#') + suf;
    si.sequence = true;
    if (!frames.empty()) {
        si.first = *frames.begin();
        si.last = *frames.rbegin();
    }
    return si;
}

bool isMovie(const std::string& path) {
    static const std::set<std::string> exts = {".mov", ".mp4", ".mkv", ".avi", ".mxf", ".webm", ".m4v"};
    return exts.count(extOf(path)) > 0;
}

void setMovieProvider(MovieInfoFn info, MovieFrameFn frame) {
    std::lock_guard<std::mutex> lk(movieMtx());
    movieInfoFn() = std::move(info);
    movieFrameFn() = std::move(frame);
}

bool movieInfo(const std::string& path, MovieInfo& out) {
    MovieInfoFn f;
    {
        std::lock_guard<std::mutex> lk(movieMtx());
        f = movieInfoFn();
    }
    if (!f) throw Error("no movie decoder available (install imageio-ffmpeg)");
    return f(path, out);
}

MutImagePtr movieFrame(const std::string& path, int frame) {
    MovieFrameFn f;
    {
        std::lock_guard<std::mutex> lk(movieMtx());
        f = movieFrameFn();
    }
    if (!f) throw Error("no movie decoder available (install imageio-ffmpeg)");
    return f(path, frame);
}

}  // namespace vc
