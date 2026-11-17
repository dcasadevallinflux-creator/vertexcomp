#pragma once
// File I/O. Readers return pixels as stored in the file (no colour
// conversion, alpha as stored); the Read op converts to the working space.

#include <functional>
#include <string>

#include "vc/image.h"

namespace vc {

struct FileImage {
    MutImagePtr image;  // bbox = data window
    Format format;      // display window size
    bool hasAlpha = false;
    bool floatData = false;  // EXR/HDR: scene-linear by convention
};

FileImage readImage(const std::string& path);

struct WriteOptions {
    std::string fileType = "exr";      // exr png jpg tga bmp hdr
    std::string dataType = "16 bit half";  // exr: "16 bit half" / "32 bit float"; png: "8 bit" / "16 bit"
    std::string compression = "zip";   // exr: none zip zips piz
    int quality = 90;                  // jpg
};

// `chans` must be present in img. For 8/16-bit formats rgba is written (alpha
// only if present in chans).
void writeImage(const std::string& path, const Image& img, const Format& fmt, const ChannelList& chans,
                const WriteOptions& opt);

// Sequence patterns: name.####.exr, name.%04d.exr, name.$F4.exr, name.@@@@.exr
bool isPattern(const std::string& path);
std::string framePath(const std::string& pattern, int frame);

struct SeqInfo {
    std::string pattern;
    int first = 1, last = 1;
    bool sequence = false;
};
// From a concrete file of a sequence (or a pattern), discover the range.
SeqInfo detectSequence(const std::string& path);
bool fileExists(const std::string& path);
int64_t fileMTime(const std::string& path);

bool isMovie(const std::string& path);

// Movie decoding is delegated to a provider registered by the host (the
// Python app installs an FFmpeg-backed one). Frames are 0-based.
struct MovieInfo {
    int frames = 0, width = 0, height = 0;
    double fps = 24.0;
};
using MovieInfoFn = std::function<bool(const std::string&, MovieInfo&)>;
using MovieFrameFn = std::function<MutImagePtr(const std::string&, int)>;
void setMovieProvider(MovieInfoFn info, MovieFrameFn frame);
bool movieInfo(const std::string& path, MovieInfo& out);
MutImagePtr movieFrame(const std::string& path, int frame);

}  // namespace vc
