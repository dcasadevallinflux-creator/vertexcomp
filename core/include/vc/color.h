#pragma once
// Built-in colour management: transfer functions plus primaries conversion
// between Rec.709/sRGB, ACEScg (AP1) and ACES2065-1 (AP0). This is the
// fallback used when no OCIO config is wired in (see docs/decisions.md).
//
// The working space is scene-linear with either Rec.709 or AP1 primaries.

#include <array>
#include <string>
#include <vector>

namespace vc {

using Mat3 = std::array<double, 9>;

struct ColorSpace {
    std::string name;
    int transfer;   // see color.cpp
    int primaries;  // 0 Rec.709, 1 AP1, 2 AP0
};

const std::vector<ColorSpace>& colorSpaces();
std::vector<std::string> colorSpaceNames();
const ColorSpace& colorSpace(const std::string& name);  // throws vc::Error

void setWorkingSpace(const std::string& name);  // "Linear Rec.709" or "ACEScg"
std::string workingSpace();

// Converts interleaved RGB triplets in place.
class ColorTransform {
public:
    ColorTransform(const std::string& from, const std::string& to);
    bool identity() const { return identity_; }
    void apply(float& r, float& g, float& b) const;

private:
    int decode_, encode_;
    Mat3 m_;
    bool matrix_, identity_;
};

double decodeTransfer(int transfer, double v);
double encodeTransfer(int transfer, double v);
Mat3 primariesMatrix(int from, int to);

}  // namespace vc
