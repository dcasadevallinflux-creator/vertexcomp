#include "vc/color.h"

#include <cmath>
#include <mutex>

#include "vc/util.h"

namespace vc {

namespace {

enum Transfer { TLinear, TSRGB, TRec709, TGamma22, TGamma24, TCineon, TACEScct };

const Mat3 kIdent{1, 0, 0, 0, 1, 0, 0, 0, 1};
// AP1 -> Rec.709 (Bradford D60 -> D65), and its inverse.
const Mat3 kAP1to709{1.70505, -0.62179, -0.08326, -0.13026, 1.14080, -0.01055, -0.02400, -0.12897, 1.15297};
const Mat3 k709toAP1{0.61309, 0.33952, 0.04737, 0.07019, 0.91635, 0.01345, 0.02062, 0.10957, 0.86961};
const Mat3 kAP0toAP1{1.4514393161, -0.2365107469, -0.2149285693, -0.0765537734, 1.1762296998,
                     -0.0996759264, 0.0083161484, -0.0060324498, 0.9977163014};
const Mat3 kAP1toAP0{0.6954522414, 0.1406786965, 0.1638690622, 0.0447945634, 0.8596711185,
                     0.0955343182, -0.0055258826, 0.0040252103, 1.0015006723};

Mat3 mul(const Mat3& a, const Mat3& b) {
    Mat3 r{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) r[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
    return r;
}

Mat3 toAP1(int p) { return p == 0 ? k709toAP1 : p == 2 ? kAP0toAP1 : kIdent; }
Mat3 fromAP1(int p) { return p == 0 ? kAP1to709 : p == 2 ? kAP1toAP0 : kIdent; }

std::string& working() {
    static std::string w = "Linear Rec.709";
    return w;
}
std::mutex& workingMtx() {
    static std::mutex m;
    return m;
}

const double kCineonBlack = std::pow(10.0, (95.0 - 685.0) / 300.0);

}  // namespace

const std::vector<ColorSpace>& colorSpaces() {
    static const std::vector<ColorSpace> cs = {
        {"Linear Rec.709", TLinear, 0}, {"sRGB", TSRGB, 0},          {"Rec.709", TRec709, 0},
        {"Gamma 2.2", TGamma22, 0},     {"Gamma 2.4", TGamma24, 0},  {"Cineon", TCineon, 0},
        {"ACEScg", TLinear, 1},         {"ACEScct", TACEScct, 1},    {"ACES2065-1", TLinear, 2},
        {"raw", TLinear, -1},
    };
    return cs;
}

std::vector<std::string> colorSpaceNames() {
    std::vector<std::string> n;
    for (auto& c : colorSpaces()) n.push_back(c.name);
    return n;
}

const ColorSpace& colorSpace(const std::string& name) {
    for (auto& c : colorSpaces())
        if (c.name == name) return c;
    throw Error("unknown colorspace '" + name + "'");
}

void setWorkingSpace(const std::string& name) {
    colorSpace(name);
    std::lock_guard<std::mutex> lk(workingMtx());
    working() = name;
}

std::string workingSpace() {
    std::lock_guard<std::mutex> lk(workingMtx());
    return working();
}

double decodeTransfer(int t, double v) {
    switch (t) {
        case TSRGB: return v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4);
        case TRec709: return v < 0.081 ? v / 4.5 : std::pow((v + 0.099) / 1.099, 1.0 / 0.45);
        case TGamma22: return v < 0 ? -std::pow(-v, 2.2) : std::pow(v, 2.2);
        case TGamma24: return v < 0 ? -std::pow(-v, 2.4) : std::pow(v, 2.4);
        case TCineon: return (std::pow(10.0, (v * 1023.0 - 685.0) / 300.0) - kCineonBlack) / (1.0 - kCineonBlack);
        case TACEScct:
            if (v <= 0.155251141552511) return (v - 0.0729055341958355) / 10.5402377416545;
            return std::pow(2.0, v * 17.52 - 9.72);
        default: return v;
    }
}

double encodeTransfer(int t, double v) {
    switch (t) {
        case TSRGB: return v <= 0.0031308 ? v * 12.92 : 1.055 * std::pow(v, 1.0 / 2.4) - 0.055;
        case TRec709: return v < 0.018 ? v * 4.5 : 1.099 * std::pow(v, 0.45) - 0.099;
        case TGamma22: return v < 0 ? -std::pow(-v, 1 / 2.2) : std::pow(v, 1 / 2.2);
        case TGamma24: return v < 0 ? -std::pow(-v, 1 / 2.4) : std::pow(v, 1 / 2.4);
        case TCineon: {
            double x = v * (1.0 - kCineonBlack) + kCineonBlack;
            return (685.0 + 300.0 * std::log10(std::max(x, 1e-10))) / 1023.0;
        }
        case TACEScct:
            if (v <= 0.0078125) return 10.5402377416545 * v + 0.0729055341958355;
            return (std::log2(v) + 9.72) / 17.52;
        default: return v;
    }
}

Mat3 primariesMatrix(int from, int to) {
    if (from == to || from < 0 || to < 0) return kIdent;
    return mul(fromAP1(to), toAP1(from));
}

ColorTransform::ColorTransform(const std::string& from, const std::string& to) {
    const ColorSpace& a = colorSpace(from);
    const ColorSpace& b = colorSpace(to);
    decode_ = a.transfer;
    encode_ = b.transfer;
    m_ = primariesMatrix(a.primaries, b.primaries);
    matrix_ = m_ != kIdent;
    identity_ = a.primaries < 0 || b.primaries < 0 || (from == to) ||
                (decode_ == TLinear && encode_ == TLinear && !matrix_);
    if (a.primaries < 0 || b.primaries < 0) identity_ = true;
}

void ColorTransform::apply(float& r, float& g, float& b) const {
    if (identity_) return;
    double x = decodeTransfer(decode_, r), y = decodeTransfer(decode_, g), z = decodeTransfer(decode_, b);
    if (matrix_) {
        double nx = m_[0] * x + m_[1] * y + m_[2] * z;
        double ny = m_[3] * x + m_[4] * y + m_[5] * z;
        double nz = m_[6] * x + m_[7] * y + m_[8] * z;
        x = nx; y = ny; z = nz;
    }
    r = float(encodeTransfer(encode_, x));
    g = float(encodeTransfer(encode_, y));
    b = float(encodeTransfer(encode_, z));
}

}  // namespace vc
