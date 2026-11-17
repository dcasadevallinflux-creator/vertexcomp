#pragma once
// Roto shapes: cubic Bezier splines with per-key point animation, a shape
// transform with its own animation curves, and a signed-distance rasterizer.
//
// Coverage: for a pixel centre at signed distance d from the outline
// (negative inside), hard-edge coverage is clamp(0.5 - d, 0, 1), which is the
// exact area coverage for a straight edge crossing the pixel and a close
// approximation along curves. Feather extends the falloff outwards
// (feather > 0) or inwards (feather < 0) over |feather| pixels.

#include <map>
#include <string>
#include <vector>

#include "vc/graph.h"
#include "vc/image.h"

namespace vc {

struct RotoPoint {
    double x = 0, y = 0;
    double inX = 0, inY = 0;    // tangent handles relative to the point
    double outX = 0, outY = 0;
    double feather = 0;         // extra feather at this point, added to shape feather
    bool smooth = true;         // UI hint: handles move together
};

enum class RotoBlend { Over, Add, Subtract, Max, Min, Multiply };

struct RotoShape {
    std::string name = "Shape";
    bool visible = true, locked = false, closed = true, invert = false;
    std::vector<double> color{1.0, 0.6, 0.1};  // overlay colour
    RotoBlend blend = RotoBlend::Over;
    double opacity = 1.0;
    double feather = 0.0;
    int featherFalloff = 0;  // 0 smooth, 1 linear
    double blur = 0.0;       // gaussian sigma applied to this shape's coverage
    int motionBlurSamples = 1;
    double shutter = 0.5;

    // Point animation: frame -> full point list. Every key holds the same
    // number of points; edits insert/remove in all keys.
    std::map<double, std::vector<RotoPoint>> keys;

    // Shape transform, independent of point animation.
    Curve tx, ty, rot, sx, sy, cx, cy;
    double txv = 0, tyv = 0, rotv = 0, sxv = 1, syv = 1, cxv = 0, cyv = 0;

    std::vector<RotoPoint> pointsAt(double frame) const;      // untransformed
    std::vector<RotoPoint> transformedAt(double frame) const;  // with shape transform
    void transformAt(double frame, double m[6]) const;         // 2x3 affine
};

struct RotoData {
    std::vector<RotoShape> shapes;
    void hash(Hasher& h, double frame) const;
};

// Rasterize all visible shapes into `out` (single plane over `box`).
void rasterizeRoto(const RotoData& data, double frame, const Box& box, float* out, const CancelToken& cancel);

// Flatten a closed or open Bezier outline to a polyline (absolute coords).
std::vector<std::pair<double, double>> flattenShape(const std::vector<RotoPoint>& pts, bool closed,
                                                    std::vector<double>* featherOut = nullptr);

}  // namespace vc
