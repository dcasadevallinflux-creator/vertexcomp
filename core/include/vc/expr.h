#pragma once
// Small, safe expression language used by knob expressions and the
// Expression node. Compiled once to a stack program; evaluation does not
// allocate. Supports + - * / % ^, comparisons (result 0/1), unary minus,
// ternary `c ? a : b`, numeric literals, variables and these functions:
//   sin cos tan asin acos atan atan2 sqrt pow exp log log10 abs floor ceil
//   round min max clamp mix step smoothstep fract mod sign
// Unknown identifiers are resolved through a callback at compile time.

#include <functional>
#include <string>
#include <vector>

namespace vc {

class Expr {
public:
    // Resolver: returns a variable slot index for a name, or -1 if unknown.
    using Resolver = std::function<int(const std::string&)>;

    Expr() = default;
    // Throws vc::Error with a readable message on syntax errors.
    Expr(const std::string& src, const Resolver& resolve);

    double eval(const double* vars) const;
    bool empty() const { return code_.empty(); }
    const std::string& source() const { return src_; }

    // Names referenced that were not single identifiers resolved by the
    // resolver (used for "Node.knob" dependencies).
    const std::vector<std::string>& references() const { return refs_; }

    struct Ins {
        int op;
        double v;
    };

private:
    std::string src_;
    std::vector<Ins> code_;
    std::vector<std::string> refs_;
    int maxStack_ = 0;
};

}  // namespace vc
