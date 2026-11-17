#include "vc/expr.h"

#include <cctype>
#include <cmath>
#include <cstdlib>

#include "vc/util.h"

namespace vc {

namespace {

enum Op {
    PUSH, VAR, NEG, NOT, ADD, SUB, MUL, DIV, MOD, POW, LT, GT, LE, GE, EQ, NE, AND, OR, SELECT,
    F_SIN, F_COS, F_TAN, F_ASIN, F_ACOS, F_ATAN, F_ATAN2, F_SQRT, F_POW, F_EXP, F_LOG, F_LOG10,
    F_ABS, F_FLOOR, F_CEIL, F_ROUND, F_MIN, F_MAX, F_CLAMP, F_MIX, F_STEP, F_SMOOTH, F_FRACT,
    F_MOD, F_SIGN
};

struct FnDef {
    const char* name;
    int op, argc;
};

const FnDef kFns[] = {
    {"sin", F_SIN, 1},     {"cos", F_COS, 1},     {"tan", F_TAN, 1},       {"asin", F_ASIN, 1},
    {"acos", F_ACOS, 1},   {"atan", F_ATAN, 1},   {"atan2", F_ATAN2, 2},   {"sqrt", F_SQRT, 1},
    {"pow", F_POW, 2},     {"exp", F_EXP, 1},     {"log", F_LOG, 1},       {"log10", F_LOG10, 1},
    {"abs", F_ABS, 1},     {"floor", F_FLOOR, 1}, {"ceil", F_CEIL, 1},     {"round", F_ROUND, 1},
    {"min", F_MIN, 2},     {"max", F_MAX, 2},     {"clamp", F_CLAMP, 3},   {"mix", F_MIX, 3},
    {"step", F_STEP, 2},   {"smoothstep", F_SMOOTH, 3}, {"fract", F_FRACT, 1}, {"mod", F_MOD, 2},
    {"sign", F_SIGN, 1},
};

struct Parser {
    const std::string& s;
    size_t i = 0;
    const Expr::Resolver& resolve;
    std::vector<Expr::Ins>& code;
    std::vector<std::string>& refs;
    int depth = 0, maxDepth = 0;

    void emit(int op, double v = 0) {
        code.push_back({op, v});
        if (op == PUSH || op == VAR) depth++;
        else if (op >= ADD && op <= OR) depth--;
        else if (op == SELECT) depth -= 2;
        maxDepth = std::max(maxDepth, depth);
    }
    void emitFn(int op, int argc) {
        code.push_back({op, double(argc)});
        depth -= argc - 1;
    }

    [[noreturn]] void fail(const std::string& msg) {
        throw Error("expression: " + msg + " at column " + std::to_string(i + 1) + " in '" + s + "'");
    }
    void ws() {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    }
    bool eat(const char* tok) {
        ws();
        size_t n = std::strlen(tok);
        if (s.compare(i, n, tok) == 0) {
            i += n;
            return true;
        }
        return false;
    }

    void ternary() {
        logical();
        if (eat("?")) {
            ternary();
            if (!eat(":")) fail("expected ':'");
            ternary();
            emit(SELECT);
        }
    }
    void logical() {
        compare();
        for (;;) {
            if (eat("&&")) { compare(); emit(AND); }
            else if (eat("||")) { compare(); emit(OR); }
            else break;
        }
    }
    void compare() {
        sum();
        for (;;) {
            if (eat("<=")) { sum(); emit(LE); }
            else if (eat(">=")) { sum(); emit(GE); }
            else if (eat("==")) { sum(); emit(EQ); }
            else if (eat("!=")) { sum(); emit(NE); }
            else if (eat("<")) { sum(); emit(LT); }
            else if (eat(">")) { sum(); emit(GT); }
            else break;
        }
    }
    void sum() {
        product();
        for (;;) {
            if (eat("+")) { product(); emit(ADD); }
            else if (eat("-")) { product(); emit(SUB); }
            else break;
        }
    }
    void product() {
        unary();
        for (;;) {
            if (eat("*")) { unary(); emit(MUL); }
            else if (eat("/")) { unary(); emit(DIV); }
            else if (eat("%")) { unary(); emit(MOD); }
            else break;
        }
    }
    void unary() {
        if (eat("-")) { unary(); emit(NEG); return; }
        if (eat("+")) { unary(); return; }
        if (eat("!")) { unary(); emit(NOT); return; }
        power();
    }
    void power() {
        primary();
        if (eat("^")) {
            unary();  // right associative
            emit(POW);
        }
    }
    void primary() {
        ws();
        if (i >= s.size()) fail("unexpected end");
        char c = s[i];
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
            char* end = nullptr;
            double v = std::strtod(s.c_str() + i, &end);
            if (end == s.c_str() + i) fail("bad number");
            i = size_t(end - s.c_str());
            emit(PUSH, v);
            return;
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            size_t b = i;
            while (i < s.size() && (std::isalnum(static_cast<unsigned char>(s[i])) || s[i] == '_' || s[i] == '.')) ++i;
            std::string name = s.substr(b, i - b);
            if (eat("(")) {
                for (auto& f : kFns) {
                    if (name == f.name) {
                        int argc = 0;
                        if (!eat(")")) {
                            do {
                                ternary();
                                argc++;
                            } while (eat(","));
                            if (!eat(")")) fail("expected ')'");
                        }
                        if (argc != f.argc) fail(name + "() takes " + std::to_string(f.argc) + " arguments");
                        emitFn(f.op, argc);
                        return;
                    }
                }
                fail("unknown function '" + name + "'");
            }
            if (name == "pi") { emit(PUSH, kPi); return; }
            if (name == "e") { emit(PUSH, 2.718281828459045); return; }
            int slot = resolve ? resolve(name) : -1;
            if (slot < 0) fail("unknown name '" + name + "'");
            if (name.find('.') != std::string::npos) refs.push_back(name);
            emit(VAR, slot);
            return;
        }
        if (eat("(")) {
            ternary();
            if (!eat(")")) fail("expected ')'");
            return;
        }
        fail(std::string("unexpected '") + c + "'");
    }
};

}  // namespace

Expr::Expr(const std::string& src, const Resolver& resolve) : src_(src) {
    Parser p{src, 0, resolve, code_, refs_};
    p.ternary();
    p.ws();
    if (p.i != src.size()) p.fail("trailing characters");
    if (p.maxDepth > 64) throw Error("expression too deep");
    maxStack_ = p.maxDepth;
}

double Expr::eval(const double* vars) const {
    double st[64];
    int sp = 0;
    for (const Ins& in : code_) {
        switch (in.op) {
            case PUSH: st[sp++] = in.v; break;
            case VAR: st[sp++] = vars[int(in.v)]; break;
            case NEG: st[sp - 1] = -st[sp - 1]; break;
            case NOT: st[sp - 1] = st[sp - 1] == 0.0 ? 1.0 : 0.0; break;
            case ADD: sp--; st[sp - 1] += st[sp]; break;
            case SUB: sp--; st[sp - 1] -= st[sp]; break;
            case MUL: sp--; st[sp - 1] *= st[sp]; break;
            case DIV: sp--; st[sp - 1] = st[sp] == 0.0 ? 0.0 : st[sp - 1] / st[sp]; break;
            case MOD: sp--; st[sp - 1] = st[sp] == 0.0 ? 0.0 : std::fmod(st[sp - 1], st[sp]); break;
            case POW: sp--; st[sp - 1] = std::pow(st[sp - 1], st[sp]); break;
            case LT: sp--; st[sp - 1] = st[sp - 1] < st[sp]; break;
            case GT: sp--; st[sp - 1] = st[sp - 1] > st[sp]; break;
            case LE: sp--; st[sp - 1] = st[sp - 1] <= st[sp]; break;
            case GE: sp--; st[sp - 1] = st[sp - 1] >= st[sp]; break;
            case EQ: sp--; st[sp - 1] = st[sp - 1] == st[sp]; break;
            case NE: sp--; st[sp - 1] = st[sp - 1] != st[sp]; break;
            case AND: sp--; st[sp - 1] = (st[sp - 1] != 0.0 && st[sp] != 0.0); break;
            case OR: sp--; st[sp - 1] = (st[sp - 1] != 0.0 || st[sp] != 0.0); break;
            case SELECT: sp -= 2; st[sp - 1] = st[sp - 1] != 0.0 ? st[sp] : st[sp + 1]; break;
            case F_SIN: st[sp - 1] = std::sin(st[sp - 1]); break;
            case F_COS: st[sp - 1] = std::cos(st[sp - 1]); break;
            case F_TAN: st[sp - 1] = std::tan(st[sp - 1]); break;
            case F_ASIN: st[sp - 1] = std::asin(st[sp - 1]); break;
            case F_ACOS: st[sp - 1] = std::acos(st[sp - 1]); break;
            case F_ATAN: st[sp - 1] = std::atan(st[sp - 1]); break;
            case F_ATAN2: sp--; st[sp - 1] = std::atan2(st[sp - 1], st[sp]); break;
            case F_SQRT: st[sp - 1] = std::sqrt(std::max(0.0, st[sp - 1])); break;
            case F_POW: sp--; st[sp - 1] = std::pow(st[sp - 1], st[sp]); break;
            case F_EXP: st[sp - 1] = std::exp(st[sp - 1]); break;
            case F_LOG: st[sp - 1] = st[sp - 1] > 0 ? std::log(st[sp - 1]) : 0.0; break;
            case F_LOG10: st[sp - 1] = st[sp - 1] > 0 ? std::log10(st[sp - 1]) : 0.0; break;
            case F_ABS: st[sp - 1] = std::fabs(st[sp - 1]); break;
            case F_FLOOR: st[sp - 1] = std::floor(st[sp - 1]); break;
            case F_CEIL: st[sp - 1] = std::ceil(st[sp - 1]); break;
            case F_ROUND: st[sp - 1] = std::round(st[sp - 1]); break;
            case F_MIN: sp--; st[sp - 1] = std::min(st[sp - 1], st[sp]); break;
            case F_MAX: sp--; st[sp - 1] = std::max(st[sp - 1], st[sp]); break;
            case F_CLAMP: sp -= 2; st[sp - 1] = std::min(std::max(st[sp - 1], st[sp]), st[sp + 1]); break;
            case F_MIX: sp -= 2; st[sp - 1] = st[sp - 1] + (st[sp] - st[sp - 1]) * st[sp + 1]; break;
            case F_STEP: sp--; st[sp - 1] = st[sp] >= st[sp - 1] ? 1.0 : 0.0; break;
            case F_SMOOTH: {
                sp -= 2;
                double e0 = st[sp - 1], e1 = st[sp], x = st[sp + 1];
                double t = e1 == e0 ? (x >= e1 ? 1.0 : 0.0) : std::min(1.0, std::max(0.0, (x - e0) / (e1 - e0)));
                st[sp - 1] = t * t * (3 - 2 * t);
                break;
            }
            case F_FRACT: st[sp - 1] = st[sp - 1] - std::floor(st[sp - 1]); break;
            case F_MOD: sp--; st[sp - 1] = st[sp] == 0.0 ? 0.0 : st[sp - 1] - st[sp] * std::floor(st[sp - 1] / st[sp]); break;
            case F_SIGN: st[sp - 1] = (st[sp - 1] > 0) - (st[sp - 1] < 0); break;
        }
    }
    return sp > 0 ? st[sp - 1] : 0.0;
}

}  // namespace vc
