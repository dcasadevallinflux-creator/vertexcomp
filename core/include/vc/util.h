#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace vc {

constexpr double kPi = 3.14159265358979323846;

struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// An error already attributed to a node ("Name: message"); passes through
// downstream nodes unchanged so only the root cause is flagged.
struct NodeError : Error {
    using Error::Error;
};

struct Cancelled : std::runtime_error {
    Cancelled() : std::runtime_error("cancelled") {}
};

class CancelToken {
public:
    void cancel() { flag_.store(true, std::memory_order_relaxed); }
    bool cancelled() const { return flag_.load(std::memory_order_relaxed); }
    void check() const {
        if (cancelled()) throw Cancelled();
    }

private:
    std::atomic<bool> flag_{false};
};

// FNV-1a over bytes, then mixed; good enough for cache keys.
struct Hasher {
    uint64_t h = 1469598103934665603ull;
    Hasher& bytes(const void* p, size_t n) {
        auto* c = static_cast<const unsigned char*>(p);
        for (size_t i = 0; i < n; ++i) {
            h ^= c[i];
            h *= 1099511628211ull;
        }
        return *this;
    }
    Hasher& add(double v) {
        if (v == 0.0) v = 0.0;  // fold -0
        return bytes(&v, sizeof v);
    }
    Hasher& add(int64_t v) { return bytes(&v, sizeof v); }
    Hasher& add(uint64_t v) { return bytes(&v, sizeof v); }
    Hasher& add(int v) { return add(int64_t(v)); }
    Hasher& add(const std::string& s) {
        add(uint64_t(s.size()));
        return bytes(s.data(), s.size());
    }
    uint64_t value() const {
        uint64_t x = h;
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdull;
        x ^= x >> 33;
        return x;
    }
};

// Work-sharing pool. parallelFor splits a range into chunks; the calling
// thread works on chunks too, so nested calls cannot deadlock.
class ThreadPool {
public:
    explicit ThreadPool(int threads = 0);
    ~ThreadPool();
    void resize(int threads);
    int size() const { return int(workers_.size()) + 1; }

    // fn(begin, end) is called on disjoint sub-ranges covering [begin, end).
    void parallelFor(int begin, int end, const std::function<void(int, int)>& fn,
                     const CancelToken* cancel = nullptr, int grain = 0);

    static ThreadPool& global();

private:
    void start(int n);
    void stop();
    void loop();

    std::vector<std::thread> workers_;
    std::deque<std::function<void()>> queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool quit_ = false;
};

// Deterministic integer hash noise: depends only on its arguments.
inline uint32_t hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
inline uint32_t hash32(int x, int y, int z, uint32_t seed) {
    return hash32(uint32_t(x) * 0x8da6b343U ^ hash32(uint32_t(y) * 0xd8163841U ^ hash32(uint32_t(z) * 0xcb1ab31fU ^ seed)));
}
inline float hashFloat(uint32_t h) { return float(h >> 8) * (1.0f / 16777216.0f); }

}  // namespace vc
