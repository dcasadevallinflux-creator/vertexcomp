#include <algorithm>
#include <exception>
#include <memory>

#include "vc/util.h"

namespace vc {

ThreadPool::ThreadPool(int threads) { start(threads); }

ThreadPool::~ThreadPool() { stop(); }

void ThreadPool::start(int n) {
    if (n <= 0) n = int(std::max(1u, std::thread::hardware_concurrency()));
    quit_ = false;
    for (int i = 0; i < n - 1; ++i) workers_.emplace_back([this] { loop(); });
}

void ThreadPool::stop() {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        quit_ = true;
    }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
    workers_.clear();
}

void ThreadPool::resize(int threads) {
    stop();
    start(threads);
}

void ThreadPool::loop() {
    for (;;) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [this] { return quit_ || !queue_.empty(); });
            if (quit_ && queue_.empty()) return;
            job = std::move(queue_.front());
            queue_.pop_front();
        }
        job();
    }
}

namespace {
struct ForState {
    std::atomic<int> next;
    std::atomic<int> remaining;
    std::mutex mtx;
    std::condition_variable done;
    std::exception_ptr error;
    int end, chunk;
};
}  // namespace

void ThreadPool::parallelFor(int begin, int end, const std::function<void(int, int)>& fn,
                             const CancelToken* cancel, int grain) {
    if (end <= begin) return;
    int n = end - begin;
    int threads = size();
    if (grain <= 0) grain = std::max(1, n / (threads * 4));
    int chunks = (n + grain - 1) / grain;
    if (threads == 1 || chunks == 1) {
        for (int b = begin; b < end; b += grain) {
            if (cancel) cancel->check();
            fn(b, std::min(end, b + grain));
        }
        return;
    }
    auto st = std::make_shared<ForState>();
    st->next = begin;
    st->remaining = chunks;
    st->end = end;
    st->chunk = grain;

    auto work = [st, &fn, cancel]() {
        for (;;) {
            int b = st->next.fetch_add(st->chunk);
            if (b >= st->end) return;
            int e = std::min(st->end, b + st->chunk);
            try {
                if (cancel) cancel->check();
                bool failed;
                {
                    std::lock_guard<std::mutex> lk(st->mtx);
                    failed = bool(st->error);
                }
                if (!failed) fn(b, e);
            } catch (...) {
                std::lock_guard<std::mutex> lk(st->mtx);
                if (!st->error) st->error = std::current_exception();
            }
            if (st->remaining.fetch_sub(1) == 1) {
                std::lock_guard<std::mutex> lk(st->mtx);
                st->done.notify_all();
            }
        }
    };
    int helpers = std::min(int(workers_.size()), chunks - 1);
    {
        std::lock_guard<std::mutex> lk(mtx_);
        for (int i = 0; i < helpers; ++i) queue_.emplace_back(work);
    }
    cv_.notify_all();
    work();
    {
        std::unique_lock<std::mutex> lk(st->mtx);
        st->done.wait(lk, [&] { return st->remaining.load() <= 0; });
    }
    if (st->error) std::rethrow_exception(st->error);
}

ThreadPool& ThreadPool::global() {
    static ThreadPool pool(0);
    return pool;
}

}  // namespace vc
