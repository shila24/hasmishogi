#ifndef MCTS_POOL_HPP
#define MCTS_POOL_HPP

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace mcts {
// Each task owns one tree. Publication and completion use the same mutex, so
// evaluation never races descent/backup and no tree statistics need atomics.
class Pool {
public:
    explicit Pool(int count) {
        try {
            for (int i = 0; i < count - 1; ++i)
                workers_.emplace_back([this, i, count] { worker(static_cast<std::size_t>(i + 1), count); });
        } catch (...) {
            stop();
            throw;
        }
    }
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    ~Pool() { stop(); }

    void run(std::size_t size, const std::function<void(std::size_t)>& task) {
        if (size < 2 || workers_.empty()) {
            for (std::size_t i = 0; i < size; ++i) task(i);
            return;
        }
        {
            std::lock_guard lock(mutex_);
            task_ = task;
            size_ = size;
            errors_.assign(workers_.size() + 1, nullptr);
            remaining_ = workers_.size();
            ++generation_;
        }
        ready_.notify_all();
        execute(0, workers_.size() + 1);
        {
            std::unique_lock lock(mutex_);
            finished_.wait(lock, [this] { return remaining_ == 0; });
            task_ = {};
        }
        for (const auto& error : errors_) if (error) std::rethrow_exception(error);
    }
private:
    void execute(std::size_t index, std::size_t stride) noexcept {
        try {
            for (auto i = index; i < size_; i += stride) task_(i);
        } catch (...) { errors_[index] = std::current_exception(); }
    }
    void worker(std::size_t index, std::size_t stride) {
        std::size_t seen = 0;
        for (;;) {
            std::unique_lock lock(mutex_);
            ready_.wait(lock, [this, seen] { return stopping_ || generation_ != seen; });
            if (stopping_) return;
            seen = generation_;
            lock.unlock();
            execute(index, stride);
            lock.lock();
            if (--remaining_ == 0) finished_.notify_one();
        }
    }
    void stop() noexcept {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& worker : workers_) if (worker.joinable()) worker.join();
    }
    std::mutex mutex_;
    std::condition_variable ready_, finished_;
    std::vector<std::thread> workers_;
    std::function<void(std::size_t)> task_;
    std::vector<std::exception_ptr> errors_;
    std::size_t generation_{}, size_{}, remaining_{};
    bool stopping_{};
};
} // namespace mcts
#endif
