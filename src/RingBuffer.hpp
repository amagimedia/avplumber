#pragma once

#include "picture_buffer.hpp"
#include <assert.h>
#include <atomic>
#include <algorithm>
#include <unistd.h>

template<typename T> class BufferResult {
protected:
    T* buffer_;
    size_t count_;
    bool more_available_;
public:
    using iterator = T*;
    BufferResult(T* const buffer, const size_t count, const bool more_available):
        buffer_(buffer),
        count_(count),
        more_available_(more_available) {
        assert( (buffer!=nullptr) || (count==0) );
    }
    BufferResult(std::nullptr_t): buffer_(nullptr), count_(0), more_available_(false) {
    }
    bool isMoreAvailable() const noexcept {
        return more_available_;
    }
    size_t count() const noexcept {
        return count_;
    }
    T* begin() noexcept {
        return buffer_;
    }
    T* end() noexcept {
        return buffer_ + count_;
    }
    const T* begin() const noexcept {
        return buffer_;
    }
    const T* end() const noexcept {
        return buffer_ + count_;
    }
    T& operator[](const size_t index) noexcept {
        return buffer_[index];
    }
    const T& operator[](const size_t index) const noexcept {
        return buffer_[index];
    }
    bool isEmpty() const noexcept {
        return count_==0;
    }
    operator bool() const noexcept {
        return !isEmpty();
    }
    template<typename Elem> explicit operator BufferResult<Elem>() noexcept {
        return BufferResult<Elem>(reinterpret_cast<Elem*>(buffer_), count_ * sizeof(T) / sizeof(Elem), more_available_);
    }
};

template<typename Elem, typename Underlying = uint8_t, size_t elem_size = sizeof(Elem)> class RingBufferView;

template <typename T> class RingBuffer {
protected:
    size_t size_;
    std::atomic_size_t read_pos_ {0};
    std::atomic_size_t write_pos_ {0};
    T* buffer_;
public:
    RingBuffer(const size_t size): size_(size) {
        buffer_ = new T[size_];
        std::fill(buffer_, buffer_ + size_, 0);
    }
    RingBuffer(RingBuffer &&movefrom):
        size_(movefrom.size_),
        read_pos_(movefrom.read_pos_.load()),
        write_pos_(movefrom.write_pos_.load()),
        buffer_(movefrom.buffer_)
        {
    }
    size_t size() const { return size_; }
    size_t fillCount() const {
        if (!buffer_) return 0;
        return (size_ + write_pos_.load(std::memory_order_acquire) - read_pos_.load(std::memory_order_relaxed)) % size_;
    }
    size_t emptyCount() const {
        if (!buffer_) return 0;
        return (size_ + read_pos_.load(std::memory_order_acquire) - write_pos_.load(std::memory_order_relaxed) - 1) % size_;
    }
    size_t readableCount() const {
        return fillCount();
    }
    size_t writableCount() const {
        return emptyCount();
    }

protected:
    BufferResult<T> getBufferCommon(const size_t min_count, const size_t max_count, const size_t cur_pos, size_t count) {
        if (count >= min_count) {
            size_t till_end = size_ - cur_pos;
            bool more_available = false;
            if (till_end < count) {
                count = till_end;
                more_available = true;
            }
            if (count > max_count) {
                count = max_count;
                more_available = true;
            }
            return BufferResult<T>(buffer_ + cur_pos, count, more_available);
        } else {
            return nullptr;
        }
    }
public:
    /**
     * Get write buffer
     * \param min_count Require space for at least min_count elements. If it's not available, return empty BufferResult.
     * \param max_count
    */
    BufferResult<T> getWriteBuffer(const size_t min_count = 1, const size_t max_count = SIZE_MAX) {
        return getBufferCommon(min_count, max_count, write_pos_.load(std::memory_order_relaxed), emptyCount());
    }
    
    BufferResult<T> getReadBuffer(const size_t min_count = 1, const size_t max_count = SIZE_MAX) {
        return getBufferCommon(min_count, max_count, read_pos_.load(std::memory_order_relaxed), fillCount());
    }
    
    const T& readableItem(const size_t offset) const {
        return buffer_[ (read_pos_.load(std::memory_order_relaxed) + offset) % size_ ];
    }
    T& readableItem(const size_t offset) {
        return buffer_[ (read_pos_.load(std::memory_order_relaxed) + offset) % size_ ];
    }
    T& writableItem(const size_t offset) {
        return buffer_[ (write_pos_.load(std::memory_order_relaxed) + offset) % size_ ];
    }
    
    void writeDone(const size_t count) {
        size_t newpos = (write_pos_.load(std::memory_order_relaxed) + count) % size_;
        write_pos_.store(newpos, std::memory_order_release);
    }
    void readDone(const size_t count) {
        size_t newpos = (read_pos_.load(std::memory_order_relaxed) + count) % size_;
        read_pos_.store(newpos, std::memory_order_release);
    }
    bool writeFrom(const T* begin, const T* end) {
        size_t count = end - begin;
        auto dst_buffer = getWriteBuffer(count, count);
        if (dst_buffer.isEmpty()) {
            return false;
        }
        std::copy(begin, begin + dst_buffer.count(), dst_buffer.begin());
        writeDone(dst_buffer.count());
        if (dst_buffer.count() != count) {
            size_t remaining = count - dst_buffer.count();
            auto dst_buffer2 = getWriteBuffer(remaining, remaining);
            std::copy(begin + dst_buffer.count(), begin + dst_buffer.count() + dst_buffer2.count(), dst_buffer2.begin());
            writeDone(dst_buffer2.count());
        }
        return true;
    }
    bool fill(const size_t count, const T value) {
        auto dst_buffer = getWriteBuffer(count, count);
        if (dst_buffer.isEmpty()) {
            return false;
        }
        std::fill(dst_buffer.begin(), dst_buffer.end(), value);
        writeDone(dst_buffer.count());
        if (dst_buffer.count() != count) {
            size_t remaining = count - dst_buffer.count();
            auto dst_buffer2 = getWriteBuffer(remaining, remaining);
            std::fill(dst_buffer2.begin(), dst_buffer2.end(), value);
            writeDone(dst_buffer.count());
        }
        return true;
    }
    bool readTo(T* begin, T* end) {
        size_t count = end - begin;
        auto src_buffer = getReadBuffer(count, count);
        if (src_buffer.isEmpty()) {
            return false;
        }
        std::copy(src_buffer.begin(), src_buffer.end(), begin);
        readDone(src_buffer.count());
        if (src_buffer.count() != count) {
            size_t remaining = count - src_buffer.count();
            auto src_buffer2 = getReadBuffer(remaining, remaining);
            std::copy(src_buffer2.begin(), src_buffer2.end(), begin + src_buffer.count());
            readDone(src_buffer2.count());
        }
        return true;
    }
    template<typename Elem> RingBufferView<Elem, T> view() {
        return RingBufferView<Elem, T>(*this);
    }
    ~RingBuffer() {
        delete[] buffer_;
        buffer_ = nullptr;
    }
};

template<typename Elem, typename Underlying /*= uint8_t*/, size_t elem_size /*= sizeof(Elem)*/> class RingBufferView {
protected:
    RingBuffer<Underlying> &rb_;
#define UNDERLYING_TO_VIEW * sizeof(Underlying) / elem_size
#define VIEW_TO_UNDERLYING * elem_size / sizeof(Underlying)
public:
    RingBufferView(RingBuffer<Underlying> &ringbuffer): rb_(ringbuffer) {
    }
    size_t fillCount() const {
        return rb_.fillCount() UNDERLYING_TO_VIEW;
    }
    size_t emptyCount() const {
        return rb_.emptyCount() UNDERLYING_TO_VIEW;
    }
    size_t readableCount() const {
        return fillCount();
    }
    size_t writableCount() const {
        return emptyCount();
    }
    BufferResult<Elem> getWriteBuffer(const size_t min_count, const size_t max_count) {
        return (BufferResult<Elem>)(rb_.getWriteBuffer(min_count VIEW_TO_UNDERLYING, max_count VIEW_TO_UNDERLYING));
    }
    BufferResult<Elem> getWriteBuffer(const size_t min_count = 1) {
        return (BufferResult<Elem>)(rb_.getWriteBuffer(min_count VIEW_TO_UNDERLYING));
    }
    BufferResult<Elem> getReadBuffer(const size_t min_count, const size_t max_count) {
        return (BufferResult<Elem>)(rb_.getReadBuffer(min_count VIEW_TO_UNDERLYING, max_count VIEW_TO_UNDERLYING));
    }
    BufferResult<Elem> getReadBuffer(const size_t min_count = 1) {
        return (BufferResult<Elem>)(rb_.getReadBuffer(min_count VIEW_TO_UNDERLYING));
    }
    
    const Elem& readableItem(const size_t offset) const {
        return *((const Elem*)(&(rb_.readableItem(offset VIEW_TO_UNDERLYING))));
    }
    Elem& readableItem(const size_t offset) {
        return *((Elem*)(&(rb_.readableItem(offset VIEW_TO_UNDERLYING))));
    }
    Elem& writableItem(const size_t offset) {
        return *((Elem*)(&(rb_.writableItem(offset VIEW_TO_UNDERLYING))));
    }
    
    void writeDone(const size_t count) {
        rb_.writeDone(count VIEW_TO_UNDERLYING);
    }
    void readDone(const size_t count) {
        rb_.readDone(count VIEW_TO_UNDERLYING);
    }
#undef UNDERLYING_TO_VIEW
#undef VIEW_TO_UNDERLYING
};