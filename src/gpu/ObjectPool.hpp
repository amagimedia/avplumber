#pragma once
#include <vector>
#include <queue>
#include <functional>
#include <mutex>
#include <optional>
#include <memory>
#include <cstddef>
#
// Simple generic object pool with externalized create/destroy.
// - reset(size, create, destroy) reinitializes the pool
// - acquire() returns index of an available entry, or std::nullopt if none
// - release(index) returns an entry to the free list
template <typename Entry>
class ObjectPool {
private:
	std::vector<Entry> entries_;
	std::queue<size_t> free_indices_;
	std::mutex mtx_;
public:
	using CreateFn = std::function<bool(Entry&)>;
	using DestroyFn = std::function<void(Entry&)>;
	// Destroys previous entries using destroy() and re-creates size entries using create()
	bool reset(size_t size, CreateFn create, DestroyFn destroy) {
		std::unique_lock<std::mutex> lock(mtx_);
		// destroy old
		for (auto &e : entries_) {
			destroy(e);
		}
		entries_.clear();
		while (!free_indices_.empty()) free_indices_.pop();
		// create new
		entries_.resize(size);
		for (size_t i = 0; i < size; ++i) {
			if (!create(entries_[i])) {
				// cleanup if failure
				for (size_t j = 0; j <= i; ++j) destroy(entries_[j]);
				entries_.clear();
				return false;
			}
			free_indices_.push(i);
		}
		return true;
	}
	size_t size() const {
		return entries_.size();
	}
	std::optional<size_t> acquire() {
		std::unique_lock<std::mutex> lock(mtx_);
		if (free_indices_.empty()) return std::nullopt;
		size_t idx = free_indices_.front();
		free_indices_.pop();
		return idx;
	}
	void release(size_t idx) {
		std::unique_lock<std::mutex> lock(mtx_);
		free_indices_.push(idx);
	}
	Entry& entry(size_t idx) {
		return entries_[idx];
	}
	const Entry& entry(size_t idx) const {
		return entries_[idx];
	}
};



