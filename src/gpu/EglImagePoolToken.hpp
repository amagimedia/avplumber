#pragma once
#include <functional>
#
// Lightweight token passed along with EglImageFrame so that consumers
// can return the underlying image back to its pool by calling release().
struct EglImagePoolToken {
	std::function<void()> release;
};



