CXX ?= c++
EIGEN_INCLUDE ?= /usr/include/eigen3

CPPFLAGS += -I../include -I$(EIGEN_INCLUDE)
CXXFLAGS += -std=c++17 -O2 -Wall -Wextra

BUILD_DIR := build
TEST_BINARY := $(BUILD_DIR)/test_tracker_iou
TRACKER_SOURCES := \
	../src/BYTETracker.cpp \
	../src/STrack.cpp \
	../src/kalmanFilter.cpp \
	../src/lapjv.cpp \
	../src/utils.cpp

.PHONY: test clean

test: $(TEST_BINARY)
	./$(TEST_BINARY)

$(TEST_BINARY): test_tracker_iou.cpp $(TRACKER_SOURCES)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

clean:
	rm -f $(TEST_BINARY)
