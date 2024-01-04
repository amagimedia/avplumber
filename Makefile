
BUILD_TYPE = Debug
HAVE_CUDA = 1

ifeq ($(BUILD_TYPE),Debug)
OPTIMIZATION_FLAGS = -O0 -ftrapv
else
OPTIMIZATION_FLAGS = -O3 -flto
endif

override CXXFLAGS += -g -rdynamic -fPIC -std=c++17 -Ideps/include -I/apps/ffmpeg/include -Ideps/cpr/build/cpr_generated_includes $(OPTIMIZATION_FLAGS)
override LFLAGS += -L/apps/ffmpeg/lib -Wl,-rpath,/apps/ffmpeg/lib $(OPTIMIZATION_FLAGS)
PKG_CONFIG_PATH := /apps/ffmpeg/lib/pkgconfig$(if PKG_CONFIG_PATH,:)$(PKG_CONFIG_PATH)

BUILD_DATE_FILE = builddate.h
#SRCDIR = $(dir $(firstword $(MAKEFILE_LIST)))src
SRCDIR = src

nodes_SRC = $(shell find $(SRCDIR)/nodes -maxdepth 1 -name '*.cpp')

ifeq ($(EMBED_IN),obs)
nodes_SRC += $(shell find $(SRCDIR)/nodes/obs -maxdepth 1 -name '*.cpp')
override CXXFLAGS += -DEMBED_IN_OBS=1 -I$(LIBOBS_INCLUDE_DIR) -I$(LIBOBS_INCLUDE_DIR)/../deps/glad/include
endif

ifeq ($(BUILD_TYPE),Debug)
nodes_SRC += $(shell find $(SRCDIR)/nodes/debug -maxdepth 1 -name '*.cpp')
override CXXFLAGS += -DSYNCMETER=1
endif

nodes_list_file = graph_factory.generated.cpp
CPPSRC = avplumber.cpp util.cpp avutils.cpp graph_core.cpp graph_mgmt.cpp stats.cpp output_control.cpp instance_shared.cpp hwaccel_mgmt.cpp EventLoop.cpp TickSource.cpp
DEPS_LIBS = deps/cpr/build/lib/libcpr.a deps/avcpp/build/src/libavcpp.a
LIBS_FLAGS = -lpthread -lcurl -lssl -lcrypto -lboost_thread -lboost_system -lavcodec -lavfilter -lavutil -lavformat -lavdevice -lswscale -lswresample -ldl

ifeq ($(HAVE_CUDA),1)
nodes_SRC += $(shell find $(SRCDIR)/nodes/cuda -maxdepth 1 -name '*.cpp')
override CPPSRC += cuda.cpp
override CXXFLAGS += -DHAVE_CUDA=1
override DEPS_LIBS += deps/cuda_loader/cuda_drvapi_dynlink.o
endif

EXE = avplumber
STATIC_LIBRARY = libavplumber.a
CPPSRC_LIB = $(addprefix src/,$(CPPSRC)) $(nodes_list_file) $(nodes_SRC)
CPPSRC_EXE = src/main.cpp $(CPPSRC_LIB)
CPPSRC_ALL = $(CPPSRC_EXE)

DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td
DEPDIR := objs
POSTCOMPILE = @mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d && touch $@

.PHONY: builddate build static_library install clean
.DEFAULT_GOAL := build

builddate:
	( /bin/date '+#define COMPILE_DATE "%Y-%m-%d %H:%M:%S %z"' && \
	echo '#define GIT_VERSION "$(shell git describe --abbrev --dirty --always --tags)"' ) > $(BUILD_DATE_FILE)


$(BUILD_DATE_FILE): builddate

$(patsubst %.cpp,objs/%.o,$(CPPSRC_EXE)): objs/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c -o $@ $<
	$(POSTCOMPILE)

objs/src/app_version.o: src/app_version.cpp builddate $(BUILD_DATE_FILE)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c -o $@ $< -include $(BUILD_DATE_FILE)


$(nodes_list_file): ./generate_node_list $(nodes_SRC)
	./generate_node_list $(nodes_SRC) > $(nodes_list_file)

$(EXE): $(patsubst %.cpp,objs/%.o,$(CPPSRC_EXE)) objs/src/app_version.o $(DEPS_LIBS)
	$(CXX) $(CXXFLAGS) $(LFLAGS) -o $@ $^ $(LIBS_FLAGS)

build: $(EXE) compile_flags.txt

$(STATIC_LIBRARY): $(patsubst %.cpp,objs/%.o,$(CPPSRC_LIB)) objs/src/app_version.o $(DEPS_LIBS)
	ar -rcs $@ $^

static_library: $(STATIC_LIBRARY)

install: build
	mkdir -p "$(DESTDIR)/apps/tools"
	cp "$(EXE)" "$(DESTDIR)/apps/tools/"

clean:
	rm $(EXE) $(STATIC_LIBRARY) $(BUILD_DATE_FILE) $(nodes_list_file) compile_flags.txt || true
	rm -r objs || true

clean_deps:
	rm -r deps/cpr/build || true
	rm -r deps/avcpp/build || true

deps/cpr/build/lib/libcpr.a:
	mkdir -p deps/cpr/build
	cd deps/cpr/build && cmake -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DCMAKE_CXX_FLAGS="$(CXXFLAGS)" -DCMAKE_EXE_LINKER_FLAGS="$(LFLAGS)" -DUSE_SYSTEM_CURL=ON -DCPR_FORCE_USE_SYSTEM_CURL=ON -DBUILD_CPR_TESTS=OFF -DCPR_BUILD_TESTS=OFF -DCPR_BUILD_TESTS_SSL=OFF -DCMAKE_AR=`which gcc-ar` -DCMAKE_RANLIB=`which gcc-ranlib` .. && make VERBOSE=1

deps/avcpp/build/src/libavcpp.a:
	rm -r deps/avcpp/build || true
	mkdir -p deps/avcpp/build
	cd deps/avcpp/build && PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DCMAKE_CXX_FLAGS="$(CXXFLAGS)" -DCMAKE_EXE_LINKER_FLAGS="$(LFLAGS)" -DCMAKE_AR=`which gcc-ar` -DCMAKE_RANLIB=`which gcc-ranlib` ..
	$(MAKE) -C deps/avcpp/build avcpp-static VERBOSE=1

deps/cuda_loader/cuda_drvapi_dynlink.o: deps/cuda_loader/cuda_drvapi_dynlink.c
	$(CXX) $(CXXFLAGS) -c -o $@ $<

compile_flags.txt:
	echo "$(CXXFLAGS)" | tr ' ' '\n' > $@

# anything that requires cpr headers must be compiled after cpr is configured
objs/src/nodes/sentinel.o: deps/cpr/build/lib/libcpr.a
objs/src/stats.o: deps/cpr/build/lib/libcpr.a

.PRECIOUS: objs/%.d

include $(wildcard $(patsubst %.cpp,objs/%.d,$(CPPSRC_ALL)))
