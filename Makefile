
BUILD_TYPE = Debug
HAVE_CUDA = 1
HAVE_VAAPI = 0
HAVE_DRM = 0
# Optional bundled deps/features
# - HAVE_SCTE35 controls whether we build/link libklvanc + libklscte35 and enable the SCTE-35 parser node.
HAVE_SCTE35 = 1
# - HAVE_KAFKA controls whether we build/link librdkafka (+ lz4/zstd) and enable the store_metadata node.
HAVE_KAFKA = 0
# Build NvOFFRUC-based frame interpolation node (requires CUDA + Optical_Flow_SDK_5.0.7 headers at build time,
# and libNvOFFRUC.so available at runtime)
HAVE_NVOF_FRUC ?= 1
# Path to NVIDIA Optical Flow SDK root dir (can be overridden by environment)
OPTICAL_FLOW_SDK_DIR_NAME ?= deps/Optical_Flow_SDK_5.0.7
# HAVE_CUDA does not require any system dependencies, but nvcc does
HAVE_NVCC = 0
HAVE_TENSORRT = 0
# Build neural_net nodes except sport_specific (draw, yolo/rtdetr, preprocess, utils)
NEURAL_NET_COMMON ?= 0
# Build neural_net sport-specific nodes
NEURAL_NET_SPECIFIC ?= 0
TENSORRT_ROOT =
NVCC ?= /usr/local/cuda/bin/nvcc
ifeq ($(HAVE_VAAPI),1)
HAVE_GL = 1
else
HAVE_GL = 0
endif

ifeq ($(BUILD_TYPE),Debug)
OPTIMIZATION_FLAGS = -O0 -ftrapv
else
OPTIMIZATION_FLAGS = -O3 -flto
endif

override CXXFLAGS += -g -rdynamic -fPIC -std=c++17 -Ideps/include -I/usr/local/include -Ideps/cpr/build/cpr_generated_includes $(OPTIMIZATION_FLAGS)
override LFLAGS += -L/usr/local/lib -Wl,-rpath,/usr/local/lib $(OPTIMIZATION_FLAGS)
PYTHON_MODULE_EXTRA_CXXFLAGS = -Ideps/json/single_include -Ideps/pybind11/include -Ideps/pybind11_json/include $(shell python3-config --includes) -fvisibility=hidden
PYTHON_MODULE_EXTRA_LFLAGS = $(shell python3-config --ldflags)
PKG_CONFIG_PATH := /usr/local/lib/pkgconfig$(if PKG_CONFIG_PATH,:)$(PKG_CONFIG_PATH)

BUILD_DATE_FILE = builddate.h
#SRCDIR = $(dir $(firstword $(MAKEFILE_LIST)))src
SRCDIR = src

NODES_SRC = $(shell find $(SRCDIR)/nodes -maxdepth 1 -name '*.cpp')
PYTHON_NODE_SRCS = $(shell find $(SRCDIR)/nodes/python -maxdepth 1 -name '*.cpp')

# Out-of-tree nodes:
# Downstream projects can inject extra node sources via
# EXTRA_NODES_SRC without forking. generate_node_list is path-agnostic, so DECLNODE()
# macros there are picked up automatically. EXTRA_NODES_INCLUDES adds -I flags so the
# extra files can resolve upstream headers like 'node_common.hpp'.
NODES_SRC += $(EXTRA_NODES_SRC)
override CXXFLAGS += $(addprefix -I,$(EXTRA_NODES_INCLUDES))

# Python node sources are needed in the node list/factories only for the python_module goal.
ifneq ($(filter python_module,$(MAKECMDGOALS)),)
NODES_SRC += $(PYTHON_NODE_SRCS)
endif
ifeq ($(NEURAL_NET_SPECIFIC),1)
NODES_SRC += $(shell find $(SRCDIR)/nodes/neural_net/sport_specific -maxdepth 1 -name '*.cpp')
NODES_SRC += $(shell find $(SRCDIR)/nodes/neural_net/sport_specific/metadata_dump -maxdepth 1 -name '*.cpp')
BYTETRACK_SRC = $(wildcard deps/bytetrack/src/*.cpp)
override CXXFLAGS += -I/usr/include/eigen3 -Ideps/bytetrack/include
endif
ifeq ($(NEURAL_NET_COMMON),1)
NODES_SRC += $(SRCDIR)/nodes/neural_net/utils/smooth_crop_viewport.cpp
endif

# hwaccel nodes moved from nodes/cuda to nodes/hwaccel
DRM_PRIME_TO_CUDA_SRC = $(SRCDIR)/nodes/hwaccel/drm_prime_to_cuda.cpp
IPC_CUDA_SOURCE_SRC = $(SRCDIR)/nodes/hwaccel/ipc_cuda_source.cpp
IPC_DMABUF_SOURCE_SRC = $(SRCDIR)/nodes/hwaccel/ipc_dmabuf_source.cpp

ifeq ($(EMBED_IN),obs)
NODES_SRC += $(shell find $(SRCDIR)/nodes/obs -maxdepth 1 -name '*.cpp')
override CXXFLAGS += -DEMBED_IN_OBS=1 -I$(LIBOBS_INCLUDE_DIR) -I$(LIBOBS_INCLUDE_DIR)/../deps/glad/include
endif

ifeq ($(BUILD_TYPE),Debug)
NODES_SRC += $(shell find $(SRCDIR)/nodes/debug -maxdepth 1 -name '*.cpp')
override CXXFLAGS += -DSYNCMETER=1
endif

nodes_list_file = graph_factory.generated.cpp
CPPSRC = avplumber.cpp util.cpp avutils.cpp graph_core.cpp graph_mgmt.cpp stats.cpp output_control.cpp instance_shared.cpp hwaccel_mgmt.cpp EventLoop.cpp TickSource.cpp rest_client.cpp mixer_orchestrator.cpp
DEPS_LIBS = deps/cpr/build/lib/libcpr.a deps/avcpp/build/src/libavcpp.a
# Python extension links via PYTHON_MODULE_EXTRA_LFLAGS (python3-config; -lpython3 is not a valid soname on many distros).
LIBS_FLAGS = -lpthread -lcurl -lssl -lcrypto -lboost_thread -lboost_system -lavcodec -lavfilter -lavutil -lavformat -lavdevice -lswscale -lswresample -ldl -lz

ifeq ($(HAVE_KAFKA),1)
DEPS_LIBS += deps/librdkafka/build/src/librdkafka.a
override CXXFLAGS += -Ideps/librdkafka/src -DHAVE_KAFKA=1
override LIBS_FLAGS += -lzstd -llz4
else
NODES_SRC := $(filter-out $(SRCDIR)/nodes/store_metadata.cpp,$(NODES_SRC))
override CXXFLAGS += -DHAVE_KAFKA=0
endif

ifeq ($(HAVE_SCTE35),1)
DEPS_LIBS += deps/libklscte35/src/.libs/libklscte35.a deps/libklvanc/src/.libs/libklvanc.a
override CXXFLAGS += -DHAVE_SCTE35=1
else
# The SCTE-35 node is the only thing that needs these libs; dropping it avoids building the autotools deps.
NODES_SRC := $(filter-out $(SRCDIR)/nodes/scte35_parse.cpp,$(NODES_SRC))
override CXXFLAGS += -DHAVE_SCTE35=0
endif

ifeq ($(HAVE_JACK),1)
NODES_SRC += $(shell find $(SRCDIR)/nodes/jack -maxdepth 1 -name '*.cpp')
override CXXFLAGS += -DHAVE_JACK=1
override LIBS_FLAGS += -ljack
endif

# PTX kernel build function: compile .cu to .ptx, embed as C header via xxd
# Usage: $(eval $(call ptx_kernel,cu_source,symbol_prefix,dependent_objects))
define ptx_kernel
ALL_PTX_H += objs/$(patsubst %.cu,%.ptx.h,$(1))

objs/$(patsubst %.cu,%.ptx,$(1)): $(1)
	@mkdir -p $$(dir $$@)
	$$(NVCC) -ptx -o $$@ $$<

objs/$(patsubst %.cu,%.ptx.h,$(1)): objs/$(patsubst %.cu,%.ptx,$(1))
	@mkdir -p $$(dir $$@)
	@if [ ! -s $$< ]; then echo "Error: PTX file $$< is empty or missing" >&2; exit 1; fi
	xxd -i $$< | sed -E 's/unsigned int [a-zA-Z0-9_]*_ptx_len/const unsigned int $(2)_len/; s/unsigned char [a-zA-Z0-9_]*_ptx/const char $(2)/' > $$@
	@if [ ! -s $$@ ]; then echo "Error: Generated header $$@ is empty" >&2; exit 1; fi

$(3): objs/$(patsubst %.cu,%.ptx.h,$(1))
endef

ALL_PTX_H =

ifeq ($(HAVE_CUDA)$(HAVE_GL)$(HAVE_NVCC),111)
NODES_SRC += $(SRCDIR)/nodes/hwaccel/cuda_to_egl_image.cpp
$(eval $(call ptx_kernel,$(SRCDIR)/nodes/hwaccel/yuv_to_rgba_surface.cu,avpl_yuv_rgba_ptx,objs/src/nodes/hwaccel/cuda_to_egl_image.o))
endif

ifeq ($(HAVE_CUDA)$(NEURAL_NET_COMMON),11)
NODES_SRC += $(SRCDIR)/nodes/neural_net/draw/cuda_overlay_base.cpp
NODES_SRC += $(SRCDIR)/nodes/neural_net/draw/draw_bbox.cpp
NODES_SRC += $(SRCDIR)/nodes/neural_net/draw/draw_bbox_labels.cpp
NODES_SRC += $(SRCDIR)/nodes/neural_net/draw/draw_segmask.cpp
NODES_SRC += $(SRCDIR)/nodes/neural_net/draw/draw_keypoints.cpp
NODES_SRC += $(SRCDIR)/nodes/neural_net/draw/draw_trail.cpp
NODES_SRC += $(SRCDIR)/nodes/neural_net/draw/draw_tactical_court.cpp
$(eval $(call ptx_kernel,$(SRCDIR)/nodes/neural_net/draw/draw_bbox.cu,avpl_draw_bbox_ptx,objs/src/nodes/neural_net/draw/draw_bbox.o))
$(eval $(call ptx_kernel,$(SRCDIR)/nodes/neural_net/draw/draw_text.cu,avpl_draw_text_ptx,objs/src/nodes/neural_net/draw/draw_text.o objs/src/nodes/neural_net/draw/draw_bbox_labels.o))
$(eval $(call ptx_kernel,$(SRCDIR)/nodes/neural_net/draw/draw_segmask.cu,avpl_draw_segmask_ptx,objs/src/nodes/neural_net/draw/draw_segmask.o))
$(eval $(call ptx_kernel,$(SRCDIR)/nodes/neural_net/draw/draw_keypoints.cu,avpl_draw_keypoints_ptx,objs/src/nodes/neural_net/draw/draw_keypoints.o))
$(eval $(call ptx_kernel,$(SRCDIR)/nodes/neural_net/draw/draw_trail.cu,avpl_draw_trail_ptx,objs/src/nodes/neural_net/draw/draw_trail.o))
$(eval $(call ptx_kernel,$(SRCDIR)/nodes/neural_net/draw/draw_tactical_court.cu,avpl_draw_tactical_court_ptx,objs/src/nodes/neural_net/draw/draw_tactical_court.o))
endif

ifeq ($(HAVE_CUDA)$(NEURAL_NET_COMMON),11)
NODES_SRC += $(SRCDIR)/nodes/neural_net/common/infer_trt_base.cpp
NODES_SRC += $(SRCDIR)/nodes/neural_net/yolo/infer_yolo.cpp
NODES_SRC += $(SRCDIR)/nodes/neural_net/rtdetr/infer_rtdetr.cpp
NODES_SRC += $(SRCDIR)/nodes/neural_net/utils/amagi_reframer.cpp
$(eval $(call ptx_kernel,$(SRCDIR)/nodes/neural_net/preprocess/nv12_to_nchw.cu,avpl_yolo_preprocess_ptx,objs/src/nodes/neural_net/common/infer_trt_base.o objs/src/nodes/neural_net/yolo/infer_yolo.o))
$(eval $(call ptx_kernel,$(SRCDIR)/nodes/neural_net/preprocess/mask_assemble.cu,avpl_yolo_mask_assemble_ptx,objs/src/nodes/neural_net/common/infer_trt_base.o objs/src/nodes/neural_net/yolo/infer_yolo.o))
$(eval $(call ptx_kernel,$(SRCDIR)/nodes/neural_net/utils/amagi_reframer.cu,avpl_reframer_ptx,objs/src/nodes/neural_net/utils/amagi_reframer.o))
endif

ifeq ($(HAVE_CUDA)$(NEURAL_NET_SPECIFIC)$(HAVE_NVCC),111)
$(eval $(call ptx_kernel,$(SRCDIR)/nodes/neural_net/sport_specific/jersey_color_extract.cu,avpl_jersey_uv_mean_ptx,objs/src/nodes/neural_net/sport_specific/jersey_color_extract.o))
$(eval $(call ptx_kernel,$(SRCDIR)/nodes/neural_net/sport_specific/player_feet_seg.cu,avpl_player_feet_seg_ptx,objs/src/nodes/neural_net/sport_specific/player_feet_seg.o))
$(eval $(call ptx_kernel,$(SRCDIR)/nodes/neural_net/sport_specific/player_torso_seg.cu,avpl_player_torso_seg_ptx,objs/src/nodes/neural_net/sport_specific/player_torso_seg.o))
$(eval $(call ptx_kernel,$(SRCDIR)/nodes/neural_net/sport_specific/tracknet_ball_preprocess.cu,avpl_tracknet_ball_preprocess_ptx,objs/src/nodes/neural_net/sport_specific/tracknet_ball.o))
$(eval $(call ptx_kernel,$(SRCDIR)/nodes/neural_net/sport_specific/luma_diff.cu,avpl_luma_diff_ptx,objs/src/nodes/neural_net/sport_specific/luma_diff.o))
$(eval $(call ptx_kernel,$(SRCDIR)/nodes/neural_net/preprocess/nv12_crop_resize_pad.cu,avpl_ocr_crop_ptx,objs/src/nodes/neural_net/sport_specific/scoreboard_ocr.o))
endif

ifeq ($(HAVE_CUDA),1)
NODES_SRC += $(IPC_CUDA_SOURCE_SRC)
NODES_SRC += $(SRCDIR)/nodes/hwaccel/cuda_rect_overlay.cpp
override CPPSRC += cuda.cpp
override CXXFLAGS += -DHAVE_CUDA=1 -Iobjs
override DEPS_LIBS += deps/cuda_loader/cuda_drvapi_dynlink.o
endif

ifeq ($(HAVE_CUDA),1)
ifneq (,$(wildcard $(SRCDIR)/nodes/nvjpeg_enc.cpp))
override LIBS_FLAGS += -lnvjpeg -lcudart
endif
else
NODES_SRC := $(filter-out $(SRCDIR)/nodes/nvjpeg_enc.cpp,$(NODES_SRC))
endif

ifeq ($(NEURAL_NET_COMMON),1)
override CXXFLAGS += -DHAVE_TENSORRT=1
ifneq ($(strip $(TENSORRT_ROOT)),)
override CXXFLAGS += -I$(TENSORRT_ROOT)/include
override LFLAGS += -L$(TENSORRT_ROOT)/lib -Wl,-rpath,$(TENSORRT_ROOT)/lib
endif
override LIBS_FLAGS += -lnvinfer -lnvinfer_plugin
endif

# NvOFFRUC (Frame Rate Up-Conversion) node, built only when headers are present
ifeq ($(HAVE_CUDA)$(HAVE_NVOF_FRUC),11)
ifneq (,$(wildcard $(OPTICAL_FLOW_SDK_DIR_NAME)/NvOFFRUC/Interface/NvOFFRUC.h))
NODES_SRC += $(SRCDIR)/nodes/neural_net/nvof/nvof_fruc.cpp
override CXXFLAGS += -DHAVE_NVOF_FRUC=1 -I$(OPTICAL_FLOW_SDK_DIR_NAME)/NvOFFRUC/Interface
else
override CXXFLAGS += -DHAVE_NVOF_FRUC=0
endif
endif

ifeq ($(HAVE_DRM),1)
NODES_SRC += $(IPC_DMABUF_SOURCE_SRC)
endif

ifeq ($(HAVE_DRM)$(HAVE_GL),11)
NODES_SRC += $(SRCDIR)/nodes/hwaccel/drm_prime_to_egl_image.cpp
endif

# drm_prime_to_cuda requires CUDA + GL + DRM
ifeq ($(HAVE_CUDA)$(HAVE_GL)$(HAVE_DRM),111)
NODES_SRC += $(DRM_PRIME_TO_CUDA_SRC)
endif

ifeq ($(HAVE_VAAPI),1)
override CXXFLAGS += -DHAVE_VAAPI=1
override LIBS_FLAGS += -lva
endif

ifeq ($(HAVE_GL),1)
override CXXFLAGS += -DHAVE_GL=1
override LIBS_FLAGS += -lGL -lEGL -lGLESv2
endif

EXE = avplumber
STATIC_LIBRARY = libavplumber.a
CPPSRC_LIB = $(addprefix src/,$(CPPSRC)) $(nodes_list_file) $(NODES_SRC) $(BYTETRACK_SRC)
CPPSRC_EXE = src/main.cpp $(CPPSRC_LIB)
# Python extension translation units (not linked into the avplumber binary/static library).
CPPSRC_PYTHON = src/avplumber_pybind.cpp
CPPSRC_COMPILE := $(CPPSRC_EXE)
CPPSRC_ALL = $(CPPSRC_COMPILE) $(CPPSRC_PYTHON)

PYTHON_EXT_SUFFIX := $(shell python3-config --extension-suffix 2>/dev/null)
# Must match PYBIND11_MODULE name in src/avplumber_pybind.cpp (avplumber.py imports _avplumber).
PYTHON_MODULE := _avplumber$(PYTHON_EXT_SUFFIX)
# Build python-specific variants for translation units that depend on PYTHON_MODULE.
PYTHON_MODULE_DEFINE_SRCS = src/avplumber.cpp src/graph_mgmt.cpp $(PYTHON_NODE_SRCS)
PYTHON_MODULE_DEFINE_OBJS = $(patsubst src/%.cpp,objs/python/src/%.o,$(PYTHON_MODULE_DEFINE_SRCS))
PYTHON_MODULE_COMMON_OBJS = $(filter-out $(patsubst %.cpp,objs/%.o,$(PYTHON_MODULE_DEFINE_SRCS)),$(patsubst %.cpp,objs/%.o,$(CPPSRC_LIB)))
PYTHON_MODULE_OBJS = $(PYTHON_MODULE_COMMON_OBJS) $(PYTHON_MODULE_DEFINE_OBJS) objs/src/app_version.o $(patsubst %.cpp,objs/%.o,$(CPPSRC_PYTHON))

DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td
DEPDIR := objs
POSTCOMPILE = @mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d && touch $@

.PHONY: builddate build static_library python_module install clean
.DEFAULT_GOAL := build

builddate:
	( /bin/date '+#define COMPILE_DATE "%Y-%m-%d %H:%M:%S %z"' && \
	echo '#define GIT_VERSION "$(shell git describe --abbrev --dirty --always --tags)"' ) > $(BUILD_DATE_FILE)


$(BUILD_DATE_FILE): builddate

$(patsubst %.cpp,objs/%.o,$(CPPSRC_PYTHON)): objs/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(PYTHON_MODULE_EXTRA_CXXFLAGS) -DPYTHON_MODULE $(DEPFLAGS) -c -o $@ $<
	$(POSTCOMPILE)

$(patsubst %.cpp,objs/%.o,$(CPPSRC_COMPILE)): objs/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c -o $@ $<
	$(POSTCOMPILE)

$(PYTHON_MODULE_DEFINE_OBJS): objs/python/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(PYTHON_MODULE_EXTRA_CXXFLAGS) -DPYTHON_MODULE -c -o $@ $<

objs/src/app_version.o: src/app_version.cpp builddate $(BUILD_DATE_FILE)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c -o $@ $< -include $(BUILD_DATE_FILE)


$(nodes_list_file): ./generate_node_list Makefile src/edge_types.hpp $(NODES_SRC)
	./generate_node_list $(NODES_SRC) > $(nodes_list_file)

$(EXE): $(patsubst %.cpp,objs/%.o,$(CPPSRC_EXE)) objs/src/app_version.o $(DEPS_LIBS) $(ALL_PTX_H)
	$(CXX) $(CXXFLAGS) $(LFLAGS) -o $@ $^ $(LIBS_FLAGS)

build: $(EXE) compile_flags.txt


$(STATIC_LIBRARY): $(patsubst %.cpp,objs/%.o,$(CPPSRC_LIB)) objs/src/app_version.o $(DEPS_LIBS) $(ALL_PTX_H)
	ar -rcs $@ $^

static_library: $(STATIC_LIBRARY)

$(PYTHON_MODULE): $(PYTHON_MODULE_OBJS) $(DEPS_LIBS) $(ALL_PTX_H)
	$(CXX) $(CXXFLAGS) $(LFLAGS) $(PYTHON_MODULE_EXTRA_LFLAGS) -shared -o $@ $(PYTHON_MODULE_OBJS) $(DEPS_LIBS) $(LIBS_FLAGS)

python_module: $(PYTHON_MODULE)

install: build
	mkdir -p "$(DESTDIR)/apps/tools"
	cp "$(EXE)" "$(DESTDIR)/apps/tools/"

clean:
	rm $(EXE) $(STATIC_LIBRARY) $(PYTHON_MODULE) $(BUILD_DATE_FILE) $(nodes_list_file) compile_flags.txt || true
	rm -r objs || true

clean_deps:
	rm -r deps/cpr/build || true
	rm -r deps/avcpp/build || true
	rm deps/cuda_loader/*.o || true
	cd deps/libklvanc && git clean -xdf || true
	cd deps/libklscte35 && git clean -xdf || true
	rm -rf deps/librdkafka/build || true

deps/cpr/build/lib/libcpr.a:
	mkdir -p deps/cpr/build
	cd deps/cpr/build && cmake -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DCMAKE_CXX_FLAGS="$(CXXFLAGS)" -DCMAKE_EXE_LINKER_FLAGS="$(LFLAGS)" -DUSE_SYSTEM_CURL=ON -DCPR_FORCE_USE_SYSTEM_CURL=ON -DBUILD_CPR_TESTS=OFF -DCPR_BUILD_TESTS=OFF -DCPR_BUILD_TESTS_SSL=OFF -DCMAKE_AR=`which gcc-ar` -DCMAKE_RANLIB=`which gcc-ranlib` .. && make VERBOSE=1

deps/avcpp/build/src/libavcpp.a:
	rm -r deps/avcpp/build || true
	mkdir -p deps/avcpp/build
	cd deps/avcpp/build && PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DCMAKE_CXX_FLAGS="$(CXXFLAGS)" -DCMAKE_EXE_LINKER_FLAGS="$(LFLAGS)" -DCMAKE_AR=`which gcc-ar` -DCMAKE_RANLIB=`which gcc-ranlib` ..
	$(MAKE) -C deps/avcpp/build avcpp-static VERBOSE=1

deps/libklvanc/src/.libs/libklvanc.a:
	cd deps/libklvanc && git clean -xdf || true
	cd deps/libklvanc && ./autogen.sh --build && ./configure --enable-shared=no --enable-static && make

deps/libklscte35/src/.libs/libklscte35.a: deps/libklvanc/src/.libs/libklvanc.a
	cd deps/libklscte35 && git clean -xdf || true
	export CFLAGS="-I$(shell readlink -f deps/include)" && export LDFLAGS="-L$(shell readlink -f deps/libklvanc/src/.libs)" && cd deps/libklscte35 && ./autogen.sh --build && ./configure --enable-shared=no --libdir=$(shell readlink -f deps/libklvanc/src/.libs) && make

deps/librdkafka/build/src/librdkafka.a:
	mkdir -p deps/librdkafka/build
	cd deps/librdkafka/build && cmake \
		-DBUILD_SHARED_LIBS=OFF \
		-DRDKAFKA_BUILD_STATIC=ON \
		-DRDKAFKA_BUILD_TESTS=OFF \
		-DRDKAFKA_BUILD_EXAMPLES=OFF \
		-DWITH_SASL=OFF \
		-DWITH_SSL=ON \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_AR=`which gcc-ar` \
		-DCMAKE_RANLIB=`which gcc-ranlib` \
		.. && $(MAKE) rdkafka VERBOSE=1

ifeq ($(HAVE_KAFKA),1)
# store_metadata.cpp needs librdkafka headers
objs/src/nodes/store_metadata.o: deps/librdkafka/build/src/librdkafka.a
endif

deps/cuda_loader/cuda_drvapi_dynlink.o: deps/cuda_loader/cuda_drvapi_dynlink.c
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# PTX build rules are generated by ptx_kernel calls above

compile_flags.txt:
	echo "$(CXXFLAGS)" | tr ' ' '\n' > $@

# anything that requires cpr headers must be compiled after cpr is configured
objs/src/rest_client.o: deps/cpr/build/lib/libcpr.a

.PRECIOUS: objs/%.d

include $(wildcard $(patsubst %.cpp,objs/%.d,$(CPPSRC_COMPILE) $(CPPSRC_PYTHON)))
