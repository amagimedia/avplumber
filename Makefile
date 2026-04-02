
BUILD_TYPE = Debug
HAVE_CUDA = 1
HAVE_VAAPI = 0
HAVE_DRM = 0
# Optional bundled deps/features
# - HAVE_SCTE35 controls whether we build/link libklvanc + libklscte35 and enable the SCTE-35 parser node.
HAVE_SCTE35 = 1
# Build NvOFFRUC-based frame interpolation node (requires CUDA + Optical_Flow_SDK_5.0.7 headers at build time,
# and libNvOFFRUC.so available at runtime)
HAVE_NVOF_FRUC ?= 1
# Path to NVIDIA Optical Flow SDK root dir (can be overridden by environment)
OPTICAL_FLOW_SDK_DIR_NAME ?= deps/Optical_Flow_SDK_5.0.7
# HAVE_CUDA does not require any system dependencies, but nvcc does
HAVE_NVCC = 0
HAVE_TENSORRT = 0
TENSORRT_ROOT =
ifeq ($(HAVE_NVCC),1)
NVCC ?= /usr/local/cuda/bin/nvcc
endif
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

override CXXFLAGS += -g -rdynamic -fPIC -std=c++17 -Ideps/include -I/usr/include/ffmpeg -I/apps/ffmpeg/include -Ideps/cpr/build/cpr_generated_includes -Ideps/pybind11/include -Ideps/pybind11_json/include `python3-config --includes` $(OPTIMIZATION_FLAGS)
override LFLAGS += -L/apps/ffmpeg/lib -Wl,-rpath,/apps/ffmpeg/lib `python3-config --ldflags` $(OPTIMIZATION_FLAGS)
PKG_CONFIG_PATH := /apps/ffmpeg/lib/pkgconfig$(if PKG_CONFIG_PATH,:)$(PKG_CONFIG_PATH)

BUILD_DATE_FILE = builddate.h
#SRCDIR = $(dir $(firstword $(MAKEFILE_LIST)))src
SRCDIR = src

NODES_SRC = $(shell find $(SRCDIR)/nodes -maxdepth 1 -name '*.cpp')

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
CPPSRC = avplumber.cpp util.cpp avutils.cpp graph_core.cpp graph_core_pybind.cpp graph_mgmt.cpp stats.cpp output_control.cpp instance_shared.cpp hwaccel_mgmt.cpp EventLoop.cpp TickSource.cpp rest_client.cpp
DEPS_LIBS = deps/cpr/build/lib/libcpr.a deps/avcpp/build/src/libavcpp.a
# Python is linked via python3-config in LFLAGS (-lpython3 is not a valid soname on many distros).
LIBS_FLAGS = -lpthread -lcurl -lssl -lcrypto -lboost_thread -lboost_system -lavcodec -lavfilter -lavutil -lavformat -lavdevice -lswscale -lswresample -ldl

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

ifeq ($(HAVE_CUDA)$(HAVE_GL)$(HAVE_NVCC),111)
NODES_SRC += $(SRCDIR)/nodes/hwaccel/cuda_to_egl_image.cpp
# Build PTX and embed as header for driver-side kernel launch (no cudart)
BUILD_PTX = 1
CUDA_KERNEL = $(SRCDIR)/nodes/hwaccel/yuv_to_rgba_surface.cu
PTX = objs/$(SRCDIR)/nodes/hwaccel/yuv_to_rgba_surface.ptx
PTX_H = objs/$(SRCDIR)/nodes/hwaccel/yuv_to_rgba_surface.ptx.h
else
BUILD_PTX = 0
endif

ifeq ($(HAVE_CUDA)$(HAVE_NVCC),11)
NODES_SRC += $(SRCDIR)/nodes/hwaccel/draw_bbox.cpp
NODES_SRC += $(SRCDIR)/nodes/hwaccel/draw_text.cpp
NODES_SRC += $(SRCDIR)/nodes/hwaccel/draw_segmask.cpp
NODES_SRC += $(SRCDIR)/nodes/hwaccel/draw_keypoints.cpp
BUILD_DRAW_BBOX_PTX = 1
DRAW_BBOX_KERNEL = $(SRCDIR)/nodes/hwaccel/draw_bbox.cu
DRAW_BBOX_PTX = objs/$(SRCDIR)/nodes/hwaccel/draw_bbox.ptx
DRAW_BBOX_PTX_H = objs/$(SRCDIR)/nodes/hwaccel/draw_bbox.ptx.h
BUILD_DRAW_TEXT_PTX = 1
DRAW_TEXT_KERNEL = $(SRCDIR)/nodes/hwaccel/draw_text.cu
DRAW_TEXT_PTX = objs/$(SRCDIR)/nodes/hwaccel/draw_text.ptx
DRAW_TEXT_PTX_H = objs/$(SRCDIR)/nodes/hwaccel/draw_text.ptx.h
BUILD_DRAW_SEGMASK_PTX = 1
DRAW_SEGMASK_KERNEL = $(SRCDIR)/nodes/hwaccel/draw_segmask.cu
DRAW_SEGMASK_PTX = objs/$(SRCDIR)/nodes/hwaccel/draw_segmask.ptx
DRAW_SEGMASK_PTX_H = objs/$(SRCDIR)/nodes/hwaccel/draw_segmask.ptx.h
BUILD_DRAW_KEYPOINTS_PTX = 1
DRAW_KEYPOINTS_KERNEL = $(SRCDIR)/nodes/hwaccel/draw_keypoints.cu
DRAW_KEYPOINTS_PTX = objs/$(SRCDIR)/nodes/hwaccel/draw_keypoints.ptx
DRAW_KEYPOINTS_PTX_H = objs/$(SRCDIR)/nodes/hwaccel/draw_keypoints.ptx.h
else
BUILD_DRAW_BBOX_PTX = 0
BUILD_DRAW_TEXT_PTX = 0
BUILD_DRAW_SEGMASK_PTX = 0
BUILD_DRAW_KEYPOINTS_PTX = 0
endif

ifeq ($(HAVE_CUDA)$(HAVE_TENSORRT)$(HAVE_NVCC),111)
NODES_SRC += $(SRCDIR)/nodes/hwaccel/cuda_infer_yolo_base.cpp
NODES_SRC += $(SRCDIR)/nodes/hwaccel/cuda_infer_yolo.cpp
NODES_SRC += $(SRCDIR)/nodes/hwaccel/vert_infer.cpp
NODES_SRC += $(SRCDIR)/nodes/hwaccel/reframer.cpp
BUILD_YOLO_PTX = 1
YOLO_PREPROCESS_KERNEL = $(SRCDIR)/nodes/hwaccel/yolo_preprocess.cu
YOLO_PREPROCESS_PTX = objs/$(SRCDIR)/nodes/hwaccel/yolo_preprocess.ptx
YOLO_PREPROCESS_PTX_H = objs/$(SRCDIR)/nodes/hwaccel/yolo_preprocess.ptx.h
YOLO_MASK_ASSEMBLE_KERNEL = $(SRCDIR)/nodes/hwaccel/yolo_mask_assemble.cu
YOLO_MASK_ASSEMBLE_PTX = objs/$(SRCDIR)/nodes/hwaccel/yolo_mask_assemble.ptx
YOLO_MASK_ASSEMBLE_PTX_H = objs/$(SRCDIR)/nodes/hwaccel/yolo_mask_assemble.ptx.h
BUILD_VERT_PTX = 1
VERT_PREPROCESS_KERNEL = $(SRCDIR)/nodes/hwaccel/vert_preprocess.cu
VERT_PREPROCESS_PTX = objs/$(SRCDIR)/nodes/hwaccel/vert_preprocess.ptx
VERT_PREPROCESS_PTX_H = objs/$(SRCDIR)/nodes/hwaccel/vert_preprocess.ptx.h
BUILD_REFRAMER_PTX = 1
REFRAMER_PREPROCESS_KERNEL = $(SRCDIR)/nodes/hwaccel/reframer.cu
REFRAMER_PREPROCESS_PTX = objs/$(SRCDIR)/nodes/hwaccel/reframer.ptx
REFRAMER_PREPROCESS_PTX_H = objs/$(SRCDIR)/nodes/hwaccel/reframer.ptx.h
else
BUILD_YOLO_PTX = 0
BUILD_VERT_PTX = 0
BUILD_REFRAMER_PTX = 0
endif

ifeq ($(HAVE_CUDA),1)
NODES_SRC += $(IPC_CUDA_SOURCE_SRC)
override CPPSRC += cuda.cpp
override CXXFLAGS += -DHAVE_CUDA=1 -Iobjs
override DEPS_LIBS += deps/cuda_loader/cuda_drvapi_dynlink.o
endif

ifeq ($(HAVE_TENSORRT),1)
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
NODES_SRC += $(SRCDIR)/nodes/hwaccel/nvof_fruc.cpp
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
CPPSRC_LIB = $(addprefix src/,$(CPPSRC)) $(nodes_list_file) $(NODES_SRC)
CPPSRC_EXE = src/main.cpp $(CPPSRC_LIB)
# Python extension: single TU with PYBIND11_MODULE (not linked into the avplumber binary).
CPPSRC_PYTHON = src/avplumber_pybind.cpp
CPPSRC_COMPILE := $(CPPSRC_EXE) $(CPPSRC_PYTHON)
CPPSRC_ALL = $(CPPSRC_COMPILE)

PYTHON_EXT_SUFFIX := $(shell python3-config --extension-suffix 2>/dev/null)
# Must match PYBIND11_MODULE name in src/avplumber_pybind.cpp (avplumber.py imports _avplumber).
PYTHON_MODULE := _avplumber$(PYTHON_EXT_SUFFIX)
PYTHON_MODULE_OBJS = $(patsubst %.cpp,objs/%.o,$(CPPSRC_LIB)) objs/src/app_version.o objs/src/avplumber_pybind.o

DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td
DEPDIR := objs
POSTCOMPILE = @mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d && touch $@

.PHONY: builddate build static_library python_module install clean
.DEFAULT_GOAL := build

builddate:
	( /bin/date '+#define COMPILE_DATE "%Y-%m-%d %H:%M:%S %z"' && \
	echo '#define GIT_VERSION "$(shell git describe --abbrev --dirty --always --tags)"' ) > $(BUILD_DATE_FILE)


$(BUILD_DATE_FILE): builddate

$(patsubst %.cpp,objs/%.o,$(CPPSRC_COMPILE)): objs/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c -o $@ $<
	$(POSTCOMPILE)

objs/src/app_version.o: src/app_version.cpp builddate $(BUILD_DATE_FILE)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c -o $@ $< -include $(BUILD_DATE_FILE)


$(nodes_list_file): ./generate_node_list Makefile src/edge_types.hpp $(NODES_SRC)
	./generate_node_list $(NODES_SRC) > $(nodes_list_file)

$(EXE): $(patsubst %.cpp,objs/%.o,$(CPPSRC_EXE)) objs/src/app_version.o $(DEPS_LIBS) $(PTX_H) $(DRAW_BBOX_PTX_H) $(DRAW_TEXT_PTX_H) $(YOLO_PREPROCESS_PTX_H) $(YOLO_MASK_ASSEMBLE_PTX_H) $(VERT_PREPROCESS_PTX_H) $(REFRAMER_PREPROCESS_PTX_H)
	$(CXX) $(CXXFLAGS) $(LFLAGS) -o $@ $^ $(LIBS_FLAGS)

build: $(EXE) compile_flags.txt


$(STATIC_LIBRARY): $(patsubst %.cpp,objs/%.o,$(CPPSRC_LIB)) objs/src/app_version.o $(DEPS_LIBS) $(PTX_H) $(DRAW_BBOX_PTX_H) $(DRAW_TEXT_PTX_H) $(YOLO_PREPROCESS_PTX_H) $(YOLO_MASK_ASSEMBLE_PTX_H) $(VERT_PREPROCESS_PTX_H) $(REFRAMER_PREPROCESS_PTX_H)
	ar -rcs $@ $^

static_library: $(STATIC_LIBRARY)

# Import as: PYTHONPATH=. python3 -c "import avplumber"  (same dir as the .so)
$(PYTHON_MODULE): $(PYTHON_MODULE_OBJS) $(DEPS_LIBS) $(PTX_H)
	$(CXX) $(CXXFLAGS) $(LFLAGS) -shared -o $@ $(PYTHON_MODULE_OBJS) $(DEPS_LIBS) $(LIBS_FLAGS)

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
	cd deps/libklvanc && git clean -xdf || true
	cd deps/libklscte35 && git clean -xdf || true

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

deps/cuda_loader/cuda_drvapi_dynlink.o: deps/cuda_loader/cuda_drvapi_dynlink.c
	$(CXX) $(CXXFLAGS) -c -o $@ $<

ifeq ($(BUILD_PTX),1)
# Build PTX once and turn it into a C header array
$(PTX): $(CUDA_KERNEL)
	@mkdir -p $(dir $@)
	$(NVCC) -ptx -o $@ $<

$(PTX_H): $(PTX)
	@mkdir -p $(dir $@)
	@if [ ! -s $< ]; then echo "Error: PTX file $< is empty or missing" >&2; exit 1; fi
	xxd -i $< | sed -E 's/unsigned int objs_src_nodes_hwaccel_yuv_to_rgba_surface_ptx_len/const unsigned int avpl_yuv_rgba_ptx_len/; s/unsigned char objs_src_nodes_hwaccel_yuv_to_rgba_surface_ptx/const char avpl_yuv_rgba_ptx/' > $@
	@if [ ! -s $@ ]; then echo "Error: Generated header $@ is empty. Check PTX file: $<" >&2; exit 1; fi

# Ensure the sink object rebuilds if the generated header changes
objs/src/nodes/hwaccel/cuda_to_egl_image.o: $(PTX_H)
endif

ifeq ($(BUILD_DRAW_BBOX_PTX),1)
$(DRAW_BBOX_PTX): $(DRAW_BBOX_KERNEL)
	@mkdir -p $(dir $@)
	$(NVCC) -ptx -o $@ $<

$(DRAW_BBOX_PTX_H): $(DRAW_BBOX_PTX)
	@mkdir -p $(dir $@)
	@if [ ! -s $< ]; then echo "Error: PTX file $< is empty or missing" >&2; exit 1; fi
	xxd -i $< | sed -E 's/unsigned int objs_src_nodes_hwaccel_draw_bbox_ptx_len/const unsigned int avpl_draw_bbox_ptx_len/; s/unsigned char objs_src_nodes_hwaccel_draw_bbox_ptx/const char avpl_draw_bbox_ptx/' > $@
	@if [ ! -s $@ ]; then echo "Error: Generated header $@ is empty. Check PTX file: $<" >&2; exit 1; fi

objs/src/nodes/hwaccel/draw_bbox.o: $(DRAW_BBOX_PTX_H)
endif

ifeq ($(BUILD_DRAW_TEXT_PTX),1)
$(DRAW_TEXT_PTX): $(DRAW_TEXT_KERNEL)
	@mkdir -p $(dir $@)
	$(NVCC) -ptx -o $@ $<

$(DRAW_TEXT_PTX_H): $(DRAW_TEXT_PTX)
	@mkdir -p $(dir $@)
	@if [ ! -s $< ]; then echo "Error: PTX file $< is empty or missing" >&2; exit 1; fi
	xxd -i $< | sed -E 's/unsigned int objs_src_nodes_hwaccel_draw_text_ptx_len/const unsigned int avpl_draw_text_ptx_len/; s/unsigned char objs_src_nodes_hwaccel_draw_text_ptx/const char avpl_draw_text_ptx/' > $@
	@if [ ! -s $@ ]; then echo "Error: Generated header $@ is empty. Check PTX file: $<" >&2; exit 1; fi

objs/src/nodes/hwaccel/draw_text.o: $(DRAW_TEXT_PTX_H)
endif

ifeq ($(BUILD_DRAW_SEGMASK_PTX),1)
$(DRAW_SEGMASK_PTX): $(DRAW_SEGMASK_KERNEL)
	@mkdir -p $(dir $@)
	$(NVCC) -ptx -o $@ $<

$(DRAW_SEGMASK_PTX_H): $(DRAW_SEGMASK_PTX)
	@mkdir -p $(dir $@)
	@if [ ! -s $< ]; then echo "Error: PTX file $< is empty or missing" >&2; exit 1; fi
	xxd -i $< | sed -E 's/unsigned int objs_src_nodes_hwaccel_draw_segmask_ptx_len/const unsigned int avpl_draw_segmask_ptx_len/; s/unsigned char objs_src_nodes_hwaccel_draw_segmask_ptx/const char avpl_draw_segmask_ptx/' > $@
	@if [ ! -s $@ ]; then echo "Error: Generated header $@ is empty. Check PTX file: $<" >&2; exit 1; fi

objs/src/nodes/hwaccel/draw_segmask.o: $(DRAW_SEGMASK_PTX_H)
endif

ifeq ($(BUILD_DRAW_KEYPOINTS_PTX),1)
$(DRAW_KEYPOINTS_PTX): $(DRAW_KEYPOINTS_KERNEL)
	@mkdir -p $(dir $@)
	$(NVCC) -ptx -o $@ $<

$(DRAW_KEYPOINTS_PTX_H): $(DRAW_KEYPOINTS_PTX)
	@mkdir -p $(dir $@)
	@if [ ! -s $< ]; then echo "Error: PTX file $< is empty or missing" >&2; exit 1; fi
	xxd -i $< | sed -E 's/unsigned int objs_src_nodes_hwaccel_draw_keypoints_ptx_len/const unsigned int avpl_draw_keypoints_ptx_len/; s/unsigned char objs_src_nodes_hwaccel_draw_keypoints_ptx/const char avpl_draw_keypoints_ptx/' > $@
	@if [ ! -s $@ ]; then echo "Error: Generated header $@ is empty. Check PTX file: $<" >&2; exit 1; fi

objs/src/nodes/hwaccel/draw_keypoints.o: $(DRAW_KEYPOINTS_PTX_H)
endif

ifeq ($(BUILD_YOLO_PTX),1)
$(YOLO_PREPROCESS_PTX): $(YOLO_PREPROCESS_KERNEL)
	@mkdir -p $(dir $@)
	$(NVCC) -ptx -o $@ $<

$(YOLO_PREPROCESS_PTX_H): $(YOLO_PREPROCESS_PTX)
	@mkdir -p $(dir $@)
	@if [ ! -s $< ]; then echo "Error: PTX file $< is empty or missing" >&2; exit 1; fi
	xxd -i $< | sed -E 's/unsigned int objs_src_nodes_hwaccel_yolo_preprocess_ptx_len/const unsigned int avpl_yolo_preprocess_ptx_len/; s/unsigned char objs_src_nodes_hwaccel_yolo_preprocess_ptx/const char avpl_yolo_preprocess_ptx/' > $@
	@if [ ! -s $@ ]; then echo "Error: Generated header $@ is empty. Check PTX file: $<" >&2; exit 1; fi

objs/src/nodes/hwaccel/cuda_infer_yolo_base.o: $(YOLO_PREPROCESS_PTX_H) $(YOLO_MASK_ASSEMBLE_PTX_H)
objs/src/nodes/hwaccel/cuda_infer_yolo.o: $(YOLO_PREPROCESS_PTX_H) $(YOLO_MASK_ASSEMBLE_PTX_H)

$(YOLO_MASK_ASSEMBLE_PTX): $(YOLO_MASK_ASSEMBLE_KERNEL)
	@mkdir -p $(dir $@)
	$(NVCC) -ptx -o $@ $<

$(YOLO_MASK_ASSEMBLE_PTX_H): $(YOLO_MASK_ASSEMBLE_PTX)
	@mkdir -p $(dir $@)
	@if [ ! -s $< ]; then echo "Error: PTX file $< is empty or missing" >&2; exit 1; fi
	xxd -i $< | sed -E 's/unsigned int objs_src_nodes_hwaccel_yolo_mask_assemble_ptx_len/const unsigned int avpl_yolo_mask_assemble_ptx_len/; s/unsigned char objs_src_nodes_hwaccel_yolo_mask_assemble_ptx/const char avpl_yolo_mask_assemble_ptx/' > $@
	@if [ ! -s $@ ]; then echo "Error: Generated header $@ is empty. Check PTX file: $<" >&2; exit 1; fi
endif

ifeq ($(BUILD_VERT_PTX),1)
$(VERT_PREPROCESS_PTX): $(VERT_PREPROCESS_KERNEL)
	@mkdir -p $(dir $@)
	$(NVCC) -ptx -o $@ $<

$(VERT_PREPROCESS_PTX_H): $(VERT_PREPROCESS_PTX)
	@mkdir -p $(dir $@)
	@if [ ! -s $< ]; then echo "Error: PTX file $< is empty or missing" >&2; exit 1; fi
	xxd -i $< | sed -E 's/unsigned int objs_src_nodes_hwaccel_vert_preprocess_ptx_len/const unsigned int avpl_vert_preprocess_ptx_len/; s/unsigned char objs_src_nodes_hwaccel_vert_preprocess_ptx/const char avpl_vert_preprocess_ptx/' > $@
	@if [ ! -s $@ ]; then echo "Error: Generated header $@ is empty. Check PTX file: $<" >&2; exit 1; fi

objs/src/nodes/hwaccel/vert_infer.o: $(VERT_PREPROCESS_PTX_H)
endif

ifeq ($(BUILD_REFRAMER_PTX),1)
$(REFRAMER_PREPROCESS_PTX): $(REFRAMER_PREPROCESS_KERNEL)
	@mkdir -p $(dir $@)
	$(NVCC) -ptx -o $@ $<

$(REFRAMER_PREPROCESS_PTX_H): $(REFRAMER_PREPROCESS_PTX)
	@mkdir -p $(dir $@)
	@if [ ! -s $< ]; then echo "Error: PTX file $< is empty or missing" >&2; exit 1; fi
	xxd -i $< | sed -E 's/unsigned int objs_src_nodes_hwaccel_reframer_ptx_len/const unsigned int avpl_reframer_ptx_len/; s/unsigned char objs_src_nodes_hwaccel_reframer_ptx/const char avpl_reframer_ptx/' > $@
	@if [ ! -s $@ ]; then echo "Error: Generated header $@ is empty. Check PTX file: $<" >&2; exit 1; fi

objs/src/nodes/hwaccel/reframer.o: $(REFRAMER_PREPROCESS_PTX_H)
endif

compile_flags.txt:
	echo "$(CXXFLAGS)" | tr ' ' '\n' > $@

# anything that requires cpr headers must be compiled after cpr is configured
objs/src/rest_client.o: deps/cpr/build/lib/libcpr.a

.PRECIOUS: objs/%.d

include $(wildcard $(patsubst %.cpp,objs/%.d,$(CPPSRC_COMPILE)))
