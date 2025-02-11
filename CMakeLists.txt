cmake_minimum_required(VERSION 3.10)
project(avplumber)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL Debug)
    set(OPTIMIZATION_FLAGS "-O0 -ftrapv")
else()
    set(OPTIMIZATION_FLAGS "-O3 -flto")
endif()

# Include directories
include_directories(
    deps/include
    /apps/ffmpeg/include
)

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -ftrapv")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -flto")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -rdynamic")

# Linker flags
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -L/apps/ffmpeg/lib -Wl,-rpath,/apps/ffmpeg/lib")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -L/apps/ffmpeg/lib -Wl,-rpath,/apps/ffmpeg/lib")

# Source files
file(GLOB NODES_SRC src/nodes/*.cpp)

if(DEFINED EMBED_IN AND EMBED_IN STREQUAL obs)
    file(GLOB OBS_NODES_SRC src/nodes/obs/*.cpp)
    list(APPEND NODES_SRC ${OBS_NODES_SRC})
    include_directories(${LIBOBS_INCLUDE_DIR} ${LIBOBS_INCLUDE_DIR}/../deps/glad/include)
    add_compile_definitions(EMBED_IN_OBS=1)
endif()

if(CMAKE_BUILD_TYPE STREQUAL Debug)
    file(GLOB DEBUG_NODES_SRC src/nodes/debug/*.cpp)
    list(APPEND NODES_SRC ${DEBUG_NODES_SRC})
    add_compile_definitions(SYNCMETER=1)
endif()

if(DEFINED HAVE_JACK AND HAVE_JACK EQUAL 1)
    file(GLOB JACK_NODES_SRC src/nodes/jack/*.cpp)
    list(APPEND NODES_SRC ${JACK_NODES_SRC})
    add_compile_definitions(HAVE_JACK=1)
    list(APPEND LIBS_FLAGS jack)
endif()

if(DEFINED HAVE_CUDA AND HAVE_CUDA EQUAL 1)
    file(GLOB CUDA_NODES_SRC src/nodes/cuda/*.cpp)
    list(APPEND NODES_SRC ${CUDA_NODES_SRC})
    list(APPEND CPPSRC src/cuda.cpp)
    add_compile_definitions(HAVE_CUDA=1)
    list(APPEND DEPS_LIBS deps/cuda_loader/cuda_drvapi_dynlink.o)
endif()

set(CPPSRC
    src/avplumber.cpp
    src/util.cpp
    src/avutils.cpp
    src/graph_core.cpp
    src/graph_mgmt.cpp
    src/stats.cpp
    src/output_control.cpp
    src/instance_shared.cpp
    src/hwaccel_mgmt.cpp
    src/EventLoop.cpp
    src/TickSource.cpp
    ${NODES_SRC}
)


include(ExternalProject)

ExternalProject_Add(
    klvanc
    PREFIX ${CMAKE_BINARY_DIR}/deps/libklvanc
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/deps/libklvanc
    CONFIGURE_COMMAND cd ${CMAKE_SOURCE_DIR}/deps/libklvanc && ./autogen.sh --build && ./configure --enable-shared=no --enable-static
    BUILD_COMMAND cd ${CMAKE_SOURCE_DIR}/deps/libklvanc && make
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/deps/libklvanc/src/.libs/libklvanc.a
)

ExternalProject_Add(
    klscte35
    PREFIX ${CMAKE_BINARY_DIR}/deps/libklscte35
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/deps/libklscte35
    CONFIGURE_COMMAND export CFLAGS=-I${CMAKE_SOURCE_DIR}/deps/include && export LDFLAGS=-L${CMAKE_BINARY_DIR}/deps/libklvanc/src/.libs && cd ${CMAKE_SOURCE_DIR}/deps/libklscte35 && ./autogen.sh --build && ./configure --enable-shared=no --libdir=${CMAKE_BINARY_DIR}/deps/libklvanc/src/.libs
    BUILD_COMMAND cd ${CMAKE_SOURCE_DIR}/deps/libklvanc && make
    INSTALL_COMMAND ""
    DEPENDS klvanc
    BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/deps/libklscte35/src/.libs/libklscte35.a
)

set(DEPS_LIBS
    cpr::cpr
    avcpp::avcpp
    ${CMAKE_BINARY_DIR}/deps/libklvanc/src/.libs/libklvanc.a
    ${CMAKE_BINARY_DIR}/deps/libklscte35/src/.libs/libklscte35.a
)

set(LIBS_FLAGS
    pthread
    curl
    ssl
    crypto
    boost_thread
    boost_system
    avcodec
    avfilter
    avutil
    avformat
    avdevice
    swscale
    swresample
    dl
)


# Executable and static library
add_executable(avplumber src/main.cpp ${CPPSRC})
target_link_libraries(avplumber ${DEPS_LIBS} ${LIBS_FLAGS})

add_library(avplumber_static STATIC ${CPPSRC})
target_link_libraries(avplumber_static ${DEPS_LIBS} ${LIBS_FLAGS})


add_dependencies(avplumber klvanc klscte35)
add_dependencies(avplumber_static klvanc klscte35)


# Add dependencies as external projects
add_subdirectory(deps/cpr)
add_subdirectory(deps/avcpp)

# Custom target for build date
add_custom_target(builddate
    COMMAND /bin/date '+#define COMPILE_DATE "%Y-%m-%d %H:%M:%S %z"' > builddate.h
    COMMAND echo '#define GIT_VERSION "$(shell git describe --abbrev --dirty --always --tags)"' >> builddate.h
    BYPRODUCTS builddate.h
)

# Custom command for generating node list
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/graph_factory.generated.cpp
    COMMAND ./generate_node_list ${NODES_SRC} > ${CMAKE_BINARY_DIR}/graph_factory.generated.cpp
    DEPENDS generate_node_list ${NODES_SRC}
)
