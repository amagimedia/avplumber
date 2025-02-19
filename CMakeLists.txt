cmake_minimum_required(VERSION 3.10)
project(avplumber)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(HAVE_CUDA ON CACHE BOOL "compile with CUDA support (built-in dynlink loader, no dependencies needed)")
set(HAVE_JACK ON CACHE BOOL "compile with JACK support (system library needed)")
set(HAVE_VAAPI ON CACHE BOOL "compile with VA-API support (system libraries needed: va GL EGL GLESv2)")

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

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

set(CPPSRC
    src/app_version.cpp
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
    ${CMAKE_BINARY_DIR}/graph_factory.generated.cpp
)

set(SYSTEM_LIBS
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

if(DEFINED EMBED_IN AND EMBED_IN STREQUAL obs)
    file(GLOB OBS_NODES_SRC src/nodes/obs/*.cpp)
    list(APPEND NODES_SRC ${OBS_NODES_SRC})
    include_directories(${LIBOBS_INCLUDE_DIR} ${LIBOBS_INCLUDE_DIR}/../deps/glad/include)
endif()

if(CMAKE_BUILD_TYPE STREQUAL Debug)
    file(GLOB DEBUG_NODES_SRC src/nodes/debug/*.cpp)
    list(APPEND NODES_SRC ${DEBUG_NODES_SRC})
    set_property(SOURCE src/stats.cpp APPEND PROPERTY COMPILE_DEFINITIONS SYNCMETER=1)
endif()

if(${HAVE_JACK})
    file(GLOB JACK_NODES_SRC src/nodes/jack/*.cpp)
    list(APPEND NODES_SRC ${JACK_NODES_SRC})
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(jack REQUIRED IMPORTED_TARGET GLOBAL jack)
    list(APPEND SYSTEM_LIBS PkgConfig::jack)
endif()

if(${HAVE_CUDA})
    file(GLOB CUDA_NODES_SRC src/nodes/cuda/*.cpp)
    list(APPEND NODES_SRC ${CUDA_NODES_SRC})
    list(APPEND CPPSRC src/cuda.cpp deps/cuda_loader/cuda_drvapi_dynlink.c)
    list(APPEND DEPS_LIBS deps/cuda_loader/cuda_drvapi_dynlink.o)
endif()

set(CPPSRC_ALL ${CPPSRC} ${NODES_SRC})

if(${HAVE_CUDA})
    set_property(SOURCE ${CPPSRC_ALL} APPEND PROPERTY COMPILE_DEFINITIONS HAVE_CUDA=1)
endif()

if(${HAVE_VAAPI})
    set_property(SOURCE ${CPPSRC_ALL} APPEND PROPERTY COMPILE_DEFINITIONS HAVE_VAAPI=1)
    list(APPEND SYSTEM_LIBS va GL EGL GLESv2)
endif()

if(DEFINED EMBED_IN AND EMBED_IN STREQUAL obs)
    set_property(SOURCE ${CPPSRC_ALL} APPEND PROPERTY COMPILE_DEFINITIONS EMBED_IN_OBS=1)
endif()


execute_process(COMMAND /bin/date "+%Y-%m-%d %H:%M:%S %z" OUTPUT_VARIABLE COMPILE_DATE OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND git describe --abbrev --dirty --always --tags OUTPUT_VARIABLE GIT_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
set_property(SOURCE src/app_version.cpp APPEND PROPERTY COMPILE_DEFINITIONS COMPILE_DATE="${COMPILE_DATE}" GIT_VERSION="${GIT_VERSION}")

include(ExternalProject)

ExternalProject_Add(
    klvanc
    PREFIX ${CMAKE_BINARY_DIR}/deps/libklvanc
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps/libklvanc
    CONFIGURE_COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR}/deps/libklvanc && ./autogen.sh --build && cd ${CMAKE_BINARY_DIR}/deps/libklvanc && ${CMAKE_CURRENT_SOURCE_DIR}/deps/libklvanc/configure --enable-shared=no --enable-static
    BUILD_COMMAND cd ${CMAKE_BINARY_DIR}/deps/libklvanc && make
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/deps/libklvanc/src/.libs/libklvanc.a
)

ExternalProject_Add(
    klscte35
    PREFIX ${CMAKE_BINARY_DIR}/deps/libklscte35
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps/libklscte35
    CONFIGURE_COMMAND export CFLAGS=-I${CMAKE_CURRENT_SOURCE_DIR}/deps/include && export LDFLAGS=-L${CMAKE_BINARY_DIR}/deps/libklvanc/src/.libs && cd ${CMAKE_CURRENT_SOURCE_DIR}/deps/libklscte35 && ./autogen.sh --build && cd ${CMAKE_BINARY_DIR}/deps/libklscte35 && ${CMAKE_CURRENT_SOURCE_DIR}/deps/libklscte35/configure --enable-shared=no --libdir=${CMAKE_BINARY_DIR}/deps/libklvanc
    BUILD_COMMAND export CFLAGS=-I${CMAKE_CURRENT_SOURCE_DIR}/deps/include && export LDFLAGS=-L${CMAKE_BINARY_DIR}/deps/libklscte35 && cd ${CMAKE_BINARY_DIR}/deps/libklscte35 && make
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


# Executable and static library

add_library(avplumber_static STATIC ${CPPSRC_ALL})
target_link_libraries(avplumber_static ${DEPS_LIBS} ${SYSTEM_LIBS})

add_executable(avplumber src/main.cpp)
target_link_libraries(avplumber avplumber_static)

add_dependencies(avplumber_static klvanc klscte35 cpr::cpr avcpp::avcpp)


# Add dependencies as external projects
set(CPR_USE_SYSTEM_CURL ON CACHE INTERNAL "")
set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "")
add_subdirectory(deps/cpr)

set(AV_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(AV_ENABLE_STATIC ON CACHE INTERNAL "")
set(AV_ENABLE_SHARED OFF CACHE INTERNAL "")
add_subdirectory(deps/avcpp)

# Custom command for generating node list
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/graph_factory.generated.cpp
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_node_list ${NODES_SRC} > ${CMAKE_BINARY_DIR}/graph_factory.generated.cpp
    DEPENDS generate_node_list ${NODES_SRC}
)
set_property(SOURCE ${CMAKE_BINARY_DIR}/graph_factory.generated.cpp APPEND PROPERTY INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR})
