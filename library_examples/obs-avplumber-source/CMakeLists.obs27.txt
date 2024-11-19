project(avplumber-source)

include_directories(SYSTEM "${CMAKE_SOURCE_DIR}/libobs")

set(avplumber-source_SOURCES
	avplumber-source.cpp)


add_library(avplumber-source MODULE
	${avplumber-source_SOURCES})


# Based on https://stackoverflow.com/a/7883815
# CMake, are you joking? Does it really have to be so convoluted?

add_custom_target(build_libavplumber
	ALL COMMAND make -j`nproc` compile_flags.txt static_library EMBED_IN=obs LIBOBS_INCLUDE_DIR=${CMAKE_SOURCE_DIR}/libobs
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/avplumber
	BYPRODUCTS
	${CMAKE_CURRENT_SOURCE_DIR}/avplumber/libavplumber.a
	${CMAKE_CURRENT_SOURCE_DIR}/avplumber/deps/cpr/build/lib/libcpr.a
	${CMAKE_CURRENT_SOURCE_DIR}/avplumber/deps/avcpp/build/src/libavcpp.a)

add_library(libavplumber STATIC IMPORTED)
set_property(TARGET libavplumber APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(libavplumber PROPERTIES
	IMPORTED_LOCATION_NOCONFIG ${CMAKE_CURRENT_SOURCE_DIR}/avplumber/libavplumber.a)
add_dependencies(libavplumber build_libavplumber)


target_link_libraries(avplumber-source
	${avplumber-source_PLATFORM_DEPS}
	libavplumber
	${CMAKE_CURRENT_SOURCE_DIR}/avplumber/deps/cpr/build/lib/libcpr.a
	${CMAKE_CURRENT_SOURCE_DIR}/avplumber/deps/avcpp/build/src/libavcpp.a
	pthread curl boost_thread boost_system avcodec avfilter avutil avformat avdevice swscale swresample
	libobs)

set_target_properties(avplumber-source PROPERTIES FOLDER "avplumber source")

install_obs_plugin_with_data(avplumber-source data)
