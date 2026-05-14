#pragma once

#include <avcpp/packet.h>
#include <avcpp/frame.h>
#include "hwaccel/EglImageFrame.hpp"
#include "metadata_frame.hpp"

// Central registry of queue data types
#define EDGE_DATA_TYPES(X) \
	X(av::Packet) \
	X(av::VideoFrame) \
	X(av::AudioSamples) \
	X(EglImageFrame) \
	X(MetadataFrame)

// Raw-only data types (frames/samples, not encoded packets)
#define EDGE_RAW_DATA_TYPES(X) \
	X(av::VideoFrame) \
	X(av::AudioSamples)

// Comma-separated application over all data types (useful for base class lists)
#define EDGE_DATA_TYPES_AS_BASES(APPLY) \
	APPLY(av::Packet), \
	APPLY(av::VideoFrame), \
	APPLY(av::AudioSamples), \
	APPLY(EglImageFrame), \
	APPLY(MetadataFrame)

// Stable names for types (avoid typeid(T).name() variance)
template<typename T> constexpr const char* edgeTypeName();

#define EDGE_TYPE_NAME_SPEC(T, name_literal) \
	template<> inline constexpr const char* edgeTypeName<T>() { return name_literal; }

EDGE_TYPE_NAME_SPEC(av::Packet, "av::Packet")
EDGE_TYPE_NAME_SPEC(av::VideoFrame, "av::VideoFrame")
EDGE_TYPE_NAME_SPEC(av::AudioSamples, "av::AudioSamples")
EDGE_TYPE_NAME_SPEC(EglImageFrame, "EglImageFrame")
EDGE_TYPE_NAME_SPEC(MetadataFrame, "MetadataFrame")

