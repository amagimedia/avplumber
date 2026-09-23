#pragma once

#include <cstddef>
#include <cstdint>

constexpr uint32_t DMABUF_RELEASE_ACK_MAGIC =
	static_cast<uint32_t>('A') |
	(static_cast<uint32_t>('C') << 8) |
	(static_cast<uint32_t>('K') << 16) |
	(static_cast<uint32_t>('1') << 24);
constexpr size_t DMABUF_RELEASE_ACK_BYTES = 16;
enum class DmabufAckKind : uint32_t { Frame = 0, Drain = 1, Drained = 2 };

inline uint32_t dmabufLoadLe32(const uint8_t *data) {
	return static_cast<uint32_t>(data[0]) |
	       (static_cast<uint32_t>(data[1]) << 8) |
	       (static_cast<uint32_t>(data[2]) << 16) |
	       (static_cast<uint32_t>(data[3]) << 24);
}

inline uint64_t dmabufLoadLe64(const uint8_t *data) {
	uint64_t value = 0;
	for (unsigned int index = 0; index < 8; ++index)
		value |= static_cast<uint64_t>(data[index]) << (index * 8);
	return value;
}

inline void dmabufStoreLe32(uint8_t *data, uint32_t value) {
	for (unsigned int index = 0; index < 4; ++index)
		data[index] = static_cast<uint8_t>(value >> (index * 8));
}

inline void dmabufStoreLe64(uint8_t *data, uint64_t value) {
	for (unsigned int index = 0; index < 8; ++index)
		data[index] = static_cast<uint8_t>(value >> (index * 8));
}

inline void dmabufEncodeReleaseAck(uint8_t *data, uint64_t frame_count,
                                  DmabufAckKind kind = DmabufAckKind::Frame) {
	dmabufStoreLe32(data, DMABUF_RELEASE_ACK_MAGIC);
	dmabufStoreLe32(data + 4, static_cast<uint32_t>(kind));
	dmabufStoreLe64(data + 8, frame_count);
}

inline bool dmabufDecodeReleaseAck(const uint8_t *data, uint64_t &frame_count,
                                   DmabufAckKind *kind = nullptr) {
	const uint32_t flags = dmabufLoadLe32(data + 4);
	if (dmabufLoadLe32(data) != DMABUF_RELEASE_ACK_MAGIC ||
        flags > static_cast<uint32_t>(DmabufAckKind::Drained) || (!kind && flags != 0))
		return false;
	if (kind) *kind = static_cast<DmabufAckKind>(flags);
	frame_count = dmabufLoadLe64(data + 8);
	return true;
}
