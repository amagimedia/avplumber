#include "BYTETracker.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using bytetrack::BYTETracker;
using bytetrack::Object;
using bytetrack::STrack;

Object detection(float x, float y, float width, float height)
{
	return Object{x, y, width, height, 0, 0.95f, 0};
}

int single_track_id(const std::vector<STrack>& tracks, const std::string& frame)
{
	if (tracks.size() != 1)
	{
		throw std::runtime_error(frame + ": expected exactly one active track, got " +
			std::to_string(tracks.size()));
	}
	return tracks.front().track_id;
}

std::pair<int, int> track_pair(const Object& first, const Object& second)
{
	BYTETracker tracker;
	const int first_id = single_track_id(tracker.update({first}), "first frame");
	const int second_id = single_track_id(tracker.update({second}), "second frame");
	return {first_id, second_id};
}

void require(bool condition, const std::string& message)
{
	if (!condition)
	{
		throw std::runtime_error(message);
	}
}

void overlapping_normalized_boxes_keep_identity()
{
	const auto ids = track_pair(
		detection(0.10f, 0.10f, 0.20f, 0.40f),
		detection(0.12f, 0.10f, 0.20f, 0.40f));
	require(ids.first == ids.second,
		"overlapping normalized boxes should retain the track ID");
}

void disjoint_normalized_boxes_get_distinct_identities()
{
	const auto ids = track_pair(
		detection(0.10f, 0.10f, 0.10f, 0.30f),
		detection(0.50f, 0.10f, 0.10f, 0.30f));
	require(ids.first != ids.second,
		"disjoint normalized boxes should not retain the track ID");
}

void coordinate_scale_does_not_change_association()
{
	const auto normalized_ids = track_pair(
		detection(0.10f, 0.10f, 0.10f, 0.30f),
		detection(0.50f, 0.10f, 0.10f, 0.30f));
	const auto pixel_ids = track_pair(
		detection(100.0f, 100.0f, 100.0f, 300.0f),
		detection(500.0f, 100.0f, 100.0f, 300.0f));
	require((normalized_ids.first == normalized_ids.second) ==
		(pixel_ids.first == pixel_ids.second),
		"equivalent normalized and pixel boxes should have the same association result");
}

void zero_area_boxes_do_not_manufacture_overlap()
{
	const auto ids = track_pair(
		detection(0.10f, 0.10f, 0.0f, 0.30f),
		detection(0.10f, 0.10f, 0.0f, 0.30f));
	require(ids.first != ids.second,
		"zero-area boxes should have zero IoU and must not retain the track ID");
}

} // namespace

int main()
{
	try
	{
		overlapping_normalized_boxes_keep_identity();
		disjoint_normalized_boxes_get_distinct_identities();
		coordinate_scale_does_not_change_association();
		zero_area_boxes_do_not_manufacture_overlap();
	}
	catch (const std::exception& error)
	{
		std::cerr << "ByteTrack IoU regression test failed: " << error.what() << '\n';
		return EXIT_FAILURE;
	}

	std::cout << "ByteTrack IoU regression tests passed\n";
	return EXIT_SUCCESS;
}
