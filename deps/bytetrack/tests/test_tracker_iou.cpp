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

void scheduled_skip_predicts_without_a_false_miss()
{
	BYTETracker tracker(60, 90);
	const int id = single_track_id(tracker.update({detection(100, 100, 60, 160)}), "start");
	auto observed = tracker.update({detection(104, 100, 60, 160)});
	const int hits = observed.front().tracklet_len;
	const float score = observed.front().score;
	auto predicted = tracker.predict_only();
	require(single_track_id(predicted, "skip") == id, "skip must retain identity");
	require(predicted.front().state == bytetrack::TrackState::Tracked, "skip must not mark lost");
	require(predicted.front().tracklet_len == hits, "prediction must not add detector support");
	require(predicted.front().score == score, "prediction must not increase confidence");
	require(predicted.front().tlbr[0] > observed.front().tlbr[0], "skip must advance motion");
	require(tracker.get_lost_stracks().empty(), "skip must not populate lost tracks");
	auto resumed = tracker.update({detection(112, 100, 60, 160)});
	require(single_track_id(resumed, "resumed") == id, "next observation must retain identity");
	require(resumed.front().frame_id == 4, "skip must advance the source-frame clock");
	require(tracker.update({}).empty(), "a real empty observation is still a miss");
	require(tracker.predict_only().empty(), "prediction must not resurrect a lost track");
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
		scheduled_skip_predicts_without_a_false_miss();
	}
	catch (const std::exception& error)
	{
		std::cerr << "ByteTrack IoU regression test failed: " << error.what() << '\n';
		return EXIT_FAILURE;
	}

	std::cout << "ByteTrack IoU regression tests passed\n";
	return EXIT_SUCCESS;
}
