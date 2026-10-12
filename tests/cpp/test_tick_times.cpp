#include "mixer/primitives/TickTimes.hpp"
#include <iostream>
#include <stdexcept>

#define CHECK(expr) do { if (!(expr)) throw std::runtime_error(#expr); } while (false)

using avp::mixer::TickTimes;

void a_duration_falls_into_the_first_bucket_whose_bound_it_does_not_exceed() {
    TickTimes times(40000000);   // 25 fps: bounds 10, 20, 30, 40, 50, 60, 80 and 120 ms
    CHECK(times.bucket(0) == 0);
    CHECK(times.bucket(10000000) == 0);
    CHECK(times.bucket(10000001) == 1);
    CHECK(times.bucket(40000000) == 3);   // exactly one period still fits its slot
    CHECK(times.bucket(40000001) == 4);
    CHECK(times.bucket(80000000) == 6);
    CHECK(times.bucket(120000000) == 7);
    CHECK(times.bucket(120000001) == TickTimes::kBuckets - 1);
}

void counts_sum_and_maximum_follow_what_was_added() {
    TickTimes times(40000000);
    for (const int64_t ns : {5000000, 15000000, 15000000, 70000000, 500000000}) times.add(ns);
    CHECK(times.count() == 5);
    CHECK(times.counts()[0] == 1 && times.counts()[1] == 2 && times.counts()[6] == 1);
    CHECK(times.counts()[TickTimes::kBuckets - 1] == 1);
    CHECK(times.sumNs() == 605000000 && times.maxNs() == 500000000);
}

int main() {
    a_duration_falls_into_the_first_bucket_whose_bound_it_does_not_exceed();
    counts_sum_and_maximum_follow_what_was_added();
    std::cout << "tick times OK\n";
}
