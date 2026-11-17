#include "ClockOrigin.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

#define CHECK(expr) do { if (!(expr)) throw std::runtime_error(#expr); } while (false)

static bool refused(const std::string &text) {
    try {
        avp::parseShowRate(text);
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

void exact_rates_parse_and_decimals_are_refused() {
    const AVRational ntsc = avp::parseShowRate("30000/1001");
    CHECK(ntsc.num == 30000 && ntsc.den == 1001);
    const AVRational whole = avp::parseShowRate("60");
    CHECK(whole.num == 60 && whole.den == 1);
    const AVRational reduced = avp::parseShowRate("60/2");
    CHECK(reduced.num == 30 && reduced.den == 1);
    // "29.97" would be 2997/100 to av_parse_ratio: a rate whose ticks drift from 30000/1001.
    for (const char *bad : {"29.97", "2997/100", "30/0", "0", "-30", "", "30000/1001x", "1/0"})
        CHECK(refused(bad));
}

void every_show_rate_is_whole_in_the_shared_unit() {
    for (const char *rate : {"25", "30000/1001", "30", "50", "60000/1001", "60", "24000/1001"}) {
        const AVRational fps = avp::parseShowRate(rate);
        const long long num = 1LL * fps.den * avp::kShareTimeBase.den, den = 1LL * fps.num * avp::kShareTimeBase.num;
        CHECK(num % den == 0);
    }
}

void the_show_ticks_at_one_rate() {
    avp::ShowSync show;
    CHECK(show.rate().den == 0 && show.latencyNs() == -1);
    show.registerRate({30000, 1001}, "mixer mixer");
    show.registerRate({30000, 1001}, "realtime.team.sync");   // the same rate: fine
    bool threw = false;
    try {
        show.registerRate({30, 1}, "realtime.team.sync");    // a sync given "30" for a 29.97 program
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
    show.registerLatency(66733333);
    show.registerLatency(50000000);                           // the first program compositor's stays
    CHECK(show.latencyNs() == 66733333);
    avp::ShowSync other;                                      // per instance: another show is unaffected
    other.registerRate({60, 1}, "mixer other");
    CHECK(other.rate().num == 60);
}

void show_ticks_are_zero_without_a_clock_origin() {
    CHECK(avp::showZeroTicks({30000, 1001}) == 0);            // AVP_CLOCK_ORIGIN unset in the test
}

int main() {
    exact_rates_parse_and_decimals_are_refused();
    every_show_rate_is_whole_in_the_shared_unit();
    the_show_ticks_at_one_rate();
    show_ticks_are_zero_without_a_clock_origin();
    std::cout << "show rate tests passed\n";
}
