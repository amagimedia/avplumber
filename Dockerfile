FROM alpine:3.21 AS builder

RUN apk add ffmpeg-dev git g++ cmake build-base curl-dev openssl-dev libssl3 boost-dev perl bash automake autoconf libtool

COPY . /build/

RUN --mount=type=cache,target=/build/build  cd /build && cmake -B build -DHAVE_JACK=OFF -DHAVE_VAAPI=OFF && cmake --build build -j`nproc` && cp /build/build/avplumber /build/avplumber


FROM alpine:3.21

RUN apk add ffmpeg libcurl libssl3 musl boost-thread
COPY --from=builder /build/avplumber /usr/local/bin/
ENTRYPOINT ["/usr/local/bin/avplumber"]
