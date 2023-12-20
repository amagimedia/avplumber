FROM alpine:3.19 as builder

RUN apk add ffmpeg-dev git g++ cmake build-base curl-dev openssl-dev libssl3 boost-dev perl

# We build dependencies first because they'll probably change less often than src/
# so we can use Docker build cache to save some time

# FIXME is it possible to do it in single COPY command without using COPY . ?
COPY deps /build/deps
COPY Makefile /build/

RUN make -C /build -j `nproc` deps/cpr/build/lib/libcpr.a deps/avcpp/build/src/libavcpp.a deps/cuda_loader/cuda_drvapi_dynlink.o


COPY src /build/src
COPY generate_node_list /build/
COPY .git /build/.git

RUN make -C /build -j `nproc`


FROM alpine:3.19

RUN apk add ffmpeg libcurl libssl3 musl
COPY --from=builder /build/avplumber /usr/local/bin/
ENTRYPOINT ["/usr/local/bin/avplumber"]
