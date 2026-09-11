ARG WOLFI_IMAGE=public.ecr.aws/amagi-media-labs-sec/secure-container-base/wolfi-base

FROM ${WOLFI_IMAGE} AS builder

RUN apk add --no-cache \
    ffmpeg-dev git gcc cmake build-base \
    curl-dev openssl-dev boost-dev \
    perl bash automake autoconf libtool

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


FROM ${WOLFI_IMAGE}

RUN apk add --no-cache ffmpeg libcurl-openssl4 boost-thread
COPY --from=builder /build/avplumber /usr/local/bin/
ENTRYPOINT ["/usr/local/bin/avplumber"]
