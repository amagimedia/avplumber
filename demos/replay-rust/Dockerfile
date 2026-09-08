# syntax=docker/dockerfile:1
#
# The replay demo's player image on the Rust core: the Rust `avplumber`
# executable and the Python TUI that drives it over the control protocol.
# Software H.264 in and libx264 out, from RPM Fusion's FFmpeg; no GPU. Janus
# and the browser preview are their own containers (`compose.yaml`).
#
# Build from the repository root (Compose does this for you):
#
#   docker build -f demos/replay-rust/Dockerfile -t avplumber-replay-rust:latest .
#
# Rebuilds are incremental: the cargo registry and the Rust build directory
# live in BuildKit cache mounts, and the dependencies are fetched in a layer
# keyed on the lock file, before the sources are copied in.

ARG FEDORA_VERSION=42

# ------------------------------------------------------- Rust avplumber
FROM fedora:${FEDORA_VERSION} AS rust-builder

# RPM Fusion's FFmpeg carries libx264; Fedora's own build does not. bindgen
# (through rusty_ffmpeg) loads libclang at build time and parses the FFmpeg
# headers with clang's own `stddef.h`: `clang-libs` plus
# `clang-resource-filesystem`, not the whole LLVM development set. The mirror
# options keep a slow mirror from stretching this first-time step.
RUN dnf -y install "https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm" \
    && dnf -y install --allowerasing --setopt=install_weak_deps=False \
        --setopt=fastestmirror=True --setopt=max_parallel_downloads=10 \
        ca-certificates clang-libs clang-resource-filesystem curl gcc git pkgconf-pkg-config \
        ffmpeg-devel \
    && dnf clean all

# The crates use Rust 2024 and let chains: a current stable toolchain, not the
# distribution's.
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH=/root/.cargo/bin:${PATH} \
    CARGO_TERM_COLOR=never \
    LIBCLANG_PATH=/usr/lib64

WORKDIR /src

# Dependencies first, from the manifests alone: this layer changes only when
# the lock file does, and the downloads land in the registry cache mount.
COPY Cargo.toml Cargo.lock ./
COPY avplumber_f7k/Cargo.toml avplumber_f7k/
COPY avplumber_nodes/Cargo.toml avplumber_nodes/
RUN mkdir -p avplumber_f7k/src avplumber_nodes/src/bin \
    && touch avplumber_f7k/src/lib.rs avplumber_nodes/src/lib.rs \
    && echo 'fn main() {}' > avplumber_nodes/src/bin/avplumber.rs \
    && echo 'fn main() {}' > avplumber_f7k/build.rs
RUN --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,target=/root/.cargo/git,sharing=locked \
    cargo fetch

# Now the real sources. The build directory is a cache mount too, so only the
# two workspace crates recompile after a source change; the binary is copied
# out because a cache mount is not part of the image.
COPY avplumber_f7k avplumber_f7k
COPY avplumber_nodes avplumber_nodes
# ffmpeg7_1: RPM Fusion's FFmpeg on Fedora 42 is the 7.1 series; the feature
# must match the libraries the crate links against.
# Fedora's libclang, loaded by bindgen, resolves its resource directory to the
# relative "../lib/clang/NN" and so misses clang's own stddef.h; the extra
# argument names that directory outright.
RUN --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,target=/root/.cargo/git,sharing=locked \
    --mount=type=cache,target=/src/target,id=avplumber-replay-rust-target,sharing=locked \
    BINDGEN_EXTRA_CLANG_ARGS="-isystem $(ls -d /usr/lib/clang/*/include | tail -n 1)" \
    cargo build --release --locked -p avplumber_nodes --features ffmpeg7_1,async --bin avplumber \
    && mkdir -p /out \
    && cp target/release/avplumber /out/avplumber \
    && /out/avplumber --help 2>&1 | grep -q usage

# ---------------------------------------------------------------- runtime
FROM fedora:${FEDORA_VERSION}

RUN dnf -y install "https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm" \
    && dnf -y install --allowerasing --setopt=install_weak_deps=False \
        --setopt=fastestmirror=True --setopt=max_parallel_downloads=10 \
        bash ca-certificates curl ffmpeg ffmpeg-libs python3 python3-pip \
    && dnf clean all

# The TUI dependency, before the demo sources so an edit to them keeps this layer.
COPY demos/replay/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY --from=rust-builder /out/avplumber /usr/local/bin/avplumber
COPY demos/replay/*.py demos/replay/requirements.txt demos/replay/README.md /opt/avplumber/demos/replay/
COPY demos/replay-rust/entrypoint.sh /usr/local/bin/replay-entrypoint
RUN chmod +x /usr/local/bin/replay-entrypoint \
    && /usr/local/bin/avplumber --help 2>&1 | grep -q usage \
    && ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx264 \
    && python3 -c 'import textual'

ENV AVPLUMBER_BIN=/usr/local/bin/avplumber \
    PYTHONUNBUFFERED=1 \
    JANUS_HOST=127.0.0.1 \
    JANUS_HTTP_PORT=8088 \
    JANUS_VIDEO_PORT=5004

WORKDIR /opt/avplumber/demos/replay
VOLUME ["/media"]
ENTRYPOINT ["/usr/local/bin/replay-entrypoint"]
CMD ["play", "--recording", "/media/replay.ts"]
