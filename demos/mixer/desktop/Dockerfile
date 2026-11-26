# syntax=docker/dockerfile:1
FROM node:24-bookworm AS native
WORKDIR /src/demos/mixer/desktop
COPY --from=source-tree src/nodes/hwaccel/TexInfoMessage.hpp /src/src/nodes/hwaccel/
COPY --from=source-tree deps/dma-browser/addons/fdpass/texinfo.h /src/deps/dma-browser/addons/fdpass/
COPY receiver.cc .
RUN c++ -std=c++17 -O2 -fPIC -shared -I/usr/local/include/node receiver.cc -o receiver.node

# Compose builds the HTML input browser first. Share its Linux Electron binary
# and graphics libraries. The GUI has its own process, profile and display.
FROM browser-runtime
WORKDIR /opt/mixer-desktop
COPY --chmod=0444 *.cjs *.html package.json ./
COPY --from=native --chmod=0555 /src/demos/mixer/desktop/receiver.node .
RUN node -e 'if (require("./package.json").devDependencies.electron !== require("/opt/dma-browser/node_modules/electron/package.json").version) process.exit(1)' \
    && chmod 4755 /opt/dma-browser/node_modules/electron/dist/chrome-sandbox
HEALTHCHECK NONE
ENTRYPOINT ["/bin/sh", "-ec", "mkdir -p -m 700 \"$XDG_RUNTIME_DIR\" && exec /opt/dma-browser/node_modules/electron/dist/electron --ozone-platform=\"$MIXER_GUI_PLATFORM\" /opt/mixer-desktop"]
