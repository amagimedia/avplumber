FROM fedora:44

RUN dnf -y install --setopt=install_weak_deps=False \
        python3 \
    && dnf clean all

WORKDIR /srv/janus-preview
COPY index.html ./
COPY cut-latency.mjs ./
COPY receiver-stats.mjs ./
COPY janus-socket.mjs ./
COPY output-selection.mjs ./
COPY preview-server.py ./

# Python handles SIGINT as PID 1, so stopping the player closes its WebSockets promptly.
STOPSIGNAL SIGINT
CMD ["python3", "preview-server.py"]
