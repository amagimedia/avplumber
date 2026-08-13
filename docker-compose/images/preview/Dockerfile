FROM fedora:42

RUN dnf -y install --setopt=install_weak_deps=False \
        python3 \
    && dnf clean all

WORKDIR /srv/janus-preview
COPY index.html ./
COPY preview-server.py ./

CMD ["python3", "preview-server.py"]
