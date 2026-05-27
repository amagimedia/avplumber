FROM fedora:42

RUN dnf -y install --setopt=install_weak_deps=False \
        python3 \
    && dnf clean all

WORKDIR /srv/janus-preview
COPY preview/index.html ./
COPY preview/preview-server.py ./

EXPOSE 8080/tcp

CMD ["python3", "preview-server.py"]
