# syntax=docker/dockerfile:1.7

ARG AVP_BASE_IMAGE=fanpulse-recorder:latest
FROM ${AVP_BASE_IMAGE}

ARG CUDA_REPO_URL=https://developer.download.nvidia.com/compute/cuda/repos/fedora42/x86_64/cuda-fedora42.repo
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG VLLM_PACKAGE=vllm==0.22.1
ARG CUPY_PACKAGE=cupy-cuda13x==14.1.1
ARG PIP_EXTRA_PACKAGES=

USER root

ENV CUDA_HOME=/usr/local/cuda-13.0
ENV CUDA_PATH=/usr/local/cuda-13.0
ENV HF_HOME=/models/huggingface
ENV LD_LIBRARY_PATH=/usr/local/lib:/opt/tensorrt/lib:/usr/local/cuda-13.0/lib64:/usr/local/cuda-13.0/targets/x86_64-linux/lib
ENV PATH=/usr/local/bin:/usr/local/cuda-13.0/bin:${PATH}
ENV PYTHONPATH=/opt/avplumber

RUN dnf -y install --setopt=install_weak_deps=False \
        ca-certificates \
        curl \
        dnf-plugins-core \
    && dnf config-manager addrepo --from-repofile="${CUDA_REPO_URL}" \
    && dnf -y install --setopt=install_weak_deps=False \
        cuda-cccl-13-0 \
        cuda-cudart-devel-13-0 \
        cuda-driver-devel-13-0 \
        cuda-nvrtc-devel-13-0 \
        libnvjitlink-devel-13-0 \
        libnvptxcompiler-13-0 \
        libnvvm-13-0 \
    && dnf clean all

RUN python3 -m pip install --no-cache-dir --upgrade pip uv \
    && uv pip install --system --index-strategy unsafe-best-match \
        "${VLLM_PACKAGE}" \
        "${CUPY_PACKAGE}" \
        --torch-backend=auto \
        --extra-index-url "${TORCH_INDEX_URL}" \
    && if [ -n "${PIP_EXTRA_PACKAGES}" ]; then \
        uv pip install --system --index-strategy unsafe-best-match ${PIP_EXTRA_PACKAGES}; \
    fi \
    && python3 -m pip cache purge || true \
    && rm -rf /root/.cache/pip /root/.cache/uv

RUN mkdir -p \
        /opt/avp-molmo \
        /opt/avplumber/pyplumber/examples \
        /opt/avplumber/src/nodes/neural_net/preprocess \
        /opt/avplumber/tests \
        /artifacts \
        /models/huggingface

COPY pyplumber/molmo_vllm.py /opt/avplumber/pyplumber/molmo_vllm.py
COPY pyplumber/examples/molmo-vllm-node.py /opt/avplumber/pyplumber/examples/molmo-vllm-node.py
COPY src/nodes/neural_net/preprocess/molmo2_preprocess.cu /opt/avplumber/src/nodes/neural_net/preprocess/molmo2_preprocess.cu
COPY tests/test_molmo_vllm.py /opt/avplumber/tests/test_molmo_vllm.py
COPY docker/molmo/run-smoke.sh /opt/avp-molmo/run-smoke.sh

RUN chmod +x /opt/avp-molmo/run-smoke.sh \
    && ldconfig

WORKDIR /opt/avplumber

ENTRYPOINT ["/opt/avp-molmo/run-smoke.sh"]
