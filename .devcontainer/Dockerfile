
# The default base image supports MacBook M1 (ARM).
# To run on RunPod GPU node (x86), set the `BASE_IMAGE` to:
# `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
# For building instructions see the README.md file.
ARG BASE_IMAGE=mcr.microsoft.com/devcontainers/base:jammy
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

RUN DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install -y build-essential --no-install-recommends make \
    ca-certificates \
    git \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev

# Python and poetry installation
ARG USERNAME=vscode
RUN if getent passwd ${USERNAME} > /dev/null 2>&1; then \
      echo "User ${USERNAME} exists"; \
    else \
      groupadd -r ${USERNAME} && \
      useradd -r -g ${USERNAME} ${USERNAME} -m -s /bin/bash; \
      echo "User ${USERNAME} created"; \
    fi

USER $USERNAME
ARG HOME="/home/$USERNAME"
ARG PYTHON_VERSION=3.11

ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${HOME}/.local/bin:$PATH"

RUN echo "done 0" \
    && curl https://pyenv.run | bash \
    && echo "done 1" \
    && pyenv install ${PYTHON_VERSION} \
    && echo "done 2" \
    && pyenv global ${PYTHON_VERSION} \
    && echo "done 3" \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.in-project true