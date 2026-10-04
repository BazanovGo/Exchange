# A container that cannot produce an irreproducible result by accident.
#
# PYTHONHASHSEED and the BLAS thread counts are fixed in the image rather
# than left to the caller, because both change numeric output and both are
# easy to forget. The build itself runs `exchange doctor` and the quick
# control suite, so an image that cannot find a known edge, or that lets a
# planted lookahead through, fails to build at all.

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so source edits do not invalidate the layer.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY configs ./configs
COPY examples ./examples
COPY tests ./tests
COPY docs ./docs

# A research container has no business writing outside its own workspace.
RUN useradd --create-home --shell /usr/sbin/nologin researcher \
    && mkdir -p /workspace/experiments /workspace/reports \
    && chown -R researcher:researcher /workspace /app
USER researcher
WORKDIR /workspace
ENV EXCHANGE_HOME=/workspace

# Calibration is part of the build. An image whose controls fail is not a
# usable research environment, and finding that out later costs more.
RUN cp -r /app/configs /workspace/configs \
    && exchange doctor \
    && exchange controls --quick \
    && rm -rf /workspace/experiments/* /workspace/reports/*

ENTRYPOINT ["exchange"]
CMD ["--help"]
