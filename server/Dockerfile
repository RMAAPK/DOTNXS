FROM python:3.12-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN python -m pip install --no-cache-dir scikit-build-core pybind11
RUN python -m pip wheel --no-build-isolation --no-deps \
    --wheel-dir /wheels "dotnxs[runtime] @ git+https://github.com/RMAAPK/DOTNXS.git@main"

WORKDIR /src
COPY . .
RUN g++ -O2 -std=c++17 -pthread -I. cpp_server.cpp -o /dotnxs-gateway

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    DOTNXS_WORKER_PORT=8001 \
    DOTNXS_TORCH_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    MALLOC_ARENA_MAX=2
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY . /app
COPY --from=build /wheels /wheels
COPY --from=build /dotnxs-gateway /usr/local/bin/dotnxs-gateway
RUN python -m pip install --no-cache-dir -r /app/requirements.txt \
    && python -m pip install --no-cache-dir --no-deps /wheels/dotnxs-*.whl \
    && chmod +x /app/start.sh
EXPOSE 10000
ENTRYPOINT ["/app/start.sh"]
