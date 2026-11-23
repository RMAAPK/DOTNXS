FROM python:3.12-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .
RUN python -m pip install --no-cache-dir scikit-build-core pybind11
RUN python -m pip wheel --no-build-isolation --no-deps . --wheel-dir /wheels
RUN g++ -O2 -std=c++17 -pthread -Iserver server/cpp_server.cpp -o /dotnxs-gateway

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    DOTNXS_WORKER_PORT=8001
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY . /app
COPY --from=build /wheels /wheels
COPY --from=build /dotnxs-gateway /usr/local/bin/dotnxs-gateway
RUN python -m pip install --no-cache-dir -r server/requirements.txt \
    && python -m pip install --no-cache-dir --no-deps /wheels/dotnxs-*.whl \
    && chmod +x /app/server/start.sh
EXPOSE 10000
ENTRYPOINT ["/app/server/start.sh"]
