FROM python:3.12-slim AS build
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh && sh /tmp/rustup.sh -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
COPY requirements-czsc-runtime.lock /app/
RUN python -m pip install --no-cache-dir -r requirements-czsc-runtime.lock
COPY vendor/czsc /app/vendor/czsc
COPY czsc-source-manifest.json build_czsc_runtime.py /app/
RUN python build_czsc_runtime.py --out /app/vendor/wheels

FROM python:3.12-slim AS runtime
WORKDIR /app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TZ=Asia/Shanghai \
    KHQUANT_DATA_ROOT=/khquant-data KHQUANT_RAW_DB=/khquant-data/raw/khquant_raw.db \
    KHQUANT_METADATA_ROOT=/khquant-data/metadata KHQUANT_ARTIFACT_ROOT=/khquant-artifacts
COPY --from=build /app/requirements-czsc-runtime.lock /app/
RUN python -m pip install --no-cache-dir -r requirements-czsc-runtime.lock
COPY --from=build /app/vendor /app/vendor
COPY --from=build /app/czsc-source-manifest.json /app/
RUN python -m pip install --no-deps /app/vendor/wheels/*.whl
COPY my_strategy /app/my_strategy
RUN mkdir -p /khquant-data/raw /khquant-data/processed/czsc /khquant-data/metadata /khquant-artifacts/runs
ENTRYPOINT ["python", "-m", "my_strategy.cli"]
CMD ["doctor", "--check-write", "--strict", "--json"]
