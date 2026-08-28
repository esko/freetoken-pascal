FROM --platform=linux/amd64 python:3.12.11-slim-bookworm@sha256:c00fc7b44d844b6da22861ec24af43968a5200eac4ec607b4725d585165d6b49

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --no-install-recommends --yes binutils g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/freetoken-pascal
COPY requirements/cpu.lock requirements/cpu.lock
RUN python -m pip install --no-cache-dir --require-hashes -r requirements/cpu.lock
COPY . .

CMD ["bash"]
