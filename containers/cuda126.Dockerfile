FROM --platform=linux/amd64 python:3.12.11-slim-bookworm@sha256:c00fc7b44d844b6da22861ec24af43968a5200eac4ec607b4725d585165d6b49 AS python

FROM --platform=linux/amd64 nvidia/cuda:12.6.3-devel-ubuntu24.04@sha256:badf6c452e8b1efea49d0bb956bef78adcf60e7f87ac77333208205f00ac9ade

COPY --from=python /usr/local /usr/local
ENV PATH=/usr/local/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TORCH_CUDA_ARCH_LIST=6.1 \
    CMAKE_CUDA_ARCHITECTURES=61 \
    CUDAARCHS=61 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/freetoken-pascal
COPY requirements/cpu.lock requirements/cpu.lock
COPY requirements/cuda126.lock requirements/cuda126.lock
RUN python -m pip install --no-cache-dir --require-hashes -r requirements/cpu.lock
RUN python -m pip install --no-cache-dir --require-hashes \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    -r requirements/cuda126.lock
COPY . .

CMD ["bash"]
