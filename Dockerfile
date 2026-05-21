FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    zip \
    unzip \
    tar \
    pkg-config \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV VCPKG_ROOT=/opt/vcpkg
RUN git clone https://github.com/microsoft/vcpkg.git $VCPKG_ROOT \
    && $VCPKG_ROOT/bootstrap-vcpkg.sh

WORKDIR /src

COPY vcpkg.json .
RUN $VCPKG_ROOT/vcpkg install --triplet x64-linux

COPY . .
RUN cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
    && cmake --build build --config Release

# Optimized Runtime Environment
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pull the compiled binary
COPY --from=builder /src/build/main /app/chnsw_app

ENV OMP_NUM_THREADS=8
ENV OMP_PLACES=cores
ENV OMP_PROC_BIND=close

ENTRYPOINT ["/app/chnsw_app"]