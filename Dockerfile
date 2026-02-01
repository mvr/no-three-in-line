FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

RUN apt-get update --yes && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
      cmake \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN cmake -S /app -B /app/build && \
    cmake --build /app/build --target three three_frontier && \
    cp /app/build/three /app/three && \
    cp /app/build/three_frontier /app/three_frontier

COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

CMD ["/app/run.sh"]
