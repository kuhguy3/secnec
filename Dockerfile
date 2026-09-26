FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libportaudio2 libsndfile1 curl \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY barkwatch ./barkwatch
COPY scripts ./scripts
RUN pip install --no-cache-dir ".[yamnet,flac,doa]" && ./scripts/download_model.sh
VOLUME /app/data
ENTRYPOINT ["python", "-m", "barkwatch", "-c", "/app/config.yaml"]
CMD ["run"]
