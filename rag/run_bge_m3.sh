#!/usr/bin/env bash

set -euo pipefail

CONTAINER_NAME=${RAG_BGE_M3_CONTAINER_NAME:-bge-m3}
MODEL_ID=${RAG_EMBED_MODEL:-BAAI/bge-m3}
PORT=${RAG_EMBED_PORT:-8085}
IMAGE=${RAG_BGE_M3_IMAGE:-ghcr.io/huggingface/text-embeddings-inference:latest}
HF_CACHE_DIR=${HF_HOME:-${HOME}/.cache/huggingface}

HEALTH_URL="http://127.0.0.1:${PORT}/health"

if docker ps --filter "name=^/${CONTAINER_NAME}$" --filter "status=running" --format "{{.ID}}" | grep -q .; then
    echo "[rag] ${CONTAINER_NAME} already running"
    exit 0
fi

if docker ps -a --filter "name=^/${CONTAINER_NAME}$" --filter "status=exited" --format "{{.ID}}" | grep -q .; then
    echo "[rag] restarting existing ${CONTAINER_NAME} container"
    docker start "${CONTAINER_NAME}" >/dev/null
else
    if docker ps -a --filter "name=^/${CONTAINER_NAME}$" --format "{{.ID}}" | grep -q .; then
        docker rm "${CONTAINER_NAME}" >/dev/null
    fi

    echo "[rag] starting ${CONTAINER_NAME} with model ${MODEL_ID}"
    docker run -d --name "${CONTAINER_NAME}" \
        --gpus all \
        -p "${PORT}:80" \
        -v "${HF_CACHE_DIR}:/data" \
        "${IMAGE}" \
        --model-id "${MODEL_ID}" \
        --pooling cls \
        --dtype float16 \
        --max-batch-tokens 16384
fi

for attempt in $(seq 1 180); do
    if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
        echo "[rag] ${CONTAINER_NAME} ready on ${HEALTH_URL}"
        exit 0
    fi
    sleep 1
done

echo "[rag] ERROR: ${CONTAINER_NAME} failed to become ready at ${HEALTH_URL}" >&2
exit 1

