# syntax=docker/dockerfile:1
FROM python:3.12-slim

# CUDA wheels keep the service usable on a GPU and still import on a CPU-only
# host (torch.cuda.is_available() is simply false there). Build with
# --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu for a much
# smaller CPU-only image; the runtime falls back to CPU automatically.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "torch==2.14.1" --index-url ${TORCH_INDEX_URL} \
    && pip install \
    sentence-transformers \
    fastapi \
    uvicorn[standard] \
    requests

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('jinaai/jina-embeddings-v5-text-nano-retrieval')"

# Preload the reranker so the first ambiguous recall does not pay a ~700 MB
# download inside the request. float32 is deliberate: sentence-transformers
# otherwise picks float16 on CPU, which measured ~5x slower for this model.
RUN python -c "import torch; from sentence_transformers import CrossEncoder; CrossEncoder('mixedbread-ai/mxbai-rerank-base-v1', max_length=256, model_kwargs={'dtype': torch.float32})"

COPY app.py llm.py /app/
WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
