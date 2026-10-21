FROM python:3.12-slim

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    sentence-transformers \
    fastapi \
    uvicorn[standard] \
    requests

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('jinaai/jina-embeddings-v5-text-nano-retrieval')"
RUN pip install --no-cache-dir groq

COPY app.py /app/app.py
WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
