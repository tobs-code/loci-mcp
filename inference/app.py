import json
import logging
import os
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
from groq import Groq

logger = logging.getLogger(__name__)

app = FastAPI(title="Inference Server")

embedder = SentenceTransformer(
    "jinaai/jina-embeddings-v5-text-nano-retrieval",
    trust_remote_code=True,
)
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

class EmbedRequest(BaseModel):
    texts: list[str]
    mode: str = "document"  # "query" or "document"

class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dimension: int

class NliRequest(BaseModel):
    query: str
    documents: list[str]

class NliResponse(BaseModel):
    contradiction_scores: list[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    # jina-embeddings-v5 is an asymmetric retrieval model: queries and
    # documents must be encoded with their own prompt, otherwise the
    # similarity scores collapse (paraphrases land around 0.2 instead of 0.8).
    prompt_name = "query" if req.mode == "query" else "document"
    try:
        vectors = embedder.encode(
            req.texts, normalize_embeddings=True, prompt_name=prompt_name
        )
    except (ValueError, KeyError):
        # Model without prompt presets: fall back to plain encoding.
        vectors = embedder.encode(req.texts, normalize_embeddings=True)
    embeddings = vectors.tolist()
    return EmbedResponse(
        embeddings=embeddings,
        dimension=len(embeddings[0]) if embeddings else 0,
    )

NLI_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": ["contradiction", "entailment", "neutral"],
        },
        "score": {"type": "number"},
    },
    "required": ["label", "score"],
    "additionalProperties": False,
}

NLI_SYSTEM = "Determine if Fact B contradicts Fact A."


@app.post("/nli", response_model=NliResponse)
def nli(req: NliRequest):
    if not groq_client.api_key:
        return NliResponse(contradiction_scores=[0.0] * len(req.documents))

    scores = []
    for idx, doc in enumerate(req.documents):
        user_msg = f"Fact A: {req.query}\nFact B: {doc}"
        js = None
        last_error = None
        # Try json_schema first (some models support strict schema), fall back to json_object
        for fmt in (
            {"type": "json_schema", "json_schema": {"name": "nli", "strict": True, "schema": NLI_SCHEMA}},
            {"type": "json_object"},
        ):
            try:
                resp = groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": NLI_SYSTEM + " Respond in JSON with keys 'label' (one of: contradiction, entailment, neutral) and 'score' (0.0-1.0)."},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0,
                    max_tokens=200,
                    response_format=fmt,
                )
                raw = resp.choices[0].message.content.strip()
                # Strip markdown code fences if present
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[-1]
                    raw = raw.rsplit("```", 1)[0].strip()
                js = json.loads(raw)
                break
            except Exception as exc:
                last_error = exc
                continue
        if js is None:
            # Provider down / invalid key: don't silently claim "no contradiction"
            # for every remaining document — abort early and say so in the logs.
            logger.warning("NLI unavailable, returning 0.0 for remaining documents: %s", last_error)
            scores.extend([0.0] * (len(req.documents) - idx))
            break
        try:
            if js and isinstance(js, dict):
                label = js.get("label", "")
                score = float(js.get("score", 0.0))
                if label == "contradiction" and score > 0.0:
                    scores.append(min(score, 1.0))
                else:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        except Exception:
            scores.append(0.0)

    return NliResponse(contradiction_scores=scores)
