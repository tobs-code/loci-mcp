import json
import logging
import os
import re
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

NLI_SYSTEM = (
    "You validate facts for a memory system. The two facts may describe the "
    "same subject (a service, person or system) or entirely different ones. "
    "First decide whether they refer to the same subject at all; compare "
    "attributes only after that."
)
NLI_INSTRUCTION = (
    "Two facts about DIFFERENT subjects remain neutral no matter how their "
    "attributes differ: both can be true at the same time (a staging API on "
    "one port never contradicts a database service on another). A staging "
    "fact and a production fact are different subjects even when both name "
    "the same service, so a differing port or host between them is neutral; "
    "the environment is part of the subject, not a value to compare across "
    "environments. A differing "
    "attribute (environment, host, port, version, date/time, number, state) "
    "or an explicit negation is a contradiction (score 0.9-1.0) ONLY when "
    "BOTH facts describe the SAME subject. Facts that merely omit details or "
    "add new ones are NOT contradictions (neutral unless the overlap clearly "
    "contradicts). Identical facts are entailment, not contradiction. "
    "Almost identical wording alone does NOT make two facts the same "
    "subject: facts that describe different instances of the same pattern "
    "(two different test facts, tickets, incidents or examples - each "
    "referring to its own item) name no shared concrete object and stay "
    "neutral even when only one word differs. Answer contradiction only "
    "when both facts clearly refer to the very same item or object. "
    "Both facts must state the SAME attribute of that item for a value "
    "difference to be a contradiction: 'API version is 2.3.1' vs 'API "
    "version is 3.0.0' contradicts (same attribute, different value), but "
    "'API version is 2.3.1' vs 'deployment runs through GitHub Actions' "
    "does not (different attributes/aspects, both can be true). "
    "Respond in JSON with keys 'label' (one of: contradiction, entailment, "
    "neutral) and 'score' (0.0-1.0)."
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

_PROVIDERS = {
    "groq": {
        "style": "openai",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-20b",
        "key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
    },
    "openai": {
        "style": "openai",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
        "model_env": None,
    },
    "anthropic": {
        "style": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-3-5-haiku-latest",
        "key_env": "ANTHROPIC_API_KEY",
        "model_env": None,
    },
    "ollama": {
        "style": "openai",
        "base_url": "http://host.docker.internal:11434/v1",
        "model": "llama3.2",
        "key_env": None,
        "model_env": None,
    },
    "openai-compatible": {
        "style": "openai",
        "base_url": "",
        "model": "",
        "key_env": "LLM_API_KEY",
        "model_env": None,
    },
}

_NO_KEY_PROVIDERS = {"ollama", "openai-compatible"}

# HTTP states where a fresh request succeeds after a brief wait: rate limits
# and provider-side hiccups. A 400-class validation error never recovers and
# must be surfaced immediately.
RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str
    base_url: str
    style: str

    @property
    def configured(self) -> bool:
        if self.provider == "none" or not self.base_url or not self.model:
            return False
        if self.provider in _NO_KEY_PROVIDERS:
            return True
        return bool(self.api_key)


def _detect_provider(env) -> str:
    for provider, key_env in (
        ("groq", "GROQ_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
    ):
        if (env.get(key_env) or "").strip():
            return provider
    if (env.get("LLM_BASE_URL") or "").strip():
        return "openai-compatible"
    if (env.get("OLLAMA_BASE_URL") or "").strip():
        return "ollama"
    return "groq"


def _normalize_ollama_url(url: str) -> str:
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    return url


def load_config(env=None) -> LLMConfig:
    env = os.environ if env is None else env
    provider = (env.get("LLM_PROVIDER") or "groq").strip().lower()
    if provider == "auto":
        provider = _detect_provider(env)
    if provider in ("none", "off", "disabled"):
        return LLMConfig("none", "", "", "", "none")
    if provider not in _PROVIDERS:
        logger.warning("Unknown LLM_PROVIDER %r, using openai-compatible", provider)
        provider = "openai-compatible"

    defaults = _PROVIDERS[provider]
    model = (env.get("LLM_MODEL") or "").strip()
    if not model and defaults["model_env"]:
        model = (env.get(defaults["model_env"]) or "").strip()
    if not model:
        model = defaults["model"]

    api_key = (env.get("LLM_API_KEY") or "").strip()
    if not api_key and defaults["key_env"]:
        api_key = (env.get(defaults["key_env"]) or "").strip()

    base_url = (env.get("LLM_BASE_URL") or "").strip()
    if not base_url and provider == "ollama":
        base_url = (env.get("OLLAMA_BASE_URL") or "").strip()
    if not base_url:
        base_url = defaults["base_url"]
    if provider == "ollama" and base_url:
        base_url = _normalize_ollama_url(base_url)

    return LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        style=defaults["style"],
    )


def build_chat_request(config: LLMConfig, system: str, user: str,
                       response_format: dict | None = None):
    base = config.base_url.rstrip("/")
    if config.style == "anthropic":
        url = f"{base}/messages"
        headers = {
            "content-type": "application/json",
            "x-api-key": config.api_key,
            "anthropic-version": "2023-06-01",
        }
        body = {
            "model": config.model,
            "max_tokens": 300,
            "temperature": 0.0,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        return url, headers, body

    url = f"{base}/chat/completions"
    headers = {"content-type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    body = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": 300,
    }
    if response_format is not None:
        body["response_format"] = response_format
    return url, headers, body


def extract_text(config: LLMConfig, payload: dict) -> str:
    if config.style == "anthropic":
        blocks = payload.get("content") or []
        if isinstance(blocks, list):
            return "".join(
                block.get("text", "") for block in blocks
                if isinstance(block, dict)
            )
        return ""
    choices = payload.get("choices") or []
    if not choices:
        return ""
    content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content
            if isinstance(part, dict)
        )
    return ""


def _loose_nli_object(text: str) -> dict | None:
    """Recover label/score from a response that is not valid JSON.

    Small models sometimes answer `label: contradiction, score: 0.9` or leave
    a trailing comma. The values are unambiguous, so parsing them keeps a
    usable judgment instead of failing the whole comparison. Only a real
    label and a numeric score are accepted - a guessed value would be worse
    than an honest "not judged".
    """
    label_match = re.search(
        r'"?label"?\s*[:=]\s*"?([A-Za-z_ -]+)', text, re.IGNORECASE)
    score_match = re.search(
        r'"?score"?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)', text, re.IGNORECASE)
    if not label_match or not score_match:
        return None
    label = label_match.group(1).strip().lower()
    if label not in ("contradiction", "neutral", "entailment"):
        return None
    try:
        score = float(score_match.group(1))
    except ValueError:
        return None
    return {"label": label, "score": score}


_BARE_LABELS = {"contradiction", "neutral", "entailment"}


def _bare_label_object(text: str) -> dict | None:
    """A reply that is only the label word, e.g. `contradiction` or `"neutral"`.

    Some providers ignore the JSON instruction on retries and answer with the
    bare class name. Only an exact (quoted or unquoted) label is accepted; a
    sentence that merely contains the word is not, because "not a
    contradiction" would otherwise turn into a contradiction.
    """
    cleaned = text.strip().strip("`'\"").strip().lower()
    if cleaned in _BARE_LABELS:
        return {"label": cleaned,
                "score": 0.9 if cleaned == "contradiction" else 0.0}
    return None


def _relaxed_json_loads(text: str):
    """json.loads that also tolerates trailing commas and control chars.

    Both are harmless in LLM replies: a trailing comma before `}` or `]` is
    a formatting slip, and raw newlines inside a JSON string (deepseek's
    invalid literal break was the same class of failure) must not lose an
    otherwise complete judgment. Anything structural still raises.
    """
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(cleaned, strict=False)


def parse_nli_json(raw: str | None) -> dict | None:
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        parsed = _relaxed_json_loads(text)
    except json.JSONDecodeError:
        parsed = None
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = _relaxed_json_loads(text[start:end + 1])
            except json.JSONDecodeError:
                parsed = None
    if isinstance(parsed, dict):
        return parsed
    return _loose_nli_object(text) or _bare_label_object(text)


def nli_score(payload: dict | None) -> float:
    if not payload:
        return 0.0
    try:
        label = str(payload.get("label", "")).lower()
        score = float(payload.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0
    if label != "contradiction" or score <= 0.0:
        return 0.0
    if score > 1.0:
        # Some providers answer in percent (e.g. 95). Normalize instead of
        # silently clamping 95 to 1.0, which presented a malformed score as a
        # perfect contradiction. An out-of-range value is understated rather
        # than over-trusted (a false contradiction supersedes a good memory).
        if score <= 100.0:
            score /= 100.0
        else:
            return 0.0
    return min(score, 1.0)


def nli_user_message(query: str, document: str,
                     query_entity: str | None = None,
                     document_entity: str | None = None) -> str:
    """Build the user prompt for one fact comparison.

    Entity labels are only a hint, not an assertion: when they differ the
    model is asked not to read differing attribute values of two different
    subjects as a contradiction. A mislabelled or missing entity never
    changes the judgment - the texts themselves stay decisive.
    """
    qa = f" [entity: {query_entity}]" if query_entity else ""
    da = f" [entity: {document_entity}]" if document_entity else ""
    message = f"Fact A{qa}: {query}\nFact B{da}: {document}"
    if (query_entity and document_entity
            and query_entity.strip().lower() != document_entity.strip().lower()):
        message += (
            "\nThe entity labels differ, so first check what the texts "
            "themselves say. When both texts plainly discuss the same "
            "concrete item (the same feature, plan, incident, system or "
            "request is named or described in both), the labels do not "
            "matter and you judge normally: a differing date/number or a "
            "direct negation between them is a contradiction. Treat the "
            "facts as different subjects only when the texts themselves "
            "indicate different instances (no shared named item)."
        )
    return message


def classify(config: LLMConfig, query: str, document: str,
             query_entity: str | None = None,
             document_entity: str | None = None) -> float:
    if not config.configured:
        return 0.0

    system = f"{NLI_SYSTEM} {NLI_INSTRUCTION}"
    user = nli_user_message(query, document, query_entity, document_entity)

    if config.style == "anthropic":
        attempts = [None]
    else:
        attempts = [
            {"type": "json_schema", "json_schema": {
                "name": "nli", "strict": True, "schema": NLI_SCHEMA,
            }},
            {"type": "json_object"},
            None,
        ]

    last_error = None
    for response_format in attempts:
        url, headers, body = build_chat_request(
            config, system, user, response_format,
        )
        # The response format is deterministic, the provider answer is not:
        # an occasional non-JSON reply (rate-limit page, truncated decode)
        # is retried with the same format before falling back to the next
        # one, so a transient hiccup does not fail the comparison. Retries
        # sleep briefly with linear backoff - an immediate resend hits the
        # same rate limit or truncation state and burns the attempt.
        for retry in range(3):
            if retry:
                time.sleep(0.5 * retry)
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=60)
                if resp.status_code >= 400:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    # Groq's strict structured output intermittently rejects
                    # the model's own (often empty) generation as
                    # `json_validate_failed`. That is a transient generation
                    # failure, not a bad request: retry the same format. A
                    # real 400 (bad model, bad parameter) is still not
                    # retried and falls through to the next format.
                    transient_generation = (
                        resp.status_code == 400
                        and any(marker in resp.text
                                for marker in ("json_validate_failed",
                                               "tool_use_failed",
                                               "failed_generation"))
                    )
                    if (resp.status_code in RETRYABLE_HTTP_STATUS
                            or transient_generation):
                        continue
                    break
                parsed = parse_nli_json(extract_text(config, resp.json()))
                if parsed is None:
                    last_error = "response was not valid NLI JSON"
                    continue
                return nli_score(parsed)
            except requests.Timeout as exc:
                # A timed-out request is the classic transient LLM hiccup:
                # wait briefly and resend before falling back.
                last_error = f"request timed out: {exc}"
                continue
            except requests.ConnectionError as exc:
                last_error = f"connection failed: {exc}"
                continue
            except requests.exceptions.JSONDecodeError as exc:
                last_error = f"response body was not JSON: {exc}"
                continue
            except Exception as exc:
                last_error = exc
                break
    raise RuntimeError(f"LLM request failed: {last_error}")
