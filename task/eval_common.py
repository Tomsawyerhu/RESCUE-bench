import json
import re
import time
from json import JSONDecoder
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMPTY_TARGET_LABEL = "__empty__"
GENERIC_GROUP_LABELS = {"Couple", "Family"}
SENSITIVE_WORD_PATTERNS = [
    re.compile(r"\bfuck(?:ing|ed|er|ers)?\b", re.IGNORECASE),
    re.compile(r"\bshit(?:ty)?\b", re.IGNORECASE),
    re.compile(r"\bbitch(?:es)?\b", re.IGNORECASE),
    re.compile(r"\basshole(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bdick\b", re.IGNORECASE),
    re.compile(r"\bpussy\b", re.IGNORECASE),
    re.compile(r"\bsex(?:ual)?\b", re.IGNORECASE),
    re.compile(r"\bpenis\b", re.IGNORECASE),
    re.compile(r"\bvagina\b", re.IGNORECASE),
    re.compile(r"\brape(?:d|s)?\b", re.IGNORECASE),
    re.compile(r"\bporn(?:ography|ographic)?\b", re.IGNORECASE),
]


def normalize_api_base(api_base: str) -> str:
    base = (api_base or "").strip().rstrip("/")
    if not base:
        raise ValueError("Missing API base.")
    return base if base.endswith("/v1") else f"{base}/v1"


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def extract_text_from_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("text", "content", "value", "output_text"):
            value = content.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, (list, dict)):
                nested = extract_text_from_message_content(value)
                if nested:
                    return nested
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            text = extract_text_from_message_content(item)
            if clean_text(text):
                parts.append(clean_text(text))
        if parts:
            return "\n".join(parts)
        return json.dumps(content, ensure_ascii=False)
    return str(content or "")


def parse_json_response_content(content: Any) -> Dict[str, Any]:
    cleaned = extract_text_from_message_content(content).strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.split("```json", 1)[1]
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 1)[1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()
    if not cleaned:
        raise json.JSONDecodeError("Empty JSON content", cleaned, 0)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = JSONDecoder()
    for index, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise json.JSONDecodeError("Could not parse JSON object from model response", cleaned, 0)


def mask_sensitive_terms(text: str) -> str:
    masked = text
    for pattern in SENSITIVE_WORD_PATTERNS:
        masked = pattern.sub(lambda match: match.group(0)[0] + "*" * max(1, len(match.group(0)) - 1), masked)
    return masked


def load_sample_id_filter(path: Optional[str]) -> set:
    if not path:
        return set()
    file_path = Path(path).expanduser().resolve()
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    try:
        payload = json.loads(text)
    except Exception:
        payload = None
    if isinstance(payload, list):
        return {clean_text(item) for item in payload if clean_text(item)}
    if isinstance(payload, dict) and isinstance(payload.get("sample_ids"), list):
        return {clean_text(item) for item in payload["sample_ids"] if clean_text(item)}
    return {clean_text(line) for line in text.splitlines() if clean_text(line)}


def canonical_target(values: Sequence[Any]) -> Tuple[str, ...]:
    labels = []
    for value in values or []:
        label = clean_text(value)
        if not label or label == "Therapist":
            continue
        labels.append(label)
    return tuple(labels)


def infer_group_mode(*, sample_id: str = "", rows: Optional[Sequence[Dict[str, Any]]] = None) -> str:
    normalized_sample_id = clean_text(sample_id).lower()
    if normalized_sample_id.startswith("family_therapy/") or "/family_therapy/" in normalized_sample_id:
        return "family"
    if rows:
        for row in rows:
            values: List[Any] = [row.get("primary_speaker")]
            values.extend(row.get("target") or [])
            for item in row.get("support_strategy") or []:
                if isinstance(item, dict):
                    values.extend(item.get("target") or [])
            for value in values:
                if clean_text(value) == "Family":
                    return "family"
    return "couple"


def group_label_for_mode(group_mode: str) -> str:
    return "Family" if clean_text(group_mode).lower() == "family" else "Couple"


def target_label(target: Tuple[str, ...], group_label: str = "Couple") -> str:
    if not target:
        return ""
    if len(target) == 1:
        return target[0]
    return group_label_for_mode(group_label)


def ensure_unique_ordered(values: Sequence[Any]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        label = clean_text(value)
        if not label or label in seen:
            continue
        seen.add(label)
        result.append(label)
    return result


def infer_participants(rows: Sequence[Dict[str, Any]]) -> List[str]:
    participants: List[str] = []
    seen = set()

    def add(label: Any) -> None:
        normalized = clean_text(label)
        if not normalized or normalized == "Therapist" or normalized in GENERIC_GROUP_LABELS or normalized in seen:
            return
        seen.add(normalized)
        participants.append(normalized)

    for row in rows:
        add(row.get("primary_speaker"))
        for target in row.get("target") or []:
            add(target)
        for item in row.get("support_strategy") or []:
            for target in item.get("target") or []:
                add(target)
    return participants


def build_candidate_target_labels(
    rows: Sequence[Dict[str, Any]],
    participants: Sequence[str],
    group_label: str = "Couple",
) -> List[str]:
    ordered_targets: List[Tuple[str, ...]] = []
    seen = set()

    def add(target_values: Sequence[Any]) -> None:
        target = canonical_target(target_values)
        if not target or target in seen:
            return
        seen.add(target)
        ordered_targets.append(target)

    for participant in participants:
        add([participant])
    add([group_label_for_mode(group_label)])

    for row in rows:
        if clean_text(row.get("primary_speaker")) == "Therapist":
            add(row.get("target") or [])
            for item in row.get("support_strategy") or []:
                add(item.get("target") or [])

    labels = [target_label(item, group_label=group_label) for item in ordered_targets]
    if not labels:
        return [EMPTY_TARGET_LABEL]
    return labels


def load_data_samples(data_root: Path, root_dir: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for path in sorted(data_root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = payload.get("rows", [])
        if not isinstance(rows, list):
            continue
        sample_id = path.relative_to(data_root).with_suffix("").as_posix()
        group_mode = infer_group_mode(sample_id=sample_id, rows=rows)
        group_label = group_label_for_mode(group_mode)
        participants = infer_participants(rows)
        samples.append(
            {
                "sample_id": sample_id,
                "couple": path.parent.name,
                "file_path": path.resolve(),
                "file_rel": path.relative_to(root_dir).as_posix(),
                "rows": rows,
                "group_mode": group_mode,
                "group_label": group_label,
                "participants": participants,
                "candidate_target_labels": build_candidate_target_labels(rows, participants, group_label=group_label),
            }
        )
    return samples


def call_json_model(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    disable_thinking: bool = False,
    retries: int = 5,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
            if disable_thinking:
                payload["enable_thinking"] = False
            response = requests.post(
                f"{api_base}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=240,
            )
            if response.status_code == 429:
                raise requests.HTTPError("429", response=response)
            response.raise_for_status()
            response_payload = response.json()
            message = response_payload["choices"][0]["message"]
            content = message.get("content")
            try:
                return parse_json_response_content(content)
            except json.JSONDecodeError as exc:
                content_preview = extract_text_from_message_content(content)[:1000]
                reasoning_preview = extract_text_from_message_content(message.get("reasoning_content"))[:1000]
                error_message = (
                    f"{exc.msg}; content_preview={content_preview!r}; "
                    f"reasoning_preview={reasoning_preview!r}"
                )
                raise json.JSONDecodeError(error_message, content_preview, 0) from exc
        except Exception as exc:
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                response_text = exc.response.text[:2000]
                exc = requests.HTTPError(f"{exc} | response={response_text}", response=exc.response)
            last_error = exc
            if attempt == retries:
                raise exc
            time.sleep(4 * attempt)
    raise RuntimeError(str(last_error))


def rank_metrics(gold: str, predicted: Sequence[str]) -> Dict[str, float]:
    try:
        position = list(predicted).index(gold) + 1
    except ValueError:
        position = 0
    return {
        "gold_rank": position,
        "recall_at_1": 1.0 if position == 1 else 0.0,
        "recall_at_2": 1.0 if 1 <= position <= 2 else 0.0,
        "recall_at_3": 1.0 if 1 <= position <= 3 else 0.0,
        "mrr": 1.0 / position if position else 0.0,
    }
