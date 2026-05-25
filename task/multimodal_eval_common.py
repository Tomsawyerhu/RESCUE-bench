import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from annotate.annotate_srt_speakers import upload_local_file_for_model
from task.eval_common import call_json_model, clean_text, mask_sensitive_terms


REVIEW_ASSETS_ROOT = ROOT_DIR / "manual_review_system" / "review_assets"
DEFAULT_UPLOAD_CACHE = ROOT_DIR / "analysis" / "multimodal_video_upload_cache.json"
MODE_ALIASES = {
    "text": "dialogue",
    "frames": "turn_video_audio",
    "speaker_frames_audio": "turn_video_audio",
}
VALID_TEXT_MODES = {"dialogue", "audio", "visual", "audiovisual"}
VALID_MODES = VALID_TEXT_MODES | {"turn_video_audio"}


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.split("```json", 1)[1]
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 1)[1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def load_json_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def save_json_cache(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def call_video_json_model_with_sanitize_retry(
    api_base: str,
    api_key: str,
    model: str,
    video_urls: Sequence[str],
    prompt: str,
    max_tokens: int,
    disable_thinking: bool = False,
) -> Dict[str, Any]:
    prompts = [prompt, mask_sensitive_terms(prompt)]
    last_error: Optional[Exception] = None

    for prompt_index, current_prompt in enumerate(prompts):
        for attempt in range(1, 4):
            try:
                content: List[Dict[str, str]] = []
                for url in video_urls:
                    content.append(
                        {
                            "type": "video_url",
                            "video_url": {"url": url},
                        }
                    )
                content.append({"type": "text", "text": current_prompt})
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"},
                }
                if disable_thinking:
                    payload["enable_thinking"] = False
                response = requests.post(
                    f"{api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "X-DashScope-OssResourceResolve": "enable",
                    },
                    json=payload,
                    timeout=600,
                )
                response.raise_for_status()
                content_text = response.json()["choices"][0]["message"]["content"]
                return extract_json_object(content_text)
            except requests.HTTPError as exc:
                last_error = exc
                if "data_inspection_failed" in str(exc) and prompt_index == 0:
                    break
                if attempt == 3:
                    raise
                time.sleep(3 * attempt)
            except (requests.RequestException, OSError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt == 3:
                    raise
                time.sleep(3 * attempt)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Unknown error in call_video_json_model_with_sanitize_retry")


def load_review_manifest(sample_id: str) -> Dict[int, Dict[str, Any]]:
    manifest_path = REVIEW_ASSETS_ROOT / sample_id / "manifest.json"
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    result: Dict[int, Dict[str, Any]] = {}
    for item in payload.get("rows", []):
        try:
            row_index = int(item.get("index") or 0)
        except Exception:
            row_index = 0
        if row_index > 0:
            result[row_index] = item
    return result


def review_clip_info(manifest_rows: Dict[int, Dict[str, Any]], row_index: int) -> tuple[str, float]:
    manifest_item = manifest_rows.get(row_index) or {}
    clip_rel = clean_text(manifest_item.get("clip_path"))
    clip_path = ""
    if clip_rel:
        resolved = (ROOT_DIR / clip_rel).resolve()
        if resolved.exists():
            clip_path = str(resolved)
    try:
        duration_sec = float(manifest_item.get("duration_sec") or 0.0)
    except Exception:
        duration_sec = 0.0
    return clip_path, max(0.0, duration_sec)


def normalize_background_dialogue(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        speaker = clean_text(item.get("speaker"))
        target = [clean_text(value) for value in item.get("target") or [] if clean_text(value)]
        content = clean_text(item.get("content"))
        if not speaker or not target or not content:
            continue
        normalized.append({"speaker": speaker, "target": target, "content": content})
    return normalized


def normalize_named_descriptions(items: Sequence[Dict[str, Any]], person_key: str = "person") -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        person = clean_text(item.get(person_key))
        description = clean_text(item.get("description"))
        if not person or not description:
            continue
        normalized.append({person_key: person, "description": description})
    return normalized


def normalize_interaction_behavior(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        initiator = clean_text(item.get("initiator"))
        target = [clean_text(value) for value in item.get("target") or [] if clean_text(value)]
        description = clean_text(item.get("description"))
        if not initiator or not target or not description:
            continue
        normalized.append({"initiator": initiator, "target": target, "description": description})
    return normalized


def row_timestamp(row: Dict[str, Any]) -> str:
    return "%.3f-%.3f" % (float(row.get("start_time") or 0.0), float(row.get("end_time") or 0.0))


def build_prior_context_rows(
    rows: Sequence[Dict[str, Any]],
    manifest_rows: Dict[int, Dict[str, Any]],
    current_row_index: int,
    context_size: int,
) -> List[Dict[str, Any]]:
    if context_size == 0:
        return []

    context_start = 1 if context_size < 0 else max(1, current_row_index - context_size)
    prior_context: List[Dict[str, Any]] = []
    for prior_index in range(context_start, current_row_index):
        row = rows[prior_index - 1]
        clip_path, clip_duration_sec = review_clip_info(manifest_rows, prior_index)
        prior_context.append(
            {
                "row_index": prior_index,
                "timestamp": row_timestamp(row),
                "primary_speaker": clean_text(row.get("primary_speaker")),
                "target": [clean_text(item) for item in row.get("target") or [] if clean_text(item)],
                "dialogue_cleaned": clean_text(row.get("dialogue_cleaned")),
                "background_dialogue": normalize_background_dialogue(row.get("background_dialogue") or []),
                "tone_of_voice": clean_text(row.get("tone_of_voice")),
                "body_posture": normalize_named_descriptions(row.get("body_posture") or []),
                "facial_expressions": normalize_named_descriptions(row.get("facial_expressions") or []),
                "self_directed_behavior": normalize_named_descriptions(row.get("self_directed_behavior") or []),
                "interaction_behavior": normalize_interaction_behavior(row.get("interaction_behavior") or []),
                "clip_path": clip_path,
                "clip_duration_sec": clip_duration_sec,
            }
        )
    return prior_context


def prompt_record_for_mode(item: Dict[str, Any], mode: str) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "row_index": item["row_index"],
        "primary_speaker": item["primary_speaker"],
        "target": item["target"],
        "dialogue_cleaned": item["dialogue_cleaned"],
        "background_dialogue": item["background_dialogue"],
    }
    timestamp = clean_text(item.get("timestamp"))
    if timestamp:
        record["timestamp"] = timestamp

    if mode in {"visual", "audiovisual"}:
        record["body_posture"] = item["body_posture"]
        record["facial_expressions"] = item["facial_expressions"]
        record["self_directed_behavior"] = item["self_directed_behavior"]
        record["interaction_behavior"] = item["interaction_behavior"]

    if mode in {"audio", "audiovisual"}:
        record["tone_of_voice"] = item["tone_of_voice"]

    return record


def ensure_upload_url(
    cache: Dict[str, Any],
    cache_path: Path,
    api_key: str,
    model: str,
    clip_path: Path,
) -> str:
    key = str(clip_path.resolve())
    cached = clean_text(cache.get(key))
    if cached:
        return cached
    uploaded = upload_local_file_for_model(api_key, model, clip_path)
    cache[key] = uploaded
    save_json_cache(cache_path, cache)
    return uploaded
