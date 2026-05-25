import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests
from openpyxl import Workbook
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional for script progress.
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else _NullProgress()


    class _NullProgress:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, n=1):
            return None

        def set_postfix_str(self, *args, **kwargs):
            return None


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from annotate.annotate_srt_speakers import upload_local_file_for_model
from task.eval_common import (
    DEFAULT_API_BASE,
    call_json_model,
    clean_text,
    load_sample_id_filter,
    mask_sensitive_terms,
    normalize_api_base,
)
from utils.project_env import load_project_env

load_project_env(ROOT_DIR)


REVIEW_ASSETS_ROOT = ROOT_DIR / "manual_review_system" / "review_assets"
DEFAULT_UPLOAD_CACHE = ROOT_DIR / "analysis" / "user_viewpoints_video_upload_cache.json"
MODE_ALIASES = {
    "text": "dialogue",
    "frames": "turn_video_audio",
    "speaker_frames_audio": "turn_video_audio",
}
VALID_MODES = {"dialogue", "audio", "visual", "audiovisual", "turn_video_audio"}


@dataclass
class UserViewpointCheckpoint:
    sample_id: str
    couple: str
    row_index: int
    timestamp: str
    speaker: str
    participants: List[str]
    target: List[str]
    dialogue: str
    background_dialogue: List[Dict[str, Any]]
    tone_of_voice: str
    body_posture: List[Dict[str, Any]]
    facial_expressions: List[Dict[str, Any]]
    self_directed_behavior: List[Dict[str, Any]]
    interaction_behavior: List[Dict[str, Any]]
    prior_context: List[Dict[str, Any]]
    gold_viewpoints: List[Dict[str, str]]
    clip_path: str = ""
    clip_duration_sec: float = 0.0


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


def normalize_object_list(payload: Dict[str, Any], field: str) -> List[Dict[str, Any]]:
    raw_items = payload.get(field, [])
    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    if not isinstance(raw_items, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, dict):
            normalized.append(item)
            continue
        if isinstance(item, str):
            try:
                parsed = json.loads(item)
            except Exception:
                continue
            if isinstance(parsed, dict):
                normalized.append(parsed)
    return normalized


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


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def call_json_model_with_sanitize_retry(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    disable_thinking: bool = False,
) -> Dict[str, Any]:
    prompts = [prompt, mask_sensitive_terms(prompt)]
    last_error: Optional[Exception] = None

    for prompt_index, current_prompt in enumerate(prompts):
        for attempt in range(1, 4):
            try:
                return call_json_model(
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    prompt=current_prompt,
                    max_tokens=max_tokens,
                    disable_thinking=disable_thinking,
                )
            except requests.HTTPError as exc:
                last_error = exc
                if "data_inspection_failed" in str(exc) and prompt_index == 0:
                    break
                if attempt == 3:
                    raise
                time.sleep(3 * attempt)
            except (requests.RequestException, OSError) as exc:
                last_error = exc
                if attempt == 3:
                    raise
                time.sleep(3 * attempt)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Unknown error in call_json_model_with_sanitize_retry")


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
                content: List[Dict[str, Any]] = []
                for video_url in video_urls:
                    content.append(
                        {
                            "type": "video_url",
                            "video_url": {"url": video_url},
                            "fps": 1,
                        }
                    )
                content.append(
                    {
                        "type": "text",
                        "text": current_prompt,
                    }
                )
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
                content = response.json()["choices"][0]["message"]["content"]
                return extract_json_object(content)
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


def valid_client_speaker(label: Any) -> bool:
    speaker = clean_text(label)
    return bool(speaker) and speaker not in {"Therapist", "Couple", "Family"}


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


def extract_participants(rows: Sequence[Dict[str, Any]]) -> List[str]:
    participants: List[str] = []
    seen = set()

    def add(label: Any) -> None:
        value = clean_text(label)
        if not value or value in {"Couple", "Family"}:
            return
        if value in seen:
            return
        seen.add(value)
        participants.append(value)

    for row in rows or []:
        add(row.get("primary_speaker"))
        for value in row.get("target") or []:
            add(value)
        for item in row.get("viewpoints_attitudes") or []:
            if not isinstance(item, dict):
                continue
            add(item.get("source"))
            add(item.get("target"))
    return participants


def normalize_gold_viewpoints(row: Dict[str, Any], speaker: str) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for item in row.get("viewpoints_attitudes") or []:
        if not isinstance(item, dict):
            continue
        source = clean_text(item.get("source"))
        target = clean_text(item.get("target"))
        viewpoint = clean_text(item.get("viewpoint"))
        attitude = clean_text(item.get("attitude"))
        if source != speaker or not target or not viewpoint:
            continue
        normalized.append(
            {
                "source": source,
                "target": target,
                "viewpoint": viewpoint,
                "attitude": attitude,
            }
        )
    return normalized


def normalize_predicted_viewpoints(raw_items: Any, default_source: str = "") -> List[Dict[str, str]]:
    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    if not isinstance(raw_items, list):
        return []

    normalized: List[Dict[str, str]] = []
    for item in raw_items:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except Exception:
                continue
        if not isinstance(item, dict):
            continue
        source = clean_text(item.get("source")) or default_source
        target = clean_text(item.get("target"))
        viewpoint = clean_text(item.get("viewpoint"))
        attitude = clean_text(item.get("attitude"))
        if not target or not viewpoint:
            continue
        normalized.append(
            {
                "source": source,
                "target": target,
                "viewpoint": viewpoint,
                "attitude": attitude,
            }
        )
    return normalized


def build_prior_context_rows(
    rows: Sequence[Dict[str, Any]],
    manifest_rows: Dict[int, Dict[str, Any]],
    current_row_index: int,
    context_size: int,
) -> List[Dict[str, Any]]:
    if context_size <= 0:
        return []

    context_start = max(1, current_row_index - context_size)
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


def load_json_samples(data_root: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for path in sorted(data_root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = payload.get("rows", [])
        if not isinstance(rows, list):
            continue
        samples.append(
            {
                "sample_id": path.relative_to(data_root).with_suffix("").as_posix(),
                "couple": path.parent.name,
                "file_rel": path.relative_to(ROOT_DIR).as_posix(),
                "participants": extract_participants(rows),
                "rows": rows,
            }
        )
    return samples


def pick_evenly(items: Sequence[UserViewpointCheckpoint], limit: int) -> List[UserViewpointCheckpoint]:
    if limit <= 0 or len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[0]]
    selected: List[UserViewpointCheckpoint] = []
    for rank in range(limit):
        pos = round(rank * (len(items) - 1) / (limit - 1))
        selected.append(items[pos])
    seen = set()
    deduped: List[UserViewpointCheckpoint] = []
    for item in selected:
        if item.row_index in seen:
            continue
        seen.add(item.row_index)
        deduped.append(item)
    if len(deduped) < limit:
        for item in items:
            if item.row_index in seen:
                continue
            seen.add(item.row_index)
            deduped.append(item)
            if len(deduped) >= limit:
                break
    return deduped[:limit]


def build_checkpoint(
    sample: Dict[str, Any],
    row_index: int,
    row: Dict[str, Any],
    manifest_rows: Dict[int, Dict[str, Any]],
    context_size: int,
) -> Optional[UserViewpointCheckpoint]:
    speaker = clean_text(row.get("primary_speaker"))
    if not valid_client_speaker(speaker):
        return None
    gold_viewpoints = normalize_gold_viewpoints(row, speaker)
    if not gold_viewpoints:
        return None
    clip_path, clip_duration_sec = review_clip_info(manifest_rows, row_index)
    return UserViewpointCheckpoint(
        sample_id=sample["sample_id"],
        couple=sample["couple"],
        row_index=row_index,
        timestamp=row_timestamp(row),
        speaker=speaker,
        participants=list(sample["participants"]),
        target=[clean_text(item) for item in row.get("target") or [] if clean_text(item)],
        dialogue=clean_text(row.get("dialogue_cleaned")),
        background_dialogue=normalize_background_dialogue(row.get("background_dialogue") or []),
        tone_of_voice=clean_text(row.get("tone_of_voice")),
        body_posture=normalize_named_descriptions(row.get("body_posture") or []),
        facial_expressions=normalize_named_descriptions(row.get("facial_expressions") or []),
        self_directed_behavior=normalize_named_descriptions(row.get("self_directed_behavior") or []),
        interaction_behavior=normalize_interaction_behavior(row.get("interaction_behavior") or []),
        prior_context=build_prior_context_rows(sample["rows"], manifest_rows, row_index, context_size),
        gold_viewpoints=gold_viewpoints,
        clip_path=clip_path,
        clip_duration_sec=clip_duration_sec,
    )


def select_checkpoints(
    sample: Dict[str, Any],
    max_checkpoints: int,
    require_video: bool,
    context_size: int,
) -> List[UserViewpointCheckpoint]:
    manifest_rows = load_review_manifest(sample["sample_id"])
    candidates: List[UserViewpointCheckpoint] = []
    for row_index, row in enumerate(sample["rows"], start=1):
        checkpoint = build_checkpoint(sample, row_index, row, manifest_rows, context_size)
        if checkpoint is None:
            continue
        if require_video and not checkpoint.clip_path:
            continue
        candidates.append(checkpoint)
    return pick_evenly(candidates, max_checkpoints)


def checkpoint_id(checkpoint: UserViewpointCheckpoint) -> str:
    return f"{checkpoint.sample_id}__row{checkpoint.row_index}"


def prompt_record_for_mode(item: Dict[str, Any], mode: str) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "row_index": item["row_index"],
        "primary_speaker": item["primary_speaker"],
        "participants": item["participants"],
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


def checkpoint_prompt_record(checkpoint: UserViewpointCheckpoint, mode: str, context_size: int) -> Dict[str, Any]:
    record = prompt_record_for_mode(
        {
            "row_index": checkpoint.row_index,
            "timestamp": checkpoint.timestamp,
            "primary_speaker": checkpoint.speaker,
            "participants": checkpoint.participants,
            "target": checkpoint.target,
            "dialogue_cleaned": checkpoint.dialogue,
            "background_dialogue": checkpoint.background_dialogue,
            "tone_of_voice": checkpoint.tone_of_voice,
            "body_posture": checkpoint.body_posture,
            "facial_expressions": checkpoint.facial_expressions,
            "self_directed_behavior": checkpoint.self_directed_behavior,
            "interaction_behavior": checkpoint.interaction_behavior,
        },
        mode,
    )
    record["checkpoint_id"] = checkpoint_id(checkpoint)
    if context_size > 0 and checkpoint.prior_context:
        record["prior_context"] = [
            prompt_record_for_mode(
                {
                    "row_index": item["row_index"],
                    "timestamp": item["timestamp"],
                    "primary_speaker": item["primary_speaker"],
                    "participants": checkpoint.participants,
                    "target": item["target"],
                    "dialogue_cleaned": item["dialogue_cleaned"],
                    "background_dialogue": item["background_dialogue"],
                    "tone_of_voice": item["tone_of_voice"],
                    "body_posture": item["body_posture"],
                    "facial_expressions": item["facial_expressions"],
                    "self_directed_behavior": item["self_directed_behavior"],
                    "interaction_behavior": item["interaction_behavior"],
                },
                mode,
            )
            for item in checkpoint.prior_context
        ]
    return record


def build_turn_video_prompt_bundle(
    checkpoint: UserViewpointCheckpoint,
    context_size: int,
    upload_cache: Dict[str, Any],
    upload_cache_path: Path,
    api_key: str,
    model: str,
) -> tuple[Dict[str, Any], List[str]]:
    record = checkpoint_prompt_record(checkpoint, "turn_video_audio", context_size)
    video_urls: List[str] = []
    attachment_index = 1

    if context_size > 0 and checkpoint.prior_context and record.get("prior_context"):
        enriched_prior_context: List[Dict[str, Any]] = []
        prompt_prior_context = record.get("prior_context", [])
        for source_item, prompt_item in zip(checkpoint.prior_context, prompt_prior_context):
            enriched_item = dict(prompt_item)
            clip_path = clean_text(source_item.get("clip_path"))
            if clip_path and Path(clip_path).exists():
                video_urls.append(ensure_upload_url(upload_cache, upload_cache_path, api_key, model, Path(clip_path)))
                enriched_item["video_attachment_index"] = attachment_index
                try:
                    duration = float(source_item.get("clip_duration_sec") or 0.0)
                except Exception:
                    duration = 0.0
                if duration > 0:
                    enriched_item["video_duration_sec"] = round(duration, 3)
                attachment_index += 1
            enriched_prior_context.append(enriched_item)
        record["prior_context"] = enriched_prior_context

    current_clip_path = Path(checkpoint.clip_path)
    video_urls.append(ensure_upload_url(upload_cache, upload_cache_path, api_key, model, current_clip_path))
    record["video_attachment_index"] = attachment_index
    if checkpoint.clip_duration_sec > 0:
        record["video_duration_sec"] = round(checkpoint.clip_duration_sec, 3)
    return record, video_urls


def prediction_prompt(sample_id: str, mode: str, records: Sequence[Dict[str, Any]], context_size: int) -> str:
    mode_guidance = {
        "dialogue": "Use only language-side evidence from primary_speaker, participants, target, dialogue_cleaned, and background_dialogue.",
        "audio": "Use language-side evidence together with tone_of_voice. Do not use visual annotations or video evidence.",
        "visual": "Use language plus visual annotation evidence from body_posture, facial_expressions, self_directed_behavior, and interaction_behavior.",
        "audiovisual": "Use language, visual annotations, and tone_of_voice.",
        "turn_video_audio": "Use the attached row-level video clip together with primary_speaker, participants, target, dialogue_cleaned, and background_dialogue. The clip already contains audio.",
    }
    context_guidance = (
        "No prior dialogue context is provided.\n"
        if context_size <= 0
        else f"If prior_context is provided, use at most the previous {context_size} rows only to resolve references or clarify who is being described. Current-turn evidence should dominate.\n"
    )
    video_guidance = (
        ""
        if mode != "turn_video_audio"
        else "For turn_video_audio, attached video clips appear in attachment order. Any prior_context item with video_attachment_index refers to its matching attached clip, ordered from oldest to newest. The current checkpoint record also has its own video_attachment_index and is the main target.\n"
    )
    return (
        "You are evaluating user-viewpoint understanding in a therapy session.\n"
        "For each checkpoint, identify the CURRENT speaker's viewpoint about another on-screen person in this turn.\n"
        "Ask directly: who is expressing the view, who is the view about, and what is that view?\n"
        "The source should normally be the row's primary_speaker.\n"
        "If there are multiple distinct viewpoints toward different people in the same row, output all of them.\n"
        "Do not use future rows or outside context.\n"
        f"{context_guidance}"
        f"{video_guidance}"
        "Return short, concrete English viewpoint phrases such as 'she is perpetually dissatisfied and impossible to please'.\n"
        f"Mode: {mode}\n"
        f"{mode_guidance[mode]}\n"
        "Output JSON only with schema:\n"
        "{"
        "\"sample_id\":\"alan_and_evelyn/001_s01e01_ct25_segment04\","
        "\"predictions\":["
        "{"
        "\"checkpoint_id\":\"alan_and_evelyn/001_s01e01_ct25_segment04__row2\","
        "\"predicted_viewpoints\":["
        "{"
        "\"source\":\"Alan\","
        "\"target\":\"Evelyn\","
        "\"viewpoint\":\"she is perpetually dissatisfied and impossible to please\""
        "}"
        "],"
        "\"reason\":\"short English reason\""
        "}"
        "]"
        "}\n"
        f"Sample: {sample_id}\n"
        f"Checkpoints: {json.dumps(list(records), ensure_ascii=False)}"
    )


def judge_prompt(sample_id: str, predictions: Sequence[Dict[str, Any]], checkpoints: Sequence[UserViewpointCheckpoint]) -> str:
    gold_lookup = {
        checkpoint_id(checkpoint): {
            "speaker": checkpoint.speaker,
            "timestamp": checkpoint.timestamp,
            "dialogue": checkpoint.dialogue,
            "gold_viewpoints": checkpoint.gold_viewpoints,
        }
        for checkpoint in checkpoints
    }
    return (
        "You are grading viewpoint predictions for a therapy user-understanding task.\n"
        "The task is to identify who holds what view of whom in the CURRENT speaker's turn.\n"
        "Compare predicted_viewpoints against gold_viewpoints.\n"
        "Be lenient about paraphrases and near-synonyms.\n"
        "Rubric:\n"
        "source_target_score: 0 if source-target pairings are mostly wrong or missing, 1 if partially recovered, 2 if all gold pairings are essentially recovered.\n"
        "viewpoint_score: 0 wrong, 1 weakly related, 2 close, 3 essentially same, 4 almost exact.\n"
        "viewpoint_total_score = source_target_score + viewpoint_score, range 0-6.\n"
        "Output JSON only with schema:\n"
        "{"
        "\"sample_id\":\"alan_and_evelyn/001_s01e01_ct25_segment04\","
        "\"graded\":["
        "{"
        "\"checkpoint_id\":\"alan_and_evelyn/001_s01e01_ct25_segment04__row2\","
        "\"source_target_score\":2,"
        "\"viewpoint_score\":3,"
        "\"viewpoint_total_score\":5,"
        "\"verdict\":\"strong|pass|borderline|fail\","
        "\"notes\":\"short English note\""
        "}"
        "]"
        "}\n"
        f"Sample: {sample_id}\n"
        f"Predictions: {json.dumps(list(predictions), ensure_ascii=False)}\n"
        f"Gold annotations: {json.dumps(gold_lookup, ensure_ascii=False)}"
    )


def normalize_prediction_items(
    sample_id: str,
    raw_items: Sequence[Dict[str, Any]],
    mode: str,
    default_source_by_checkpoint: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in raw_items:
        checkpoint_name = clean_text(item.get("checkpoint_id"))
        if not checkpoint_name:
            continue
        default_source = ""
        if default_source_by_checkpoint:
            default_source = clean_text(default_source_by_checkpoint.get(checkpoint_name))
        normalized.append(
            {
                "checkpoint_id": checkpoint_name,
                "predicted_viewpoints": normalize_predicted_viewpoints(
                    item.get("predicted_viewpoints", []),
                    default_source=default_source,
                ),
                "reason": clean_text(item.get("reason")),
                "_prediction_mode": mode,
                "_sample_id": sample_id,
            }
        )
    return normalized


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


def predict_textual_sample(
    api_base: str,
    api_key: str,
    model: str,
    sample_id: str,
    checkpoints: Sequence[UserViewpointCheckpoint],
    mode: str,
    context_size: int,
    max_tokens: int,
    disable_thinking: bool,
) -> List[Dict[str, Any]]:
    records = [checkpoint_prompt_record(checkpoint, mode, context_size) for checkpoint in checkpoints]
    default_source_by_checkpoint = {checkpoint_id(checkpoint): checkpoint.speaker for checkpoint in checkpoints}
    payload = call_json_model_with_sanitize_retry(
        api_base=api_base,
        api_key=api_key,
        model=model,
        prompt=prediction_prompt(sample_id, mode, records, context_size),
        max_tokens=max_tokens,
        disable_thinking=disable_thinking,
    )
    return normalize_prediction_items(
        sample_id,
        normalize_object_list(payload, "predictions"),
        mode,
        default_source_by_checkpoint=default_source_by_checkpoint,
    )


def predict_textual_checkpoint(
    api_base: str,
    api_key: str,
    model: str,
    checkpoint: UserViewpointCheckpoint,
    mode: str,
    context_size: int,
    max_tokens: int,
    disable_thinking: bool,
) -> Dict[str, Any]:
    predictions = predict_textual_sample(
        api_base=api_base,
        api_key=api_key,
        model=model,
        sample_id=checkpoint.sample_id,
        checkpoints=[checkpoint],
        mode=mode,
        context_size=context_size,
        max_tokens=max_tokens,
        disable_thinking=disable_thinking,
    )
    expected_id = checkpoint_id(checkpoint)
    for prediction in predictions:
        if clean_text(prediction.get("checkpoint_id")) == expected_id:
            return prediction
    if len(predictions) == 1:
        prediction = dict(predictions[0])
        prediction["checkpoint_id"] = expected_id
        prediction["_sample_id"] = checkpoint.sample_id
        prediction["_prediction_mode"] = mode
        return prediction
    raise RuntimeError(f"Missing prediction for {expected_id}")


def predict_turn_video_checkpoint(
    api_base: str,
    api_key: str,
    model: str,
    checkpoint: UserViewpointCheckpoint,
    upload_cache: Dict[str, Any],
    upload_cache_path: Path,
    context_size: int,
    max_tokens: int,
    disable_thinking: bool,
) -> Dict[str, Any]:
    if not checkpoint.clip_path:
        raise RuntimeError(f"Missing row clip for {checkpoint_id(checkpoint)}")
    clip_path = Path(checkpoint.clip_path)
    if not clip_path.exists():
        raise RuntimeError(f"Missing row clip file for {checkpoint_id(checkpoint)}: {clip_path}")
    prompt_record, video_urls = build_turn_video_prompt_bundle(
        checkpoint=checkpoint,
        context_size=context_size,
        upload_cache=upload_cache,
        upload_cache_path=upload_cache_path,
        api_key=api_key,
        model=model,
    )
    prompt = prediction_prompt(checkpoint.sample_id, "turn_video_audio", [prompt_record], context_size)
    payload = call_video_json_model_with_sanitize_retry(
        api_base=api_base,
        api_key=api_key,
        model=model,
        video_urls=video_urls,
        prompt=prompt,
        max_tokens=max_tokens,
        disable_thinking=disable_thinking,
    )
    predictions = normalize_prediction_items(
        checkpoint.sample_id,
        normalize_object_list(payload, "predictions"),
        "turn_video_audio",
        default_source_by_checkpoint={checkpoint_id(checkpoint): checkpoint.speaker},
    )
    if not predictions:
        raise RuntimeError(f"Missing video prediction for {checkpoint_id(checkpoint)}")
    prediction = predictions[0]
    prediction["_video_clip_path"] = checkpoint.clip_path
    prediction["_video_duration_sec"] = round(checkpoint.clip_duration_sec, 3)
    return prediction


def judge_checkpoint(
    *,
    api_base: str,
    api_key: str,
    model: str,
    sample_id: str,
    prediction: Dict[str, Any],
    checkpoint: UserViewpointCheckpoint,
    max_tokens: int,
    disable_thinking: bool,
) -> Dict[str, Any]:
    expected_id = checkpoint_id(checkpoint)
    payload = call_json_model_with_sanitize_retry(
        api_base=api_base,
        api_key=api_key,
        model=model,
        prompt=judge_prompt(sample_id, [prediction], [checkpoint]),
        max_tokens=max_tokens,
        disable_thinking=disable_thinking,
    )
    for grade in normalize_object_list(payload, "graded"):
        if clean_text(grade.get("checkpoint_id")) == expected_id:
            return grade
    grades = normalize_object_list(payload, "graded")
    if len(grades) == 1:
        grade = dict(grades[0])
        grade["checkpoint_id"] = expected_id
        return grade
    raise RuntimeError(f"Missing grade for {expected_id}")


def summarize_sample_details(sample_id: str, couple: str, details: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    checkpoint_count = len(details)
    if checkpoint_count == 0:
        return {
            "sample_id": sample_id,
            "couple": couple,
            "checkpoint_count": 0,
            "avg_viewpoint_total_score": 0.0,
            "avg_source_target_score": 0.0,
            "avg_viewpoint_score": 0.0,
            "details": [],
        }
    return {
        "sample_id": sample_id,
        "couple": couple,
        "checkpoint_count": checkpoint_count,
        "avg_viewpoint_total_score": round(
            sum(item["viewpoint_total_score"] for item in details) / checkpoint_count, 3
        ),
        "avg_source_target_score": round(sum(item["source_target_score"] for item in details) / checkpoint_count, 3),
        "avg_viewpoint_score": round(sum(item["viewpoint_score"] for item in details) / checkpoint_count, 3),
        "details": list(details),
    }


def normalize_failed_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in rows or []:
        if not isinstance(item, dict):
            continue
        sample_id = clean_text(item.get("sample_id"))
        error = clean_text(item.get("error"))
        if not sample_id or not error:
            continue
        normalized.append({"sample_id": sample_id, "error": error})
    return normalized


def validate_resume_payload(
    payload: Dict[str, Any],
    *,
    mode: str,
    prediction_model: str,
    judge_model: str,
    data_root: Path,
    checkpoints_per_sample: int,
    context_size: int,
) -> None:
    if clean_text(payload.get("task")) != "user_understanding_current_speaker_viewpoint":
        raise ValueError("Existing output_json task does not match evaluate_user_viewpoints.")
    existing_mode = clean_text(payload.get("mode"))
    if existing_mode and existing_mode != mode:
        raise ValueError(f"Existing output_json mode={existing_mode}, current mode={mode}.")
    existing_prediction_model = clean_text(payload.get("prediction_model"))
    if existing_prediction_model and existing_prediction_model != prediction_model:
        raise ValueError(
            f"Existing output_json prediction_model={existing_prediction_model}, current model={prediction_model}."
        )
    existing_judge_model = clean_text(payload.get("judge_model"))
    if existing_judge_model and existing_judge_model != judge_model:
        raise ValueError(f"Existing output_json judge_model={existing_judge_model}, current judge_model={judge_model}.")
    existing_data_root = clean_text(payload.get("data_root"))
    if existing_data_root and Path(existing_data_root).expanduser().resolve() != data_root:
        raise ValueError(f"Existing output_json data_root={existing_data_root}, current data_root={data_root}.")
    existing_checkpoints = payload.get("checkpoints_per_sample")
    if existing_checkpoints not in {"", None} and int(existing_checkpoints) != checkpoints_per_sample:
        raise ValueError(
            "Existing output_json checkpoints_per_sample=%s, current checkpoints_per_sample=%s."
            % (existing_checkpoints, checkpoints_per_sample)
        )
    existing_context_size = payload.get("context_size")
    if existing_context_size not in {"", None} and int(existing_context_size) != context_size:
        raise ValueError(
            "Existing output_json context_size=%s, current context_size=%s."
            % (existing_context_size, context_size)
        )


def load_existing_progress(
    output_json: Path,
    *,
    mode: str,
    prediction_model: str,
    judge_model: str,
    data_root: Path,
    checkpoints_per_sample: int,
    context_size: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not output_json.exists():
        return [], []

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    validate_resume_payload(
        payload,
        mode=mode,
        prediction_model=prediction_model,
        judge_model=judge_model,
        data_root=data_root,
        checkpoints_per_sample=checkpoints_per_sample,
        context_size=context_size,
    )

    sample_rows: List[Dict[str, Any]] = []
    for item in payload.get("samples", []) or []:
        if not isinstance(item, dict):
            continue
        sample_id = clean_text(item.get("sample_id"))
        couple = clean_text(item.get("couple"))
        details = item.get("details", [])
        if not sample_id or not couple or not isinstance(details, list):
            continue
        sample_rows.append(item)

    failed_rows = normalize_failed_rows(payload.get("failed", []) or [])
    return sample_rows, failed_rows


def build_output_payload(
    *,
    mode: str,
    prediction_model: str,
    prediction_api_base: str,
    judge_model: str,
    judge_api_base: str,
    data_root: Path,
    checkpoints_per_sample: int,
    context_size: int,
    all_sample_rows: Sequence[Dict[str, Any]],
    failed_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    normalized_failed_rows = normalize_failed_rows(failed_rows)
    return {
        "task": "user_understanding_current_speaker_viewpoint",
        "mode": mode,
        "prediction_model": prediction_model,
        "prediction_api_base": prediction_api_base,
        "judge_model": judge_model,
        "judge_api_base": judge_api_base,
        "data_root": str(data_root),
        "checkpoints_per_sample": checkpoints_per_sample,
        "context_size": context_size,
        "checkpoint_selection_policy": "Sample evenly from all eligible checkpoints across the full sample span; no late-turn cutoff is applied.",
        "checkpoint_eligibility": "Non-therapist rows with at least one viewpoints_attitudes item whose source matches primary_speaker.",
        "completed_sample_count": len(all_sample_rows),
        "failed_sample_count": len(normalized_failed_rows),
        "metric_denominator_policy": "Only completed samples in `samples` are counted in denominators. Failed or skipped samples listed in `failed` are excluded.",
        "input_policy": {
            "dialogue": [
                "primary_speaker",
                "participants",
                "target",
                "dialogue_cleaned",
                "background_dialogue",
            ],
            "audio": [
                "primary_speaker",
                "participants",
                "target",
                "dialogue_cleaned",
                "background_dialogue",
                "tone_of_voice",
            ],
            "visual": [
                "primary_speaker",
                "participants",
                "target",
                "dialogue_cleaned",
                "background_dialogue",
                "body_posture",
                "facial_expressions",
                "self_directed_behavior",
                "interaction_behavior",
            ],
            "audiovisual": [
                "primary_speaker",
                "participants",
                "target",
                "dialogue_cleaned",
                "background_dialogue",
                "body_posture",
                "facial_expressions",
                "self_directed_behavior",
                "interaction_behavior",
                "tone_of_voice",
            ],
            "turn_video_audio": [
                "primary_speaker",
                "participants",
                "target",
                "dialogue_cleaned",
                "background_dialogue",
                "row_level_video_clip",
                "prior_context_row_level_video_clips",
            ],
        },
        "timing_assumption": "Use the current speaking turn as the main evidence for the speaker's viewpoint(s) toward another on-screen person.",
        "samples": sorted(list(all_sample_rows), key=lambda item: clean_text(item.get("sample_id"))),
        "failed": sorted(normalized_failed_rows, key=lambda item: clean_text(item.get("sample_id"))),
    }


def persist_outputs(
    *,
    output_json: Path,
    output_xlsx: Path,
    mode: str,
    prediction_model: str,
    prediction_api_base: str,
    judge_model: str,
    judge_api_base: str,
    data_root: Path,
    checkpoints_per_sample: int,
    context_size: int,
    all_sample_rows: Sequence[Dict[str, Any]],
    failed_rows: Sequence[Dict[str, Any]],
    write_xlsx: bool,
) -> None:
    payload = build_output_payload(
        mode=mode,
        prediction_model=prediction_model,
        prediction_api_base=prediction_api_base,
        judge_model=judge_model,
        judge_api_base=judge_api_base,
        data_root=data_root,
        checkpoints_per_sample=checkpoints_per_sample,
        context_size=context_size,
        all_sample_rows=all_sample_rows,
        failed_rows=failed_rows,
    )
    atomic_write_json(output_json, payload)

    if not write_xlsx:
        return

    summary_rows = [
        {
            "sample_id": item["sample_id"],
            "couple": item["couple"],
            "checkpoint_count": item["checkpoint_count"],
            "avg_viewpoint_total_score": item["avg_viewpoint_total_score"],
            "avg_source_target_score": item["avg_source_target_score"],
            "avg_viewpoint_score": item["avg_viewpoint_score"],
        }
        for item in sorted(all_sample_rows, key=lambda item: clean_text(item.get("sample_id")))
    ]
    detail_rows: List[Dict[str, Any]] = []
    for item in sorted(all_sample_rows, key=lambda row: clean_text(row.get("sample_id"))):
        detail_rows.extend(item.get("details", []))
    write_result_workbook(output_xlsx, detail_rows, summary_rows)


def write_result_workbook(output_path: Path, detailed_rows: Sequence[Dict[str, Any]], summary_rows: Sequence[Dict[str, Any]]) -> None:
    workbook = Workbook()
    summary_sheet = workbook.active
    summary_sheet.title = "summary"
    summary_sheet.append(
        [
            "sample_id",
            "couple",
            "checkpoint_count",
            "avg_viewpoint_total_score",
            "avg_source_target_score",
            "avg_viewpoint_score",
        ]
    )
    for row in summary_rows:
        summary_sheet.append(
            [
                row["sample_id"],
                row["couple"],
                row["checkpoint_count"],
                row["avg_viewpoint_total_score"],
                row["avg_source_target_score"],
                row["avg_viewpoint_score"],
            ]
        )

    details_sheet = workbook.create_sheet("details")
    details_sheet.append(
        [
            "sample_id",
            "couple",
            "checkpoint_id",
            "row_index",
            "speaker",
            "turn_target",
            "timestamp",
            "dialogue",
            "gold_viewpoints",
            "predicted_viewpoints",
            "prediction_mode",
            "video_clip_path",
            "video_duration_sec",
            "source_target_score",
            "viewpoint_score",
            "viewpoint_total_score",
            "verdict",
            "prediction_reason",
            "judge_notes",
        ]
    )
    for row in detailed_rows:
        details_sheet.append(
            [
                row["sample_id"],
                row["couple"],
                row["checkpoint_id"],
                row["row_index"],
                row["speaker"],
                ", ".join(row["turn_target"]),
                row["timestamp"],
                row["dialogue"],
                json.dumps(row["gold_viewpoints"], ensure_ascii=False),
                json.dumps(row["predicted_viewpoints"], ensure_ascii=False),
                row["prediction_mode"],
                row.get("video_clip_path", ""),
                row.get("video_duration_sec", ""),
                row["source_target_score"],
                row["viewpoint_score"],
                row["viewpoint_total_score"],
                row["verdict"],
                row["prediction_reason"],
                row["judge_notes"],
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate current-speaker user viewpoints on source-target-viewpoint understanding."
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-xlsx", required=True)
    parser.add_argument("--sample-ids-file", default="")
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--checkpoints-per-sample", type=int, default=5)
    parser.add_argument("--context-size", type=int, default=3)
    parser.add_argument(
        "--mode",
        choices=sorted(VALID_MODES | set(MODE_ALIASES)),
        default="dialogue",
        help="dialogue | audio | visual | audiovisual | turn_video_audio. 'text' and 'frames' are kept as aliases.",
    )
    parser.add_argument("--upload-cache", default=str(DEFAULT_UPLOAD_CACHE))
    parser.add_argument(
        "--api-base",
        default=os.getenv("BAILIAN_API_BASE")
        or os.getenv("DASHSCOPE_API_BASE")
        or os.getenv("OPENAI_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
        or DEFAULT_API_BASE,
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("BAILIAN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY"),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("USER_VIEWPOINT_MODEL") or os.getenv("USER_UNDERSTANDING_MODEL") or "deepseek-v4-flash",
    )
    parser.add_argument("--judge-api-base", default="")
    parser.add_argument("--judge-api-key", default="")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--prediction-max-tokens", type=int, default=2200)
    parser.add_argument("--judge-max-tokens", type=int, default=1800)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--judge-disable-thinking", action="store_true")
    args = parser.parse_args()

    mode = MODE_ALIASES.get(args.mode, args.mode)
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode: {args.mode}")
    if args.context_size < 0:
        raise ValueError("context_size must be >= 0.")
    if not args.api_key:
        raise ValueError("Missing API key.")
    if not args.model:
        raise ValueError("Missing model.")

    data_root = Path(args.data_root).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_xlsx = Path(args.output_xlsx).expanduser().resolve()
    upload_cache_path = Path(args.upload_cache).expanduser().resolve()
    upload_cache = load_json_cache(upload_cache_path)

    api_base = normalize_api_base(args.api_base)
    judge_model = clean_text(args.judge_model) or "gpt-5.4"
    judge_base_candidate = args.judge_api_base
    if not judge_base_candidate and judge_model != args.model:
        judge_base_candidate = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or args.api_base
    judge_api_base = normalize_api_base(judge_base_candidate or args.api_base)
    judge_api_key = args.judge_api_key or (
        (os.getenv("OPENAI_API_KEY") or args.api_key) if judge_model != args.model else args.api_key
    )
    allowed_sample_ids = load_sample_id_filter(args.sample_ids_file)

    samples = load_json_samples(data_root)
    if allowed_sample_ids:
        samples = [sample for sample in samples if sample["sample_id"] in allowed_sample_ids]
    if args.limit_samples > 0:
        samples = samples[: args.limit_samples]

    all_sample_rows, failed = load_existing_progress(
        output_json,
        mode=mode,
        prediction_model=args.model,
        judge_model=judge_model,
        data_root=data_root,
        checkpoints_per_sample=args.checkpoints_per_sample,
        context_size=args.context_size,
    )
    completed_sample_ids = {
        clean_text(item.get("sample_id")) for item in all_sample_rows if clean_text(item.get("sample_id"))
    }
    failed_by_sample = {
        clean_text(item.get("sample_id")): item for item in failed if clean_text(item.get("sample_id"))
    }
    resumed_completed_count = len(completed_sample_ids)
    newly_completed_count = 0

    if completed_sample_ids or failed_by_sample:
        print(
            f"[resume] completed={len(completed_sample_ids)} failed={len(failed_by_sample)} output={output_json}",
            flush=True,
        )

    checkpoints_by_sample: Dict[str, List[Checkpoint]] = {}
    total_checkpoint_count = sum(
        int(item.get("checkpoint_count") or len(item.get("details", []) or []))
        for item in all_sample_rows
    )
    require_video = mode == "turn_video_audio"
    for sample in samples:
        sample_id = sample["sample_id"]
        if sample_id in completed_sample_ids:
            continue
        try:
            checkpoints = select_checkpoints(
                sample,
                args.checkpoints_per_sample,
                require_video=require_video,
                context_size=args.context_size,
            )
        except Exception:
            checkpoints = []
        if checkpoints:
            checkpoints_by_sample[sample_id] = checkpoints
            total_checkpoint_count += len(checkpoints)

    predict_progress = tqdm(
        total=total_checkpoint_count,
        initial=sum(len(item.get("details", []) or []) for item in all_sample_rows),
        desc="viewpoints predict total",
        unit="checkpoint",
        dynamic_ncols=True,
    )
    judge_progress = tqdm(
        total=total_checkpoint_count,
        initial=sum(len(item.get("details", []) or []) for item in all_sample_rows),
        desc="viewpoints judge total",
        unit="checkpoint",
        dynamic_ncols=True,
    )

    for sample in samples:
        sample_id = sample["sample_id"]
        if sample_id in completed_sample_ids:
            continue
        try:
            checkpoints = checkpoints_by_sample.get(sample_id, [])
            if not checkpoints:
                raise RuntimeError("no valid checkpoints")

            if mode == "turn_video_audio":
                predictions = []
                with tqdm(total=len(checkpoints), desc=f"viewpoints predict {sample_id}", unit="checkpoint", dynamic_ncols=True, disable=True) as progress:
                    for checkpoint in checkpoints:
                        predictions.append(
                            predict_turn_video_checkpoint(
                                api_base=api_base,
                                api_key=args.api_key,
                                model=args.model,
                                checkpoint=checkpoint,
                                upload_cache=upload_cache,
                                upload_cache_path=upload_cache_path,
                                context_size=args.context_size,
                                max_tokens=args.prediction_max_tokens,
                                disable_thinking=args.disable_thinking,
                            )
                        )
                        progress.update(1)
                        predict_progress.update(1)
                        progress.set_postfix_str(f"pred={len(predictions)}/{len(checkpoints)}")
                        predict_progress.set_postfix_str(f"sample={sample_id} pred={len(predictions)}/{len(checkpoints)}")
                        time.sleep(0.8)
            else:
                with tqdm(total=len(checkpoints), desc=f"viewpoints predict {sample_id}", unit="checkpoint", dynamic_ncols=True, disable=True) as progress:
                    predictions = predict_textual_sample(
                        api_base=api_base,
                        api_key=args.api_key,
                        model=args.model,
                        sample_id=sample_id,
                        checkpoints=checkpoints,
                        mode=mode,
                        context_size=args.context_size,
                        max_tokens=args.prediction_max_tokens,
                        disable_thinking=args.disable_thinking,
                    )
                    delta = min(len(predictions), len(checkpoints))
                    progress.update(delta)
                    predict_progress.update(delta)
                    progress.set_postfix_str(f"pred={len(predictions)}/{len(checkpoints)}")
                    predict_progress.set_postfix_str(f"sample={sample_id} pred={len(predictions)}/{len(checkpoints)}")

            with tqdm(total=len(checkpoints), desc=f"viewpoints judge {sample_id}", unit="checkpoint", dynamic_ncols=True, disable=True) as progress:
                prediction_map = {clean_text(item.get("checkpoint_id")): item for item in predictions}
                checkpoint_by_id = {checkpoint_id(checkpoint): checkpoint for checkpoint in checkpoints}
                expected_ids = set(checkpoint_by_id)
                missing_predictions = sorted(expected_ids - set(prediction_map))
                if missing_predictions:
                    for missing_id in missing_predictions:
                        checkpoint = checkpoint_by_id[missing_id]
                        if mode == "turn_video_audio":
                            prediction = predict_turn_video_checkpoint(
                                api_base=api_base,
                                api_key=args.api_key,
                                model=args.model,
                                checkpoint=checkpoint,
                                upload_cache=upload_cache,
                                upload_cache_path=upload_cache_path,
                                context_size=args.context_size,
                                max_tokens=args.prediction_max_tokens,
                                disable_thinking=args.disable_thinking,
                            )
                        else:
                            prediction = predict_textual_checkpoint(
                                api_base=api_base,
                                api_key=args.api_key,
                                model=args.model,
                                checkpoint=checkpoint,
                                mode=mode,
                                context_size=args.context_size,
                                max_tokens=min(args.prediction_max_tokens, 1200),
                                disable_thinking=args.disable_thinking,
                            )
                        prediction_map[clean_text(prediction.get("checkpoint_id")) or missing_id] = prediction
                        predict_progress.update(1)
                        predict_progress.set_postfix_str(f"sample={sample_id} fallback_pred={len(prediction_map)}/{len(checkpoints)}")
                        time.sleep(0.3)
                    missing_predictions = sorted(expected_ids - set(prediction_map))
                    if missing_predictions:
                        raise RuntimeError(f"missing predictions for {missing_predictions}")
                    predictions = [prediction_map[checkpoint_id(checkpoint)] for checkpoint in checkpoints]

                judge_payload = call_json_model_with_sanitize_retry(
                    api_base=judge_api_base,
                    api_key=judge_api_key,
                    model=judge_model,
                    prompt=judge_prompt(sample_id, predictions, checkpoints),
                    max_tokens=args.judge_max_tokens,
                    disable_thinking=args.judge_disable_thinking,
                )
                grades = normalize_object_list(judge_payload, "graded")
                grade_map = {clean_text(item.get("checkpoint_id")): item for item in grades}
                graded_count = len(set(grade_map) & expected_ids)
                progress.update(graded_count)
                judge_progress.update(graded_count)
                progress.set_postfix_str(f"graded={graded_count}/{len(checkpoints)}")
                judge_progress.set_postfix_str(f"sample={sample_id} graded={graded_count}/{len(checkpoints)}")
                missing_grades = sorted(expected_ids - set(grade_map))
                if missing_grades:
                    for missing_id in missing_grades:
                        checkpoint = checkpoint_by_id[missing_id]
                        last_grade_error: Optional[Exception] = None
                        for attempt in range(1, 4):
                            try:
                                grade = judge_checkpoint(
                                    api_base=judge_api_base,
                                    api_key=judge_api_key,
                                    model=judge_model,
                                    sample_id=sample_id,
                                    prediction=prediction_map[missing_id],
                                    checkpoint=checkpoint,
                                    max_tokens=min(args.judge_max_tokens, 900),
                                    disable_thinking=args.judge_disable_thinking,
                                )
                                break
                            except Exception as exc:
                                last_grade_error = exc
                                if attempt == 3:
                                    raise
                                time.sleep(2 * attempt)
                        else:
                            raise RuntimeError(str(last_grade_error))
                        grade_map[clean_text(grade.get("checkpoint_id")) or missing_id] = grade
                        progress.update(1)
                        judge_progress.update(1)
                        progress.set_postfix_str(f"fallback_graded={len(set(grade_map) & expected_ids)}/{len(checkpoints)}")
                        judge_progress.set_postfix_str(f"sample={sample_id} fallback_graded={len(set(grade_map) & expected_ids)}/{len(checkpoints)}")
                        time.sleep(0.3)
                    missing_grades = sorted(expected_ids - set(grade_map))
                    if missing_grades:
                        raise RuntimeError(f"missing grades for {missing_grades}")

            details: List[Dict[str, Any]] = []
            for checkpoint in checkpoints:
                key = checkpoint_id(checkpoint)
                prediction = prediction_map[key]
                grade = grade_map[key]
                source_target_score = grade.get("source_target_score", "")
                viewpoint_score = grade.get("viewpoint_score", "")
                total_score = grade.get("viewpoint_total_score", "")
                if source_target_score == "" or viewpoint_score == "" or total_score == "":
                    raise RuntimeError(f"missing judge score fields for {key}")
                details.append(
                    {
                        "sample_id": sample_id,
                        "couple": sample["couple"],
                        "checkpoint_id": key,
                        "row_index": checkpoint.row_index,
                        "speaker": checkpoint.speaker,
                        "turn_target": checkpoint.target,
                        "timestamp": checkpoint.timestamp,
                        "dialogue": checkpoint.dialogue,
                        "gold_viewpoints": checkpoint.gold_viewpoints,
                        "predicted_viewpoints": prediction.get("predicted_viewpoints", []),
                        "prediction_mode": clean_text(prediction.get("_prediction_mode")) or mode,
                        "video_clip_path": clean_text(prediction.get("_video_clip_path")),
                        "video_duration_sec": prediction.get("_video_duration_sec", ""),
                        "prediction_reason": clean_text(prediction.get("reason")),
                        "source_target_score": float(source_target_score),
                        "viewpoint_score": float(viewpoint_score),
                        "viewpoint_total_score": float(total_score),
                        "verdict": clean_text(grade.get("verdict")),
                        "judge_notes": clean_text(grade.get("notes")),
                    }
                )

            sample_row = summarize_sample_details(sample_id, sample["couple"], details)
            all_sample_rows.append(sample_row)
            completed_sample_ids.add(sample_id)
            newly_completed_count += 1
            failed_by_sample.pop(sample_id, None)
            persist_outputs(
                output_json=output_json,
                output_xlsx=output_xlsx,
                mode=mode,
                prediction_model=args.model,
                prediction_api_base=api_base,
                judge_model=judge_model,
                judge_api_base=judge_api_base,
                data_root=data_root,
                checkpoints_per_sample=args.checkpoints_per_sample,
                context_size=args.context_size,
                all_sample_rows=all_sample_rows,
                failed_rows=list(failed_by_sample.values()),
                write_xlsx=False,
            )
            if newly_completed_count % 20 == 0:
                print(
                    f"[progress] newly_completed={newly_completed_count} "
                    f"total_completed={resumed_completed_count + newly_completed_count} "
                    f"last_sample={sample_id} "
                    f"last_avg_total={sample_row['avg_viewpoint_total_score']} "
                    f"last_avg_source_target={sample_row['avg_source_target_score']} "
                    f"last_avg_viewpoint={sample_row['avg_viewpoint_score']}",
                    flush=True,
                )
        except Exception as exc:
            failed_by_sample[sample_id] = {"sample_id": sample_id, "error": str(exc)}
            persist_outputs(
                output_json=output_json,
                output_xlsx=output_xlsx,
                mode=mode,
                prediction_model=args.model,
                prediction_api_base=api_base,
                judge_model=judge_model,
                judge_api_base=judge_api_base,
                data_root=data_root,
                checkpoints_per_sample=args.checkpoints_per_sample,
                context_size=args.context_size,
                all_sample_rows=all_sample_rows,
                failed_rows=list(failed_by_sample.values()),
                write_xlsx=False,
            )
            print(f"[failed] {sample_id}: {exc}", flush=True)

    predict_progress.close()
    judge_progress.close()

    if newly_completed_count % 20 != 0 and newly_completed_count > 0:
        last_sample_row = all_sample_rows[-1]
        print(
            f"[progress] newly_completed={newly_completed_count} "
            f"total_completed={resumed_completed_count + newly_completed_count} "
            f"last_sample={last_sample_row['sample_id']} "
            f"last_avg_total={last_sample_row['avg_viewpoint_total_score']} "
            f"last_avg_source_target={last_sample_row['avg_source_target_score']} "
            f"last_avg_viewpoint={last_sample_row['avg_viewpoint_score']}",
            flush=True,
        )

    persist_outputs(
        output_json=output_json,
        output_xlsx=output_xlsx,
        mode=mode,
        prediction_model=args.model,
        prediction_api_base=api_base,
        judge_model=judge_model,
        judge_api_base=judge_api_base,
        data_root=data_root,
        checkpoints_per_sample=args.checkpoints_per_sample,
        context_size=args.context_size,
        all_sample_rows=all_sample_rows,
        failed_rows=list(failed_by_sample.values()),
        write_xlsx=True,
    )
    print(f"[written] {output_json}", flush=True)
    print(f"[written] {output_xlsx}", flush=True)


if __name__ == "__main__":
    main()
