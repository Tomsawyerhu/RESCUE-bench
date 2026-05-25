import json
from typing import Any, Dict, List, Sequence, Tuple

from task.eval_common import (
    DEFAULT_API_BASE,
    call_json_model,
    canonical_target,
    clean_text,
    ensure_unique_ordered,
    group_label_for_mode,
    load_data_samples,
    load_sample_id_filter,
    mask_sensitive_terms,
    normalize_api_base,
    rank_metrics,
    target_label,
)

COUPLE_STRATEGY_DEFINITIONS: Dict[str, str] = {
    "counterbalance": "Restore relational symmetry by bringing a sidelined participant back into the interaction.",
    "safeguard": "Protect a vulnerable participant from being overrun, shamed, cornered, or overwhelmed in the moment.",
    "goal_align": "Orient participants toward a shared task, purpose, or therapeutic goal.",
    "track": "Explicitly follow the live interaction sequence, negative cycle, or relational pattern in the room.",
    "reframe": "Shift blame or a rigid story into a relational frame, interaction pattern, or shared dilemma.",
    "evoke": "Invite deeper vulnerability, hurt, fear, longing, shame, or need rather than staying at the surface level.",
    "enact": "Help one participant say something directly to another participant in the room.",
    "join": "Foster softer empathic contact and emotional joining between participants.",
    "detach": "Help participants step back together and view the problem as a shared pattern rather than fight as opponents.",
    "repair": "Soften rupture, clarify intent, apologize, reconnect, or otherwise repair relational strain.",
}

FAMILY_STRATEGY_DEFINITIONS: Dict[str, str] = {
    "safeguard": "Protect safety or vulnerability when a child, parent, or other family member is being overrun, shamed, threatened, scapegoated, or emotionally overwhelmed.",
    "join": "Build trust, reduce defensiveness, or strengthen the therapist's working alliance so the family can stay engaged in the conversation.",
    "track": "Explicitly name or follow the live family interaction pattern, escalation sequence, coalition, or recurring relational cycle in the room.",
    "counterbalance": "Rebalance voice, participation, or influence by bringing a less-heard or lower-power family member back into the interaction.",
    "enact": "Move family members into direct in-room interaction, asking one person to say, hear, or respond to another rather than only speaking through the therapist.",
    "boundary": "Clarify or restructure boundaries, roles, hierarchy, coalitions, or subsystem functioning, especially across generations or between parents and children.",
    "reframe": "Shift the meaning of the problem away from individual blame by reframing it systemically or externalizing it as a shared family challenge.",
    "repair": "Support apology, acknowledgement of hurt, reconnection, or relational healing after rupture, injury, or disconnection.",
}

STRATEGY_DEFINITIONS_BY_MODE: Dict[str, Dict[str, str]] = {
    "couple": COUPLE_STRATEGY_DEFINITIONS,
    "family": FAMILY_STRATEGY_DEFINITIONS,
}


def batched(seq: Sequence[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    if size <= 0:
        raise ValueError("batch size must be >= 1")
    return [list(seq[i : i + size]) for i in range(0, len(seq), size)]


def current_row_strategy(row: Dict[str, Any], group_mode: str = "") -> Tuple[str, Tuple[str, ...], str]:
    resolved_group_mode = clean_text(group_mode)
    if resolved_group_mode not in STRATEGY_DEFINITIONS_BY_MODE:
        resolved_group_mode = "family" if "Family" in (row.get("target") or []) else "couple"
    strategy_definitions = STRATEGY_DEFINITIONS_BY_MODE[resolved_group_mode]
    items = row.get("support_strategy") or []
    if not items:
        return "", tuple(), "none"
    first = items[0] if isinstance(items[0], dict) else {}
    strategy = clean_text(first.get("strategy_type"))
    if strategy not in strategy_definitions:
        return "", tuple(), "none"
    target = canonical_target(first.get("target") or [])
    if target:
        return strategy, target, "support_strategy"
    fallback_target = canonical_target(row.get("target") or [])
    if fallback_target:
        return strategy, fallback_target, "row_target_fallback"
    return strategy, tuple(), "none"


def iter_prediction_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_predictions = payload.get("predictions", [])
    if isinstance(raw_predictions, dict):
        raw_predictions = [raw_predictions]
    if not isinstance(raw_predictions, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in raw_predictions:
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


def strategy_definitions_for_sample(sample: Dict[str, Any]) -> Dict[str, str]:
    group_mode = clean_text(sample.get("group_mode") or "couple")
    return STRATEGY_DEFINITIONS_BY_MODE[group_mode]


def group_label_for_sample(sample: Dict[str, Any]) -> str:
    return group_label_for_mode(clean_text(sample.get("group_mode") or "couple"))
