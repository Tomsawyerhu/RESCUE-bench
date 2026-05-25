# RESCUE-BENCH Evaluation Code

This repository contains the evaluation scripts for **RESCUE-BENCH:
Relation-aware Emotional Support Conversation Understanding and Evaluation
Benchmark**. RESCUE-BENCH evaluates whether LLMs can understand evolving
multi-party relational dynamics and use them for relation-sensitive emotional
support decisions.

The benchmark contains two groups of tasks:

- **Relational Understanding**
  - **ER**: Emotion Recognition
  - **VP**: Viewpoint Prediction
  - **RPP**: Relation Pattern Prediction
- **Relation-Sensitive Support**
  - **ITP**: Intervention Time Prediction
  - **STP**: Support Target Prediction
  - **SSP**: Support Strategy Prediction

## Repository Layout

```text
task/
  eval_common.py
  multimodal_eval_common.py
  evaluate_intervention_timing_multimodal.py
  evaluate_intervention_timing.py
  evaluate_relation_cycle_prediction_multimodal.py
  evaluate_support_target_prediction_multimodal.py
  evaluate_strategy_prediction.py
  evaluate_strategy_prediction_multimodal.py
  evaluate_user_understanding.py
  evaluate_user_viewpoints.py
utils/
  project_env.py
data/
  <scenario_or_couple>/<sample>.json
```

`task/evaluate_strategy_prediction.py` is a helper module used by the SSP
evaluation script. It defines support strategy labels and shared parsing logic;
it is not normally run directly.

## Setup

Use Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install requests openpyxl tqdm bert-score torch transformers
```

`bert-score`, `torch`, and `transformers` are required for ER because
`evaluate_user_understanding.py` computes BERTScore. If you only run
classification or ranking tasks, these packages are not needed.

Copy `.env.example` to `.env` and fill in the endpoint and model you use:

```bash
cp .env.example .env
```

The scripts use OpenAI-compatible chat completion APIs. Supported environment
variables include:

```text
BAILIAN_API_BASE
BAILIAN_API_KEY
DASHSCOPE_API_KEY
OPENAI_API_BASE
OPENAI_BASE_URL
OPENAI_API_KEY
EVAL_MODEL
BAILIAN_MODEL
USER_UNDERSTANDING_MODEL
USER_VIEWPOINT_MODEL
```

For judge-based ER and VP evaluation, you may optionally set separate judge
credentials:

```text
JUDGE_API_BASE
JUDGE_API_KEY
JUDGE_MODEL
```

Avoid passing API keys directly on the command line, because command histories
and process listings can expose them. Prefer `.env` or shell environment
variables.

## Data Format

Most tasks read JSON samples from `--data-root`, recursively. Each sample is a
JSON file with a top-level `rows` list:

```json
{
  "rows": [
    {
      "start_time": 0.0,
      "end_time": 4.2,
      "primary_speaker": "Therapist",
      "target": ["Partner A"],
      "dialogue_cleaned": "What comes up for you when you hear that?",
      "utterance_type": "...",
      "background_dialogue": [],
      "tone_of_voice": "...",
      "body_posture": "...",
      "facial_expressions": "...",
      "self_directed_behavior": "...",
      "interaction_behavior": [],
      "internal_emotion": [],
      "viewpoints_attitudes": [],
      "support_strategy": [],
      "relation_cycle_state": "",
      "relation_cycle_reason": "",
      "relation_cycle_evidence_rows": []
    }
  ]
}
```

Fields are used according to the evaluation mode:

- `dialogue`: speaker, target, dialogue text, and structured dialogue context.
- `audio`: dialogue fields plus `tone_of_voice`.
- `visual`: dialogue fields plus posture, facial expression, self-directed
  behavior, and interaction behavior.
- `audiovisual`: dialogue, audio, and visual annotation fields.
- `turn_video_audio`: row-level video/audio clips. This mode is supported by
  ER, VP, and SSP when compatible video assets are available.

Sample IDs are derived from paths relative to `--data-root`. For example,
`data/dale_and_india/004_s03e06_ct09_segment05.json` becomes:

```text
dale_and_india/004_s03e06_ct09_segment05
```

To restrict evaluation to a subset, pass `--sample-ids-file`. The file may be a
plain text list, a JSON list, or a JSON object with `sample_ids`.

## Paper Metrics

The primary metrics follow the paper:

| Task | Type | Primary metrics |
| --- | --- | --- |
| ITP | Binary classification | Precision, Recall, F1 |
| RPP | Multiclass classification | Accuracy |
| STP | Ranking | Recall, MRR |
| SSP | Ranking | Recall, MRR |
| ER | Generation | LLM-as-judge, BERTScore |
| VP | Generation | LLM-as-judge, BERTScore |

For ranking tasks, the scripts ask the model to return a top-3 list and compute
Recall@1, Recall@2, Recall@3, and MRR. For paper-style reporting, use the top-1
recall field (`target_recall_at_1` for STP and `strategy_recall_at_1` for SSP)
as the main Rec value unless an experiment table explicitly states another k.

For RPP, the paper reports multiclass accuracy. In the script output, top-1
accuracy is represented by `relation_recall_at_1`, because it is computed using
the same top-k helper as ranking tasks. Treat this as RPP accuracy when
reporting the paper metric.

## Common CLI Arguments

Most evaluation scripts share these arguments:

```text
--data-root              Root directory containing sample JSON files.
--output-json            Path to write machine-readable results.
--output-xlsx            Path to write spreadsheet results.
--sample-ids-file        Optional subset of samples to evaluate.
--limit-samples          Optional debugging limit.
--context-size           Number of previous rows to include; -1 means all prior rows.
--mode                   dialogue, audio, visual, audiovisual, or turn_video_audio where supported.
--batch-size             Number of targets/checkpoints per model request.
--api-base               OpenAI-compatible API base URL.
--api-key                API key, preferably from environment variables.
--model                  Prediction model.
--max-tokens             Completion token budget.
--disable-thinking       Pass enable_thinking=false for compatible endpoints.
```

The JSON output is resumable for most scripts: if `--output-json` already exists
and its configuration matches the current run, completed samples are skipped.

## ITP: Intervention Time Prediction

**Paper task.** Given the prior conversation context at a candidate segment,
predict whether the therapist should intervene.

**Primary script.**

```bash
python task/evaluate_intervention_timing_multimodal.py \
  --data-root data \
  --mode dialogue \
  --context-size 3 \
  --output-json outputs/itp_dialogue.json \
  --output-xlsx outputs/itp_dialogue.xlsx
```

**Inputs.**

- `--data-root`: annotated sample JSON files.
- Candidate instances are constructed from non-therapist turns.
- Positive gold label `yes`: the actual next row is a therapist turn.
- Negative gold label `no`: the next row is still a non-therapist turn.
- `--max-candidates-per-class-per-sample`: optional per-sample class balancing
  cap. `0` means no cap beyond the script's default candidate construction.

**Supported modes.**

```text
dialogue, audio, visual, audiovisual
```

The alias `text` maps to `dialogue`.

**Expected model output.**

The model returns JSON predictions with:

```json
{
  "predictions": [
    {
      "candidate_id": "...",
      "should_speak": "yes",
      "confidence": 0.8,
      "reason": "..."
    }
  ]
}
```

**Outputs.**

- JSON fields: `overall`, `couples`, `samples`, `details`, `confusions`.
- XLSX sheets: `overall`, `couples`, `samples`, `details`, `confusions`.

**Paper metrics.**

- Precision, Recall, and F1 for the positive intervention-needed class.

**Additional script metrics.**

- `accuracy`
- `precision_yes`, `recall_yes`, `f1_yes`
- `precision_no`, `recall_no`, `f1_no`
- `balanced_accuracy`
- `tp`, `tn`, `fp`, `fn`

**Legacy script.**

`task/evaluate_intervention_timing.py` evaluates the same binary decision from a
prebuilt `--gold-json` file:

```bash
python task/evaluate_intervention_timing.py \
  --gold-json path/to/gold.json \
  --output-json outputs/itp_legacy.json \
  --output-xlsx outputs/itp_legacy.xlsx
```

Use the multimodal script above for the paper-style benchmark pipeline unless
you are reproducing an older prebuilt-gold experiment.

## RPP: Relation Pattern Prediction

**Paper task.** Given the conversation context, predict the current group-level
relation pattern, such as escalation, withdrawal, repair, or alignment.

**Script.**

```bash
python task/evaluate_relation_cycle_prediction_multimodal.py \
  --data-root data \
  --group-mode couple \
  --target-speaker-filter therapist \
  --mode dialogue \
  --context-size -1 \
  --output-json outputs/rpp_dialogue.json \
  --output-xlsx outputs/rpp_dialogue.xlsx
```

**Inputs.**

- `--data-root`: annotated sample JSON files.
- By default, gold relation labels are read from `rows[*].relation_cycle_state`
  in the dataset. Optional fields `relation_cycle_reason` and
  `relation_cycle_evidence_rows` are used as gold rationale/evidence metadata
  when present.
- `--relation-gold-glob`: optional legacy external JSON glob for older runs
  where relation-cycle labels were not yet merged into `--data-root`.
- `--group-mode`: `couple` or `family`, selecting the relation taxonomy.
- `--target-speaker-filter`:
  - `therapist`: evaluate only therapist rows with non-empty relation-cycle gold.
  - `any`: evaluate any row with non-empty relation-cycle gold.
- `--include-current-turn`: optional flag to include the target turn itself.
  Without it, the task uses prior context only.

**Supported modes.**

```text
dialogue, audio, visual, audiovisual
```

The alias `text` maps to `dialogue`.

**Expected model output.**

The model returns top-3 relation labels:

```json
{
  "predictions": [
    {
      "turn_id": "...",
      "relation_top3": ["pursue_withdraw", "attack_attack", "mixed_transition"],
      "reason": "..."
    }
  ]
}
```

**Outputs.**

- JSON fields: `overall`, `couples`, `samples`, `details`,
  `relation_confusions`, `top1_class_metrics`.
- XLSX sheets: `overall`, `couples`, `samples`, `details`,
  `relation_confusions`, `class_metrics`.

**Paper metric.**

- Multiclass accuracy.

**How to read script metrics for the paper.**

- Use `relation_recall_at_1` as top-1 accuracy.

**Additional script metrics.**

- `relation_recall_at_2`
- `relation_recall_at_3`
- `relation_mrr`
- `top1_macro_precision`
- `top1_macro_recall`
- `top1_macro_f1`
- Per-label precision, recall, F1, and support in `top1_class_metrics`.

## STP: Support Target Prediction

**Paper task.** Once intervention is needed, predict whom the therapist should
support: an individual participant, a dyad/subgroup, or the whole couple/family.

**Script.**

```bash
python task/evaluate_support_target_prediction_multimodal.py \
  --data-root data \
  --mode dialogue \
  --context-size 3 \
  --output-json outputs/stp_dialogue.json \
  --output-xlsx outputs/stp_dialogue.xlsx
```

**Inputs.**

- `--data-root`: annotated sample JSON files.
- Evaluated rows are therapist turns with non-empty `support_strategy`
  annotations.
- Gold target is read from the first `support_strategy` target when available,
  with row-level `target` as fallback.
- Candidate target labels are inferred from participants plus group-level labels
  such as `Couple` or `Family`.

**Supported modes.**

```text
dialogue, audio, visual, audiovisual
```

The alias `text` maps to `dialogue`.

**Expected model output.**

```json
{
  "predictions": [
    {
      "turn_id": "...",
      "target_top3": ["Partner A", "Couple", "Partner B"],
      "reason": "..."
    }
  ]
}
```

**Outputs.**

- JSON fields: `overall`, `couples`, `samples`, `details`,
  `target_confusions`.
- XLSX sheets: `overall`, `couples`, `samples`, `details`,
  `target_confusions`.

**Paper metrics.**

- Recall and MRR.

**Script metric fields.**

- `target_recall_at_1`
- `target_recall_at_2`
- `target_recall_at_3`
- `target_mrr`

## SSP: Support Strategy Prediction

**Paper task.** Given the context and the selected support target, predict which
support strategy the therapist should use.

**Script.**

```bash
python task/evaluate_strategy_prediction_multimodal.py \
  --data-root data \
  --mode dialogue \
  --context-size 3 \
  --output-json outputs/ssp_dialogue.json \
  --output-xlsx outputs/ssp_dialogue.xlsx
```

**Inputs.**

- `--data-root`: annotated sample JSON files.
- Evaluated rows are therapist turns with non-empty `support_strategy`.
- The gold support target is provided to the model as `known_support_target`.
  The script evaluates strategy ranking only.
- Strategy taxonomies are defined in `task/evaluate_strategy_prediction.py`.
- `--include-target-internal-emotion`: optionally includes the most recent
  available internal-emotion annotation for the known support target.

**Supported modes.**

```text
dialogue, audio, visual, audiovisual, turn_video_audio
```

Aliases:

```text
text -> dialogue
frames -> turn_video_audio
speaker_frames_audio -> turn_video_audio
```

**Expected model output.**

```json
{
  "predictions": [
    {
      "turn_id": "...",
      "strategy_top3": ["track", "reframe", "evoke"],
      "reason": "..."
    }
  ]
}
```

**Outputs.**

- JSON fields: `overall`, `couples`, `samples`, `details`,
  `strategy_confusions`, `taxonomy`.
- XLSX sheets: `overall`, `couples`, `samples`, `details`,
  `strategy_confusions`.

**Paper metrics.**

- Recall and MRR.

**Script metric fields.**

- `strategy_recall_at_1`
- `strategy_recall_at_2`
- `strategy_recall_at_3`
- `strategy_mrr`

## ER: Emotion Recognition

**Paper task.** Given the dialogue history and multimodal evidence for a
segment, predict the current speaker's internal emotion and emotional intensity.

**Script.**

```bash
python task/evaluate_user_understanding.py \
  --data-root data \
  --mode dialogue \
  --context-size 3 \
  --checkpoints-per-sample 5 \
  --output-json outputs/er_dialogue.json \
  --output-xlsx outputs/er_dialogue.xlsx
```

**Inputs.**

- `--data-root`: annotated sample JSON files.
- Checkpoints are rows with usable `dialogue_cleaned` and gold
  `internal_emotion`.
- `--min-dialogue-words`: minimum words in the current turn for checkpoint
  eligibility.
- `--checkpoints-per-sample`: maximum checkpoints selected per sample.
- `--judge-model`: optional separate judge model. If omitted, the script uses
  the prediction model or compatible defaults.
- `--bertscore-model-type`: BERTScore backbone, default
  `distilbert-base-uncased`.

**Supported modes.**

```text
dialogue, audio, visual, audiovisual, turn_video_audio
```

Aliases:

```text
text -> dialogue
frames -> turn_video_audio
speaker_frames_audio -> turn_video_audio
```

**Expected prediction output.**

```json
{
  "predictions": [
    {
      "checkpoint_id": "...",
      "predicted_internal_emotion": "hurt and defensive",
      "predicted_intensity_score": 6,
      "reason": "..."
    }
  ]
}
```

**Judge output.**

The judge returns an `emotion_score` from 1 to 5. Intensity is not judged by the
LLM; it is computed by formula:

```text
intensity_score_match = max(1, 5 - abs(predicted_intensity - gold_intensity))
```

**Outputs.**

- JSON fields: `samples`, `failed`, configuration metadata, and per-sample
  `details`.
- XLSX sheets: `summary`, `details`.

**Paper metrics.**

- LLM-as-judge score.
- BERTScore.

**Script metric fields.**

- `avg_emotion_score`: average 1-5 LLM judge emotion score.
- `avg_intensity_match`: average formula-based intensity match score.
- `avg_bertscore_f1`: average BERTScore F1.
- Detail-level fields include `emotion_score`, `intensity_score_match`, and
  `bertscore_f1`.

## VP: Viewpoint Prediction

**Paper task.** Given the current speaker and context, predict directed
interpersonal viewpoints: who holds what view toward whom.

**Script.**

```bash
python task/evaluate_user_viewpoints.py \
  --data-root data \
  --mode dialogue \
  --context-size 3 \
  --checkpoints-per-sample 5 \
  --output-json outputs/vp_dialogue.json \
  --output-xlsx outputs/vp_dialogue.xlsx
```

**Inputs.**

- `--data-root`: annotated sample JSON files.
- Checkpoints are non-therapist rows with at least one
  `viewpoints_attitudes` item whose source matches `primary_speaker`.
- `--checkpoints-per-sample`: maximum checkpoints selected per sample.
- `--judge-model`: optional separate judge model.

**Supported modes.**

```text
dialogue, audio, visual, audiovisual, turn_video_audio
```

Aliases:

```text
text -> dialogue
frames -> turn_video_audio
speaker_frames_audio -> turn_video_audio
```

**Expected prediction output.**

```json
{
  "predictions": [
    {
      "checkpoint_id": "...",
      "predicted_viewpoints": [
        {
          "source": "Partner A",
          "target": "Partner B",
          "viewpoint": "he is dismissive of her concerns"
        }
      ],
      "reason": "..."
    }
  ]
}
```

**Judge output.**

The judge scores two components:

- `source_target_score`: 0-2, whether source-target pairings are recovered.
- `viewpoint_score`: 0-4, semantic match of viewpoint descriptions.

The total score is:

```text
viewpoint_total_score = source_target_score + viewpoint_score
```

with range 0-6.

**Outputs.**

- JSON fields: `samples`, `failed`, configuration metadata, and per-sample
  `details`.
- XLSX sheets: `summary`, `details`.

**Paper metrics.**

- LLM-as-judge score.
- BERTScore.

**Script metric fields.**

- `avg_viewpoint_total_score`
- `avg_source_target_score`
- `avg_viewpoint_score`
- Detail-level fields include `source_target_score`, `viewpoint_score`, and
  `viewpoint_total_score`.

Note: the current VP script implements the LLM-as-judge score. If BERTScore is
reported for VP in a paper table, compute it consistently with the paper's VP
BERTScore protocol.

## Multimodal Video Mode

`turn_video_audio` mode uses row-level video/audio clips when the released data
package includes compatible video assets. Each video-asset manifest row should
include at least:

```json
{
  "index": 1,
  "clip_path": "path/to/row_001.mp4",
  "duration_sec": 3.2
}
```

The video upload helper caches uploaded object URLs. Default cache files include:

```text
analysis/multimodal_video_upload_cache.json
analysis/user_understanding_video_upload_cache.json
analysis/user_viewpoints_video_upload_cache.json
```

Do not commit upload cache files. They may contain local absolute paths and
cloud object URLs.

## Output Hygiene

The scripts may write absolute local paths into result JSON/XLSX files, such as
`data_root`, `gold_json`, `video_clip_path`, or upload cache keys. Before
publishing generated outputs, scrub machine-specific paths and do not publish
API keys, `.env`, or upload caches.

Recommended `.gitignore` entries:

```gitignore
.env
analysis/*upload_cache*.json
outputs/
results/
*.xlsx
```

## Reproducing All Six Tasks

Example commands for dialogue-only evaluation:

```bash
mkdir -p outputs

python task/evaluate_intervention_timing_multimodal.py \
  --data-root data --mode dialogue --context-size 3 \
  --output-json outputs/itp_dialogue.json \
  --output-xlsx outputs/itp_dialogue.xlsx

python task/evaluate_relation_cycle_prediction_multimodal.py \
  --data-root data --mode dialogue --context-size -1 \
  --group-mode couple \
  --output-json outputs/rpp_dialogue.json \
  --output-xlsx outputs/rpp_dialogue.xlsx

python task/evaluate_support_target_prediction_multimodal.py \
  --data-root data --mode dialogue --context-size 3 \
  --output-json outputs/stp_dialogue.json \
  --output-xlsx outputs/stp_dialogue.xlsx

python task/evaluate_strategy_prediction_multimodal.py \
  --data-root data --mode dialogue --context-size 3 \
  --output-json outputs/ssp_dialogue.json \
  --output-xlsx outputs/ssp_dialogue.xlsx

python task/evaluate_user_understanding.py \
  --data-root data --mode dialogue --context-size 3 \
  --checkpoints-per-sample 5 \
  --output-json outputs/er_dialogue.json \
  --output-xlsx outputs/er_dialogue.xlsx

python task/evaluate_user_viewpoints.py \
  --data-root data --mode dialogue --context-size 3 \
  --checkpoints-per-sample 5 \
  --output-json outputs/vp_dialogue.json \
  --output-xlsx outputs/vp_dialogue.xlsx
```

Use `--sample-ids-file` and `--limit-samples` for smoke tests before running the
full benchmark.
