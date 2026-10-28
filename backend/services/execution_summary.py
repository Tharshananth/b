"""
Turns a raw executionInfo document into a compact, derived summary.

The stored document is a nested state machine with no overall status field, so
"is my scan done?" has to be computed by walking the stage tree. That walk is
deterministic and belongs in code, not in a model call. The LLM receives this
summary rather than the raw document: smaller, ordered, and with durations and
anomalies already worked out.
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Pipeline order. A stage absent from the document has not been reached.
STAGE_ORDER = [
    "dataUpload",
    "dicomToNifti",
    "preprocess",
    "preprocessStatusSet",
    "inference",
    "inferenceStatusSet",
]

# Event markers, not pipeline stages. Verified against all 88 UAT records:
# both are 'Started' in 100% of documents and carry an 'end' in none of them.
# They record that a status transition fired, so they never complete and must
# be excluded from stall detection and from the overall verdict - otherwise
# every job in the collection reads as permanently stalled.
MARKER_STAGES = frozenset({"preprocessStatusSet", "inferenceStatusSet"})

# Timestamps are stored as day-first local strings with no timezone.
TS_FORMAT = "%d/%m/%Y, %I:%M:%S %p"

# A running stage older than this, or a finished stage that took longer, is
# reported as stalled.
STALL_AFTER = timedelta(hours=24)

SUCCESS = "success"
FAILED = "failed"
RUNNING = "running"
UNKNOWN = "unknown"


def parse_ts(value: Optional[str]) -> Optional[datetime]:
    """Parse a stored timestamp. Returns None rather than raising."""
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, TS_FORMAT)
    except ValueError:
        logger.warning("unparseable timestamp: %r", value)
        return None


def classify(state: Any) -> str:
    """
    Map a status string onto an outcome.

    The full status vocabulary is not yet known, so this matches on substrings
    and returns UNKNOWN rather than guessing. Unknown values are collected by
    the caller so the vocabulary can be filled in from real data.
    """
    if not isinstance(state, str):
        return UNKNOWN
    s = state.lower()
    if "fail" in s or "error" in s or "abort" in s or "cancel" in s:
        return FAILED
    if "success" in s or "complete" in s or "done" in s or "finish" in s:
        return SUCCESS
    if "start" in s or "running" in s or "progress" in s or "queue" in s:
        return RUNNING
    return UNKNOWN


def _flatten(status: Dict[str, Any], timespan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Walk the status tree into an ordered flat list of stages.

    A stage value is either a status string (dataUpload: 'UploadSuccess') or an
    object of sub-tasks keyed by modality or task name (dicomToNifti: {dMRI,
    sMRI}). Both shapes flatten to the same record.
    """
    stages: List[Dict[str, Any]] = []
    seen = set()

    def add(stage: str, task: Optional[str], state: Any, span: Any):
        span = span if isinstance(span, dict) else {}
        start = parse_ts(span.get("start"))
        end = parse_ts(span.get("end"))
        duration = (end - start).total_seconds() if start and end else None
        stages.append({
            "stage": stage,
            "task": task,
            "state": state,
            "outcome": classify(state),
            "is_marker": stage in MARKER_STAGES,
            "start": start,
            "end": end,
            "duration_seconds": duration,
        })

    ordered = [s for s in STAGE_ORDER if s in status]
    ordered += [s for s in status if s not in STAGE_ORDER]  # unknown stages last

    for stage in ordered:
        if stage in seen:
            continue
        seen.add(stage)
        state = status[stage]
        span = timespan.get(stage, {}) if isinstance(timespan, dict) else {}

        if isinstance(state, dict):
            for task, task_state in state.items():
                task_span = span.get(task, {}) if isinstance(span, dict) else {}
                add(stage, task, task_state, task_span)
        else:
            add(stage, None, state, span)

    return stages


def _find_problems(
    stages: List[Dict[str, Any]],
    expected_modalities: List[str],
    now: Optional[datetime],
) -> List[str]:
    """Human-readable anomalies, worked out from the stage list."""
    problems: List[str] = []

    for st in stages:
        if st["is_marker"]:
            continue
        label = st["stage"] + ("." + st["task"] if st["task"] else "")

        if st["outcome"] == FAILED:
            problems.append(label + " failed with status '" + str(st["state"]) + "'")

        elif st["outcome"] == RUNNING and st["start"] and not st["end"]:
            age = (now - st["start"]) if now else None
            if age and age > STALL_AFTER:
                problems.append(
                    label + " started " + _humanize(age.total_seconds())
                    + " ago and has not finished"
                )

        if st["duration_seconds"] and st["duration_seconds"] > STALL_AFTER.total_seconds():
            problems.append(
                label + " took " + _humanize(st["duration_seconds"])
                + ", far longer than expected"
            )

        if st["outcome"] == UNKNOWN and st["state"] is not None:
            problems.append(label + " has an unrecognised status '" + str(st["state"]) + "'")

    # Timestamps that run backwards mean the record cannot be trusted for timing.
    # Only compare across different stages - tasks within one stage run in
    # parallel and legitimately start in any order.
    timed = [s for s in stages if s["start"]]
    for a, b in zip(timed, timed[1:]):
        if b["stage"] != a["stage"] and b["start"] < a["start"]:
            problems.append(
                "timestamps are out of order: "
                + b["stage"] + " starts before " + a["stage"]
            )
            break

    # A modality listed in dockerArray that produced no stage never ran.
    # dockerArray names acquisition modalities (sMRI, fMRI, dMRI) plus the
    # pseudo-entry sMRI_Inferencing, which is satisfied by the inference stage
    # rather than by a preprocess task of that name.
    ran = {s["task"] for s in stages if s["task"]}
    inference_ran = any(s["stage"] == "inference" for s in stages)

    for mod in expected_modalities:
        if mod.endswith("_Inferencing"):
            if not inference_ran:
                problems.append(
                    "inference was expected for '" + mod + "' but never started"
                )
            continue
        # preprocess names tasks as <modality>_<run>, e.g. fMRI_1, fMRI_2
        if not any(t == mod or t.startswith(mod + "_") for t in ran if t):
            problems.append("expected modality '" + mod + "' has no pipeline stages")

    return problems


def explain_failure(state: str, stage: str) -> Dict[str, str]:
    """
    Turn a failure status string into a cause and a suggested next step.

    The stored strings encode the reason inline, e.g.
        UploadFail-ConsentForms-sMRI T1c-T1w No further processing
        UploadFail-sMRI T1c-flair
        Failed-internalPreprocessingError
        InternalError
    so the cause is parsed out rather than guessed. Anything unrecognised falls
    back to reporting the raw string, never to inventing an explanation.
    """
    raw = str(state)
    s = raw.lower()

    # Terminal marker the pipeline appends when it will not retry.
    terminal = "no further processing" in s
    body = raw
    for marker in ("No further processing", "no further processing"):
        body = body.replace(marker, "").strip(" -")

    if "consentforms" in s.replace(" ", ""):
        series = body.split("-", 2)[-1].replace("-", ", ").strip()
        return {
            "cause": "consent forms are missing for " + (series or "the uploaded series"),
            "action": (
                "Upload the consent forms for those series, then re-run the case."
                + (" The pipeline stopped and will not retry on its own."
                   if terminal else "")
            ),
        }

    if raw.startswith("UploadFail"):
        series = body.split("-", 1)[-1].replace("-", ", ").strip()
        return {
            "cause": "upload failed for " + (series or "one or more series"),
            "action": "Re-upload the affected series and check they are complete "
                      "and correctly labelled.",
        }

    if "internal" in s and ("error" in s or "fail" in s):
        stage_names = {
            "preprocess": "preprocessing",
            "inference": "inferencing",
            "dicomToNifti": "DICOM to NIfTI conversion",
            "dataUpload": "upload",
        }
        if "preprocess" in s:
            which = "preprocessing"
        elif "inferenc" in s:
            which = "inferencing"
        else:
            which = stage_names.get(stage, stage)
        return {
            "cause": "an internal " + which + " error on the platform side",
            "action": "This is not something the upload can fix. Raise it with "
                      "BrainSightAI support with the job ID.",
        }

    return {
        "cause": "the pipeline reported '" + raw + "'",
        "action": "Raise this with BrainSightAI support with the job ID.",
    }


def _humanize(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 90:
        return str(seconds) + " seconds"
    if seconds < 5400:
        return str(round(seconds / 60)) + " minutes"
    if seconds < 172800:
        return str(round(seconds / 3600, 1)) + " hours"
    return str(round(seconds / 86400, 1)) + " days"


def summarize(doc: Dict[str, Any], now: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Derive a compact summary from one executionInfo document.

    `now` is injected so stall detection is testable; defaults to the local
    clock, matching the naive local timestamps in the data.
    """
    now = now or datetime.now()
    status = doc.get("status") or {}
    timespan = doc.get("timespan") or {}
    modalities = doc.get("dockerArray") or []

    stages = _flatten(status, timespan)
    problems = _find_problems(stages, modalities, now)

    real = [s for s in stages if not s["is_marker"]]
    outcomes = [s["outcome"] for s in real]
    if FAILED in outcomes:
        overall = FAILED
    elif RUNNING in outcomes:
        overall = "stalled" if problems else RUNNING
    elif outcomes and all(o == SUCCESS for o in outcomes):
        overall = "completed"
    else:
        overall = UNKNOWN

    incomplete = [s for s in real if s["outcome"] != SUCCESS]
    current = None
    if incomplete:
        current = incomplete[0]["stage"]
        if incomplete[0]["task"]:
            current += "." + incomplete[0]["task"]

    started = min((s["start"] for s in stages if s["start"]), default=None)
    finished = max((s["end"] for s in stages if s["end"]), default=None)

    # Diagnosis: one entry per distinct failure cause, not per failed task.
    # Three tasks failing on the same internal error is one problem to report.
    diagnosis: List[Dict[str, Any]] = []
    for st in stages:
        if st["is_marker"] or st["outcome"] != FAILED:
            continue
        exp = explain_failure(st["state"], st["stage"])
        existing = next(
            (d for d in diagnosis if d["cause"] == exp["cause"]), None
        )
        label = st["stage"] + ("." + st["task"] if st["task"] else "")
        if existing:
            existing["affected"].append(label)
        else:
            diagnosis.append({
                "cause": exp["cause"],
                "action": exp["action"],
                "affected": [label],
                "raw_status": st["state"],
            })

    return {
        "job_id": doc.get("_id"),
        "patient_id": doc.get("patientID"),
        "study_id": doc.get("studyID"),
        "trial_id": doc.get("trialID"),
        "execution_id": doc.get("executionID"),
        "disorder": doc.get("disorder"),
        "modalities": modalities,
        "overall": overall,
        "current_stage": current,
        "started_at": started.isoformat() if started else None,
        "last_activity_at": finished.isoformat() if finished else None,
        "elapsed": _humanize((now - started).total_seconds()) if started else None,
        "diagnosis": diagnosis,
        "problems": problems,
        "stages": [
            {
                "stage": s["stage"],
                "task": s["task"],
                "state": s["state"],
                "outcome": s["outcome"],
                "duration": (
                    _humanize(s["duration_seconds"])
                    if s["duration_seconds"] is not None else None
                ),
            }
            for s in stages
        ],
    }


def collect_status_vocabulary(docs: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Gather every distinct status string per stage across documents.

    Used to discover the real vocabulary - especially the failure values, which
    the UAT sample does not contain.
    """
    vocab: Dict[str, set] = {}
    for doc in docs:
        for st in _flatten(doc.get("status") or {}, {}):
            vocab.setdefault(st["stage"], set()).add(str(st["state"]))
    return {k: sorted(v) for k, v in sorted(vocab.items())}
