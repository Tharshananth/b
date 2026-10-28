"""
Job status service — the `get_job_status` tool's implementation.

This is Agent 2 (scoped retrieval) and Agent 3 (scope validation) from the MVP
plan, both as deterministic code rather than model calls. The LLM chooses the
*search terms*; it never chooses the *scope*. Scope arrives from the session via
TenantScope and is injected server-side, after any model-supplied filter is
applied, so a model-supplied field can never widen the query.
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from pymongo.errors import PyMongoError

from config import get_config
from database.mongo import get_job_db

logger = logging.getLogger(__name__)


class ScopeViolation(Exception):
    """A document outside the caller's tenant scope was returned. Always fatal."""


@dataclass(frozen=True)
class TenantScope:
    """
    Resolved identity for one request. Built server-side from the authenticated
    session — never from the chat message, the LLM, or the request body.
    """
    hospital: str
    study_id: Optional[str] = None
    patient_id: Optional[str] = None

    def as_filter(self) -> Dict[str, Any]:
        f: Dict[str, Any] = {get_config().job_db.tenant_field: self.hospital}
        if self.study_id:
            f["studyID"] = self.study_id
        if self.patient_id:
            f["patientID"] = self.patient_id
        return f


# Fields the LLM is permitted to filter on. Anything else is dropped, not merged.
QUERYABLE_FIELDS = frozenset({
    "patientID", "executionID", "studyID", "trialID", "disorder",
})

# Full documents are returned. dockerArray was excluded at first as "heavy", but
# it is two short strings naming the expected modalities, and the summariser
# needs it to notice that an expected pipeline never ran.
PROJECTION = None


def get_job_status(
    scope: TenantScope,
    *,
    patient_id: Optional[str] = None,
    execution_id: Optional[str] = None,
    study_id: Optional[str] = None,
    trial_id: Optional[int] = None,
    disorder: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch execution records visible to `scope`.

    Raises ScopeViolation if the database returns anything outside the scope —
    fail closed, do not filter-and-continue. Raises PyMongoError on DB failure
    so the caller can tell the user the job system is unreachable rather than
    answering as if nothing were wrong.
    """
    cfg = get_config().job_db
    limit = min(limit or cfg.default_limit, cfg.max_limit)

    # 1. Model-supplied narrowing, whitelisted.
    candidate = {
        "patientID": patient_id,
        "executionID": execution_id,
        "studyID": study_id,
        "trialID": trial_id,
        "disorder": disorder,
    }
    query = {
        k: v for k, v in candidate.items()
        if v is not None and k in QUERYABLE_FIELDS
    }

    # 2. Scope injected last, so it overwrites any colliding model-supplied key.
    query.update(scope.as_filter())

    logger.info("job_status query=%s limit=%s", query, limit)

    # No database-side sort by time is possible: timestamps live inside the
    # nested `timespan` object as day-first strings, and `_id` begins with
    # studyID+hospital, so sorting on it orders by study and patient, not by
    # recency. The page is ordered in Python instead - see _newest_first.
    docs = list(get_job_db()[cfg.collection].find(query, PROJECTION).limit(limit))

    _assert_in_scope(docs, scope)
    docs = _newest_first(docs)
    logger.info("job_status returned %d docs", len(docs))
    return docs


def get_user_job_status(
    scope: TenantScope,
    *,
    patient_id: Optional[str] = None,
    execution_id: Optional[str] = None,
    study_id: Optional[str] = None,
    trial_id: Optional[int] = None,
    disorder: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    What the `get_job_status` tool actually returns to the LLM.

    Scoped query, then a derived summary per record instead of the raw nested
    document. The raw form is ~40 nested keys of state-machine detail; the
    summary is the answer to "what is happening and what is wrong with it".
    """
    from services.execution_summary import summarize

    docs = get_job_status(
        scope,
        patient_id=patient_id,
        execution_id=execution_id,
        study_id=study_id,
        trial_id=trial_id,
        disorder=disorder,
        limit=limit,
    )
    return [summarize(doc) for doc in docs]


def _newest_first(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Order the returned page newest-first by earliest recorded start time.

    This sorts only the rows already fetched, not the whole collection, so
    "the 5 most recent jobs" is not answerable until the pipeline writes a
    real BSON date. Good enough while a scope returns few rows; revisit if
    that stops being true.
    """
    from services.execution_summary import parse_ts

    def sort_key(doc):
        span = doc.get("timespan") or {}
        starts = []
        for value in span.values():
            if isinstance(value, dict):
                if "start" in value:
                    starts.append(parse_ts(value.get("start")))
                else:  # nested per-task spans
                    for sub in value.values():
                        if isinstance(sub, dict):
                            starts.append(parse_ts(sub.get("start")))
        real = [s for s in starts if s]
        return min(real) if real else datetime.min

    return sorted(docs, key=sort_key, reverse=True)


def _assert_in_scope(docs: List[Dict[str, Any]], scope: TenantScope) -> None:
    """
    Agent 3, hard half: re-check every returned document against the scope.

    The query already filtered. This catches the cases the query cannot: a bad
    index, a view misconfiguration, a future refactor that drops the injection.
    It is cheap and it is the last line of defense before tenant data reaches a
    model's context window.
    """
    tenant_field = get_config().job_db.tenant_field
    expected = scope.hospital

    for doc in docs:
        actual = doc.get(tenant_field)
        if actual != expected:
            logger.critical(
                "SCOPE VIOLATION: doc %s has %s=%r, session scope is %r",
                doc.get("_id"), tenant_field, actual, expected,
            )
            raise ScopeViolation(
                f"Document {doc.get('_id')} is outside the caller's scope"
            )


def health() -> Dict[str, Any]:
    """Job-database health for the /api/health endpoint."""
    from database.mongo import ping
    cfg = get_config().job_db
    try:
        reachable = ping()
        count = (
            get_job_db()[cfg.collection].estimated_document_count()
            if reachable else None
        )
        return {
            "status": "healthy" if reachable else "unhealthy",
            "collection": cfg.collection,
            "documents": count,
        }
    except (PyMongoError, RuntimeError) as e:
        return {"status": "unhealthy", "error": str(e)}
