"""
Citation verification service for BSAI chatbot.

Verifies that every technical citation references an approved document
actually retrieved in the current turn, removing ungrounded or hallucinated
citations before they reach the user.
"""
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Matches [Source: <title>] and [Source: <title>, p. <n>] (also "p <n>", "p.<n>", case-insensitive)
CITATION_RE = re.compile(
    r"\[source:\s*([^,\]]+?)(?:,\s*p\.?\s*(\d+))?\s*\]",
    re.IGNORECASE
)


@dataclass
class CitationCheck:
    text: str
    valid: int
    invalid: int
    missing_page: int


def normalize_title(s: str) -> str:
    """Normalize a document title for robust matching.

    - Unicode NFKC, then lowercase.
    - Replace '_' and '-' with spaces.
    - Remove trailing file extension (.pdf, .docx, .md, .txt).
    - Collapse all whitespace runs to a single space, and strip.
    """
    if not s:
        return ""
    text = unicodedata.normalize("NFKC", str(s)).lower()
    text = re.sub(r"\.(pdf|docx|md|txt)$", "", text, flags=re.IGNORECASE)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _titles_match(cited_norm: str, retrieved_norm: str) -> bool:
    """Check if cited title matches retrieved title.

    Matching rule:
    The titles are equal after normalization, OR one fully contains
    the other AND the shorter one has at least 3 words.
    Never match on fewer than 3 words by containment.
    """
    if not cited_norm or not retrieved_norm:
        return False
    if cited_norm == retrieved_norm:
        return True
    if cited_norm in retrieved_norm or retrieved_norm in cited_norm:
        shorter = cited_norm if len(cited_norm) <= len(retrieved_norm) else retrieved_norm
        if len(shorter.split()) >= 3:
            return True
    return False


def _clean_whitespace(text: str) -> str:
    """Clean up leftover double spaces and spaces before punctuation."""
    text = re.sub(r" +", " ", text)
    text = re.sub(r" +([.,;?!])", r"\1", text)
    text = re.sub(r" +$", "", text, flags=re.MULTILINE)
    return text


def verify_citations(
    text: Any, sources: Any, message_id: Optional[str] = None
) -> CitationCheck:
    """
    Verify citations in text against retrieved sources.

    - Title matching uses normalize_title with exact or >=3-word containment.
    - If title is valid and page matches (or no page requested): kept as is.
    - If title is valid but page was not retrieved: keep citation, remove only the page part.
      Counts as missing_page (not invalid).
    - If title matches nothing retrieved: citation is removed. Counts as invalid.
    - Never raises; on any error returns the text unchanged.
    """
    if not isinstance(text, str):
        return CitationCheck(text=text, valid=0, invalid=0, missing_page=0)

    msg_id = message_id or "unknown"

    try:
        # Build list of retrieved documents: [(original_title, normalized_title, {pages})]
        retrieved_data: List[Tuple[str, str, Set[Optional[int]]]] = []
        retrieved_titles_log: List[str] = []

        if sources:
            for s in sources:
                if isinstance(s, dict):
                    t = s.get("title", "")
                    p = s.get("page")
                else:
                    t = getattr(s, "title", "")
                    p = getattr(s, "page", None)

                if t:
                    orig_t = str(t).strip()
                    retrieved_titles_log.append(orig_t)
                    norm_t = normalize_title(orig_t)

                    page_val = None
                    if p is not None:
                        try:
                            page_val = int(p)
                        except (ValueError, TypeError):
                            page_val = None

                    # Find existing entry or add new
                    existing = None
                    for entry in retrieved_data:
                        if entry[1] == norm_t:
                            existing = entry
                            break

                    if existing:
                        existing[2].add(page_val)
                    else:
                        retrieved_data.append((orig_t, norm_t, {page_val}))

        valid_count = 0
        invalid_count = 0
        missing_page_count = 0

        # Find all citation matches
        matches = list(CITATION_RE.finditer(text))
        if not matches:
            logger.info(
                "citations message_id=%s valid=0 invalid=0 missing_page=0 retrieved_titles=%s",
                msg_id,
                retrieved_titles_log,
            )
            return CitationCheck(text=text, valid=0, invalid=0, missing_page=0)

        cleaned = text
        # Process matches in reverse order so character offsets do not shift
        for m in reversed(matches):
            cited_title_raw = m.group(1).strip() if m.group(1) else ""
            raw_page = m.group(2)
            cited_page = int(raw_page) if raw_page else None
            cited_norm = normalize_title(cited_title_raw)

            # Match against retrieved sources
            matched_source = None
            for orig_t, norm_t, pages in retrieved_data:
                if _titles_match(cited_norm, norm_t):
                    matched_source = (orig_t, norm_t, pages)
                    break

            start, end = m.span()

            if matched_source is None:
                # Title matches nothing retrieved: remove whole citation
                invalid_count += 1
                logger.warning(
                    'citation removed message_id=%s cited_title="%s" reason=no_match',
                    msg_id,
                    cited_title_raw,
                )
                cleaned = cleaned[:start] + cleaned[end:]
            else:
                # Title matches!
                orig_t, norm_t, pages = matched_source
                if cited_page is not None and cited_page not in pages:
                    # Valid title, wrong page: keep citation, remove only page part
                    valid_count += 1
                    missing_page_count += 1
                    replacement = f"[Source: {cited_title_raw}]"
                    cleaned = cleaned[:start] + replacement + cleaned[end:]
                else:
                    valid_count += 1
                    if cited_page is None:
                        missing_page_count += 1
                    # Keep citation as is

        cleaned = _clean_whitespace(cleaned)

        logger.info(
            "citations message_id=%s valid=%d invalid=%d missing_page=%d retrieved_titles=%s",
            msg_id,
            valid_count,
            invalid_count,
            missing_page_count,
            retrieved_titles_log,
        )

        return CitationCheck(
            text=cleaned,
            valid=valid_count,
            invalid=invalid_count,
            missing_page=missing_page_count,
        )

    except Exception as e:
        logger.error("verify_citations error: %s", e, exc_info=True)
        return CitationCheck(text=text, valid=0, invalid=0, missing_page=0)
