"""Fast path processing for instant responses without LLM invocation."""
from .greetings import (
    normalize,
    match_greeting,
    GREETING_WORDS,
    THANKS_WORDS,
    BYE_WORDS,
    FILLER_WORDS,
)

__all__ = [
    "normalize",
    "match_greeting",
    "GREETING_WORDS",
    "THANKS_WORDS",
    "BYE_WORDS",
    "FILLER_WORDS",
]
