"""
Instant-greeting matcher (offline, no AI, no network).
Extracts greetings, thanks, and goodbye messages to bypass LLM and guardrail.
"""
import unicodedata

GREETING_WORDS = {"hi", "hii", "hiii", "hello", "helo", "hey", "heya", "hola",
                  "morning", "afternoon", "evening", "vanakkam", "namaste", "namaskar"}
THANKS_WORDS = {"thanks", "thank", "thx", "ty", "tq", "nandri", "dhanyavad", "dhanyawad",
                "shukriya", "appreciated"}
BYE_WORDS = {"bye", "goodbye", "cya", "tata"}
FILLER_WORDS = {"good", "you", "so", "much", "a", "lot", "very", "there", "all", "team",
                "again", "ok", "okay", "got", "it", "great", "cool", "nice", "and", "for",
                "the", "help", "your", "sir", "madam", "bot", "everyone", "see", "later",
                "ji", "romba", "bahut"}


def normalize(text: str) -> list:
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    return text.split()


def match_greeting(text: str):
    """Return 'greeting' | 'thanks' | 'bye' if the whole message is one, else None."""
    words = normalize(text)
    if not words or len(words) > 6:
        return None
    core = set(words) & (GREETING_WORDS | THANKS_WORDS | BYE_WORDS)
    if not core:
        return None
    if any(w not in GREETING_WORDS | THANKS_WORDS | BYE_WORDS | FILLER_WORDS for w in words):
        return None
    if core & THANKS_WORDS:
        return "thanks"
    if core & BYE_WORDS:
        return "bye"
    return "greeting"
