"""
Custom codec for serializing/deserializing Message and ToolCall objects.
Handles raw bytes in ToolCall provider_data (e.g. Gemini thought_signature)
without data loss.
"""
import base64
import json
from dataclasses import fields
from typing import Any, List
from llm.base import Message, ToolCall


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return {"__bytes__": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _from_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        if set(obj) == {"__bytes__"}:
            return base64.b64decode(obj["__bytes__"])
        return {k: _from_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_jsonable(v) for v in obj]
    return obj


def message_to_dict(m: Message) -> dict:
    d = {f.name: getattr(m, f.name) for f in fields(m)}
    if m.tool_calls:
        d["tool_calls"] = [{f.name: getattr(tc, f.name) for f in fields(tc)} for tc in m.tool_calls]
    return _to_jsonable(d)


def message_from_dict(d: dict) -> Message:
    d = _from_jsonable(d)
    if d.get("tool_calls"):
        d["tool_calls"] = [ToolCall(**tc) for tc in d["tool_calls"]]
    return Message(**d)


def dumps(messages: List[Message]) -> str:
    return json.dumps([message_to_dict(m) for m in messages], ensure_ascii=False)


def loads(raw: str) -> List[Message]:
    return [message_from_dict(d) for d in json.loads(raw)]


def trim(messages: List[Message], k: int) -> List[Message]:
    """
    Trim conversation history to the last k user turns.
    Preserves all tool turns attached to each retained user turn.
    """
    starts = [i for i, m in enumerate(messages) if m.role == "user"]
    if len(starts) > k:
        return messages[starts[-k]:]
    return messages

