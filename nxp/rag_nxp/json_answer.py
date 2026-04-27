"""Parse JSON-shaped model output: {"answer": "...", "cited": [...]}.

No ML imports — pure string handling so it is cheap to unit-test.
"""

from __future__ import annotations

import json
import re
from json import JSONDecoder


def _parse_cited(obj: dict) -> list[int] | None:
    cited_out: list[int] = []
    cited = obj.get("cited")
    if isinstance(cited, list):
        for x in cited:
            if isinstance(x, int):
                cited_out.append(x)
            elif isinstance(x, str) and x.strip().isdigit():
                cited_out.append(int(x.strip()))
    return cited_out or None


def _scan_json_objects(s: str) -> list[dict]:
    """Find every JSON object and keep those with a non-empty "answer" string."""
    decoder = JSONDecoder()
    out: list[dict] = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == "{":
            try:
                obj, end = decoder.raw_decode(s, i)
                if (
                    isinstance(obj, dict)
                    and isinstance(obj.get("answer"), str)
                    and obj["answer"].strip()
                ):
                    out.append(obj)
                i = end
                continue
            except ValueError:
                pass
        i += 1
    return out


def _unescape_answer_field(inner: str) -> str:
    """Best-effort decode of a string value captured from `"answer": "..."`."""
    inner = inner.strip()
    try:
        return json.loads(f'"{inner}"')
    except json.JSONDecodeError:
        return (
            inner.replace("\\n", "\n")
            .replace('\\"', '"')
            .replace("\\\\", "\\")
        )


def decode_model_json_reply(raw: str) -> tuple[str | None, list[int] | None]:
    """Return (answer, cited) parsed from model text, or (None, None).

    Handles:
    - whole-string JSON
    - JSON wrapped in ```json ... ``` fences
    - prose before/after JSON (picks the last valid object with an "answer")
    - single greedy `{...}` block
    - truncated JSON where only a partial "answer": "..." pair survives
    """
    s = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s)
    if fence:
        s = fence.group(1).strip()

    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and isinstance(obj.get("answer"), str) and obj["answer"].strip():
            return obj["answer"].strip(), _parse_cited(obj)
    except json.JSONDecodeError:
        pass

    objs = _scan_json_objects(s)
    if objs:
        obj = objs[-1]
        return obj["answer"].strip(), _parse_cited(obj)

    try:
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            obj = json.loads(m.group(0))
            if (
                isinstance(obj, dict)
                and isinstance(obj.get("answer"), str)
                and obj["answer"].strip()
            ):
                return obj["answer"].strip(), _parse_cited(obj)
    except json.JSONDecodeError:
        pass

    matches = list(re.finditer(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', s, re.DOTALL))
    if matches:
        inner = matches[-1].group(1)
        return _unescape_answer_field(inner), None

    # Smaller models sometimes emit a botched JSON wrapper like
    #   {"Congress is responsible for making laws."}
    # i.e. braces around a string with no `"answer":` key. Recognise that
    # shape and recover the inner string as the answer.
    braced = re.match(r'^\s*\{\s*"((?:[^"\\]|\\.)+)"\s*\}\s*$', s, re.DOTALL)
    if braced:
        return _unescape_answer_field(braced.group(1)), None

    return None, None
