"""Canned replies for hi / thanks / bye / small-talk so we skip the LLM.

Design:
- Aggressively normalize the message (case, punctuation, repeated letters).
- Detect by *category token*: the message is a greeting if it's short and a
  greeting word appears in it.
- Always defer to RAG when the message looks like a real document question.
"""

from __future__ import annotations

import re

# Words that, if present, mean the message is a real document question and
# should bypass greeting handling entirely.
_TOPIC_MARKERS = (
    "document",
    "section",
    "chapter",
    "page",
    "clause",
    "paragraph",
    "article",
    "amendment",
    "law",
    "policy",
    "procedure",
    "specification",
    "spec",
    "requirement",
    "datasheet",
    "manual",
    "constitution",
    "congress",
    "senate",
    "president",
    "government",
)


# Greeting token vocabularies. Keep canonical (lowercase, no punctuation,
# no repeated trailing letters). The normalizer below collapses real-world
# variants into these forms.
_HELLO_TOKENS = frozenset(
    {
        "hi",
        "hii",  # post-collapse safety
        "hey",
        "hello",
        "helo",
        "hiya",
        "howdy",
        "yo",
        "sup",
        "wassup",
        "whatsup",
        "greetings",
        "namaste",
        "namaskar",
        "hola",
        "bonjour",
        "ciao",
        "salaam",
        "salam",
        "shalom",
        "konnichiwa",
        "annyeong",
        "aloha",
        "gday",
        "gm",
        "ga",
        "ge",
        "morning",
        "morn",
        "afternoon",
        "evening",
        "eve",
        "night",
        "nite",
    }
)

_THANKS_TOKENS = frozenset(
    {
        "thanks",
        "thank",
        "thankyou",
        "thx",
        "ty",
        "tysm",
        "tyvm",
        "thnx",
        "thnks",
        "tnx",
        "appreciate",
        "appreciated",
        "cheers",
        "gracias",
        "merci",
        "danke",
        "arigato",
        "shukriya",
        "dhanyavaad",
    }
)

_BYE_TOKENS = frozenset(
    {
        "bye",
        "byebye",
        "goodbye",
        "cya",
        "later",
        "ttyl",
        "adios",
        "ciao",  # also a hello in Italian, ambiguous; we treat as hello first
        "farewell",
        "peace",
    }
)

_HOW_ARE_YOU_TRIGGERS = (
    "how are you",
    "how r u",
    "how ru",
    "how is it going",
    "hows it going",
    "how are things",
    "you good",
    "u good",
    "all good",
    "nice to meet you",
    "pleased to meet you",
    "how have you been",
    "hows everything",
)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_REPEAT_LETTER = re.compile(r"([a-z])\1{2,}")  # 3+ same letter -> 1
_REPEAT_PAIR = re.compile(r"([a-z])\1")        # 2 same letter -> 1 (gentle)


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace and repeated letters.

    Examples
    --------
    "Hi!"        -> "hi"
    "Hellooo!"   -> "helo"  (then matched in HELLO_TOKENS via "helo")
    "HEYYY"      -> "hey"
    "  thx  "    -> "thx"
    "Good Morning, friend." -> "good morning friend"
    """
    s = text.strip().lower()
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = _REPEAT_LETTER.sub(r"\1", s)
    s = _REPEAT_PAIR.sub(r"\1", s)
    return s


def _looks_like_document_question(t: str) -> bool:
    """True → should use RAG, not greeting shortcuts."""
    if len(t) > 120:
        return True
    if any(m in t for m in _TOPIC_MARKERS):
        return True
    if "?" in t and len(t) > 25:
        return True
    factual_starts = (
        "what is ",
        "what are ",
        "what does ",
        "what was ",
        "why ",
        "how does ",
        "how do ",
        "when did ",
        "when does ",
        "where does ",
        "where is ",
        "who is ",
        "who are ",
        "which ",
        "explain ",
        "define ",
        "does the ",
        "did the ",
        "compare ",
        "list ",
        "summarize ",
        "describe ",
        "tell me about ",
    )
    return any(t.startswith(p) for p in factual_starts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def greeting_reply(text: str) -> str | None:
    """Return a short canned reply for small talk, or None for normal RAG routing."""
    raw = text.strip()
    if not raw:
        return None
    if len(raw) > 120:
        return None

    t = _normalize(raw)
    if not t or _looks_like_document_question(t):
        return None

    # Multi-word triggers (must be checked before token logic).
    if any(trigger in t for trigger in _HOW_ARE_YOU_TRIGGERS):
        return _msg_how_are_you()

    words = t.split()
    if not words:
        return None

    # If the message is short, classify by which greeting category its words
    # belong to. We require the first or last word to be a greeting token, so
    # "tell me about X" can't accidentally match because of a stray "thanks".
    is_short = len(words) <= 6

    first, last = words[0], words[-1]

    def has_token(tokens: frozenset[str]) -> bool:
        return first in tokens or last in tokens or any(w in tokens for w in words[:3])

    if is_short:
        # Order matters: thanks/bye take priority over hello if both
        # categories appear (e.g. "thanks bye").
        if has_token(_THANKS_TOKENS):
            return _msg_thanks()
        if first in _BYE_TOKENS or last in _BYE_TOKENS:
            return _msg_bye()
        if has_token(_HELLO_TOKENS):
            return _msg_welcome()

    # "good morning/afternoon/evening/night" forms (and minor variants).
    if any(t.startswith(p) for p in (
        "good morning",
        "good afternoon",
        "good evening",
        "good night",
        "good day",
    )):
        return _msg_welcome()

    # Bare exact-match fallback (catches anything we missed).
    if t in _HELLO_TOKENS:
        return _msg_welcome()
    if t in _THANKS_TOKENS:
        return _msg_thanks()
    if t in _BYE_TOKENS:
        return _msg_bye()

    return None


def _msg_welcome() -> str:
    return (
        "Hi — I'm your document assistant. Ask me anything answered by the "
        "uploaded PDF and I'll ground my answer in its content."
    )


def _msg_thanks() -> str:
    return "Glad to help! Ask another question about the document whenever you want."


def _msg_bye() -> str:
    return "Goodbye — reload the page anytime to keep chatting about your PDF."


def _msg_how_are_you() -> str:
    return (
        "I'm doing well, thanks — ready to answer questions about your uploaded "
        "document. What would you like to know?"
    )
