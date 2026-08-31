"""Centralized generic helpers shared across services - no domain knowledge
of any particular feature lives here.
"""

from __future__ import annotations

import re
import unicodedata

# --- Input sanitization for text sent to an LLM -----------------------------
#
# Three layers, all meant to run BEFORE the text reaches the model so a
# malicious or out-of-scope input never costs a model call:
#
# 1. Character cleaning - strips hidden/control characters and invalid
#    Unicode, collapses redundant whitespace. Purely mechanical, never
#    rejects.
# 2. Prompt-injection defense - flags attempts to override instructions,
#    leak the system prompt, hijack the model's role, or dictate its output
#    directly.
# 3. Scope guardrails - flags content trying to redirect the model to an
#    unrelated task, or embed a payload meant for a different execution
#    context (HTML/script, SQL, shell).
#
# Callers that also instruct their model to ignore embedded instructions
# (e.g. via their own system prompt) get this as a second, independent layer
# that blocks the obvious cases outright instead of relying on the model
# alone.

# Zero-width / invisible characters sometimes used to break up a blocked
# phrase or hide a payload (e.g. "ig​nore previous instructions").
_ZERO_WIDTH_CHARS = [
    0x200B,  # zero width space
    0x200C,  # zero width non-joiner
    0x200D,  # zero width joiner
    0x2060,  # word joiner
    0xFEFF,  # BOM / zero width no-break space
    0x00AD,  # soft hyphen
]
_ZERO_WIDTH_RE = re.compile("[" + "".join(chr(cp) for cp in _ZERO_WIDTH_CHARS) + "]")

# C0/C1 control characters other than tab/LF/CR, which are kept and later
# collapsed into ordinary whitespace.
_CONTROL_CHARS_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(raw: str) -> str:
    """Normalise Unicode, strip invisible/control characters, collapse
    redundant whitespace. Purely mechanical - does not judge content."""
    text = unicodedata.normalize("NFKC", raw)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _CONTROL_CHARS_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _compile_all(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns]


# An injection attempt is an IMPERATIVE addressed to the model, so it opens a
# sentence (or the whole query), optionally behind a filler word. Ordinary
# legal narrative mentions the same verbs mid-sentence - "my friend asked me to
# act as a guarantor", "the company refused to show the rules I was fired
# under" - and anchoring here is what keeps those out of the blocked bucket.
_SENTENCE_START = r"(?:^|[.!?;\n]\s*)(?:please\s+|now\s+|also\s+|and\s+|then\s+)*"


# Phrasings that try to override system instructions, leak the system prompt,
# redefine the model's role, or dictate the output JSON directly.
_INJECTION_PATTERNS = _compile_all(
    [
        # Instruction override.
        _SENTENCE_START
        + r"(ignore|disregard|forget|override)\s+(all\s+|any\s+)?(the\s+|your\s+)?(previous|prior|above|earlier|system|original)\s+(instructions?|rules?|prompts?)",
        _SENTENCE_START
        + r"forget\s+(everything|all)\s+(you\s+)?(were\s+told|know|above)",
        # Prompt leaking. The possessive ("your", or "the system/original ...")
        # is required: without it this matches employment and contract disputes.
        r"(reveal|print|show|repeat|leak|output|give\s+me|tell\s+me)\s+(me\s+)?(your|the)\s+(system|initial|original)\s+(prompt|instructions?|rules?)",
        r"(reveal|print|repeat|leak|output)\s+your\s+(system\s+)?(prompt|instructions?)",
        r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions?|rules?)",
        # Role hijacking. Note "act as if you are" only - bare "act as a/an"
        # is ordinary legal vocabulary (guarantor, witness, nominee, agent).
        _SENTENCE_START + r"you\s+are\s+now\s+(a|an)\b",
        _SENTENCE_START + r"act\s+as\s+if\s+you\s+(are|were)\b",
        _SENTENCE_START + r"pretend\s+(to\s+be|you\s+are)\b",
        r"\bdeveloper\s+mode\b",
        r"\bjailbreak\b",
        r"\bDAN\s+mode\b",
        # Output hijacking.
        _SENTENCE_START + r"new\s+instructions?\s*:",
        r"^\s*(system|assistant|user)\s*:",  # line-anchored role marker
        r"<\|im_(start|end)\|>",
        r"\[/?(system|assistant|instructions?)\]",
        r"###\s*(system|instruction)",
        r"\bset\s+is_valid\s+to\b",
        _SENTENCE_START
        + r"(output|respond\s+with|reply\s+with|return)\s+(only\s+)?(the\s+following\s+)?json",
        _SENTENCE_START + r"respond\s+only\s+with\b",
    ]
)

# Content that redirects the model to an unrelated task, or embeds a payload
# meant for a different execution context (browser, database, shell).
#
# Task redirection is imperative too, so it gets the same anchor - "my employer
# asked me to write a program and now claims the IP" is a real IPR query, not
# an attempt to turn the classifier into a code generator. The payload patterns
# below stay unanchored: they are not phrased as instructions to anyone.
_SCOPE_PATTERNS = _compile_all(
    [
        r"<\s*script\b",
        r"javascript\s*:",
        r"\bon(error|click|load)\s*=",
        r"\bunion\s+select\b",
        r";\s*drop\s+table\b",
        r"'\s*or\s+'?1'?\s*=\s*'?1",
        r"\$\([^)]*\)",  # $(command substitution)
        r"`[^`]*`",  # backtick command substitution
        r";\s*rm\s+-rf\b",
        _SENTENCE_START
        + r"translate\s+(this|the\s+following)\s+(text|sentence|paragraph)?\s*(into|to)\s+\w+",
        _SENTENCE_START
        + r"(write|generate|compose)\s+(me\s+)?(a|an)?\s*(poem|song|essay|story|code|program|script)\b",
        _SENTENCE_START + r"generate\s+code\s+for\b",
        _SENTENCE_START
        + r"summarize\s+(this|the\s+following)\s+(article|text|document)\b",
    ]
)


def find_security_issue(text: str) -> str | None:
    """First matched category ("prompt_injection" / "scope_violation"), or
    None if the text looks safe. The label is for internal use (printing) -
    never surface it in an API response, so a caller can't learn which exact
    rule to dodge."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return "prompt_injection"
    for pattern in _SCOPE_PATTERNS:
        if pattern.search(text):
            return "scope_violation"
    return None
