"""Centralized generic helpers shared across services - no domain knowledge
of any particular feature lives here.
"""

from __future__ import annotations

import re
import unicodedata

# --- Input sanitization for text sent to an LLM -----------------------------
#
# Three steps, run in THIS ORDER, all before the text reaches the model so a
# malicious or out-of-scope input never costs a model call:
#
# 1. clean_text - strips hidden/control characters and invalid Unicode and
#    collapses redundant spacing, but KEEPS line breaks. Purely mechanical,
#    never rejects.
# 2. find_security_issue - flags prompt-injection attempts (overriding
#    instructions, leaking the system prompt, hijacking the model's role or
#    output) and scope violations (redirecting the model to an unrelated task,
#    or embedding a payload meant for another execution context).
# 3. flatten - collapses the surviving line breaks, producing the single-line
#    string actually sent to the model.
#

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
# collapsed into ordinary whitespace. These are DELETED rather than replaced
# with a space: they are used to splice a blocked word apart the same way the
# zero-width characters above are, so replacing them would leave "ig\x01nore"
# as "ig nore" and defeat the scanner, while deleting rejoins it to "ignore".
_CONTROL_CHARS_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

_CARRIAGE_RETURN_RE = re.compile(r"\r\n?")
# Horizontal whitespace only - "\s" minus the newline we are preserving.
_HORIZONTAL_SPACE_RE = re.compile(r"[^\S\n]+")
_LINE_BREAK_RUN_RE = re.compile(r" ?\n[\s\n]*")
_ALL_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(raw: str) -> str:
    """Normalise Unicode, strip invisible/control characters, collapse
    redundant spacing. Line breaks are preserved for `find_security_issue`;
    call `flatten` afterwards for the text to send to the model. Purely
    mechanical - does not judge content."""
    text = unicodedata.normalize("NFKC", raw)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _CARRIAGE_RETURN_RE.sub("\n", text)
    text = _HORIZONTAL_SPACE_RE.sub(" ", text)
    text = _LINE_BREAK_RUN_RE.sub("\n", text)
    return text.strip()


def flatten(text: str) -> str:
    """Collapse every remaining whitespace run, including line breaks, into
    single spaces - the single-line form sent to the model."""
    return _ALL_WHITESPACE_RE.sub(" ", text).strip()


def _compile_all(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns]


# An injection attempt is an IMPERATIVE addressed to the model, so it opens a
# sentence, a line, or the whole query, optionally behind a filler word.
# Ordinary legal narrative mentions the same verbs mid-sentence - "my friend
# asked me to act as a guarantor", "the company refused to show the rules I was
# fired under" - and anchoring here is what keeps those out of the blocked
# bucket.
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
# meant for a different execution context (browser, database).
#
# Task redirection is imperative too, so it gets the same anchor - "my employer
# asked me to write a program and now claims the IP" is a real IPR query, not
# an attempt to turn the classifier into a code generator. The payload patterns
# below stay unanchored: they are not phrased as instructions to anyone.
#
# There are deliberately no shell-payload patterns here. Backtick and $(...)
# spans only mean something to a shell, and this text is never handed to one -
# it goes to an LLM. Matching them blocked ordinary quoted text and figures
# like "$(50,000)" while preventing nothing.
_SCOPE_PATTERNS = _compile_all(
    [
        r"<\s*script\b",
        r"javascript\s*:",
        r"\bon(error|click|load)\s*=",
        r"\bunion\s+select\b",
        r";\s*drop\s+table\b",
        r"'\s*or\s+'?1'?\s*=\s*'?1",
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
    None if the text looks safe. Expects `clean_text` output, with its line
    breaks intact. The label is for internal use (printing) - never surface it
    in an API response, so a caller can't learn which exact rule to dodge."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return "prompt_injection"
    for pattern in _SCOPE_PATTERNS:
        if pattern.search(text):
            return "scope_violation"
    return None
