"""Treat every tool result as untrusted evidence rather than instruction."""

from __future__ import annotations

import re
import unicodedata
from enum import StrEnum

from pydantic import BaseModel, Field

CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
COLLAPSIBLE_WHITESPACE = re.compile(r"[ \t]{3,}")


class InjectionFlag(StrEnum):
    """Patterns that suggest retrieved content is trying to steer the agent."""

    INSTRUCTION_OVERRIDE = "instruction_override"
    ROLE_IMPERSONATION = "role_impersonation"
    CHAT_TEMPLATE_MARKER = "chat_template_marker"
    TOOL_DIRECTIVE = "tool_directive"
    CREDENTIAL_REQUEST = "credential_request"
    HIDDEN_CHARACTERS = "hidden_characters"


INJECTION_PATTERNS: tuple[tuple[InjectionFlag, re.Pattern[str]], ...] = (
    (
        InjectionFlag.INSTRUCTION_OVERRIDE,
        re.compile(
            r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}"
            r"\b(previous|prior|earlier|above|all)\b[^.\n]{0,40}"
            r"\b(instruction|instructions|prompt|prompts|rule|rules)\b",
            re.IGNORECASE,
        ),
    ),
    (
        InjectionFlag.ROLE_IMPERSONATION,
        re.compile(r"^\s*(system|assistant|developer)\s*:", re.IGNORECASE | re.MULTILINE),
    ),
    (
        InjectionFlag.CHAT_TEMPLATE_MARKER,
        re.compile(r"<\|[a-z_]+\|>|\[/?INST\]|<<SYS>>", re.IGNORECASE),
    ),
    (
        InjectionFlag.TOOL_DIRECTIVE,
        re.compile(
            r"\b(call|invoke|execute|run)\b[^.\n]{0,30}\b(tool|function|capability|command)\b",
            re.IGNORECASE,
        ),
    ),
    (
        InjectionFlag.CREDENTIAL_REQUEST,
        re.compile(
            r"\b(reveal|print|output|send|exfiltrate|leak)\b[^.\n]{0,40}"
            r"\b(api[ _-]?key|secret|token|password|credential|credentials)\b",
            re.IGNORECASE,
        ),
    ),
)


class UntrustedContent(BaseModel):
    """Source text kept deliberately separate from any tool-control message."""

    model_config = {"frozen": True}

    text: str
    byte_length: int = Field(ge=0)
    truncated: bool = False
    injection_flags: frozenset[InjectionFlag] = Field(default_factory=frozenset)

    @property
    def is_suspicious(self) -> bool:
        return bool(self.injection_flags)


def _strip_hidden_characters(text: str) -> tuple[str, bool]:
    """Remove zero-width and control characters used to hide instructions from a reviewer."""
    without_controls = CONTROL_CHARACTERS.sub("", text)
    visible = "".join(
        character
        for character in without_controls
        if unicodedata.category(character) != "Cf" or character == "\n"
    )
    return visible, visible != text


def detect_injection(text: str) -> frozenset[InjectionFlag]:
    return frozenset(flag for flag, pattern in INJECTION_PATTERNS if pattern.search(text))


def sanitize_result(text: str, *, max_bytes: int) -> UntrustedContent:
    """Normalize a tool result and flag steering attempts without silently discarding it.

    The text is never rewritten into something that looks safe. It is cleaned of hidden
    characters, bounded in size, and returned with explicit flags so the caller can decide
    whether to quarantine it.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    visible, hid_characters = _strip_hidden_characters(text)
    normalized = COLLAPSIBLE_WHITESPACE.sub("  ", visible)

    encoded = normalized.encode("utf-8")
    truncated = len(encoded) > max_bytes
    if truncated:
        normalized = encoded[:max_bytes].decode("utf-8", errors="ignore")
        encoded = normalized.encode("utf-8")

    flags = set(detect_injection(normalized))
    if hid_characters:
        flags.add(InjectionFlag.HIDDEN_CHARACTERS)

    return UntrustedContent(
        text=normalized,
        byte_length=len(encoded),
        truncated=truncated,
        injection_flags=frozenset(flags),
    )
