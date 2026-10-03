"""Evidence is what a tool returned, not what an agent says a tool returned.

Section 9 has the platform "normalize, classify and hash evidence", and section 4 rules
out accepting model-generated citations without verifying them. So a researcher does not
get to hand in a finished evidence record. It hands in a *claim*: an excerpt, and the
identifier of the tool call it says the excerpt came from. Everything else is decided
here, from what the platform itself observed:

- the tool call must be one this task actually made, recorded as it happened;
- the excerpt must really appear in what that call returned;
- the source is taken from the call's own arguments, not from the agent;
- the content hash is computed here, over the normalized excerpt;
- trust and access classification come from the capability that produced it.

A claim that fails any of these is not downgraded to a weaker kind of evidence. It is
refused, and the agent is told which claim and why - without the excerpt being echoed
back, since that text came from an untrusted source.
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from research_platform.agents.contracts import EvidenceSubmission
from research_platform.domain.invocations import ToolInvocation, argument_digest
from research_platform.domain.models import (
    AccessClass,
    EvidenceRecordCreate,
    NonEmptyText,
    TrustLevel,
)
from research_platform.mcp.registry import Capability

SUMMARISING_CAPABILITIES = frozenset({("web-research", "search")})
"""Capabilities whose output describes a source rather than being one.

A search result is a third party's snippet about a page, so an excerpt taken from it is
secondary evidence; an excerpt of a fetched page, a file or a query result is the source
itself.
"""


def normalize_text(text: str) -> str:
    """Canonical form used both to match an excerpt and to hash it.

    Unicode is composed and every run of whitespace becomes one space, so an excerpt an
    agent re-wrapped or re-indented still matches, and the same words always hash alike.
    """
    return " ".join(unicodedata.normalize("NFC", text).split())


def hash_content(text: str) -> str:
    """The content hash of an excerpt, over its normalized form."""
    return f"sha256:{hashlib.sha256(normalize_text(text).encode('utf-8')).hexdigest()}"


def source_of(capability: Capability, tenant_id: str, arguments: dict[str, Any]) -> str:
    """Where an invocation's content came from, derived from the call that fetched it."""
    key = (capability.server, capability.name)
    if key == ("web-research", "fetch"):
        return str(arguments.get("url", "")).strip()
    if key == ("filesystem", "read_document"):
        return f"workspace://{tenant_id}/{str(arguments.get('path', '')).lstrip('/')}"
    if capability.server == "github":
        return f"https://github.com/{str(arguments.get('repository', '')).strip()}"
    # Anything else - a search, a schema, a query, a calculation - has no address of its
    # own, so it is identified by the exact call that produced it.
    digest = argument_digest(arguments).removeprefix("sha256:")[:16]
    return f"{capability.server}://{capability.name}/{digest}"


@dataclass(frozen=True)
class ObservedContent:
    """What one successful tool call returned, as the platform itself saw it."""

    invocation: ToolInvocation
    capability: Capability
    arguments: dict[str, Any]
    text: str

    @property
    def source_uri(self) -> str:
        return source_of(self.capability, self.invocation.tenant_id, self.arguments)

    @property
    def trust_level(self) -> TrustLevel:
        if (self.capability.server, self.capability.name) in SUMMARISING_CAPABILITIES:
            return TrustLevel.SECONDARY
        return TrustLevel.PRIMARY

    @property
    def access_class(self) -> AccessClass:
        return self.capability.max_access_class


@dataclass
class EvidenceLedger:
    """Every successful tool call one task made, kept for the length of that task."""

    _observed: dict[UUID, ObservedContent] = field(default_factory=dict)

    def record(
        self,
        invocation: ToolInvocation,
        capability: Capability,
        arguments: dict[str, Any],
        text: str,
    ) -> None:
        self._observed[invocation.id] = ObservedContent(
            invocation=invocation, capability=capability, arguments=dict(arguments), text=text
        )

    def get(self, invocation_id: UUID) -> ObservedContent | None:
        return self._observed.get(invocation_id)

    def __len__(self) -> int:
        return len(self._observed)


class EvidenceClaim(BaseModel):
    """What a researcher asserts: this excerpt came from that tool call."""

    model_config = {"frozen": True}

    excerpt: NonEmptyText
    tool_invocation_id: UUID
    title: str | None = Field(default=None, max_length=500)
    author: str | None = Field(default=None, max_length=200)
    published_at: datetime | None = None


class EvidenceClaims(BaseModel):
    """A researcher's output for one task, before the platform has verified any of it."""

    model_config = {"frozen": True}

    claims: list[EvidenceClaim] = Field(min_length=1, max_length=100)
    unmet_requirements: list[NonEmptyText] = Field(default_factory=list, max_length=20)


class UnverifiableEvidence(ValueError):
    """One or more claimed excerpts could not be traced to a tool call that returned them."""


def verify_claims(
    claims: EvidenceClaims, *, ledger: EvidenceLedger, task_id: UUID
) -> EvidenceSubmission:
    """Turn verified claims into evidence records, or refuse the whole submission."""
    problems: list[str] = []
    records: list[EvidenceRecordCreate] = []
    seen: set[str] = set()
    for index, claim in enumerate(claims.claims):
        observed = ledger.get(claim.tool_invocation_id)
        if observed is None:
            problems.append(
                f"claims.{index}: {claim.tool_invocation_id} is not a tool call this task "
                "made; cite the tool_invocation_id printed at the top of a tool result"
            )
            continue
        if normalize_text(claim.excerpt) not in normalize_text(observed.text):
            problems.append(
                f"claims.{index}: the excerpt does not appear in the output of tool call "
                f"{claim.tool_invocation_id}; quote the result exactly"
            )
            continue
        content_hash = hash_content(claim.excerpt)
        if content_hash in seen:
            problems.append(f"claims.{index}: the same excerpt was submitted more than once")
            continue
        seen.add(content_hash)
        records.append(
            EvidenceRecordCreate(
                excerpt=normalize_text(claim.excerpt),
                source_uri=observed.source_uri,
                title=claim.title,
                author=claim.author,
                published_at=claim.published_at,
                trust_level=observed.trust_level,
                access_class=observed.access_class,
                content_hash=content_hash,
                producing_task_id=task_id,
                tool_invocation_id=claim.tool_invocation_id,
            )
        )
    if problems:
        raise UnverifiableEvidence("; ".join(problems))
    return EvidenceSubmission(records=records, unmet_requirements=list(claims.unmet_requirements))
