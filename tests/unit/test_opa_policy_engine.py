from collections.abc import Callable
from uuid import uuid4

import httpx
import pytest

from research_platform.domain.models import AccessClass
from research_platform.domain.tasks import AgentRole
from research_platform.identity import Principal, Role
from research_platform.mcp.opa import (
    UNAVAILABLE_VERSION,
    AllOfPolicyEngine,
    OpaPolicyEngine,
    build_policy_input,
)
from research_platform.mcp.policy import (
    AuthorizationRequest,
    PolicyDecision,
    RegistryPolicyEngine,
)
from research_platform.mcp.registry import Capability

JOB_ID = uuid4()
TASK_ID = uuid4()

CAPABILITY = Capability(
    server="web-research",
    name="fetch",
    description="Fetch an approved public URL.",
    required_roles=frozenset({Role.REQUESTER}),
    allowed_agents=frozenset({AgentRole.RESEARCHER}),
)

PRINCIPAL = Principal(
    tenant_id="acme",
    subject_id="user-1",
    roles=frozenset({Role.REQUESTER}),
    clearance=AccessClass.INTERNAL,
)


def make_request(principal: Principal = PRINCIPAL) -> AuthorizationRequest:
    return AuthorizationRequest(
        principal=principal,
        capability=CAPABILITY,
        job_id=JOB_ID,
        task_id=TASK_ID,
        arguments={"url": "https://vendor.test/pricing", "api_key": "live-secret"},
    )


def engine_returning(handler: Callable[[httpx.Request], httpx.Response]) -> OpaPolicyEngine:
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://opa.test")
    return OpaPolicyEngine("http://opa.test", client=client)


def responding(payload: object, status_code: int = 200) -> OpaPolicyEngine:
    return engine_returning(lambda _: httpx.Response(status_code, json=payload))


def test_a_base_url_is_required() -> None:
    with pytest.raises(ValueError, match="base URL is required"):
        OpaPolicyEngine("")


def test_the_policy_input_describes_the_identity_and_capability() -> None:
    document = build_policy_input(make_request())["input"]

    assert document["principal"] == {
        "tenant_id": "acme",
        "subject_id": "user-1",
        "roles": ["requester"],
        "agent_role": None,
        "clearance": "internal",
    }
    assert document["capability"]["server"] == "web-research"
    assert document["capability"]["allowed_agents"] == ["researcher"]
    assert document["job_id"] == str(JOB_ID)


def test_the_policy_input_names_arguments_without_their_values() -> None:
    """Argument values would copy sensitive data into a second service and its logs."""
    document = build_policy_input(make_request())["input"]

    assert document["argument_names"] == ["api_key", "url"]
    assert "live-secret" not in str(document)
    assert "vendor.test" not in str(document)


def test_the_acting_agent_role_is_sent() -> None:
    document = build_policy_input(make_request(PRINCIPAL.for_agent(AgentRole.CRITIC)))["input"]

    assert document["principal"]["agent_role"] == "critic"


def test_an_allow_decision_is_honoured() -> None:
    engine = responding({"result": {"allow": True, "policy_version": "opa-2026-08-01"}})

    decision = engine.evaluate(make_request())

    assert decision.allowed is True
    assert decision.policy_version == "opa-2026-08-01"
    assert engine.policy_version == "opa-2026-08-01"


def test_a_deny_decision_carries_the_policy_reason() -> None:
    engine = responding(
        {"result": {"allow": False, "reason": "no held role permits", "policy_version": "v1"}}
    )

    decision = engine.evaluate(make_request())

    assert decision.allowed is False
    assert decision.reason == "no held role permits"
    assert decision.policy_version == "v1"


def test_a_deny_without_a_reason_still_refuses() -> None:
    engine = responding({"result": {"allow": False, "policy_version": "v1"}})

    decision = engine.evaluate(make_request())

    assert decision.allowed is False
    assert "no reason was given" in decision.reason


def test_the_request_is_posted_to_the_decision_path() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"result": {"allow": True, "policy_version": "v1"}})

    engine_returning(handler).evaluate(make_request())

    assert seen[0].method == "POST"
    assert seen[0].url.path == "/v1/data/research/authz/decision"


def test_a_timeout_refuses_the_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    decision = engine_returning(handler).evaluate(make_request())

    assert decision.allowed is False
    assert "did not respond in time" in decision.reason
    assert decision.policy_version == UNAVAILABLE_VERSION


def test_a_transport_error_refuses_the_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    decision = engine_returning(handler).evaluate(make_request())

    assert decision.allowed is False
    assert "could not be reached" in decision.reason


@pytest.mark.parametrize("status_code", [400, 404, 500, 503])
def test_an_error_status_refuses_the_call(status_code: int) -> None:
    engine = responding({"result": {"allow": True}}, status_code=status_code)

    decision = engine.evaluate(make_request())

    assert decision.allowed is False
    assert str(status_code) in decision.reason


def test_a_non_json_response_refuses_the_call() -> None:
    engine = engine_returning(lambda _: httpx.Response(200, text="not json"))

    decision = engine.evaluate(make_request())

    assert decision.allowed is False
    assert "could not be reached" in decision.reason


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"result": {}},
        {"result": {"allow": "yes"}},
        {"result": "allow"},
        [],
    ],
)
def test_an_unreadable_decision_refuses_the_call(payload: object) -> None:
    decision = responding(payload).evaluate(make_request())

    assert decision.allowed is False
    assert decision.policy_version == UNAVAILABLE_VERSION


def test_the_reported_version_resets_when_the_service_becomes_unavailable() -> None:
    responses = iter(
        [
            httpx.Response(200, json={"result": {"allow": True, "policy_version": "v9"}}),
            httpx.Response(503, json={}),
        ]
    )
    engine = engine_returning(lambda _: next(responses))

    engine.evaluate(make_request())
    assert engine.policy_version == "v9"

    engine.evaluate(make_request())
    assert engine.policy_version == UNAVAILABLE_VERSION


def test_the_engine_closes_a_client_it_owns() -> None:
    with OpaPolicyEngine("http://opa.test") as engine:
        assert engine.policy_version == UNAVAILABLE_VERSION


class StubEngine:
    def __init__(self, decision: PolicyDecision) -> None:
        self._decision = decision
        self.calls = 0

    @property
    def policy_version(self) -> str:
        return self._decision.policy_version

    def evaluate(self, request: AuthorizationRequest) -> PolicyDecision:
        self.calls += 1
        return self._decision


def test_at_least_one_engine_is_required() -> None:
    with pytest.raises(ValueError, match="at least one policy engine"):
        AllOfPolicyEngine([])


def test_all_engines_must_permit_the_call() -> None:
    engine = AllOfPolicyEngine([RegistryPolicyEngine(), StubEngine(PolicyDecision.allow("opa-v1"))])

    assert engine.evaluate(make_request()).allowed is True


def test_a_single_refusal_denies_the_call() -> None:
    engine = AllOfPolicyEngine(
        [RegistryPolicyEngine(), StubEngine(PolicyDecision.deny("blocked by policy", "opa-v1"))]
    )

    decision = engine.evaluate(make_request())

    assert decision.allowed is False
    assert "blocked by policy" in decision.reason


def test_the_registry_boundary_still_applies_when_the_bundle_permits() -> None:
    """A misconfigured bundle cannot widen access beyond the registered capability."""
    engine = AllOfPolicyEngine([RegistryPolicyEngine(), StubEngine(PolicyDecision.allow("opa-v1"))])
    reporter = PRINCIPAL.for_agent(AgentRole.REPORTER)

    decision = engine.evaluate(make_request(reporter))

    assert decision.allowed is False
    assert "may not execute" in decision.reason


def test_every_refusal_is_reported() -> None:
    engine = AllOfPolicyEngine(
        [
            StubEngine(PolicyDecision.deny("first refusal", "a")),
            StubEngine(PolicyDecision.deny("second refusal", "b")),
        ]
    )

    decision = engine.evaluate(make_request())

    assert "first refusal" in decision.reason
    assert "second refusal" in decision.reason


def test_the_combined_version_names_every_engine() -> None:
    engine = AllOfPolicyEngine(
        [StubEngine(PolicyDecision.allow("local-v1")), StubEngine(PolicyDecision.allow("opa-v2"))]
    )

    assert engine.policy_version == "local-v1+opa-v2"


def test_every_engine_is_consulted_even_after_a_refusal() -> None:
    """Reporting all refusals at once avoids a caller fixing one and hitting the next."""
    first = StubEngine(PolicyDecision.deny("first refusal", "a"))
    second = StubEngine(PolicyDecision.deny("second refusal", "b"))

    AllOfPolicyEngine([first, second]).evaluate(make_request())

    assert first.calls == 1
    assert second.calls == 1
