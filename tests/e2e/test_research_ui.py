import hashlib
import os
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator

import httpx
import pytest
from playwright.sync_api import Page, expect


def wait_until_ready(host: str, port: int, timeout_seconds: float = 10) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        with socket.socket() as connection:
            connection.settimeout(0.2)
            if connection.connect_ex((host, port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"application did not start on {host}:{port}")


@pytest.fixture(scope="session")
def base_url() -> Iterator[str]:
    configured_url = os.getenv("E2E_BASE_URL")
    if configured_url:
        yield configured_url
        return

    host, port = "127.0.0.1", 8765
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "research_platform.main:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_until_ready(host, port)
        yield f"http://{host}:{port}"
    finally:
        process.terminate()
        process.wait(timeout=5)


def new_tenant() -> str:
    """A tenant of the test's own, so tests sharing one server never share jobs."""
    return f"ui-{uuid.uuid4().hex[:8]}"


def api(base_url: str, tenant: str, **extra: str) -> httpx.Client:
    """Talk to the same server the browser does, as one identity in one tenant."""
    return httpx.Client(
        base_url=f"{base_url}/api/v1",
        headers={"X-Tenant-ID": tenant, "X-Requester-ID": "local-user", **extra},
    )


def as_tenant(page: Page, base_url: str, tenant: str) -> None:
    page.goto(base_url)
    page.get_by_text("Signed in as").click()
    page.get_by_label("Tenant").fill(tenant)
    page.get_by_label("Tenant").press("Tab")


def evidence(excerpt: str, access_class: str = "public") -> dict[str, str]:
    digest = hashlib.sha256(" ".join(excerpt.split()).encode()).hexdigest()
    return {
        "excerpt": excerpt,
        "source_uri": "https://vendor.test/pricing",
        "access_class": access_class,
        "content_hash": f"sha256:{digest}",
        "producing_task_id": str(uuid.uuid4()),
        "tool_invocation_id": str(uuid.uuid4()),
    }


def awaiting_review(client: httpx.Client) -> None:
    """Create a job and move it by hand to the reviewer checkpoint."""
    job = client.post("/jobs", json={"question": "Is the contract price current?"}).json()
    for target in ("planning", "researching", "analyzing", "review_required"):
        client.post(f"/jobs/{job['id']}/transitions", params={"target": target})


@pytest.mark.e2e
def test_requester_can_create_and_see_assignment(page: Page, base_url: str) -> None:
    as_tenant(page, base_url, new_tenant())
    page.get_by_label("Research question").fill("Compare primary-source release claims")
    page.get_by_role("button", name="Create assignment").click()

    expect(page.get_by_role("status")).to_have_text("Assignment created.")
    card = page.locator("article")
    expect(card.get_by_role("heading")).to_have_text("Compare primary-source release claims")
    expect(card).to_contain_text("created")


@pytest.mark.e2e
def test_an_assignment_carries_its_constraints_and_budget(page: Page, base_url: str) -> None:
    tenant = new_tenant()
    as_tenant(page, base_url, tenant)
    page.get_by_label("Research question").fill("What does the vendor charge?")
    page.get_by_label("Constraints").fill("Published after 2024\nUSD only")
    page.get_by_label("Source requirements").fill("At least one primary source")
    page.get_by_label("Tool-call budget").fill("7")
    page.get_by_role("button", name="Create assignment").click()
    expect(page.get_by_role("status")).to_have_text("Assignment created.")

    [job] = api(base_url, tenant).get("/jobs").json()

    assert job["constraints"] == ["Published after 2024", "USD only"]
    assert job["source_requirements"] == ["At least one primary source"]
    assert job["budget"]["max_tool_calls"] == 7
    assert job["tenant_id"] == tenant


@pytest.mark.e2e
def test_opening_an_assignment_shows_its_evidence_usage_and_empty_sections(
    page: Page, base_url: str
) -> None:
    tenant = new_tenant()
    client = api(base_url, tenant)
    job = client.post("/jobs", json={"question": "What does the vendor charge?"}).json()
    client.post(f"/jobs/{job['id']}/evidence", json=evidence("Vendor pricing is 20 USD per seat."))

    as_tenant(page, base_url, tenant)
    page.get_by_role("button", name="Open").click()

    detail = page.get_by_label("Assignment detail")
    expect(detail.get_by_role("heading", level=2)).to_have_text("What does the vendor charge?")
    expect(detail).to_contain_text("Vendor pricing is 20 USD per seat.")
    expect(detail).to_contain_text("https://vendor.test/pricing · unverified · public")
    expect(detail).to_contain_text("0 tool calls · 0 tokens")
    expect(detail).to_contain_text("No findings yet.")
    expect(detail).to_contain_text("No tool calls yet.")
    expect(detail).to_contain_text("this research job has not published a report")
    expect(detail.get_by_role("button", name="Approve")).to_be_hidden()


@pytest.mark.e2e
def test_evidence_above_the_viewers_clearance_is_counted_but_not_shown(
    page: Page, base_url: str
) -> None:
    tenant = new_tenant()
    cleared = api(base_url, tenant, **{"X-Clearance": "internal"})
    job = cleared.post("/jobs", json={"question": "What is the negotiated price?"}).json()
    cleared.post(
        f"/jobs/{job['id']}/evidence",
        json=evidence("The negotiated price is 14 USD per seat.", "internal"),
    )

    as_tenant(page, base_url, tenant)
    page.get_by_role("button", name="Open").click()
    detail = page.get_by_label("Assignment detail")

    expect(detail).to_contain_text("1 withheld: above your clearance.")
    expect(detail).not_to_contain_text("14 USD")

    page.get_by_label("Clearance").select_option("internal")

    expect(detail).to_contain_text("The negotiated price is 14 USD per seat.")
    expect(detail).not_to_contain_text("withheld")


@pytest.mark.e2e
def test_a_reviewer_is_offered_a_decision_and_told_when_it_cannot_be_delivered(
    page: Page, base_url: str
) -> None:
    """This server runs no workflows, so the decision is refused - and the page says so."""
    tenant = new_tenant()
    awaiting_review(api(base_url, tenant, **{"X-Requester-ID": "someone-else"}))

    as_tenant(page, base_url, tenant)
    page.get_by_label("Role").select_option("reviewer")
    page.get_by_role("button", name="Open").click()
    detail = page.get_by_label("Assignment detail")
    expect(detail.locator("#detail-status")).to_have_text("review required")

    detail.get_by_role("button", name="Approve").click()

    expect(page.get_by_role("status")).to_have_text("workflows are not enabled")


@pytest.mark.e2e
def test_a_requester_cannot_decide_a_review(page: Page, base_url: str) -> None:
    tenant = new_tenant()
    awaiting_review(api(base_url, tenant))

    as_tenant(page, base_url, tenant)
    page.get_by_role("button", name="Open").click()
    page.get_by_label("Assignment detail").get_by_role("button", name="Reject").click()

    expect(page.get_by_role("status")).to_have_text("only a reviewer may decide a review")


@pytest.mark.e2e
def test_one_tenant_never_sees_another_tenants_assignments(page: Page, base_url: str) -> None:
    api(base_url, new_tenant()).post("/jobs", json={"question": "Only its own tenant sees this"})

    as_tenant(page, base_url, new_tenant())

    expect(page.locator("article")).to_have_count(0)
