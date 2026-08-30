import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator

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


@pytest.mark.e2e
def test_requester_can_create_and_see_assignment(page: Page, base_url: str) -> None:
    page.goto(base_url)
    page.get_by_label("Research question").fill("Compare primary-source release claims")
    page.get_by_role("button", name="Create assignment").click()

    expect(page.get_by_role("status")).to_have_text("Assignment created.")
    card = page.locator("article")
    expect(card.get_by_role("heading")).to_have_text("Compare primary-source release claims")
    expect(card).to_contain_text("created")
