"""The boundary the Python analysis server must be used within.

Section 8 requires an ephemeral container, disabled network and resource limits. This
module declares those limits and screens the submitted code for the capabilities the
sandbox is meant to withhold.

The screen is a fast refusal for obvious attempts, not a security boundary. Python code
cannot be made safe by inspection, so the container isolation and resource limits
declared here are what actually contain the calculation.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass

MAX_CODE_LENGTH = 20_000

FORBIDDEN_MODULES = frozenset(
    {
        "asyncio",
        "ctypes",
        "http",
        "importlib",
        "multiprocessing",
        "os",
        "pathlib",
        "pickle",
        "requests",
        "shutil",
        "socket",
        "subprocess",
        "sys",
        "urllib",
        "webbrowser",
    }
)

FORBIDDEN_NAMES = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "globals",
        "input",
        "locals",
        "open",
        "vars",
    }
)

FORBIDDEN_ATTRIBUTES = frozenset({"__builtins__", "__class__", "__globals__", "__subclasses__"})


class CalculationNotAllowed(ValueError):
    """Raised when submitted analysis code is refused before it is containerized."""


@dataclass(frozen=True)
class SandboxLimits:
    """The resource ceiling an analysis container runs under."""

    cpu_seconds: int = 10
    memory_mib: int = 512
    wall_clock_seconds: int = 30
    max_output_bytes: int = 262_144
    network_enabled: bool = False

    def __post_init__(self) -> None:
        if self.network_enabled:
            raise ValueError("the analysis sandbox must never be given network access")
        for name, value in (
            ("cpu_seconds", self.cpu_seconds),
            ("memory_mib", self.memory_mib),
            ("wall_clock_seconds", self.wall_clock_seconds),
            ("max_output_bytes", self.max_output_bytes),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.cpu_seconds > self.wall_clock_seconds:
            raise ValueError("cpu_seconds cannot exceed wall_clock_seconds")


def _module_root(name: str) -> str:
    return name.split(".", 1)[0]


def screen_calculation(code: str) -> str:
    """Refuse obviously disallowed analysis code, or return it unchanged.

    A syntactically invalid submission is refused here rather than costing a container
    start, and imports of the capabilities the sandbox withholds are named explicitly so
    the analyst learns why the calculation was rejected.
    """
    candidate = code.strip()
    if not candidate:
        raise CalculationNotAllowed("a calculation must not be empty")
    if len(candidate) > MAX_CODE_LENGTH:
        raise CalculationNotAllowed(f"a calculation must not exceed {MAX_CODE_LENGTH} characters")

    try:
        tree = ast.parse(candidate)
    except SyntaxError as error:
        raise CalculationNotAllowed(f"a calculation must be valid Python: {error.msg}") from error

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            _reject_modules(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            _reject_modules([node.module or ""])
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise CalculationNotAllowed(f"a calculation must not use {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
            raise CalculationNotAllowed(f"a calculation must not reach {node.attr}")

    return candidate


def _reject_modules(names: Iterable[str]) -> None:
    for name in names:
        root = _module_root(name)
        if root in FORBIDDEN_MODULES:
            raise CalculationNotAllowed(f"a calculation must not import {root}")
