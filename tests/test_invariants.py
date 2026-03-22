"""Contract tests for the P0-2 invariant: single auditable path to execution.

These tests enforce architectural boundaries by statically analysing the source
tree with the ``ast`` module.  They do NOT execute production code — they parse
it.  A failure here means a module boundary that must never be crossed has been
crossed.

Invariants verified
-------------------
* ``execution/`` must NOT import from ``signals``.
* ``data/`` must NOT import from ``signals``.
* :meth:`~execution.engine.ExecutionEngine.execute` must accept an
  ``ApprovedOrder`` parameter.
* ``_compute_gate_token`` must only be called from ``risk/base.py`` and
  ``risk/gate.py`` (plus test files).
* ``execution/engine.py`` must call ``verify_gate_token``.
* No file inside ``signals/`` may import from ``bus``.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Project root — derive it relative to this test file so the suite works
# regardless of the working directory.
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _python_files(directory: Path) -> list[Path]:
    """Return all ``.py`` files under *directory*, recursively.

    Args:
        directory: Root directory to walk.

    Returns:
        Sorted list of absolute :class:`~pathlib.Path` objects.
    """
    return sorted(directory.rglob("*.py"))


def _imports_from_module(source_path: Path, target_module: str) -> list[str]:
    """Return every import statement in *source_path* that references *target_module*.

    Both ``import target_module`` and ``from target_module[.sub] import …``
    forms are detected.

    Args:
        source_path: Path to a Python source file.
        target_module: Top-level module name to search for (e.g. ``"signals"``).

    Returns:
        List of human-readable import description strings; empty when no such
        import exists.
    """
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []

    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == target_module or alias.name.startswith(f"{target_module}."):
                    hits.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == target_module or module.startswith(f"{target_module}."):
                names = ", ".join(a.name for a in node.names)
                hits.append(f"from {module} import {names}")
    return hits


def _calls_to_name(source_path: Path, func_name: str) -> list[int]:
    """Return line numbers where *func_name* is called in *source_path*.

    Detects both bare calls ``func_name(…)`` and attribute calls
    ``obj.func_name(…)``.

    Args:
        source_path: Path to a Python source file.
        func_name: Function name to search for.

    Returns:
        Sorted list of line numbers (1-indexed).
    """
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []

    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == func_name:
            lines.append(node.lineno)
        elif isinstance(func, ast.Attribute) and func.attr == func_name:
            lines.append(node.lineno)
    return sorted(lines)


# ---------------------------------------------------------------------------
# Invariant tests
# ---------------------------------------------------------------------------


class TestSignalEngineNotImportedByExecution:
    """``execution/`` must not import anything from the ``signals`` package."""

    def test_signal_engine_not_imported_by_execution(self) -> None:
        """Scan every .py file in execution/ and assert no signals imports."""
        execution_dir = _ROOT / "execution"
        assert execution_dir.is_dir(), f"execution/ directory not found at {execution_dir}"

        violations: list[str] = []
        for py_file in _python_files(execution_dir):
            hits = _imports_from_module(py_file, "signals")
            for hit in hits:
                violations.append(f"{py_file.relative_to(_ROOT)}: {hit}")

        assert not violations, (
            "execution/ must not import from signals — P0-2 violation.\n"
            + "\n".join(violations)
        )


class TestSignalEngineNotImportedByData:
    """``data/`` must not import anything from the ``signals`` package."""

    def test_signal_engine_not_imported_by_data(self) -> None:
        """Scan every .py file in data/ and assert no signals imports."""
        data_dir = _ROOT / "data"
        assert data_dir.is_dir(), f"data/ directory not found at {data_dir}"

        violations: list[str] = []
        for py_file in _python_files(data_dir):
            hits = _imports_from_module(py_file, "signals")
            for hit in hits:
                violations.append(f"{py_file.relative_to(_ROOT)}: {hit}")

        assert not violations, (
            "data/ must not import from signals — architectural boundary violation.\n"
            + "\n".join(violations)
        )


class TestExecutionEngineRequiresApprovedOrder:
    """``ExecutionEngine.execute`` must declare an ``ApprovedOrder`` parameter."""

    def test_execution_engine_requires_approved_order(self) -> None:
        """Inspect the signature of ExecutionEngine.execute at import time.

        ``from __future__ import annotations`` makes all annotations lazy strings
        in Python 3.12, so ``inspect.signature`` alone returns string literals.
        ``typing.get_type_hints`` is used to resolve them to actual types.
        """
        import typing

        from execution.engine import ExecutionEngine
        from risk.base import ApprovedOrder

        sig = inspect.signature(ExecutionEngine.execute)
        params = dict(sig.parameters)

        non_self_params = {k: v for k, v in params.items() if k != "self"}
        assert non_self_params, "ExecutionEngine.execute has no non-self parameters."

        # Resolve string annotations to their actual runtime types.
        try:
            hints = typing.get_type_hints(ExecutionEngine.execute)
        except Exception as exc:
            pytest.fail(
                f"typing.get_type_hints(ExecutionEngine.execute) raised: {exc}"
            )

        # Verify that at least one non-return parameter resolves to ApprovedOrder.
        param_types = {k: v for k, v in hints.items() if k != "return"}
        approved_order_param = next(
            (name for name, typ in param_types.items() if typ is ApprovedOrder),
            None,
        )
        assert approved_order_param is not None, (
            f"ExecutionEngine.execute must have an ApprovedOrder-annotated parameter. "
            f"Resolved types: {param_types}"
        )


class TestApprovedOrderOnlyCreatedInRiskGate:
    """``_compute_gate_token`` must only be called from ``risk/base.py`` and ``risk/gate.py``."""

    def test_approved_order_only_created_in_risk_gate(self) -> None:
        """Walk all .py files; assert _compute_gate_token is called only in approved locations."""
        func_name = "_compute_gate_token"

        # Permitted call sites (relative to project root).
        permitted_relative: set[str] = {
            "risk/base.py",
            "risk/gate.py",
        }

        violations: list[str] = []
        for py_file in _python_files(_ROOT):
            rel = str(py_file.relative_to(_ROOT))

            # Skip __pycache__ directories.
            if "__pycache__" in rel:
                continue

            # Tests are allowed to call _compute_gate_token (e.g. to build fixtures).
            if rel.startswith("tests/"):
                continue

            # Skip the explicitly permitted files.
            if rel in permitted_relative:
                continue

            lines = _calls_to_name(py_file, func_name)
            if lines:
                violations.append(
                    f"{rel}: {func_name} called at line(s) {lines}"
                )

        assert not violations, (
            f"{func_name} must only be called in risk/base.py and risk/gate.py — "
            "P0-2 violation: token minting outside the risk gate.\n"
            + "\n".join(violations)
        )


class TestVerifyGateTokenCalledInExecution:
    """``execution/engine.py`` must call ``verify_gate_token``."""

    def test_verify_gate_token_called_in_execution(self) -> None:
        """Assert that execution/engine.py contains a call to verify_gate_token."""
        engine_path = _ROOT / "execution" / "engine.py"
        assert engine_path.exists(), f"execution/engine.py not found at {engine_path}"

        content = engine_path.read_text(encoding="utf-8")
        assert "verify_gate_token" in content, (
            "execution/engine.py does not call verify_gate_token — "
            "P0-2 invariant is not enforced at the execution boundary."
        )

        # Additionally confirm it is an actual call, not just a comment.
        lines = _calls_to_name(engine_path, "verify_gate_token")
        assert lines, (
            "execution/engine.py imports verify_gate_token but never calls it — "
            "P0-2 invariant is not enforced."
        )


class TestNoDirectBusToSignalEngine:
    """``signals/`` must not import from ``bus/`` — data flows via the feature store only."""

    def test_no_direct_bus_to_signal_engine(self) -> None:
        """Scan every .py file in signals/ and assert no bus imports."""
        signals_dir = _ROOT / "signals"
        assert signals_dir.is_dir(), f"signals/ directory not found at {signals_dir}"

        violations: list[str] = []
        for py_file in _python_files(signals_dir):
            hits = _imports_from_module(py_file, "bus")
            for hit in hits:
                violations.append(f"{py_file.relative_to(_ROOT)}: {hit}")

        assert not violations, (
            "signals/ must not import from bus/ — signals must receive market data "
            "exclusively via the feature store, never directly from the event bus.\n"
            + "\n".join(violations)
        )
