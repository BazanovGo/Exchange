"""Environment capture.

Part of an experiment record: what interpreter, what package versions, what
commit produced a result. Captured as plain data so it can be diffed
between the original run and a reproduction attempt.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path

# Packages whose version can change a numeric result. Kept short on purpose:
# a diff of 300 transitive pins is noise, not evidence.
TRACKED_PACKAGES = ("numpy", "pandas", "scipy", "pyarrow")


@dataclass(frozen=True)
class GitInfo:
    """Repository state at capture time."""

    commit: str | None = None
    branch: str | None = None
    dirty: bool | None = None
    remote: str | None = None

    @property
    def is_clean_checkout(self) -> bool:
        return self.commit is not None and self.dirty is False


@dataclass(frozen=True)
class EnvironmentInfo:
    """Everything about the environment that can move a number."""

    python_version: str
    python_implementation: str
    platform: str
    machine: str
    packages: dict[str, str] = field(default_factory=dict)
    git: GitInfo = field(default_factory=GitInfo)

    def to_dict(self) -> dict:
        return asdict(self)

    def diff(self, other: EnvironmentInfo) -> dict[str, tuple[object, object]]:
        """Fields that differ between two captures, as ``{field: (a, b)}``."""
        out: dict[str, tuple[object, object]] = {}
        for key in ("python_version", "python_implementation", "platform", "machine"):
            a, b = getattr(self, key), getattr(other, key)
            if a != b:
                out[key] = (a, b)
        for name in sorted(set(self.packages) | set(other.packages)):
            a, b = self.packages.get(name), other.packages.get(name)
            if a != b:
                out[f"packages.{name}"] = (a, b)
        if self.git.commit != other.git.commit:
            out["git.commit"] = (self.git.commit, other.git.commit)
        return out


def capture_git(repo: str | Path | None = None) -> GitInfo:
    """Read commit/branch/dirty state, or blanks outside a repository."""
    cwd = Path(repo) if repo is not None else Path.cwd()

    def run(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return GitInfo()
    status = run("status", "--porcelain")
    return GitInfo(
        commit=commit,
        branch=run("rev-parse", "--abbrev-ref", "HEAD"),
        dirty=None if status is None else bool(status),
        remote=run("config", "--get", "remote.origin.url"),
    )


def package_versions(names: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str]:
    """Installed versions of the tracked packages (missing ones omitted)."""
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def capture_environment(repo: str | Path | None = None) -> EnvironmentInfo:
    """Snapshot the current environment for an experiment record."""
    return EnvironmentInfo(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=f"{platform.system()} {platform.release()}",
        machine=platform.machine(),
        packages=package_versions(),
        git=capture_git(repo),
    )


def environment_summary() -> str:
    """One-line description used by ``exchange doctor``."""
    info = capture_environment()
    pkgs = ", ".join(f"{k}={v}" for k, v in info.packages.items())
    return f"Python {info.python_version} on {info.platform} ({info.machine}); {pkgs or 'no tracked packages'}"


def check_environment() -> list[tuple[str, bool, str]]:
    """Environment checks for ``exchange doctor``: ``(name, ok, detail)``."""
    checks: list[tuple[str, bool, str]] = []

    ok_py = sys.version_info >= (3, 11)
    checks.append(("python>=3.11", ok_py, platform.python_version()))

    for name in TRACKED_PACKAGES:
        try:
            version = metadata.version(name)
            checks.append((name, True, version))
        except metadata.PackageNotFoundError:
            required = name in {"numpy", "pandas", "scipy"}
            checks.append((name, not required, "not installed"))

    git = capture_git()
    checks.append(("git checkout", git.commit is not None, git.commit or "not a git repository"))
    if git.commit is not None:
        checks.append((
            "git clean",
            git.dirty is False,
            "clean" if git.dirty is False else "uncommitted changes (results will not be reproducible)",
        ))
    return checks
