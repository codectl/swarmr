"""Credential discovery: the one thing between a fresh checkout and a run.

`load_env` writes into `os.environ`, so every test here restores the process
environment itself — `monkeypatch` only reverts changes it made, not the ones
made by the code under test. `build_model` is tested only as far as choosing
the provider's client, which constructs without a network call.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from swarmr.core.model import build_model, load_env


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A cwd with no .env above it, an empty home, and no model settings set."""
    saved = dict(os.environ)
    for key in [
        key
        for key in os.environ
        if key.startswith(("KIMI_", "SWARMR_")) or key.endswith("_API_KEY")
    ]:
        del os.environ[key]
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    yield work
    os.environ.clear()
    os.environ.update(saved)


def test_an_already_set_key_short_circuits_without_reading_any_file(
    isolated_env: Path,
) -> None:
    """The environment is authoritative: a stale .env must not shadow the key a
    caller exported for this run."""
    (isolated_env / ".env").write_text("KIMI_API_KEY=from-file\nKIMI_MODEL=from-file\n")
    os.environ["KIMI_API_KEY"] = "from-environment"

    load_env()

    assert os.environ["KIMI_API_KEY"] == "from-environment"
    assert "KIMI_MODEL" not in os.environ, "the file must not have been read at all"


def test_the_env_file_override_beats_the_file_in_the_working_directory(
    isolated_env: Path, tmp_path: Path
) -> None:
    """No path is baked in, so a caller can point at the file holding the key."""
    (isolated_env / ".env").write_text("KIMI_API_KEY=from-cwd\n")
    override = tmp_path / "elsewhere.env"
    override.write_text("KIMI_API_KEY=from-override\n")
    os.environ["KIMI_ENV_FILE"] = str(override)

    load_env()

    assert os.environ["KIMI_API_KEY"] == "from-override"


def test_a_key_is_found_by_walking_up_from_the_working_directory(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running from a subdirectory of the checkout must work like running from
    its root: the .env lives at the top."""
    (isolated_env.parent / ".env").write_text("KIMI_API_KEY=from-parent\n")
    nested = isolated_env / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    load_env()

    assert os.environ["KIMI_API_KEY"] == "from-parent"


def test_only_prefixed_keys_are_imported(isolated_env: Path) -> None:
    """A shared .env holds other projects' secrets; loading them all would put
    unrelated credentials into this process."""
    (isolated_env / ".env").write_text(
        "AWS_SECRET_ACCESS_KEY=nope\nOTHER=nope\nKIMI_API_KEY=k\n"
    )

    load_env()

    assert os.environ["KIMI_API_KEY"] == "k"
    assert os.environ.get("AWS_SECRET_ACCESS_KEY") != "nope"
    assert "OTHER" not in os.environ


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("KIMI_API_KEY=plain", "plain"),
        ('KIMI_API_KEY="quoted"', "quoted"),
        ("KIMI_API_KEY='quoted'", "quoted"),
        ("  KIMI_API_KEY = spaced  ", "spaced"),
        ('KIMI_API_KEY = "both"  ', "both"),
    ],
    ids=["plain", "double-quoted", "single-quoted", "spaced", "spaced-and-quoted"],
)
def test_quotes_and_surrounding_whitespace_are_stripped(
    isolated_env: Path, line: str, expected: str
) -> None:
    """A quoted value is shell syntax, not part of the key: sending the quotes
    to the provider fails the request with an unhelpful 401."""
    (isolated_env / ".env").write_text(f"{line}\n")

    load_env()

    assert os.environ["KIMI_API_KEY"] == expected


def test_comments_and_lines_without_an_assignment_are_ignored(
    isolated_env: Path,
) -> None:
    (isolated_env / ".env").write_text(
        "# KIMI_API_KEY=commented-out\n\nKIMI_JUNK\nKIMI_API_KEY=real\n"
    )

    load_env()

    assert os.environ["KIMI_API_KEY"] == "real"
    assert "KIMI_JUNK" not in os.environ
    assert not any(key.startswith("#") for key in os.environ)


def test_a_value_already_in_the_environment_is_never_overwritten(
    isolated_env: Path,
) -> None:
    """The file fills gaps; it does not win over an explicit export."""
    (isolated_env / ".env").write_text("KIMI_MODEL=from-file\nKIMI_API_KEY=k\n")
    os.environ["KIMI_MODEL"] = "from-environment"

    load_env()

    assert os.environ["KIMI_MODEL"] == "from-environment"
    assert os.environ["KIMI_API_KEY"] == "k"


def test_no_file_anywhere_leaves_the_environment_untouched(isolated_env: Path) -> None:
    """`build_model` then raises with an actionable message; `load_env` does not."""
    load_env()

    assert not [key for key in os.environ if key.startswith("KIMI_")]


def test_a_file_can_name_any_provider_and_carry_its_key(isolated_env: Path) -> None:
    """`SWARMR_MODEL` and the provider's own key variable come from the same
    .env; a line that is neither a setting nor a key never reaches the process."""
    (isolated_env / ".env").write_text(
        "SWARMR_MODEL=anthropic:some-model\nANTHROPIC_API_KEY=k\nDATABASE_URL=x\n"
    )

    model = build_model()

    assert type(model).__name__ == "ChatAnthropic"
    assert "DATABASE_URL" not in os.environ


def test_without_a_named_provider_the_model_is_kimi(isolated_env: Path) -> None:
    (isolated_env / ".env").write_text("KIMI_API_KEY=k\n")

    assert type(build_model()).__name__ == "ChatOpenAI"


def test_a_model_without_a_provider_is_refused_with_the_format(
    isolated_env: Path,
) -> None:
    os.environ["SWARMR_MODEL"] = "some-model"

    with pytest.raises(RuntimeError, match="<provider>:<model>"):
        build_model()
