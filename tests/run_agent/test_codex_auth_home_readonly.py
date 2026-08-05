"""Offline regressions for the explicit read-only Codex auth snapshot."""

from __future__ import annotations

import base64
import json
import logging
import os
import socket
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import agent.auxiliary_client as auxiliary_client
import hermes_cli.auth as auth
import run_agent


def _jwt(expires_at: float, *, sentinel: str = "") -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').decode().rstrip("=")
    claims = {"exp": expires_at}
    if sentinel:
        claims["sentinel"] = sentinel
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"{header}.{payload}.offline-signature"


def _write_auth_home(
    root: Path,
    *,
    token: str | None = None,
    document: object | None = None,
) -> tuple[Path, str]:
    auth_home = root / "codex-auth"
    auth_home.mkdir(mode=0o700, parents=True)
    token = token or _jwt(time.time() + 3600)
    if document is None:
        document = {
            "providers": {
                "openai-codex": {
                    "tokens": {
                        "access_token": token,
                        "refresh_token": "unused-offline-fixture",
                    }
                }
            }
        }
    auth_file = auth_home / "auth.json"
    auth_file.write_text(json.dumps(document), encoding="utf-8")
    auth_file.chmod(0o600)
    return auth_home, token


def _tree_snapshot(root: Path) -> list[tuple[str, int, bytes]]:
    return [
        (
            str(path.relative_to(root)),
            path.stat().st_mode & 0o777,
            b"" if path.is_dir() else path.read_bytes(),
        )
        for path in sorted(root.rglob("*"))
    ]


def _fake_routed_client(*, api_key: str, base_url: str, **kwargs):
    return SimpleNamespace(
        api_key=api_key,
        base_url=base_url,
        _custom_headers=dict(kwargs.get("default_headers") or {}),
    )


def _forbidden(label: str):
    def _raise(*args, **kwargs):
        raise AssertionError(f"forbidden auth path reached: {label}")

    return _raise


def _patch_agent_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **kwargs: [])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})
    monkeypatch.setattr(
        run_agent.AIAgent,
        "_create_openai_client",
        lambda self, client_kwargs, **kwargs: SimpleNamespace(**client_kwargs),
    )
    monkeypatch.setattr(auxiliary_client, "_create_openai_client", _fake_routed_client)


def test_snapshot_uses_one_bounded_descriptor_relative_read(tmp_path, monkeypatch):
    auth_home, token = _write_auth_home(tmp_path)
    before = _tree_snapshot(auth_home)
    real_open = os.open
    real_read = os.read
    open_calls = []
    read_calls = []

    def recording_open(path, flags, mode=0o777, *, dir_fd=None):
        open_calls.append((path, flags, dir_fd))
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def recording_read(fd, amount):
        read_calls.append((fd, amount))
        return real_read(fd, amount)

    monkeypatch.setattr(auth.os, "open", recording_open)
    monkeypatch.setattr(auth.os, "read", recording_read)

    result = auth.resolve_codex_readonly_credentials(auth_home)

    assert result["api_key"] == token
    assert result["provider"] == "openai-codex"
    assert result["source"] == "readonly_auth_snapshot"
    assert len(open_calls) == 2
    assert os.fspath(open_calls[0][0]) == os.fspath(auth_home)
    assert open_calls[1][0] == "auth.json"
    assert open_calls[1][2] is not None
    assert open_calls[1][1] & os.O_NOFOLLOW
    assert not open_calls[1][1] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)
    assert len(read_calls) == 1
    assert read_calls[0][1] == auth.CODEX_READONLY_AUTH_MAX_BYTES
    assert _tree_snapshot(auth_home) == before


def test_default_raw_codex_path_is_unchanged(monkeypatch):
    token = _jwt(time.time() + 3600)
    ordinary_reads = []
    static_reads = []
    monkeypatch.setattr(
        auxiliary_client,
        "_read_codex_access_token",
        lambda: ordinary_reads.append(True) or token,
    )
    monkeypatch.setattr(
        auth,
        "resolve_codex_readonly_credentials",
        lambda *_a, **_k: static_reads.append(True),
    )
    monkeypatch.setattr(auxiliary_client, "_create_openai_client", _fake_routed_client)

    client, model = auxiliary_client.resolve_provider_client(
        "openai-codex", model="gpt-5.6-sol", raw_codex=True
    )

    assert client.api_key == token
    assert model == "gpt-5.6-sol"
    assert ordinary_reads == [True]
    assert static_reads == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"provider": "openrouter"}, "provider=openai-codex"),
        ({"provider": "openai-codex", "api_key": "explicit"}, "explicit API key"),
        (
            {
                "provider": "openai-codex",
                "fallback_model": {"provider": "openrouter", "model": "fallback"},
            },
            "provider fallback",
        ),
        ({"provider": "openai-codex", "credential_pool": object()}, "credential pool"),
    ],
)
def test_invalid_agent_combinations_fail_during_initialization(
    tmp_path, monkeypatch, kwargs, message
):
    auth_home, _ = _write_auth_home(tmp_path)
    static_reads = []
    monkeypatch.setattr(
        auth,
        "resolve_codex_readonly_credentials",
        lambda *_a, **_k: static_reads.append(True),
    )

    with pytest.raises(ValueError, match=message):
        run_agent.AIAgent(
            model="gpt-5.6-sol",
            codex_auth_home=auth_home,
            enabled_toolsets=[],
            skip_context_files=True,
            skip_memory=True,
            **kwargs,
        )

    assert static_reads == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"provider": "openrouter", "raw_codex": True},
        {"provider": "openai-codex", "raw_codex": False},
        {
            "provider": "openai-codex",
            "raw_codex": True,
            "explicit_api_key": "explicit",
        },
    ],
)
def test_router_rejects_invalid_explicit_auth_home_combinations(tmp_path, kwargs):
    auth_home, _ = _write_auth_home(tmp_path)
    with pytest.raises(ValueError, match="codex_auth_home"):
        auxiliary_client.resolve_provider_client(
            model="gpt-5.6-sol", codex_auth_home=auth_home, **kwargs
        )


def test_agent_construction_uses_static_snapshot_once_without_side_effects(
    tmp_path, monkeypatch, capsys, caplog
):
    sentinel = "".join(("credential", "-sentinel-", "never-log"))
    auth_home, token = _write_auth_home(
        tmp_path, token=_jwt(time.time() + 3600, sentinel=sentinel)
    )
    before = _tree_snapshot(auth_home)
    _patch_agent_bootstrap(monkeypatch)
    real_resolver = auth.resolve_codex_readonly_credentials
    resolutions = []

    def recording_resolver(path):
        resolutions.append(Path(path))
        return real_resolver(path)

    monkeypatch.setattr(auth, "resolve_codex_readonly_credentials", recording_resolver)
    for name in (
        "_auth_file_path",
        "_recover_codex_tokens_from_cli",
        "_pool_codex_access_token",
        "_probe_codex_quota_restored",
        "resolve_codex_runtime_credentials",
    ):
        monkeypatch.setattr(auth, name, _forbidden(name))
    monkeypatch.setattr(
        auxiliary_client, "_read_codex_access_token", _forbidden("ordinary_codex_read")
    )
    monkeypatch.setattr(auxiliary_client, "load_pool", _forbidden("credential_pool"))
    monkeypatch.setattr(socket, "create_connection", _forbidden("network"))
    caplog.set_level(logging.DEBUG)

    agent = run_agent.AIAgent(
        provider="openai-codex",
        model="gpt-5.6-sol",
        codex_auth_home=auth_home,
        enabled_toolsets=[],
        max_iterations=1,
        max_tokens=900,
        quiet_mode=False,
        skip_context_files=True,
        skip_memory=True,
        load_soul_identity=False,
    )

    captured = capsys.readouterr()
    emitted = "\n".join((captured.out, captured.err, caplog.text, repr(agent)))
    assert resolutions == [auth_home]
    assert agent.api_key == token
    assert agent._codex_auth_static_snapshot is True
    assert agent._credential_pool is None
    assert agent.tools == []
    assert "read-only Codex OAuth snapshot" in captured.out
    assert token not in emitted
    assert sentinel not in emitted
    assert _tree_snapshot(auth_home) == before


def test_static_snapshot_disables_401_credential_reentry(monkeypatch):
    agent = run_agent.AIAgent.__new__(run_agent.AIAgent)
    agent._codex_auth_static_snapshot = True
    agent.api_mode = "codex_responses"
    agent.provider = "openai-codex"
    monkeypatch.setattr(
        auth,
        "resolve_codex_runtime_credentials",
        _forbidden("401 credential re-entry"),
    )

    assert agent._try_refresh_codex_client_credentials(force=True) is False


@pytest.mark.parametrize(
    ("fixture", "failure_class"),
    [
        ("missing", "file_open_failed"),
        ("oversized", "file_oversized"),
        ("malformed", "malformed"),
        ("symlink_file", "file_open_failed"),
        ("wrong_owner", "directory_wrong_owner"),
        ("unsafe_directory", "directory_unsafe_permissions"),
        ("unsafe_file", "file_unsafe_permissions"),
        ("expiry_unparseable", "expiry_unparseable"),
        ("expired", "expired"),
        ("inside_margin", "insufficient_lifetime"),
    ],
)
def test_invalid_snapshots_fail_closed_without_credential_material(
    tmp_path, monkeypatch, caplog, fixture, failure_class
):
    sentinel = "".join(("invalid", "-credential-", "sentinel"))
    now = 2_000_000_000.0
    auth_home, _ = _write_auth_home(tmp_path)
    auth_file = auth_home / "auth.json"

    if fixture == "missing":
        auth_file.unlink()
    elif fixture == "oversized":
        auth_file.write_bytes(b"x" * (auth.CODEX_READONLY_AUTH_MAX_BYTES + 1))
    elif fixture == "malformed":
        auth_file.write_text(sentinel, encoding="utf-8")
    elif fixture == "symlink_file":
        target = tmp_path / "target.json"
        target.write_text("{}", encoding="utf-8")
        target.chmod(0o600)
        auth_file.unlink()
        auth_file.symlink_to(target)
    elif fixture == "wrong_owner":
        current_uid = os.getuid()
        monkeypatch.setattr(auth.os, "getuid", lambda: current_uid + 1)
    elif fixture == "unsafe_directory":
        auth_home.chmod(0o720)
    elif fixture == "unsafe_file":
        auth_file.chmod(0o640)
    elif fixture == "expiry_unparseable":
        _replace_token(auth_file, _jwt_claims({"exp": sentinel}))
    elif fixture == "expired":
        _replace_token(auth_file, _jwt(now - 1, sentinel=sentinel))
    elif fixture == "inside_margin":
        _replace_token(auth_file, _jwt(now + 899, sentinel=sentinel))

    caplog.set_level(logging.DEBUG)
    with pytest.raises(auth.AuthError) as caught:
        auth.resolve_codex_readonly_credentials(auth_home, now=now)

    rendered = "\n".join((str(caught.value), repr(caught.value), caplog.text))
    assert failure_class in str(caught.value)
    assert os.fspath(auth_file) in str(caught.value)
    assert sentinel not in rendered


def _jwt_claims(claims: dict) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"{header}.{payload}.offline-signature"


def _replace_token(auth_file: Path, token: str) -> None:
    document = json.loads(auth_file.read_text(encoding="utf-8"))
    document["providers"]["openai-codex"]["tokens"]["access_token"] = token
    auth_file.write_text(json.dumps(document), encoding="utf-8")


def test_symlinked_auth_directory_fails_closed(tmp_path):
    real_home, _ = _write_auth_home(tmp_path / "real")
    symlink_home = tmp_path / "linked-auth"
    symlink_home.symlink_to(real_home, target_is_directory=True)

    with pytest.raises(auth.AuthError, match="directory_open_failed") as caught:
        auth.resolve_codex_readonly_credentials(symlink_home)

    assert os.fspath(symlink_home / "auth.json") in str(caught.value)


def test_wrong_owner_file_descriptor_fails_closed(tmp_path, monkeypatch):
    auth_home, _ = _write_auth_home(tmp_path)
    real_fstat = os.fstat
    calls = 0

    def wrong_file_owner(fd):
        nonlocal calls
        calls += 1
        result = real_fstat(fd)
        if calls == 2:
            values = list(result)
            values[4] = result.st_uid + 1
            return os.stat_result(values)
        return result

    monkeypatch.setattr(auth.os, "fstat", wrong_file_owner)

    with pytest.raises(auth.AuthError, match="file_wrong_owner"):
        auth.resolve_codex_readonly_credentials(auth_home)


def test_non_regular_auth_file_fails_without_reading(tmp_path, monkeypatch):
    auth_home, _ = _write_auth_home(tmp_path)
    auth_file = auth_home / "auth.json"
    auth_file.unlink()
    auth_file.mkdir(mode=0o700)
    reads = []
    monkeypatch.setattr(auth.os, "read", lambda *_a: reads.append(True))

    with pytest.raises(auth.AuthError, match="file_not_regular"):
        auth.resolve_codex_readonly_credentials(auth_home)

    assert reads == []


def test_snapshot_changed_during_read_fails_closed(tmp_path, monkeypatch):
    auth_home, _ = _write_auth_home(tmp_path)
    auth_file = auth_home / "auth.json"
    real_read = os.read

    def racing_read(fd, amount):
        snapshot = real_read(fd, amount)
        auth_file.write_bytes(snapshot + b" ")
        return snapshot

    monkeypatch.setattr(auth.os, "read", racing_read)

    with pytest.raises(auth.AuthError, match="file_changed_during_read"):
        auth.resolve_codex_readonly_credentials(auth_home)


def test_expiry_beyond_margin_is_accepted(tmp_path):
    now = 2_000_000_000.0
    auth_home, token = _write_auth_home(tmp_path, token=_jwt(now + 901))

    result = auth.resolve_codex_readonly_credentials(auth_home, now=now)

    assert result["api_key"] == token
