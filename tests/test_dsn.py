"""DSN parsing, including a CostKey served under a path."""
import pytest

from costkey.client import _parse_dsn


def test_root_dsn():
    assert _parse_dsn("https://ck_abc@costkey.example.com/proj_1") == (
        "https://costkey.example.com/api/v1/events", "ck_abc", "proj_1")


def test_dsn_under_a_path():
    assert _parse_dsn("https://ck_abc@portal-dev.example.com/costkey/proj_123") == (
        "https://portal-dev.example.com/costkey/api/v1/events", "ck_abc", "proj_123")
    assert _parse_dsn("http://k@localhost:4300/tools/costkey/proj_9/") == (
        "http://localhost:4300/tools/costkey/api/v1/events", "k", "proj_9")


def test_dsn_missing_project():
    with pytest.raises(ValueError, match="missing project ID"):
        _parse_dsn("https://k@costkey.example.com/")
