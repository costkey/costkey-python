"""Commit and repository detection, and what each batch says about them."""
import gzip
import json
import os

import httpx
import pytest

import costkey.client as client_module
import costkey.transport as transport_module
from costkey.source import (
    COMMIT_ENV_VARS,
    commit_from_git,
    detect_commit,
    detect_repository,
    normalize_commit,
    normalize_repository,
    repository_from_git,
)
from costkey.transport import Transport

SHA = "0123456789abcdef0123456789abcdef01234567"
OTHER = "fedcba9876543210fedcba9876543210fedcba98"


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(text)


def _fake_repo(root, head, *, loose=None, packed=None, config=None):
    git = os.path.join(root, ".git")
    _write(os.path.join(git, "HEAD"), head)
    for ref, sha in (loose or {}).items():
        _write(os.path.join(git, *ref.split("/")), sha + "\n")
    if packed is not None:
        _write(os.path.join(git, "packed-refs"), packed)
    if config is not None:
        _write(os.path.join(git, "config"), config)
    return git


# --- commit ------------------------------------------------------------------

def test_commit_must_be_7_to_40_hex_and_is_lowercased():
    assert normalize_commit(SHA.upper()) == SHA
    assert normalize_commit("  abc1234\n") == "abc1234"
    assert normalize_commit("abc123") is None
    assert normalize_commit(SHA + "0") is None
    assert normalize_commit("main") is None
    assert normalize_commit("pr/123") is None
    assert normalize_commit(None) is None
    assert normalize_commit(42) is None


def test_commit_precedence(tmp_path):
    _fake_repo(str(tmp_path), OTHER + "\n")
    env = {"COSTKEY_COMMIT": "a" * 40, "GITHUB_SHA": "b" * 40, "GIT_SHA": "c" * 40}
    assert detect_commit("D" * 40, env=env, start=str(tmp_path)) == "d" * 40
    assert detect_commit(None, env=env, start=str(tmp_path)) == "a" * 40
    del env["COSTKEY_COMMIT"]
    assert detect_commit(None, env=env, start=str(tmp_path)) == "b" * 40
    del env["GITHUB_SHA"]
    assert detect_commit(None, env=env, start=str(tmp_path)) == "c" * 40
    assert detect_commit(None, env={}, start=str(tmp_path)) == OTHER


def test_platform_variables_are_checked_in_order():
    for index, key in enumerate(COMMIT_ENV_VARS):
        env = {later: "e" * 40 for later in COMMIT_ENV_VARS[index + 1:]}
        env[key] = SHA
        assert detect_commit(env=env, start="/") == SHA, key


def test_an_invalid_value_falls_through(tmp_path):
    env = {"COSTKEY_COMMIT": "not-a-sha", "SOURCE_VERSION": "refs/heads/main", "COMMIT_SHA": SHA}
    assert detect_commit("nope", env=env, start=str(tmp_path)) == SHA


def test_unknown_commit_is_none(tmp_path):
    assert detect_commit(env={}, start=str(tmp_path)) is None


def test_git_head_on_a_loose_branch_ref(tmp_path):
    _fake_repo(str(tmp_path), "ref: refs/heads/main\n", loose={"refs/heads/main": SHA})
    nested = tmp_path / "apps" / "service" / "src"
    nested.mkdir(parents=True)
    assert commit_from_git(str(nested)) == SHA


def test_git_head_on_a_packed_ref(tmp_path):
    _fake_repo(str(tmp_path), "ref: refs/heads/feature/x\n", packed=(
        "# pack-refs with: peeled fully-peeled sorted\n"
        f"{OTHER} refs/heads/main\n"
        f"{SHA.upper()} refs/heads/feature/x\n"
        f"^{OTHER}\n"
    ))
    assert commit_from_git(str(tmp_path)) == SHA


def test_loose_ref_wins_over_packed(tmp_path):
    _fake_repo(str(tmp_path), "ref: refs/heads/main\n", loose={"refs/heads/main": SHA},
               packed=f"{OTHER} refs/heads/main\n")
    assert commit_from_git(str(tmp_path)) == SHA


def test_detached_head(tmp_path):
    _fake_repo(str(tmp_path), SHA + "\n")
    assert commit_from_git(str(tmp_path)) == SHA


def test_linked_worktree(tmp_path):
    main = tmp_path / "main"
    common = _fake_repo(str(main), "ref: refs/heads/main\n", packed=f"{SHA} refs/heads/wt\n",
                        config='[remote "origin"]\n\turl = git@github.com:acme/app.git\n')
    worktree_git = os.path.join(common, "worktrees", "wt")
    _write(os.path.join(worktree_git, "HEAD"), "ref: refs/heads/wt\n")
    _write(os.path.join(worktree_git, "commondir"), "../..\n")
    checkout = tmp_path / "wt"
    checkout.mkdir()
    (checkout / ".git").write_text(f"gitdir: {worktree_git}\n")
    assert commit_from_git(str(checkout)) == SHA
    assert repository_from_git(str(checkout)) == "https://github.com/acme/app"


def test_unborn_or_broken_head_is_unknown(tmp_path):
    _fake_repo(str(tmp_path), "ref: refs/heads/main\n")
    assert commit_from_git(str(tmp_path)) is None
    (tmp_path / ".git" / "HEAD").write_text("garbage")
    assert commit_from_git(str(tmp_path)) is None


def test_ref_cannot_escape_the_git_dir(tmp_path):
    _write(str(tmp_path / "secret"), SHA)
    _fake_repo(str(tmp_path / "repo"), "ref: ../../secret\n")
    assert commit_from_git(str(tmp_path / "repo")) is None


# --- repository --------------------------------------------------------------

@pytest.mark.parametrize("remote, expected", [
    ("git@github.com:org/repo.git", "https://github.com/org/repo"),
    ("git@github.com:org/repo", "https://github.com/org/repo"),
    ("https://x@github.com/org/repo.git", "https://github.com/org/repo"),
    ("https://user:ghp_secret@github.com/org/repo.git/", "https://github.com/org/repo"),
    ("https://oauth2:glpat-abc@gitlab.com/group/sub/repo.git", "https://gitlab.com/group/sub/repo"),
    ("http://bitbucket.org/team/repo", "https://bitbucket.org/team/repo"),
    ("ssh://git@github.com/org/repo.git", "https://github.com/org/repo"),
    ("ssh://git@ghe.example.com:7999/org/repo.git", "https://ghe.example.com/org/repo"),
    ("git://github.com/org/repo.git", "https://github.com/org/repo"),
    ("https://GitHub.com/Org/Repo?token=abc#frag", "https://github.com/Org/Repo"),
    ("https://git.example.com:8443/org/repo.git", "https://git.example.com:8443/org/repo"),
    ("/srv/git/repo.git", None),
    ("file:///srv/git/repo.git", None),
    ("C:\\repos\\app", None),
    ("https://github.com/", None),
    ("", None),
    (None, None),
])
def test_repository_normalisation(remote, expected):
    assert normalize_repository(remote) == expected


def test_credentials_never_survive():
    for remote in ("https://user:hunter2@github.com/org/repo.git",
                   "https://hunter2@github.com/org/repo"):
        assert "hunter2" not in normalize_repository(remote)
    env = {"COSTKEY_REPOSITORY": "https://ci:hunter2@gitlab.com/org/repo.git"}
    assert detect_repository(env=env, start="/") == "https://gitlab.com/org/repo"


def test_repository_precedence(tmp_path):
    _fake_repo(str(tmp_path), SHA, config='[remote "origin"]\n\turl = git@github.com:from/git.git\n')
    env = {
        "COSTKEY_REPOSITORY": "https://github.com/from/costkey",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_REPOSITORY": "from/actions",
    }
    start = str(tmp_path)
    assert detect_repository("git@github.com:from/arg.git", env=env, start=start) == "https://github.com/from/arg"
    assert detect_repository(env=env, start=start) == "https://github.com/from/costkey"
    del env["COSTKEY_REPOSITORY"]
    assert detect_repository(env=env, start=start) == "https://github.com/from/actions"
    assert detect_repository(env={}, start=start) == "https://github.com/from/git"


@pytest.mark.parametrize("env, expected", [
    ({"GITHUB_SERVER_URL": "https://ghe.acme.com", "GITHUB_REPOSITORY": "o/r"}, "https://ghe.acme.com/o/r"),
    ({"VERCEL_GIT_PROVIDER": "gitlab", "VERCEL_GIT_REPO_OWNER": "o", "VERCEL_GIT_REPO_SLUG": "r"}, "https://gitlab.com/o/r"),
    ({"VERCEL_GIT_PROVIDER": "bitbucket", "VERCEL_GIT_REPO_OWNER": "o", "VERCEL_GIT_REPO_SLUG": "r"}, "https://bitbucket.org/o/r"),
    ({"RENDER_GIT_REPO_SLUG": "o/r"}, "https://github.com/o/r"),
    ({"RAILWAY_GIT_REPO_OWNER": "o", "RAILWAY_GIT_REPO_NAME": "r"}, "https://github.com/o/r"),
    ({"CI_PROJECT_URL": "https://gitlab.acme.com/group/r"}, "https://gitlab.acme.com/group/r"),
    ({"BITBUCKET_GIT_HTTP_ORIGIN": "http://bitbucket.org/o/r"}, "https://bitbucket.org/o/r"),
    ({"CIRCLE_REPOSITORY_URL": "git@github.com:o/r.git"}, "https://github.com/o/r"),
])
def test_platform_repository_variables(env, expected, tmp_path):
    assert detect_repository(env=env, start=str(tmp_path)) == expected


def test_git_config_reads_only_origin(tmp_path):
    _fake_repo(str(tmp_path), SHA, config=(
        "[core]\n\trepositoryformatversion = 0\n"
        '[remote "upstream"]\n\turl = git@github.com:up/stream.git\n'
        '[remote "origin"]\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n'
        "\turl = https://naman:token@github.com/thrive-money/core.git\n"
        '[branch "main"]\n\tremote = origin\n'
    ))
    assert repository_from_git(str(tmp_path)) == "https://github.com/thrive-money/core"


def test_no_origin_is_unknown(tmp_path):
    _fake_repo(str(tmp_path), SHA, config='[remote "upstream"]\n\turl = git@github.com:up/stream.git\n')
    assert repository_from_git(str(tmp_path)) is None


# --- the batch ---------------------------------------------------------------

def _posted_body(monkeypatch, transport):
    posted = []

    def post(_url, *, content, headers, **_kwargs):
        if headers.get("Content-Encoding") == "gzip":
            content = gzip.decompress(content)
        posted.append(json.loads(content))
        return httpx.Response(202)

    monkeypatch.setattr(transport_module.httpx, "post", post)
    assert transport._send_batch([{"id": "e1"}])
    return posted[0]


def test_batch_carries_commit_and_repository(monkeypatch):
    transport = Transport("https://ingest.test/api/v1/events", "key", 50, 5, False,
                          release=SHA, commit=SHA, repository="https://github.com/o/r")
    body = _posted_body(monkeypatch, transport)
    assert body["release"] == SHA
    assert body["commit"] == SHA
    assert body["repository"] == "https://github.com/o/r"
    assert body["events"] == [{"id": "e1"}]


def test_batch_omits_unknown_fields(monkeypatch):
    body = _posted_body(monkeypatch, Transport("https://ingest.test/api/v1/events", "key", 50, 5, False))
    assert set(body) == {"sdkVersion", "events"}


@pytest.fixture()
def init_transport(monkeypatch, tmp_path):
    """Run init without patching clients or touching the network; yield the
    Transport kwargs it chose."""
    created = {}

    class FakeTransport:
        def __init__(self, **kwargs):
            created.update(kwargs)

        def start(self):
            pass

    monkeypatch.setattr(client_module, "Transport", FakeTransport)
    monkeypatch.setattr(client_module, "patch", lambda **kwargs: created.setdefault("context", kwargs["default_context"]))
    monkeypatch.setattr(client_module, "start_runtime_config_fetch", lambda *a, **k: None)
    for key in ("COSTKEY_RELEASE", "COSTKEY_COMMIT", "COSTKEY_REPOSITORY", "GITHUB_REPOSITORY",
                "GITHUB_SERVER_URL", "VERCEL_GIT_PROVIDER", "RENDER_GIT_REPO_SLUG", "RAILWAY_GIT_REPO_OWNER",
                "CI_PROJECT_URL", "BITBUCKET_GIT_HTTP_ORIGIN", "CIRCLE_REPOSITORY_URL", *COMMIT_ENV_VARS):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(client_module, "_initialized", False)

    def run(**kwargs):
        created.clear()
        client_module._initialized = False
        client_module.init("https://k@ingest.test/proj", scan_callgraph=False,
                           project_root=str(tmp_path), **kwargs)
        return dict(created)

    yield run
    client_module._initialized = False
    client_module._transport = None


def test_init_release_defaults_to_the_detected_commit(init_transport, monkeypatch, tmp_path):
    _fake_repo(str(tmp_path), "ref: refs/heads/main\n", loose={"refs/heads/main": SHA},
               config='[remote "origin"]\n\turl = git@github.com:acme/app.git\n')
    chosen = init_transport()
    assert chosen["commit"] == SHA
    assert chosen["release"] == SHA
    assert chosen["repository"] == "https://github.com/acme/app"
    assert chosen["context"]["release"] == SHA


def test_init_release_precedence(init_transport, monkeypatch):
    monkeypatch.setenv("COSTKEY_COMMIT", SHA)
    monkeypatch.setenv("COSTKEY_RELEASE", "v1.2.3")
    assert init_transport()["release"] == "v1.2.3"
    chosen = init_transport(release="v2", commit=OTHER, repository="https://t:secret@github.com/o/r.git")
    assert chosen["release"] == "v2"
    assert chosen["commit"] == OTHER
    assert chosen["repository"] == "https://github.com/o/r"


def test_init_with_nothing_known(init_transport):
    chosen = init_transport()
    assert chosen["release"] is None
    assert chosen["commit"] is None
    assert chosen["repository"] is None
