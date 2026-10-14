"""Which commit, in which repository, the running code came from.

CostKey shows the deploy each call came from and links it to the source. The
commit and repository are read once at ``init`` from, in order: the explicit
argument, ``COSTKEY_COMMIT`` / ``COSTKEY_REPOSITORY``, the variables CI and
hosting platforms set, and finally the ``.git`` directory found by walking up
from the project root. No subprocess is run. Nothing here raises: anything that
cannot be read is unknown.
"""
from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

_SHA = re.compile(r"[0-9a-f]{7,40}")

# Checked after COSTKEY_COMMIT, in this order.
COMMIT_ENV_VARS = (
    "GITHUB_SHA",                         # GitHub Actions
    "VERCEL_GIT_COMMIT_SHA",              # Vercel
    "RENDER_GIT_COMMIT",                  # Render
    "RAILWAY_GIT_COMMIT_SHA",             # Railway
    "CF_PAGES_COMMIT_SHA",                # Cloudflare Pages
    "CI_COMMIT_SHA",                      # GitLab CI
    "BITBUCKET_COMMIT",                   # Bitbucket Pipelines
    "CIRCLE_SHA1",                        # CircleCI
    "COMMIT_REF",                         # Netlify
    "HEROKU_SLUG_COMMIT",                 # Heroku (dyno metadata)
    "SOURCE_VERSION",                     # Heroku builds, AWS CodeBuild
    "CODEBUILD_RESOLVED_SOURCE_VERSION",  # AWS CodeBuild
    "SOURCE_COMMIT",                      # Coolify, Docker Hub builds
    "GIT_COMMIT",                         # Jenkins
    "GIT_SHA",
    "COMMIT_SHA",                         # Google Cloud Build
)

_PROVIDER_HOSTS = {
    "github": "github.com",
    "gitlab": "gitlab.com",
    "bitbucket": "bitbucket.org",
}


def normalize_commit(value: object) -> Optional[str]:
    """A 7–40 character hex SHA in lowercase, or None."""
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()
    return candidate if _SHA.fullmatch(candidate) else None


def normalize_repository(value: object) -> Optional[str]:
    """``https://host/owner/repo`` for any https, http, ssh, git or scp-style
    remote, with credentials, port (for ssh), query and ``.git`` removed."""
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        if "://" not in raw:
            # scp-style: [user@]host:owner/repo.git
            match = re.fullmatch(r"(?:[^@/\s]+@)?([^:/\s]+):(?!//)(.+)", raw)
            if not match:
                return None
            host, path = match.group(1), match.group(2)
            port = None
        else:
            parts = urlsplit(raw)
            if parts.scheme.lower() not in ("https", "http", "ssh", "git", "git+ssh", "ssh+git"):
                return None
            host = parts.hostname or ""
            path = parts.path
            port = None
            if parts.scheme.lower() in ("https", "http"):
                try:
                    port = parts.port
                except ValueError:
                    port = None
                if port in (80, 443):
                    port = None
        host = host.lower()
        path = path.strip("/")
        if path.endswith(".git"):
            path = path[: -len(".git")]
        path = path.strip("/")
        if not host or not path or "\\" in path or ".." in path.split("/"):
            return None
        netloc = f"{host}:{port}" if port else host
        return urlunsplit(("https", netloc, "/" + path, "", ""))
    except Exception:
        return None


def _repo_from_env(env: Mapping[str, str]) -> Optional[str]:
    explicit = normalize_repository(env.get("COSTKEY_REPOSITORY"))
    if explicit:
        return explicit

    github = env.get("GITHUB_REPOSITORY")
    if github:
        server = (env.get("GITHUB_SERVER_URL") or "https://github.com").rstrip("/")
        found = normalize_repository(f"{server}/{github}")
        if found:
            return found

    owner, slug = env.get("VERCEL_GIT_REPO_OWNER"), env.get("VERCEL_GIT_REPO_SLUG")
    host = _PROVIDER_HOSTS.get((env.get("VERCEL_GIT_PROVIDER") or "").lower())
    if host and owner and slug:
        found = normalize_repository(f"https://{host}/{owner}/{slug}")
        if found:
            return found

    render = env.get("RENDER_GIT_REPO_SLUG")
    if render:
        found = normalize_repository(f"https://github.com/{render}")
        if found:
            return found

    owner, name = env.get("RAILWAY_GIT_REPO_OWNER"), env.get("RAILWAY_GIT_REPO_NAME")
    if owner and name:
        found = normalize_repository(f"https://github.com/{owner}/{name}")
        if found:
            return found

    for key in ("CI_PROJECT_URL", "BITBUCKET_GIT_HTTP_ORIGIN", "CIRCLE_REPOSITORY_URL"):
        found = normalize_repository(env.get(key))
        if found:
            return found
    return None


def _read(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read(1024 * 1024)
    except Exception:
        return None


def find_git_dirs(start: Optional[str] = None) -> Optional[tuple[str, str]]:
    """(git dir, common dir) for the repository containing ``start``.

    The common dir is where refs, packed-refs and config live; it differs from
    the git dir for a linked worktree.
    """
    try:
        current = os.path.abspath(start or os.getcwd())
    except Exception:
        return None
    for _ in range(64):
        candidate = os.path.join(current, ".git")
        git_dir: Optional[str] = None
        if os.path.isdir(candidate):
            git_dir = candidate
        elif os.path.isfile(candidate):
            # Worktrees and submodules: ".git" is a file naming the real dir.
            content = _read(candidate) or ""
            if content.startswith("gitdir:"):
                pointed = content[len("gitdir:"):].strip()
                git_dir = os.path.normpath(os.path.join(current, pointed))
        if git_dir and os.path.isdir(git_dir):
            common = git_dir
            pointer = _read(os.path.join(git_dir, "commondir"))
            if pointer and pointer.strip():
                common = os.path.normpath(os.path.join(git_dir, pointer.strip()))
            return git_dir, common
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent
    return None


def _resolve_ref(git_dir: str, common_dir: str, ref: str) -> Optional[str]:
    if ".." in ref.split("/"):
        return None
    for base in (git_dir, common_dir):
        found = normalize_commit(_read(os.path.join(base, *ref.split("/"))))
        if found:
            return found
    packed = _read(os.path.join(common_dir, "packed-refs")) or ""
    for line in packed.splitlines():
        if not line or line[0] in "#^":
            continue
        sha, _, name = line.partition(" ")
        if name.strip() == ref:
            return normalize_commit(sha)
    return None


def commit_from_git(start: Optional[str] = None) -> Optional[str]:
    try:
        dirs = find_git_dirs(start)
        if not dirs:
            return None
        git_dir, common_dir = dirs
        head = (_read(os.path.join(git_dir, "HEAD")) or "").strip()
        if head.startswith("ref:"):
            return _resolve_ref(git_dir, common_dir, head[len("ref:"):].strip())
        return normalize_commit(head)
    except Exception:
        return None


def repository_from_git(start: Optional[str] = None) -> Optional[str]:
    try:
        dirs = find_git_dirs(start)
        if not dirs:
            return None
        config = _read(os.path.join(dirs[1], "config")) or ""
        in_origin = False
        for line in config.splitlines():
            stripped = line.strip()
            if not stripped or stripped[0] in "#;":
                continue
            if stripped.startswith("["):
                in_origin = re.fullmatch(r'\[\s*remote\s+"origin"\s*\]', stripped) is not None
                continue
            if in_origin:
                key, sep, value = stripped.partition("=")
                if sep and key.strip().lower() == "url":
                    return normalize_repository(value.strip().strip('"'))
        return None
    except Exception:
        return None


def detect_commit(explicit: object = None, env: Optional[Mapping[str, str]] = None,
                  start: Optional[str] = None) -> Optional[str]:
    """The commit the running code was built from, or None."""
    try:
        environ = os.environ if env is None else env
        for value in (explicit, environ.get("COSTKEY_COMMIT"),
                      *(environ.get(key) for key in COMMIT_ENV_VARS)):
            found = normalize_commit(value)
            if found:
                return found
        return commit_from_git(start)
    except Exception:
        return None


def detect_repository(explicit: object = None, env: Optional[Mapping[str, str]] = None,
                      start: Optional[str] = None) -> Optional[str]:
    """The source repository as ``https://host/owner/repo``, or None."""
    try:
        found = normalize_repository(explicit)
        if found:
            return found
        found = _repo_from_env(os.environ if env is None else env)
        if found:
            return found
        return repository_from_git(start)
    except Exception:
        return None
