"""CostKey CLI."""
from __future__ import annotations

import json
import os
import secrets
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


TOKEN_DIR = Path.home() / ".costkey"
TOKEN_FILE = TOKEN_DIR / "token"


def _arg(args: list[str], name: str) -> str | None:
    try:
        idx = args.index(name)
    except ValueError:
        return None
    return args[idx + 1] if idx + 1 < len(args) else None


def _host(args: list[str]) -> str:
    return _arg(args, "--host") or os.environ.get("COSTKEY_HOST") or "https://app.costkey.dev"


def _load_token(args: list[str]) -> str | None:
    if token := _arg(args, "--token"):
        return token
    if token := os.environ.get("COSTKEY_TOKEN"):
        return token
    try:
        return TOKEN_FILE.read_text().strip()
    except OSError:
        return None


def _save_token(token: str) -> None:
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(token)
    TOKEN_FILE.chmod(0o600)


def _read(path: str) -> str:
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def _find_first(paths: tuple[str, ...]) -> str | None:
    for file in paths:
        if Path(file).exists():
            return file
    return None


def _py_entry_file() -> str | None:
    return _find_first(("main.py", "app.py", "asgi.py", "wsgi.py", "manage.py", "src/main.py", "src/app.py"))


def _node_entry_file() -> str | None:
    return _find_first(
        (
            "src/instrumentation.ts",
            "src/instrumentation.js",
            "instrumentation.ts",
            "instrumentation.js",
            "src/server.ts",
            "src/server.js",
            "server.ts",
            "server.js",
            "src/index.ts",
            "src/index.js",
            "index.ts",
            "index.js",
            "app.ts",
            "app.js",
        )
    )


def _detect() -> dict[str, object]:
    reqs = (_read("requirements.txt") + "\n" + _read("pyproject.toml") + "\n" + _read("Pipfile")).lower()
    if "fastapi" in reqs:
        return {
            "label": "FastAPI",
            "language": "py",
            "entrypoint": "Initialize CostKey near the top of your FastAPI app module before AI clients are used.",
            "entry_file": _py_entry_file(),
            "needs_sourcemaps": False,
            "sidecar": 'PYTHONPATH="$PWD/.costkey:$PYTHONPATH" uvicorn app:app',
        }
    if "django" in reqs:
        return {
            "label": "Django",
            "language": "py",
            "entrypoint": "Initialize CostKey in settings.py, wsgi.py, or asgi.py before AI clients are used.",
            "entry_file": _find_first(("settings.py", "wsgi.py", "asgi.py", "manage.py")) or _py_entry_file(),
            "needs_sourcemaps": False,
            "sidecar": 'PYTHONPATH="$PWD/.costkey:$PYTHONPATH" python manage.py runserver',
        }
    if "flask" in reqs:
        return {
            "label": "Flask",
            "language": "py",
            "entrypoint": "Initialize CostKey before creating routes or AI clients.",
            "entry_file": _py_entry_file(),
            "needs_sourcemaps": False,
            "sidecar": 'PYTHONPATH="$PWD/.costkey:$PYTHONPATH" flask run',
        }
    if Path("package.json").exists():
        try:
            pkg = json.loads(Path("package.json").read_text())
            deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
        except Exception:
            deps = {}
        if "next" in deps or any(Path(name).exists() for name in ("next.config.js", "next.config.mjs", "next.config.ts")):
            return {
                "label": "Next.js",
                "language": "ts",
                "entrypoint": "Initialize CostKey in your Next.js instrumentation file, or preload it with NODE_OPTIONS before the server starts.",
                "entry_file": _find_first(("src/instrumentation.ts", "src/instrumentation.js", "instrumentation.ts", "instrumentation.js")),
                "needs_sourcemaps": True,
                "sidecar": "NODE_OPTIONS='--import costkey/register' next start",
            }
        if "vite" in deps:
            return {
                "label": "Vite / bundled TypeScript",
                "language": "ts",
                "entrypoint": "Initialize CostKey in your server entrypoint before AI clients are constructed, or preload it with NODE_OPTIONS for server-side runtimes.",
                "entry_file": _node_entry_file(),
                "needs_sourcemaps": True,
                "sidecar": "NODE_OPTIONS='--import costkey/register' node dist/server.js",
            }
        if "@nestjs/core" in deps:
            return {
                "label": "NestJS",
                "language": "ts",
                "entrypoint": "Initialize CostKey in main.ts before NestFactory creates the app, or preload it with NODE_OPTIONS.",
                "entry_file": _find_first(("src/main.ts", "main.ts", "src/main.js", "main.js")),
                "needs_sourcemaps": False,
                "sidecar": "NODE_OPTIONS='--import costkey/register' npm start",
            }
        if "@remix-run/node" in deps or "@remix-run/serve" in deps:
            return {
                "label": "Remix",
                "language": "ts",
                "entrypoint": "Initialize CostKey in the server entrypoint that starts Remix, or preload it with NODE_OPTIONS.",
                "entry_file": _node_entry_file(),
                "needs_sourcemaps": True,
                "sidecar": "NODE_OPTIONS='--import costkey/register' npm start",
            }
        if "astro" in deps:
            return {
                "label": "Astro",
                "language": "ts",
                "entrypoint": "Initialize CostKey in the server adapter entrypoint, or preload it with NODE_OPTIONS for SSR/server output.",
                "entry_file": _node_entry_file(),
                "needs_sourcemaps": True,
                "sidecar": "NODE_OPTIONS='--import costkey/register' npm start",
            }
        if "@sveltejs/kit" in deps:
            return {
                "label": "SvelteKit",
                "language": "ts",
                "entrypoint": "Initialize CostKey in the server hook or server entrypoint, or preload it with NODE_OPTIONS.",
                "entry_file": _find_first(("src/hooks.server.ts", "src/hooks.server.js", "src/server.ts", "src/server.js")),
                "needs_sourcemaps": True,
                "sidecar": "NODE_OPTIONS='--import costkey/register' npm start",
            }
        return {
            "label": "Node / TypeScript",
            "language": "ts",
            "entrypoint": "Initialize CostKey in your app entrypoint before AI calls.",
            "entry_file": _node_entry_file(),
            "needs_sourcemaps": False,
            "sidecar": "NODE_OPTIONS='--import costkey/register' node index.js",
        }
    return {
        "label": "Python",
        "language": "py",
        "entrypoint": "Initialize CostKey in your app entrypoint before AI calls.",
        "entry_file": _py_entry_file(),
        "needs_sourcemaps": False,
        "sidecar": 'PYTHONPATH="$PWD/.costkey:$PYTHONPATH" python main.py',
    }


def _json_request(url: str, *, token: str | None = None, body: dict[str, object] | None = None) -> dict[str, object]:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, data=data, headers=headers, method="POST" if body is not None else "GET")
    with urlopen(req, timeout=20) as res:
        return json.loads(res.read().decode())


def _login(host: str, args: list[str]) -> str:
    existing = _load_token(args)
    if existing:
        try:
            _json_request(f"{host}/auth/me", token=existing)
            return existing
        except Exception:
            pass

    state = secrets.token_hex(16)
    token_holder: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_OPTIONS(self) -> None:
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_POST(self) -> None:
            if self.path != "/callback":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode())
            if payload.get("state") != state:
                self.send_response(400)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b'{"error":"State mismatch"}')
                return
            token_holder["token"] = str(payload["token"])
            _save_token(token_holder["token"])
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def log_message(self, *_args: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    url = f"{host}/auth/cli?{urlencode({'port': port, 'state': state})}"
    print("\nOpening browser to authenticate with CostKey...\n")
    print(f"If the browser does not open, visit:\n{url}\n")
    webbrowser.open(url)

    while "token" not in token_holder:
        server.handle_request()
    server.server_close()
    return token_holder["token"]


def _create_project(host: str, token: str, args: list[str]) -> str:
    name = _arg(args, "--name") or Path.cwd().name or "my-app"
    print(f'Creating CostKey project "{name}"...')
    try:
        data = _json_request(f"{host}/api/v1/projects", token=token, body={"name": name})
    except HTTPError as err:
        try:
            body = json.loads(err.read().decode())
            message = body.get("error", str(err))
        except Exception:
            message = str(err)
        raise RuntimeError(message) from err
    except URLError as err:
        raise RuntimeError(str(err)) from err
    return str(data["dsn"])


def _ensure_env(dsn: str) -> None:
    env = Path(".env")
    current = env.read_text() if env.exists() else ""
    if "COSTKEY_DSN=" in current:
        print("[ok] COSTKEY_DSN already exists in .env")
        return
    prefix = "" if not current or current.endswith("\n") else "\n"
    with env.open("a") as f:
        f.write(f"{prefix}# CostKey\nCOSTKEY_DSN={dsn}\n")
    print("[ok] Added COSTKEY_DSN to .env")


def _write_guide(detection: dict[str, object], dsn: str) -> None:
    Path(".costkey").mkdir(exist_ok=True)
    if detection["language"] == "ts":
        snippet = f'import {{ CostKey }} from "costkey"\nCostKey.init({{ dsn: process.env.COSTKEY_DSN ?? "{dsn}" }})'
        fence = "ts"
    else:
        snippet = f'import costkey\ncostkey.init(dsn="{dsn}")'
        fence = "python"
    warning = ""
    if detection["needs_sourcemaps"]:
        warning = "\n\nIMPORTANT: This project appears to be bundled/minified. Configure sourcemaps so production stack traces resolve to original source: https://costkey.dev/docs/sourcemaps\n"
    detected_file = f"\nLikely entrypoint: `{detection['entry_file']}`\n" if detection.get("entry_file") else "\nNo safe entrypoint was detected. Use the sidecar/preload option or add the snippet manually.\n"
    Path(".costkey/setup.md").write_text(
        f"# CostKey setup\n\nDetected: {detection['label']}\n{detected_file}\n{detection['entrypoint']}\n\n## Code snippet\n\n```{fence}\n{snippet}\n```\n\n## No-code sidecar/preload\n\n```bash\n{detection['sidecar']}\n```{warning}\n"
    )
    if detection["language"] == "py":
        Path(".costkey/sitecustomize.py").write_text(
            "import os\nimport costkey\n\n"
            "if os.environ.get('COSTKEY_DSN'):\n"
            "    costkey.init(\n"
            "        dsn=os.environ.get('COSTKEY_DSN'),\n"
            "        release=os.environ.get('COSTKEY_RELEASE'),\n"
            "        capture_body=os.environ.get('COSTKEY_CAPTURE_BODY', 'true').lower() != 'false',\n"
            "        debug=os.environ.get('COSTKEY_DEBUG') == 'true',\n"
            "    )\n"
        )
    elif detection["language"] == "ts":
        Path(".costkey/node-options.env").write_text("NODE_OPTIONS=--import costkey/register\n")
    print("[ok] Wrote .costkey/setup.md")


def setup(args: list[str]) -> None:
    host = _host(args)
    dsn = _arg(args, "--dsn") or os.environ.get("COSTKEY_DSN")
    if not dsn:
        token = _login(host, args)
        dsn = _create_project(host, token, args)
    detection = _detect()
    _ensure_env(dsn)
    _write_guide(detection, dsn)

    print(f"\nDetected: {detection['label']}")
    if detection.get("entry_file"):
        print(f"Likely entrypoint: {detection['entry_file']}")
    print(detection["entrypoint"])
    print("\nAdd this before your first AI call:\n")
    if detection["language"] == "ts":
        print('  import { CostKey } from "costkey"')
        print(f'  CostKey.init({{ dsn: process.env.COSTKEY_DSN ?? "{dsn}" }})')
    else:
        print("  import costkey")
        print(f'  costkey.init(dsn="{dsn}")')
    if detection["needs_sourcemaps"]:
        print("\n\033[33mWarning: sourcemaps recommended\033[0m")
        print("  This looks like a bundled/minified JS app. Configure sourcemaps:")
        print("  https://costkey.dev/docs/sourcemaps")
    print("No-code sidecar/preload option:")
    print(f"  {detection['sidecar']}")
    print("\nDone. Run your app and open the CostKey dashboard to watch events land.\n")


def main() -> None:
    import sys

    args = sys.argv[1:]
    command = args[0] if args else "help"
    if command == "setup":
        setup(args)
        return
    print(
        """costkey

Usage:
  costkey setup [--name <project-name>] [--dsn <dsn>]

Examples:
  pipx run costkey setup
  pipx run costkey setup --name my-app
  pipx run costkey setup --dsn https://ck_key@app.costkey.dev/project_id
"""
    )


if __name__ == "__main__":
    main()
