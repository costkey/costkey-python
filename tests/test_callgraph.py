"""Tests for costkey.callgraph — static call graph analysis."""
from __future__ import annotations

import json
import os
import tempfile
import textwrap

import pytest

from costkey.callgraph import (
    build_call_graph,
    compute_scores,
    discover_files,
    _MAX_PAYLOAD_BYTES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_py(directory: str, relpath: str, content: str) -> str:
    """Write a .py file inside directory, creating subdirs as needed."""
    full = os.path.join(directory, relpath)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(textwrap.dedent(content))
    return full


# ---------------------------------------------------------------------------
# discover_files
# ---------------------------------------------------------------------------

class TestDiscoverFiles:
    def test_discover_files(self, tmp_path):
        """Finds .py files and skips venv directories."""
        (tmp_path / "app.py").write_text("x = 1")
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "utils.py").write_text("y = 2")
        venv = tmp_path / "venv"
        venv.mkdir()
        (venv / "bad.py").write_text("z = 3")
        dotenv = tmp_path / ".venv"
        dotenv.mkdir()
        (dotenv / "bad2.py").write_text("w = 4")

        files = discover_files(str(tmp_path))
        basenames = {os.path.basename(f) for f in files}
        assert "app.py" in basenames
        assert "utils.py" in basenames
        assert "bad.py" not in basenames
        assert "bad2.py" not in basenames

    def test_max_files_cap(self, tmp_path):
        """Caps at 500 files."""
        for i in range(600):
            (tmp_path / f"f{i}.py").write_text(f"x = {i}")

        files = discover_files(str(tmp_path))
        assert len(files) == 500

    def test_skips_egg_info(self, tmp_path):
        """Skips .egg-info directories."""
        egg = tmp_path / "mypkg.egg-info"
        egg.mkdir()
        (egg / "top.py").write_text("x = 1")
        (tmp_path / "real.py").write_text("y = 1")

        files = discover_files(str(tmp_path))
        basenames = {os.path.basename(f) for f in files}
        assert "real.py" in basenames
        assert "top.py" not in basenames


# ---------------------------------------------------------------------------
# build_call_graph
# ---------------------------------------------------------------------------

class TestBuildCallGraph:
    def test_simple(self, tmp_path):
        """Parse a file with 3 functions, one calling openai."""
        _write_py(str(tmp_path), "app.py", """\
            import openai

            def get_client():
                return openai.OpenAI()

            def call_ai(prompt):
                client = get_client()
                return client.chat.completions.create(model="gpt-4", messages=[{"role":"user","content":prompt}])

            def business_logic():
                result = call_ai("hello")
                return result
        """)

        graph = build_call_graph([str(tmp_path / "app.py")], str(tmp_path))

        assert "app.py:get_client" in graph
        assert "app.py:call_ai" in graph
        assert "app.py:business_logic" in graph

        # call_ai calls openai pattern
        assert graph["app.py:call_ai"]["calls_ai"] is True
        # business_logic calls call_ai
        assert "call_ai" in graph["app.py:business_logic"]["callees"]
        # get_client does not call AI
        assert graph["app.py:get_client"]["calls_ai"] is False

    def test_anthropic(self, tmp_path):
        """Detect client.messages.create pattern."""
        _write_py(str(tmp_path), "svc.py", """\
            import anthropic

            def ask_claude(prompt):
                client = anthropic.Anthropic()
                return client.messages.create(model="claude-3", messages=[{"role":"user","content":prompt}])
        """)

        graph = build_call_graph([str(tmp_path / "svc.py")], str(tmp_path))
        assert graph["svc.py:ask_claude"]["calls_ai"] is True

    def test_class_methods(self, tmp_path):
        """Detect class methods as ClassName.method."""
        _write_py(str(tmp_path), "service.py", """\
            import openai

            class MyService:
                def predict(self):
                    client = openai.OpenAI()
                    return client.chat.completions.create(model="gpt-4", messages=[])

                def helper(self):
                    return self.predict()
        """)

        graph = build_call_graph([str(tmp_path / "service.py")], str(tmp_path))
        assert "service.py:MyService.predict" in graph
        assert "service.py:MyService.helper" in graph
        assert graph["service.py:MyService.predict"]["calls_ai"] is True

    def test_syntax_error_skipped(self, tmp_path):
        """Files with syntax errors are silently skipped."""
        bad = tmp_path / "bad.py"
        bad.write_text("def broken(:\n    pass")
        good = tmp_path / "good.py"
        good.write_text("def ok():\n    pass\n")

        graph = build_call_graph(
            [str(bad), str(good)], str(tmp_path)
        )
        assert "good.py:ok" in graph
        # No entries from the bad file
        assert not any(k.startswith("bad.py:") for k in graph)

    def test_google_pattern(self, tmp_path):
        """Detect generate_content pattern for Google."""
        _write_py(str(tmp_path), "gen.py", """\
            import google.generativeai as genai

            def ask_gemini(prompt):
                model = genai.GenerativeModel("gemini-pro")
                return model.generate_content(prompt)
        """)

        graph = build_call_graph([str(tmp_path / "gen.py")], str(tmp_path))
        assert graph["gen.py:ask_gemini"]["calls_ai"] is True


# ---------------------------------------------------------------------------
# compute_scores
# ---------------------------------------------------------------------------

class TestComputeScores:
    def test_scoring(self):
        """Wrapper gets high score, business logic gets low score."""
        graph = {
            "app.py:call_openai": {
                "calls_ai": True,
                "callees": [],
                "lineno": 10,
            },
            "app.py:wrapper": {
                "calls_ai": False,
                "callees": ["call_openai"],
                "lineno": 20,
            },
            "app.py:handler": {
                "calls_ai": False,
                "callees": ["wrapper"],
                "lineno": 30,
            },
        }
        scored = compute_scores(graph)

        assert scored["app.py:call_openai"]["calls_ai"] is True
        assert scored["app.py:call_openai"]["ai_distance"] == 0

        # wrapper is 1 hop away (calls call_openai)
        assert scored["app.py:wrapper"]["ai_distance"] == 1
        # handler is 2 hops away
        assert scored["app.py:handler"]["ai_distance"] == 2

        # Business logic (handler) should have lower score than the AI wrapper
        assert scored["app.py:handler"]["score"] < scored["app.py:call_openai"]["score"]

    def test_ai_distance(self):
        """Verify BFS distance computation through a chain."""
        graph = {
            "a.py:direct": {"calls_ai": True, "callees": [], "lineno": 1},
            "a.py:one_hop": {"calls_ai": False, "callees": ["direct"], "lineno": 2},
            "a.py:two_hop": {"calls_ai": False, "callees": ["one_hop"], "lineno": 3},
            "a.py:three_hop": {"calls_ai": False, "callees": ["two_hop"], "lineno": 4},
            "a.py:four_hop": {"calls_ai": False, "callees": ["three_hop"], "lineno": 5},
        }
        scored = compute_scores(graph)

        assert scored["a.py:direct"]["ai_distance"] == 0
        assert scored["a.py:one_hop"]["ai_distance"] == 1
        assert scored["a.py:two_hop"]["ai_distance"] == 2
        assert scored["a.py:three_hop"]["ai_distance"] == 3
        # four_hop is 4 hops away, excluded (> 3)
        assert "a.py:four_hop" not in scored

    def test_fan_in(self):
        """Fan-in counts correctly for popular functions."""
        graph = {
            "x.py:shared": {"calls_ai": True, "callees": [], "lineno": 1},
            "x.py:a": {"calls_ai": False, "callees": ["shared"], "lineno": 2},
            "x.py:b": {"calls_ai": False, "callees": ["shared"], "lineno": 3},
            "x.py:c": {"calls_ai": False, "callees": ["shared"], "lineno": 4},
        }
        scored = compute_scores(graph)
        assert scored["x.py:shared"]["fan_in"] == 3

    def test_irrelevant_functions_excluded(self):
        """Functions with no path to AI calls are excluded."""
        graph = {
            "a.py:ai_func": {"calls_ai": True, "callees": [], "lineno": 1},
            "b.py:unrelated": {"calls_ai": False, "callees": [], "lineno": 1},
        }
        scored = compute_scores(graph)
        assert "a.py:ai_func" in scored
        assert "b.py:unrelated" not in scored


# ---------------------------------------------------------------------------
# Payload size limit
# ---------------------------------------------------------------------------

class TestPayloadSize:
    def test_payload_size_limit(self, tmp_path):
        """Large graphs get trimmed to fit 50KB."""
        # Build a graph with many functions, each calling AI
        graph: dict = {}
        for i in range(2000):
            graph[f"big/module_{i}.py:func_{i}"] = {
                "calls_ai": True,
                "callees": [],
                "lineno": i,
            }

        scored = compute_scores(graph)

        # Build the payload manually to verify trimming logic
        functions = {}
        for key, info in scored.items():
            functions[key] = {
                "line": info["line"],
                "score": info["score"],
                "fan_in": info["fan_in"],
                "calls_ai": info["calls_ai"],
            }

        payload = {"sdkVersion": "python-0.2.3", "functions": functions}
        payload_json = json.dumps(payload)

        # Without trimming it exceeds 50KB
        assert len(payload_json.encode("utf-8")) > _MAX_PAYLOAD_BYTES

        # Apply trimming logic (same as scan_and_send)
        sorted_keys = sorted(
            functions.keys(),
            key=lambda k: functions[k]["score"],
            reverse=True,
        )
        while len(payload_json.encode("utf-8")) > _MAX_PAYLOAD_BYTES and sorted_keys:
            remove_key = sorted_keys.pop()
            del functions[remove_key]
            payload["functions"] = functions
            payload_json = json.dumps(payload)

        assert len(payload_json.encode("utf-8")) <= _MAX_PAYLOAD_BYTES
        assert len(functions) > 0  # Some functions remain
