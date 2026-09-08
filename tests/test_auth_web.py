"""Tests for djcues.auth_web -- the browser-based BYOK setup wizard page.

render_auth_setup_html() is the module's entire Python-callable surface:
a zero-argument pure function returning a static HTML/CSS/JS string. All
of the wizard's actual behavior (fetching models, saving, error
handling) lives in the embedded JavaScript, which only executes in a
real browser -- there's no meaningful additional branching on the
Python side to unit-test the way cli.py/review.py/viz.py/server.py had.
What's tested here is what's real and checkable from Python: the page
renders valid, complete, deterministic markup, and it upholds the
same-origin/no-embedded-server-URL design the module's own docstring
promises (a real regression guard -- copying review.py's server-URL-
embedding pattern here by mistake would silently break that promise).
"""

from __future__ import annotations

from djcues.auth_web import render_auth_setup_html


class TestRenderAuthSetupHtml:
    def test_returns_a_complete_html_document(self):
        result = render_auth_setup_html()
        assert result.startswith("<!DOCTYPE html>")
        assert result.rstrip().endswith("</html>")
        assert "<title>djcues &mdash; Configure agentic analysis</title>" in result

    def test_css_and_js_are_actually_embedded(self):
        result = render_auth_setup_html()
        assert "<style>" in result and "</style>" in result
        assert "<script>" in result and "</script>" in result
        # Spot-check real content from each, not just the tags.
        assert ".btn-primary" in result  # CSS
        assert "fetch('/models'" in result  # JS

    def test_never_embeds_a_server_url(self):
        # The module's own docstring promises this explicitly: unlike
        # review.py's page (which needs a server URL baked in because it
        # can be opened from a file:// path), this page is only ever
        # served BY the local server it talks to, so every request is a
        # same-origin relative fetch -- no absolute URL is ever
        # constructed anywhere in the page. (The literal string
        # "127.0.0.1" does legitimately appear once, in the human-
        # readable description of where the key goes -- that's fine,
        # what matters is that it's never used to build a fetch target.)
        result = render_auth_setup_html()
        assert "http://" not in result
        assert "https://" not in result

    def test_fetches_are_same_origin_relative_paths(self):
        result = render_auth_setup_html()
        assert "fetch('/models'" in result
        assert "fetch('/save'" in result

    def test_no_hardcoded_api_key_placeholder_leaks_into_markup(self):
        # Cheap but real: nothing resembling a real provider key prefix
        # should ever appear in the static template itself.
        result = render_auth_setup_html()
        assert "sk-" not in result
        assert "AIza" not in result  # Google API key prefix

    def test_deterministic_pure_output(self):
        assert render_auth_setup_html() == render_auth_setup_html()
