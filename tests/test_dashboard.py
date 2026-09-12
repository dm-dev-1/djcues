"""Tests for djcues.dashboard -- the analysis dashboard page.

render_dashboard_html() is this module's entire Python-callable surface,
same shape as auth_web.py's render_auth_setup_html(): a (mostly) pure
function returning a static HTML/CSS/JS string. All of the dashboard's
actual interactive behavior (the playlist tree, job polling, results
rendering) lives in the embedded JavaScript, which only executes in a
real browser -- not more Python assertions to write there, matching
test_auth_web.py's own established precedent for exactly this situation.
What's tested here is what's real and checkable from Python: the page
renders valid, complete, deterministic markup, upholds the same-origin/
no-embedded-server-URL design (matching auth_web.py's pattern, not
review.py's), and safely embeds the one real piece of dynamic input it
takes (initial_playlist).
"""

from __future__ import annotations

from djcues.dashboard import render_dashboard_html


class TestRenderDashboardHtml:
    def test_returns_a_complete_html_document(self):
        result = render_dashboard_html()
        assert result.startswith("<!DOCTYPE html>")
        assert result.rstrip().endswith("</html>")
        assert "<title>djcues &mdash; Analysis dashboard</title>" in result

    def test_css_and_js_are_actually_embedded(self):
        result = render_dashboard_html()
        assert "<style>" in result and "</style>" in result
        assert "<script>" in result and "</script>" in result
        # Spot-check real content from each, not just the tags -- including
        # viz.py's own CSS being genuinely reused, not just imported and
        # discarded.
        assert ".dashboard" in result  # dashboard-specific CSS
        assert ".timeline-section" in result  # reused from viz._PAGE_CSS
        assert "loadPlaylists()" in result and "'/api/playlists'" in result  # JS (via the fetchJson() helper, not a bare fetch(...) call)

    def test_analysis_presets_and_estimate_display_are_present(self):
        """The preset picker (named shortcuts onto the same agentic/
        refine-drops/deep checkboxes) and its cost/time estimate display,
        backed by GET /api/estimate -- see analysis_cache.estimate()."""
        result = render_dashboard_html()
        for label in ("Quick", "Refine Drops", "Deep Analysis", "Agentic",
                      "Full Agentic", "Full Agentic + Deep", "Quick Check", "Verify Audio"):
            assert f">{label}<" in result
        assert 'id="estimate-display"' in result
        assert "function refreshEstimate" in result
        assert "'/api/estimate?'" in result

    def test_device_selector_is_present(self):
        """Reads/writes ~/.djcues/config.json via GET/POST /api/devices --
        the same config djcues.auth's CLI commands use, not a separate
        dashboard-only setting."""
        result = render_dashboard_html()
        assert 'id="flag-device"' in result
        for value in ("auto", "cpu", "cuda", "directml"):
            assert f'value="{value}"' in result
        assert 'id="device-status"' in result
        assert "function loadDevices" in result
        assert "'/api/devices'" in result

    def test_never_embeds_a_server_url(self):
        # Same-origin like auth_web.py's page (served BY the local server
        # it talks to), unlike review.py's file://-opened page, which
        # needs one. No absolute URL should ever be constructed here.
        result = render_dashboard_html()
        assert "http://" not in result
        assert "https://" not in result

    def test_fetches_are_same_origin_relative_paths(self):
        result = render_dashboard_html()
        for path in ("/api/playlists", "/api/tracks/", "/api/jobs/"):
            assert path in result

    def test_deterministic_pure_output_with_no_args(self):
        assert render_dashboard_html() == render_dashboard_html()

    def test_default_offset_and_loop_bars_appear_in_the_form(self):
        result = render_dashboard_html(default_offset=8, default_loop_bars=2)
        assert 'id="flag-offset" value="8"' in result
        assert 'id="flag-loop-bars" value="2"' in result

    def test_no_initial_playlist_by_default(self):
        result = render_dashboard_html()
        assert "const INITIAL_PLAYLIST = null;" in result

    def test_initial_playlist_is_auto_selected(self):
        result = render_dashboard_html(initial_playlist={"id": "723183862", "name": "Tech House"})
        assert 'const INITIAL_PLAYLIST = {"id": "723183862", "name": "Tech House"};' in result

    def test_initial_playlist_defined_before_init_is_called(self):
        # A real ordering bug: INITIAL_PLAYLIST must be assigned before
        # initDashboard() (which reads it) runs, or the auto-select
        # silently sees `undefined` instead of the real value.
        result = render_dashboard_html(initial_playlist={"id": "1", "name": "X"})
        assert result.index("INITIAL_PLAYLIST = ") < result.index("initDashboard();")

    def test_initial_playlist_name_is_safely_json_escaped(self):
        # A real regression guard: a playlist name containing a quote
        # must not be able to break out of the embedded JS string literal
        # -- json.dumps() (not raw f-string interpolation) is what's
        # supposed to guarantee this.
        tricky_name = 'Set"; alert(document.cookie); "'
        result = render_dashboard_html(initial_playlist={"id": "1", "name": tricky_name})
        # If the quote had broken out of the string literal, this would
        # appear as bare, executable JS rather than as an escaped string
        # value -- it must not.
        assert 'alert(document.cookie); "' not in result
        assert 'Set\\"; alert(document.cookie); \\"' in result  # safely escaped instead

    def test_initial_playlist_name_cannot_close_the_script_tag(self):
        # A playlist name containing a literal "</script>" must not be
        # able to end the embedded <script> block early -- only the
        # *closing*-tag count matters for this (the HTML tokenizer, once
        # inside script-data state, only watches for "</script"; a bare
        # "<script>"-shaped substring appearing again as inert text
        # inside that state re-opens nothing, so it's not itself unsafe
        # and isn't asserted on here).
        tricky_name = "</script><script>alert(1)</script>"
        result = render_dashboard_html(initial_playlist={"id": "1", "name": tricky_name})
        baseline = render_dashboard_html()
        assert result.count("</script>") == baseline.count("</script>")
