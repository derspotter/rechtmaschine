"""RenderSession-Eigenschaften ohne Browser-Launch (der echte Chromium-Lauf
ist Live-Akzeptanz im Container — auf dem Host-ARM ist Playwright-Launch
nicht verlässlich, siehe Memory playwright-server-cli-not-mcp).

Run: .venv/bin/python -m pytest tests/test_render_session.py -q
"""
import asyncio

from endpoints.research.render import RenderSession, render_fallback_enabled


def test_render_flag_default_on(monkeypatch):
    monkeypatch.delenv("RESEARCH_RENDER_FALLBACK", raising=False)
    assert render_fallback_enabled() is True
    monkeypatch.setenv("RESEARCH_RENDER_FALLBACK", "false")
    assert render_fallback_enabled() is False


def test_render_session_is_lazy_and_close_is_idempotent():
    session = RenderSession()
    assert session.launched is False
    # aclose ohne Launch wirft nie und startet nichts.
    asyncio.run(session.aclose())
    asyncio.run(session.aclose())
    assert session.launched is False
