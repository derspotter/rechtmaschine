"""Playwright-Render-Fallback fürs Verify-Fetching (Addendum 2026-07-14).

httpx bleibt der Primärpfad (94% der Quellen brauchen kein Rendering);
dieser Fallback bekommt nur Seiten, die dort als blocked/not_decision
enden. EIN lazy gestarteter Chromium pro Verify-Batch, kleine Semaphore,
hartes Seiten-Timeout. Jede Exception schlägt nach oben durch und wird in
fetch_source zur Notiz — der ehrliche Original-Status bleibt bestehen.
"""
import asyncio
import os

RENDER_PAGE_TIMEOUT_MS = int(os.getenv("RESEARCH_RENDER_TIMEOUT_MS", "30000") or "30000")


def render_fallback_enabled() -> bool:
    """Flag RESEARCH_RENDER_FALLBACK, default an."""
    return os.getenv("RESEARCH_RENDER_FALLBACK", "true").strip().lower() in {"1", "true", "yes", "on"}


class RenderSession:
    """Lazy Chromium, direkt als ``render_fn`` in fetch_source injizierbar.

    Der Browser startet erst beim ersten tatsächlichen Render-Bedarf —
    ein Verify-Batch ohne Walls zahlt keinerlei Browser-Overhead.
    Nach dem Batch ``aclose()`` aufrufen (idempotent).
    """

    def __init__(self, max_concurrency: int = 2):
        self._sem = asyncio.Semaphore(max_concurrency)
        self._lock = asyncio.Lock()
        self._pw = None
        self._browser = None

    @property
    def launched(self) -> bool:
        return self._browser is not None

    async def _ensure_browser(self):
        async with self._lock:
            if self._browser is None:
                from playwright.async_api import async_playwright

                self._pw = await async_playwright().start()
                self._browser = await self._pw.chromium.launch(headless=True)
        return self._browser

    async def __call__(self, url: str) -> str:
        browser = await self._ensure_browser()
        async with self._sem:
            page = await browser.new_page()
            try:
                await page.goto(url, wait_until="networkidle", timeout=RENDER_PAGE_TIMEOUT_MS)
                return await page.content()
            finally:
                await page.close()

    async def aclose(self) -> None:
        try:
            if self._browser is not None:
                await self._browser.close()
        except Exception:  # noqa: BLE001 — Cleanup darf nie werfen
            pass
        finally:
            self._browser = None
            try:
                if self._pw is not None:
                    await self._pw.stop()
            except Exception:  # noqa: BLE001
                pass
            finally:
                self._pw = None
