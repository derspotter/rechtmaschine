#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p tmp/playwright

if ! command -v playwright-cli >/dev/null 2>&1; then
  echo "playwright-cli not found in PATH" >&2
  exit 1
fi

# Ensure the UI is reachable (docker-compose.yml publishes 8000:8000)
if ! curl -fsS -o /dev/null http://127.0.0.1:8000/; then
  echo "UI not reachable at http://127.0.0.1:8000/ (is docker compose up?)" >&2
  exit 1
fi

TOKEN="$(docker exec rechtmaschine-app bash -lc "python3 - <<'PY'
from auth import create_access_token
print(create_access_token({'sub': 'jay'}))
PY")"

# Ensure we start from a clean default session.
playwright-cli session-stop-all >/dev/null 2>&1 || true
playwright-cli session-delete >/dev/null 2>&1 || true

# Use an isolated session profile to avoid "Browser is already in use" conflicts.
playwright-cli config --isolated --browser=chrome >/dev/null

playwright-cli open http://127.0.0.1:8000/ >/dev/null
playwright-cli resize 1280 800 >/dev/null

playwright-cli screenshot --filename tmp/playwright/01-login-overlay.png --full-page >/dev/null

playwright-cli localstorage-set rechtmaschine_auth_token "${TOKEN}" >/dev/null
playwright-cli reload >/dev/null

playwright-cli run-code "async page => {
  // Wait for cases to load and the login overlay to disappear.
  await page.waitForTimeout(600);
  await page.waitForSelector('#caseSelect option', { timeout: 10_000 });
}" >/dev/null

playwright-cli screenshot --filename tmp/playwright/02-dashboard.png --full-page >/dev/null

playwright-cli run-code "async page => {
  // Create a new case via the UI (confirm + prompt).
  page.on('dialog', async d => {
    try {
      if (d.type() === 'prompt') {
        await d.accept('PW Fall A');
      } else {
        await d.accept();
      }
    } catch (e) {
      // ignore
    }
  });

  await page.getByRole('button', { name: /Neuer Fall/ }).click();
  await page.waitForTimeout(1500);

  const ids = ['anhoerung-docs','bescheid-docs','vorinstanz-docs','rechtsprechung-docs','akte-docs','sonstiges-docs'];
  const counts = {};
  let total = 0;
  for (const id of ids) {
    const el = page.locator('#' + id);
    const c = await el.locator('.doc-card, .document-card, .doc-item').count();
    counts[id] = c;
    total += c;
  }
  if (total !== 0) {
    throw new Error('expected new case to be empty, got total=' + total + ' counts=' + JSON.stringify(counts));
  }
}" >/dev/null

playwright-cli screenshot --filename tmp/playwright/03-new-case-empty.png --full-page >/dev/null

playwright-cli run-code "async page => {
  // Switch back to the original case by label.
  await page.locator('#caseSelect').selectOption({ label: 'Fall 1' });
  await page.waitForTimeout(1500);
}" >/dev/null

playwright-cli screenshot --filename tmp/playwright/04-back-to-fall1.png --full-page >/dev/null

playwright-cli run-code "async page => {
  // Select the first Bescheid checkbox.
  const cb = page.locator('#bescheid-docs input[type=checkbox]').first();
  await cb.check();
  await page.waitForTimeout(1200); // allow case-state autosave debounce
}" >/dev/null

playwright-cli screenshot --filename tmp/playwright/05-bescheid-selected.png --full-page >/dev/null

playwright-cli run-code "async page => {
  // Switch away and back; selection must persist.
  await page.locator('#caseSelect').selectOption({ label: 'PW Fall A' });
  await page.waitForTimeout(1500);
  await page.locator('#caseSelect').selectOption({ label: 'Fall 1' });
  await page.waitForTimeout(1500);

  const cb = page.locator('#bescheid-docs input[type=checkbox]').first();
  const checked = await cb.isChecked();
  if (!checked) throw new Error('expected bescheid checkbox to stay checked after switching cases');
}" >/dev/null

playwright-cli screenshot --filename tmp/playwright/06-switch-back-still-checked.png --full-page >/dev/null

echo "OK: screenshots in tmp/playwright/"
