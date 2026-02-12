
// --- Draft History Logic ---
function renderDraftHistoryItems(list, items) {
    if (!list) return;
    if (!Array.isArray(items) || items.length === 0) {
        list.innerHTML = '<div style="font-size: 12px; color: #95a5a6; padding: 10px;">Keine Entwürfe gefunden.</div>';
        return;
    }

    list.innerHTML = items.map(draft => `
        <div class="draft-item" style="padding: 10px; border-bottom: 1px solid #eee; cursor: pointer;" onclick="loadDraft('${draft.id}')">
            <div style="font-weight: 600; font-size: 13px; color: #34495e;">${draft.document_type || 'Unbekannt'}</div>
            <div style="font-size: 11px; color: #7f8c8d; margin-bottom: 4px;">
                ${new Date(draft.created_at).toLocaleString()} • ${draft.model_used || 'N/A'}
            </div>
            <div style="font-size: 12px; color: #95a5a6; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                ${draft.user_prompt || 'Kein Prompt'}
            </div>
        </div>
    `).join('');
}

async function loadDrafts(options) {
    const opts = options || {};
    const silent = !!opts.silent;
    const expectedCaseId = opts.expectedCaseId || (typeof getActiveCaseIdLocal === 'function' ? getActiveCaseIdLocal() : null);
    const container = document.getElementById('draftHistoryContainer');
    const list = document.getElementById('draftHistoryList');
    if (!list) return [];

    if (!silent) {
        list.innerHTML = '<div style="font-size: 12px; color: #95a5a6; padding: 10px;">Laden...</div>';
    }

    try {
        const token = getAuthToken();
        const headers = {};
        if (token) headers['Authorization'] = `Bearer ${token}`;
        const params = new URLSearchParams();
        params.set('_ts', String(Date.now()));
        if (expectedCaseId) {
            params.set('_case', String(expectedCaseId));
        }
        const response = await fetch(`/drafts/?${params.toString()}`, {
            headers,
            cache: 'no-store',
        });
        if (!response.ok) throw new Error('Fehler beim Laden');

        const data = await response.json();
        const items = Array.isArray(data.items) ? data.items : [];

        const currentCaseId = typeof getActiveCaseIdLocal === 'function' ? getActiveCaseIdLocal() : null;
        if (expectedCaseId && currentCaseId && String(expectedCaseId) !== String(currentCaseId)) {
            return [];
        }

        renderDraftHistoryItems(list, items);
        return items;
    } catch (error) {
        console.error('Failed to load drafts:', error);
        if (!silent) {
            list.innerHTML = '<div style="font-size: 12px; color: #e74c3c; padding: 10px;">Fehler beim Laden der Entwürfe.</div>';
        }
        throw error;
    }
}

async function toggleDraftHistory() {
    const container = document.getElementById('draftHistoryContainer');
    if (!container) return;

    if (container.style.display === 'none') {
        container.style.display = 'block';
        try {
            await loadDrafts();
        } catch (error) {
            // loadDrafts already wrote a user-facing message into the list.
        }
    } else {
        container.style.display = 'none';
    }
}

async function loadDraft(draftId) {
    try {
        const token = getAuthToken();
        const headers = {};
        if (token) headers['Authorization'] = `Bearer ${token}`;

        const response = await fetch(`/drafts/${draftId}`, { headers });
        if (!response.ok) throw new Error('Fehler beim Laden des Entwurfs');

        const draft = await response.json();

        // Populate inputs
        const typeSelect = document.getElementById('documentTypeSelect');
        if (typeSelect) typeSelect.value = draft.document_type;

        const instructions = document.getElementById('draftInstructions');
        if (instructions) instructions.value = draft.user_prompt;

        const modelSelect = document.getElementById('modelSelect');
        if (modelSelect && draft.model_used) modelSelect.value = draft.model_used;

        // Display result
        displayDraft(draft);

        // Hide history
        document.getElementById('draftHistoryContainer').style.display = 'none';

    } catch (error) {
        console.error('Failed to restore draft:', error);
        alert('Entwurf konnte nicht geladen werden.');
    }
}
