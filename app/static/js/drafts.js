
// --- Draft History Logic ---
async function toggleDraftHistory() {
    const container = document.getElementById('draftHistoryContainer');
    const list = document.getElementById('draftHistoryList');

    if (container.style.display === 'none') {
        container.style.display = 'block';
        list.innerHTML = '<div style="font-size: 12px; color: #95a5a6; padding: 10px;">Laden...</div>';

        try {
            const token = getAuthToken();
            const headers = {};
            if (token) headers['Authorization'] = `Bearer ${token}`;

            const response = await fetch('/drafts/', { headers });
            if (!response.ok) throw new Error('Fehler beim Laden');

            const data = await response.json();
            if (!data.items || data.items.length === 0) {
                list.innerHTML = '<div style="font-size: 12px; color: #95a5a6; padding: 10px;">Keine Entwürfe gefunden.</div>';
                return;
            }

            list.innerHTML = data.items.map(draft => `
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

        } catch (error) {
            console.error('Failed to load drafts:', error);
            list.innerHTML = '<div style="font-size: 12px; color: #e74c3c; padding: 10px;">Fehler beim Laden der Entwürfe.</div>';
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
        displayDraft(draft.generated_text);

        // Hide history
        document.getElementById('draftHistoryContainer').style.display = 'none';

    } catch (error) {
        console.error('Failed to restore draft:', error);
        alert('Entwurf konnte nicht geladen werden.');
    }
}
