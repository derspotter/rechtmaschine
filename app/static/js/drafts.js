
// --- Draft History Logic ---
let draftHistoryFetchController = null;
let draftHistoryLoadRequestId = 0;

function _emptySelectedDocumentsPayload() {
    return {
        anhoerung: [],
        bescheid: { primary: null, others: [] },
        vorinstanz: { primary: null, others: [] },
        rechtsprechung: [],
        akte: [],
        sonstiges: [],
        saved_sources: [],
    };
}

function _pushUnique(target, value) {
    if (!value) return;
    if (!Array.isArray(target)) return;
    if (!target.includes(value)) target.push(value);
}

function _rebuildSelectedDocumentsFromUsed(usedDocuments) {
    const selected = _emptySelectedDocumentsPayload();
    if (!Array.isArray(usedDocuments)) {
        return selected;
    }

    for (const entry of usedDocuments) {
        if (!entry || typeof entry !== 'object') continue;
        const category = String(entry.category || '').toLowerCase();
        const filename = (entry.filename || entry.title || '').trim();
        const role = String(entry.role || '').toLowerCase();
        if (!filename) continue;

        switch (category) {
            case 'bescheid':
                if (role === 'primary' && !selected.bescheid.primary) {
                    selected.bescheid.primary = filename;
                } else {
                    _pushUnique(selected.bescheid.others, filename);
                }
                break;
            case 'vorinstanz':
                if (role === 'primary' && !selected.vorinstanz.primary) {
                    selected.vorinstanz.primary = filename;
                } else {
                    _pushUnique(selected.vorinstanz.others, filename);
                }
                break;
            case 'anhoerung':
            case 'rechtsprechung':
            case 'akte':
            case 'sonstiges':
            case 'saved_sources':
                _pushUnique(selected[category], filename);
                break;
            default:
                break;
        }
    }

    if (
        !selected.bescheid.primary &&
        Array.isArray(selected.bescheid.others) &&
        selected.bescheid.others.length > 0
    ) {
        selected.bescheid.primary = selected.bescheid.others.shift();
    }

    return selected;
}

function _ensureDraftRequestPayload(draft) {
    if (!draft || typeof draft !== 'object') {
        return null;
    }

    if (draft._requestPayload && typeof draft._requestPayload === 'object') {
        return draft._requestPayload;
    }

    if (draft.request_payload && typeof draft.request_payload === 'object') {
        draft._requestPayload = draft.request_payload;
        return draft._requestPayload;
    }

    const metadata = draft.metadata && typeof draft.metadata === 'object' ? draft.metadata : {};
    const usedDocuments = Array.isArray(metadata.used_documents) ? metadata.used_documents : [];
    const selectedDocuments = _rebuildSelectedDocumentsFromUsed(usedDocuments);

    const modelSelect = document.getElementById('modelSelect');
    const verbositySelect = document.getElementById('verbositySelect');
    const fallbackModel = modelSelect ? modelSelect.value : 'claude-opus-4-6';
    const fallbackVerbosity = verbositySelect ? verbositySelect.value : 'high';
    const fallbackLegalArea = typeof getLegalArea === 'function' ? getLegalArea() : 'migrationsrecht';

    const userPrompt = (draft.user_prompt || '').trim();
    const generatedText = (draft.generated_text || '').trim();
    const chatHistory = [];
    if (userPrompt) {
        chatHistory.push({ role: 'user', content: userPrompt });
    }
    if (generatedText) {
        chatHistory.push({ role: 'assistant', content: generatedText });
    }

    draft._requestPayload = {
        document_type: draft.document_type || 'Klagebegründung',
        user_prompt: userPrompt,
        legal_area: fallbackLegalArea || 'migrationsrecht',
        selected_documents: selectedDocuments,
        model: draft.model_used || fallbackModel,
        verbosity: fallbackVerbosity,
        chat_history: chatHistory,
    };

    return draft._requestPayload;
}

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
    const list = document.getElementById('draftHistoryList');
    if (!list) return [];

    if (!silent) {
        list.innerHTML = '<div style="font-size: 12px; color: #95a5a6; padding: 10px;">Laden...</div>';
    }

    let controller = null;
    try {
        const token = getAuthToken();
        const headers = {};
        if (token) headers['Authorization'] = `Bearer ${token}`;
        const params = new URLSearchParams();
        params.set('_ts', String(Date.now()));
        if (expectedCaseId) {
            params.set('_case', String(expectedCaseId));
        }

        if (draftHistoryFetchController) {
            draftHistoryFetchController.abort();
        }
        const requestId = ++draftHistoryLoadRequestId;
        controller = new AbortController();
        draftHistoryFetchController = controller;

        const response = await fetch(`/drafts/?${params.toString()}`, {
            headers,
            cache: 'no-store',
            signal: controller.signal,
        });

        // Ignore stale responses that arrive after a newer request has started.
        if (requestId !== draftHistoryLoadRequestId) {
            return [];
        }
        if (!response.ok) throw new Error('Fehler beim Laden');

        const data = await response.json();
        const items = Array.isArray(data.items) ? data.items : [];

        const currentCaseId = typeof getActiveCaseIdLocal === 'function' ? getActiveCaseIdLocal() : null;
        if (expectedCaseId && currentCaseId && String(expectedCaseId) !== String(currentCaseId)) {
            if (!silent) {
                renderDraftHistoryItems(list, []);
            }
            return [];
        }

        renderDraftHistoryItems(list, items);
        return items;
    } catch (error) {
        if (error && error.name === 'AbortError') {
            return [];
        }
        console.error('Failed to load drafts:', error);
        if (!silent) {
            list.innerHTML = '<div style="font-size: 12px; color: #e74c3c; padding: 10px;">Fehler beim Laden der Entwürfe.</div>';
        }
        throw error;
    } finally {
        if (draftHistoryFetchController === controller) {
            draftHistoryFetchController = null;
        }
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
        if (draftHistoryFetchController) {
            draftHistoryFetchController.abort();
        }
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

        _ensureDraftRequestPayload(draft);

        // Display result
        displayDraft(draft);

    } catch (error) {
        console.error('Failed to restore draft:', error);
        alert('Entwurf konnte nicht geladen werden.');
    }
}
