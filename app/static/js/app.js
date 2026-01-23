const debugLog = (...args) => {
    const ts = new Date().toISOString();
    console.log(`[Rechtmaschine] ${ts}`, ...args);
};

const debugError = (...args) => {
    const ts = new Date().toISOString();
    console.error(`[Rechtmaschine] ${ts}`, ...args);
};

// --- Auth Logic ---
const AUTH_TOKEN_KEY = 'rechtmaschine_auth_token';
const LEGAL_AREA_KEY = 'rechtmaschine_legal_area';

function getAuthToken() {
    return localStorage.getItem(AUTH_TOKEN_KEY);
}

function setAuthToken(token) {
    localStorage.setItem(AUTH_TOKEN_KEY, token);
}

function clearAuthToken() {
    localStorage.removeItem(AUTH_TOKEN_KEY);
}

function getLegalArea() {
    const value = localStorage.getItem(LEGAL_AREA_KEY);
    return value === 'sozialrecht' ? 'sozialrecht' : 'migrationsrecht';
}

function setLegalArea(area) {
    localStorage.setItem(LEGAL_AREA_KEY, area);
}

function initLegalAreaToggle() {
    const toggle = document.getElementById('legalAreaToggle');
    const label = document.getElementById('legalAreaLabel');
    if (!toggle || !label) return;

    const current = getLegalArea();
    toggle.checked = current === 'sozialrecht';
    label.textContent = current === 'sozialrecht' ? 'Sozialrecht' : 'Migrationsrecht';

    toggle.addEventListener('change', () => {
        const area = toggle.checked ? 'sozialrecht' : 'migrationsrecht';
        setLegalArea(area);
        label.textContent = area === 'sozialrecht' ? 'Sozialrecht' : 'Migrationsrecht';
    });
}

initLegalAreaToggle();

function showLoginOverlay() {
    const overlay = document.getElementById('loginOverlay');
    if (overlay) overlay.style.display = 'flex';
}

function hideLoginOverlay() {
    const overlay = document.getElementById('loginOverlay');
    if (overlay) overlay.style.display = 'none';
}

async function handleLogin() {
    const emailInput = document.getElementById('loginEmail');
    const passwordInput = document.getElementById('loginPassword');
    const errorText = document.getElementById('loginError');

    const email = emailInput.value.trim();
    const password = passwordInput.value;

    if (!email || !password) {
        errorText.textContent = 'Bitte E-Mail und Passwort eingeben.';
        errorText.style.display = 'block';
        return;
    }

    try {
        const formData = new FormData();
        formData.append('username', email);
        formData.append('password', password);

        const response = await fetch('/token', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            setAuthToken(data.access_token);
            hideLoginOverlay();
            errorText.style.display = 'none';
            // Reload data
            loadDocuments();
            loadSources();
        } else {
            errorText.textContent = 'Login fehlgeschlagen. Bitte überprüfen Sie Ihre Daten.';
            errorText.style.display = 'block';
        }
    } catch (error) {
        console.error('Login error:', error);
        errorText.textContent = 'Ein Fehler ist aufgetreten.';
        errorText.style.display = 'block';
    }
}

// function to securely view/download documents
async function viewDocument(filename, event) {
    if (event) event.preventDefault();

    // Use existing helper
    const token = getAuthToken();
    if (!token) {
        showLoginOverlay();
        return;
    }

    try {
        const response = await fetch(`/documents/${filename}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (response.status === 401) {
            // Check if handleLogout exists, otherwise clear token and show login
            // Using clearAuthToken derived from file content
            clearAuthToken();
            showLoginOverlay();
            return;
        }

        if (!response.ok) {
            throw new Error(`Fehler beim Laden: ${response.statusText}`);
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);

        // Create hidden link to force download name
        const a = document.createElement('a');
        a.href = url;
        a.download = filename; // Use the filename from argument
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

        // Clean up URL after a delay
        setTimeout(() => window.URL.revokeObjectURL(url), 1000);

    } catch (error) {
        console.error('Download failed:', error);
        alert('Dokument konnte nicht geladen werden: ' + error.message);
    }
}

// Intercept fetch to add Authorization header
const originalFetch = window.fetch;
window.fetch = async function (url, options = {}) {
    // Skip auth for token endpoint
    if (url === '/token') {
        return originalFetch(url, options);
    }

    const token = getAuthToken();
    if (token) {
        options.headers = options.headers || {};
        // Handle Headers object or plain object
        if (options.headers instanceof Headers) {
            options.headers.set('Authorization', `Bearer ${token}`);
        } else {
            options.headers['Authorization'] = `Bearer ${token}`;
        }
    }

    const response = await originalFetch(url, options);

    if (response.status === 401) {
        clearAuthToken();
        showLoginOverlay();
    }

    return response;
};

// Check auth on load
window.addEventListener('DOMContentLoaded', () => {
    if (!getAuthToken()) {
        showLoginOverlay();
    } else {
        hideLoginOverlay();
    }
});
// --- End Auth Logic ---

const escapeJsString = (value) => String(value)
    .split('\\\\').join('\\\\\\\\')
    .split("'").join("\\\\'")
    .split('"').join('\\\\"');

const categoryToKey = {
    'Anhörung': 'anhoerung',
    'Bescheid': 'bescheid',
    'Vorinstanz': 'vorinstanz',
    'Rechtsprechung': 'rechtsprechung',
    'Akte': 'akte',
    'Sonstiges': 'sonstiges',
    'Sonstige gespeicherte Quellen': 'sonstiges'
};

const categoryToServerValue = {
    'Sonstiges': 'Sonstige gespeicherte Quellen'
};

const ANONYMIZABLE_CATEGORIES = new Set([
    'Anhörung',
    'Bescheid',
    'Sonstige gespeicherte Quellen',
    'Sonstiges'
]);

const selectionState = {
    anhoerung: new Set(),
    bescheid: {
        primary: null,
        others: new Set()
    },
    vorinstanz: {
        primary: null,
        others: new Set()
    },
    rechtsprechung: new Set(),
    sonstiges: new Set(),
    akte: new Set(),
    saved_sources: new Set()
};

let globalDocuments = {};
let globalSources = [];

const DOCUMENT_CONTAINER_IDS = {
    'Anhörung': 'anhoerung-docs',
    'Bescheid': 'bescheid-docs',
    'Vorinstanz': 'vorinstanz-docs',
    'Akte': 'akte-docs',
    'Rechtsprechung': 'rechtsprechung-docs',
    'Sonstiges': 'sonstiges-docs',
    'Sonstige gespeicherte Quellen': 'sonstiges-docs'
};

const DOCUMENT_POLL_INTERVAL_MS = 20000;
const SOURCES_POLL_INTERVAL_MS = 45000;
const STREAM_RETRY_BASE_MS = 4000;
const STREAM_MAX_RETRY_MS = 30000;
const STREAM_STALE_THRESHOLD_MS = 25000;
const STREAM_DISABLE_AFTER_FAILURES = 3;
const STREAM_DISABLE_DURATION_MS = 5 * 60 * 1000;

let documentPollingTimer = null;
let sourcesPollingTimer = null;
let documentFetchInFlight = false;
let sourcesFetchInFlight = false;
let lastDocumentSnapshotDigest = '';
let lastSourcesSnapshotDigest = '';
let documentFetchPromise = null;
let sourcesFetchPromise = null;
let documentStreamSource = null;
let documentStreamRetryTimer = null;
let lastStreamMessageAt = 0;
let documentStreamFailures = 0;
let documentStreamDisabledUntil = 0;

function resetSelectionState() {
    selectionState.anhoerung.clear();
    selectionState.vorinstanz.primary = null;
    selectionState.vorinstanz.others.clear();
    selectionState.rechtsprechung.clear();
    selectionState.akte.clear();
    selectionState.sonstiges.clear();
    selectionState.saved_sources.clear();
    selectionState.bescheid.primary = null;
    selectionState.bescheid.others.clear();
}

async function legacyResetCleanup() {
    const summary = {
        documentsDeleted: 0,
        sourcesDeleted: 0,
        errors: []
    };

    try {
        debugLog('legacyResetCleanup: fetching documents');
        const response = await fetch('/documents');
        if (response.ok) {
            const data = await response.json();
            const documents = [];
            if (data && typeof data === 'object') {
                Object.values(data).forEach(categoryDocs => {
                    if (Array.isArray(categoryDocs)) {
                        categoryDocs.forEach(doc => documents.push(doc));
                    }
                });
            }
            debugLog('legacyResetCleanup: deleting documents', { count: documents.length });
            for (const doc of documents) {
                if (!doc || !doc.filename) continue;
                const encoded = encodeURIComponent(doc.filename);
                const deleteResp = await fetch(`/documents/${encoded}`, { method: 'DELETE' });
                if (deleteResp.ok) {
                    summary.documentsDeleted += 1;
                } else {
                    summary.errors.push(`Dokument ${doc.filename} konnte nicht gelöscht werden (HTTP ${deleteResp.status})`);
                }
            }
        } else {
            summary.errors.push(`Dokumentliste konnte nicht geladen werden (HTTP ${response.status})`);
        }
    } catch (error) {
        debugError('legacyResetCleanup: document cleanup failed', error);
        summary.errors.push(`Dokumente: ${error.message}`);
    }

    try {
        debugLog('legacyResetCleanup: deleting all sources');
        const response = await fetch('/sources', { method: 'DELETE' });
        if (response.ok) {
            const data = await response.json().catch(() => ({}));
            const count = data.count ?? (Array.isArray(data) ? data.length : 0);
            summary.sourcesDeleted = typeof count === 'number' ? count : 0;
        } else if (response.status !== 404) {
            summary.errors.push(`Quellen konnten nicht gelöscht werden (HTTP ${response.status})`);
        }
    } catch (error) {
        debugError('legacyResetCleanup: source cleanup failed', error);
        summary.errors.push(`Quellen: ${error.message}`);
    }

    const baseMessage = `Fallback-Bereinigung abgeschlossen (Dokumente: ${summary.documentsDeleted}, Quellen: ${summary.sourcesDeleted})`;
    const message = summary.errors.length
        ? `${baseMessage}. Einige Elemente konnten nicht gelöscht werden.`
        : baseMessage;

    if (summary.errors.length) {
        debugError('legacyResetCleanup: encountered issues', summary.errors);
    } else {
        debugLog('legacyResetCleanup: completed successfully');
    }

    return { message, details: summary };
}

function pruneDocumentSelections(documentsByCategory) {
    const available = {
        anhoerung: new Set(),
        bescheid: new Set(),
        vorinstanz: new Set(),
        rechtsprechung: new Set()
    };

    for (const [category, documents] of Object.entries(documentsByCategory || {})) {
        const key = categoryToKey[category];
        if (!key || !(key in available)) continue;
        documents.forEach(doc => {
            if (doc && doc.filename) {
                available[key].add(doc.filename);
            }
        });
    }

    selectionState.anhoerung = new Set([...selectionState.anhoerung].filter(filename => available.anhoerung.has(filename)));

    if (selectionState.bescheid.primary && !available.bescheid.has(selectionState.bescheid.primary)) {
        selectionState.bescheid.primary = null;
    }
    selectionState.bescheid.others = new Set([...selectionState.bescheid.others].filter(filename => available.bescheid.has(filename)));

    if (selectionState.vorinstanz.primary && !available.vorinstanz.has(selectionState.vorinstanz.primary)) {
        selectionState.vorinstanz.primary = null;
    }
    selectionState.vorinstanz.others = new Set([...selectionState.vorinstanz.others].filter(filename => available.vorinstanz.has(filename)));

    selectionState.rechtsprechung = new Set([...selectionState.rechtsprechung].filter(filename => available.rechtsprechung.has(filename)));
}

function pruneSourceSelections(sources) {
    const available = new Set();
    sources.forEach(source => {
        if (source && source.id) {
            available.add(source.id);
        }
    });
    selectionState.saved_sources = new Set([...selectionState.saved_sources].filter(id => available.has(id)));
}

function toggleDocumentSelection(categoryKey, filename, isChecked, isPrimary = false) {
    debugLog('toggleDocumentSelection', { categoryKey, filename, isChecked, isPrimary });

    if (categoryKey === 'bescheid') {
        if (isPrimary) {
            if (isChecked) {
                selectionState.bescheid.primary = filename;
                if (selectionState.bescheid.others.has(filename)) {
                    selectionState.bescheid.others.delete(filename);
                }
            } else if (selectionState.bescheid.primary === filename) {
                selectionState.bescheid.primary = null;
            }
        } else {
            if (isChecked) {
                selectionState.bescheid.others.add(filename);
                if (selectionState.bescheid.primary === filename) {
                    selectionState.bescheid.primary = null;
                }
            } else {
                selectionState.bescheid.others.delete(filename);
            }
        }
    } else if (categoryKey === 'vorinstanz') {
        const encoded = encodeURIComponent(filename);
        const container = document.getElementById('vorinstanz-docs');

        if (isPrimary) { // Radio logic (Urteil selected)
            if (isChecked) {
                selectionState.vorinstanz.primary = filename;
                if (selectionState.vorinstanz.others.has(filename)) {
                    selectionState.vorinstanz.others.delete(filename);
                }
                // Force check the checkbox (because Primary implies Selected)
                if (container) {
                    const checkbox = container.querySelector(`input[type="checkbox"][data-filename="${encoded}"]`);
                    if (checkbox) checkbox.checked = true;
                }
            }
        } else { // Checkbox logic (Verwenden toggled)
            if (isChecked) {
                if (!selectionState.vorinstanz.primary) {
                    // Auto-promote to primary if none selected
                    selectionState.vorinstanz.primary = filename;
                    // Update Radio DOM
                    if (container) {
                        const radio = container.querySelector(`input[type="radio"][name="vorinstanz-primary"][data-filename="${encoded}"]`);
                        if (radio) radio.checked = true;
                    }
                } else if (selectionState.vorinstanz.primary !== filename) {
                    selectionState.vorinstanz.others.add(filename);
                }
            } else {
                selectionState.vorinstanz.others.delete(filename);
                if (selectionState.vorinstanz.primary === filename) {
                    selectionState.vorinstanz.primary = null;
                    // Update Radio DOM
                    if (container) {
                        const radio = container.querySelector(`input[type="radio"][name="vorinstanz-primary"][data-filename="${encoded}"]`);
                        if (radio) radio.checked = false;
                    }
                }
            }
        }
    } else {
        const set = selectionState[categoryKey];
        if (set) {
            if (isChecked) {
                set.add(filename);
            } else {
                set.delete(filename);
            }
        }
    }
}

function toggleSavedSourceSelection(sourceId, isChecked) {
    if (!sourceId) return;
    if (isChecked) selectionState.saved_sources.add(sourceId);
    else selectionState.saved_sources.delete(sourceId);
}

function toggleBescheidSelection(filename, isChecked) {
    // This function is now deprecated by the new toggleDocumentSelection
    // but keeping it for now if there are other call sites not covered by the change.
    // Ideally, all calls should be migrated to toggleDocumentSelection.
    if (!filename) return;
    if (isChecked) {
        if (!selectionState.bescheid.primary) {
            selectionState.bescheid.primary = filename;
            const encoded = encodeURIComponent(filename);
            const radio = document.querySelector(`input[type="radio"][name="bescheidPrimary"][data-filename="${encoded}"]`);
            if (radio) radio.checked = true;
        } else if (selectionState.bescheid.primary !== filename) {
            selectionState.bescheid.others.add(filename);
        }
    } else {
        selectionState.bescheid.others.delete(filename);
        if (selectionState.bescheid.primary === filename) {
            selectionState.bescheid.primary = null;
            const encoded = encodeURIComponent(filename);
            const radio = document.querySelector(`input[type="radio"][name="bescheidPrimary"][data-filename="${encoded}"]`);
            if (radio) radio.checked = false;
        }
    }
}

function setPrimaryBescheid(filename) {
    // This function is now deprecated by the new toggleDocumentSelection
    // but keeping it for now if there are other call sites not covered by the change.
    // Ideally, all calls should be migrated to toggleDocumentSelection.
    if (!filename) return;
    selectionState.bescheid.primary = filename;
    selectionState.bescheid.others.delete(filename);
    selectionState.bescheid.others = new Set(selectionState.bescheid.others);
    const encoded = encodeURIComponent(filename);
    const checkbox = document.querySelector(`input[type="checkbox"][data-bescheid-checkbox="${encoded}"]`);
    if (checkbox) {
        checkbox.checked = true;
    }
}

function isDocumentSelected(categoryKey, filename) {
    if (!categoryKey || !filename) return false;
    if (categoryKey === 'bescheid') {
        return selectionState.bescheid.primary === filename || selectionState.bescheid.others.has(filename);
    }
    if (categoryKey === 'vorinstanz') {
        return selectionState.vorinstanz.primary === filename || selectionState.vorinstanz.others.has(filename);
    }
    const set = selectionState[categoryKey];
    return set ? set.has(filename) : false;
}

function isSourceSelected(sourceId) {
    return selectionState.saved_sources.has(sourceId);
}

function getSelectedDocumentsPayload() {
    const payload = {
        anhoerung: Array.from(selectionState.anhoerung),
        bescheid: {
            primary: selectionState.bescheid.primary,
            others: Array.from(selectionState.bescheid.others)
        },
        vorinstanz: {
            primary: selectionState.vorinstanz.primary,
            others: Array.from(selectionState.vorinstanz.others)
        },
        rechtsprechung: Array.from(selectionState.rechtsprechung),
        akte: Array.from(selectionState.akte),
        sonstiges: Array.from(selectionState.sonstiges),
        saved_sources: Array.from(selectionState.saved_sources)
    };
    return payload;
}

function validatePrimaryBescheid() {
    const primaryBescheid = selectionState.bescheid.primary;
    if (!primaryBescheid) {
        return null;
    }
    const encodedFilename = encodeURIComponent(primaryBescheid);
    const checkbox = document.querySelector(`input[type="checkbox"][data-bescheid-checkbox="${encodedFilename}"]`);
    if (!checkbox) {
        debugLog('validatePrimaryBescheid: not found in DOM, clearing', { primaryBescheid });
        selectionState.bescheid.primary = null;
        return null;
    }
    return primaryBescheid;
}

function showError(context, error) {
    debugError(`${context}: request error`, error);
    alert(`❌ Fehler: ${error.message}`);
}

let jlawyerTemplatesPromise = null;
let pollingActive = false;

async function ensureJLawyerTemplates() {
    if (Array.isArray(window.jlawyerTemplates)) {
        return window.jlawyerTemplates;
    }
    if (!jlawyerTemplatesPromise) {
        jlawyerTemplatesPromise = (async () => {
            try {
                debugLog('ensureJLawyerTemplates: fetching /jlawyer/templates');
                const response = await fetch('/jlawyer/templates');
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                const data = await response.json();
                const templates = Array.isArray(data.templates) ? data.templates : [];
                window.jlawyerTemplates = templates;
                return templates;
            } catch (error) {
                debugError('ensureJLawyerTemplates: failed', error);
                window.jlawyerTemplates = [];
                return [];
            } finally {
                jlawyerTemplatesPromise = null;
            }
        })();
    }
    return jlawyerTemplatesPromise;
}

function handleDocumentCheckboxChange(categoryKey, element) {
    // This function is now deprecated by the new toggleDocumentSelection
    // but keeping it for now if there are other call sites not covered by the change.
    // Ideally, all calls should be migrated to toggleDocumentSelection.
    if (!element) return;
    const encoded = element.dataset?.filename || '';
    let filename = encoded;
    try { filename = decodeURIComponent(encoded); } catch (e) { /* ignore */ }
    toggleDocumentSelection(categoryKey, filename, element.checked);
}

function handleBescheidCheckboxChange(element) {
    // This function is now deprecated by the new toggleDocumentSelection
    // but keeping it for now if there are other call sites not covered by the change.
    // Ideally, all calls should be migrated to toggleDocumentSelection.
    if (!element) return;
    const encoded = element.dataset?.filename || '';
    let filename = encoded;
    try { filename = decodeURIComponent(encoded); } catch (e) { /* ignore */ }
    toggleBescheidSelection(filename, element.checked);
}

function handlePrimaryBescheidSelect(element) {
    // This function is now deprecated by the new toggleDocumentSelection
    // but keeping it for now if there are other call sites not covered by the change.
    // Ideally, all calls should be migrated to toggleDocumentSelection.
    if (!element) return;
    const encoded = element.dataset?.filename || '';
    let filename = encoded;
    try { filename = decodeURIComponent(encoded); } catch (e) { /* ignore */ }
    setPrimaryBescheid(filename);
}

function handleSavedSourceCheckboxChange(element) {
    if (!element) return;
    const encoded = element.dataset?.sourceId || '';
    let sourceId = encoded;
    try { sourceId = decodeURIComponent(encoded); } catch (e) { /* ignore */ }
    toggleSavedSourceSelection(sourceId, element.checked);
}

function handleDocumentSnapshot(payload) {
    const data = payload && typeof payload === 'object' ? payload : {};
    const reason = data.reason || 'update';

    if (reason === 'reset') {
        debugLog('handleDocumentSnapshot: reset detected, clearing all selections');
        resetSelectionState();
        // Force clear digest to ensure UI redraws
        lastDocumentSnapshotDigest = '';
    }

    if (data.documents) {
        renderDocuments(data.documents, { force: reason === 'reset' });
    } else {
        loadDocuments({ placeholders: false }).catch((err) => debugError('handleDocumentSnapshot: reload failed', err));
    }
}

function handleSourceSnapshot(payload) {
    const data = payload && typeof payload === 'object' ? payload : {};
    const reason = data.reason || 'update';

    if (reason === 'reset') {
        selectionState.saved_sources.clear();
        lastSourcesSnapshotDigest = '';
    }

    if (data.sources) {
        renderSources(data.sources, { force: reason === 'reset' });
    } else {
        loadSources({ placeholders: false }).catch((err) => debugError('handleSourceSnapshot: reload failed', err));
    }
}

function renderSources(sources, options) {
    const force = !!(options && options.force);
    const sourcesArray = Array.isArray(sources) ? sources : [];

    pruneSourceSelections(sourcesArray);

    const digest = JSON.stringify(sourcesArray);
    if (!force && digest === lastSourcesSnapshotDigest) {
        debugLog('renderSources: snapshot unchanged, skipping redraw');
        return;
    }

    lastSourcesSnapshotDigest = digest;

    const container = document.getElementById('sources-docs');
    if (!container) return;

    let root = container.querySelector('[data-sources-root]');
    if (!root) {
        root = document.createElement('div');
        root.setAttribute('data-sources-root', 'true');
        container.appendChild(root);
    }

    if (sourcesArray.length === 0) {
        root.innerHTML = '<div class="empty-message">Keine Quellen gespeichert</div>';
    } else {
        root.innerHTML = sourcesArray.map(source => createSourceCard(source)).join('');
    }
}

function startDocumentStream(delayMs) {
    const now = Date.now();
    if (now < documentStreamDisabledUntil) {
        const waitMs = Math.max(0, documentStreamDisabledUntil - now);
        debugLog('documentStream: temporarily disabled, retry scheduled', { waitMs });
        if (!documentStreamRetryTimer) {
            documentStreamRetryTimer = setTimeout(() => {
                documentStreamRetryTimer = null;
                startDocumentStream();
            }, waitMs);
        }
        return;
    }

    const initialDelay = typeof delayMs === 'number' ? Math.max(0, delayMs) : 0;
    const nextDelay = Math.min(
        STREAM_MAX_RETRY_MS,
        initialDelay > 0 ? initialDelay * 1.5 : STREAM_RETRY_BASE_MS
    );

    if (documentStreamRetryTimer) {
        clearTimeout(documentStreamRetryTimer);
        documentStreamRetryTimer = null;
    }

    const connect = () => {
        if (documentStreamRetryTimer) {
            clearTimeout(documentStreamRetryTimer);
            documentStreamRetryTimer = null;
        }
        if (documentStreamSource) {
            try { documentStreamSource.close(); } catch (err) { /* ignore */ }
            documentStreamSource = null;
        }

        try {
            const token = getAuthToken();
            const url = token ? `/documents/stream?token=${token}` : '/documents/stream';
            const source = new EventSource(url);
            documentStreamSource = source;
            lastStreamMessageAt = Date.now();
            documentStreamFailures = 0;
            documentStreamDisabledUntil = 0;
            debugLog('documentStream: connected');

            source.onmessage = (event) => {
                lastStreamMessageAt = Date.now();
                if (!event || !event.data) {
                    return;
                }
                let payload = null;
                try {
                    payload = JSON.parse(event.data);
                } catch (parseError) {
                    debugError('unified stream: failed to parse payload', parseError);
                    return;
                }
                if (payload?.type === 'documents_snapshot') {
                    handleDocumentSnapshot(payload);
                } else if (payload?.type === 'sources_snapshot') {
                    handleSourceSnapshot(payload);
                } else {
                    debugLog('unified stream: unknown event type', payload?.type);
                }
            };

            source.onerror = (error) => {
                debugError('documentStream: error', error);
                lastStreamMessageAt = 0;
                try { source.close(); } catch (err) { /* ignore */ }
                documentStreamSource = null;
                documentStreamFailures += 1;
                if (documentStreamFailures >= STREAM_DISABLE_AFTER_FAILURES) {
                    documentStreamDisabledUntil = Date.now() + STREAM_DISABLE_DURATION_MS;
                    documentStreamFailures = 0;
                    debugError('documentStream: disabled after repeated failures, falling back to polling', {
                        disabledUntil: new Date(documentStreamDisabledUntil).toISOString()
                    });
                    if (!documentStreamRetryTimer) {
                        const resumeDelay = Math.max(0, documentStreamDisabledUntil - Date.now());
                        documentStreamRetryTimer = setTimeout(() => {
                            documentStreamRetryTimer = null;
                            startDocumentStream();
                        }, resumeDelay);
                    }
                    return;
                }
                documentStreamRetryTimer = setTimeout(() => startDocumentStream(nextDelay), nextDelay);
            };
        } catch (error) {
            debugError('documentStream: connection failed', error);
            documentStreamRetryTimer = setTimeout(() => startDocumentStream(nextDelay), nextDelay);
        }
    };

    if (initialDelay > 0) {
        documentStreamRetryTimer = setTimeout(connect, initialDelay);
    } else {
        connect();
    }
}

function startAutoRefresh() {
    if (pollingActive) {
        return;
    }
    pollingActive = false; // Rely exclusively on SSE; no polling fallback
    debugLog('Polling DISABLED - relying entirely on SSE');
}

// Load documents and sources on page load
window.addEventListener('DOMContentLoaded', () => {
    debugLog('DOMContentLoaded: initializing interface');

    // Check if marked.js loaded successfully
    if (typeof marked !== 'undefined' && marked.parse) {
        debugLog('Marked.js library loaded successfully - markdown rendering enabled');
    } else {
        console.warn('Marked.js library not available - falling back to plain text rendering');
    }

    loadDocuments();
    loadSources();
    startDocumentStream();
    startAutoRefresh();

    // Setup model selection handler for showing/hiding verbosity
    const modelSelect = document.getElementById('modelSelect');
    const verbosityGroup = document.getElementById('verbosityGroup');
    if (modelSelect && verbosityGroup) {
        modelSelect.addEventListener('change', function () {
            const isGPT = this.value.startsWith('gpt');
            verbosityGroup.style.display = isGPT ? 'flex' : 'none';
            debugLog('Model changed:', this.value, 'showing GPT controls:', isGPT);
        });
    }
});

window.addEventListener('beforeunload', () => {
    if (documentStreamRetryTimer) {
        clearTimeout(documentStreamRetryTimer);
        documentStreamRetryTimer = null;
    }
    if (documentStreamSource) {
        try { documentStreamSource.close(); } catch (err) { /* ignore */ }
        documentStreamSource = null;
    }
    if (documentPollingTimer) {
        clearInterval(documentPollingTimer);
        documentPollingTimer = null;
    }
    if (sourcesPollingTimer) {
        clearInterval(sourcesPollingTimer);
        sourcesPollingTimer = null;
    }
});

function renderUnifiedSonstiges() {
    const container = document.getElementById('sonstiges-docs');
    if (!container) return;

    // Get documents from "Sonstige gespeicherte Quellen"
    const documents = globalDocuments['Sonstige gespeicherte Quellen'] || [];

    // Get all sources
    const sources = globalSources || [];

    // Combine and sort by date (newest first)
    const combined = [
        ...documents.map(d => ({ ...d, _type: 'document', timestamp: d.timestamp || d.created_at })),
        ...sources.map(s => ({ ...s, _type: 'source', timestamp: s.timestamp || s.created_at }))
    ].sort((a, b) => {
        const dateA = new Date(a.timestamp || 0);
        const dateB = new Date(b.timestamp || 0);
        return dateB - dateA; // Descending
    });

    if (combined.length === 0) {
        container.innerHTML = '<div class="empty-message">Keine Dokumente oder Quellen</div>';
        return;
    }

    const html = combined.map(item => {
        if (item._type === 'document') return createDocumentCard(item);
        if (item._type === 'source') return createSourceCard(item);
        return '';
    }).join('');

    container.innerHTML = html;
}

function renderDocuments(grouped, options) {
    const force = !!(options && options.force);
    const data = grouped && typeof grouped === 'object' ? grouped : {};
    globalDocuments = data; // Update global state

    // Don't prune selections if this is a forced render after reset
    // (selections were already cleared by resetSelectionState)
    if (!force) {
        pruneDocumentSelections(data);
    }

    const digest = JSON.stringify(data);
    if (!force && digest === lastDocumentSnapshotDigest) {
        debugLog('renderDocuments: snapshot unchanged, skipping redraw');
        return;
    }

    lastDocumentSnapshotDigest = digest;

    Object.entries(DOCUMENT_CONTAINER_IDS).forEach(([category, elementId]) => {
        // Skip Sonstiges/Sources - handled by unified renderer
        if (category === 'Sonstiges' || category === 'Sonstige gespeicherte Quellen') return;

        const el = document.getElementById(elementId);
        if (!el) return;
        const docsArray = Array.isArray(data[category]) ? data[category] : [];
        if (docsArray.length === 0) {
            el.innerHTML = '<div class="empty-message">Keine Dokumente</div>';
        } else {
            const cards = docsArray
                .map(doc => createDocumentCard(doc))
                .filter(Boolean)
                .join('');
            el.innerHTML = cards || '<div class="empty-message">Keine Dokumente</div>';
        }
    });

    renderUnifiedSonstiges();
}

function renderSources(sourcesList) {
    globalSources = Array.isArray(sourcesList) ? sourcesList : [];
    // pruneSourceSelections(globalSources); // Optional: if we want to clean up selections
    renderUnifiedSonstiges();
}

async function loadDocuments(options) {
    const placeholders = !(options && options.placeholders === false);
    debugLog('loadDocuments: start', { placeholders });

    if (documentFetchInFlight && documentFetchPromise) {
        debugLog('loadDocuments: reusing in-flight request');
        return documentFetchPromise;
    }

    if (placeholders) {
        Object.values(DOCUMENT_CONTAINER_IDS).forEach((elementId) => {
            const el = document.getElementById(elementId);
            if (el) {
                el.innerHTML = '<div class="empty-message">⏳ Lädt …</div>';
            }
        });
    }

    documentFetchInFlight = true;
    documentFetchPromise = (async () => {
        try {
            debugLog('loadDocuments: fetching /documents');
            const response = await fetch('/documents', { cache: 'no-store' });
            debugLog('loadDocuments: response status', response.status);
            if (response.status === 429) {
                const retryAfterHeader = response.headers.get('Retry-After');
                const retryAfterSeconds = retryAfterHeader ? parseInt(retryAfterHeader, 10) : NaN;
                const delayMs = Number.isFinite(retryAfterSeconds) && retryAfterSeconds > 0
                    ? retryAfterSeconds * 1000
                    : DOCUMENT_POLL_INTERVAL_MS * 2;
                debugError('loadDocuments: rate limited, scheduling retry', { delayMs });
                setTimeout(() => {
                    loadDocuments({ placeholders: false }).catch((err) => debugError('loadDocuments retry failed', err));
                }, delayMs);
                return null;
            }
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            debugLog('loadDocuments: received data', data);

            renderDocuments(data, { force: placeholders });
            return data;
        } catch (error) {
            debugError('loadDocuments: failed', error);
            lastDocumentSnapshotDigest = '';
            Object.values(DOCUMENT_CONTAINER_IDS).forEach((elementId) => {
                const el = document.getElementById(elementId);
                if (el) {
                    el.innerHTML = '<div class="empty-message">⚠️ Konnte Dokumente nicht laden</div>';
                }
            });
            return null;
        } finally {
            documentFetchInFlight = false;
            documentFetchPromise = null;
        }
    })();

    return documentFetchPromise;
}

function createDocumentCard(doc) {
    if (!doc || !doc.filename) {
        debugLog('createDocumentCard: skipping entry without filename', doc);
        return '';
    }

    const categoryKey = categoryToKey[doc.category];
    const encodedFilename = encodeURIComponent(doc.filename);
    const jsSafeFilename = escapeJsString(doc.filename);
    const confidenceValue = typeof doc.confidence === 'number'
        ? `${(doc.confidence * 100).toFixed(0)}% Konfidenz`
        : 'Konfidenz unbekannt';
    debugLog('createDocumentCard', { filename: doc.filename, confidence: doc.confidence, categoryKey });
    const showAnonymizeBtn = ANONYMIZABLE_CATEGORIES.has(doc.category);
    const isAnonymized = !!doc.anonymized;

    const anonymizedBadge = isAnonymized
        ? `<div class="status-badge anonymized">✅ Anonymisiert</div>`
        : '';

    const needsOcr = !!doc.needs_ocr;
    const ocrApplied = !!doc.ocr_applied;

    const ocrBadge = ocrApplied
        ? `<div class="status-badge ocr-completed">✅ OCR durchgeführt</div>`
        : (needsOcr ? `<div class="status-badge ocr-needed">📄 OCR benötigt</div>` : '');

    const ocrButton = needsOcr
        ? `
            <button class="ocr-btn"
                    onclick="performOcrOnDocument('${doc.id}', this)"
                    title="OCR durchführen">
                📄 OCR starten
            </button>
        `
        : '';

    const anonymizeButton = showAnonymizeBtn
        ? `
            <button class="anonymize-btn${isAnonymized ? ' secondary' : ''}"
                    onclick="anonymizeDocument('${doc.id}', this)"
                    title="${isAnonymized ? 'Erneut anonymisieren' : 'Anonymisieren'}">
                ${isAnonymized ? '🔄 Erneut anonymisieren' : '🔒 Anonymisieren'}
            </button>
        `
        : '';

    let selectionControls = '';
    if (categoryKey === 'bescheid') {
        const isPrimary = selectionState.bescheid.primary === doc.filename;
        const isSelected = isDocumentSelected('bescheid', doc.filename);
        selectionControls = `
            <div class="selection-controls">
                <label class="selection-option">
                    <input type="checkbox"
                           data-bescheid-checkbox="${encodedFilename}"
                           data-filename="${encodedFilename}"
                           ${isSelected ? 'checked' : ''}
                           onchange="handleBescheidCheckboxChange(this)">
                    Verwenden
                </label>
                <label class="selection-option">
                    <input type="radio"
                           name="bescheidPrimary"
                           data-filename="${encodedFilename}"
                           ${isPrimary ? 'checked' : ''}
                           onclick="handlePrimaryBescheidSelect(this)">
                    Anlage K2
                </label>
            </div>
        `;
    } else if (categoryKey === 'vorinstanz') {
        const isPrimary = selectionState.vorinstanz.primary === doc.filename;
        const isSelected = isDocumentSelected('vorinstanz', doc.filename); // Check if selected as primary OR other

        // For the checkbox, we want it checked if it's in 'others' OR 'primary' (conceptually 'used')
        // But in Bescheid logic:
        // - Checkbox 'Verwenden' usually maps to 'others'.
        // - Radio 'Anlage K2' maps to 'primary'.
        // However, the user said "same functionality".
        // In Bescheid box:
        // Checkbox is for "Verwenden" (adds to others).
        // Radio is for "Anlage K2" (sets primary).
        // If I click Radio, it becomes primary.
        // If I click Checkbox, it becomes other.

        // Let's look at isSelected for Bescheid:
        // const isSelected = isDocumentSelected('bescheid', doc.filename);
        // isDocumentSelected returns true if primary OR in others.

        // So if I select Primary, the Checkbox "Verwenden" should also be checked?
        // In the Bescheid implementation I restored:
        // ${isSelected ? 'checked' : ''} for the checkbox.
        // So yes, if it is primary, "Verwenden" is ALSO checked visually.

        selectionControls = `
            <div class="selection-controls">
                <label class="selection-option">
                    <input type="checkbox"
                           data-filename="${encodedFilename}"
                           ${isSelected ? 'checked' : ''}
                           onchange="toggleDocumentSelection('vorinstanz', '${jsSafeFilename}', this.checked, false)">
                    Verwenden
                </label>
                <label class="selection-option">
                    <input type="radio" name="vorinstanz-primary" 
                           ${isPrimary ? 'checked' : ''} 
                           data-filename="${encodedFilename}"
                           onchange="toggleDocumentSelection('vorinstanz', '${jsSafeFilename}', true, true)">
                    Urteil
                </label>
            </div>
        `;
    } else if (categoryKey === 'anhoerung' || categoryKey === 'rechtsprechung' || categoryKey === 'saved_sources' || categoryKey === 'akte' || categoryKey === 'sonstiges') {
        const checked = isDocumentSelected(categoryKey, doc.filename) ? 'checked' : '';
        selectionControls = `
            <label class="selection-option">
                <input type="checkbox"
                       ${checked}
                       data-filename="${encodedFilename}"
                       onchange="toggleDocumentSelection('${categoryKey}', '${jsSafeFilename}', this.checked)">
                Verwenden
            </label>
        `;
    }

    return `
        <div class="document-card">
            <button class="delete-btn" onclick="deleteDocument('${jsSafeFilename}')" title="Löschen">×</button>
            ${selectionControls ? `<div class="selection-wrapper">${selectionControls}</div>` : ''}
            <div class="filename" title="${escapeAttribute(doc.filename)}">
                <a href="#" onclick="viewDocument('${encodedFilename}', event)" title="Dokument ansehen">${escapeHtml(doc.filename)}</a>
            </div>
            <div class="confidence">${escapeHtml(confidenceValue)}</div>
            ${anonymizedBadge}
            ${ocrBadge}
            ${ocrButton}
            ${anonymizeButton}
        </div>
    `;
}

async function loadSources(options) {
    const placeholders = !(options && options.placeholders === false);
    debugLog('loadSources: start');
    const container = document.getElementById('sources-docs');

    if (sourcesFetchInFlight && sourcesFetchPromise) {
        debugLog('loadSources: reusing in-flight request');
        return sourcesFetchPromise;
    }

    if (placeholders && container) {
        container.innerHTML = '';
    }

    const ensureSourcesRoot = () => {
        if (!container) return null;
        let root = container.querySelector('[data-sources-root]');
        if (!root) {
            root = document.createElement('div');
            root.setAttribute('data-sources-root', 'true');
            container.appendChild(root);
        }
        return root;
    };
    const sourcesRoot = ensureSourcesRoot();

    if (placeholders && container) {
        if (sourcesRoot) {
            sourcesRoot.innerHTML = '<div class="empty-message">⏳ Lädt …</div>';
        } else {
            container.innerHTML = '<div class="empty-message">⏳ Lädt …</div>';
        }
    }

    sourcesFetchInFlight = true;
    sourcesFetchPromise = (async () => {
        try {
            debugLog('loadSources: fetching /sources');
            const response = await fetch('/sources', { cache: 'no-store' });
            debugLog('loadSources: response status', response.status);
            if (response.status === 429) {
                const retryAfterHeader = response.headers.get('Retry-After');
                const retryAfterSeconds = retryAfterHeader ? parseInt(retryAfterHeader, 10) : NaN;
                const delayMs = Number.isFinite(retryAfterSeconds) && retryAfterSeconds > 0
                    ? retryAfterSeconds * 1000
                    : SOURCES_POLL_INTERVAL_MS * 2;
                debugError('loadSources: rate limited, scheduling retry', { delayMs });
                setTimeout(() => {
                    loadSources({ placeholders: false }).catch((err) => debugError('loadSources retry failed', err));
                }, delayMs);
                return null;
            }
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const payload = await response.json();
            const sources = Array.isArray(payload) ? payload : (payload.sources || []);
            const count = Array.isArray(payload) ? payload.length : (payload.count ?? sources.length);
            debugLog('loadSources: received payload', { count, sources });

            debugLog('loadSources: target container located', container);

            pruneSourceSelections(sources);

            const digest = JSON.stringify(sources);
            if (digest === lastSourcesSnapshotDigest) {
                debugLog('loadSources: snapshot unchanged, skipping redraw');
                return sources;
            }
            lastSourcesSnapshotDigest = digest;

            if (sourcesRoot) {
                if (sources.length === 0) {
                    debugLog('loadSources: no sources stored');
                    sourcesRoot.innerHTML = '<div class="empty-message">Keine Quellen gespeichert</div>';
                } else {
                    debugLog('loadSources: rendering source cards');
                    sourcesRoot.innerHTML = sources.map(source => createSourceCard(source)).join('');
                }
            } else if (container) {
                container.innerHTML = sources.length === 0
                    ? '<div class="empty-message">Keine Quellen gespeichert</div>'
                    : sources.map(source => createSourceCard(source)).join('');
            }
            return sources;
        } catch (error) {
            debugError('loadSources: failed', error);
            lastSourcesSnapshotDigest = '';
            if (sourcesRoot) {
                sourcesRoot.innerHTML = '<div class="empty-message">⚠️ Konnte Quellen nicht laden</div>';
            } else if (container) {
                container.innerHTML = '<div class="empty-message">⚠️ Konnte Quellen nicht laden</div>';
            }
            return null;
        } finally {
            sourcesFetchInFlight = false;
            sourcesFetchPromise = null;
        }
    })();

    return sourcesFetchPromise;
}

function createSourceCard(source) {
    debugLog('createSourceCard', source);
    const statusEmoji = {
        'completed': '✅',
        'downloading': '⏳',
        'pending': '📥',
        'failed': '❌',
        'skipped': '⏭️'
    }[source.download_status] || '📄';

    const downloadButton = source.download_status === 'completed'
        ? `<a href="/sources/download/${source.id}" target="_blank" style="color: #2c3e50; text-decoration: none; font-size: 12px;">📥 PDF</a>`
        : '';
    const pdfLinkButton = source.pdf_url
        ? `<a href="${escapeAttribute(source.pdf_url)}" target="_blank" style="color: #2c3e50; text-decoration: none; font-size: 12px; margin-left: 6px;">🔗 Original-PDF</a>`
        : '';
    const descriptionHtml = source.description
        ? `<div style="margin-top: 6px; color: #555; font-size: 13px; line-height: 1.4;">${escapeForTemplate(source.description)}</div>`
        : '';

    const isSelected = isSourceSelected(source.id);
    const encodedId = encodeURIComponent(source.id || '');
    const jsSafeId = escapeJsString(source.id || '');

    return `
        <div class="document-card">
            <button class="delete-btn" onclick="deleteSource('${jsSafeId}')" title="Löschen">×</button>
            <div class="selection-wrapper">
                <label class="selection-option">
                    <input type="checkbox"
                           ${isSelected ? 'checked' : ''}
                           data-source-id="${encodedId}"
                           onchange="handleSavedSourceCheckboxChange(this)">
                    Verwenden
                </label>
            </div>
            <div class="filename">
                <a href="${escapeAttribute(source.url)}" target="_blank" style="color: inherit; text-decoration: none;">
                    ${escapeHtml(source.title)}
                </a>
                ${downloadButton}${pdfLinkButton}
            </div>
            <div class="confidence">
                ${statusEmoji} ${source.document_type || 'Quelle'}
            </div>
            ${descriptionHtml}
        </div>
    `;
}

async function addSourceFromResults(evt, index) {
    const sources = window.latestResearchSources || [];
    const source = sources[index];
    if (!source) {
        debugError('addSourceFromResults: no source found for index', { index });
        alert('❌ Quelle konnte nicht gefunden werden.');
        return;
    }

    const button = evt?.target;
    if (button) {
        button.disabled = true;
        button.textContent = '⏳ Wird hinzugefügt...';
    }

    let addedSuccessfully = false;
    debugLog('addSourceFromResults: start', source);
    try {
        const response = await fetch('/sources', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                title: source.title,
                url: source.url,
                description: source.description,
                pdf_url: source.pdf_url,
                document_type: source.document_type || 'Rechtsprechung',
                research_query: window.latestResearchQuery || 'Recherche',
                auto_download: !!source.pdf_url
            })
        });
        debugLog('addSourceFromResults: response status', response.status);
        const data = await response.json();

        if (response.ok) {
            addedSuccessfully = true;
            debugLog('addSourceFromResults: success', data);
            alert('✅ Quelle gespeichert. Download startet im Hintergrund.');
            if (button) {
                button.disabled = true;
                button.textContent = '✅ Quelle hinzugefügt';
            }
            // Sources list will auto-update via SSE
        } else {
            debugError('addSourceFromResults: server error', data);
            alert(`❌ Quelle konnte nicht gespeichert werden: ${data.detail || 'Unbekannter Fehler'}`);
        }
    } catch (error) {
        showError('addSourceFromResults', error);
    } finally {
        if (button && !addedSuccessfully) {
            button.disabled = false;
            button.textContent = '➕ Zu gespeicherten Quellen';
        }
    }
}

async function deleteSource(sourceId) {
    debugLog('deleteSource: requested', sourceId);
    if (!confirm('Quelle wirklich löschen?')) {
        debugLog('deleteSource: user cancelled', sourceId);
        return;
    }

    try {
        debugLog('deleteSource: sending DELETE', sourceId);
        const response = await fetch(`/sources/${sourceId}`, {
            method: 'DELETE'
        });
        debugLog('deleteSource: response status', response.status);

        if (response.ok) {
            debugLog('deleteSource: success');
            // Sources list will auto-update via SSE
        } else {
            const data = await response.json();
            debugError('deleteSource: server error', data);
            alert(`❌ Fehler: ${data.detail || 'Löschen fehlgeschlagen'}`);
        }
    } catch (error) {
        showError('deleteSource', error);
    }
}

async function deleteAllSources() {
    debugLog('deleteAllSources: requested');
    if (!confirm('Wirklich ALLE gespeicherten Quellen löschen? Diese Aktion kann nicht rückgängig gemacht werden.')) {
        debugLog('deleteAllSources: user cancelled');
        return;
    }

    try {
        debugLog('deleteAllSources: sending DELETE /sources');
        const response = await fetch('/sources', {
            method: 'DELETE'
        });
        debugLog('deleteAllSources: response status', response.status);

        if (response.ok) {
            const data = await response.json();
            debugLog('deleteAllSources: success', data);
            alert(`✅ ${data.count} Quellen gelöscht`);
            // Sources list will auto-update via SSE
        } else {
            const data = await response.json();
            debugError('deleteAllSources: server error', data);
            alert(`❌ Fehler: ${data.detail || 'Löschen fehlgeschlagen'}`);
        }
    } catch (error) {
        showError('deleteAllSources', error);
    }
}

async function resetApplication() {
    debugLog('resetApplication: requested');
    if (!confirm('Wirklich alle Dokumente, Quellen und Downloads löschen? Diese Aktion kann nicht rückgängig gemacht werden.')) {
        debugLog('resetApplication: user cancelled');
        return;
    }

    try {
        debugLog('resetApplication: sending DELETE /reset');
        const response = await fetch('/reset', { method: 'DELETE' });
        let data = {};
        try {
            data = await response.json();
        } catch (parseError) {
            debugLog('resetApplication: could not parse JSON response', parseError);
        }
        debugLog('resetApplication: response status', response.status, data);

        if (response.ok) {
            debugLog('resetApplication: backend success');
            alert(`✅ ${data.message || 'Alle Daten wurden gelöscht.'}`);
            return;
        }

        if (response.status === 404) {
            debugLog('resetApplication: /reset missing, running legacy cleanup fallback');
            const fallbackResult = await legacyResetCleanup();
            resetSelectionState();
            debugLog('resetApplication: fallback manual refresh');
            await Promise.all([loadDocuments(), loadSources()]);
            debugLog('resetApplication: fallback refresh completed');
            alert(`✅ ${fallbackResult.message}`);
            return;
        }

        debugError('resetApplication: server error', data);
        alert(`❌ Fehler: ${data.detail || data.message || 'Löschen fehlgeschlagen'}`);
    } catch (error) {
        showError('resetApplication', error);
    }
}

async function uploadFile() {
    debugLog('uploadFile: start');
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    if (!file) {
        debugLog('uploadFile: no file selected');
        alert('Bitte wählen Sie eine PDF-Datei aus');
        return;
    }

    const loading = document.getElementById('loading');
    debugLog('uploadFile: showing loading indicator');
    loading.style.display = 'block';

    const formData = new FormData();
    formData.append('file', file);
    debugLog('uploadFile: prepared form data', { filename: file.name, size: file.size });

    try {
        debugLog('uploadFile: sending POST /classify');
        const response = await fetch('/classify', {
            method: 'POST',
            body: formData
        });
        debugLog('uploadFile: response status', response.status);
        const data = await response.json();
        debugLog('uploadFile: response body', data);

        if (response.ok) {
            debugLog('uploadFile: classification succeeded');
            // Clear file input
            fileInput.value = '';
            // Show success message
            alert(`✅ Dokument klassifiziert als: ${data.category} (${(data.confidence * 100).toFixed(0)}%)`);
        } else {
            debugError('uploadFile: classification failed', data);
            alert(`❌ Fehler: ${data.detail || 'Unbekannter Fehler'}`);
        }
    } catch (error) {
        showError('uploadFile', error);
    } finally {
        debugLog('uploadFile: hiding loading indicator');
        loading.style.display = 'none';
    }
}

function triggerDirectUpload(category) {
    debugLog('triggerDirectUpload: category', category);
    const categoryKey = category.toLowerCase().replace('ö', 'oe');
    const fileInputId = `file-${categoryKey}`;
    const fileInput = document.getElementById(fileInputId);

    if (!fileInput) {
        debugError('triggerDirectUpload: file input not found', fileInputId);
        return;
    }

    fileInput.click();
}

async function uploadDirectFile(category, inputElement) {
    debugLog('uploadDirectFile: start', { category });
    if (!inputElement || !inputElement.files || inputElement.files.length === 0) {
        debugLog('uploadDirectFile: no file selected');
        return;
    }

    const files = Array.from(inputElement.files);
    debugLog('uploadDirectFile: files selected', { count: files.length });

    for (const file of files) {
        debugLog('uploadDirectFile: processing file', { filename: file.name, size: file.size });
        const formData = new FormData();
        formData.append('file', file);
        const serverCategory = categoryToServerValue[category] || category;
        formData.append('category', serverCategory);

        try {
            debugLog('uploadDirectFile: sending POST /upload-direct');
            const response = await fetch('/upload-direct', {
                method: 'POST',
                body: formData
            });

            debugLog('uploadDirectFile: response status', response.status);
            const data = await response.json();
            debugLog('uploadDirectFile: response body', data);

            if (response.ok) {
                debugLog('uploadDirectFile: upload succeeded', file.name);
            } else {
                debugError('uploadDirectFile: upload failed', { filename: file.name, error: data });
                alert(`❌ Upload fehlgeschlagen für ${file.name}: ${data.detail || 'Unbekannter Fehler'}`);
            }
        } catch (error) {
            showError(`uploadDirectFile (${file.name})`, error);
        }
    }

    // Refresh document list
    await loadDocuments({ placeholders: false });

    // Reset input
    inputElement.value = '';
}

async function deleteDocument(filename) {
    debugLog('deleteDocument: requested', filename);
    if (!confirm(`Möchten Sie "${filename}" wirklich löschen?`)) {
        debugLog('deleteDocument: user cancelled', filename);
        return;
    }

    try {
        debugLog('deleteDocument: sending DELETE', filename);
        const response = await fetch(`/documents/${encodeURIComponent(filename)}`, {
            method: 'DELETE'
        });
        debugLog('deleteDocument: response status', response.status);

        if (response.ok) {
            debugLog('deleteDocument: success');
            // Documents list will auto-update via SSE
        } else {
            const data = await response.json();
            debugError('deleteDocument: server error', data);
            alert(`❌ Fehler: ${data.detail || 'Löschen fehlgeschlagen'}`);
        }
    } catch (error) {
        showError('deleteDocument', error);
    }
}

async function performOcrOnDocument(docId, buttonElement) {
    debugLog('performOcrOnDocument: requested', docId);

    if (!confirm('OCR für dieses Dokument durchführen? Dies kann 30-60 Sekunden dauern.')) {
        debugLog('performOcrOnDocument: user cancelled');
        return;
    }

    const button = buttonElement;
    const originalText = button ? button.innerHTML : null;

    try {
        if (button) {
            button.disabled = true;
            button.innerHTML = '⏳ OCR läuft...';
        }

        debugLog('performOcrOnDocument: calling POST /documents/' + docId + '/ocr');
        const response = await fetch(`/documents/${docId}/ocr`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        debugLog('performOcrOnDocument: response status', response.status);

        if (response.status === 429) {
            const retryAfterHeader = response.headers.get('Retry-After');
            const retryAfterSeconds = retryAfterHeader ? parseInt(retryAfterHeader, 10) : NaN;
            const retryMsg = !isNaN(retryAfterSeconds)
                ? `Zu viele Anfragen. Bitte warten Sie ${retryAfterSeconds} Sekunden.`
                : 'Zu viele Anfragen. Bitte versuchen Sie es später erneut.';
            throw new Error(retryMsg);
        }

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Unbekannter Fehler' }));
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();
        debugLog('performOcrOnDocument: success', data);

        const textLength = data.extracted_text ? data.extracted_text.length : 0;
        alert(`OCR erfolgreich abgeschlossen!\n\nExtrahierte Zeichen: ${textLength}\n\nDas Dokument wurde verarbeitet.`);

        // Refresh documents to update UI (removes OCR badge/button)
        await loadDocuments({ placeholders: false });

    } catch (error) {
        debugError('performOcrOnDocument: error', error);
        alert(`OCR fehlgeschlagen:\n\n${error.message}`);
    } finally {
        if (button && originalText) {
            button.disabled = false;
            button.innerHTML = originalText;
        }
    }
}

async function anonymizeDocument(docId, buttonElement, options) {
    const opts = options || {};
    const skipConfirm = !!opts.skipConfirm;
    const isRetry = !!opts.isRetry;

    debugLog('anonymizeDocument: requested', docId);

    if (!skipConfirm) {
        if (!confirm('Dokument anonymisieren? Dies kann 30-60 Sekunden dauern.')) {
            debugLog('anonymizeDocument: user cancelled');
            return;
        }
    }

    // Find the button and show loading state
    const button = buttonElement || (typeof event !== 'undefined' ? event.target : null);
    const originalText = button ? button.innerHTML : null;

    if (button) {
        button.innerHTML = '⏳ Verarbeite...';
        button.disabled = true;
    }

    try {
        debugLog('anonymizeDocument: sending POST', docId);
        const response = await fetch(`/documents/${docId}/anonymize`, {
            method: 'POST'
        });
        debugLog('anonymizeDocument: response status', response.status);

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Anonymisierung fehlgeschlagen');
        }

        const result = await response.json();
        debugLog('anonymizeDocument: success', result);

        // Show result in modal
        showAnonymizationResult(result);
        // Document list will auto-update via SSE (anonymized badge appears)

    } catch (error) {
        if (button && originalText !== null) {
            button.innerHTML = originalText;
            button.disabled = false;
        }

        debugError('anonymizeDocument: error', error);

        // Provide helpful error message for common cases
        let errorMsg = error.message;
        if (errorMsg.toLowerCase().includes('parse model response')) {
            alert(`Anonymisierung fehlgeschlagen

Die Antwort des KI-Dienstes konnte nicht als gültiges JSON interpretiert werden.

Bitte versuchen Sie es erneut. Wenn das Problem bestehen bleibt, prüfen Sie die Logdateien des Anonymisierungsdienstes.`);
            return;
        }

        if (!isRetry && (errorMsg.includes('scanned') || errorMsg.includes('OCR'))) {
            const handled = await handleScannedDocumentRetry(docId, button, originalText);
            if (handled) {
                return;
            }
        } else if (errorMsg.includes('503') || errorMsg.includes('unavailable')) {
            alert(`Anonymisierungsdienst nicht erreichbar

Der Anonymisierungsdienst auf dem Home-PC ist nicht erreichbar.

Bitte stellen Sie sicher, dass der Dienst läuft und Tailscale verbunden ist.`);
        } else {
            alert(`Fehler bei der Anonymisierung:

` + errorMsg);
        }
    } finally {
        // Restore button
        if (button && originalText !== null) {
            button.innerHTML = originalText;
            button.disabled = false;
        }
    }
}

async function handleScannedDocumentRetry(docId, button, originalText) {
    const wantsOcr = confirm(`Gescanntes Dokument erkannt

Dieses PDF enthält vermutlich nur gescannte Seiten ohne echten Text.

Möchten Sie jetzt eine OCR-Verarbeitung starten und danach die Anonymisierung erneut versuchen?`);

    if (!wantsOcr) {
        return true; // handled by user cancellation
    }

    try {
        if (button) {
            button.innerHTML = '🔄 OCR läuft...';
            button.disabled = true;
        }

        debugLog('anonymizeDocument: manual OCR trigger', docId);
        const response = await fetch(`/documents/${docId}/ocr`, {
            method: 'POST'
        });
        let data = {};
        try {
            data = await response.json();
        } catch (parseError) {
            debugError('anonymizeDocument: OCR response parse error', parseError);
        }

        if (!response.ok) {
            throw new Error(data.detail || 'OCR fehlgeschlagen. Bitte prüfen Sie den OCR-Dienst.');
        }

        debugLog('anonymizeDocument: OCR completed', data);
        const textLength = data.text_length ? data.text_length.toLocaleString('de-DE') : 'unbekannt';
        alert(`OCR abgeschlossen.\n\nExtrahierte Zeichen: ${textLength}\nAnonymisierung wird erneut gestartet.`);

        if (button && originalText !== null) {
            button.innerHTML = originalText;
            button.disabled = false;
        }

        await anonymizeDocument(docId, button, { skipConfirm: true, isRetry: true });
        return true;
    } catch (ocrError) {
        debugError('anonymizeDocument: manual OCR failed', ocrError);
        alert(`OCR-Verarbeitung fehlgeschlagen:

${ocrError.message}`);
        return true;
    }
}

function showAnonymizationResult(result) {
    debugLog('showAnonymizationResult', result);

    // Create modal overlay
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0,0,0,0.5); display: flex; align-items: center;
        justify-content: center; z-index: 1000;
    `;

    // Format plaintiff names
    const namesHtml = result.plaintiff_names && result.plaintiff_names.length > 0
        ? result.plaintiff_names.join(', ')
        : '<em>Keine Namen gefunden</em>';

    // Format confidence indicator
    const confidencePercent = (result.confidence * 100).toFixed(1);
    const confidenceColor = result.confidence < 0.7 ? '#E74C3C' : '#27AE60';
    const confidenceIcon = result.confidence < 0.7
        ? '⚠️ Niedrig - Manuelle Überprüfung empfohlen'
        : '✓ Gut';

    // Cached indicator
    const cachedBadge = result.cached
        ? '<span style="font-size: 14px; color: #888; margin-left: 10px;">(aus Cache)</span>'
        : '';

    // Create modal content
    modal.innerHTML = `
        <div style="background: white; padding: 30px; border-radius: 8px;
                    max-width: 900px; max-height: 85vh; overflow-y: auto;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
            <h2 style="margin-top: 0; color: #333;">
                Anonymisiertes Dokument
                ${cachedBadge}
            </h2>

            <div style="background: #f0f7ff; padding: 15px; border-radius: 6px;
                        margin-bottom: 20px; border-left: 4px solid #4A90E2;">
                <p style="margin: 5px 0;">
                    <strong>Gefundene Namen:</strong>
                    ${namesHtml}
                </p>
                <p style="margin: 5px 0;">
                    <strong>Konfidenz:</strong>
                    ${confidencePercent}%
                    <span style="color: ${confidenceColor};">${confidenceIcon}</span>
                </p>
                ${result.ocr_used ? `
                    <p style="margin: 5px 0; color: #4A90E2;">
                        <strong>📄 OCR verwendet:</strong> Ja (Gescanntes Dokument)
                    </p>
                ` : ''}
            </div>

            <div style="background: #f9f9f9; padding: 20px; border-radius: 6px;
                        border: 1px solid #ddd; max-height: 500px; overflow-y: auto;">
                <h3 style="margin-top: 0; color: #555;">Anonymisierter Text:</h3>
                <div style="white-space: pre-wrap; font-family: 'Courier New', monospace;
                            font-size: 13px; line-height: 1.6; color: #333;">
                    ${escapeHtml(result.anonymized_text)}
                </div>
            </div>

            <div style="margin-top: 20px; text-align: right;">
                <button onclick="this.closest('div').parentElement.remove()"
                        style="padding: 10px 25px; background-color: #4A90E2;
                               color: white; border: none; border-radius: 5px;
                               cursor: pointer; font-size: 14px;">
                    Schließen
                </button>
            </div>
        </div>
    `;

    // Add to page
    document.body.appendChild(modal);

    // Close on background click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.remove();
        }
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function escapeAttribute(value) {
    return String(value || '')
        .replace(/&/g, '&amp;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function escapeForTemplate(value) {
    // Escape string for safe use in template literals
    return escapeHtml(String(value || ''))
        .replace(/\n/g, '<br>')
        .replace(/\r/g, '');
}

// Load asyl.net suggestions
const asylnetKeywordsInput = document.getElementById('asylnetKeywords');
if (asylnetKeywordsInput) {
    asylnetKeywordsInput.addEventListener('input', function () {
        const val = this.value;
        if (!val || val.length < 2) return;

        fetch(`/research/suggestions?q=${encodeURIComponent(val)}`)
            .then(r => r.json())
            .then(suggestions => {
                const datalist = document.getElementById('asylnetSuggestions');
                if (datalist) {
                    datalist.innerHTML = suggestions.map(s => `<option value="${s}">`).join('');
                }
            })
            .catch(err => console.error("Failed to load suggestions", err));
    });
}

async function generateDocument() {
    debugLog('generateDocument: starting research');
    const loadingDiv = document.getElementById('researchLoading');
    const outputDiv = document.getElementById('outputDescription');
    const searchEngineSelect = document.getElementById('searchEngineSelect');
    const asylnetKeywordsInput = document.getElementById('asylnetKeywords');

    // Hide previous results
    const resultsContainer = document.getElementById('researchResults');
    if (resultsContainer) resultsContainer.innerHTML = '';

    if (!outputDiv || !searchEngineSelect) return;

    const query = outputDiv.value.trim();
    const searchEngine = searchEngineSelect.value;
    const manualKeywords = asylnetKeywordsInput ? asylnetKeywordsInput.value.trim() : null;
    const researchSelect = document.getElementById('researchDocumentSelect');
    // Deprecated: const referenceDocId = researchSelect ? researchSelect.value : null;

    // Use standard selection payload
    const selectedDocuments = getSelectedDocumentsPayload();

    // Check if any relevant document is selected for context
    const hasContext = selectedDocuments.bescheid.primary ||
        selectedDocuments.vorinstanz.primary ||
        (selectedDocuments.rechtsprechung && selectedDocuments.rechtsprechung.length > 0) ||
        (selectedDocuments.akte && selectedDocuments.akte.length > 0);

    if (!query && !hasContext) {
        alert('Bitte geben Sie eine Rechercheanfrage ein oder wählen Sie mindestens ein Dokument aus (Bescheid, Urteil, etc.).');
        return;
    }

    loadingDiv.style.display = 'block';

    const payload = {
        query: query,
        selected_documents: selectedDocuments,
        primary_bescheid: null, // Legacy field
        reference_document_id: null, // Legacy field
        search_engine: searchEngine,
        asylnet_keywords: manualKeywords
    };

    debugLog('generateDocument: sending request', payload);

    try {
        const response = await fetch('/research', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        loadingDiv.style.display = 'none';

        if (response.ok) {
            debugLog('generateDocument: research successful');
            displayResearchResults(data);
        } else {
            console.error('Research failed', data);
            alert(`❌ Fehler: ${data.detail || 'Recherche fehlgeschlagen'}`);
        }
    } catch (error) {
        loadingDiv.style.display = 'none';
        showError('Recherche', error);
    }
}

async function createDraft() {
    debugLog('createDraft: start');
    const textarea = document.getElementById('draftInstructions');
    const documentTypeSelect = document.getElementById('documentTypeSelect');
    const modelSelect = document.getElementById('modelSelect');
    const verbositySelect = document.getElementById('verbositySelect');
    const userPrompt = (textarea?.value || '').trim();
    const documentType = documentTypeSelect ? documentTypeSelect.value : 'Klagebegründung';
    const model = modelSelect ? modelSelect.value : 'claude-sonnet-4-5';
    const verbosity = verbositySelect ? verbositySelect.value : 'high';
    const legalArea = getLegalArea();

    if (!userPrompt) {
        debugLog('createDraft: user prompt missing');
        alert('Bitte geben Sie Anweisungen für den Entwurf ein.');
        return;
    }

    const needsBescheid = documentType.toLowerCase().includes('klage');
    if (needsBescheid && !validatePrimaryBescheid()) {
        alert('Bitte wählen Sie einen Hauptbescheid (Anlage K2) aus.');
        return;
    }

    const payload = getSelectedDocumentsPayload();

    const evt = typeof event !== 'undefined' ? event : null;
    const button = evt?.target || null;
    const originalText = button ? button.textContent : null;
    if (button) {
        button.disabled = true;
        button.textContent = '✍️ Generiere Entwurf...';
    }
    debugLog('createDraft: sending POST /generate', { documentType, model, verbosity, legalArea, payload });

    // Initialize empty draft object for UI reference
    const modalKey = `draft-${Date.now()}`;
    const requestPayload = {
        document_type: documentType,
        user_prompt: userPrompt,
        legal_area: legalArea,
        selected_documents: payload,
        model: model,
        verbosity: verbosity,
        chat_history: [] // Not supported in createDraft yet? Or inferred?
    };

    const initialData = {
        document_type: documentType,
        user_prompt: userPrompt,
        used_documents: [], // Will be populated from stream metadata later
        generated_text: "",
        thinking_text: "",
        metadata: { token_usage: {} } // Initial structure
    };
    initialData._requestPayload = requestPayload;

    // Create UI immediately
    await displayDraft(initialData, modalKey);
    const modal = document.querySelector(`[data-modal-key="${modalKey}"]`);

    // Find output elements in the modal to update incrementally
    // Note: displayDraft creates the structure. We need to find the specific containers.
    // displayDraft returns void, but we can query by ID or class inside the modal.
    // For now, let's assume we can re-render the content area or append.
    // Actually, appending usage is tricky if we re-render Markdown.

    // Let's modify the streaming loop to accumulate text and update the innerHTML of markdownContainer.
    // To prevent markdown flickering, maybe update every X chunks or just use textContent if plain? 
    // Models return markdown, so we need `marked.parse`. updating innerHTML with `marked.parse(accumulated)` is fine.

    // For thinking:
    // If thinking section doesn't exist (because initially empty), displayDraft might simply omit it?
    // In `displayDraft`: `const thinkingHtml = thinkingText ? ... : '';`
    // So if we start with empty thinking, the container won't exist.
    // We should initialize `thinking_text` with a placeholder if we expect thinking?
    // Or simpler: Re-call `displayDraft`? No, that destroys input fields and state.
    // Better: Update `displayDraft` to create the container even if empty (hidden), OR handle it here by injecting.

    // Let's try to inject the thinking container if missing.
    // But `displayDraft` currently creates strings.

    // NOTE: I will rely on `displayDraft` creating the modal structure. 
    // I need to ensure `displayDraft` handles "partial" updates or I manipulate DOM directly.
    // Direct DOM manipulation is best for streaming.

    // Find output elements in the modal to update incrementally
    // Note: displayDraft creates the structure. We need to find the specific containers.
    // displayDraft returns void, but we can query by ID or class inside the modal.
    const markdownContainer = modal.querySelector('.markdown-content');
    const thinkingContainer = modal.querySelector('details div');
    const thinkingSummary = modal.querySelector('details summary');
    const statsContainer = modal.querySelector('.token-stats-container'); // Need to add this class or find by style? 
    // displayDraft doesn't add neat classes everywhere, so we might need to adjust displayDraft OR query robustly.
    // For now, let's assume we can re-render the content area or append.
    // Actually, appending usage is tricky if we re-render Markdown.

    // Let's modify the streaming loop to accumulate text and update the innerHTML of markdownContainer.
    // To prevent markdown flickering, maybe update every X chunks or just use textContent if plain? 
    // Models return markdown, so we need `marked.parse`. updating innerHTML with `marked.parse(accumulated)` is fine.

    // For thinking:
    // If thinking section doesn't exist (because initially empty), displayDraft might simply omit it?
    // In `displayDraft`: `const thinkingHtml = thinkingText ? ... : '';`
    // So if we start with empty thinking, the container won't exist.
    // We should initialize `thinking_text` with a placeholder if we expect thinking?
    // Or simpler: Re-call `displayDraft`? No, that destroys input fields and state.
    // Better: Update `displayDraft` to create the container even if empty (hidden), OR handle it here by injecting.

    // Let's try to inject the thinking container if missing.
    // But `displayDraft` currently creates strings.

    // NOTE: I will rely on `displayDraft` creating the modal structure. 
    // I need to ensure `displayDraft` handles "partial" updates or I manipulate DOM directly.
    // Direct DOM manipulation is best for streaming.

    try {
        const response = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestPayload)
        });

        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`Server error: ${response.status} ${errText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let done = false;

        // Accumulators
        let fullThinking = "";
        let fullText = "";

        while (!done) {
            const { value, done: readerDone } = await reader.read();
            if (readerDone) {
                done = true;
                break;
            }

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const data = JSON.parse(line);

                    if (data.type === 'hearing' || data.type === 'thinking') { // Support 'hearing' alias just in case
                        const text = data.text || "";
                        fullThinking += text;
                        if (thinkingContainer) {
                            thinkingContainer.textContent = fullThinking;
                            thinkingContainer.parentElement.style.display = 'block'; // Ensure visible
                            // Update summary count
                            if (thinkingSummary) {
                                thinkingSummary.textContent = `🧠 Claude's Denkprozess (${fullThinking.length.toLocaleString()} Zeichen)`;
                            }
                        } else {
                            // Lazily handle missing container? 
                            // Ideally update `displayDraft` to always render the details block but hidden.
                            // For now, if missing, we skip visual update until final refresh?
                        }
                    } else if (data.type === 'text') {
                        const text = data.text || "";
                        fullText += text;
                        if (markdownContainer) {
                            // Render markdown
                            if (typeof marked !== 'undefined' && marked.parse) {
                                markdownContainer.innerHTML = marked.parse(fullText);
                            } else {
                                markdownContainer.textContent = fullText;
                            }
                        }
                    } else if (data.type === 'usage' || data.type === 'metadata') {
                        // usage data from stream
                        if (data.type === 'usage') {
                            initialData.metadata.token_usage = data.data; // Update data object
                        } else {
                            // Full metadata update
                            Object.assign(initialData.metadata, data.data);
                        }
                        // Update stats UI if possible? 
                        // The stats panel logic is in `displayDraft`. 
                        // We might need to refresh just that part.
                    } else if (data.type === 'done') {
                        console.log("Stream done", data.draft_id);
                        initialData.id = data.draft_id;
                    } else if (data.type === 'error') {
                        throw new Error(data.message);
                    }
                } catch (e) {
                    console.warn("Error parsing stream line", e, line);
                }
            }
        }

        // Finalize: Ensure everything is updated and saved state is correct
        initialData.generated_text = fullText;
        initialData.thinking_text = fullThinking;

        // Persist chat history for future ameliorations
        const chatHistory = [];
        if (requestPayload.user_prompt) {
            chatHistory.push({ role: 'user', content: requestPayload.user_prompt });
        }
        if (fullText) {
            chatHistory.push({ role: 'assistant', content: fullText });
        }
        requestPayload.chat_history = chatHistory;
        initialData._requestPayload = requestPayload;

        // Update the global drafts store
        if (window.generatedDrafts) {
            window.generatedDrafts[modalKey] = initialData;
        }

        // Just calling displayDraft again might interrupt user selection/scroll?
        // But it updates the metadata/citations sections which we couldn't easily stream.
        // Let's call displayDraft one last time to sanitize/finalize UI components like "Warnings".
        // To avoid replacing the whole modal (and losing scroll), we might want to target specific parts.
        // But `displayDraft` replaces `content.innerHTML`.
        // If we want to be safe, we can just leave it as is if it looks good, OR re-render.
        // Re-rendering is safer for "citations" and "warnings" which only come at the end.
        closeDraftModal(modal); // Close the temp one?
        displayDraft(initialData, modalKey); // Re-open (this might flicker)

        debugLog('createDraft: generation successful');
    } catch (error) {
        showError('createDraft', error);
        if (modal) closeDraftModal(modal); // Close partial modal on error
    } finally {
        debugLog('createDraft: restoring button state');
        if (button) {
            button.disabled = false;
            button.textContent = originalText;
        }
    }
}

async function displayDraft(data, overrideModalKey = null) {
    debugLog('displayDraft: rendering draft modal', data);
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 1000;';

    const content = document.createElement('div');
    content.style.cssText = 'background: white; padding: 30px; border-radius: 10px; max-width: 900px; max-height: 85vh; overflow-y: auto; box-shadow: 0 4px 6px rgba(0,0,0,0.1);';

    const modalKey = overrideModalKey || `draft-${Date.now()}`;
    window.generatedDrafts = window.generatedDrafts || {};
    window.generatedDrafts[modalKey] = data;

    // Helper function to copy draft text
    window.copyDraftText = function (key) {
        const draft = window.generatedDrafts[key];
        if (draft && draft.generated_text) {
            navigator.clipboard.writeText(draft.generated_text)
                .then(() => console.log('Text copied to clipboard'))
                .catch(err => console.error('Failed to copy text:', err));
        }
    };

    const usedDocuments = Array.isArray(data.used_documents) ? data.used_documents : [];
    const friendlyCategories = {
        anhoerung: 'Anhörung',
        bescheid: 'Bescheid',
        rechtsprechung: 'Rechtsprechung',
        saved_sources: 'Gespeicherte Quelle'
    };
    let usedHtml = '';
    if (usedDocuments.length > 0) {
        usedHtml = '<h3 style="margin-top: 20px; color: #2c3e50;">🗂️ Verwendete Dokumente:</h3><ul style="list-style: none; padding: 0;">';
        usedDocuments.forEach(doc => {
            const parts = [];
            if (doc.category) parts.push(friendlyCategories[doc.category] || escapeHtml(doc.category));
            if (doc.role) parts.push(doc.role === 'primary' ? 'Anlage K2' : escapeHtml(doc.role));
            const meta = parts.length ? ` <span style="color: #7f8c8d; font-size: 12px;">(${parts.join(' · ')})</span>` : '';
            usedHtml += `<li style="margin: 6px 0; color: #2c3e50;">${escapeHtml(doc.filename || 'Unbenannt')}${meta}</li>`;
        });
        usedHtml += '</ul>';
    }

    const metadata = data.metadata || {};
    const warnings = Array.isArray(metadata.warnings) ? metadata.warnings : [];
    const missing = Array.isArray(metadata.missing_citations) ? metadata.missing_citations : [];
    const wordCount = metadata.word_count != null ? metadata.word_count : '-';
    const tokenCount = metadata.token_count != null ? metadata.token_count : '-';
    const citationsFound = metadata.citations_found != null ? metadata.citations_found : 0;
    const generatedText = data.generated_text || '(Kein Text erzeugt)';

    // Render markdown if marked library is available, otherwise fall back to plain text
    const renderedContent = (typeof marked !== 'undefined' && marked.parse)
        ? marked.parse(generatedText)
        : `<pre style="white-space: pre-wrap;">${escapeHtml(generatedText)}</pre>`;

    const templateOptions = await ensureJLawyerTemplates();
    const defaultTemplateName = templateOptions.length > 0 ? templateOptions[0] : 'Klagebegründung_Vorlage.odt';
    const defaultFileName = `${(data.document_type || 'Klagebegründung').replace(/\s+/g, '_')}_${new Date().toISOString().slice(0, 10)}.odt`;
    const caseInputId = `jlawyer-case-${modalKey}`;
    const templateInputId = `jlawyer-template-${modalKey}`;
    const templateSelectId = `jlawyer-template-select-${modalKey}`;
    const fileInputId = `jlawyer-file-${modalKey}`;
    const statusId = `jlawyer-status-${modalKey}`;

    const templateSelectHtml = templateOptions.length > 0
        ? `
            <label style="font-size: 13px; color: #2c3e50;">Template auswählen
                <select id="${templateSelectId}"
                        style="width: 100%; margin-top: 4px; padding: 8px 10px; border: 1px solid #bdc3c7; border-radius: 5px;">
                    <option value="">-- Template wählen --</option>
                    ${templateOptions.map((name, idx) => {
            const escaped = escapeAttribute(name);
            const label = escapeHtml(name);
            const selected = idx === 0 ? 'selected' : '';
            return `<option value="${escaped}" ${selected}>${label}</option>`;
        }).join('')}
                </select>
            </label>
            <label style="font-size: 13px; color: #2c3e50;">Alternativer Template-Name (optional)
                <input id="${templateInputId}" type="text" placeholder="Eigenes Template"
                       style="width: 100%; margin-top: 4px; padding: 8px 10px; border: 1px solid #bdc3c7; border-radius: 5px;" />
            </label>
        `
        : `
            <label style="font-size: 13px; color: #2c3e50;">Template-Name (ODT)
                <input id="${templateInputId}" type="text" value="${escapeAttribute(defaultTemplateName)}"
                       style="width: 100%; margin-top: 4px; padding: 8px 10px; border: 1px solid #bdc3c7; border-radius: 5px;" />
            </label>
        `;

    const warningsHtml = warnings.length
        ? `
            <div style="background: #fff5e6; border: 1px solid #f5c16c; padding: 12px; border-radius: 6px; margin-top: 16px;">
                <strong style="color: #d35400;">Hinweise:</strong>
                <ul style="margin: 8px 0 0 16px; color: #d35400; font-size: 14px;">
                    ${warnings.map(w => `<li>${escapeHtml(w)}</li>`).join('')}
                </ul>
            </div>
        ` : '';

    const missingHtml = missing.length
        ? `
            <div style="background: #fdecea; border: 1px solid #f5a8a4; padding: 12px; border-radius: 6px; margin-top: 16px;">
                <strong style="color: #c0392b;">Nicht zitierte Dokumente:</strong>
                <ul style="margin: 8px 0 0 16px; color: #c0392b; font-size: 14px;">
                    ${missing.map(m => `<li>${escapeHtml(m)}</li>`).join('')}
                </ul>
            </div>
        ` : '';

    // Claude extended thinking display (collapsible)
    const thinkingText = data.thinking_text != null ? data.thinking_text : '';
    // Always render, but hide if empty initially (streaming will unhide it)
    const displayStyle = thinkingText ? 'block' : 'none';
    const thinkingHtml = `
        <details style="margin-top: 16px; background: #f3e5f5; border: 1px solid #ce93d8; border-radius: 6px; padding: 12px; display: ${displayStyle}">
            <summary style="cursor: pointer; font-weight: 600; color: #7b1fa2;">🧠 Claude's Denkprozess (${thinkingText.length.toLocaleString()} Zeichen)</summary>
            <div style="margin-top: 12px; padding: 12px; background: white; border-radius: 4px; font-size: 13px; line-height: 1.5; white-space: pre-wrap; max-height: 400px; overflow-y: auto;">
                ${escapeHtml(thinkingText)}
            </div>
        </details>
    `;

    // Build token usage display (detailed if available)
    const tokenUsage = metadata.token_usage;
    let statsHtml = '';
    if (tokenUsage && tokenUsage.input_tokens) {
        const costFormatted = tokenUsage.cost_usd != null
            ? `$${tokenUsage.cost_usd.toFixed(4)}`
            : '-';
        statsHtml = `
            <div style="margin-bottom: 12px; padding: 12px; background: #e3f2fd; border-radius: 6px; font-size: 13px;">
                <strong style="color: #1565c0;">📊 Token Usage:</strong>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; margin-top: 8px;">
                    <span title="Input tokens">📥 Input: ${tokenUsage.input_tokens.toLocaleString()}</span>
                    <span title="Output tokens (includes thinking)">📤 Output: ${tokenUsage.output_tokens.toLocaleString()}</span>
                    ${tokenUsage.cache_read_tokens > 0 ? `<span title="Cache read tokens">💾 Cache Read: ${tokenUsage.cache_read_tokens.toLocaleString()}</span>` : ''}
                    ${tokenUsage.cache_write_tokens > 0 ? `<span title="Cache write tokens">📝 Cache Write: ${tokenUsage.cache_write_tokens.toLocaleString()}</span>` : ''}
                </div>
                <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #bbdefb;">
                    <strong>Gesamt:</strong> ${tokenUsage.total_tokens.toLocaleString()} Token · <strong>Kosten:</strong> ${costFormatted}
                    ${tokenUsage.model ? ` <span style="color: #7f8c8d;">(${tokenUsage.model})</span>` : ''}
                </div>
            </div>
        `;
    } else {
        // Fallback to simple display
        statsHtml = `
            <div style="margin-bottom: 12px; color: #34495e; font-size: 13px;">
                <strong>Statistik:</strong> ${citationsFound} Zitate · ${wordCount} Wörter · ${tokenCount} Token
            </div>
        `;
    }

    content.innerHTML = `
        <h2 style="color: #2c3e50; margin-bottom: 15px;">✍️ ${escapeHtml(data.document_type || 'Entwurf')}</h2>
        <div style="background: #eaf7ec; padding: 12px; border-radius: 5px; margin-bottom: 12px;">
            <strong>Aufgabenstellung:</strong> ${escapeForTemplate(data.user_prompt || '—')}
        </div>
        ${statsHtml}
        <div style="line-height: 1.6; background: #f8f9fa; padding: 16px; border-radius: 6px; border: 1px solid #e1e4e8;" class="markdown-content">${renderedContent}</div>
        ${usedHtml}
        ${warningsHtml}
        ${missingHtml}
        ${thinkingHtml}
        <div style="margin-top: 24px; padding: 18px; border-radius: 8px; border: 1px solid #dfe6e9; background: #fff8e1;">
            <h3 style="margin-top: 0; margin-bottom: 12px; color: #d35400;">✨ Interaktive Verbesserung</h3>
            <p style="font-size: 13px; color: #7f8c8d; margin-bottom: 10px;">
                Geben Sie Änderungswünsche ein, um den Text zu überarbeiten (z.B. "Füge Argument X hinzu" oder "Kürze den zweiten Absatz").
            </p>
            <textarea id="amelioration-prompt-${modalKey}"
                      placeholder="Was soll verbessert werden?"
                      style="width: 100%; min-height: 80px; padding: 10px; border: 1px solid #bdc3c7; border-radius: 5px; font-family: Arial; margin-bottom: 10px;"></textarea>
            <button onclick="ameliorateDraft('${modalKey}', this)"
                    style="background: #e67e22; color: white; border: none; padding: 8px 16px; border-radius: 5px; cursor: pointer;">
                🔄 Text überarbeiten
            </button>
        </div>
        <div style="margin-top: 24px; padding: 18px; border-radius: 8px; border: 1px solid #dfe6e9; background: #fbfcfd;">
            <h3 style="margin-top: 0; margin-bottom: 12px; color: #2c3e50;">📨 An j-lawyer senden</h3>
            <div style="display: flex; flex-direction: column; gap: 10px;">
                <label style="font-size: 13px; color: #2c3e50;">Aktenzeichen (case_id)
                    <input id="${caseInputId}" type="text" placeholder="z.B. 12345-2024"
                           style="width: 100%; margin-top: 4px; padding: 8px 10px; border: 1px solid #bdc3c7; border-radius: 5px;" />
                </label>
                ${templateSelectHtml}
                <label style="font-size: 13px; color: #2c3e50;">Dateiname (Ergebnis)
                    <input id="${fileInputId}" type="text" value="${escapeAttribute(defaultFileName)}"
                           style="width: 100%; margin-top: 4px; padding: 8px 10px; border: 1px solid #bdc3c7; border-radius: 5px;" />
                </label>
                <div id="${statusId}" style="font-size: 12px; color: #7f8c8d;"></div>
                <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                    <button onclick="copyDraftText('${modalKey}')"
                            style="background: #2ecc71; color: white; border: none; padding: 10px 14px; border-radius: 5px; cursor: pointer;">Kopieren</button>
                    <button onclick="sendDraftToJLawyer('${modalKey}', this)"
                            style="background: #e67e22; color: white; border: none; padding: 10px 14px; border-radius: 5px; cursor: pointer;">An j-lawyer senden</button>
                    <button onclick="closeDraftModal(this)"
                            style="background: #3498db; color: white; border: none; padding: 10px 14px; border-radius: 5px; cursor: pointer;">Schließen</button>
                </div>
            </div>
        </div>
    `;

    const wrapper = document.createElement('div');
    wrapper.className = 'modal-content-wrapper';
    wrapper.appendChild(content);

    modal.dataset.modalKey = modalKey;
    modal.appendChild(wrapper);
    document.body.appendChild(modal);
    modal.onclick = (e) => {
        if (e.target === modal) {
            closeDraftModal(modal);
        }
    };
}

function closeDraftModal(trigger) {
    const modal = trigger && trigger.classList && trigger.classList.contains('modal-overlay')
        ? trigger
        : trigger?.closest?.('.modal-overlay');
    if (!modal) return;
    const key = modal.dataset?.modalKey;
    if (key && window.generatedDrafts) {
        delete window.generatedDrafts[key];
    }
    modal.remove();
}

async function sendDraftToJLawyer(modalKey, button) {
    debugLog('sendDraftToJLawyer: start', { modalKey });
    const drafts = window.generatedDrafts || {};
    const draft = drafts[modalKey];
    if (!draft) {
        alert('❌ Kein Entwurf verfügbar. Bitte generieren Sie zuerst den Text.');
        return;
    }

    const caseInput = document.getElementById(`jlawyer-case-${modalKey}`);
    const templateSelect = document.getElementById(`jlawyer-template-select-${modalKey}`);
    const templateInput = document.getElementById(`jlawyer-template-${modalKey}`);
    const fileInput = document.getElementById(`jlawyer-file-${modalKey}`);
    const statusEl = document.getElementById(`jlawyer-status-${modalKey}`);

    const caseId = (caseInput?.value || '').trim();
    const templateName = ((templateSelect && templateSelect.value) || (templateInput?.value || '')).trim();
    const fileName = (fileInput?.value || '').trim();

    if (!caseId || !templateName || !fileName) {
        alert('Bitte füllen Sie Aktenzeichen, Template und Dateinamen aus.');
        return;
    }

    if (statusEl) {
        statusEl.style.color = '#7f8c8d';
        statusEl.textContent = 'Sende an j-lawyer...';
    }

    if (button) {
        button.disabled = true;
        button.textContent = '📨 Sende...';
    }

    try {
        const response = await fetch('/send-to-jlawyer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                case_id: caseId,
                template_name: templateName,
                file_name: fileName,
                generated_text: draft.generated_text || draft.draft_text || ''
            })
        });
        const data = await response.json();
        debugLog('sendDraftToJLawyer: response', { status: response.status, data });

        if (response.ok && data.success) {
            if (statusEl) {
                statusEl.style.color = '#27ae60';
                statusEl.textContent = data.message || 'Erfolgreich an j-lawyer gesendet.';
            }
            if (button) {
                button.textContent = '✅ Gesendet';
            }
        } else {
            const detail = Array.isArray(data.detail)
                ? data.detail.map(item => item.msg || item).join('\n')
                : (data.detail || data.message || 'Versand fehlgeschlagen');
            if (statusEl) {
                statusEl.style.color = '#c0392b';
                statusEl.textContent = `Fehler: ${detail}`;
            }
            if (button) {
                button.disabled = false;
                button.textContent = 'An j-lawyer senden';
            }
        }
    } catch (error) {
        debugError('sendDraftToJLawyer: request error', error);
        if (statusEl) {
            statusEl.style.color = '#c0392b';
            statusEl.textContent = `Fehler: ${error.message}`;
        }
        if (button) {
            button.disabled = false;
            button.textContent = 'An j-lawyer senden';
        }
    }
}

async function addDocumentFromSource(evt, index) {
    const sources = window.latestResearchSources || [];
    const source = sources[index];
    if (!source) {
        debugError('addDocumentFromSource: no source found', { index });
        return;
    }

    const button = evt?.target;
    if (button) {
        button.disabled = true;
        button.textContent = '⏳ ...';
    }

    try {
        const response = await fetch('/documents/from-url', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                title: source.title,
                url: source.pdf_url || source.url,
                category: 'Rechtsprechung',
                auto_download: true
            })
        });

        const data = await response.json();

        if (response.ok) {
            alert('✅ Dokument zu "Rechtsprechung" hinzugefügt.');
            if (button) {
                button.textContent = '✅ Hinzugefügt';
                button.disabled = true;
            }
            // Document list will auto-update
        } else {
            console.error('Add document failed', data);
            alert(`❌ Fehler: ${data.detail || 'Konnte nicht hinzugefügt werden'}`);
            if (button) {
                button.textContent = '❌ Fehler';
                button.disabled = false;
            }
        }
    } catch (e) {
        console.error('Add document error', e);
        alert('❌ Fehler beim Hinzufügen');
        if (button) {
            button.textContent = '❌ Fehler';
            button.disabled = false;
        }
    }
}

function displayResearchResults(data) {
    debugLog('displayResearchResults: showing results', { query: data.query, sourceCount: (data.sources || []).length });
    const incomingSources = Array.isArray(data.sources) ? data.sources : [];
    console.log('DEBUG: First source:', incomingSources[0]);
    window.latestResearchSources = [];
    window.latestResearchQuery = data.query || '';
    const modal = document.createElement('div');
    modal.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 1000;';

    const content = document.createElement('div');
    content.style.cssText = 'background: white; padding: 30px; border-radius: 10px; max-width: 900px; max-height: 85vh; overflow-y: auto; box-shadow: 0 4px 6px rgba(0,0,0,0.1);';

    let sourcesHtml = '';
    if (incomingSources.length > 0) {
        console.log('DEBUG: Entering sources rendering, source count:', incomingSources.length);
        sourcesHtml = '<h3 style="margin-top: 20px; color: #2c3e50;">📚 Relevante Quellen:</h3>';
        sourcesHtml += '<div style="color: #7f8c8d; font-size: 13px; margin-bottom: 12px;">💾 Hinweis: Hochwertige Quellen werden automatisch als PDF gespeichert und erscheinen in "Gespeicherte Quellen"</div>';

        // Group sources by origin
        const sourceGroups = {
            'Grok': [],
            'Gemini': [],
            'asyl.net': [],
            'Gesetzestext': []
        };

        const knownOrigins = new Set(Object.keys(sourceGroups));
        incomingSources.forEach(source => {
            const origin = (source.source || 'Gemini').trim();
            console.log('Source origin:', origin, 'for source:', source.title);
            const targetGroup = knownOrigins.has(origin) ? origin : 'Gemini';
            if (!knownOrigins.has(origin)) {
                console.warn('Unknown source origin:', origin, '- defaulting to Gemini');
            }
            sourceGroups[targetGroup].push(source);
        });

        console.log('Grouped sources:', Object.entries(sourceGroups).map(([k, v]) => `${k}: ${v.length}`).join(', '));

        const groupMeta = {
            'Grok': { emoji: '🤖', label: 'Grok' },
            'Gemini': { emoji: '✨', label: 'Gemini' },
            'asyl.net': { emoji: '⚖️', label: 'asyl.net' },
            'Gesetzestext': { emoji: '📜', label: 'Gesetzestexte' }
        };

        const orderedSources = [];

        // Display each group with header
        Object.entries(sourceGroups).forEach(([groupName, sources]) => {
            if (sources.length === 0) return;

            // Add group header
            const meta = groupMeta[groupName] || { emoji: '🧭', label: groupName };
            const headerEmoji = meta.emoji;
            sourcesHtml += `<div style="margin-top: 25px; margin-bottom: 15px;">
                <h4 style="color: #34495e; font-size: 16px; font-weight: 600; border-bottom: 2px solid #3498db; padding-bottom: 8px;">
                    ${headerEmoji} ${meta.label} (${sources.length})
                </h4>
            </div>`;
            sourcesHtml += '<div style="display: flex; flex-direction: column; gap: 15px;">';

            sources.forEach((source) => {
                const index = orderedSources.length;
                orderedSources.push(source);

                const description = escapeForTemplate(source.description || 'Relevante Quelle für Ihre Recherche');
                const addButton = `<button onclick="addSourceFromResults(event, ${index})" style="display: inline-block; background: #27ae60; color: white; padding: 6px 12px; border-radius: 4px; border: none; cursor: pointer; font-size: 13px; font-weight: 500;">➕ Zu gespeicherten Quellen</button>`;
                const addDocButton = `<button onclick="addDocumentFromSource(event, ${index})" style="display: inline-block; background: #8e44ad; color: white; padding: 6px 12px; border-radius: 4px; border: none; cursor: pointer; font-size: 13px; font-weight: 500; margin-left: 8px;">➕ Zu Rechtsprechung</button>`;
                const pdfLink = source.pdf_url || (source.url && source.url.toLowerCase().endsWith('.pdf') ? source.url : null);
                if (pdfLink && !source.pdf_url) {
                    source.pdf_url = pdfLink;
                }
                const pdfButton = pdfLink
                    ? `<a href="${escapeAttribute(pdfLink)}" target="_blank" style="display: inline-block; background: #2ecc71; color: white; padding: 6px 12px; border-radius: 4px; text-decoration: none; font-size: 13px; font-weight: 500;">📄 PDF öffnen</a>`
                    : '';
                const pdfBadge = pdfLink ? '<span style="color: #2ecc71; font-size: 12px; font-weight: 600; margin-left: 8px;">📄 PDF erkannt</span>' : '';
                const safeUrl = escapeAttribute(source.url);
                const displayUrl = escapeHtml(source.url.length > 50 ? source.url.substring(0, 50) + '...' : source.url);
                sourcesHtml += `
                <div style="padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #3498db;">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px;">
                        <a href="${safeUrl}" target="_blank" style="color: #2c3e50; text-decoration: none; font-weight: 600; font-size: 15px; flex: 1;">
                            ${index + 1}. ${escapeHtml(source.title)}
                        </a>
                        ${pdfBadge}
                    </div>
                    <p style="color: #555; margin: 8px 0 10px 0; line-height: 1.5; font-size: 14px;">${description}</p>
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <a href="${safeUrl}" target="_blank"
                           style="display: inline-block; background: #3498db; color: white; padding: 6px 12px; border-radius: 4px; text-decoration: none; font-size: 13px; font-weight: 500;">
                            🔗 Zur Quelle
                        </a>
                        ${pdfButton}
                        ${addButton}
                        ${addDocButton}
                        <span style="color: #7f8c8d; font-size: 12px;">
                            ${displayUrl}
                        </span>
                    </div>
                </div>
            `;
            });

            sourcesHtml += '</div>';
        });

        window.latestResearchSources = orderedSources;
    } else {
        sourcesHtml = '<p style="color: #7f8c8d; margin-top: 20px;">Keine Quellen gefunden.</p>';
    }

    const summaryHtml = data.summary || '';

    content.innerHTML = `
        <h2 style="color: #2c3e50; margin-bottom: 15px;">🔍 Rechercheergebnisse</h2>
        <div style="background: #e3f2fd; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
            <strong>Ihre Anfrage:</strong> ${escapeForTemplate(data.query)}
        </div>
        ${summaryHtml ? `<div style=\"margin-bottom: 20px;\"><strong>Zusammenfassung:</strong><div style=\"margin-top: 10px; line-height: 1.6;\">${summaryHtml}</div></div>` : ''}
        ${sourcesHtml}
        <div style="margin-top: 20px; display: flex; gap: 10px;">
            <button onclick="this.parentElement.parentElement.parentElement.remove();"
                    style="background: #27ae60; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-weight: 500;">
                📥 Zu gespeicherten Quellen
            </button>
            <button onclick="this.parentElement.parentElement.parentElement.remove()"
                    style="background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-weight: 500;">
                Schließen
            </button>
        </div>
    `;

    modal.appendChild(content);
    document.body.appendChild(modal);

    modal.onclick = (e) => {
        if (e.target === modal) modal.remove();
    };
}

async function ameliorateDraft(modalKey, button) {
    debugLog('ameliorateDraft: start', { modalKey });
    const drafts = window.generatedDrafts || {};
    const draft = drafts[modalKey];
    if (!draft) {
        alert('❌ Kein Entwurf verfügbar.');
        return;
    }

    const promptInput = document.getElementById(`amelioration-prompt-${modalKey}`);
    const ameliorationPrompt = (promptInput?.value || '').trim();

    if (!ameliorationPrompt) {
        alert('Bitte geben Sie einen Änderungswunsch ein.');
        return;
    }

    const requestPayload = draft._requestPayload;
    if (!requestPayload) {
        alert('Fehler: Original-Anfragedaten nicht gefunden.');
        return;
    }

    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = '🔄 Überarbeite...';

    // Clone history to avoid mutating original until success
    const currentHistory = [...(requestPayload.chat_history || [])];
    // Note: We do NOT add the new prompt to history yet. It is sent as 'user_prompt'.

    console.log('DEBUG: ameliorationPrompt value:', ameliorationPrompt);
    console.log('DEBUG: requestPayload before merge:', JSON.stringify(requestPayload, null, 2));

    const newPayload = {
        ...requestPayload,
        user_prompt: ameliorationPrompt,
        chat_history: currentHistory
    };

    // Remove legacy fields if they exist
    delete newPayload.previous_generated_text;
    delete newPayload.amelioration_prompt;

    debugLog('ameliorateDraft: sending POST /generate', newPayload);
    console.log('DEBUG: Amelioration Payload:', JSON.stringify(newPayload, null, 2));

    try {
        const response = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newPayload)
        });
        debugLog('ameliorateDraft: response status', response.status);

        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`Server error: ${response.status} ${errText}`);
        }

        // Handle streaming response (same as createDraft)
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let done = false;
        let fullThinking = "";
        let fullText = "";
        let draftId = null;
        let metadata = {};

        while (!done) {
            const { value, done: readerDone } = await reader.read();
            if (readerDone) {
                done = true;
                break;
            }

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const streamData = JSON.parse(line);

                    if (streamData.type === 'thinking' || streamData.type === 'hearing') {
                        fullThinking += streamData.text || "";
                    } else if (streamData.type === 'text') {
                        fullText += streamData.text || "";
                    } else if (streamData.type === 'usage') {
                        metadata.token_usage = streamData.data;
                    } else if (streamData.type === 'metadata') {
                        Object.assign(metadata, streamData.data);
                    } else if (streamData.type === 'done') {
                        draftId = streamData.draft_id;
                    } else if (streamData.type === 'error') {
                        throw new Error(streamData.message);
                    }
                } catch (e) {
                    console.warn("Error parsing ameliorate stream line", e, line);
                }
            }
        }

        debugLog('ameliorateDraft: stream complete', { textLen: fullText.length, thinkingLen: fullThinking.length });

        // Build result data object
        const data = {
            document_type: newPayload.document_type,
            user_prompt: ameliorationPrompt,
            generated_text: fullText,
            thinking_text: fullThinking,
            used_documents: [],
            metadata: metadata,
            id: draftId
        };

        debugLog('ameliorateDraft: generation successful');
        // Update history with user prompt AND assistant response
        currentHistory.push({ role: 'user', content: ameliorationPrompt });
        currentHistory.push({ role: 'assistant', content: data.generated_text });

        // Update payload with new history
        newPayload.chat_history = currentHistory;
        data._requestPayload = newPayload;

        // Close old modal
        closeDraftModal(button);
        // Show new modal
        await displayDraft(data);
    } catch (error) {
        showError('ameliorateDraft', error);
    } finally {
        if (button) {
            button.disabled = false;
            button.textContent = originalText;
        }
    }
}

function selectAllSources() {
    console.log('selectAllSources: triggered');
    const container = document.getElementById('sources-docs');
    if (!container) {
        console.error('selectAllSources: #sources-docs container not found');
        return;
    }

    const checkboxes = Array.from(container.querySelectorAll('input[type="checkbox"]'));
    if (checkboxes.length === 0) {
        console.warn('selectAllSources: no checkboxes found');
        return;
    }

    // Check if all are currently selected
    const allSelected = checkboxes.every(cb => cb.checked);
    const targetState = !allSelected; // If all selected, we want to deselect (false). Otherwise select (true).

    console.log(`selectAllSources: allSelected=${allSelected}, targetState=${targetState}`);

    let count = 0;
    checkboxes.forEach(cb => {
        if (cb.checked !== targetState) {
            cb.checked = targetState;
            try {
                handleSavedSourceCheckboxChange(cb);
                count++;
            } catch (e) {
                console.error('selectAllSources: error updating checkbox', e);
            }
        }
    });
    console.log(`selectAllSources: updated ${count} checkboxes`);
}

function selectAllVorinstanz() {
    console.log('selectAllVorinstanz: triggered');
    const container = document.getElementById('vorinstanz-docs');
    if (!container) {
        console.error('selectAllVorinstanz: #vorinstanz-docs container not found');
        return;
    }

    const checkboxes = Array.from(container.querySelectorAll('input[type="checkbox"]'));
    if (checkboxes.length === 0) {
        console.warn('selectAllVorinstanz: no checkboxes found');
        return;
    }

    // Check if all are currently selected
    const allSelected = checkboxes.every(cb => cb.checked);
    const targetState = !allSelected;

    console.log(`selectAllVorinstanz: allSelected=${allSelected}, targetState=${targetState}`);

    let count = 0;
    checkboxes.forEach(cb => {
        if (cb.checked !== targetState) {
            cb.checked = targetState;
            // For regular documents, we use toggleDocumentSelection
            // But toggleDocumentSelection takes (categoryKey, filename, isChecked)
            // We need to extract filename from the checkbox or its parent
            const filename = decodeURIComponent(cb.getAttribute('data-filename'));
            if (filename) {
                toggleDocumentSelection('vorinstanz', filename, targetState);
                count++;
            }
        }
    });
    console.log(`selectAllVorinstanz: updated ${count} checkboxes`);
}

async function queryGemini() {
    debugLog('queryGemini: start');
    const queryInput = document.getElementById('queryInput');
    const query = queryInput.value.trim();

    if (!query) {
        alert('Bitte geben Sie eine Frage ein.');
        return;
    }

    const payload = getSelectedDocumentsPayload();
    // Validate that something is selected
    const totalDocs = payload.anhoerung.length +
        (payload.bescheid.primary ? 1 : 0) + payload.bescheid.others.length +
        (payload.vorinstanz.primary ? 1 : 0) + payload.vorinstanz.others.length +
        payload.rechtsprechung.length +
        payload.saved_sources.length +
        payload.akte.length +
        payload.sonstiges.length;

    if (totalDocs === 0) {
        alert('Bitte wählen Sie mindestens ein Dokument (oder eine Quelle) aus, das/die befragt werden soll (Häkchen unten setzen!).');
        return;
    }

    const loadingDiv = document.getElementById('queryLoading');
    const resultDiv = document.getElementById('queryAnswerResult');
    const textDiv = document.getElementById('queryAnswerText');
    const btn = document.querySelector('button[onclick="queryGemini()"]');

    resultDiv.style.display = 'block';
    loadingDiv.style.display = 'block';
    textDiv.innerHTML = '';
    if (btn) btn.disabled = true;

    try {
        debugLog('queryGemini: sending request', { query, docCount: totalDocs });

        const response = await fetch('/query-documents', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                selected_documents: payload,
                model: 'gemini-3-flash-preview'
            })
        });

        debugLog('queryGemini: response status', response.status);

        if (!response.ok) {
            const errText = await response.text();
            throw new Error(errText || 'Fehler bei der Anfrage.');
        }

        // Reading the stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let accumulatedText = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            accumulatedText += chunk;

            // Render on the fly
            let htmlContent = accumulatedText;
            if (typeof marked !== 'undefined' && marked.parse) {
                try {
                    htmlContent = marked.parse(accumulatedText);
                } catch (e) {
                    htmlContent = `<pre>${accumulatedText}</pre>`;
                }
            }
            textDiv.innerHTML = htmlContent;
        }

        debugLog('queryGemini: stream complete');

    } catch (error) {
        showError('queryGemini', error);
        textDiv.innerHTML += `<div style="color: red; margin-top: 10px;">Fehler: ${error.message}</div>`;
    } finally {
        loadingDiv.style.display = 'none';
        if (btn) btn.disabled = false;
    }
}
