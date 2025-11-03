const debugLog = (...args) => {
    const ts = new Date().toISOString();
    console.log(`[Rechtmaschine] ${ts}`, ...args);
};

const debugError = (...args) => {
    const ts = new Date().toISOString();
    console.error(`[Rechtmaschine] ${ts}`, ...args);
};

const escapeJsString = (value) => String(value)
    .split('\\\\').join('\\\\\\\\')
    .split("'").join("\\\\'")
    .split('"').join('\\\\"');

const categoryToKey = {
    'Anhörung': 'anhoerung',
    'Bescheid': 'bescheid',
    'Rechtsprechung': 'rechtsprechung',
    'Akte': 'akte',
    'Sonstiges': 'saved_sources'
};

const selectionState = {
    anhoerung: new Set(),
    bescheid: {
        primary: null,
        others: new Set()
    },
    rechtsprechung: new Set(),
    saved_sources: new Set()
};

const DOCUMENT_CONTAINER_IDS = {
    'Anhörung': 'anhoerung-docs',
    'Bescheid': 'bescheid-docs',
    'Akte': 'akte-docs',
    'Rechtsprechung': 'rechtsprechung-docs',
    'Sonstiges': 'sonstiges-docs'
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
    selectionState.rechtsprechung.clear();
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

    selectionState.bescheid.others = new Set([...selectionState.bescheid.others].filter(filename => available.bescheid.has(filename)));
    if (!available.bescheid.has(selectionState.bescheid.primary)) {
        selectionState.bescheid.primary = null;
    }

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

function toggleDocumentSelection(categoryKey, filename, isChecked) {
    if (!categoryKey || !filename) return;
    if (categoryKey === 'anhoerung') {
        if (isChecked) selectionState.anhoerung.add(filename);
        else selectionState.anhoerung.delete(filename);
    } else if (categoryKey === 'rechtsprechung') {
        if (isChecked) selectionState.rechtsprechung.add(filename);
        else selectionState.rechtsprechung.delete(filename);
    }
}

function toggleSavedSourceSelection(sourceId, isChecked) {
    if (!sourceId) return;
    if (isChecked) selectionState.saved_sources.add(sourceId);
    else selectionState.saved_sources.delete(sourceId);
}

function toggleBescheidSelection(filename, isChecked) {
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
    if (categoryKey === 'anhoerung') return selectionState.anhoerung.has(filename);
    if (categoryKey === 'rechtsprechung') return selectionState.rechtsprechung.has(filename);
    if (categoryKey === 'bescheid') {
        return selectionState.bescheid.primary === filename || selectionState.bescheid.others.has(filename);
    }
    return false;
}

function isSourceSelected(sourceId) {
    return selectionState.saved_sources.has(sourceId);
}

function getSelectedDocumentsPayload() {
    const primary = selectionState.bescheid.primary;
    const others = Array.from(selectionState.bescheid.others).filter(name => name !== primary);
    return {
        anhoerung: Array.from(selectionState.anhoerung),
        bescheid: {
            primary: primary,
            others: others
        },
        rechtsprechung: Array.from(selectionState.rechtsprechung),
        saved_sources: Array.from(selectionState.saved_sources)
    };
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
    if (!element) return;
    const encoded = element.dataset?.filename || '';
    let filename = encoded;
    try { filename = decodeURIComponent(encoded); } catch (e) { /* ignore */ }
    toggleDocumentSelection(categoryKey, filename, element.checked);
}

function handleBescheidCheckboxChange(element) {
    if (!element) return;
    const encoded = element.dataset?.filename || '';
    let filename = encoded;
    try { filename = decodeURIComponent(encoded); } catch (e) { /* ignore */ }
    toggleBescheidSelection(filename, element.checked);
}

function handlePrimaryBescheidSelect(element) {
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

    const container = document.getElementById('sonstiges-docs');
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
            const source = new EventSource('/documents/stream');
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
    pollingActive = false; // DISABLED: SSE handles all updates
    debugLog('Polling DISABLED - relying entirely on SSE');

    // Polling disabled to test SSE reliability
    // if (documentPollingTimer) {
    //     clearInterval(documentPollingTimer);
    // }
    // documentPollingTimer = setInterval(() => {
    //     const now = Date.now();
    //     const streamHealthy = documentStreamSource && (now - lastStreamMessageAt) < STREAM_STALE_THRESHOLD_MS;
    //     const streamDisabled = now < documentStreamDisabledUntil;
    //     if (!streamHealthy || streamDisabled) {
    //         loadDocuments({ placeholders: false }).catch((err) => debugError('document poll failed', err));
    //     }
    // }, DOCUMENT_POLL_INTERVAL_MS);

    // if (sourcesPollingTimer) {
    //     clearInterval(sourcesPollingTimer);
    // }
    // sourcesPollingTimer = setInterval(() => {
    //     loadSources({ placeholders: false }).catch((err) => debugError('sources poll failed', err));
    // }, SOURCES_POLL_INTERVAL_MS);
}

// Load documents and sources on page load
window.addEventListener('DOMContentLoaded', () => {
    debugLog('DOMContentLoaded: initializing interface');
    loadDocuments();
    loadSources();
    startDocumentStream();
    startAutoRefresh();
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

function renderDocuments(grouped, options) {
    const force = !!(options && options.force);
    const data = grouped && typeof grouped === 'object' ? grouped : {};

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
        const el = document.getElementById(elementId);
        if (!el) return;
        const docsArray = Array.isArray(data[category]) ? data[category] : [];
        if (category === 'Sonstiges' && docsArray.length === 0) {
            // Avoid overwriting saved sources area when no misc documents are present
            return;
        }
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
    const showAnonymizeBtn = doc.category === 'Anhörung' || doc.category === 'Bescheid';
    const isAnonymized = !!doc.anonymized;

    let selectionControls = '';
    if (categoryKey === 'anhoerung' || categoryKey === 'rechtsprechung') {
        const checked = isDocumentSelected(categoryKey, doc.filename) ? 'checked' : '';
        selectionControls = `
            <label class="selection-option">
                <input type="checkbox"
                       ${checked}
                       data-filename="${encodedFilename}"
                       onchange="handleDocumentCheckboxChange('${categoryKey}', this)">
                Verwenden
            </label>
        `;
    } else if (categoryKey === 'bescheid') {
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
    }

    const anonymizedBadge = isAnonymized
        ? `<div class="status-badge anonymized">✅ Anonymisiert</div>`
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

    return `
        <div class="document-card">
            <button class="delete-btn" onclick="deleteDocument('${jsSafeFilename}')" title="Löschen">×</button>
            ${selectionControls ? `<div class="selection-wrapper">${selectionControls}</div>` : ''}
            <div class="filename">${escapeHtml(doc.filename)}</div>
            <div class="confidence">${escapeHtml(confidenceValue)}</div>
            ${anonymizedBadge}
            ${anonymizeButton}
        </div>
    `;
}

async function loadSources(options) {
    const placeholders = !(options && options.placeholders === false);
    debugLog('loadSources: start');
    const container = document.getElementById('sonstiges-docs');

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
        debugError('addSourceFromResults: request error', error);
        alert(`❌ Fehler: ${error.message}`);
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
        debugError('deleteSource: failed', error);
        alert(`❌ Fehler: ${error.message}`);
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
        debugError('deleteAllSources: failed', error);
        alert(`❌ Fehler: ${error.message}`);
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
            // UI will auto-update via SSE (documents and sources snapshots)
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
        debugError('resetApplication: failed', error);
        alert(`❌ Fehler: ${error.message}`);
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
        debugError('uploadFile: request error', error);
        alert(`❌ Fehler: ${error.message}`);
    } finally {
        debugLog('uploadFile: hiding loading indicator');
        loading.style.display = 'none';
    }
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
        debugError('deleteDocument: request error', error);
        alert(`❌ Fehler: ${error.message}`);
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

async function generateDocument() {
    debugLog('generateDocument: start');
    const description = document.getElementById('outputDescription').value.trim();

    const payload = {};

    if (description) {
        payload.query = description;
    } else {
        const primaryBescheid = selectionState.bescheid.primary;
        if (!primaryBescheid) {
            debugLog('generateDocument: no description and no primary Bescheid selected');
            alert('Bitte wählen Sie einen Hauptbescheid (Anlage K2) aus oder geben Sie eine Recherchefrage ein.');
            return;
        }

        // Validate that the selected bescheid actually exists in the current DOM
        const encodedFilename = encodeURIComponent(primaryBescheid);
        const checkbox = document.querySelector(`input[type="checkbox"][data-bescheid-checkbox="${encodedFilename}"]`);
        if (!checkbox) {
            debugLog('generateDocument: selected Bescheid not found in DOM, clearing selection', { primaryBescheid });
            alert('Der ausgewählte Bescheid wurde nicht gefunden. Bitte wählen Sie einen Bescheid aus der Liste aus.');
            selectionState.bescheid.primary = null;
            return;
        }

        payload.primary_bescheid = primaryBescheid;
    }

    // Show loading state
    const button = event.target;
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = '🔍 Recherchiere...';
    debugLog('generateDocument: sending POST /research', payload);

    try {
        const response = await fetch('/research', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        debugLog('generateDocument: response status', response.status);
        const data = await response.json();
        debugLog('generateDocument: response body', data);

        if (response.ok) {
            debugLog('generateDocument: research successful, displaying results');
            displayResearchResults(data);
            // Sources list will auto-update via SSE when user adds sources from results
        } else {
            debugError('generateDocument: research failed', data);
            alert(`❌ Fehler: ${data.detail || 'Recherche fehlgeschlagen'}`);
        }
    } catch (error) {
        debugError('generateDocument: request error', error);
        alert(`❌ Fehler: ${error.message}`);
    } finally {
        debugLog('generateDocument: restoring button state');
        button.disabled = false;
        button.textContent = originalText;
    }
}

async function createDraft() {
    debugLog('createDraft: start');
    const textarea = document.getElementById('outputDescription');
    const documentTypeSelect = document.getElementById('documentTypeSelect');
    const userPrompt = (textarea?.value || '').trim();
    const documentType = documentTypeSelect ? documentTypeSelect.value : 'Klagebegründung';

    if (!userPrompt) {
        debugLog('createDraft: user prompt missing');
        alert('Bitte beschreiben Sie das gewünschte Dokument');
        return;
    }

    const payload = getSelectedDocumentsPayload();
    if (!payload.bescheid.primary) {
        alert('Bitte wählen Sie einen Hauptbescheid (Anlage K2) aus.');
        return;
    }

    const evt = typeof event !== 'undefined' ? event : null;
    const button = evt?.target || null;
    const originalText = button ? button.textContent : null;
    if (button) {
        button.disabled = true;
        button.textContent = '✍️ Generiere Entwurf...';
    }
    debugLog('createDraft: sending POST /generate', { documentType, payload });

    try {
        const response = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                document_type: documentType,
                user_prompt: userPrompt,
                selected_documents: payload
            })
        });
        debugLog('createDraft: response status', response.status);
        const data = await response.json();
        debugLog('createDraft: response body', data);

        if (response.ok) {
            debugLog('createDraft: generation successful');
            await displayDraft(data);
        } else {
            debugError('createDraft: generation failed', data);
            const detail = Array.isArray(data.detail)
                ? data.detail.map(item => item.msg || item).join('\n')
                : (data.detail || data.message || 'Generierung fehlgeschlagen');
            alert(`❌ Fehler: ${detail}`);
        }
    } catch (error) {
        debugError('createDraft: request error', error);
        alert(`❌ Fehler: ${error.message}`);
    } finally {
        debugLog('createDraft: restoring button state');
        if (button) {
            button.disabled = false;
            button.textContent = originalText;
        }
    }
}

async function displayDraft(data) {
    debugLog('displayDraft: rendering draft modal', data);
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 1000;';

    const content = document.createElement('div');
    content.style.cssText = 'background: white; padding: 30px; border-radius: 10px; max-width: 900px; max-height: 85vh; overflow-y: auto; box-shadow: 0 4px 6px rgba(0,0,0,0.1);';

    const modalKey = `draft-${Date.now()}`;
    window.generatedDrafts = window.generatedDrafts || {};
    window.generatedDrafts[modalKey] = data;

    // Helper function to copy draft text
    window.copyDraftText = function(key) {
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
    const citationsFound = metadata.citations_found != null ? metadata.citations_found : 0;
    const generatedText = data.generated_text || '(Kein Text erzeugt)';

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
                <strong style="color: #c0392b;">Nicht gefundene Quellen:</strong>
                <ul style="margin: 8px 0 0 16px; color: #c0392b; font-size: 14px;">
                    ${missing.map(m => `<li>${escapeHtml(m)}</li>`).join('')}
                </ul>
            </div>
        ` : '';

    content.innerHTML = `
        <h2 style="color: #2c3e50; margin-bottom: 15px;">✍️ ${escapeHtml(data.document_type || 'Entwurf')}</h2>
        <div style="background: #eaf7ec; padding: 12px; border-radius: 5px; margin-bottom: 12px;">
            <strong>Aufgabenstellung:</strong> ${escapeForTemplate(data.user_prompt || '—')}
        </div>
        <div style="margin-bottom: 12px; color: #34495e; font-size: 13px;">
            <strong>Statistik:</strong> ${citationsFound} Zitate · ${wordCount} Wörter
        </div>
        <pre style="white-space: pre-wrap; line-height: 1.45; background: #f8f9fa; padding: 16px; border-radius: 6px; border: 1px solid #e1e4e8;">${escapeHtml(generatedText)}</pre>
        ${usedHtml}
        ${warningsHtml}
        ${missingHtml}
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

function displayResearchResults(data) {
    debugLog('displayResearchResults: showing results', { query: data.query, sourceCount: (data.sources || []).length });
    window.latestResearchSources = Array.isArray(data.sources) ? data.sources : [];
    window.latestResearchQuery = data.query || '';
    const modal = document.createElement('div');
    modal.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 1000;';

    const content = document.createElement('div');
    content.style.cssText = 'background: white; padding: 30px; border-radius: 10px; max-width: 900px; max-height: 85vh; overflow-y: auto; box-shadow: 0 4px 6px rgba(0,0,0,0.1);';

    let sourcesHtml = '';
    if (data.sources && data.sources.length > 0) {
        sourcesHtml = '<h3 style="margin-top: 20px; color: #2c3e50;">📚 Relevante Quellen:</h3>';
        sourcesHtml += '<div style="color: #7f8c8d; font-size: 13px; margin-bottom: 12px;">💾 Hinweis: Hochwertige Quellen werden automatisch als PDF gespeichert und erscheinen in "Gespeicherte Quellen"</div>';
        sourcesHtml += '<div style="display: flex; flex-direction: column; gap: 15px;">';

        data.sources.forEach((source, index) => {
            const description = escapeForTemplate(source.description || 'Relevante Quelle für Ihre Recherche');
            const canAddToSources = !!source.pdf_url;
            const addButton = `<button onclick="addSourceFromResults(event, ${index})" style="display: inline-block; background: #27ae60; color: white; padding: 6px 12px; border-radius: 4px; border: none; cursor: pointer; font-size: 13px; font-weight: 500;">➕ Zu gespeicherten Quellen</button>`;
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
                        <span style="color: #7f8c8d; font-size: 12px;">
                            ${displayUrl}
                        </span>
                    </div>
                </div>
            `;
        });

        sourcesHtml += '</div>';
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
