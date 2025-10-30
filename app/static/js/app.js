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

// Load documents and sources on page load
window.addEventListener('DOMContentLoaded', () => {
    debugLog('DOMContentLoaded: initializing interface');
    loadDocuments();
    loadSources();
});

async function loadDocuments() {
    debugLog('loadDocuments: start');
    try {
        debugLog('loadDocuments: fetching /documents');
        const response = await fetch('/documents');
        debugLog('loadDocuments: response status', response.status);
        const data = await response.json();
        debugLog('loadDocuments: received data', data);

        pruneDocumentSelections(data);

        // Clear all boxes
        document.getElementById('anhoerung-docs').innerHTML = '';
        document.getElementById('bescheid-docs').innerHTML = '';
        document.getElementById('akte-docs').innerHTML = '';
        document.getElementById('rechtsprechung-docs').innerHTML = '';
        document.getElementById('sonstiges-docs').innerHTML = '';

        // Populate each category
        const categoryMap = {
            'Anhörung': 'anhoerung-docs',
            'Bescheid': 'bescheid-docs',
            'Akte': 'akte-docs',
            'Rechtsprechung': 'rechtsprechung-docs',
            'Sonstiges': 'sonstiges-docs'
        };

        for (const [category, documents] of Object.entries(data)) {
            const docsArray = Array.isArray(documents) ? documents : [];
            debugLog(`loadDocuments: rendering category ${category}`, { count: docsArray.length });
            const boxId = categoryMap[category];
            const box = document.getElementById(boxId);

            if (box) {
                if (docsArray.length === 0) {
                    box.innerHTML = '<div class="empty-message">Keine Dokumente</div>';
                } else {
                    const cards = docsArray
                        .map(doc => createDocumentCard(doc))
                        .filter(card => !!card)
                        .join('');
                    box.innerHTML = cards || '<div class="empty-message">Keine Dokumente</div>';
                }
            }
        }
    } catch (error) {
        debugError('loadDocuments: failed', error);
    }
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

    return `
        <div class="document-card">
            <button class="delete-btn" onclick="deleteDocument('${jsSafeFilename}')" title="Löschen">×</button>
            ${selectionControls ? `<div class="selection-wrapper">${selectionControls}</div>` : ''}
            <div class="filename">${escapeHtml(doc.filename)}</div>
            <div class="confidence">${escapeHtml(confidenceValue)}</div>
            ${showAnonymizeBtn ? `
                <button class="anonymize-btn" onclick="anonymizeDocument('${doc.id}', this)" title="Anonymisieren">
                    🔒 Anonymisieren
                </button>
            ` : ''}
        </div>
    `;
}

async function loadSources() {
    debugLog('loadSources: start');
    try {
        debugLog('loadSources: fetching /sources');
        const response = await fetch('/sources');
        debugLog('loadSources: response status', response.status);
        const payload = await response.json();
        const sources = Array.isArray(payload) ? payload : (payload.sources || []);
        const count = Array.isArray(payload) ? payload.length : (payload.count ?? sources.length);
        debugLog('loadSources: received payload', { count, sources });

        const container = document.getElementById('sonstiges-docs');
        debugLog('loadSources: target container located', container);

        pruneSourceSelections(sources);

        if (sources.length === 0) {
            debugLog('loadSources: no sources stored');
            container.innerHTML = '<div class="empty-message">Keine Quellen gespeichert</div>';
        } else {
            debugLog('loadSources: rendering source cards');
            container.innerHTML = sources.map(source => createSourceCard(source)).join('');
        }
    } catch (error) {
        debugError('loadSources: failed', error);
    }
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
            loadSources();
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
            debugLog('deleteSource: success, refreshing sources');
            loadSources();
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
            loadSources();
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
            // Reload documents to show the new one
            await loadDocuments();
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
            debugLog('deleteDocument: success, refreshing documents');
            // Reload documents
            await loadDocuments();
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

    if (!description) {
        debugLog('generateDocument: description missing');
        alert('Bitte beschreiben Sie das gewünschte Dokument');
        return;
    }

    // Show loading state
    const button = event.target;
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = '🔍 Recherchiere...';
    debugLog('generateDocument: sending POST /research', { query: description });

    try {
        const response = await fetch('/research', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: description })
        });

        debugLog('generateDocument: response status', response.status);
        const data = await response.json();
        debugLog('generateDocument: response body', data);

        if (response.ok) {
            debugLog('generateDocument: research successful, displaying results');
            displayResearchResults(data);
            loadSources();
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
            <button onclick="loadSources(); this.parentElement.parentElement.parentElement.remove();"
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
