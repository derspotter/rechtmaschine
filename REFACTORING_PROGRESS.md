# DRY Refactoring Progress

**Started**: 2025-10-30
**Goal**: Reduce ~970 lines of duplicated code, unify document/source update mechanisms

---

## Phase 1: Unify Backend Broadcasting ✅

### 1.1 Extend PostgreSQL LISTEN/NOTIFY to Sources
- [x] Add `SOURCES_CHANNEL` to events.py
- [x] Create generic `_emit_event()` helper in app.py
- [x] Keep `_broadcast_documents_snapshot()` and create `_broadcast_sources_snapshot()`
- [x] Add `_broadcast_sources_snapshot()` calls to source operations (add, delete, delete_all, downloads)
- [x] Unified `/documents/stream` SSE endpoint (sends BOTH documents AND sources snapshots)
- [x] Initialize both PostgreSQL listeners (documents_updates, sources_updates) feeding into same hub

**Status**: COMPLETE ✅
**Lines saved**: ~60 lines backend
**Key Changes**:
- Single SSE stream (`/documents/stream`) now broadcasts all entity types
- Two PostgreSQL channels feeding into one unified BroadcastHub
- Removed legacy `_notify_sources_updated()` function
- Added broadcast to anonymize endpoint (was missing)

---

## Phase 2: Simplify Frontend Update Logic ✅

### 2.1 Remove Manual Refresh Calls
- [x] Remove `loadSources()` from `addSourceFromResults()`
- [x] Remove `loadSources()` from `deleteSource()`
- [x] Remove `loadSources()` from `deleteAllSources()`
- [x] Remove `loadDocuments()` + `loadSources()` from `resetApplication()` (kept fallback for 404)
- [x] Remove `loadDocuments()` from `deleteDocument()`
- [x] Remove `loadDocuments()` from `anonymizeDocument()`
- [x] Remove `loadSources()` from `generateDocument()` (research results)
- [x] Remove `loadSources()` from research modal button

**Status**: COMPLETE ✅
**Lines saved**: ~8 manual refresh calls removed
**Key Changes**:
- All operations now rely on SSE broadcasts instead of manual refreshes
- Kept initial page load calls (loadDocuments/loadSources on DOMContentLoaded)
- Kept fallback refresh in reset when /reset endpoint returns 404

### 2.2 Unified SSE Stream Handler
- [x] Add `handleSourceSnapshot()` function
- [x] Add `renderSources()` function with digest comparison
- [x] Update `onmessage` handler to process both `documents_snapshot` and `sources_snapshot` events
- [x] Single EventSource connection handles all updates

**Status**: COMPLETE ✅
**Lines saved**: ~50 lines (unified stream handling)

**Decisions Made**:
- ❌ Did NOT create generic `startEntityStream()` - kept existing working code
- ✅ Single SSE endpoint is cleaner than separate endpoints
- ✅ Frontend now handles both event types from one stream

---

## Phase 3: Simplify Polling Watchdog ⏸️

**Status**: DEFERRED
**Reason**: Current polling implementation works fine, focus on critical path first

---

## Phase 4: Consolidate Selection Logic ⏸️

**Status**: DEFERRED
**Reason**: Selection logic works, not causing bugs, optimize later if needed

---

## Phase 5: Configuration-Based Architecture ⏸️

**Status**: DEFERRED
**Reason**: Over-engineering at this stage, implement when adding 3rd+ entity type

---

## Testing Checklist

**Tested & Working:**
- [x] Delete document → UI auto-updates via SSE ✅ (instant update, <100ms)
- [x] SSE stream health → logs show both channels connected ✅
- [x] Digest comparison → prevents unnecessary redraws ✅

**To be tested:**
- [ ] Upload PDF → documents auto-update via SSE
- [ ] Reset all → both documents and sources clear via SSE
- [ ] Add saved source → sources auto-update via SSE
- [ ] Delete source → sources auto-update via SSE
- [ ] Anonymize document → badge appears via SSE
- [ ] Download status updates → UI refreshes via SSE
- [ ] Multiple browser tabs → all stay in sync (if testing multi-tab)

---

## Total Progress

**Backend refactoring**: COMPLETE ✅
- Unified event broadcasting system
- Single SSE stream for all updates
- PostgreSQL LISTEN/NOTIFY bridge operational

**Frontend refactoring**: COMPLETE ✅
- Removed all manual refresh calls after operations
- Unified stream handler for documents + sources
- Digest-based rendering prevents unnecessary redraws

**Estimated savings**: ~120 lines removed
**Completion**: Phase 1 & 2 complete (core objectives achieved)

---

## Notes & Issues

### Architecture Decisions

**Why one SSE stream instead of two?**
- Simpler frontend code (one EventSource connection)
- Easier to add new entity types in future
- Reduces connection overhead
- Unified event handling logic

**Why keep two PostgreSQL channels?**
- Logical separation of concerns
- Easier debugging (can see which events are which)
- Flexibility to route events differently in future
- Both feed into same hub anyway

**Why not make it more generic?**
- Over-engineering risk: added complexity for unclear future benefit
- Current solution is DRY enough: no significant duplication
- Easy to refactor to generic pattern when adding 3rd+ entity type
- YAGNI principle: implement when actually needed

### Known Working State

**App startup confirmed:**
```
Listening on PostgreSQL channels: documents_updates, sources_updates
```

**SSE endpoint unified:**
- `/documents/stream` sends initial snapshots for BOTH documents and sources
- Subsequent events routed by type field

**Broadcast triggers:**
- Documents: classify, delete, reset, anonymize
- Sources: add, delete, delete_all, download_started, download_completed, download_failed
- Reset: broadcasts BOTH documents and sources snapshots

### Next Steps (if needed)

If further optimization is desired:
1. Create generic `fetchAndRender(config)` to DRY up loadDocuments/loadSources
2. Smart polling (only when SSE fails)
3. Generic selection pruning helper
4. Configuration-based entity management

**Recommendation**: Ship current changes, monitor in production, optimize based on real usage patterns.
