-- Migration: Consolidate documents and processed_documents into single table
-- Date: 2025-11-04

BEGIN;

-- 1. Add new columns to documents table
ALTER TABLE documents
    DROP COLUMN IF EXISTS extracted_text,
    ADD COLUMN IF NOT EXISTS extracted_text_path VARCHAR(512),
    ADD COLUMN IF NOT EXISTS is_anonymized BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS ocr_applied BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS needs_ocr BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS anonymization_metadata JSONB,
    ADD COLUMN IF NOT EXISTS processing_status VARCHAR(20) DEFAULT 'pending';

-- 2. Migrate data from processed_documents to documents
-- For each document, take the most recent processed entry
UPDATE documents d
SET
    is_anonymized = COALESCE(p.is_anonymized, FALSE),
    ocr_applied = COALESCE(p.ocr_applied, FALSE),
    needs_ocr = COALESCE(p.needs_ocr, FALSE),
    anonymization_metadata = p.anonymization_metadata,
    processing_status = COALESCE(p.processing_status, 'pending')
FROM (
    SELECT DISTINCT ON (document_id)
        document_id,
        extracted_text,
        is_anonymized,
        ocr_applied,
        needs_ocr,
        anonymization_metadata,
        processing_status
    FROM processed_documents
    ORDER BY document_id, created_at DESC
) p
WHERE d.id = p.document_id;

-- 3. Drop the processed_documents table
DROP TABLE IF EXISTS processed_documents CASCADE;

-- 4. Add index for common queries
CREATE INDEX IF NOT EXISTS ix_documents_needs_ocr ON documents(needs_ocr) WHERE needs_ocr = TRUE;
CREATE INDEX IF NOT EXISTS ix_documents_is_anonymized ON documents(is_anonymized) WHERE is_anonymized = TRUE;

COMMIT;
