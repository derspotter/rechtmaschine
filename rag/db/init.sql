CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id TEXT PRIMARY KEY,
    collection TEXT NOT NULL DEFAULT 'rag_chunks',
    text TEXT NOT NULL,
    context_header TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    provenance JSONB NOT NULL DEFAULT '[]'::jsonb,
    dense vector(1024) NOT NULL,
    sparse JSONB,
    search_text TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('german', coalesce(context_header, '') || ' ' || coalesce(text, ''))
    ) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS rag_chunks_collection_idx ON rag_chunks (collection);
CREATE INDEX IF NOT EXISTS rag_chunks_metadata_idx ON rag_chunks USING gin (metadata);
CREATE INDEX IF NOT EXISTS rag_chunks_search_text_idx ON rag_chunks USING gin (search_text);
CREATE INDEX IF NOT EXISTS rag_chunks_dense_idx ON rag_chunks USING hnsw (dense vector_cosine_ops);
