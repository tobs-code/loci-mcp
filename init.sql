CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

LOAD 'age';
SET search_path TO ag_catalog, public;

CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL DEFAULT '',
    embedding vector(768),
    mem_type TEXT NOT NULL DEFAULT 'fact',
    scope TEXT NOT NULL DEFAULT 'global',
    entity TEXT,
    importance FLOAT DEFAULT 1.0,
    metadata JSONB DEFAULT '{}',
    access_count INT DEFAULT 0,
    superseded_by TEXT,
    deleted_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE entities (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    type TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE relations (
    id SERIAL PRIMARY KEY,
    source_id INT REFERENCES entities(id) ON DELETE CASCADE,
    target_id INT REFERENCES entities(id) ON DELETE CASCADE,
    relation_type TEXT,
    weight FLOAT DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_memories_embedding ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_memories_tsvector ON memories USING GIN (to_tsvector('german', content));
CREATE INDEX idx_memories_type ON memories(mem_type);
CREATE INDEX idx_memories_entity ON memories(entity);
CREATE INDEX idx_memories_hash ON memories(content_hash);
CREATE INDEX idx_memories_active ON memories(superseded_by) WHERE superseded_by IS NULL;
CREATE INDEX idx_entities_name ON entities(name);
CREATE INDEX idx_entities_type ON entities(type);
CREATE INDEX idx_relations_source ON relations(source_id);
CREATE INDEX idx_relations_target ON relations(target_id);
CREATE INDEX idx_relations_type ON relations(relation_type);

SELECT create_graph('memory_graph');