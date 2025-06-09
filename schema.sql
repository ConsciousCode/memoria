PRAGMA foreign_keys=ON;

/**
 * Multimodal files uploaded to memory.
**/
CREATE TABLE IF NOT EXISTS files (
    rowid INTEGER PRIMARY KEY,
    cid BLOB NOT NULL UNIQUE,
    filename TEXT, /* filename at time of upload */
    mimetype TEXT NOT NULL,
    metadata JSONB,
    size INTEGER NOT NULL,
    content BLOB /* actual file content - NULL = external storage */
);

/**
 * Main table for storing and coordinating memories.
**/
CREATE TABLE IF NOT EXISTS memories (
    rowid INTEGER PRIMARY KEY,
    cid BLOB UNIQUE, -- NULL indicates an incomplete memory
    timestamp REAL,
    kind TEXT NOT NULL CHECK (object_type IN ('self', 'other', 'text', 'file')),
    data JSONB NOT NULL,
    importance REAL
);

/**
 * Memory dependencies which enable grounding and context for memories.
**/
CREATE TABLE IF NOT EXISTS edges (
    src_id INTEGER REFERENCES memories(rowid) ON DELETE CASCADE,
    dst_id INTEGER REFERENCES memories(rowid) ON DELETE CASCADE,
    label TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,

    PRIMARY KEY (src_id, label)
);

/**
 * A sona is a fuzzy-isolated component of a total self which processes
 * incoming information independently, but can share information with
 * other sonas via memories. They act as de-facto centroids of their name
 * embeddings for emergent compartmentalization of the self.
 *
 * active_id is the currently active incomplete memory receiving ACT updates
 * pending_id is the next memory to be processed
**/
CREATE TABLE IF NOT EXISTS sonas (
    rowid INTEGER PRIMARY KEY,
    uuid BLOB NOT NULL UNIQUE
);

/**
 * Memories seen or recalled at some point by the sona. This makes them easier
 * for the sona to recall later. This is a strict superset of the memories
 * referenced by the sona's ACT threads.
**/
CREATE TABLE IF NOT EXISTS sona_memories (
    sona_id INTEGER REFERENCES sonas(rowid) ON DELETE CASCADE,
    memory_id INTEGER REFERENCES memories(id) ON DELETE CASCADE,
);

/**
 * Linked list of the linear chain of processes within a sona, its Autonomous
 * Cognitive Thread (ACT). The memory is the response, and the prev_id is
 * the previous link in the chain, or NULL at the end.
**/
CREATE TABLE IF NOT EXISTS acthreads (
    rowid INTEGER PRIMARY KEY,
    cid BLOB UNIQUE, -- ACT CID depends on memory CID which may be NULL for staged memories
    sona_id INTEGER REFERENCES sonas(rowid) ON DELETE CASCADE,
    memory_id INTEGER REFERENCES memories(id) ON DELETE CASCADE,
    prev_id INTEGER REFERENCES acthreads(rowid) ON DELETE CASCADE
);

-------------
-- Indices --
-------------

/**
 * FTS index for searching memory content by exact keywords.
**/
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(content);

/**
 * Note: Multiple embeddings can refer to the same memory, so we store the
 * rowid and memory_id separately.
**/
CREATE VIRTUAL TABLE IF NOT EXISTS memory_vss USING vec0 (
    rowid INTEGER PRIMARY KEY,
    memory_id INTEGER NOT NULL REFERENCES memories(rowid) ON DELETE CASCADE,
    embedding FLOAT[1536] NOT NULL,

    UNIQUE(memory_id, embedding)
);

/**
 * Embeddings of the various names given to sonas for fuzzy matching.
**/
CREATE VIRTUAL TABLE IF NOT EXISTS sona_vss USING vec0(
    sona_id INTEGER PRIMARY KEY REFERENCES sonas(rowid) ON DELETE CASCADE,
    embedding FLOAT[1536] NOT NULL
);