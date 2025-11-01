PRAGMA foreign_keys=ON;

/**
 * Main table for storing and coordinating memories.
**/
CREATE TABLE IF NOT EXISTS memories (
    rowid INTEGER PRIMARY KEY,
    cid BLOB UNIQUE, -- NULL indicates an incomplete memory
    data JSONB NOT NULL,
    timestamp REAL,
    metadata JSONB
);

/**
 * Memories marked as requiring updates in the merkledag.
**/
CREATE TABLE IF NOT EXISTS invalid_memories (
    memory_id INTEGER PRIMARY KEY REFERENCES memories(rowid) ON DELETE CASCADE
);

/**
 * Memory dependencies which enable grounding and context for memories.
**/
CREATE TABLE IF NOT EXISTS edges (
    src_id INTEGER REFERENCES memories(rowid) ON DELETE CASCADE,
    dst_id INTEGER REFERENCES memories(rowid) ON DELETE CASCADE,

    PRIMARY KEY (src_id, dst_id)
);

/**
 * Root CIDs of the IPFS files stored in the flatfs blockstore.
**/
CREATE TABLE IF NOT EXISTS ipfs_files (
    cid BLOB PRIMARY KEY,
    filename TEXT, /* filename at time of upload */
    mimetype TEXT NOT NULL,
    filesize INTEGER NOT NULL,
    overhead INTEGER NOT NULL /* Total size of the file including IPFS overhead */
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
    memory_id INTEGER NOT NULL REFERENCES memories(rowid) ON DELETE CASCADE,
    embedding FLOAT[768]
);
