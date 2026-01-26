PRAGMA foreign_keys=ON;

/**
 * Main table for storing and coordinating memories.
**/
CREATE TABLE IF NOT EXISTS memories (
    rowid INTEGER PRIMARY KEY,
    uuid BLOB UNIQUE NOT NULL,
    data JSONB NOT NULL
);

/**
 * Memory dependencies which enable grounding and context for memories.
**/
CREATE TABLE IF NOT EXISTS edges (
    src_id INTEGER REFERENCES memories(rowid) ON DELETE CASCADE,
    dst_id INTEGER REFERENCES memories(rowid) ON DELETE CASCADE,

    PRIMARY KEY (src_id, dst_id)
);
