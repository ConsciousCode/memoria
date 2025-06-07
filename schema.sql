PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS files (
    rowid INTEGER PRIMARY KEY,
    cid BLOB NOT NULL UNIQUE,
    filename TEXT, /* filename at time of upload */
    mimetype TEXT NOT NULL,
    metadata JSONB,
    size INTEGER NOT NULL,
    content BLOB /* actual file content - NULL = external storage */
);

CREATE TABLE IF NOT EXISTS memories (
    rowid INTEGER PRIMARY KEY,
    cid BLOB UNIQUE, -- NULL indicates an incomplete memory
    timestamp REAL,
    kind TEXT NOT NULL,
    data JSONB NOT NULL,
    importance REAL
);

CREATE TABLE IF NOT EXISTS edges (
    src_id INTEGER REFERENCES memories(rowid) ON DELETE CASCADE,
    dst_id INTEGER REFERENCES memories(rowid) ON DELETE CASCADE,
    label TEXT NOT NULL,
    weight REAL NOT NULL,

    PRIMARY KEY (src_id, label)
);

CREATE TABLE IF NOT EXISTS sonas (
    rowid INTEGER PRIMARY KEY,
    uuid BLOB NOT NULL UNIQUE,
    last_id INTEGER REFERENCES memories(rowid) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS acthreads (
    sona_id INTEGER REFERENCES sonas(rowid) ON DELETE CASCADE,
    memory_id INTEGER REFERENCES memories(id) ON DELETE CASCADE,
    last_id INTEGER REFERENCES memories(rowid) ON DELETE CASCADE,

    PRIMARY KEY (sona_id, memory_id),
);

-------------
-- Indices --
-------------

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(content);

-- Multiple embeddings can refer to the same memory, so we store the rowid
--  and memory_id separately.
CREATE VIRTUAL TABLE IF NOT EXISTS vss_nomic_v1_5_index USING vec0 (
    rowid INTEGER PRIMARY KEY,
    memory_id INTEGER NOT NULL REFERENCES memories(rowid) ON DELETE CASCADE,
    embedding FLOAT[1536] NOT NULL,

    UNIQUE(memory_id, embedding)
);

CREATE VIRTUAL TABLE IF NOT EXISTS sona_embedding USING vec0(
    sona_id INTEGER PRIMARY KEY REFERENCES sonas(rowid) ON DELETE CASCADE,
    embedding FLOAT[1536] NOT NULL
);