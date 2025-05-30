PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS files (
    rowid INTEGER PRIMARY KEY,
    multihash TEXT NOT NULL,
    filename TEXT, /* filename at time of upload */
    mimetype TEXT NOT NULL,
    metadata JSONB,
    size INTEGER NOT NULL,
    content BLOB, /* actual file content - NULL = external storage */

    UNIQUE(multihash)
);

CREATE TABLE IF NOT EXISTS memories (
    rowid INTEGER PRIMARY KEY,
    timestamp REAL,
    kind TEXT NOT NULL,
    data JSONB NOT NULL,
    importance REAL
);

CREATE TABLE IF NOT EXISTS edges (
    src_id INTEGER NOT NULL,
    dst_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    weight REAL NOT NULL,

    PRIMARY KEY (src_id, label),
    FOREIGN KEY (src_id) REFERENCES memories(rowid),
    FOREIGN KEY (dst_id) REFERENCES memories(rowid)
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(content);

CREATE VIRTUAL TABLE IF NOT EXISTS vss_nomic_v1_5_index USING vec0 (
    memory_id INTEGER PRIMARY KEY,
    embedding FLOAT[1536]
);