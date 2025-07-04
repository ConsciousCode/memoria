CREATE TABLE IF NOT EXISTS anthropic_convos (
    uuid BLOB NOT NULL PRIMARY KEY,
    memory BLOB UNIQUE NOT NULL,
    uploaded_at INT NOT NULL
);

CREATE TABLE IF NOT EXISTS anthropic_messages (
    uuid BLOB NOT NULL PRIMARY KEY,
    memory BLOB UNIQUE NOT NULL,
    uploaded_at INT NOT NULL,
    convo BLOB NOT NULL REFERENCES anthropic_convos(uuid) ON DELETE CASCADE
);