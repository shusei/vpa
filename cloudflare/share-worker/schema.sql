CREATE TABLE IF NOT EXISTS shares (
  id TEXT PRIMARY KEY,
  created_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL,
  locale TEXT NOT NULL,
  title TEXT NOT NULL,
  description TEXT NOT NULL,
  alt TEXT NOT NULL,
  target_url TEXT NOT NULL,
  image BLOB NOT NULL
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS shares_expires_at_idx
ON shares (expires_at);
