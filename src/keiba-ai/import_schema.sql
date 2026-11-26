-- Normalized schema used only by src/keiba-ai/import_csv.py.
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS races (
    race_id TEXT PRIMARY KEY,
    race_date TEXT NOT NULL,
    venue TEXT NOT NULL,
    race_number INTEGER NOT NULL,
    race_name TEXT,
    surface TEXT,
    distance_m INTEGER,
    going TEXT,
    field_size INTEGER,
    trifecta_payout_yen INTEGER,
    source_name TEXT NOT NULL,
    imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS horses (
    horse_id TEXT PRIMARY KEY,
    horse_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entries (
    race_id TEXT NOT NULL,
    horse_id TEXT NOT NULL,
    horse_number INTEGER,
    frame_number INTEGER,
    jockey_name TEXT,
    carried_weight REAL,
    win_odds REAL,
    popularity INTEGER,
    finish_position INTEGER,
    PRIMARY KEY (race_id, horse_id),
    FOREIGN KEY (race_id) REFERENCES races(race_id),
    FOREIGN KEY (horse_id) REFERENCES horses(horse_id)
);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    race_id TEXT NOT NULL,
    horse_id TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    win_odds REAL,
    popularity INTEGER,
    PRIMARY KEY (race_id, horse_id, captured_at),
    FOREIGN KEY (race_id, horse_id)
        REFERENCES entries(race_id, horse_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id TEXT NOT NULL,
    predicted_at TEXT NOT NULL,
    cutoff_at TEXT NOT NULL,
    chaos_probability REAL NOT NULL,
    uncertainty_score REAL,
    decision TEXT NOT NULL,
    model_version TEXT NOT NULL,
    FOREIGN KEY (race_id) REFERENCES races(race_id)
);

CREATE TABLE IF NOT EXISTS model_evaluations (
    evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    train_until TEXT NOT NULL,
    test_from TEXT NOT NULL,
    test_until TEXT NOT NULL,
    skip_rate REAL,
    chaos_recall REAL,
    chaos_precision REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_races_date
    ON races(race_date);

CREATE INDEX IF NOT EXISTS idx_entries_horse
    ON entries(horse_id);

CREATE INDEX IF NOT EXISTS idx_predictions_race
    ON predictions(race_id, predicted_at);
