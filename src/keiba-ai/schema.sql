PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS races (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE,
    race_date TEXT,
    venue TEXT,
    race_name TEXT,
    distance_m INTEGER,
    surface TEXT,
    direction TEXT,
    crawled_at TEXT,
    race_number INTEGER,
    start_time TEXT,
    weather TEXT,
    turf_condition TEXT,
    dirt_condition TEXT,
    race_grade TEXT,
    race_conditions TEXT
);

CREATE TABLE IF NOT EXISTS horses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER NOT NULL,
    finish INTEGER,
    frame INTEGER,
    horse_number INTEGER,
    horse_name TEXT,
    sex_age TEXT,
    weight REAL,
    jockey TEXT,
    time TEXT,
    margin TEXT,
    last_3f REAL,
    body_weight INTEGER,
    weight_change INTEGER,
    trainer TEXT,
    popularity INTEGER,
    odds REAL,
    horse_mark TEXT,
    corner_order TEXT,
    horse_url TEXT,
    jockey_url TEXT,
    trainer_url TEXT,
    source_fields TEXT
);

CREATE TABLE IF NOT EXISTS payouts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER NOT NULL,
    bet_type TEXT,
    combination TEXT,
    amount INTEGER,
    popularity INTEGER
);

CREATE TABLE IF NOT EXISTS crawl_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT,
    error TEXT,
    occurred_at TEXT
);

CREATE TABLE IF NOT EXISTS crawl_retry_queue (
    url TEXT PRIMARY KEY,
    retry_count INTEGER NOT NULL DEFAULT 0,
    next_retry_at TEXT NOT NULL,
    last_error TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER NOT NULL,
    predicted_at TEXT NOT NULL,
    cutoff_at TEXT,
    chaos_probability REAL NOT NULL DEFAULT 0,
    uncertainty_score REAL,
    decision TEXT NOT NULL DEFAULT 'unavailable',
    model_version TEXT NOT NULL,
    top3_predictions TEXT,
    upset_factors TEXT,
    evaluated_at TEXT
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

CREATE TABLE IF NOT EXISTS prediction_evaluations (
    evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    test_from TEXT NOT NULL,
    test_until TEXT NOT NULL,
    evaluated_races INTEGER NOT NULL,
    mean_correct_picks REAL NOT NULL,
    precision_at_3 REAL NOT NULL,
    win_return_rate REAL,
    quinella_return_rate REAL,
    trio_return_rate REAL,
    upset_races INTEGER NOT NULL DEFAULT 0,
    upset_brier_score REAL,
    upset_top_10pct_rate REAL
);

CREATE INDEX IF NOT EXISTS idx_races_date ON races(race_date);
CREATE INDEX IF NOT EXISTS idx_horses_race ON horses(race_id);
CREATE INDEX IF NOT EXISTS idx_payouts_race ON payouts(race_id);
CREATE INDEX IF NOT EXISTS idx_crawl_retry_due
    ON crawl_retry_queue(next_retry_at);
CREATE INDEX IF NOT EXISTS idx_predictions_race ON predictions(race_id);
CREATE INDEX IF NOT EXISTS idx_predictions_time
    ON predictions(predicted_at);
CREATE INDEX IF NOT EXISTS idx_prediction_evaluations_model
    ON prediction_evaluations(model_version);
