import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from scripts import model_features, train_model, train_upset_model


class HistoricalFeatureTests(unittest.TestCase):
    def test_uses_only_results_from_earlier_dates(self):
        horses = pd.DataFrame(
            [
                {
                    "race_date": "2020-01-01",
                    "finish": 1,
                    "horse_name": "Horse A",
                    "jockey": "Jockey A",
                    "trainer": "Trainer A",
                },
                {
                    "race_date": "2020-01-01",
                    "finish": 4,
                    "horse_name": "Horse B",
                    "jockey": "Jockey A",
                    "trainer": "Trainer A",
                },
                {
                    "race_date": "2020-01-02",
                    "finish": 4,
                    "horse_name": "Horse A",
                    "jockey": "Jockey A",
                    "trainer": "Trainer A",
                },
                {
                    "race_date": "2020-01-03",
                    "finish": np.nan,
                    "horse_name": "Horse A",
                    "jockey": "Jockey A",
                    "trainer": "Trainer A",
                },
            ]
        )

        result = model_features.add_historical_performance_features(horses)

        self.assertEqual(result.loc[0, "jockey_prior_starts"], 0)
        self.assertEqual(result.loc[1, "jockey_prior_starts"], 0)
        self.assertEqual(result.loc[2, "jockey_prior_starts"], 2)
        self.assertEqual(result.loc[2, "jockey_prior_win_rate"], 0.5)
        self.assertEqual(result.loc[2, "horse_prior_top3_rate"], 1.0)
        self.assertEqual(result.loc[3, "horse_prior_starts"], 2)
        self.assertEqual(result.loc[3, "horse_prior_top3_rate"], 0.5)

    def test_parses_jra_combined_sex_and_age(self):
        result = model_features.add_sex_age_features(
            pd.DataFrame({"sex_age": ["牡4", "牝3", "セン5", None]})
        )

        self.assertEqual(result["sex"].tolist()[:3], ["牡", "牝", "セン"])
        self.assertEqual(result["age"].tolist()[:3], [4, 3, 5])
        self.assertTrue(pd.isna(result.loc[3, "age"]))

    def test_prediction_rows_include_only_prior_history(self):
        race = {
            "id": 2,
            "race_date": "2020-01-02",
            "venue": "Tokyo",
            "distance_m": 1600,
            "surface": "Turf",
            "race_number": 11,
            "start_time": "10:30",
            "weather": "晴",
            "turf_condition": "良",
            "race_conditions": "{\"category\":\"重賞\"}",
        }
        horses = pd.DataFrame(
            [
                {
                    "horse_number": 1,
                    "horse_name": "Horse A",
                    "sex_age": "牡4",
                    "weight": 56.0,
                    "body_weight": 470,
                    "weight_change": 2,
                    "jockey": "Jockey A",
                    "trainer": "Trainer A",
                    "horse_mark": "(父)",
                    "popularity": 2,
                    "horse_prior_starts": 999,
                }
            ]
        )
        historical = pd.DataFrame(
            [
                {
                    "race_id": 1,
                    "race_date": "2020-01-01",
                    "finish": 1,
                    "horse_name": "Horse A",
                    "jockey": "Jockey A",
                    "trainer": "Trainer A",
                },
                {
                    "race_id": 3,
                    "race_date": "2020-01-02",
                    "finish": 3,
                    "horse_name": "Horse A",
                    "jockey": "Jockey A",
                    "trainer": "Trainer A",
                },
            ]
        )

        result = model_features.make_prediction_feature_rows(
            race,
            horses,
            historical,
        )

        self.assertEqual(result.loc[0, "horse_prior_starts"], 1)
        self.assertEqual(result.loc[0, "horse_prior_win_rate"], 1)
        self.assertEqual(result.loc[0, "jockey_prior_starts"], 1)
        self.assertEqual(result.loc[0, "weight"], 56.0)
        self.assertEqual(result.loc[0, "distance_m"], 1600)
        self.assertEqual(result.loc[0, "race_number"], 11)
        self.assertEqual(result.loc[0, "start_minutes"], 630)
        self.assertEqual(result.loc[0, "weather"], "晴")
        self.assertEqual(result.loc[0, "horse_mark"], "(父)")

    def test_recent_form_and_body_weight_trend_use_only_prior_dates(self):
        horses = pd.DataFrame(
            [
                {
                    "race_id": day,
                    "race_date": f"2020-01-0{day}",
                    "finish": finish,
                    "horse_name": "Horse A",
                    "jockey": "Jockey A",
                    "trainer": "Trainer A",
                    "venue": "Tokyo",
                    "race_name": "Same Stakes",
                    "surface": "Turf",
                    "distance_m": 1600,
                    "body_weight": 468 + day,
                    "last_3f": 35.0 + day,
                    "time": f"1:{30 + day}.0",
                }
                for day, finish in enumerate([1, 4, 3, 4, 4, 4], start=1)
            ]
            + [
                {
                    "race_id": 7,
                    "race_date": "2020-01-07",
                    "finish": np.nan,
                    "horse_name": "Horse A",
                    "jockey": "Jockey A",
                    "trainer": "Trainer A",
                    "venue": "Tokyo",
                    "race_name": "Same Stakes",
                    "surface": "Turf",
                    "distance_m": 1600,
                    "body_weight": 475,
                    "last_3f": np.nan,
                    "time": None,
                }
            ]
        )

        result = model_features.add_historical_performance_features(horses)
        current = result.iloc[-1]

        self.assertAlmostEqual(
            current["horse_recent_top3_rate_5"],
            2 / 15,
        )
        self.assertEqual(current["horse_prior_days_since_last_start"], 1)
        self.assertEqual(current["horse_prior_time_seconds"], 96)
        self.assertEqual(current["horse_prior_last_3f"], 41.0)
        self.assertEqual(current["horse_prior_body_weight_change"], 1)
        self.assertEqual(current["horse_course_distance_prior_starts"], 6)
        self.assertEqual(current["jockey_horse_prior_starts"], 6)

    def test_race_name_encoding_uses_earlier_races_only(self):
        history = pd.DataFrame(
            [
                {
                    "race_id": race_id,
                    "race_date": race_date,
                    "race_name": "Same Stakes",
                    "finish": finish,
                }
                for race_id, race_date, finishes in (
                    (1, "2020-01-01", [1, 2, 3, 4]),
                    (2, "2020-01-02", [1, 2, 3, 4]),
                )
                for finish in finishes
            ]
        )

        result = model_features.add_race_name_history_features(history)

        self.assertTrue(result.loc[:3, "race_name_prior_starts"].eq(0).all())
        self.assertTrue(result.loc[4:, "race_name_prior_starts"].eq(4).all())
        self.assertTrue(
            result.loc[4:, "race_name_prior_top3_rate"].eq(0.75).all()
        )

    def test_relative_features_rank_entries_within_each_race(self):
        entries = pd.DataFrame(
            {
                "race_id": [1, 1, 2, 2],
                "odds": [2.0, 10.0, 3.0, 6.0],
            }
        )

        result = model_features.add_race_relative_features(entries, ["odds"])

        self.assertEqual(result["odds_race_percentile"].tolist(), [0.5, 1, 0.5, 1])


class TopThreeModelTests(unittest.TestCase):
    def test_loads_optional_fields_and_excludes_result_leakage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "races.sqlite"
            with sqlite3.connect(database_path) as conn:
                conn.executescript(
                    """
                    CREATE TABLE races (
                        id INTEGER PRIMARY KEY, race_date TEXT, venue TEXT,
                        distance_m INTEGER, surface TEXT, direction TEXT,
                        race_name TEXT, race_number INTEGER, weather TEXT
                    );
                    CREATE TABLE horses (
                        id INTEGER PRIMARY KEY, race_id INTEGER, finish INTEGER,
                        frame INTEGER, horse_number INTEGER, horse_name TEXT,
                        sex_age TEXT, weight REAL, jockey TEXT, trainer TEXT,
                        body_weight INTEGER, weight_change INTEGER,
                        popularity INTEGER, odds REAL, time TEXT, margin TEXT,
                        last_3f REAL
                    );
                    """
                )
                for race_id, date in enumerate(
                    ("2020-01-01", "2020-01-02", "2020-01-03"), start=1
                ):
                    conn.execute(
                        "INSERT INTO races VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            race_id,
                            date,
                            "Tokyo",
                            1600,
                            "Turf",
                            "Left",
                            "Test Stakes",
                            11,
                            "Sunny",
                        ),
                    )
                    for number in range(1, 5):
                        conn.execute(
                            """
                            INSERT INTO horses VALUES (
                                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                ?, ?, ?
                            )
                            """,
                            (
                                race_id * 10 + number,
                                race_id,
                                number,
                                1,
                                number,
                                f"Horse {number}",
                                f"牡{number + 2}",
                                56.0,
                                f"Jockey {number}",
                                f"Trainer {number}",
                                470 + number,
                                number - 2,
                                number,
                                None,
                                "1:30.0",
                                None,
                                34.0,
                            ),
                        )

            original_db_path = train_model.DB_PATH
            train_model.DB_PATH = database_path
            try:
                data, features, categorical, numeric = train_model.load_data()
            finally:
                train_model.DB_PATH = original_db_path

        self.assertEqual(data["race_id"].nunique(), 3)
        self.assertIn("popularity", numeric)
        self.assertIn("body_weight", numeric)
        self.assertIn("weather", categorical)
        self.assertNotIn("finish", features)
        self.assertNotIn("time", features)
        self.assertNotIn("margin", features)
        self.assertNotIn("last_3f", features)
        self.assertTrue(data["horse_prior_starts"].ge(0).all())

    def test_top_three_metrics_rank_each_race_separately(self):
        predictions = pd.DataFrame(
            {
                "race_id": [1, 1, 1, 1, 2, 2, 2, 2],
                "horse_number": [1, 2, 3, 4, 1, 2, 3, 4],
                "finish": [4, 1, 2, 3, 3, 2, 1, 4],
                "score": [0.1, 0.9, 0.8, 0.7, 0.7, 0.8, 0.9, 0.1],
            }
        )

        metrics = train_model._top_three_metrics(predictions, "score")

        self.assertEqual(metrics["races"], 2)
        self.assertEqual(metrics["mean_correct_picks"], 3)
        self.assertEqual(metrics["winner_hit_rate"], 1)
        self.assertEqual(metrics["exact_top3_set_rate"], 1)
        self.assertEqual(metrics["excluded_races"], 0)

    def test_top_three_metrics_excludes_races_without_complete_top_three(self):
        predictions = pd.DataFrame(
            {
                "race_id": [1, 1, 1, 1, 2, 2, 2],
                "horse_number": [1, 2, 3, 4, 1, 2, 3],
                "finish": [4, 1, 2, 3, 2, 3, 4],
                "score": [0.1, 0.9, 0.8, 0.7, 0.9, 0.8, 0.7],
            }
        )

        metrics = train_model._top_three_metrics(predictions, "score")

        self.assertEqual(metrics["races"], 1)
        self.assertEqual(metrics["excluded_races"], 1)
        self.assertEqual(metrics["winner_hit_rate"], 1)

    def test_popularity_baseline_excludes_whole_races_with_missing_popularity(self):
        class EqualProbabilityModel:
            def predict_proba(self, features):
                return np.full((len(features), 2), 0.5)

        test_data = pd.DataFrame(
            {
                "race_id": [1] * 4 + [2] * 4,
                "race_date": pd.to_datetime(
                    ["2025-01-01"] * 4 + ["2025-01-02"] * 4
                ),
                "venue": ["Tokyo"] * 8,
                "race_name": ["Test"] * 8,
                "horse_number": [1, 2, 3, 4] * 2,
                "finish": [1, 2, 3, 4] * 2,
                "target_top3": [1, 1, 1, 0] * 2,
                "popularity": [1, 2, 3, 4, np.nan, 1, 2, 3],
                "feature": range(8),
            }
        )

        metrics, _ = train_model.evaluate_model(
            EqualProbabilityModel(),
            test_data,
            ["feature"],
        )

        baseline = metrics["popularity_baseline"]
        self.assertEqual(baseline["races"], 1)
        self.assertEqual(baseline["excluded_races"], 0)
        self.assertEqual(baseline["excluded_incomplete_market_races"], 1)
        self.assertEqual(
            metrics["race_level_top3_on_market_coverage"]["races"],
            baseline["races"],
        )

    def test_prediction_output_breaks_score_ties_by_horse_number(self):
        class EqualProbabilityModel:
            def predict_proba(self, features):
                return np.full((len(features), 2), 0.5)

        test_data = pd.DataFrame(
            {
                "race_id": [1, 1, 1, 1],
                "race_date": pd.to_datetime(["2024-01-01"] * 4),
                "venue": ["Tokyo"] * 4,
                "race_name": ["Test"] * 4,
                "horse_number": [4, 2, 1, 3],
                "finish": [4, 2, 1, 3],
                "target_top3": [0, 1, 1, 1],
                "feature": [1, 2, 3, 4],
            }
        )

        _, predictions = train_model.evaluate_model(
            EqualProbabilityModel(),
            test_data,
            ["feature"],
        )
        ordered = predictions.sort_values("predicted_rank_in_race")

        self.assertEqual(
            ordered["horse_number"].tolist(),
            [1, 2, 3, 4],
        )

    def test_payout_simulation_reports_return_rate_for_win_quinella_and_trio(self):
        predictions = pd.DataFrame(
            {
                "race_id": [1, 1, 1, 1, 2, 2, 2, 2],
                "horse_number": [3, 1, 2, 4, 3, 1, 2, 4],
                "predicted_rank_in_race": [1, 2, 3, 4, 1, 2, 3, 4],
            }
        )
        payouts = pd.DataFrame(
            {
                "race_id": [1, 1, 1],
                "bet_type": ["単勝", "馬連", "3連複"],
                "combination": ["3", "1-3", "1-2-3"],
                "amount": [150, 400, 1200],
            }
        )

        result = train_model.simulate_payout_returns(predictions, payouts)

        self.assertEqual(result["単勝"]["return_rate"], 1.5)
        self.assertEqual(result["馬連"]["return_rate"], 4.0)
        self.assertEqual(result["3連複"]["return_rate"], 12.0)
        self.assertEqual(result["単勝"]["stake_yen"], 100)
        self.assertEqual(result["単勝"]["coverage"], 0.5)

    def test_evaluates_ranker_scores_and_reports_ndcg(self):
        class RankingModel:
            def predict(self, features):
                return features["feature"].to_numpy()

        test_data = pd.DataFrame(
            {
                "race_id": [1, 1, 1, 1],
                "race_date": pd.to_datetime(["2024-01-01"] * 4),
                "venue": ["Tokyo"] * 4,
                "race_name": ["Test"] * 4,
                "horse_number": [1, 2, 3, 4],
                "finish": [1, 2, 3, 4],
                "target_top3": [1, 1, 1, 0],
                "feature": [4, 3, 2, 1],
            }
        )

        metrics, _ = train_model.evaluate_model(
            RankingModel(),
            test_data,
            ["feature"],
        )

        self.assertEqual(metrics["race_level_top3"]["mean_correct_picks"], 3)
        self.assertEqual(metrics["race_level_top3"]["ndcg_at_3"], 1)


class UpsetFeatureTests(unittest.TestCase):
    def test_chronological_split_falls_back_when_configured_train_period_is_empty(self):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        data = pd.DataFrame(
            {
                "race_date": list(dates),
                "target_race_upset": [0, 1] * 10,
            }
        )

        train, validation, test, split = (
            train_upset_model.chronological_split(data)
        )

        self.assertEqual(
            split["strategy"],
            "chronological_date_proportions",
        )
        self.assertGreater(len(train), 0)
        self.assertGreater(len(validation), 0)
        self.assertGreater(len(test), 0)
        self.assertLess(train["race_date"].max(), validation["race_date"].min())
        self.assertLess(
            validation["race_date"].max(),
            test["race_date"].min(),
        )
        self.assertEqual(
            len(train) + len(validation) + len(test),
            len(data),
        )

    def test_chronological_split_keeps_configured_periods_when_available(self):
        dates = pd.to_datetime(
            ["2020-06-01", "2023-06-01", "2025-06-01"]
        )
        data = pd.DataFrame(
            {
                "race_date": dates,
                "target_race_upset": [0, 1, 0],
            }
        )

        train, validation, test, split = (
            train_upset_model.chronological_split(data)
        )

        self.assertEqual(split["strategy"], "configured_date_ranges")
        self.assertEqual(len(train), 1)
        self.assertEqual(len(validation), 1)
        self.assertEqual(len(test), 1)

    def test_selects_threshold_by_validation_balanced_accuracy(self):
        labels = pd.Series([0, 0, 1, 1, 0, 1])
        probabilities = np.array([0.1, 0.4, 0.35, 0.8, 0.45, 0.7])

        threshold = train_upset_model.choose_balanced_accuracy_threshold(
            labels,
            probabilities,
        )
        predictions = (probabilities >= threshold).astype(int)

        self.assertGreaterEqual(
            balanced_accuracy_score(labels, predictions),
            balanced_accuracy_score(labels, probabilities >= 0.5),
        )

    def test_upset_model_sigmoid_calibration_preserves_probability_contract(self):
        features = pd.DataFrame({"signal": np.linspace(-2, 2, 20)})
        labels = pd.Series([0] * 10 + [1] * 10)
        estimator = train_upset_model.make_pipeline([], ["signal"])
        estimator.fit(features, labels)
        raw_probability = estimator.predict_proba(features)[:, 1]
        calibrator = train_upset_model.fit_probability_calibrator(
            labels,
            raw_probability,
        )

        model = train_upset_model.SigmoidCalibratedModel(
            estimator,
            calibrator,
        )
        probabilities = model.predict_proba(features)

        self.assertEqual(probabilities.shape, (20, 2))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
        self.assertIs(
            model.named_steps["classifier"],
            estimator.named_steps["classifier"],
        )
        self.assertEqual(
            type(model).__module__,
            "scripts.model_calibration",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "upset_model.joblib"
            joblib.dump(model, model_path)
            scripts_directory = (
                Path(__file__).resolve().parents[1] / "scripts"
            )
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import joblib, sys; "
                        "import train_upset_model; "
                        "joblib.load(sys.argv[1])"
                    ),
                    str(model_path),
                ],
                cwd=scripts_directory,
                check=True,
                capture_output=True,
                text=True,
            )

    def test_aggregates_pre_race_field_without_using_current_popularity(self):
        races = pd.DataFrame(
            [
                {
                    "id": "1",
                    "race_id": "1",
                    "race_date": pd.Timestamp("2020-01-01"),
                    "venue": "Tokyo",
                    "race_name": "Same Stakes",
                    "surface": "Turf",
                    "distance_m": 1600,
                    "field_size": 3,
                    "target_race_upset": 1,
                },
                {
                    "id": "2",
                    "race_id": "2",
                    "race_date": pd.Timestamp("2020-01-02"),
                    "venue": "Tokyo",
                    "race_name": "Same Stakes",
                    "surface": "Turf",
                    "distance_m": 1600,
                    "field_size": 3,
                    "target_race_upset": 0,
                },
            ]
        )
        horses = pd.DataFrame(
            [
                {
                    "race_id": race_id,
                    "finish": finish,
                    "horse_number": number,
                    "horse_name": f"Horse {number}",
                    "sex_age": "牡4",
                    "frame": number,
                    "weight": 56.0,
                    "body_weight": 470 + number,
                    "weight_change": number - 2,
                    "jockey": f"Jockey {number}",
                    "trainer": f"Trainer {number}",
                    "popularity": number + 5,
                    "odds": None,
                }
                for race_id, finishes in (("1", (1, 2, 3)), ("2", (1, 2, 3)))
                for number, finish in enumerate(finishes, start=1)
            ]
        )
        horses["race_date"] = horses["race_id"].map(
            {"1": pd.Timestamp("2020-01-01"), "2": pd.Timestamp("2020-01-02")}
        )

        featured, categorical, numeric = train_upset_model.add_race_features(
            races,
            races[["id", "race_date"]].rename(columns={"id": "id"}),
            horses,
        )

        self.assertIn("field_body_weight_mean", numeric)
        self.assertIn("field_jockey_prior_top3_rate_mean", numeric)
        self.assertIn("race_name_prior_upset_rate", numeric)
        self.assertIn("venue", categorical)
        self.assertNotIn("race_name", categorical)
        self.assertNotIn("popularity", categorical + numeric)
        self.assertNotIn("odds", categorical + numeric)
        self.assertNotIn("finish", categorical + numeric)
        self.assertAlmostEqual(featured.loc[0, "field_body_weight_mean"], 472)
        self.assertEqual(featured.loc[0, "field_horse_prior_starts_mean"], 0)
        self.assertGreater(featured.loc[1, "field_horse_prior_starts_mean"], 0)
        self.assertEqual(featured.loc[0, "race_name_prior_races"], 0)
        self.assertEqual(featured.loc[1, "race_name_prior_races"], 1)
        self.assertEqual(featured.loc[1, "race_name_prior_upset_rate"], 1)


if __name__ == "__main__":
    unittest.main()
