import json
import sqlite3
import unittest

from scripts.database import initialize_schema
from scripts.predict_upcoming import evaluate_saved_predictions


class SavedPredictionEvaluationTests(unittest.TestCase):
    def test_evaluates_mature_predictions_once_and_records_payout_returns(self):
        with sqlite3.connect(":memory:") as connection:
            initialize_schema(connection)
            connection.execute(
                "INSERT INTO races (id, race_date) VALUES (1, '2026-01-03')"
            )
            connection.executemany(
                """
                INSERT INTO horses (race_id, horse_number, finish)
                VALUES (1, ?, ?)
                """,
                [(1, 1), (2, 2), (3, 3), (4, 4)],
            )
            connection.executemany(
                """
                INSERT INTO payouts (race_id, bet_type, combination, amount)
                VALUES (1, ?, ?, ?)
                """,
                [
                    ("単勝", "1", 300),
                    ("馬連", "1-2", 500),
                    ("3連複", "1-2-3", 800),
                ],
            )
            ranked = [
                {"rank": number, "horse_number": horse_number}
                for number, horse_number in enumerate((1, 2, 4, 3), start=1)
            ]
            connection.execute(
                """
                INSERT INTO predictions (
                    race_id, predicted_at, model_version, top3_predictions
                ) VALUES (1, '2026-01-01T10:00:00+09:00', 'model-v1', ?)
                """,
                (json.dumps(ranked),),
            )

            results = evaluate_saved_predictions(connection)
            evaluated_count = connection.execute(
                "SELECT COUNT(*) FROM prediction_evaluations"
            ).fetchone()[0]
            row_evaluated_at = connection.execute(
                "SELECT evaluated_at FROM predictions"
            ).fetchone()[0]
            second_run = evaluate_saved_predictions(connection)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["mean_correct_picks"], 2)
        self.assertEqual(
            results[0]["payout_simulation"]["馬連"]["return_rate"],
            5.0,
        )
        self.assertEqual(evaluated_count, 1)
        self.assertIsNotNone(row_evaluated_at)
        self.assertEqual(second_run, [])


if __name__ == "__main__":
    unittest.main()
