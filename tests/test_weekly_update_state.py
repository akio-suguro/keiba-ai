import sqlite3
import unittest

from scripts.weekly_update_state import mark_completed, should_run


class WeeklyUpdateStateTests(unittest.TestCase):
    def test_completion_prevents_duplicate_weekly_run(self):
        with sqlite3.connect(":memory:") as connection:
            self.assertTrue(should_run(connection, "2026-W45"))

            mark_completed(connection, "2026-W45")

            self.assertFalse(should_run(connection, "2026-W45"))
            self.assertTrue(should_run(connection, "2026-W46"))


if __name__ == "__main__":
    unittest.main()
