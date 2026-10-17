
import sqlite3

DB_PATH = "/app/data/keiba.db"


def main():
    conn = sqlite3.connect(DB_PATH)

    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' ORDER BY name"
        ).fetchall()

        print("===== TABLES =====")
        for (table_name,) in tables:
            print(f"- {table_name}")

        print("\n===== SCHEMA =====")
        for (table_name,) in tables:
            print(f"\n--- TABLE: {table_name} ---")
            columns = conn.execute(
                f'PRAGMA table_info("{table_name}")'
            ).fetchall()

            for column in columns:
                # cid, name, type, notnull, default_value, primary_key
                print(
                    f"name={column[1]}, "
                    f"type={column[2]}, "
                    f"notnull={column[3]}, "
                    f"primary_key={column[5]}"
                )

            count = conn.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            ).fetchone()[0]
            print(f"row_count={count}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
