"""Crawl current-month results and published entries for upcoming races."""

import sys

if __package__:
    from . import crawl_jra
else:
    import crawl_jra


def main() -> int:
    """Run the current-month crawler without the full crawler's size cap."""
    return crawl_jra.main(
        current_month_only=True,
        enforce_database_size_limit=False,
    )


if __name__ == "__main__":
    sys.exit(main())
