from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import requests


RELEASE_DOWNLOAD_BASE = (
    "https://github.com/akio-suguro/keiba-ai/releases/latest/download"
)
MODEL_ARTIFACTS = {
    "top3_model.joblib": Path("top3_model.joblib"),
    "upset_model.joblib": Path("upset_model.joblib"),
    "upset_metrics.json": Path("upset_metrics.json"),
}


def download_model_artifacts(
    cache_dir: Path,
    *,
    timeout: int = 30,
) -> None:
    """Download one published release's model artifacts to the project."""
    downloads: list[tuple[Path, Path]] = []
    try:
        for asset_name, relative_path in MODEL_ARTIFACTS.items():
            response = requests.get(
                f"{RELEASE_DOWNLOAD_BASE}/{asset_name}",
                timeout=timeout,
            )
            response.raise_for_status()
            if not response.content:
                raise RuntimeError(
                    f"GitHub Release asset is empty: {asset_name}"
                )

            destination = cache_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                downloads.append((temporary_path, destination))
                temporary.write(response.content)

        for temporary_path, destination in downloads:
            temporary_path.replace(destination)
    finally:
        for temporary_path, _ in downloads:
            temporary_path.unlink(missing_ok=True)
