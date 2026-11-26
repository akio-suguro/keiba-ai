import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.model_artifacts import (
    MODEL_ARTIFACTS,
    RELEASE_DOWNLOAD_BASE,
    download_model_artifacts,
)


class ModelArtifactDownloadTests(unittest.TestCase):
    @patch("scripts.model_artifacts.requests.get")
    def test_downloads_assets_to_project_paths(self, get):
        get.return_value.content = b"artifact"
        get.return_value.raise_for_status.return_value = None

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.assertTrue(download_model_artifacts(root))

            for asset_name, relative_path in MODEL_ARTIFACTS.items():
                self.assertEqual(
                    (root / relative_path).read_bytes(),
                    b"artifact",
                )
            self.assertEqual(
                [call.args[0] for call in get.call_args_list],
                [
                    f"{RELEASE_DOWNLOAD_BASE}/{asset_name}"
                    for asset_name in MODEL_ARTIFACTS
                ],
            )

        self.assertEqual(get.call_count, len(MODEL_ARTIFACTS))

    @patch("scripts.model_artifacts.requests.get")
    def test_download_failure_keeps_existing_model_set(self, get):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for relative_path in MODEL_ARTIFACTS.values():
                destination = root / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"old")

            response = Mock(content=b"new")
            get.side_effect = [response, RuntimeError("download failed")]

            with self.assertRaisesRegex(RuntimeError, "download failed"):
                download_model_artifacts(root)

            for relative_path in MODEL_ARTIFACTS.values():
                self.assertEqual(
                    (root / relative_path).read_bytes(),
                    b"old",
                )
            self.assertFalse(list(root.rglob("*.tmp")))

    @patch("scripts.model_artifacts.requests.get")
    def test_missing_release_does_not_replace_cached_artifacts(self, get):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for relative_path in MODEL_ARTIFACTS.values():
                destination = root / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"old")

            response = Mock(status_code=404)
            get.side_effect = [
                Mock(status_code=200, content=b"new"),
                response,
            ]

            self.assertFalse(download_model_artifacts(root))

            for relative_path in MODEL_ARTIFACTS.values():
                self.assertEqual(
                    (root / relative_path).read_bytes(),
                    b"old",
                )
            self.assertFalse(list(root.rglob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
