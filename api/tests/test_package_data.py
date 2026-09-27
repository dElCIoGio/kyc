from __future__ import annotations

from pathlib import Path
import tomllib
import unittest


class PackageDataTests(unittest.TestCase):
    def test_verifier_distribution_files_are_declared_as_package_data(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        configuration = tomllib.loads((project_root / "pyproject.toml").read_text("utf-8"))
        package_data = configuration["tool"]["setuptools"]["package-data"]["kyc_api"]

        self.assertIn("verify/dist/index.html", package_data)
        self.assertIn("verify/dist/assets/**/*", package_data)


if __name__ == "__main__":
    unittest.main()
