"""Verify that a built API wheel contains the hosted verifier production bundle."""

from __future__ import annotations

import sys
from pathlib import Path
from zipfile import ZipFile


def main(wheel_path: str) -> None:
    with ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())

    index = "kyc_api/verify/dist/index.html"
    javascript = sorted(
        name
        for name in names
        if name.startswith("kyc_api/verify/dist/assets/") and name.endswith(".js")
    )
    if index not in names or not javascript:
        missing = []
        if index not in names:
            missing.append(index)
        if not javascript:
            missing.append("kyc_api/verify/dist/assets/*.js")
        raise SystemExit(f"wheel is missing verifier production assets: {', '.join(missing)}")

    print(f"verified {Path(wheel_path).name}: {index}, {javascript[0]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_verifier_wheel.py WHEEL")
    main(sys.argv[1])
