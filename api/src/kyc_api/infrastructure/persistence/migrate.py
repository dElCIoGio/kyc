from __future__ import annotations

import argparse
import os

from alembic import command

from .database import migration_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the packaged KYC API database schema")
    parser.add_argument("command", choices=("upgrade", "downgrade"), nargs="?", default="upgrade")
    parser.add_argument("revision", nargs="?", default="head")
    args = parser.parse_args()
    database_url = os.environ.get("KYC_DATABASE_URL")
    if not database_url:
        parser.error("KYC_DATABASE_URL is required")
    config = migration_config(database_url)
    if args.command == "upgrade":
        command.upgrade(config, args.revision)
    else:
        command.downgrade(config, args.revision)


if __name__ == "__main__":
    main()
