"""HTTP orchestration service for the Angolan KYC engine."""

__version__ = "0.1.0"

from .main import create_app

__all__ = ["create_app"]
