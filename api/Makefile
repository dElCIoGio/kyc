.PHONY: install-library run

ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)

install-library:
	python -m pip install -e "$(ROOT_DIR)[ocr]"

run:
	uvicorn kyc_api.main:app --app-dir api/src --host 127.0.0.1 --port 8000 --no-access-log
