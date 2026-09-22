.PHONY: install-library run docker-build docker-up docker-down

ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)

install-library:
	python -m pip install -e "$(ROOT_DIR)[ocr]"

run:
	uvicorn kyc_api.main:app --app-dir api/src --host 127.0.0.1 --port 8000 --no-access-log

docker-build:
	docker compose -f "$(ROOT_DIR)/docker-compose.yml" build api

docker-up:
	docker compose -f "$(ROOT_DIR)/docker-compose.yml" up --build

docker-down:
	docker compose -f "$(ROOT_DIR)/docker-compose.yml" down
