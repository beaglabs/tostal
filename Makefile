.PHONY: dev dev-api dev-gateway build test lint typecheck clean migrate train-facies train-segmenter evaluate

dev: dev-api dev-gateway

dev-api:
	uvicorn api.app.main:app --reload --host 0.0.0.0 --port 8000

dev-gateway:
	cd gateway && npm run dev

build:
	docker compose build

test:
	cd api && pytest tests/ -v

lint:
	cd api && ruff check app/ tests/
	cd gateway && npm run lint

typecheck:
	cd api && mypy app/
	cd gateway && npm run typecheck

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name node_modules -exec rm -rf {} +

migrate:
	cd api && alembic upgrade head

migrate-new:
	cd api && alembic revision --autogenerate -m "$(msg)"

seed:
	cd api && python -m storage.scripts.seed_data

docker-up:
	docker compose up -d

docker-down:
	docker compose down

train-facies:
	python training/scripts/train_xgboost_facies.py

train-segmenter:
	python training/scripts/train_segmenter.py

evaluate:
	python training/scripts/evaluate.py