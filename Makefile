.PHONY: install train seed test api docker-build docker-run clean

install:
	pip install -r requirements.txt

train:
	python -m src.train --epochs 5

train-quick:
	python -m src.train --quick

seed:
	python -m src.seed_catalog

test:
	python tests/test_smoke.py

api:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t product-recognition:latest .

docker-run:
	docker run --rm -p 8000:8000 product-recognition:latest

clean:
	rm -rf data/ models/*.pth catalog/*.db __pycache__ src/__pycache__ tests/__pycache__
