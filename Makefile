.PHONY: help install test lint format clean download preprocess train predict results

help:
	@echo "Available targets:"
	@echo "  install       Install dependencies"
	@echo "  test          Run unit tests"
	@echo "  lint          Run code quality checks"
	@echo "  format        Format code with black"
	@echo "  clean         Clean generated files"
	@echo "  download      Download MIMIC-IV Demo data"
	@echo "  preprocess    Run data preprocessing"
	@echo "  train         Train models"
	@echo "  predict       Generate predictions"
	@echo "  results       Analyze results"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=app --cov-report=html

lint:
	flake8 app/ tests/
	black --check app/ tests/

format:
	black app/ tests/
	isort app/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

download:
	python -m app.download_demo --dest data/raw/mimic-iv-demo

preprocess:
	python preprocess.py

train:
	python train.py --model coxph

predict:
	python predict.py --model models/coxph_model.pkl

results:
	python result.py --predictions results/predictions/test_pred.csv

