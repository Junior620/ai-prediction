.PHONY: help install test lint format clean run-api run-redis init-db

help:
	@echo "Cocoa Price Prediction System - Available Commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run linters"
	@echo "  make format      - Format code"
	@echo "  make clean       - Clean temporary files"
	@echo "  make run-api     - Start FastAPI server"
	@echo "  make run-redis   - Start Redis server"
	@echo "  make init-db     - Initialize Supabase database"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-redis:
	redis-server config/redis_config.conf

init-db:
	@echo "Please run the SQL script in your Supabase SQL Editor:"
	@echo "File: config/supabase_init.sql"
