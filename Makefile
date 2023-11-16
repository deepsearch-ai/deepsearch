# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PROJECT_NAME := deepsearchai

# Targets
.PHONY: install format lint clean test ci_lint ci_test coverage

install:
	poetry install

install_all:
	poetry install --all-extras

shell:
	poetry shell

py_shell:
	poetry run python

format:
	$(PYTHON) -m black .
	$(PYTHON) -m isort .

clean:
	rm -rf dist build *.egg-info

lint:
	poetry run ruff .

test:
	poetry run pytest $(file)

coverage:
	poetry run pytest --cov=$(PROJECT_NAME) --cov-report=html