"""
Task automation using invoke.

This module provides task automation for common workflows.

Usage:
    invoke preprocess
    invoke train --model coxph
    invoke predict --model coxph
    invoke test
"""

from invoke import task


@task
def download(c, dest="data/raw/mimic-iv-demo"):
    """Download MIMIC-IV Demo dataset."""
    c.run(f"python -m app.download_demo --dest {dest}")


@task
def preprocess(c):
    """Run data preprocessing pipeline."""
    c.run("python preprocess.py")


@task
def train(c, model="coxph"):
    """Train a survival model."""
    c.run(f"python train.py --model {model}")


@task
def predict(c, model="coxph"):
    """Generate predictions."""
    c.run(f"python predict.py --model models/{model}_model.pkl")


@task
def test(c):
    """Run unit tests."""
    c.run("pytest tests/ -v")


@task
def lint(c):
    """Run code quality checks."""
    c.run("flake8 app/ tests/")
    c.run("black --check app/ tests/")


@task
def format_code(c):
    """Format code with black."""
    c.run("black app/ tests/")

