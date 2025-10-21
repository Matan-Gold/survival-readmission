"""
Task automation using invoke.

This module provides task automation for common workflows with Weights & Biases integration.

Usage:
    invoke preprocess
    invoke train --model cox --wandb-project survival-readmission
    invoke train-all --wandb-project survival-readmission
    invoke predict --model cox
    invoke result --model cox
    invoke test
"""

from invoke import task


@task
def download(c, dest="data/raw/mimic-iv-demo"):
    """Download MIMIC-IV Demo dataset."""
    c.run(f"python -m app.download_demo --dest {dest}")


@task
def preprocess(c, input_dir="data/raw/mimic-iv-demo", output_dir="data/processed"):
    """Run data preprocessing pipeline."""
    c.run(f"python preprocess.py --input {input_dir} --output {output_dir}")


@task
def train(c, model="cox", data="data/processed", output=None, wandb_project="survival-readmission", 
          wandb_run=None, **kwargs):
    """Train a survival model."""
    cmd = f"python train.py --model {model} --data {data}"
    
    if output:
        cmd += f" --output {output}"
    
    cmd += f" --wandb-project {wandb_project}"
    
    if wandb_run:
        cmd += f" --wandb-run {wandb_run}"
    
    # Add model-specific parameters
    for key, value in kwargs.items():
        if value is not None:
            cmd += f" --{key.replace('_', '-')} {value}"
    
    c.run(cmd)


@task
def train_all(c, data="data/processed", wandb_project="survival-readmission"):
    """Train all models sequentially."""
    models = ["cox", "xgb-cox", "xgb-aft", "rf"]
    
    for model in models:
        print(f"Training {model}...")
        c.run(f"python train.py --model {model} --data {data} --wandb-project {wandb_project}")
        print(f"Completed {model}")


@task
def predict(c, model="cox", data="data/processed/features.csv", output="results/predictions.csv"):
    """Generate predictions."""
    model_path = f"models/{model}_model.pkl"
    c.run(f"python predict.py --model {model_path} --data {data} --output {output}")


@task
def result(c, model="cox", predictions="results/predictions.csv", targets="data/processed/targets.csv", 
          output="results/analysis"):
    """Analyze results and generate reports."""
    c.run(f"python result.py --predictions {predictions} --targets {targets} --output {output} --model-name {model}")


@task
def full_pipeline(c, model="cox", data="data/processed", wandb_project="survival-readmission"):
    """Run full pipeline: preprocess -> train -> predict -> result."""
    print("Running full pipeline...")
    
    # 1. Preprocess
    print("1. Preprocessing...")
    c.run("python preprocess.py")
    
    # 2. Train
    print(f"2. Training {model}...")
    c.run(f"python train.py --model {model} --data {data} --wandb-project {wandb_project}")
    
    # 3. Predict
    print(f"3. Generating predictions for {model}...")
    c.run(f"python predict.py --model models/{model}_model.pkl --data {data}/features.csv --output results/{model}_predictions.csv")
    
    # 4. Result
    print(f"4. Analyzing results for {model}...")
    c.run(f"python result.py --predictions results/{model}_predictions.csv --targets {data}/targets.csv --output results/{model}_analysis --model-name {model}")
    
    print("Full pipeline complete!")


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


@task
def clean(c):
    """Clean up generated files."""
    c.run("rm -rf models/*.pkl")
    c.run("rm -rf results/predictions/*")
    c.run("rm -rf results/analysis/*")
    c.run("rm -rf data/processed/*.csv")
    print("Cleanup complete!")

