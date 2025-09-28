from pathlib import Path
from mlops.data_checks import run_all

if __name__ == "__main__":
    run_all(Path("data/processed"), Path("data/artifacts/analytics/data_checks.json"))
