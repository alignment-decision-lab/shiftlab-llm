import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import utils
from training_pipeline import run_training


if __name__ == "__main__":
    config = utils.load_config()
    run_training(config)