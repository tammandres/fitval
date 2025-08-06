from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
DATAPREP_DIR = PROJECT_ROOT / 'dataprep'
RESULTS_DIR = PROJECT_ROOT / 'results'
DATACUT = pd.to_datetime({'year':[2024], 'month':[2], 'day':[26]})
