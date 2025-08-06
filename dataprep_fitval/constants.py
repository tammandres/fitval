from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
#DATA_DIR = Path("C:\\Users\\5lJC\\Desktop\\dataprep_fitml_and_fitval\\data_from_airlock") 
DATA_DIR = Path("C:\\Users\\5lJC\\Desktop\\dataprep_fitml_and_fitval\\data_parquet") 
DATAPREP_DIR = Path("C:\\Users\\5lJC\\Desktop\\dataprep_fitml_and_fitval\\dataprep") 
DATACUT = pd.to_datetime({'year':[2024], 'month':[2], 'day':[26]})
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
DATE_FORMAT = '%Y-%m-%d'
DATASET_VERSION = '2024-02-26'
PARQUET = True