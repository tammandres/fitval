import numpy as np
import pandas as pd
import shutil
import os
from pathlib import Path


data_path = Path("C:/Users/5lJC/Desktop/dataprep_fitml_and_fitval/airlock_import")
os.listdir(data_path)

src1 = data_path / 'fit_airlock_part1_20240619'
items1 = [i for i in os.listdir(src1) if i not in ['gpreqs', 'demographics']]
print(items1)

src2 = data_path / 'fit_airlock_part2_20240620'
items2 = os.listdir(src2)
print(items2)

# Copy
target_path = Path("C:/Users/5lJC/Desktop/dataprep_fitml_and_fitval/data_parquet")
target_path.mkdir(exist_ok=True)

for i in items1:
    shutil.copytree(src1 / i, target_path / i)

for i in items2:
    shutil.copytree(src2 / i, target_path / i)

gpreqs_dest_path = target_path / 'gpreqs'
gpreqs_dest_path.mkdir(exist_ok=True)
fname = 'gpreqs_1.parquet'
shutil.copyfile(data_path / fname, gpreqs_dest_path / fname)

demo_dest_path = target_path / 'demographics'
demo_dest_path.mkdir(exist_ok=True)
fname = 'demographics_1.parquet'
shutil.copyfile(data_path / fname, demo_dest_path / fname)
