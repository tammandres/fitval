"""Aggregate metric tables computed for each time period"""
import numpy as np
import pandas as pd
from constants import PROJECT_ROOT
from pathlib import Path
import os
import re

# Get directories to aggregate
run_path = PROJECT_ROOT / 'results'
dirs = os.listdir(run_path)
dirs = [d for d in dirs if re.search('^(?:timecut|prepost)_fu-(?:365|180)$', d)]

# Get files to aggregate
for dir in dirs:
    print("Processing dir", dir)

    # Subdirectories to aggregate
    subdirs = os.listdir(run_path / dir)
    subdirs = [subdir for subdir in subdirs]

    # Dbl check that all subdirs have same files
    files = list(set([f for subdir in subdirs for f in os.listdir(run_path / dir / subdir)]))
    files = [f for f in files if f.endswith('.csv')]
    test = [f in os.listdir(run_path / dir / subdir) for subdir in subdirs for f in files]

    if not all(test):
        print("Some files are missing:")
        for f in files:
            for subdir in subdirs:
                if not f in os.listdir(run_path / dir / subdir):
                    print(f, subdir)
        raise ValueError("Missing files.")

    # Period labels
    periods = [re.findall(r'period-(.+)(?:$|_)', subdir)[0] for subdir in subdirs]
    assert len(periods) == len(subdirs)

    # Save path
    out_dir = dir + '_agg'
    out_path = run_path / 'agg' / out_dir
    out_path.mkdir(exist_ok=True, parents=True)
    print('...saving aggregated file to', out_path)

    # Aggregate
    for f in files:
        print('- aggregating', f)

        df_agg = pd.DataFrame()
        empty_count = 0
        for period, subdir in zip(periods, subdirs):
            file_path = run_path / dir / subdir / f
            try:  # Try reading the file
                df = pd.read_csv(file_path)
                df['period'] = period
                df_agg = pd.concat(objs=[df_agg, df], axis=0)
            except pd.errors.EmptyDataError:  # If the csv file is empty
                empty_count += 1
            
        if empty_count > 0:
            assert empty_count == len(periods)  ## To dbl check that if there were empty files, all were empty
        
        if df_agg.shape[0] > 0:
            df_agg.to_csv(out_path / f, index=False)
            print('-- saved.')
        else:
            print("-- input files were empty")
