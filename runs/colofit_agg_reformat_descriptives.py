"""Create aggregated descriptives table"""
from pathlib import Path
import numpy as np
import pandas as pd
import os
import re
from constants import PROJECT_ROOT

# Get directories in which to reformat tables
run_path = PROJECT_ROOT / 'results' / 'agg'
dirs = os.listdir(run_path)

for dir in dirs:
    print('\n', dir)

    df = pd.read_csv(run_path / dir / 'summary_table.csv')

    # Check uniqueness of index
    df['idx'] = df.group + df.characteristic
    for p in df.period.unique():
        dfsub = df.loc[df.period == p]
        assert dfsub.shape[0] == dfsub.idx.nunique()

    # Time label
    period_type = re.findall('^[a-z]+', dir)[0]
    fu = int(re.findall('fu-(\d+)', dir)[0])

    if period_type == 'prepost':
        if fu == 365:
            time_labels = {
                        'all': 'All data (2017/01 - 2023/02)',
                        'prebuff': 'Pre-buffer data (2017/01 - 2021/06)',
                        'bufftime': 'Buffer data from time (2021/10 - 2023/02)',
                        'buffcomm': 'Buffer data from comment (2022/04 - 2023/02)',
                        }
        else:
            time_labels = {
                        'all': 'All data (2017/01 - 2023/08)',
                        'prebuff': 'Pre-buffer data (2017/01 - 2021/06)',
                        'bufftime': 'Buffer data from time (2021/10 - 2023/08)',
                        'buffcomm': 'Buffer data from comment (2022/04 - 2023/08)',
                        }
    else:
        time_labels = {
                'precovid': 'Pre-COVID (2017/01 - 2020/02)',
                'covid': 'COVID (2020/03 - 2021/04)',
                'post1': 'Post-COVID (2021/05 - 2021/12)',
                'post2': '2022 H1 (2022/01 - 2022/06)',
                'post3': '2022 H2 (2022/07 - 2022/12)',
                'post4': '2023 H1 (2023/01 - 2023/06)',
                'post3-buffcomm': '2022 H2 buffer only (2022/07 - 2022/12)',
                'post4-buffcomm': '2023 H1 buffer only (2023/01 - 2023/06)'
                }

    df.period = df.period.replace(time_labels)
    nperiod = df.period.nunique()
    
    # Pivot
    df = df.pivot(index=['idx', 'group', 'characteristic'], columns=['period'], 
                  values=['No colorectal cancer', 'Colorectal cancer'])
    
    # Reorder columns
    new_col_order = [[i, i + nperiod] for i in range(nperiod)]
    new_col_order = [item for sublist in new_col_order for item in sublist]
    df = df.iloc[:, new_col_order]

    df = df.reset_index()
    df = df.fillna('')

    df = df.rename(columns={'characteristic': 'Characteristic'}) # Add capital C

    # Save
    df.to_csv(run_path / dir / 'reformat_summary_table.csv', index=False)
