"""Reformat tables of performance metrics"""
from pathlib import Path
import numpy as np
import pandas as pd
import os
import re
from constants import PROJECT_ROOT
from fitval.reformat import reformat_disc_cal
from fitval.plots import plot_rocpr_interp, plot_cal_smooth


plot = False

# Get directories in which to reformat tables
run_path = PROJECT_ROOT / 'results'
dirs = os.listdir(run_path)
dirs = [d for d in dirs if re.search('^(?:timecut|prepost)_fu-(?:365|180)$', d)]

# Loop over directories
for dir in dirs:
    print("Processing dir", dir)

    # Subdirectories to process
    subdirs = os.listdir(run_path / dir)

    # Reformat
    for subdir in subdirs:
        print('-- Reformatting tables in', subdir)
        subdir_path = run_path / dir / subdir
        reformat_disc_cal(data_path=subdir_path, save_path=subdir_path)

        # Plot 
        if plot:
            for ci in [True, False]:
                plot_rocpr_interp(subdir_path, models_incl=['nottingham-cox', 'fit'], suf='_cox', ci=ci)
                plot_rocpr_interp(subdir_path, models_incl=['nottingham-cox', 'nottingham-lr', 'nottingham-cox-boot',
                                                            'nottingham-lr-boot', 'fit'], suf='_all', ci=ci)
                plot_rocpr_interp(subdir_path, models_incl=['nottingham-cox-3.5', 'nottingham-cox-platt', 'nottingham-cox-quant',
                                                            'fit'], suf='_recal', ci=ci)
                plot_rocpr_interp(subdir_path, models_incl=['nottingham-cox', 'nottingham-fit-age-sex',
                                                            'fit'], suf='_fast', ci=ci)

            plot_cal_smooth(subdir_path, models_incl=['nottingham-cox'], suf='_cox')
            plot_cal_smooth(subdir_path, models_incl=['nottingham-cox', 'nottingham-lr', 'nottingham-cox-boot',
                                                    'nottingham-lr-boot'], suf='_all')
            plot_cal_smooth(subdir_path, models_incl=['nottingham-cox-3.5', 'nottingham-cox-platt', 'nottingham-cox-quant',
                                                    'fit'], suf='_recal')
            
