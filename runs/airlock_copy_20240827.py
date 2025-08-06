import os
import shutil
from constants import PROJECT_ROOT


target_dir = PROJECT_ROOT / 'airlock' / 'airlock_20240827'
target_dir.mkdir(exist_ok=True, parents=True)


results_dir = PROJECT_ROOT / 'results' / 'agg'
dirs = os.listdir(results_dir)
dirs = [d for d in dirs if 'timecut' in d]


# Copy updated plots (also contain 'all data' in addition to the six time periods)
pats = ['plot_roc', 'plot_cal', 'plot_dca']

for d in dirs:
    print(d)

    files = os.listdir(results_dir / d)
    files = [f for f in files for p in pats if f.startswith(p)]

    dest = target_dir / d
    dest.mkdir(exist_ok=True, parents=True)

    for f in files:
        shutil.copyfile(results_dir / d / f, dest / f)


# Copy updated test reduction plots (different thr methods on one graph)
results_dir = PROJECT_ROOT / 'results' / 'reduction_thr-external-local'
pats = ['plot_reduction_model-cox', 'reformat_model-cox']

files = os.listdir(results_dir)
files = [f for f in files for p in pats if f.startswith(p)]

dest = target_dir / 'reduction_thr-external-local'
dest.mkdir(exist_ok=True, parents=True)

for f in files:
    shutil.copyfile(results_dir / f, dest / f)