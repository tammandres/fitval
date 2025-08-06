import os
import shutil
from constants import PROJECT_ROOT


results_dir = PROJECT_ROOT / 'results'

target_dir = PROJECT_ROOT / 'airlock'
target_dir.mkdir(exist_ok=True, parents=True)


dirs = os.listdir(results_dir)
dirs = [d for d in dirs if 'agg' in d]

pats = ['reformat', 'plot', 'summary']

for d in dirs:
    print(d)
    files = os.listdir(results_dir / d)
    files = [f for f in files for p in pats if f.startswith(p)]

    dest = target_dir / d
    dest.mkdir(exist_ok=True, parents=True)

    for f in files:
        shutil.copyfile(results_dir / d / f, dest / f)


