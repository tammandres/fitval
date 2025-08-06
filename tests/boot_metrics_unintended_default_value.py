"""boot_metrics originally contains "sens" and "thr_risk" as arguments that have default values as lists.
Values are added to these lists during execution without copying them.
This changes the default values of the function so that when it is called again the added values are included.
Luckily, the function always computes "unique" so there's no repetitin of values, but it is still an unwanted consequence.
Potential solutions: 
1) When appending to default arg, create a copy.
2) Set default value to None -> and if it is none create the list insid ethe function, 
e.g. if sens is None: sens = [0.5, 0.8, 0.9]
"""
import pandas as pd
import numpy as np
import os
from fitval.dummydata import linear_data, induce_missing
from fitval.boot import boot_metrics
from constants import PROJECT_ROOT
import inspect

# Dummy data and path
x, y = linear_data(n=1500)
test_path = PROJECT_ROOT / 'tests' / 'test_boot'
test_path.mkdir(exist_ok=True, parents=True)

x = pd.DataFrame(x)
x = x.abs()
x.columns = ['v' + str(i) for i in range(x.shape[1])]
x = x.rename(columns={'v0': 'fit_val', 'v1': 'age_at_fit', 'v2': 'blood_MCV', 'v3': 'blood_PLT', 'v4': 'ind_gender_M'})
y = pd.DataFrame(y, columns=['crc'])

# Check default value for argument sens
signature = inspect.signature(boot_metrics)
parameters = signature.parameters
default_values = {k: v.default for k, v in parameters.items() if v.default is not inspect.Parameter.empty}
default_values['sens']

# Run
data_ci, data_noci = boot_metrics(x, y, model_names=['nottingham-lr', 'fit'], thr_risk=None, B=3, M=2, nchunks=1,
                                  repl_ypred_nan=True, save_path=test_path, save=False, parallel=False)

# Check default again: see that it has expanded
signature = inspect.signature(boot_metrics)
parameters = signature.parameters
default_values = {k: v.default for k, v in parameters.items() if v.default is not inspect.Parameter.empty}
default_values['sens']




