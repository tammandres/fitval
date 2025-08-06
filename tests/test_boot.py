import pandas as pd
import numpy as np
import os
from fitval.dummydata import linear_data, induce_missing
from fitval.boot import boot_metrics
from constants import PROJECT_ROOT


# Dummy data and path
x, y = linear_data(n=500, p_pred=5, p_noise=0)
test_path = PROJECT_ROOT / 'tests' / 'test_boot'
test_path.mkdir(exist_ok=True, parents=True)

x = pd.DataFrame(x)
x = x.abs()
x.columns = ['v' + str(i) for i in range(x.shape[1])]
x = x.rename(columns={'v0': 'fit_val', 'v1': 'age_at_fit', 'v2': 'blood_MCV', 'v3': 'blood_PLT', 'v4': 'ind_gender_M'})
y = pd.DataFrame(y, columns=['crc'])

xmis = induce_missing(x.copy(), seed=42)
xmis.fit_val = x.fit_val


# Run with some of the different settings
def test_boot_metrics():
    data_ci, data_noci = boot_metrics(x, y, model_names=['nottingham-lr', 'fit'], B=3, M=2,
                                      repl_ypred_nan=True, save_path=test_path, save=True)

    data_ci, data_noci = boot_metrics(x, y, model_names=['nottingham-lr', 'nottingham-cox', 'fit'], B=4, M=2,
                                      repl_ypred_nan=True, parallel=True, save_path=test_path, save=False)

    data_ci, data_noci = boot_metrics(x, y, model_names=['nottingham-lr', 'fit'], B=3, M=2,
                                      repl_ypred_nan=True, global_only=True, save_path=test_path, save=False)

    data_ci, data_noci = boot_metrics(xmis, y, model_names=['nottingham-lr', 'fit'], B=3, M=2,
                                      repl_ypred_nan=True, save_path=test_path, save=False)

    data_ci, data_noci = boot_metrics(x, y, model_names=['nottingham-lr', 'fit'], B=1, M=2,
                                      repl_ypred_nan=True, global_only=True, save_path=test_path, save=False,
                                      dca=False)
