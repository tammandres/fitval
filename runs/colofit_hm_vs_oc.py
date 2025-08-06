"""Compare difference in predicted risk when other predictors at median
but FIT 400 (max HM-Jack) vs 1146 (75th perc of CRC) or 4640 (90th perc of CRC in Nott)
"""
import pandas as pd
from fitval.models import get_model

model = get_model('nottingham-cox')


fit = [400, 1146, 4640, 8303, 18655, 69800]

nfit = len(fit)
data = {'fit_val': fit,
        'ind_gender_M': [1] * nfit,
        'age_at_fit': [74] * nfit,
        'blood_MCV': [89.1] * nfit,
        'blood_PLT': [305] * nfit}

data = pd.DataFrame(data)


data['risk_score'] = model(data) * 100
data
