import pandas as pd
import numpy as np
from fitval.models import get_model, MODEL_NAMES


def test_instantiate_model():
    # Test that prediction functions can be returned
    for m in MODEL_NAMES:
        print(m)
        model = get_model(m)
        assert model


def test_apply_model():
    # Test that prediction functions can be applied, and result is in [0, 1]

    rng = np.random.default_rng(seed=42)
    columns = ['age_at_fit', 'fit_val', 'blood_PLT', 'blood_MCV', 'ind_gender_M', 'blood_HGB', 'blood_CFER', 
              'blood_FEN']  #, 'blood_TRSAT']
    df = rng.normal(loc=100, scale=10, size=(10, len(columns)))
    df = pd.DataFrame(df, columns=columns)
    for m in MODEL_NAMES:
        print(m)
        model = get_model(m)
        y_pred = model(df)
        test = np.logical_or(y_pred <= 1, y_pred >= 0)
        assert test.all()
