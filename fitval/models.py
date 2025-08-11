"""Prediction models to be validated"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, SplineTransformer 


def _sigmoid(x):
    return 1/(1 + np.exp(-x))


# ---- Add your models here and update the get_model function accordingly
MODEL_NAMES = ['logistic-full', 'logistic-fit-age']  # list all valid model names that can be returned by the get_model function

def logistic_full(X: pd.DataFrame):
    """Dummy model that computes a risk score using columns in DataFrame X"""
    linear_predictor = -8.9 + 0.003 * X.fit_val + 0.066 * X.age + 1.1 * X.ind_gender_M - 0.00367 * X.blood_MCV + 0.000014 * X.blood_PLT
    prob = _sigmoid(linear_predictor)
    return prob


def logistic_fit_age(X: pd.DataFrame):
    """Dummy model that computes a risk score using columns in DataFrame X"""
    linear_predictor = -8.9 + 0.003 * X.fit_val + 0.066 * X.age
    prob = _sigmoid(linear_predictor)
    return prob


def get_model(model_name: str):
    """Get model function (input data -> probability) based on model name"""
    if model_name not in MODEL_NAMES:
        raise ValueError("model_name must be in " + str(MODEL_NAMES))

    if model_name == 'logistic-full':
        return logistic_full
    elif model_name == 'logistic-fit-age':
        return logistic_fit_age


# ---- Code for FIT-only spline model that will be fitted to the data to estimate risk corresponding to FIT = 10
def create_spline_model(fit: np.ndarray, crc: np.ndarray, knots: list = [10, 100]):
    """Predicts cancer (crc) from FIT test values (fit)
    by log-transforming FIT values, applying a spline transformation,
    and fitting a logistic regression model"""

    knots = [10, 100]
    knots = np.array([np.log(k + 1) for k in knots]).reshape(-1, 1)

    pipe = make_pipeline(FunctionTransformer(lambda z: z.reshape(-1, 1)),
                         FunctionTransformer(np.log1p), 
                         SplineTransformer(degree=2, knots=knots, include_bias=False, extrapolation='linear'),
                         LogisticRegression(penalty=None))
    pipe.fit(fit, crc)

    return pipe
