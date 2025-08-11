import numpy as np
import pandas as pd


def linear_data(n: int = 1000, p_pred: int = 10, p_noise: int = 10, bias: float = 0, beta: float = 0.5,
                seed: int = 42):
    """Simulate binary classification data from a linear (logistic) model
    Args:
        n: number of observations
        p_pred: number of relevant predictor variables (regression coef > 0)
        p_noise: number of 'noise' variables (regression coef = 0)
        bias: bias term
        beta: regression coefficient for relevant predictor variables
        seed: random number seed
    """
    rng = np.random.default_rng(seed=seed)

    # Sample n values for each of the p_pred + p_noise predictor variables, i.i.d ~ N(0,1)
    X = rng.normal(loc=0, scale=1, size=(n, p_pred + p_noise))

    # Set regression coefficients of relevant predictor variables to beta, and of noise variables to 0
    b_pred = np.ones((p_pred, 1))*beta
    b_noise = np.zeros((p_noise, 1))
    b = np.concatenate((b_pred, b_noise), axis=0)

    # Compute linear predictor (bias + Xb), pass through sigmoid to get probability
    f = bias + np.matmul(X, b).squeeze()
    prob = 1/(1+np.exp(-f))

    # Sample n values from Bernoulli
    y = rng.binomial(n=1, p=prob, size=(n,))
    m = '{} samples, {} positive cases, {} relevant predictors, {} noise predictors'
    print(m.format(n, y.sum(), p_pred, p_noise))
    return X, y


def induce_missing(x: np.ndarray, seed=42):
    rng = np.random.default_rng(seed=seed)
    mask = rng.binomial(n=1, p=0.3, size=x.shape)
    x[mask == 1] = np.nan
    print('Number of missing values: {}'.format(np.isnan(x).sum()))
    return x


def dummy_fit_data(n=250, bias=-6, random_state=42):
    
    # Dummy predictor variables
    rng = np.random.default_rng(seed=random_state)
    fit_val = rng.lognormal(mean=1.2, sigma=1.75, size=n)
    ind_gender = rng.binomial(n=1, p=0.4, size=n)
    blood_mcv = rng.gamma(shape=10, scale=3, size=n)
    blood_mcv = np.mean(blood_mcv) + 90 - blood_mcv
    blood_plt = rng.gamma(shape=1.2, scale=170, size=n)
    age = rng.normal(loc=70, scale=10, size=n)

    # Dummy outcome variable
    c = pd.DataFrame({'fit_val': fit_val, 'age': age, 'blood_mcv': blood_mcv, 'blood_plt': blood_plt})
    c = np.log(c + 1)
    c = (c - c.median()) / (c.quantile(0.75) - c.quantile(0.25))
    c['ind_gender'] = ind_gender
    lin = bias + c.fit_val * 2. + c.age * 0.5 + c.ind_gender * 0.5 + c.blood_plt * 1. + c.blood_mcv * 1.
    prob = 1 / (1 + np.exp(-lin))
    crc = rng.binomial(n=1, p=prob, size=n)

    # Collect to dataframe
    df = pd.DataFrame({'y_true': crc, 'fit_val': fit_val, 'age': age, 'ind_gender_M': ind_gender, 'blood_MCV': blood_mcv,
                       'blood_PLT': blood_plt})
    
    print(df.describe())
    return df
    