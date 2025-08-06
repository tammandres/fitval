import numpy as np


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
    