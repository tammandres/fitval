"""Prediction models to be validated"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, SplineTransformer 
from constants import PROJECT_ROOT


# Valid model names that can be submitted to get_model()
# Currently, platt-scaled models (and fit-ebm) cannot be retrieved using this function
# Instead, the models are implicitly created when running the recalibration function
MODEL_NAMES = ['nottingham-lr', 'nottingham-lr-boot', 'nottingham-cox', 'nottingham-cox-boot', 
               'nottingham-fit', 'nottingham-fit-age', 'nottingham-fit-age-sex', 
               'nottingham-lr-3.5', 'nottingham-lr-quant', 'nottingham-cox-3.5', 'nottingham-cox-quant',
               'nottingham-fit-3.5', 'nottingham-fit-quant',
               'nottingham-fit-age-3.5', 'nottingham-fit-age-quant',
               'nottingham-fit-age-sex-3.5', 'nottingham-fit-age-sex-quant',
               'fit-spline', 'fit']

# Path to Nottingham FIT distribution data
QUANT_NOTT = PROJECT_ROOT / 'data' / 'nottingham_quantiles.csv'

# csv files for checking model equivalence between this implementation and original
NOTT_TEST = 'model_equivalence_nottingham.csv'

# Model labels and colors for plots
model_labels = {'nottingham-lr': 'Nottingham-lr',
                'nottingham-lr-boot': 'Nottingham-lr-boot',
                'nottingham-cox': 'Nottingham-Cox',
                'nottingham-cox-boot': 'Nottingham-Cox-boot',

                'nottingham-lr-platt': 'Nottingham-lr-platt',
                'nottingham-lr-iso': 'Nottingham-lr-iso',
                'nottingham-lr-quant': 'Nottingham-lr-quant',
                'nottingham-lr-3.5': 'Nottingham-lr-3.5',
            
                'nottingham-cox-platt': 'Nottingham-Cox-platt',
                'nottingham-cox-iso': 'Nottingham-Cox-iso',
                'nottingham-cox-quant': 'Nottingham-Cox-quant',
                'nottingham-cox-3.5': 'Nottingham-Cox-3.5',

                'nottingham-fit': 'Nottingham-fit',
                'nottingham-fit-platt': 'Nottingham-fit-platt',
                'nottingham-fit-quant': 'Nottingham-fit-quant',
                'nottingham-fit-3.5': 'Nottingham-fit-3.5',

                'nottingham-fit-age-sex': 'Nottingham-fit-age-sex',
                'nottingham-fit-age-sex-platt': 'Nottingham-fit-age-sex-platt',
                'nottingham-fit-age-sex-quant': 'Nottingham-fit-age-sex-quant',
                'nottingham-fit-age-sex-3.5': 'Nottingham-fit-age-sex-3.5',

                'nottingham-fit-age': 'Nottingham-fit-age',
                'nottingham-fit-age-platt': 'Nottingham-fit-age-platt',
                'nottingham-fit-age-quant': 'Nottingham-fit-age-quant',
                'nottingham-fit-age-3.5': 'Nottingham-fit-age-3.5',

                'fit': 'FIT test',
                'fit-spline': 'FIT-spline',
                'fit-ebm': 'FIT-ebm'
                }

model_colors = {'nottingham-lr': 'C0',
                'nottingham-lr-boot': 'C8', #'C9',
                'nottingham-cox': 'C2',
                'nottingham-cox-boot': 'C4',

                'nottingham-lr-platt': 'C7', #'C5', #'C5', C8, 'olive'
                'nottingham-lr-quant': 'C8', #'C9', #'C7', #'C9', mediumseagreen
                'nottingham-lr-3.5': 'C9', #'C6', # C6,

                'nottingham-cox-platt': 'C0', # 'olive', #'C5', C8, 'olive'
                'nottingham-cox-quant': 'C1', #'C1', #'C7', #'C9', mediumseagreen
                'nottingham-cox-3.5': 'C2', # 'C7', # C6,

                'nottingham-lr-iso': 'springgreen',

                'nottingham-fit': 'springgreen',
                'nottingham-fit-age': 'orange',
                'nottingham-fit-age-sex': 'darkred',

                'nottingham-fit-platt': 'olive',
                'nottingham-fit-quant': 'darkgreen',
                'nottingham-fit-3.5': 'springgreen',

                'nottingham-fit-age-sex-platt': 'gray',
                'nottingham-fit-age-sex-quant': 'peachpuff',
                'nottingham-fit-age-sex-3.5': 'darkred',

                'nottingham-fit-age-platt': 'gray',
                'nottingham-fit-age-quant': 'gold',
                'nottingham-fit-age-3.5': 'orange',

                'fit': 'red',
                'fit-spline': 'red',
                'fit-ebm': 'red'
                }


#def _logit(x):
#    return np.log(x/(1 - x))


def _sigmoid(x):
    return 1/(1 + np.exp(-x))


def get_model(model_name: str):
    """Get model function (input data -> probability) based on model name"""
    if model_name not in MODEL_NAMES:
        raise ValueError("model_name must be in " + str(MODEL_NAMES))


    # ---- Main Nottingham models ----
    if model_name == 'nottingham-lr':
        return NottinghamLinearModel().forward_lr

    elif model_name == 'nottingham-lr-boot':
        return NottinghamLinearModel().forward_lr_boot

    elif model_name == 'nottingham-cox':
        return NottinghamLinearModel().forward_cox

    elif model_name == 'nottingham-cox-boot':
        return NottinghamLinearModel().forward_cox_boot
    

    # ---- Recalibration attempts of main models ----
    elif model_name == 'nottingham-lr-quant':
        return NottinghamLinearModel(quant_transform=True).forward_lr

    elif model_name == 'nottingham-lr-3.5':
         return NottinghamLinearModel(cfactor=3.5).forward_lr

    elif model_name == 'nottingham-cox-quant':
        return NottinghamLinearModel(quant_transform=True).forward_cox

    elif model_name == 'nottingham-cox-3.5':
         return NottinghamLinearModel(cfactor=3.5).forward_cox    

    # ---- Additional models ----
    elif model_name == 'nottingham-fit-age-sex':
        return NottinghamLinearModel().forward_fit_age_sex
    
    elif model_name == 'nottingham-fit-age':
        return NottinghamLinearModel().forward_fit_age
        
    elif model_name == 'nottingham-fit':
        return NottinghamLinearModel().forward_fit


    # ---- Recalibration attempts of additional models ----
    elif model_name == 'nottingham-fit-quant':
        return NottinghamLinearModel(quant_transform=True).forward_fit

    elif model_name == 'nottingham-fit-3.5':
        return NottinghamLinearModel(cfactor=3.5).forward_fit

    elif model_name == 'nottingham-fit-age-quant':
        return NottinghamLinearModel(quant_transform=True).forward_fit_age

    elif model_name == 'nottingham-fit-age-3.5':
        return NottinghamLinearModel(cfactor=3.5).forward_fit_age

    elif model_name == 'nottingham-fit-age-sex-quant':
        return NottinghamLinearModel(quant_transform=True).forward_fit_age_sex

    elif model_name == 'nottingham-fit-age-sex-3.5':
        return NottinghamLinearModel(cfactor=3.5).forward_fit_age_sex


    # ---- Oxford FIT-only model ----
    elif model_name == 'fit-spline':
        return FITSplineModel().forward
    
    elif model_name == 'fit':
        return OxfordFIT().forward


def _quantile_transform(fit, quant_path):

    # Quantiles and corresponding FIT values from external data
    q = pd.read_csv(quant_path)
    q['quant'] = q['percentile'] / 100
    q['fit_ox'] = np.quantile(fit, q.quant)  # Add corresponding FIT values in Oxford data

    # Simplified quantile transform
    fit_transformed = fit.copy()
    q_grouped = q.groupby('fit_ox')['fit_val'].agg(np.median).reset_index()
    n = q_grouped.shape[0]
    for i in range(n):
        if i < (n - 1):
            mask = (fit < q_grouped.fit_ox[i + 1]) & (fit >= q_grouped.fit_ox[i]) #if i > 0 else (fit <= q.fit_ox[i+1])
        else:
            mask = (fit >= q_grouped.fit_ox[i])
        fit_transformed[mask] = q_grouped.fit_val.to_numpy()[i]
    q['fit_ox_transformed'] = np.quantile(fit_transformed, q.quant)

    """
    plt.plot(q.percentile, q.fit_val, label='fit-other')
    plt.plot(q.percentile, q.fit_ox_transformed, label='fit-ox-transform')
    plt.legend(frameon=False)
    plt.ylim(0, 1000)
    plt.show()
    """
    return fit_transformed, q


class NottinghamLinearModel():
    """Dataclass for Nottingham models"""

    def __init__(self, quant_transform: bool = False, cfactor: float = None, fit_min: float = 4.):
        self.quant_transform = quant_transform
        self.cfactor = cfactor
        self.fit_min = fit_min
        if quant_transform and (cfactor is not None):
            raise ValueError("cfactor must be None when quant_transform is True")
    
    def predictor_variables(self, df: pd.DataFrame):  #, fit_min: float = 4.):
        """Get predictor variables to be used in the model
        Assumes df has columns 'age_at_fit', 'blood_PLT', 'blood_MCV', 'ind_gender_M'
        """
        fit = df.fit_val.copy()

        # Multiply FIT values by constant (previously - after replacing values less than 4 by 4)
        if self.cfactor is not None:
            fit *= self.cfactor

        # FIT values: apply a rough quantile transformation; or replace all values less than 4 with 4
        if self.quant_transform:
            fit, q = _quantile_transform(fit, QUANT_NOTT)
        else:
            #fit[fit < 4.] = 4.
            fit[fit < self.fit_min] = self.fit_min
        
        age, plat, mcv, male = df.age_at_fit, df.blood_PLT, df.blood_MCV, df.ind_gender_M
        return fit, age, plat, mcv, male

    def forward_lr(self, df: pd.DataFrame, return_fx: bool = False, bias_to_fx: bool = False):
        """Applies Nottingham logistic regression model to dataset df"""

        # Get predictors 
        fit, age, plat, mcv, male = self.predictor_variables(df.copy())

        # Apply model
        bias = 0.1216817
        a = 1.96315 * (age / 100)**3 - 15.09326 * (age / 100)**3 * np.log(age / 100)
        f = -2.19346 * (fit / 100)**(-1/2) - 0.31620 * (fit / 100)**(-1/2) * np.log(fit / 100)
        p = 1.07231 * np.log(plat / 100)
        m = -4.73172 * (mcv / 100)
        s = 0.51152 * male
        lin = bias + a + f + p + m + s
        y_pred = _sigmoid(lin)
        #y_pred = np.clip(y_pred, 0, 0.999)
        
        # Return
        if return_fx:
            cols = ['age_at_fit', 'fit_val', 'blood_PLT', 'blood_MCV', 'ind_gender_M']
            x = pd.concat(objs=[df.age_at_fit, df.fit_val, df.blood_PLT, df.blood_MCV, df.ind_gender_M], axis=1)
            x.columns = cols
            fx = np.vstack([a, f, p, m, s]).transpose()
            fx = pd.DataFrame(fx, columns=cols)
            if bias_to_fx:
                fx += bias
            return x, fx
        else: 
            return y_pred.to_numpy()

    def forward_lr_boot(self, df: pd.DataFrame, return_fx: bool = False, bias_to_fx: bool = False):
        """Applies Nottingham bootstrap averaged logistic regression model to dataset df"""

        # Get predictors 
        fit, age, plat, mcv, male = self.predictor_variables(df.copy())

        # Apply model
        bias = -1.856562
        a = 1.992754 * (age / 100)**3 - 15.81497 * (age / 100)**3 * np.log(age / 100)
        f = -0.573727 * (fit / 100)**(-1/2) - 0.07482237 * (fit / 100)**(-1/2) * np.log(fit / 100) + 0.5247163 * np.log(fit / 100)
        p = 0.4873701 * np.log(plat / 100) + 0.152428 * (plat / 100)**(1/2) + 0.1029912 * (plat / 100) + 0.01396725 * (plat / 100)**2 - 0.005743251 * (plat / 100)**2 * np.log(plat / 100)
        m = -4.8295074 * (mcv / 100)
        s = 0.5214879 * male
        lin = bias + a + f + p + m + s
        y_pred = _sigmoid(lin)
        #y_pred = np.clip(y_pred, 0, 0.999)
        
        # Return
        if return_fx:
            cols = ['age_at_fit', 'fit_val', 'blood_PLT', 'blood_MCV', 'ind_gender_M']
            x = pd.concat(objs=[df.age_at_fit, df.fit_val, df.blood_PLT, df.blood_MCV, df.ind_gender_M], axis=1)
            x.columns = cols
            fx = np.vstack([a, f, p, m, s]).transpose()
            fx = pd.DataFrame(fx, columns=cols)
            if bias_to_fx:
                fx += bias
            return x, fx
        else: 
            return y_pred.to_numpy()
    
    def forward_cox(self, df: pd.DataFrame, return_fx: bool = False, bias_to_fx: bool = False):
        
        # Get predictors 
        fit, age, plat, mcv, male = self.predictor_variables(df)

        # Predict
        base = np.exp(-0.6592014)
        a = 1.6685765 * (age/100)**3 - 13.9435406 * (age/100)**3 * np.log(age/100)
        f = -1.9965475 * (fit/100)**(-1/2) - 0.2657153 * (fit/100)**(-1/2) * np.log(fit/100)
        #f[fit < 1] = -9
        p = 0.9208493 * np.log(plat/100)
        m = -3.9007829 * (mcv/100)
        s = 0.4543275 * male
        lin = a + f + p + m + s
        y_pred = 1 - base ** np.exp(lin)

        # Return
        if return_fx:
            cols = ['age_at_fit', 'fit_val', 'blood_PLT', 'blood_MCV', 'ind_gender_M']
            x = pd.concat(objs=[df.age_at_fit, df.fit_val, df.blood_PLT, df.blood_MCV, df.ind_gender_M], axis=1)
            x.columns = cols
            fx = np.vstack([a, f, p, m, s]).transpose()
            fx = pd.DataFrame(fx, columns=cols)

            return x, fx
        else: 
            return y_pred.to_numpy()

    def forward_cox_boot(self, df: pd.DataFrame, return_fx: bool = False, bias_to_fx: bool = False):
        """Applies Nottingham bootstrapped Cox model to dataset df"""

        # Get predictors 
        fit, age, plat, mcv, male = self.predictor_variables(df)

        # Predict
        base = np.exp(-0.5453559)
        a = 1.708974 * (age/100)**3 - 14.32383 * (age/100)**3 * np.log(age/100)
        f = -1.81905 * (fit/100)**(-1/2) - 0.2444505  * (fit/100)**(-1/2) * np.log(fit/100) + 0.068396 * np.log(fit/100)
        p = 0.70926459 * np.log(plat/100) + 0.06014985 * (plat/100) + 0.01821419 * (plat/100)**2 - 0.01055294 * (plat/100)**2 * np.log(plat/100)
        m = -4.01856725 * (mcv/100)
        s = 0.46638214 * male
        lin = a + f + p + m + s
        y_pred = 1 - base ** np.exp(lin)

        # Return
        if return_fx:
            cols = ['age_at_fit', 'fit_val', 'blood_PLT', 'blood_MCV', 'ind_gender_M']
            x = pd.concat(objs=[df.age_at_fit, df.fit_val, df.blood_PLT, df.blood_MCV, df.ind_gender_M], axis=1)
            x.columns = cols
            fx = np.vstack([a, f, p, m, s]).transpose()
            fx = pd.DataFrame(fx, columns=cols)
            
            return x, fx
        else: 
            return y_pred.to_numpy()
    
    def forward_fit(self, df: pd.DataFrame, return_fx: bool = False, bias_to_fx: bool = False):

        # Get predictor variables
        fit, __, __, __, __ = self.predictor_variables(df)
        
        # Get prediction
        bias = -0.2772
        f = - 2.2325*(fit/100)**(-0.5) - 0.3109*(fit/100)**(-0.5)*np.log(fit/100)
        lin = bias + f
        y_pred = _sigmoid(lin)
        #y_pred = np.clip(y_pred, 0, 0.999)

        # Return
        if return_fx:
            cols = ['fit_val']
            x = pd.DataFrame(df.fit_val, columns=['fit_val'])
            fx = pd.DataFrame(f, columns=cols)
            if bias_to_fx:
                fx += bias
            return x, fx
        else: 
            return y_pred.to_numpy() 

    def forward_fit_age(self, df: pd.DataFrame, return_fx: bool = False, bias_to_fx: bool = False):

        # Get predictors 
        fit, age, __, __, __ = self.predictor_variables(df.copy())

        # Apply model
        bias = -5.00066
        a = 1.66788 * (age/100)**3 - 16.05224 * (age/100)**3 * np.log(age/100)
        f = 0.84294 * np.log(fit/100) - 0.08597 * np.log(fit/100)**2  
        lin = bias + a + f
        y_pred = _sigmoid(lin)
        #y_pred = np.clip(y_pred, 0, 0.999)

        # Return
        if return_fx:
            cols = ['fit_val', 'age_at_fit']
            x = pd.concat(objs=[df.fit_val, df.age_at_fit], axis=1)
            x.columns = cols
            fx = np.vstack([f, a]).transpose()
            fx = pd.DataFrame(fx, columns=cols)
            if bias_to_fx:
                fx += bias
            return x, fx
        else: 
            return y_pred.to_numpy()

    def forward_fit_age_sex(self, df: pd.DataFrame, return_fx: bool = False, bias_to_fx: bool = False):

        # Get predictors 
        fit, age, __, __, male = self.predictor_variables(df.copy())

        # Apply model
        bias = -5.09771
        a = 1.68502 * (age/100)**3 - 15.44161 * (age/100)**3 * np.log(age/100)
        f = 0.83737 * np.log(fit/100) - 0.08638 * np.log(fit/100)**2  
        s = 0.30240 * male 
        lin = bias + a + f + s
        y_pred = _sigmoid(lin)
        #y_pred = np.clip(y_pred, 0, 0.999)

        # Return
        if return_fx:
            cols = ['fit_val', 'age_at_fit', 'ind_gender_M']
            x = pd.concat(objs=[df.fit_val, df.age_at_fit, df.ind_gender_M], axis=1)
            x.columns = cols
            fx = np.vstack([f, a, s]).transpose()
            fx = pd.DataFrame(fx, columns=cols)
            if bias_to_fx:
                fx += bias
            return x, fx
        else: 
            return y_pred.to_numpy()


def model_output_test_nott(run_path: Path):
    """Produce predicted probabilities for typical covariate values:
    to be cross-checked with developing centres to ensure model formulas are correct"""

    # Nottingham 
    df = pd.DataFrame()
    df['fit_val'] = [4, 10, 1000]
    df['age_at_fit'] = [20, 60, 80]
    df['blood_PLT'] = [100, 200, 400]
    df['blood_MCV'] = [80, 90, 100]
    df['ind_gender_M'] = [0, 1, 1]

    nott_models = ['nottingham-lr', 'nottingham-lr-boot', 'nottingham-cox', 'nottingham-cox-boot',
                   'nottingham-fit', 'nottingham-fit-age', 'nottingham-fit-age-sex']
    nott_suf = ['_lr', '_lrboot', '_cox', '_coxboot', '_fit', '_fitage', '_fitagesex']
    for suf, model_name in zip(nott_suf, nott_models):
        model = get_model(model_name)
        df['y_pred' + suf] = model(df)
    
    df.to_csv(run_path / NOTT_TEST, index=False)

    return df


class FITSplineModel():
    """FIT-only model based on old Oxford data
    Predictor variable is the FIT-test
    FIT is transformed using spline(log(FIT + 1)) before applying logistic regression.
    This is kept for compatibility with older code: better to use create_spline_model() below
    """
    def __init__(self, bias: list = None, coef: list = None):
        self.model_name = 'fit-spline'
        self.bias = bias
        self.coef = coef

    def fit(self, df: pd.DataFrame, y: pd.DataFrame):

        fit = df.fit_val.to_numpy()
        y = y.iloc[:, 0].to_numpy().squeeze()

        fit_log = np.log(fit.reshape(-1, 1) + 1)
        knots = np.array([np.log(10+1), np.log(100+1)]).reshape(-1, 1)
        spline = SplineTransformer(degree=2, knots=knots, include_bias=False, extrapolation='linear')
        xs = spline.fit_transform(fit_log)

        clf_spline = LogisticRegression(penalty=None)
        clf_spline.fit(xs, y)
        
        self.bias = clf_spline.intercept_
        self.coef = clf_spline.coef_.squeeze().tolist()

    def forward(self, df: pd.DataFrame, return_fx: bool = False, bias_to_fx: bool = False):

        fit = df.fit_val.copy().to_numpy()
        
        # Spline transform, knots for FIT at 10 ug/g and 100 ug/g
        fit_log = np.log(fit.reshape(-1, 1) + 1)
        knots = np.array([np.log(10 + 1), np.log(100 + 1)]).reshape(-1, 1)
        spline = SplineTransformer(degree=2, knots=knots, include_bias=False, extrapolation='linear')
        xs = spline.fit_transform(fit_log)

        # Apply model
        if self.coef is None:
            #coef = np.array([-3.87924753, -1.07700156]).reshape(1, -1)
            coef = np.array([-3.77573562 -1.29889335]).reshape(1, -1)
        else:
            coef = np.array([self.coef]).reshape(1, -1)
        if self.bias is None:
            #bias = -1.15642343
            bias = -1.25677013
        else:
            bias = self.bias
        f = (coef * xs).sum(axis=1)
        lin = bias + f
        y_pred = _sigmoid(lin)
        #y_pred = np.clip(y_pred, 0, 0.999)

        # Return
        if return_fx:
            cols = ['fit_val']
            x = pd.DataFrame(df.fit_val, columns=['fit_val'])
            fx = pd.DataFrame(f, columns=cols)
            if bias_to_fx:
                fx += bias
            return x, fx
        else: 
            return y_pred


class OxfordFIT():
    """Returns Oxford FIT test
    Included as a separate class, so that preprocessing routines could be specified here."""

    def __init__(self):
        self.model_name = 'fit'

    def forward(self, df: pd.DataFrame):
        return df.fit_val.copy().to_numpy()


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
