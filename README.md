# fitval: validation of FIT-based prediction models

Functions for validating colorectal cancer (CRC) risk prediction models that combine the FIT test result with other variables in symptomatic patients. 


## Installation 

```
conda create -n fitval python=3.9
conda activate fitval
pip install -r requirements.txt
```

## Usage

The code can be run on a table that for each patient contains the colorectal cancer indicator, FIT test values, and predicted probabilities of CRC according to a model. The columns must be named and formatted as follows:

* `crc` : indicator variable for colorectal cancer (0: no, 1: yes), 

* `fit_val` : numerical FIT test values (1.5, 10, 20, etc), 

* `y_pred` : predicted probability of CRC according to a model, must be in the [0, 1] interval (0.03, 0.05, 0.25 etc).


The table must be in a `.csv` format. For example, if the table is located in `./data/predictions.csv`, then the code can be run as
```sh
validate -p ./data/predictions.csv
```
