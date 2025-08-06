# Prepare the OUH-FIT dataset

Andres Tamm

2024-05-21 


## Usage
Install required packages
```
conda create -n fit-dataprep python=3.9
conda activate fit-dataprep
pip install -r requirements.txt
```

## Run 

Run the scripts in this order

1. `airlock_get_data.py` - collates the data files needed for analysis, from raw airlocked files

2. `data_items.py` - cleans and prepares the tables needed for analysis

3. `dataprep_colofit.py` - applies inclusion criteria and computes the data matrix

4. `fit_timetrends.py` - summarises time trends in FIT and clinical symptoms

5. `check_nfit_20250515.py` - summarises how many subsequent FITs were available and let to a cancer diagnosis

6. `airlock_copy.py` - an example script for copying the outputs to a new dir for airlock export 


## Dataset version

Uses updated FIT dataset with data cutoff 2024-02-26, based on these airlocks: 

* gpreqs correct, 2024-06-27, 31d76491-a25a-49a2-bb04-7627e7c08c88 (fix timezone in gpreqs)
* FIT dataset update part2, 2024-05-20, b0747ccf-8744-40ed-87be-4e6ae3527035 (TNM staging)
* Fit dataset update part1, 2024-05-17, a409305d-923b-4bff-a592-825ae194a870 (nearly all data items)


## Notes 

* Most data in parquet format has date columns already in datetime format, but not always
* Sometimes, parquet files contain indicator variables (0/1) but these are in object (string format):
  have needed to transform these to integer.
