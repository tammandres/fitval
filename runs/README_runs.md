# Scripts for COLOFIT analysis


Main scripts, run in the following order:

1. Compute performance metrics: `colofit_time-cut_metrics.py` and `colofit_time-prepost_metrics.py`

2. Reformat tables of performance metrics: `colofit_reformat.py`

3. Compute descriptive stats table: `colofit_time-cut_descriptives.py`, `colofit_time-prepost_descriptives.py`
   (requires the use of a different conda environment)

4. Aggregate tables computed for each time period:  `colofit_agg.py`

5. Plot aggregated data: `colofit_time-cut_plot.py` and `colofit_time-prepost_plot.py`

6. Reformat aggregated descriptive stats table: `colofit_agg_reformat_descriptives.py`

7. Additionally reformat aggregated tables for inclusion in a report: `colofit_agg_reformat_tables.py`

8. Plot reduction in referrals with both 365 and 180 day follow-ups: `colofit_time-cut_plot-followup.py`

9. Conduct statistical tests: `colofit_sample_proportio.py` (e.g. for differences in proportion with low haemoglobin between people with cancer and no cancer)

10. Main script to compute reduction in referrals over time under different methods of computing the risk score threshold (externally estimated, locally estimated threshold from current time period, locally estimated threshold from previous time period): `colofit_reduction_external-local.py`

11. Compute reduction in referrals over time under different methods of computing the risk score threshold, this time including different subsets of the linear predictor to evaluate how the input variables contribute to reduction in referrals: `colofit_reduction-by-prdictor_external-local.py`

Additional scripts:

* `colofit_time-cut_pfit10.py`: compute the proportion with FIT >= 10 in each time period.

* `colofit_time-cut_descriptives_fix-bloods-high-low.py`: fix low/normal HGB counts that were previously incorrect in two time periods, because the gender label was 'Female or non-binary (<10)' but the blood test summarisation script expected F or M only.

* `colofit_sanitycheck.py`: a few checks, including a check that more manually calculated results give the same results as the scripts

* `colofit_hm_vs_oc.py`: explore how the predicted risk changes when FIT values are increased above 400 while other predictors are set at median. Provides insight into how predicted risks may differ for patients who did their FIT with HMJack rather than OC censor.

Andres Tamm, 1 Aug 2025



