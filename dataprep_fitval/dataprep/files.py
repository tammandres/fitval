from dataclasses import dataclass
from constants import PARQUET


if PARQUET:
    @dataclass
    class InputFiles:
        # Input data files
        fit: str = 'fit_values'
        bloods: str = 'biochemistry'
        demo: str = 'demographics'
        hist: str = 'tnm_staging_pathology'
        inpat_diag: str = 'inpat_diagnosis'
        outpat_diag: str = 'outpat_diagnosis'
        inpat_proc: str = 'inpat_procedures'
        outpat_proc: str = 'outpat_procedures'
        chemo_admin: str = 'medication_chemo_admin'
        chemo_sum: str = 'medication_chemo_summary'
        radio: str = 'radiotherapy'
        imaging: str = 'tnm_staging_radiology'
        prescribing: str = 'medication_all'
        bmi: str = 'bmi_measurements'
        crc_from_path: str = 'crc_from_path_20240618'
        crc_matches: str = 'crc_matches_20240618'
        outpat: str = 'outpat_attendances'
        infoflex: str = 'infoflex_diagnosis_and_first_treatment'
else:
    @dataclass
    class InputFiles:
        # Input data files
        fit: str = 'fit_values.csv'
        bloods: str = 'biochemistry.csv'
        demo: str = 'demographics.csv'
        hist: str = 'data_reports_pathology_redacted.csv'
        inpat_diag: str = 'inpat_diagnosis.csv'
        outpat_diag: str = 'outpat_diagnosis.csv'
        inpat_proc: str = 'inpat_procedures.csv'
        outpat_proc: str = 'outpat_procedures.csv'
        chemo_admin: str = 'medication_chemo_admin.csv'
        chemo_sum: str = 'medication_chemo_summary.csv'
        radio: str = 'radiotherapy.csv'
        imaging: str = None
        prescribing: str = 'medication_all_subset.csv'
        bmi: str = 'bmi_measurements.csv'
        crc_from_path: str = 'crc_from_path_20240529.csv'
        crc_matches: str = 'crc_matches_20240529.csv'
        outpat: str = 'outpat_attendances.csv'


@dataclass
class OutputFiles:
    """Outputted data files"""
    fit: str = 'fit.csv'   # Earliest FIT values
    #clind: str = 'clind.csv'  # Clinical details associated with FIT values
    demo: str = 'demo.csv'  # Demographics
    sym: str = 'sym.csv'  # Clinical symptoms extracted from clinical details
    diagmin: str = 'diagmin.csv'  # Earliest colorectal cancer (CRC) date and source
    diag: str = 'diag.csv'  # All sources of CRC dates
    events: str = 'events.csv'  # Event logs
    bloods: str = 'bloods.csv'  # Blood test results
    bloods_hilo: str = 'bloods_hilo.csv'   # High-low indicators for specific bloods
    diag_codes: str = 'diag_codes.csv'  # Diagnosis codes (inpat and outpat)
    proc_codes: str = 'proc_codes.csv'  # Procedure codes (inpat and outpat)
    pres_codes: str = 'pres_codes.csv'  # Prescription codes (inpat and outpat)
    bmi: str = 'bmi.csv'  # Body mass index
    sum_table: str = 'summary_table.csv'  # Summary table for CRC and no-CRC groups
    inclusion: str = 'inclusion.csv'  # Number of patients at each step of inclusion criteria
    data_matrix: str = 'data_matrix.csv'  # Matrix of predictors, outcome, subject identifier
    x: str = 'x.csv'  # Matrix of predictors
    y: str = 'y.csv'  # Matrix of outcome variable


@dataclass
class DQFiles:
    """Outputted files for data quality checks"""
    reports_crc_path: str = 'dq_reports_crc_pathology.csv'
    matches_crc_path = 'dq_matches_crc_pathology.csv'
    reports = 'dq_reports_crc.csv'
    matches_crc = 'dq_matches_crc.csv'
    matches_sym = 'dq_matches_sym.csv'
    matches_tnm = 'dq_matches_tnm.csv'
    matches_tnm_ex = 'dq_matches_tnm_excluded.csv'
    plot_days_fit_crc = 'dq_plot_days_fit_crc.png'
    plot_fit_hist = 'dq_plot_fit_histogram.png'
    plot_mis = 'dq_plot_missing.png'
    plot_mis_ind = 'dq_plot_missing_individual.png'
    table_mis = 'dq_table_missing.csv'
    bloods_sum = 'dq_bloods_summary.csv'
    code_counts = 'dq_code_counts.csv'
    bmi_dq = 'dq_bmi.csv'


# Helper files
IMG_CODE_FILE = 'NIHR-HIC_Colorectal-Cancer_imaging-types.xlsx'
