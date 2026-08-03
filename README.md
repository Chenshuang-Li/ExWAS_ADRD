# Exposome-Wide Association Study and Predictive Risk Score for Incident Alzheimer's Disease and Related Dementias

1.Code accompanying the manuscript:

Li C, Zhao C, Lan Y, Ellis RJ, Isaac S, Zhang S, Miller GW, Sun Y,
Jain S, Yadav H, Gao P*, Patel CJ*. Exposome-Wide Association Study
and Predictive Risk Score for Incident Alzheimer's Disease and
Related Dementias. *Alzheimer's & Dementia: The Journal of the
Alzheimer's Association*. (Under review).

*Corresponding authors.

2.Pipeline
Run the three scripts in order:

01_ExWAS_Cox.R — Data preparation; univariate Cox regression for each exposure (ExWAS); multiple-testing correction; correlation-based clustering to remove redundant exposures.
02_Prediction_Model.R — Case-control matching; nested cross-validation (LASSO + XGBoost) to build an exposome-based prediction model; SHAP values; comparison against covariate-only and covariate+exposure models (AUC, NRI, IDI).
03_ERS_Survival.R — Construct the Exposomic Risk Score (ERS); stratify into quartiles; Cox and Kaplan-Meier survival analysis by ERS quartile.

Each script saves an .rds handoff file that the next script reads — run them in order; do not skip steps.

bash
Rscript 01_ExWAS_Cox.R
Rscript 02_Prediction_Model.R
Rscript 03_ERS_Survival.R
Requirements

R (developed with 4.3.x). Packages:

r
install.packages(c(
  "dplyr", "tibble", "readr", "arrow", "survival",
  "MatchIt", "caret", "pROC", "xgboost", "glmnet", "shapviz"
))
Input data

Set project_dir at the top of 01_ExWAS_Cox.R to your local data folder, which should contain three parquet files keyed by eid:

outcome.parquet — eid, ADRD, time_ADRD
covariates.parquet — eid, sex, age, ethnicity, APOE_status, overall_health, and assessment_center
exposures.parquet — eid, one column per candidate exposure

Data are not included in this repository due to data use agreement restrictions.. The data used in the present study are available from UKB with restrictions applied. Data were used under license and are thus not publicly available. Access to the UKB data can be requested through a standard protocol (https://www.ukbiobank.ac.uk/register-apply/).

3.Outputs
Results are written to <project_dir>/results/ADRD/All/, including Cox regression results, selected exposures, the fitted prediction model and SHAP object, model comparison metrics, and ERS survival analysis results.

4.Citation
If you use this code, please cite:
Li C, Zhao C, Lan Y, Ellis RJ, Isaac S, Zhang S, Miller GW, Sun Y, Jain S, Yadav H, Gao P, Patel CJ. 
Exposome-Wide Association Study and Predictive Risk Score for Incident Alzheimer's Disease and Related Dementias. 
*Alzheimer's & Dementia: The Journal of the Alzheimer's Association*. [Year]; [Volume]([Issue]):[Pages].
DOI: [xxx]
