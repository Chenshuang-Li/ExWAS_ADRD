#Survival analysis of Exposomic Risk Score----------------
library(dplyr)
library(readr)
library(survival)

project_dir <- "/path/to/your/project"
output_path <- file.path(project_dir, "results/ADRD")
output_path_subgroup <- file.path(output_path, "All")
outcome_var <- "ADRD"
time_var <- "time_ADRD"
model_to_analyze <- "XGB"

handoff02 <- readRDS(file.path(output_path_subgroup, "02_prediction_handoff.rds"))
train_data_matched <- handoff02$train_data_matched
test_data <- handoff02$test_data
covariate_vars <- handoff02$covariate_vars

shap_data <- readRDS(file.path(output_path_subgroup, paste0(outcome_var, "_shap_data.rds")))
cox_results <- read_csv(file.path(output_path_subgroup, paste0(outcome_var, "_cox_results.csv")))

selected_features <- shap_data[[model_to_analyze]]$selected_features

weights_df <- cox_results %>%
  filter(Factor %in% selected_features) %>%
  mutate(Beta = log(HR)) %>%
  dplyr::select(Factor, Beta, HR, CI5, CI95, P, Z) %>%
  arrange(desc(abs(Beta)))

selected_features <- intersect(selected_features, weights_df$Factor)

write.csv(
  weights_df,
  file.path(output_path_subgroup, paste0(outcome_var, "_ERS_weights.csv")),
  row.names = FALSE
)

weights <- setNames(weights_df$Beta, weights_df$Factor)

X_train_ers <- train_data_matched %>% dplyr::select(all_of(selected_features))
X_test_ers <- test_data %>% dplyr::select(all_of(selected_features))

train_data_matched$ERS <- as.numeric(as.matrix(X_train_ers) %*% weights[selected_features])
test_data$ERS <- as.numeric(as.matrix(X_test_ers) %*% weights[selected_features])

test_data_with_ers <- test_data %>%
  dplyr::select(eid, all_of(outcome_var), all_of(time_var), 
                ERS, all_of(covariate_vars))
write.csv(
  test_data_with_ers,
  file.path(output_path_subgroup, paste0(outcome_var, "_test_data_with_ERS.csv")),
  row.names = FALSE
)

test_data$ERS_quartile <- tryCatch({
  cut(
    test_data$ERS,
    breaks = quantile(test_data$ERS, probs = c(0, 0.25, 0.5, 0.75, 1)),
    labels = c("Q1_Low", "Q2", "Q3", "Q4_High"),
    include.lowest = TRUE
  )
}, error = function(e) {
  factor(
    ntile(test_data$ERS, 4),
    levels = 1:4,
    labels = c("Q1_Low", "Q2", "Q3", "Q4_High")
  )
})

quartile_formula <- as.formula(paste0(
  "Surv(", time_var, ", ", outcome_var, " == 1) ~ ",
  "ERS_quartile + ", paste(covariate_vars, collapse = " + ")
))
cox_quartile <- coxph(quartile_formula, data = test_data)

cox_coef_table <- as.data.frame(summary(cox_quartile)$coefficients)
cox_coef_table$Variable <- rownames(cox_coef_table)
cox_coef_table <- cox_coef_table[, c("Variable", names(cox_coef_table)[1:(ncol(cox_coef_table)-1)])]
write.csv(
  cox_coef_table, 
  file.path(output_path_subgroup, paste0(outcome_var, "_ERS_cox_quartile_coefficients.csv")), 
  row.names = FALSE
)

cox_conf <- as.data.frame(summary(cox_quartile)$conf.int)
cox_conf$Variable <- rownames(cox_conf)
cox_conf <- cox_conf[, c("Variable", names(cox_conf)[1:(ncol(cox_conf)-1)])]
write.csv(
  cox_conf, 
  file.path(output_path_subgroup, paste0(outcome_var, "_ERS_cox_quartile_HR_CI.csv")), 
  row.names = FALSE
)

test_data_surv <- test_data %>%
  filter(!is.na(ERS_quartile) & 
           !is.na(!!sym(time_var)) & 
           !is.na(!!sym(outcome_var)))

surv_formula <- as.formula(paste0(
  "Surv(", time_var, ", ", outcome_var, " == 1) ~ ERS_quartile"
))

fit_quartile <- survfit(surv_formula, data = test_data_surv)
survdiff_result <- survdiff(surv_formula, data = test_data_surv)

quartile_summary <- test_data_surv %>%
  group_by(ERS_quartile) %>%
  summarise(
    N = n(),
    Events = sum(!!sym(outcome_var) == 1),
    Event_Rate = sprintf("%.1f%%", 100 * mean(!!sym(outcome_var) == 1)),
    Mean_ERS = sprintf("%.3f", mean(ERS)),
    .groups = "drop"
  )
write.csv(
  quartile_summary,
  file.path(output_path_subgroup, paste0(outcome_var, "_ERS_quartile_summary.csv")),
  row.names = FALSE
)

surv_data <- data.frame(
  time = fit_quartile$time,
  surv = fit_quartile$surv,
  upper = fit_quartile$upper,
  lower = fit_quartile$lower,
  n.risk = fit_quartile$n.risk,
  n.event = fit_quartile$n.event,
  strata = rep(names(fit_quartile$strata), fit_quartile$strata)
)

surv_data$strata <- gsub("ERS_quartile=", "", surv_data$strata)
surv_data$strata <- factor(surv_data$strata, 
                           levels = c("Q1_Low", "Q2", "Q3", "Q4_High"))

surv_data_with_origin <- surv_data %>%
  group_by(strata) %>%
  do({
    rbind(
      data.frame(time = 0, surv = 1, upper = 1, lower = 1, 
                 n.risk = first(.$n.risk), n.event = 0, strata = first(.$strata)),
      .
    )
  }) %>%
  ungroup()

write.csv(
  surv_data_with_origin,
  file.path(output_path_subgroup, paste0(outcome_var, "_ERS_survival_data.csv")),
  row.names = FALSE
)