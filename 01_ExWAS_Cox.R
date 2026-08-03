#ExWAS for ADRD (Cox)---------------------------
#Data preparation
library(dplyr)
library(tibble)
library(readr)
library(arrow)
library(survival)

project_dir <- "/path/to/your/project"
outcome_path <- file.path(project_dir, "data/outcome.parquet")
covariate_path <- file.path(project_dir, "data/covariates.parquet")
independent_path <- file.path(project_dir, "data/exposures.parquet")
output_path <- file.path(project_dir, "results/ADRD")

outcome <- read_parquet(outcome_path)
covariates <- read_parquet(covariate_path)
independent <- read_parquet(independent_path)
outcome_var <- "ADRD"
time_var <- "time_ADRD"

analysis_data <- outcome %>%
  inner_join(covariates, by = "eid") %>%
  inner_join(independent, by = "eid")

set.seed(123)
train_index <- sample(seq_len(nrow(analysis_data)), size = floor(2/3 * nrow(analysis_data)))
train_data <- analysis_data[train_index, ]
test_data  <- analysis_data[-train_index, ]

output_path_subgroup <- file.path(output_path, "All")
if(!dir.exists(output_path_subgroup)) dir.create(output_path_subgroup, recursive = TRUE)

covariate_vars <- setdiff(colnames(covariates), "eid")
independent_vars <- setdiff(colnames(independent), "eid")

#ExWAS for ADRD (Cox)

Uni_glm_model <- function(x, data, outcome_var, covariate_vars, min_cases = 10, min_total = 200, max_iter = 50) {
  
  cases <- sum(data[[outcome_var]] == 1)
  total <- nrow(data)
  
  if(cases < min_cases | total < min_total) {
    return(data.frame(Factor = x, HR = NA, CI5 = NA, CI95 = NA,
                      P = NA, Z = NA, ph_value = NA, QC_flag = FALSE))
  }
  
  FML <- as.formula(paste0("Surv(", time_var, ", ", outcome_var, " == 1) ~ ",
                           x, " + ", paste(covariate_vars, collapse = " + ")))
  
  glm1 <- tryCatch(coxph(FML, data = data), error = function(e) NULL)
  
  if(is.null(glm1) | glm1$iter > max_iter) {
    return(data.frame(Factor = x, HR = NA, CI5 = NA, CI95 = NA,
                      P = NA, Z = NA, ph_value = NA, QC_flag = FALSE))
  }
  
  GSum <- summary(glm1)
  coef <- GSum$coefficients[1, 1]
  HR <- round(GSum$coefficients[1, 2], 4)
  se <- GSum$coefficients[1, 3]
  CI5 <- round(exp(coef - 1.96 * se), 4)
  CI95 <- round(exp(coef + 1.96 * se), 4)
  Pvalue <- GSum$coefficients[1, 5]
  Z <- GSum$coefficients[1, 4]
  
  ph_test <- tryCatch(cox.zph(glm1), error = function(e) NULL)
  ph_pvalue <- if(!is.null(ph_test)) ph_test$table[1, "p"] else NA
  
  return(data.frame(Factor = x, HR, CI5, CI95, P = Pvalue, Z, ph_value = ph_pvalue, QC_flag = TRUE))
}

Uni_glm <- lapply(independent_vars, Uni_glm_model, 
                  data = train_data, 
                  outcome_var = outcome_var,
                  covariate_vars = covariate_vars)
result <- do.call(rbind, Uni_glm)

result$P <- as.numeric(result$P)
result <- result[order(result$P), ]
result$P_Bonferroni_adjust <- p.adjust(result$P, method = "bonferroni")
result$P_FDR_adjust <- p.adjust(result$P, method = "fdr")

write.csv(result, file.path(output_path_subgroup, paste0(outcome_var, "_cox_results.csv")), row.names = FALSE)

significant_factor_bonferroni <- result$Factor[result$P_Bonferroni_adjust < 0.05 & result$QC_flag]
significant_factor_fdr <- result$Factor[result$P_FDR_adjust < 0.05 & result$QC_flag]
write.csv(data.frame(Factor = significant_factor_bonferroni),
          file.path(output_path_subgroup, paste0(outcome_var, "_sig_Bonferroni.csv")), row.names = FALSE)
write.csv(data.frame(Factor = significant_factor_fdr),
          file.path(output_path_subgroup, paste0(outcome_var, "_sig_FDR.csv")), row.names = FALSE)

cor_threshold <- 0.25
sig_factors <- significant_factor_bonferroni

vars_data <- dplyr::select(train_data, eid, dplyr::all_of(sig_factors)) %>% 
  column_to_rownames("eid")

convert_to_numeric <- function(x) {
  if (is.factor(x)) as.numeric(x)
  else if (is.character(x)) as.numeric(as.factor(x))
  else if (is.logical(x)) as.numeric(x)
  else x
}
vars_data <- as.data.frame(lapply(vars_data, convert_to_numeric))

corr_matrix <- cor(vars_data, method = "spearman")

write_parquet(as.data.frame(corr_matrix),
              file.path(output_path_subgroup, paste0(outcome_var, "_corr_matrix.parquet")))

vars_data_df <- vars_data %>% rownames_to_column("eid")
write_parquet(vars_data_df,
              file.path(output_path_subgroup, paste0(outcome_var, "_vars_data_clean.parquet")))

hc <- hclust(as.dist(1 - corr_matrix), method = "average")
dist_cutoff <- 1 - cor_threshold
clusters <- cutree(hc, h = dist_cutoff)

kept_vars <- sapply(unique(clusters), function(cl) {
  vars_in_cluster <- names(clusters[clusters == cl])
  subdata <- vars_data[, vars_in_cluster, drop = FALSE]
  vars_in_cluster[which.max(apply(subdata, 2, var, na.rm = TRUE))]
})
kept_vars <- unname(kept_vars)
removed_vars <- setdiff(colnames(vars_data), kept_vars)

write_csv(data.frame(Factor = kept_vars),
          file.path(output_path_subgroup, paste0(outcome_var, "_cluster_kept_vars.csv")))
write_csv(data.frame(Factor = removed_vars),
          file.path(output_path_subgroup, paste0(outcome_var, "_cluster_removed_vars.csv")))

saveRDS(
  list(
    train_data = train_data,
    test_data = test_data,
    covariate_vars = covariate_vars,
    kept_vars = kept_vars
  ),
  file.path(output_path_subgroup, "01_data_cox_handoff.rds")
)


