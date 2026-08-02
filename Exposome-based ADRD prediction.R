#Exposome-based ADRD prediction --------------------------
library(dplyr)
library(MatchIt)
library(caret)
library(pROC)
library(xgboost)
library(glmnet)
library(shapviz)

project_dir <- "/path/to/your/project"
output_path <- file.path(project_dir, "results/ADRD")
output_path_subgroup <- file.path(output_path, "All")
outcome_var <- "ADRD"
time_var <- "time_ADRD"
model_to_analyze <- "XGB"

handoff01 <- readRDS(file.path(output_path_subgroup, "01_data_cox_handoff.rds"))
train_data <- handoff01$train_data
test_data <- handoff01$test_data
covariate_vars <- handoff01$covariate_vars
independent_vars <- handoff01$kept_vars


case_data <- train_data %>% filter(!!sym(outcome_var) == 1)
control_data <- train_data %>% filter(!!sym(outcome_var) == 0)

train_downsampled <- bind_rows(case_data, control_data) %>%
  mutate(id = row_number())

match_formula <- as.formula(paste0(outcome_var, " ~ sex + age"))

m.out <- matchit(
  formula = match_formula,
  data = train_downsampled,
  method = "nearest",
  ratio = 1,
  replace = FALSE
)

matched_data <- match.data(m.out)

train_data_matched <- dplyr::select(
  matched_data,
  -distance,
  -weights,
  -subclass,
  -id
)

y_train <- train_data_matched[[outcome_var]]

X_train <- dplyr::select(train_data_matched, dplyr::all_of(independent_vars))
X_train <- as.data.frame(X_train)

xgb_param_grid <- expand.grid(
  max_depth = c(3,5), eta = c(0.01,0.05),
  subsample = 0.7, colsample_bytree = 0.7,
  min_child_weight = 1, stringsAsFactors = FALSE
)

nested_cv <- function(X, y, outer_folds = 5, inner_folds = 5, model_type, param_grid, seed = 123) {
  set.seed(seed)
  
  outer_indices <- createFolds(y, k = outer_folds, list = TRUE, returnTrain = FALSE)
  outer_results <- list()
  
  for (i in 1:outer_folds) {
    test_idx <- outer_indices[[i]]
    train_idx <- setdiff(seq_len(nrow(X)), test_idx)
    
    X_outer_train <- X[train_idx, , drop = FALSE]
    y_outer_train <- y[train_idx]
    X_outer_test <- X[test_idx, , drop = FALSE]
    y_outer_test <- y[test_idx]
    
    best_auc <- 0
    best_params <- NULL
    best_model <- NULL
    
    for (j in 1:nrow(param_grid)) {
      inner_indices <- createFolds(y_outer_train, k = inner_folds, list = TRUE, returnTrain = FALSE)
      auc_inner <- c()
      
      for (k in 1:inner_folds) {
        inner_test_idx <- inner_indices[[k]]
        inner_train_idx <- setdiff(seq_len(nrow(X_outer_train)), inner_test_idx)
        
        X_inner_train <- X_outer_train[inner_train_idx, , drop = FALSE]
        y_inner_train <- y_outer_train[inner_train_idx]
        X_inner_test <- X_outer_train[inner_test_idx, , drop = FALSE]
        y_inner_test <- y_outer_train[inner_test_idx]
        
        lasso_fit <- cv.glmnet(as.matrix(X_inner_train), y_inner_train, alpha = 1, family = "binomial")
        selected_features <- rownames(coef(lasso_fit, s = "lambda.1se"))[coef(lasso_fit, s = "lambda.1se")[,1] != 0]
        selected_features <- selected_features[selected_features != "(Intercept)"]
        
        if (length(selected_features) == 0) next
        
        X_inner_train_sel <- X_inner_train[, selected_features, drop = FALSE]
        X_inner_test_sel <- X_inner_test[, selected_features, drop = FALSE]
        
        if (model_type == "xgb") {
          xgb_params <- list(
            objective = "binary:logistic", eval_metric = "auc",
            max_depth = param_grid$max_depth[j], eta = param_grid$eta[j],
            subsample = param_grid$subsample[j], colsample_bytree = param_grid$colsample_bytree[j],
            min_child_weight = param_grid$min_child_weight[j]
          )
          dtrain <- xgb.DMatrix(as.matrix(X_inner_train_sel), label = y_inner_train)
          model <- xgb.train(params = xgb_params, data = dtrain, nrounds = 100, verbose = 0)
          dtest <- xgb.DMatrix(as.matrix(X_inner_test_sel))
          preds <- predict(model, dtest)
        }
        
        auc_inner[k] <- auc(y_inner_test, preds)
      }
      
      mean_auc_inner <- mean(auc_inner, na.rm = TRUE)
      if (mean_auc_inner > best_auc) {
        best_auc <- mean_auc_inner
        best_params <- param_grid[j,]
        best_model <- model
        best_selected_features <- selected_features
      }
    }
    
    if (!is.null(best_model)) {
      X_outer_test_sel <- X_outer_test[, best_selected_features, drop = FALSE]
      preds_outer <- predict(best_model, xgb.DMatrix(as.matrix(X_outer_test_sel)))
      outer_results[[i]] <- list(
        auc = auc(y_outer_test, preds_outer),
        selected_features = best_selected_features,
        best_params = best_params
      )
    }
  }
  return(outer_results)
}

xgb_nested <- nested_cv(X_train, y_train, model_type = "xgb", param_grid = xgb_param_grid)
saveRDS(xgb_nested, file = file.path(output_path_subgroup, paste0(outcome_var, "_xgb_nested.rds")))

models <- list()
models[[model_to_analyze]] <- xgb_nested

y_test <- test_data[[outcome_var]]
X_test_all <- dplyr::select(test_data, dplyr::all_of(independent_vars))
X_test_all <- as.data.frame(X_test_all)

final_predictions <- list()

for (model_name in names(models)) {
  nested_result <- models[[model_name]]
  
  all_features <- lapply(nested_result, function(x) unique(x$selected_features))
  feature_table <- table(unlist(all_features))
  fold_count <- length(nested_result)
  selected_features <- names(feature_table[feature_table > fold_count/2])
  
  params_list <- lapply(nested_result, function(x) x$best_params)
  params_df <- do.call(rbind, params_list)
  best_params <- sapply(params_df, function(col) {
    names(sort(table(col), decreasing = TRUE))[1]
  })
  
  X_test_final <- X_test_all[, selected_features, drop = FALSE]
  
  xgb_params <- list(
    objective = "binary:logistic", eval_metric = "auc",
    max_depth = as.numeric(best_params["max_depth"]),
    eta = as.numeric(best_params["eta"]),
    subsample = as.numeric(best_params["subsample"]),
    colsample_bytree = as.numeric(best_params["colsample_bytree"]),
    min_child_weight = as.numeric(best_params["min_child_weight"])
  )
  dtrain <- xgb.DMatrix(as.matrix(X_train[, selected_features]), label = y_train)
  final_model <- xgb.train(params = xgb_params, data = dtrain, nrounds = 100, verbose = 0)
  dtest <- xgb.DMatrix(as.matrix(X_test_final))
  preds <- predict(final_model, dtest)
  
  final_predictions[[model_name]] <- list(
    model = final_model,
    preds = preds,
    auc = auc(y_test, preds)
  )
}

roc_data_list <- list()
for (model_name in names(final_predictions)) {
  roc_obj <- roc(y_test, final_predictions[[model_name]]$preds)
  roc_data <- data.frame(
    fpr = 1 - roc_obj$specificities,
    tpr = roc_obj$sensitivities,
    thresholds = roc_obj$thresholds
  )
  roc_data_list[[model_name]] <- list(
    roc_df = roc_data,
    auc = auc(roc_obj)
  )
}
saveRDS(
  roc_data_list,
  file = file.path(output_path_subgroup, paste0(outcome_var, "_roc_data_list.rds"))
)

shap_data <- list()
for (model_name in names(final_predictions)) {
  nested_result <- models[[model_name]]
  
  all_features <- lapply(nested_result, function(x) unique(x$selected_features))
  feature_table <- table(unlist(all_features))
  fold_count <- length(nested_result)
  selected_features <- names(feature_table[feature_table > fold_count/2])
  
  X_test_final <- X_test_all[, selected_features, drop = FALSE]
  
  shap_data[[model_name]] <- list(
    model = final_predictions[[model_name]]$model,
    selected_features = selected_features,
    X_test_final = X_test_final,
    preds = final_predictions[[model_name]]$preds,
    auc = final_predictions[[model_name]]$auc
  )
}

shap_data$X_test_all <- X_test_all
shap_data$models <- models

saveRDS(
  shap_data,
  file = file.path(output_path_subgroup, paste0(outcome_var, "_shap_data.rds"))
)

final_model <- shap_data[[model_to_analyze]]$model
selected_features <- shap_data[[model_to_analyze]]$selected_features
X_test_shap <- shap_data[[model_to_analyze]]$X_test_final

X_test_shap <- as.matrix(X_test_shap)
colnames(X_test_shap) <- selected_features

shap_obj <- shapviz(final_model, X_pred = X_test_shap, X = X_test_shap)

saveRDS(
  shap_obj,
  file = file.path(output_path_subgroup, sprintf("%s_shapviz_object.rds", outcome_var))
)

nested_model1 <- xgb_nested
all_features <- lapply(nested_model1, function(x) unique(x$selected_features))
feature_table <- table(unlist(all_features))
fold_count <- length(nested_model1)
selected_independent_vars <- names(feature_table[feature_table > fold_count/2])

model1_name <- "Independent_vars"
model2_name <- "Independent_plus_Covariates"
model3_name <- "Covariates_only"

X_train_model2 <- dplyr::select(train_data_matched, dplyr::all_of(c(selected_independent_vars, covariate_vars)))
X_train_model2 <- as.data.frame(X_train_model2)

X_train_model3 <- dplyr::select(train_data_matched, dplyr::all_of(covariate_vars))
X_train_model3 <- as.data.frame(X_train_model3)

simple_cv <- function(X, y, outer_folds = 5, inner_folds = 5, model_type, param_grid, seed = 123) {
  set.seed(seed)
  
  outer_indices <- createFolds(y, k = outer_folds, list = TRUE, returnTrain = FALSE)
  outer_results <- list()
  
  for (i in 1:outer_folds) {
    test_idx <- outer_indices[[i]]
    train_idx <- setdiff(seq_len(nrow(X)), test_idx)
    
    X_outer_train <- X[train_idx, , drop = FALSE]
    y_outer_train <- y[train_idx]
    X_outer_test <- X[test_idx, , drop = FALSE]
    y_outer_test <- y[test_idx]
    
    best_auc <- 0
    best_params <- NULL
    best_model <- NULL
    
    for (j in 1:nrow(param_grid)) {
      inner_indices <- createFolds(y_outer_train, k = inner_folds, list = TRUE, returnTrain = FALSE)
      auc_inner <- c()
      
      for (k in 1:inner_folds) {
        inner_test_idx <- inner_indices[[k]]
        inner_train_idx <- setdiff(seq_len(nrow(X_outer_train)), inner_test_idx)
        
        X_inner_train <- X_outer_train[inner_train_idx, , drop = FALSE]
        y_inner_train <- y_outer_train[inner_train_idx]
        X_inner_test <- X_outer_train[inner_test_idx, , drop = FALSE]
        y_inner_test <- y_outer_train[inner_test_idx]
        
        if (model_type == "xgb") {
          xgb_params <- list(
            objective = "binary:logistic", eval_metric = "auc",
            max_depth = param_grid$max_depth[j], eta = param_grid$eta[j],
            subsample = param_grid$subsample[j], 
            colsample_bytree = param_grid$colsample_bytree[j],
            min_child_weight = param_grid$min_child_weight[j]
          )
          dtrain <- xgb.DMatrix(as.matrix(X_inner_train), label = y_inner_train)
          model <- xgb.train(params = xgb_params, data = dtrain, nrounds = 100, verbose = 0)
          dtest <- xgb.DMatrix(as.matrix(X_inner_test))
          preds <- predict(model, dtest)
          
          auc_inner[k] <- auc(y_inner_test, preds)
        }
      }
      
      mean_auc_inner <- mean(auc_inner, na.rm = TRUE)
      if(!is.na(mean_auc_inner) && mean_auc_inner > best_auc) {
        best_auc <- mean_auc_inner
        best_params <- param_grid[j,]
        best_model <- model
      }
    }
    
    if (!is.null(best_model)) {
      preds_outer <- predict(best_model, xgb.DMatrix(as.matrix(X_outer_test)))
      outer_results[[i]] <- list(
        auc = auc(y_outer_test, preds_outer),
        selected_features = colnames(X),
        best_params = best_params
      )
    }
  }
  return(outer_results)
}

nested_model2 <- simple_cv(X_train_model2, y_train, model_type = "xgb", param_grid = xgb_param_grid)
saveRDS(nested_model2, file = file.path(output_path_subgroup, paste0(outcome_var, "_xgb_nested_model2.rds")))

nested_model3 <- simple_cv(X_train_model3, y_train, model_type = "xgb", param_grid = xgb_param_grid)
saveRDS(nested_model3, file = file.path(output_path_subgroup, paste0(outcome_var, "_xgb_nested_model3.rds")))

X_test_model1 <- dplyr::select(test_data, dplyr::all_of(selected_independent_vars))
X_test_model1 <- as.data.frame(X_test_model1)

X_test_model2 <- dplyr::select(test_data, dplyr::all_of(c(selected_independent_vars, covariate_vars)))
X_test_model2 <- as.data.frame(X_test_model2)

X_test_model3 <- dplyr::select(test_data, dplyr::all_of(covariate_vars))
X_test_model3 <- as.data.frame(X_test_model3)

evaluate_simple_model <- function(nested_result, X_train, y_train, X_test, model_name) {
  
  X_train <- as.data.frame(X_train)
  X_test <- as.data.frame(X_test)
  
  selected_features <- colnames(X_train)
  
  params_list <- lapply(nested_result, function(x) x$best_params)
  params_df <- do.call(rbind, params_list)
  best_params <- sapply(params_df, function(col) {
    names(sort(table(col), decreasing = TRUE))[1]
  })
  
  xgb_params <- list(
    objective = "binary:logistic", eval_metric = "auc",
    max_depth = as.numeric(best_params["max_depth"]),
    eta = as.numeric(best_params["eta"]),
    subsample = as.numeric(best_params["subsample"]),
    colsample_bytree = as.numeric(best_params["colsample_bytree"]),
    min_child_weight = as.numeric(best_params["min_child_weight"])
  )
  
  dtrain <- xgb.DMatrix(as.matrix(X_train), label = y_train)
  final_model <- xgb.train(params = xgb_params, data = dtrain, nrounds = 100, verbose = 0)
  
  dtest <- xgb.DMatrix(as.matrix(X_test))
  preds <- predict(final_model, dtest)
  
  return(list(
    model = final_model,
    preds = preds,
    selected_features = selected_features
  ))
}

X_train_model1 <- dplyr::select(train_data_matched, dplyr::all_of(selected_independent_vars))
result1 <- evaluate_simple_model(nested_model1, X_train_model1, y_train, X_test_model1, model1_name)
result2 <- evaluate_simple_model(nested_model2, X_train_model2, y_train, X_test_model2, model2_name)
result3 <- evaluate_simple_model(nested_model3, X_train_model3, y_train, X_test_model3, model3_name)

roc1 <- roc(y_test, result1$preds)
roc2 <- roc(y_test, result2$preds)
roc3 <- roc(y_test, result3$preds)

roc_data <- data.frame(
  FPR = c(1 - roc1$specificities, 1 - roc2$specificities, 1 - roc3$specificities),
  TPR = c(roc1$sensitivities, roc2$sensitivities, roc3$sensitivities),
  Model = rep(c(
    sprintf("Exposures (AUC=%.3f)", auc(roc1)),
    sprintf("Exposures + Demographic (AUC=%.3f)", auc(roc2)),
    sprintf("Demographic (AUC=%.3f)", auc(roc3))
  ), c(length(roc1$specificities), length(roc2$specificities), length(roc3$specificities)))
)

saveRDS(
  roc_data,
  file = file.path(output_path_subgroup, paste0(outcome_var, "_ROC_plot_data.rds"))
)

ci1 <- ci.auc(roc1)
ci2 <- ci.auc(roc2)
ci3 <- ci.auc(roc3)

comparison_results <- data.frame(
  Model = c("Independent_vars", "Independent_plus_Covariates", "Covariates_only"),
  Test_AUC = c(auc(roc1), auc(roc2), auc(roc3)),
  AUC_CI_lower = c(ci1[1], ci2[1], ci3[1]),
  AUC_CI_upper = c(ci1[3], ci2[3], ci3[3]),
  N_features = c(
    length(result1$selected_features), 
    length(result2$selected_features), 
    length(result3$selected_features)
  ),
  N_independent_vars = c(
    length(selected_independent_vars),
    length(selected_independent_vars),
    0
  ),
  N_covariates = c(
    0,
    length(covariate_vars),
    length(covariate_vars)
  )
)

write.csv(
  comparison_results,
  file = file.path(output_path_subgroup, paste0(outcome_var, "_model_comparison.csv")),
  row.names = FALSE
)

risk_thresholds <- quantile(result3$preds, probs = c(0.33, 0.67))

categorize_risk <- function(pred, thresholds) {
  cut(pred, 
      breaks = c(-Inf, thresholds[1], thresholds[2], Inf),
      labels = c("Low", "Medium", "High"),
      include.lowest = TRUE)
}

risk_cat_model2 <- categorize_risk(result2$preds, risk_thresholds)
risk_cat_model3 <- categorize_risk(result3$preds, risk_thresholds)

calculate_categorical_nri <- function(y_true, risk_old, risk_new) {
  
  risk_old_num <- as.numeric(risk_old)
  risk_new_num <- as.numeric(risk_new)
  
  events <- y_true == 1
  up_events <- sum(risk_new_num > risk_old_num & events)
  down_events <- sum(risk_new_num < risk_old_num & events)
  nri_events <- (up_events - down_events) / sum(events)
  
  non_events <- y_true == 0
  down_non_events <- sum(risk_new_num < risk_old_num & non_events)
  up_non_events <- sum(risk_new_num > risk_old_num & non_events)
  nri_non_events <- (down_non_events - up_non_events) / sum(non_events)
  
  nri_total <- nri_events + nri_non_events
  
  return(list(
    NRI_events = nri_events,
    NRI_non_events = nri_non_events,
    NRI_total = nri_total,
    up_events = up_events,
    down_events = down_events,
    up_non_events = up_non_events,
    down_non_events = down_non_events,
    total_events = sum(events),
    total_non_events = sum(non_events)
  ))
}

nri_result <- calculate_categorical_nri(y_test, risk_cat_model3, risk_cat_model2)

calculate_continuous_nri <- function(y_true, pred_old, pred_new) {
  
  events <- y_true == 1
  nri_events <- mean(pred_new[events] > pred_old[events]) - 
    mean(pred_new[events] < pred_old[events])
  
  non_events <- y_true == 0
  nri_non_events <- mean(pred_new[non_events] < pred_old[non_events]) - 
    mean(pred_new[non_events] > pred_old[non_events])
  
  nri_total <- nri_events + nri_non_events
  
  return(list(
    NRI_events = nri_events,
    NRI_non_events = nri_non_events,
    NRI_total = nri_total
  ))
}

cnri_result <- calculate_continuous_nri(y_test, result3$preds, result2$preds)

bootstrap_nri <- function(y_true, pred_old, pred_new, thresholds, n_boot = 2000) {
  set.seed(123)
  n <- length(y_true)
  
  boot_cat_nri <- numeric(n_boot)
  boot_cont_nri <- numeric(n_boot)
  
  for(i in 1:n_boot) {
    boot_idx <- sample(1:n, n, replace = TRUE)
    boot_y <- y_true[boot_idx]
    boot_pred_old <- pred_old[boot_idx]
    boot_pred_new <- pred_new[boot_idx]
    
    boot_risk_old <- categorize_risk(boot_pred_old, thresholds)
    boot_risk_new <- categorize_risk(boot_pred_new, thresholds)
    cat_nri <- calculate_categorical_nri(boot_y, boot_risk_old, boot_risk_new)
    boot_cat_nri[i] <- cat_nri$NRI_total
    
    cont_nri <- calculate_continuous_nri(boot_y, boot_pred_old, boot_pred_new)
    boot_cont_nri[i] <- cont_nri$NRI_total
  }
  
  cat_nri_ci <- quantile(boot_cat_nri, c(0.025, 0.975))
  cont_nri_ci <- quantile(boot_cont_nri, c(0.025, 0.975))
  
  return(list(
    categorical_NRI_CI = cat_nri_ci,
    continuous_NRI_CI = cont_nri_ci
  ))
}

boot_results <- bootstrap_nri(y_test, result3$preds, result2$preds, risk_thresholds)

calculate_idi <- function(y_true, pred_old, pred_new) {
  
  events <- y_true == 1
  idi_events <- mean(pred_new[events]) - mean(pred_old[events])
  
  non_events <- y_true == 0
  idi_non_events <- mean(pred_old[non_events]) - mean(pred_new[non_events])
  
  idi <- idi_events + idi_non_events
  
  return(list(
    IDI_events = idi_events,
    IDI_non_events = idi_non_events,
    IDI_total = idi
  ))
}

idi_result <- calculate_idi(y_test, result3$preds, result2$preds)

bootstrap_idi <- function(y_true, pred_old, pred_new, n_boot = 2000) {
  set.seed(123)
  n <- length(y_true)
  boot_idi <- numeric(n_boot)
  
  for(i in 1:n_boot) {
    boot_idx <- sample(1:n, n, replace = TRUE)
    idi <- calculate_idi(y_true[boot_idx], pred_old[boot_idx], pred_new[boot_idx])
    boot_idi[i] <- idi$IDI_total
  }
  
  return(quantile(boot_idi, c(0.025, 0.975)))
}

idi_ci <- bootstrap_idi(y_test, result3$preds, result2$preds)

nri_summary <- data.frame(
  Metric = c("Category-based NRI (Events)", 
             "Category-based NRI (Non-events)", 
             "Category-based NRI (Total)",
             "Continuous NRI (Events)",
             "Continuous NRI (Non-events)",
             "Continuous NRI (Total)",
             "IDI (Events)",
             "IDI (Non-events)",
             "IDI (Total)"),
  
  Value = c(nri_result$NRI_events,
            nri_result$NRI_non_events,
            nri_result$NRI_total,
            cnri_result$NRI_events,
            cnri_result$NRI_non_events,
            cnri_result$NRI_total,
            idi_result$IDI_events,
            idi_result$IDI_non_events,
            idi_result$IDI_total),
  
  CI_lower = c(NA, NA, boot_results$categorical_NRI_CI[1],
               NA, NA, boot_results$continuous_NRI_CI[1],
               NA, NA, idi_ci[1]),
  
  CI_upper = c(NA, NA, boot_results$categorical_NRI_CI[2],
               NA, NA, boot_results$continuous_NRI_CI[2],
               NA, NA, idi_ci[2])
)

write.csv(
  nri_summary,
  file = file.path(output_path_subgroup, paste0(outcome_var, "_NRI_IDI_results.csv")),
  row.names = FALSE
)

saveRDS(
  list(
    train_data_matched = train_data_matched,
    test_data = test_data,
    covariate_vars = covariate_vars
  ),
  file.path(output_path_subgroup, "02_prediction_handoff.rds")
)

