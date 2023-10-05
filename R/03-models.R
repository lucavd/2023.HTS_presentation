source('R/01-data_generation.R')

# Load required packages
library(tidyverse)
library(pROC)
library(caret)
library(mgcv)
library(ROCR)
library(lme4)
library(randomForest)
library(ranger)

# Convert 'activity' to factor variable and explicitly set the levels
final_dataset$activity <- factor(final_dataset$activity, levels = c(0, 1), labels = c("Level_0", "Level_1"))

# Bootstrap function for AUC 95%CI ----------------------------------------

bootstrap_auc <- function(auc_vec, n_bootstraps = 1000, alpha = 0.05) {
  bootstrapped_auc <- numeric(n_bootstraps)
  n = length(auc_vec)
  
  for(i in 1:n_bootstraps) {
    # Eseguire il campionamento con sostituzione
    resampled_indices <- sample(1:n, n, replace = TRUE)
    bootstrapped_auc[i] <- mean(auc_vec[resampled_indices])
  }
  
  # Calcolare gli intervalli di confidenza
  ci_lower <- quantile(bootstrapped_auc, alpha / 2)
  ci_upper <- quantile(bootstrapped_auc, 1 - alpha / 2)
  
  return(data.frame(CI_lower = ci_lower, CI_upper = ci_upper))
}


# Initialize cross validation ---------------------------------------------

set.seed(42)
folds <- createFolds(final_dataset$activity, k = 10)


# 1. GLM ------------------------------------------------------------------

# Preallocate list to store ROC values
list_roc_values <- list()

# Initialize variable to store AUC values
auc_glm <- numeric(length = 10)

for (i in seq_along(folds)) {
  # Subset the data based on the folds
  train_data <- final_dataset[-folds[[i]],]
  test_data <- final_dataset[folds[[i]],]
  
  # Fit logistic regression model
  logistic_model <- glm(activity ~ measure1 + measure2 + measure3 + measure4 + measure5, 
                        data = train_data, 
                        family = binomial())
  
  # Make predictions on the test set
  pred_probs <- predict(logistic_model, newdata = test_data, type = "response")
  
  # Compute ROC values
  pred_obj <- prediction(pred_probs, test_data$activity)
  perf_obj <- performance(pred_obj, measure = "tpr", x.measure = "fpr")
  
  # Compute AUC
  auc_glm[i] <- roc(test_data$activity, pred_probs)$auc
  
  # Store TPR and FPR
  list_roc_values[[i]] <- data.frame(TPR = unlist(perf_obj@y.values), FPR = unlist(perf_obj@x.values))
}

# Compute average AUC across all folds
mean_auc_glm <- mean(auc_glm)

ic_auc_glm <- bootstrap_auc(auc_glm)

# Display the average AUC
print(paste("Average AUC for Logistic Regression with Spline: ",
            round(mean_auc_glm, 2), "[", round(ic_auc_glm$CI_lower, 2),
            "-",
            round(ic_auc_glm$CI_upper, 2), "]"))

# Combine all TPR and FPR
all_roc_values <- do.call(rbind, list_roc_values)

# Compute mean and 95% CI
mean_roc <- all_roc_values %>% group_by(FPR) %>% summarise(Mean_TPR = mean(TPR), .groups = 'drop')
ci_roc <- all_roc_values %>% group_by(FPR) %>% summarise(CI_low = quantile(TPR, 0.025), CI_high = quantile(TPR, 0.975), .groups = 'drop')

# Plotting
ggplot(data = mean_roc, aes(x = FPR, y = mean_roc$Mean_TPR)) +
  geom_line(color = "blue") +
  geom_ribbon(data = ci_roc, aes(ymin = CI_low, ymax = CI_high), alpha = 0.2) +
  labs(title = "Mean GLM ROC Curve with 95% CI",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  annotate('text', x = .7, y = .2, label = paste("Average AUC for GLM: ",
                                                 round(mean_auc_glm, 2), "[", round(ic_auc_glm$CI_lower, 2),
                                                 "-",
                                                 round(ic_auc_glm$CI_upper, 2), "]"))



# 2. Non parametric logistic regression - GAM spline -----------

# Initialize variable to store AUC values
auc_nplr_spline <- numeric(length = 10)
list_roc_values_gam <- list()

# Loop through each fold
for(i in seq_along(folds)){
  # Subset data into training and test sets based on folds
  train_data <- final_dataset[-folds[[i]],]
  test_data <- final_dataset[folds[[i]],]
  
  # Fit non-parametric logistic regression model using spline terms
  nplr_spline_model <- gam(activity ~ s(measure1) + s(measure2) + s(measure3) + s(measure4) + s(measure5), 
                           family = binomial(link = "logit"), 
                           data = train_data)
  
  # Predict probabilities on the test set
  pred_probs_gam <- as.vector(predict(nplr_spline_model, newdata = test_data, type = "response"))
  
  # Compute ROC values
  pred_obj_gam <- prediction(pred_probs_gam, test_data$activity)
  perf_obj_gam <- performance(pred_obj_gam, measure = "tpr", x.measure = "fpr")
  
  # Compute AUC
  auc_nplr_spline[i] <- roc(test_data$activity, pred_probs_gam)$auc
  
  # Store TPR and FPR
  list_roc_values_gam[[i]] <- data.frame(TPR = unlist(perf_obj_gam@y.values), 
                                         FPR = unlist(perf_obj_gam@x.values))
}

# Compute average AUC across all folds
mean_auc_nplr_spline <- mean(auc_nplr_spline)

ic_auc_nplr_spline <- bootstrap_auc(auc_nplr_spline)

# Display the average AUC
print(paste("Average AUC for Non-Parametric Logistic Regression with Spline: ",
            round(mean_auc_nplr_spline, 2), "[", round(ic_auc_nplr_spline$CI_lower, 2),
            "-",
            round(ic_auc_nplr_spline$CI_upper, 2), "]"))

# Combine all TPR and FPR
all_roc_values <- do.call(rbind, list_roc_values_gam)

# Compute mean and 95% CI
mean_roc_gam <- all_roc_values %>% group_by(FPR) %>% summarise(Mean_TPR = mean(TPR), .groups = 'drop')
ci_roc_gam <- all_roc_values %>% group_by(FPR) %>% summarise(CI_low = quantile(TPR, 0.025), CI_high = quantile(TPR, 0.975), .groups = 'drop')

# Plotting
ggplot(data = mean_roc_gam, aes(x = FPR, y = mean_roc_gam$Mean_TPR)) +
  geom_line(color = "blue") +
  geom_ribbon(data = ci_roc_gam, aes(ymin = CI_low, ymax = CI_high), alpha = 0.2) +
  labs(title = "Mean GAM ROC Curve with 95% CI",
       x = "False Positive Rate",
       y = "True Positive Rate")+
  annotate('text', x = .7, y = .2, label = paste("Average AUC for GAM: ",
                                                 round(mean_auc_nplr_spline, 2), "[", round(ic_auc_nplr_spline$CI_lower, 2),
                                                 "-",
                                                 round(ic_auc_nplr_spline$CI_upper, 2), "]"))


# GLMM --------------------------------------------------------------------

auc_glmm <- numeric(length(folds))
list_roc_values_glmm <- list()

# Cross-Validation Loop for GLMM
for(i in seq_along(folds)){
  # Subset data into training and test sets based on folds
  train_data <- final_dataset[-folds[[i]],]
  test_data <- final_dataset[folds[[i]],]
  
  # Fit GLMM Model with random effect compound_id
  glmm_model <- glmer(activity ~ measure1 + measure2 + measure3 + measure4 + measure5 + (1|compound_id),
                      family = binomial(link="logit"),
                      data = train_data)
  
  # Use 'tryCatch' to handle potential errors during prediction
  tryCatch({
    pred_probs_glmm <- predict(glmm_model, newdata = test_data, type = "response", re.form=NA)
    
    # Compute AUC using ROCR package
    pred_obj_glmm <- prediction(pred_probs_glmm, as.numeric(test_data$activity))
    perf_obj_glmm <- performance(pred_obj_glmm, measure = "tpr", x.measure = "fpr")
    auc_obj_glmm <- performance(pred_obj_glmm, measure = "auc")
    # Store TPR and FPR
    list_roc_values_glmm[[i]] <- data.frame(TPR = unlist(perf_obj_glmm@y.values),
                                            FPR = unlist(perf_obj_glmm@x.values))
    auc_glmm[i] <- as.numeric(auc_obj_glmm@y.values)
  }, error = function(e) {
    message("Prediction failed for this fold with error: ", e)
    auc_glmm[i] <- NA
    list_roc_values_glmm[[i]] <- NA
  })
}

# Average AUC over the 10 folds (omitting failed predictions)
mean_auc_glmm <- mean(auc_glmm, na.rm = TRUE)

ic_auc_glmm <- bootstrap_auc(auc_glmm)

# Display the average AUC
print(paste("Average AUC for GLMM: ",
            round(mean_auc_glmm, 2), "[", round(ic_auc_glmm$CI_lower, 2),
            "-",
            round(ic_auc_glmm$CI_upper, 2), "]"))

# Combine all TPR and FPR
all_roc_values <- do.call(rbind, list_roc_values_glmm)

# Compute mean and 95% CI
mean_roc_glmm <- all_roc_values %>% group_by(FPR) %>% summarise(Mean_TPR = mean(TPR), .groups = 'drop')
ci_roc_glmm <- all_roc_values %>% group_by(FPR) %>% summarise(CI_low = quantile(TPR, 0.025), CI_high = quantile(TPR, 0.975), .groups = 'drop')

# Plotting
ggplot(data = mean_roc_glmm, aes(x = FPR, y = mean_roc_glmm$Mean_TPR)) +
  geom_line(color = "blue") +
  geom_ribbon(data = ci_roc_glmm, aes(ymin = CI_low, ymax = CI_high), alpha = 0.2) +
  labs(title = "Mean GLMM ROC Curve with 95% CI",
       x = "False Positive Rate",
       y = "True Positive Rate")+
  annotate('text', x = .7, y = .2, label = paste("Average AUC for GLMM: ",
                                                 round(mean_auc_glmm, 2), "[", round(ic_auc_glmm$CI_lower, 2),
                                                 "-",
                                                 round(ic_auc_glmm$CI_upper, 2), "]"))


# Random Forest -----------------------------------------------------------

auc_rf <- numeric(length(folds))
list_roc_values_rf <- list()

# Cross-Validation Loop for Random Forest
for(i in seq_along(folds)){
  # Subset data into training and test sets based on folds
  train_data <- final_dataset[-folds[[i]],]
  test_data <- final_dataset[folds[[i]],]
  
  # Fit Random Forest Model
  rf_model <- randomForest(activity ~ measure1 + measure2 + measure3 + measure4 + measure5,
                           data = train_data, 
                           ntree = 500)
  
  # Predict probabilities on the test set
  pred_probs_rf <- as.numeric(predict(rf_model, newdata = test_data, type = "response"))
  pred_obj_rf <- prediction(pred_probs_rf, test_data$activity)
  perf_obj_rf <- performance(pred_obj_rf, measure = "tpr", x.measure = "fpr")
  
  # Compute AUC using ROCR package
  
  auc_obj_rf <- performance(pred_obj_rf, measure = "auc")
  auc_rf[i] <- as.numeric(auc_obj_rf@y.values)
  
  # Store TPR and FPR
  list_roc_values_rf[[i]] <- data.frame(TPR = unlist(perf_obj_rf@y.values), 
                                        FPR = unlist(perf_obj_rf@x.values))
}

# Average AUC over the 10 folds
mean_auc_rf <- mean(auc_rf)

ic_auc_rf <- bootstrap_auc(auc_rf)

# Display the average AUC
print(paste("Average AUC for RF: ",
            round(mean_auc_rf, 2), "[", round(ic_auc_rf$CI_lower, 2),
            "-",
            round(ic_auc_rf$CI_upper, 2), "]"))

# Combine all TPR and FPR
all_roc_values <- do.call(rbind, list_roc_values_rf)

# Compute mean and 95% CI
mean_roc_rf <- all_roc_values %>% group_by(FPR) %>% summarise(Mean_TPR = mean(TPR), .groups = 'drop')
ci_roc_rf <- all_roc_values %>% group_by(FPR) %>% summarise(CI_low = quantile(TPR, 0.025), CI_high = quantile(TPR, 0.975), .groups = 'drop')

# Plotting
ggplot(data = mean_roc_rf, aes(x = FPR, y = mean_roc_rf$Mean_TPR)) +
  geom_line(color = "blue") +
  geom_ribbon(data = ci_roc_rf, aes(ymin = CI_low, ymax = CI_high), alpha = 0.2) +
  labs(title = "Mean RF ROC Curve with 95% CI",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  annotate('text', x = .7, y = .2, label = paste("Average AUC for RF: ",
                                                 round(mean_auc_rf, 2), "[", round(ic_auc_rf$CI_lower, 2),
                                                 "-",
                                                 round(ic_auc_rf$CI_upper, 2), "]"))


# Random Forest Classification with ranger --------------------------------

# Initialize an empty vector to store AUC values from each fold
auc_rf_ranger <- numeric(length(folds))
list_roc_values_ranger <- list()

# Perform 10-fold cross-validation
for(i in seq_along(folds)){
  # Subset data into training and test sets based on folds
  train_data <- final_dataset[-folds[[i]],]
  test_data <- final_dataset[folds[[i]],]
  
  # Fit the Random Forest model using ranger
  rf_ranger_model <- ranger(activity ~ measure1 + measure2 + measure3 + measure4 + measure5, 
                            data = train_data,
                            probability = TRUE, 
                            num.trees = 500)
  
  # Predict probabilities on the test set
  pred_probs_rf_ranger <- predict(rf_ranger_model, data = test_data)$predictions[,2]
  
  
  # Compute AUC using the ROCR package
  pred_obj_rf_ranger <- prediction(pred_probs_rf_ranger, test_data$activity)
  perf_obj_ranger <- performance(pred_obj_rf_ranger, measure = "tpr", x.measure = "fpr")
  auc_obj_rf_ranger <- performance(pred_obj_rf_ranger, measure = "auc")
  auc_rf_ranger[i] <- as.numeric(auc_obj_rf_ranger@y.values)
  
  # Store TPR and FPR
  list_roc_values_ranger[[i]] <- data.frame(TPR = unlist(perf_obj_ranger@y.values), 
                                            FPR = unlist(perf_obj_ranger@x.values))
}

# Calculate the mean AUC over the 10 folds
mean_auc_rf_ranger <- mean(auc_rf_ranger)

mean_auc_rf_ranger

ic_auc_rf_ranger<- bootstrap_auc(auc_rf_ranger)

# Display the average AUC
print(paste("Average AUC for RF - Ranger: ",
            round(mean_auc_rf_ranger, 2), "[", round(ic_auc_rf_ranger$CI_lower, 2),
            "-",
            round(ic_auc_rf_ranger$CI_upper, 2), "]"))

# Combine all TPR and FPR
all_roc_values <- do.call(rbind, list_roc_values_ranger)

# Compute mean and 95% CI
mean_roc_ranger <- all_roc_values %>% group_by(FPR) %>% summarise(Mean_TPR = mean(TPR), .groups = 'drop')
ci_roc_ranger <- all_roc_values %>% group_by(FPR) %>% summarise(CI_low = quantile(TPR, 0.025), CI_high = quantile(TPR, 0.975), .groups = 'drop')

# Plotting
ggplot(data = mean_roc_ranger, aes(x = FPR, y = mean_roc_ranger$Mean_TPR)) +
  geom_line(color = "blue") +
  geom_ribbon(data = ci_roc_ranger, aes(ymin = CI_low, ymax = CI_high), alpha = 0.2) +
  labs(title = "Mean RF Ranger ROC Curve with 95% CI",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  annotate('text', x = .6, y = .2, label = paste("Average AUC for RF - Ranger: ",
                                                 round(mean_auc_rf_ranger, 2), "[", round(ic_auc_rf_ranger$CI_lower, 2),
                                                 "-",
                                                 round(ic_auc_rf_ranger$CI_upper, 2), "]"))

# xgboost -----------------------------------------------------------------
library(xgboost)
#load here since masks 'slice' from dplyr

# Prepare matrix form required by xgboost
train_matrix <- as.matrix(final_dataset[, c("measure1", "measure2", "measure3", "measure4", "measure5")])
label_vector <- as.numeric(final_dataset$activity) - 1  # zero-based labels

# 10-Fold Cross-Validation Setup
auc_xgb <- numeric(length(folds))
list_roc_values_xgb <- list()

# Cross-Validation Loop for XGBoost
for(i in seq_along(folds)){
  # Subset data into training and test sets based on folds
  train_data <- train_matrix[-folds[[i]],, drop = FALSE]
  train_label <- label_vector[-folds[[i]]]
  
  test_data <- train_matrix[folds[[i]],, drop = FALSE]
  test_label <- label_vector[folds[[i]]]
  
  # Fit XGBoost model
  xgb_model <- xgboost(data = train_data, label = train_label, nrounds = 100, objective = "binary:logistic")
  
  # Predict probabilities on the test set
  pred_probs_xgb <- predict(xgb_model, newdata = test_data)
  
  # Compute AUC using ROCR package
  pred_obj_xgb <- prediction(pred_probs_xgb, test_label)
  perf_obj_xgb <- performance(pred_obj_xgb, measure = "tpr", x.measure = "fpr")
  auc_obj_xgb <- performance(pred_obj_xgb, measure = "auc")
  auc_xgb[i] <- as.numeric(auc_obj_xgb@y.values)
  
  list_roc_values_xgb[[i]] <- data.frame(TPR = unlist(perf_obj_xgb@y.values), 
                                     FPR = unlist(perf_obj_xgb@x.values))
}

# Average AUC over the 10 folds
mean_auc_xgb <- mean(auc_xgb)

ic_auc_xgb<- bootstrap_auc(auc_xgb)

# Display the average AUC
print(paste("Average AUC for XGB: ",
            round(mean_auc_xgb, 2), "[", round(ic_auc_xgb$CI_lower, 2),
            "-",
            round(ic_auc_xgb$CI_upper, 2), "]"))

# Combine all TPR and FPR
all_roc_values <- do.call(rbind, list_roc_values_xgb)

# Compute mean and 95% CI
mean_roc_xgb <- all_roc_values %>% group_by(FPR) %>% summarise(Mean_TPR = mean(TPR), .groups = 'drop')
ci_roc_xgb <- all_roc_values %>% group_by(FPR) %>% summarise(CI_low = quantile(TPR, 0.025), CI_high = quantile(TPR, 0.975), .groups = 'drop')

# Plotting
ggplot(data = mean_roc_xgb, aes(x = FPR, y = mean_roc_xgb$Mean_TPR)) +
  geom_line(color = "blue") +
  geom_ribbon(data = ci_roc_xgb, aes(ymin = CI_low, ymax = CI_high), alpha = 0.2) +
  labs(title = "Mean XGB ROC Curve with 95% CI",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  annotate('text', x = .6, y = .2, label = paste("Average AUC for XGB: ",
                                                 round(mean_auc_xgb, 2), "[", round(ic_auc_xgb$CI_lower, 2),
                                                 "-",
                                                 round(ic_auc_xgb$CI_upper, 2), "]"))

# autoML -------------------------------------------------------------

# Title: AutoML Classification with h2o

# download the latest Java SE JDK
# https://www.oracle.com/java/technologies/downloads/#jdk21-windows
# use a clean session of R (Restart R)
# XGBOOST NOT WORKING ON WINDOWS

source("R/01-data_generation.R")

# Load the necessary package
library(h2o)

# Initialize h2o
h2o.init(nthreads = -1, max_mem_size = '50G')

# Convert the dataset into h2o object
final_dataset$activity <- factor(final_dataset$activity, levels = c(0, 1), labels = c("Level_0", "Level_1"))
final_dataset_h2o <- as.h2o(final_dataset)

# Specify the predictors and response variable
predictors <- c("measure1", "measure2", "measure3", "measure4", "measure5")
response <- "activity"

# Run AutoML with 10-fold cross-validation
automl_models <- h2o.automl(x = predictors,
                            y = response,
                            training_frame = final_dataset_h2o,
                            max_runtime_secs = 600,
                            nfolds = 10,
                            sort_metric = 'AUC')


# Leaderboard
lb <- h2o.get_leaderboard(object = automl_models, extra_columns = "ALL")

# Extract the best model
best_model <- automl_models@leader

# this is equivalent to
best_model <- h2o.get_best_model(automl_models)

# Get the best model using a non-default metric
best_model_logloss <- h2o.get_best_model(automl_models, criterion = "logloss")

# Get the best XGBoost model using default sort metric
xgb <- h2o.get_best_model(automl_models, algorithm = "xgboost")

# Get the best XGBoost model, ranked by logloss
xgb_logloss <- h2o.get_best_model(automl_models, algorithm = "xgboost", criterion = "logloss")

# Extract AUC of the best model on cross-validation
auc_automl <- h2o.auc(h2o.performance(best_model, xval = TRUE))

sens_automl <- h2o.sensitivity(h2o.performance(best_model, xval = TRUE))

spec_automl <- h2o.specificity(h2o.performance(best_model, xval = TRUE))

# Shut down the h2o cluster
h2o.shutdown(prompt = FALSE)





