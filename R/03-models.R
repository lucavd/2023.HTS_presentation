# Load required packages
library(tidyverse)
library(pROC)
library(caret)



# 1. GLM ------------------------------------------------------------------

# Convert 'activity' to factor variable and explicitly set the levels
final_dataset$activity <- factor(final_dataset$activity, levels = c(0, 1), labels = c("Level_0", "Level_1"))

# Initialize parameters for k-fold cross-validation
set.seed(42)
control_lr <- trainControl(method = "cv", number = 10, classProbs = TRUE, 
                           summaryFunction = twoClassSummary, savePredictions = "final")

# Fit logistic regression model with 10-fold cross-validation
cv_logistic_model_lr <- train(activity ~ measure1 + measure2 + measure3 + measure4 + measure5, 
                              data = final_dataset, 
                              method = "glm", 
                              family = binomial(link = "logit"), 
                              trControl = control_lr,
                              metric = "ROC")

# Retrieve the AUC-ROC from the cross-validation model
auc_cv_lr <- cv_logistic_model_lr$results[1, "ROC"]



# 2. Non parametric logistic regression - spline -----------

# Load the mgcv package for GAMs
# Load the required libraries
library(mgcv)
library(caret)

# Initialize 10-fold cross-validation
set.seed(42)
folds <- createFolds(final_dataset$activity, k = 10)

# Initialize variable to store AUC values
auc_nplr_spline <- numeric(length = 10)

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
  pred_probs <- predict(nplr_spline_model, newdata = test_data, type = "response")
  
  # Compute AUC
  auc_nplr_spline[i] <- roc(test_data$activity, pred_probs)$auc
}

# Compute average AUC across all folds
mean_auc_nplr_spline <- mean(auc_nplr_spline)

# Display the average AUC
print(paste("Average AUC for Non-Parametric Logistic Regression with Spline: ", round(mean_auc_nplr_spline, 3)))



# GAM ---------------------------------------------------------------------

# Import Required Libraries
library(mgcv)
library(ROCR)
library(caret)

# 10-Fold Cross-Validation Setup
folds <- createFolds(final_dataset$activity, k = 10, list = TRUE)
auc_gam <- numeric(length(folds))

# Cross-Validation Loop for GAM
for(i in seq_along(folds)) {
  # Subset data into training and test sets based on folds
  train_data <- final_dataset[-folds[[i]],]
  test_data <- final_dataset[folds[[i]],]
  
  # Fit GAM Model
  gam_model <- gam(activity ~ s(measure1) + s(measure2) + s(measure3) + s(measure4) + s(measure5),
                   family = binomial,
                   data = train_data)
  
  # Predict Probabilities on the Test Set
  pred_probs <- predict(gam_model, newdata = test_data, type = "response")
  
  # Ensure pred_probs is a numeric vector before using with ROCR
  pred_probs <- as.numeric(pred_probs)
  
  # Compute AUC using ROCR package
  pred_obj <- prediction(pred_probs, as.numeric(test_data$activity))
  auc_obj <- performance(pred_obj, measure = "auc")
  auc_gam[i] <- as.numeric(auc_obj@y.values)
}

# Average AUC over the 10 folds
mean_auc_gam <- mean(auc_gam)

mean_auc_gam

# GLMM --------------------------------------------------------------------

# Title: Generalized Linear Mixed Models (GLMM)
library(lme4)

# 10-Fold Cross-Validation Setup
folds <- createFolds(final_dataset$activity, k = 10, list = TRUE)
auc_glmm <- numeric(length(folds))

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
    pred_probs <- predict(glmm_model, newdata = test_data, type = "response", re.form=NA)
    
    # Compute AUC using ROCR package
    pred_obj <- prediction(pred_probs, as.numeric(test_data$activity))
    auc_obj <- performance(pred_obj, measure = "auc")
    auc_glmm[i] <- as.numeric(auc_obj@y.values)
  }, error = function(e) {
    message("Prediction failed for this fold with error: ", e)
    auc_glmm[i] <- NA
  })
}

# Average AUC over the 10 folds (omitting failed predictions)
mean_auc_glmm <- mean(auc_glmm, na.rm = TRUE)

mean_auc_glmm


# Random Forest -----------------------------------------------------------

# Title: Random Forest Classification

# Load necessary package
library(randomForest)

# 10-Fold Cross-Validation Setup
folds <- createFolds(final_dataset$activity, k = 10, list = TRUE)
auc_rf <- numeric(length(folds))

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
  pred_probs <- as.numeric(predict(rf_model, newdata = test_data, type = "response"))
  
  # Compute AUC using ROCR package
  pred_obj <- prediction(pred_probs, test_data$activity)
  auc_obj <- performance(pred_obj, measure = "auc")
  auc_rf[i] <- as.numeric(auc_obj@y.values)
}

# Average AUC over the 10 folds
mean_auc_rf <- mean(auc_rf)

mean_auc_rf

# Title: Random Forest Classification with ranger

# Load the ranger package
library(ranger)

# Initialize an empty vector to store AUC values from each fold
auc_rf_ranger <- numeric(length(folds))

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
  auc_obj_rf_ranger <- performance(pred_obj_rf_ranger, measure = "auc")
  auc_rf_ranger[i] <- as.numeric(auc_obj_rf_ranger@y.values)
}

# Calculate the mean AUC over the 10 folds
mean_auc_rf_ranger <- mean(auc_rf_ranger)

mean_auc_rf_ranger

# xgboost -----------------------------------------------------------------

# Title: XGBoost Classification

# Load necessary packages
library(xgboost)

# Prepare matrix form required by xgboost
train_matrix <- as.matrix(final_dataset[, c("measure1", "measure2", "measure3", "measure4", "measure5")])
label_vector <- as.numeric(final_dataset$activity) - 1  # zero-based labels

# 10-Fold Cross-Validation Setup
folds <- createFolds(final_dataset$activity, k = 10, list = TRUE)
auc_xgb <- numeric(length(folds))

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
  pred_probs <- predict(xgb_model, newdata = test_data)
  
  # Compute AUC using ROCR package
  pred_obj <- prediction(pred_probs, test_label)
  auc_obj <- performance(pred_obj, measure = "auc")
  auc_xgb[i] <- as.numeric(auc_obj@y.values)
}

# Average AUC over the 10 folds
mean_auc_xgb <- mean(auc_xgb)

mean_auc_xgb

# autoML -------------------------------------------------------------

# Title: AutoML Classification with h2o

# download the latest Java SE JDK
# https://www.oracle.com/java/technologies/downloads/#jdk21-windows

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

# Extract the best model
best_model <- automl_models@leader

# Extract AUC of the best model on cross-validation
auc_automl <- h2o.auc(h2o.performance(best_model, xval = TRUE))

# Shut down the h2o cluster
h2o.shutdown(prompt = FALSE)



