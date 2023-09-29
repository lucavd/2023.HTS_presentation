# Load required libraries
library(Boruta)
library(caret)
library(ROCR)

# Initialize an array to store AUC values
auc_values <- numeric()

# Apply Boruta for variable selection
boruta_output <- Boruta(activity ~ ., data = final_dataset, doTrace = 0)
final_vars <- getSelectedAttributes(boruta_output, withTentative = TRUE)
final_dataset_filtered <- final_dataset[, c(final_vars, "activity")]

# 5-fold Cross Validation
folds <- createFolds(final_dataset_filtered$activity, k = 5)

# Loop through each fold
for(i in seq_along(folds)){
  train_data <- final_dataset_filtered[-folds[[i]],]
  test_data <- final_dataset_filtered[folds[[i]],]
  
  # Train logistic regression model using selected variables
  control_lr <- trainControl(method = "none", classProbs = TRUE, summaryFunction = twoClassSummary)
  logistic_model <- train(activity ~ ., data = train_data, method = "glm", family = "binomial", trControl = control_lr, metric = "ROC")
  
  # Predict on test data
  predictions <- predict(logistic_model, newdata = test_data, type = "prob")[,2]
  
  # Calculate AUC
  pred_obj <- prediction(predictions, test_data$activity)
  auc_obj <- performance(pred_obj, measure = "auc")
  auc_values[i] <- as.numeric(auc_obj@y.values)
}

# Calculate mean and confidence intervals of AUC
mean_auc <- mean(auc_values)
ci_lower <- quantile(auc_values, 0.025)
ci_upper <- quantile(auc_values, 0.975)
