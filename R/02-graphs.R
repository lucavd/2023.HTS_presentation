source('R/01-data_generation.R')

# graphs ------------------------------------------------------------------

# CORRELATED REPLICATES

# Randomly select 10 compounds
random_compound_ids <- sample(unique(final_dataset$compound_id), 10)

# Filter the dataset to include only the selected compounds
subset_dataset <- final_dataset %>% 
  filter(compound_id %in% random_compound_ids)

# Melt the dataset from wide to long format for ggplot2 compatibility
melted_dataset <- subset_dataset %>%
  gather(key = "measure_type", value = "value", measure1:measure5)

# Generate scatter plot to assess correlation between replicates for selected compounds
ggplot(melted_dataset, aes(x = replicate, y = value)) +
  geom_point(aes(color = as.factor(compound_id), shape = measure_type)) +
  facet_wrap(~ compound_id) +
  labs(title = "Correlation of Measurements Across Replicates for Randomly Selected Compounds",
       x = "Replicate",
       y = "Measure Value",
       color = "Compound ID",
       shape = "Measure Type") +
  theme_minimal()

# ACTIVITY = 1 measures are higher than ACTIVITY = 0

# Melt the dataset from wide to long format for ggplot2 compatibility
melted_dataset_a <- dataset %>%
  gather(key = "measure_type", value = "value", measure1:measure5)

# Generate boxplot to visualize the distribution of measures by activity level
ggplot(melted_dataset_a, aes(x = measure_type, y = value, fill = as.factor(activity))) +
  geom_boxplot(alpha = 0.7, position = position_dodge(width = 0.75)) +
  labs(title = "Distribution of Measures by Activity Level",
       x = "Measure Type",
       y = "Value",
       fill = "Activity Level") +
  theme_minimal()


# ALL MEASURES DISTRIBUTION

# Melt the final dataset from wide to long format for ggplot2 compatibility
melted_final_dataset <- final_dataset %>%
  gather(key = "measure_type", value = "value", measure1:measure5)

# Generate violin plot to visualize the distribution of the five measurements
ggplot(melted_final_dataset, aes(x = measure_type, y = value)) +
  geom_violin(aes(fill = measure_type), alpha = 0.5) +
  geom_boxplot(width = 0.2) +
  labs(title = "Distribution of the Five Measures Across All Compounds and Replicates",
       x = "Measure Type",
       y = "Value",
       fill = "Measure Type") +
  theme_minimal()

# Generate density plot to visualize the distribution of the five measurements
ggplot(melted_final_dataset, aes(x = value, fill = measure_type)) +
  geom_density(alpha = 0.7) +
  facet_wrap(~ measure_type, scales = "free") +
  labs(title = "Density Plot of the Five Measures Across All Compounds and Replicates",
       x = "Value",
       y = "Density",
       fill = "Measure Type") +
  theme_minimal()



# ROC GRAPH Logistic regression -------------------------------------------

library(ROCR)
library(ggplot2)
library(dplyr)

# Preallocate list to store ROC values
list_roc_values <- list()

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
  
  # Store TPR and FPR
  list_roc_values[[i]] <- data.frame(TPR = unlist(perf_obj@y.values), FPR = unlist(perf_obj@x.values))
}

# Combine all TPR and FPR
all_roc_values <- do.call(rbind, list_roc_values)

# Compute mean and 95% CI
mean_roc <- all_roc_values %>% group_by(FPR) %>% summarise(Mean_TPR = mean(TPR), .groups = 'drop')
ci_roc <- all_roc_values %>% group_by(FPR) %>% summarise(CI_low = quantile(TPR, 0.025), CI_high = quantile(TPR, 0.975), .groups = 'drop')

# Plotting
ggplot(data = mean_roc, aes(x = FPR, y = mean_roc$Mean_TPR)) +
  geom_line(color = "blue") +
  geom_ribbon(data = ci_roc, aes(ymin = CI_low, ymax = CI_high), alpha = 0.2) +
  labs(title = "Mean ROC Curve with 95% CI",
       x = "False Positive Rate",
       y = "True Positive Rate")
