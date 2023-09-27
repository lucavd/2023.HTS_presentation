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



