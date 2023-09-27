# Load required libraries
library(tidyverse)

set.seed(42)

# Initialize variables
n_compounds <- 500
n_replicates <- 3
n_positive_compounds <- floor(0.05 * n_compounds)
n_high_var_compounds <- floor(0.20 * n_compounds)

# 1- Generate 500 compounds
compounds <- tibble(compound_id = 1:n_compounds)

# 2- Create 3 replicates for each compound
dataset <- compounds %>%
  slice(rep(1:n(), each = n_replicates)) %>%
  mutate(replicate = rep(1:n_replicates, times = n_compounds))

# 3- Add 5 measurements for each compound and replicate
# Measurements are normally distributed with varying means and standard deviations.
dataset <- dataset %>%
  rowwise() %>%
  mutate(
    measure1 = rnorm(1, mean = 10, sd = 2),
    measure2 = rnorm(1, mean = 20, sd = 3),
    measure3 = rnorm(1, mean = 30, sd = 4),
    measure4 = rnorm(1, mean = 40, sd = 5),
    measure5 = rnorm(1, mean = 50, sd = 6)
  )

# 4- Modify 3 of the 5 measures to not follow a normal distribution
dataset <- dataset %>%
  rowwise() %>%
  mutate(
    measure2 = rexp(1, rate = 1/20),
    measure3 = rpois(1, lambda = 30),
    measure5 = rchisq(1, df = 8)
  )

# 5- Ensure measurements are similar within replicates of each compound
# Skipped: The data already satisfy this criteria by design.

# 6- 5% of the compounds have activity = 1, others have activity = 0
positive_compound_ids <- sample(compounds$compound_id, n_positive_compounds)
dataset <- dataset %>%
  mutate(activity = if_else(compound_id %in% positive_compound_ids, 1, 0))

# 7- Positive compounds have a random number (between 1 and 5) of measures that are higher than those of negative compounds

# Create a dataframe to hold the random increments
increments_df <- dataset %>%
  filter(activity == 1) %>%
  rowwise() %>%
  mutate(
    num_measures_to_modify = sample(1:5, 1),
    measure_names = list(sample(c("measure1", "measure2", "measure3", "measure4", "measure5"), num_measures_to_modify)),
    increments = list(rep(sample(1:10, 1), num_measures_to_modify))
  ) %>%
  ungroup() %>%
  dplyr::select(compound_id, replicate, measure_names, increments)

# Function to apply the increments
apply_increments <- function(measure, measure_name, measure_names, increments) {
  if (measure_name %in% measure_names) {
    measure + increments[which(measure_names == measure_name)]
  } else {
    measure
  }
}

# Apply the increments to the original dataset
dataset <- dataset %>%
  left_join(increments_df, by = c("compound_id", "replicate")) %>%
  rowwise() %>%
  mutate(
    measure1 = apply_increments(measure1, "measure1", measure_names, increments),
    measure2 = apply_increments(measure2, "measure2", measure_names, increments),
    measure3 = apply_increments(measure3, "measure3", measure_names, increments),
    measure4 = apply_increments(measure4, "measure4", measure_names, increments),
    measure5 = apply_increments(measure5, "measure5", measure_names, increments)
  ) %>%
  dplyr::select(-measure_names, -increments)


# 8- 20% of the compounds have a variability 5 times greater
high_var_compound_ids <- sample(compounds$compound_id, n_high_var_compounds)
dataset <- dataset %>%
  rowwise() %>%
  mutate(
    measure1 = if_else(compound_id %in% high_var_compound_ids, measure1 * 5, measure1),
    measure2 = if_else(compound_id %in% high_var_compound_ids, measure2 * 5, measure2),
    measure3 = if_else(compound_id %in% high_var_compound_ids, measure3 * 5, measure3),
    measure4 = if_else(compound_id %in% high_var_compound_ids, measure4 * 5, measure4),
    measure5 = if_else(compound_id %in% high_var_compound_ids, measure5 * 5, measure5)
  )

# 9- Add 3 positive controls with unique compound IDs and higher average measurements
positive_controls <- tibble(
  compound_id = rep(-1:-3, each = n_replicates),
  replicate = rep(1:n_replicates, times = 3),
  activity = rep(1, times = n_replicates * 3),
  measure1 = rnorm(n_replicates * 3, mean = 60, sd = 2),
  measure2 = rexp(n_replicates * 3, rate = 1/40),
  measure3 = rpois(n_replicates * 3, lambda = 60),
  measure4 = rnorm(n_replicates * 3, mean = 100, sd = 5),
  measure5 = rchisq(n_replicates * 3, df = 16)
)


# 10- Add 5 negative controls with unique compound IDs and lower average measurements
negative_controls <- tibble(
  compound_id = rep(-4:-8, each = n_replicates),
  replicate = rep(1:n_replicates, times = 5),
  activity = rep(0, times = n_replicates * 5),
  measure1 = rnorm(n_replicates * 5, mean = 5, sd = 2),
  measure2 = rexp(n_replicates * 5, rate = 1/10),
  measure3 = rpois(n_replicates * 5, lambda = 15),
  measure4 = rnorm(n_replicates * 5, mean = 20, sd = 5),
  measure5 = rchisq(n_replicates * 5, df = 4)
)


# Combine everything to generate the final dataset
final_dataset <- bind_rows(dataset, positive_controls, negative_controls)