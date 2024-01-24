library("cfcausal")

# Function to read and preprocess data
read_and_preprocess_data <- function(args, data_type) {
  file_path <- sprintf("./prepared_data/%s/%s_encoded_%s.csv", args, data_type, args)
  causal_path <- sprintf("./results/causal/%s/causalLift_%s_df_CATE_%s_Treatment1.csv", args, data_type, args)

  #tlearner_path <- sprintf("./results/causal/%s/TLearner_%s.csv", args, data_type)
  #cate_df_path <- sprintf("./results/causal/%s/cate_df_results_%s_bpic2017.csv", args, data_type)

  # Read data files
  encoded <- sapply(read.csv(file_path, sep = ";"), as.numeric)
  causal <- read.csv(causal_path, sep = ";")
  #cate_df <- read.csv(cate_df_path, sep = ";")

  # Combine data frames
  data <- cbind(encoded, causal)

  # Print head and shape of the data
  #cat("Head of", data_type, "data:\n")
  #print(head(data))
  cat("\nShape of", data_type, "data:", dim(data), "\n\n")

  # Handle missing values if needed
  data[is.na(data)] <- 0  # Replace with appropriate handling

  return(data)
}

# Function to preprocess data for a specific treatment
preprocess_data_for_treatment <- function(data, treatment_suffix) {
  #cat("tsufix", treatment_suffix)
  col_prefix <- paste0(c("Treatment1", "Proba_if_Treated", "Proba_if_Untreated"))
  T <- as.numeric(data[[col_prefix[1]]])
  Y1 <- as.numeric(data[[col_prefix[2]]])
  Y0 <- as.numeric(data[[col_prefix[3]]])
  Y <- ifelse(T == 1, Y1, Y0)
  X <- data[, !(names(data) %in% c(col_prefix, "Outcome"))]

  return(list(T = T, Y = Y, X = X))
}

# Function to calculate CATE for a specific method
calculate_CATE <- function(X, Y, T, alpha) {
  cat("Alpha...", alpha, "\n")
  CIfun <- conformalIte(
    X, Y, T,
    alpha = alpha, algo = "counterfactual", type = "CQR", quantiles = c(0.05, 0.95), outfun = "quantRF", useCV = FALSE
  )
  return(CIfun)
}

# Function to calculate and save CATE for a specific treatment
calculate_and_save_cate <- function(train_data, test_data, valid_data, args, treatment_suffix, alpha_values) {
  #cat("treatment_suffix...", treatment_suffix)
  train_data_treatment <- preprocess_data_for_treatment(train_data, treatment_suffix)
  valid_data_treatment <- preprocess_data_for_treatment(valid_data, treatment_suffix)
  test_data_treatment <- preprocess_data_for_treatment(test_data, treatment_suffix)

  # Calculate and save CATE for each alpha value
  for (alpha in alpha_values) {
    CIfun <- calculate_CATE(train_data_treatment$X, train_data_treatment$Y, train_data_treatment$T, alpha)

    # Calculate CATE for validation and test sets
    methods <- CIfun(valid_data_treatment$X, valid_data_treatment$Y, valid_data_treatment$T)
    valid_data[paste0("lower_", "counterfactual", "_", treatment_suffix, "_alpha_", alpha)] <- methods$lower
    valid_data[paste0("upper_", "counterfactual", "_", treatment_suffix, "_alpha_", alpha)] <- methods$upper

    methods <- CIfun(test_data_treatment$X, test_data_treatment$Y, test_data_treatment$T)
    test_data[paste0("lower_", "counterfactual", "_", treatment_suffix, "_alpha_", alpha)] <- methods$lower
    test_data[paste0("upper_", "counterfactual", "_", treatment_suffix, "_alpha_", alpha)] <- methods$upper
  }

  # Print head and shape of the final data for validation set
  cat("Head of final validation data:\n")
  print(head(valid_data))
  cat("\nShape of final validation data:", dim(valid_data), "\n\n")

  # Print head and shape of the final data for test set
  cat("Head of final test data:\n")
  print(head(test_data))
  cat("\nShape of final test data:", dim(test_data), "\n\n")

  # Define the file path for saving the results for validation set
  res_filename_valid <- sprintf("./results/conformal_causal/%s/", args)
  filename_valid <- sprintf("conformalizedTE_%s_%s_valid.csv", args, treatment_suffix)
  filepath_valid <- file.path(res_filename_valid, filename_valid)

  # Create results directory for validation set
  if (!dir.exists(res_filename_valid)) {
    dir.create(res_filename_valid, recursive = TRUE)
  }

  # Save the data frame to CSV for validation set
  write.csv(valid_data, file = filepath_valid, row.names = FALSE)

  # Define the file path for saving the results for test set
  res_filename_test <- sprintf("./results/conformal_causal/%s/", args)
  filename_test <- sprintf("conformalizedTE_%s_%s_test.csv", args, treatment_suffix)
  filepath_test <- file.path(res_filename_test, filename_test)

  # Create results directory for test set
  if (!dir.exists(res_filename_test)) {
    dir.create(res_filename_test, recursive = TRUE)
  }

  # Save the data frame to CSV for test set
  write.csv(test_data, file = filepath_test, row.names = FALSE)
}

# Main script
# args <- "bpic2017"  # Replace with your actual dataset name
#args <- "bpic2017"  # Replace with your actual dataset name
options(warn=-1)
args <- commandArgs(trailingOnly = TRUE)
print(args)
valid_data <- read_and_preprocess_data(args, "valid")
train_data <- read_and_preprocess_data(args, "train")
test_data <- read_and_preprocess_data(args, "test")

# Loop through treatment_suffix values
#for (treatment_suffix in c("1", "2", "3", "4")) {
for (treatment_suffix in c("1")) {
  # Define alpha values range
  cat("treatment_suffix", treatment_suffix, "\n")
  alpha_values <- seq(0.1, 1.0, by = 0.1)

  # Calculate and save CATE for each treatment and alpha value
  calculate_and_save_cate(train_data, test_data, valid_data, args, treatment_suffix, alpha_values)
}
