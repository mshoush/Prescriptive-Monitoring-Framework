# https://github.com/lihualei71/cfcausal
# https://lihualei71.github.io/cfcausal/reference/conformalIte.html
library("cfcausal")

options(warn=-1)
args <- commandArgs(trailingOnly = TRUE)
print(args)

filepath <- sprintf("results/causal/%s/train_df_CATE_%s.csv", args, args)
print(filepath)


print("read train data...")
train <- read.csv(sprintf("results/causal/%s/train_df_CATE_%s.csv", args, args), sep = ";")
#print(colnames(train))
#print(train$event)
tryCatch({
	train$event <- ifelse(train$event, 1, 0)
	train <- as.data.frame(lapply(train, as.numeric))
}, error = function(e) {
})
#train$event <- ifelse(train$event, 1, 0)
#print(sapply(train, class))
#train <- as.data.frame(lapply(train, as.numeric))
#print(train$event)

T_train <- train$Treatment

Y1_train <- train$Proba_if_Treated

Y0_train <- train$Proba_if_Untreated

Y_train <- ifelse(T_train == 1, Y1_train, Y0_train)

X_train <- train[, !(names(train) %in% c("Proba_if_Untreated", "Proba_if_Tre
    ated", "CATE", "Outcome"))]


###########################
print("read test data...")
test <- read.csv(sprintf("results/causal/%s/test_df_CATE_%s.csv", args, args), sep = ";")

tryCatch({
	test$event <- ifelse(test$event, 1, 0)
	test <- as.data.frame(lapply(test, as.numeric))
}, error = function(e) {
})


#test$event <- ifelse(test$event, 1, 0)
#test <- as.data.frame(lapply(test, as.numeric))
T_test <- test$Treatment

Y1_test <- test$Proba_if_Treated

Y0_test <- test$Proba_if_Untreated

Y_test <- ifelse(T_test == 1, Y1_test, Y0_test)

X_test <- test[, !(names(test) %in% c("Proba_if_Untreated", "Proba_if_Tre
    ated", "CATE", "Outcome"))]


######################################################################################
print("Inexact nested method...")

# Inexact nested method
CIfun <- conformalIte(X_train, Y_train, T_train,
    alpha = 0.1, algo = "nest", exact = FALSE, type = "CQR",
    quantiles = c(0.05, 0.95), outfun = "quantRF", useCV = FALSE
)

Inexact_nested_method <- CIfun(X_test)


######################################################################################

# Exact nested method
print("Exact nested method...")

CIfun <- conformalIte(X_train, Y_train, T_train, alpha = 0.1, algo = "nest", exact = TRUE, type = "CQR", quantiles = c(0.05, 0.95), outfun = "quantRF", useCV = FALSE)

Exact_nested_method <- CIfun(X_test)


######################################################################################
print("Naive method...")

# naive method
CIfun <- conformalIte(X_train, Y_train, T_train,
    alpha = 0.1, algo = "naive", type = "CQR",
    quantiles = c(0.05, 0.95), outfun = "quantRF", useCV = FALSE
)

naive_method <- CIfun(X_test)


######################################################################################
print("counterfactual method...")

# counterfactual method, Y and T needs to be observed

CIfun <- conformalIte(X_train, Y_train, T_train, alpha = 0.1, algo = "counterfactual", type = "CQR", quantiles = c(0.05, 0.95), outfun = "quantRF", useCV = FALSE)


counterfactual_method <- CIfun(X_test, Y_test, T_test)

#################################################################################
print("Save results...")
test[c("lower_counterfactual", "upper_counterfactual")] <- counterfactual_method[c("lower", "upper")]

test[c("lower_inexact", "upper_inexact")] <- Inexact_nested_method[c("lower","upper")]


test[c("lower_exact", "upper_exact")] <- Exact_nested_method[c("lower", "upper")]


test[c("lower_naive", "upper_naive")] <- naive_method[c("lower", "upper")]

# write.csv2(test, file.path("./results/conformal_causal/bpic2012/", paste0("t
#     est_", "bpic2012", ".csv")), sep = ";", row.names = FALSE)
# Save the data frame to CSV
write.csv(test, file = "./results/test2.csv", row.names = FALSE)


# Define the file path
res_filename <- sprintf("./results/conformal_causal/%s/",args)

# Create results directory
if (!dir.exists(res_filename)) {
	  dir.create(res_filename, recursive = TRUE)
}

filename <- sprintf("test2_conformalizedTE_%s.csv", args)
filepath <- file.path(res_filename, filename)

# Save the data frame to CSV
write.csv(test, file = filepath, row.names = FALSE)



#write.csv(test,file='./results/conformal_causal/bpic2012/test2.csv', row.names=FALSE)
