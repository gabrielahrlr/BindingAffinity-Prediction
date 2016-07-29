#!/usr/bin/env Rscript
###############################################################################################
#    Framework I :
#     @description: Assess the combination of different either SFs or descriptors, separately, 
#     using LASSO, Elastic Net, GAM and KRLS models.
#     Provide tools to detect the possible subset of most significant variables, 
#     analyze the independent interaction of each descriptor and SF with 
#     the protein- ligand binding affinity from different perspectives.
#
#     @details:
#     @example: Rscript --vanilla fw1.R -a set1_sfs_.csv  -b set2_sfs.csv -c set2_sfs.cs
#
#     @author: Gabriela Hernández 
#     EM-DMKM  2014-2016
#     BSC-CNS  2016
###############################################################################################
########################################################################################
# ARGUMENTS INPUT  PROCESSING
# This section performs input processing for the arguments that have to be passed and
# should NOT be modified
# ######################################################################################
library(optparse)

option_list = list(
  make_option(c("-a", "--train"), type = "character", default = NULL,
              help = "Training dataset (SFs or Descriptors) REQUIRED", metavar = "csv file"),
  make_option(c("-b", "--val"), type = "character", default = NULL, 
              help = "Validation dataset (SFs or Descriptors) OPTIONAL", metavar = "csv file"),
  make_option(c("-c", "--newdata"), type = "character", default = NULL, 
              help = "New Data dataset (SFs or Descriptors) OPTIONAL", metavar = "csv file")
);

opt_parser = OptionParser(option_list = option_list);
opt = parse_args(opt_parser);

# SFs or descriptors training datasets are mandatory to be passed!
if (is.null(opt$train)){
  print_help(opt_parser)
  stop("At least the SFs or Descriptors training datasets must be supplied", call. = FALSE)
}

########################################################################################
# READING DATASETS 
# In this section, the datasets provided are read and processed.
# If not validation set is provided, then stratified sampling on the training set is
# performed, using 70% for training and 30% for validation.
########################################################################################
train <- read.csv(opt$train, header = TRUE)
# Length of training
ntrain <- length(train)
# Length of training -1 : For excluding the response variable
train_n <- ntrain-1
complex_names <- train[1]
# Predictors and Response variables separation for training
xtrain <- train[, 2:train_n]
ytrain <- train[ntrain]


if (is.null(opt$val)){
  print("No validation dataset was provided, stratified sampling will be performed
        on the training dataset")
  library(caret)
  xtrain_fool <- xtrain
  ytrain_fool <- ytrain
  set.seed(1989)
  partition <- createDataPartition(ytrain[,1], times = 1, groups = 20, p = 0.7, list = TRUE)
  idx <- partition$Resample
  xtrain <- xtrain_fool[idx,]
  xtrain <- as.data.frame(xtrain)
  ytrain <- ytrain_fool[idx,]
  ytrain <- as.data.frame(ytrain)
  xval <- xtrain_fool[-idx,]
  xval <- as.data.frame(xval)
  yval <- ytrain_fool[-idx,]
  yval <- as.data.frame(yval)
} else{
  val <- read.csv(opt$val, header = TRUE)
  complex_val <- val[1]
  xval <- val[,2:train_n]
  yval <- val[ntrain]
}

if(is.null(opt$newdata)){
  print("NO new data is provided")
}else{
  newdata <- read.csv(opt$newdata, header = TRUE)
  n_new <- length(newdata)
  xnew <- newdata[, 2:n_new]
  xnew.z <- scale(xnew)
  xnew.z <- as.data.frame(xnew.z)
}

########################################################################################
# Pre-process
# Standardization (Z-score) of all the predictors is performed, centering all the 
# variables to zero with standard deviation of 1.
########################################################################################
library(caret)
# Training
xtrain.z <- scale(xtrain)
xtrain.z <- as.data.frame(xtrain.z)
ytrain <- as.data.frame(ytrain)
colnames(ytrain)[1] <- "Experimental"

# Validation
xval.z <- scale(xval)
xval.z <- as.data.frame(xval.z)
yval <- as.data.frame(yval)
colnames(yval)[1] <- "Experimental"


########################################################################################
# MODELS Computation
# Compute the models LASSO, Elastic Net, GAM and KRLS for descriptors and if
# demanded for SFs as well
########################################################################################

# Load Functions for models
source('models.R')
print(paste("Learning Models..."))
# LASSO
lasso_model <- lasso(x = xtrain.z, y = ytrain)
lasso_error <- sqrt(mean(lasso_model$lasso_cv$cvm))
# Elastic Net
eNet_model <- eNet(xtrain.z, ytrain)
eNet_error <- sqrt(mean(eNet_model$eNet_cv$cvm))
# GAM
gam_model <- gam_m(xtrain.z, ytrain)
gam_error <- sqrt(mean(gam_model$gam_model$gcv.ubre))
# KRLS
krls_model <- krls_m(xtrain.z, ytrain)
krls_error <- predict(krls_model$krls_model, xtrain.z)
krls_error <- evaluation(krls_error$fit, ytrain)
krls_error <- round(krls_error$RMSE, 3)


########################################################################################
# Predictions in the validation dataset (SFs or Descriptors) and if it is provided in
# the Newdata dataset
########################################################################################

# Validation Set
print(paste("Evaluation of the models..."))
val_lasso <- predict(lasso_model$lasso_model,  newx = as.matrix(xval.z))
val_eNet <- predict(eNet_model$eNet_model, newx = as.matrix(xval.z))
val_gam <- predict(gam_model$gam_model, newdata = xval.z)
val_krls <- predict(krls_model$krls_model, newdata = as.matrix(xval.z))

if(!is.null(opt$newdata)){
  # If xnew is not null, then prediction of the new dataset is performed, 
  # output will be stored as newdata_predictions.csv into the same folder
  print(paste("Prediction of the New Dataset, output will be stored as newdata_predictions.csv into the same folder!"))
  new_lasso <- predict(lasso_model$lasso_model, newx = as.matrix(xnew.z))
  new_eNet <- predict(eNet_model$eNet_model, newx = as.matrix(xnew.z))
  new_gam <- predict(gam_model$gam_model, newdata = xnew.z)
  new_krls <- predict(krls_model$krls_model, newdata = as.matrix(xnew.z))
  df_pred <- data.frame(new_lasso[,1], new_eNet[,1], new_gam, new_krls$fit)
  colnames(df_pred) <- c("LASSO", "Elastic_Net", "GAM", "KRLS")
  write.csv(df_pred, "predictions.csv")
}


eval_lasso <- evaluation(val_lasso[,1], yval)
eval_eNet <- evaluation(val_eNet[,1], yval)
eval_gam <- evaluation(val_gam, yval)
eval_krls <- evaluation(val_krls$fit, yval)




# Features Selected by LASSO and Elastic Net with beta Coefficients
lasso_vars <- lasso_model$lasso_model$beta[which(lasso_model$lasso_model$beta!= 0),]
eNet_vars <- eNet_model$eNet_model$beta[which(eNet_model$eNet_model$beta!= 0),]


# Print the evaluations
print(paste("LASSO Training Error (RMSE):"))
print(round(lasso_error, 3))
print(paste("Evaluation Validation set LASSO (Rp):"))
print(round(eval_lasso$Rp[1], 3))
print(paste("Evaluation LASSO (RMSE):"))
print(round(eval_lasso$RMSE, 3))
print(paste("Elastic Net Training Error (RMSE):"))
print(round(eNet_error, 3))
print(paste("Evaluation Elastic Net (Rp):"))
print(round(eval_eNet$Rp[1], 3))
print(paste("Evaluation Elastic Net (RMSE):"))
print(round(eval_eNet$RMSE, 3))
print(paste("GAM Training Error (RMSE):"))
print(round(gam_error, 3))
print(paste("Evaluation GAM (Rp):"))
print(round(eval_gam$Rp[1],3))
print(paste("Evaluation GAM (RMSE):"))
print(round(eval_gam$RMSE, 3))
print(paste("KRLS Training Error (RMSE):"))
print(round(krls_error, 3))
print(paste("Evaluation KRLS (Rp):"))
print(round(eval_krls$Rp[1], 3))
print(paste("Evaluation KRLS (RMSE):"))
print(round(eval_krls$RMSE, 3))
print(paste("Features Selected by LASSO:"))
print(lasso_vars)
print(paste("Features Selected by Elastic Net:"))
print(eNet_vars)


# Report the plots of all the models in PDF
pdf("Models_Plots.pdf")
par(mfrow=c(2,2))
plot(lasso_model$lasso_cv$glmnet.fit, "lambda", label=TRUE, main = "LASSO 10-Fold CV Lambda")
plot(lasso_model$lasso_cv, main = "LASSO shrinkage of coefficients")
par(mfrow=c(2,2))
plot(eNet_model$eNet_cv$glmnet.fit, "lambda", label=TRUE)
plot(eNet_model$eNet_cv)
par(mfrow=c(2,2))
plot(gam_model$gam_model, scheme = 1,residuals=TRUE, all.terms = TRUE)
par(mfrow=c(2,2))
plot(krls_model$krls_model)
dev.off()

