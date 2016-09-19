#!/usr/bin/env Rscript
########################################################################################
# STACKING FRAMEWORK
# @author: Gabriela HERNANDEZ LARIOS
# @description: 
# @example with
# @example with validation dataset: Rscript --vanilla stacking.R -a set1_sfs.csv -b set2_sfs.csv 
#           -d set1_all_descriptors.csv -e set2_all_descriptors.csv 
#
# ######################################################################################
library(caret)
library(hydroGOF)
library(optparse)

.thisfile_rscript <- function() {
  # Get the file path of this script.
  #
  # This functions must be called only inside this script file, and assumes
  # that, in case of a command-line script, it is the top-level script.
  #
  # Returns:
  #   If this script was called as a command-line script, it returns its file path.
  #   Otherwise, it returns NULL.
  #
  # This code was copied from http://stackoverflow.com/a/36075028
  cmdArgs <- commandArgs(trailingOnly = FALSE)
  cmdArgsTrailing <- commandArgs(trailingOnly = TRUE)
  cmdArgs <- cmdArgs[seq.int(from=1, length.out=length(cmdArgs) - length(cmdArgsTrailing))]
  res <- gsub("^(?:--file=(.*)|.*)$", "\\1", cmdArgs)

  # If multiple --file arguments are given, R uses the last one
  res <- tail(res[res != ""], 1)
  if (length(res) > 0)
    return (res)

  NULL
}

.dummyGetThisDirname <- function() {
  # Get the path of the directory of this script.
  #
  # This functions must be called only inside this script file, and assumes
  # that, in case of a command-line script, it is the top-level script.
  #
  # Returns:
  #   The path of the directory of this script.
  thisDir <- getSrcDirectory(.dummyGetThisDirname)

  if (length(thisDir) == 0) {
    return (dirname(.thisfile_rscript()))
  } else {
    return (thisDir)
  }
}

# Load Functions for models
source(file.path(.dummyGetThisDirname(), 'models.R'))

main <- function(opt)
{
  # SFs and descriptors training datasets are mandatory to be passed!
  if (is.null(opt$sfs) | is.null(opt$des)){
    print_help(opt_parser)
    stop("At least the SFs and Descriptors training datasets must be supplied", call. = FALSE)
  }
  
  ########################################################################################
  # READING DATASETS 
  # In this section, the datasets provided are read and processed.
  # If not validation sets are provided, then stratified sampling on the training sets is
  # performed, using 70% for training and 30% for validation.
  ########################################################################################
  sfs <- read.csv(opt$sfs, header = TRUE)
  des <- read.csv(opt$des, header = TRUE)
  # Length of SFs
  nsfs <- length(sfs)
  # Length of SFs -1
  sfs_n <- nsfs-1
  complex_names <- sfs[1]
  # Predictors and Response variables separation for SFs
  xsfs <- sfs[,2:sfs_n]
  ysfs <- sfs[nsfs]
  # Length of descriptors
  ndes <- length(des)
  des_n <- ndes-1
  # Predictors and Response variables separation for descriptors
  xdes <- des[,2:des_n]
  ydes <- des[ndes]
  
  if (is.null(opt$sfs_val) | is.null(opt$des_val)){
    print("No validation datasets were provided, stratified sampling will be performed
          on the SFs and Descriptors training datasets")
    xsfs_fool <- xsfs
    ysfs_fool <- ysfs
    xdes_fool <- xdes
    ydes_fool <- ydes
    partition <- createDataPartition(ysfs[,1], times = 1, groups = 20, p = 0.7, list = TRUE)
    idx <- partition$Resample
    xsfs <- xsfs_fool[idx,]
    xsfs <- as.data.frame(xsfs)
    ysfs <- ysfs_fool[idx,]
    ysfs <- as.data.frame(ysfs)
    xdes <- xdes_fool[idx,]
    xdes <- as.data.frame(xdes)
    ydes <- ydes_fool[idx,]
    ydes <- as.data.frame(ydes)
    xsfs_val <- xsfs_fool[-idx,]
    xsfs_val <- as.data.frame(xsfs_val)
    ysfs_val <- ysfs_fool[-idx,]
    ysfs_val <- as.data.frame(ysfs_val)
    xdes_val <- xdes_fool[-idx,]
    xdes_val <- as.data.frame(xdes_val)
    ydes_val <- ydes_fool[-idx,]
    ydes_val <- as.data.frame(ydes_val)
  }
  else{
    sfs_val <- read.csv(opt$sfs_val, header = TRUE)
    des_val <- read.csv(opt$des_val, header = TRUE)
    complex_val <- sfs_val[1]
    xsfs_val <- sfs_val[,2:sfs_n]
    ysfs_val <- sfs_val[nsfs]
    xdes_val <- des_val[,2:des_n]
    ydes_val <- des_val[ndes]
  }

  if(is.null(opt$sfs_new) | is.null(opt$des_new)){
    print("NO new data is provided")
  }
  else{
    # Read the new data, if the option -g or --new_data is used the program will
    # use the last column in the new_data file as the experimental values for this
    # and compute the statistics.
    if (is.null(opt$new_data)){
      # Scoring Functions 
      sfs_new <- read.csv(opt$sfs_new, header = TRUE)
      nsfs_new <- length(sfs_new)
      xsfs_new <- sfs_new[, 2:nsfs_new]
      
      # Descriptors
      des_new <- read.csv(opt$des_new, header = TRUE)
      ndes_new <- length(des_new)
      xdes_new <- des_new[, 2:ndes_new]
    }
    else{
      # Scoring Functions 
      sfs_new <- read.csv(opt$sfs_new, header = TRUE)
      nsfs_new_all <- length(sfs_new)
      nsfs_new <- nsfs_new_all - 1
      xsfs_new <- sfs_new[, 2:nsfs_new]
      ysfs_new <- sfs_new[nsfs_new_all]

      # Descriptors
      des_new <- read.csv(opt$des_new, header = TRUE)
      ndes_new_all <- length(des_new)
      ndes_new <- ndes_new_all - 1
      xdes_new <- des_new[, 2:ndes_new]
      ydes_new <- des_new[ndes_new_all]
      ydes_new <- as.data.frame(ydes_new)
    }
  }
  
  ########################################################################################
  # Pre-process Phase
  # Standardization (Z-score) of all the predictors is performed, centering all the 
  # variables to zero with standard deviation on 1. Use N as denominator for the standard
  # deviation
  ########################################################################################
  # SFs
  sfs_standardization_center <- apply(xsfs, 2, mean)
  sfs_xtrain_variance <- apply(xsfs, 2, var)
  sfs_xtrain_datapoints <- nrow(xsfs)
  sfs_standardization_scale <- sqrt((sfs_xtrain_datapoints-1)/sfs_xtrain_datapoints)*sqrt(sfs_xtrain_variance)
  xsfs.z <- scale(xsfs, center=sfs_standardization_center, scale=sfs_standardization_scale)
  xsfs.z <- as.data.frame(xsfs.z)
  xsfs_val.z <- scale(xsfs_val, center=sfs_standardization_center, scale=sfs_standardization_scale)
  xsfs_val.z <- as.data.frame(xsfs_val.z)
  ysfs <- as.data.frame(ysfs)
  ysfs_val <- as.data.frame(ysfs_val)

  if(!is.null(opt$sfs_new) & !is.null(opt$des_new)) {
    xsfs_new.z <- scale(xsfs_new, center=sfs_standardization_center, scale=sfs_standardization_scale)
    xsfs_new.z <- as.data.frame(xsfs_new.z)
  }
  
  colnames(ysfs)[1] <- "Experimental"
  colnames(ysfs_val)[1] <- "Experimental"
  if (!is.null(opt$new_data)) {
    colnames(ysfs_new)[1] <- "Experimental"
  }
  
  # Descriptors
  des_standardization_center <- apply(xdes, 2, mean)
  des_xtrain_variance <- apply(xdes, 2, var)
  des_xtrain_datapoints <- nrow(xdes)
  des_standardization_scale <- sqrt((des_xtrain_datapoints-1)/des_xtrain_datapoints)*sqrt(des_xtrain_variance)
  xdes.z <- scale(xdes, center=des_standardization_center, scale=des_standardization_scale)
  xdes.z <- as.data.frame(xdes.z)
  xdes_val.z <- scale(xdes_val, center=des_standardization_center, scale=des_standardization_scale)
  xdes_val.z <- as.data.frame(xdes_val.z)

  if(!is.null(opt$sfs_new) & !is.null(opt$des_new)) {
    xdes_new.z <- scale(xdes_new, center=des_standardization_center, scale=des_standardization_scale)
    xdes_new.z <- as.data.frame(xdes_new.z)
  }
  
  colnames(ydes)[1] <- "Experimental"
  colnames(ydes_val)[1] <- "Experimental"
  if(!is.null(opt$sfs_new) & !is.null(opt$des_new) & !is.null(opt$new_data)) {
    colnames(ydes_new)[1] <- "Experimental"
  }
  
  # ALL
  xall.z <- cbind(xsfs.z, xdes.z)
  yall <- ydes
  xall_val.z <- cbind(xsfs_val.z, xdes_val.z)
  yall_val <- ydes_val
  if(!is.null(opt$sfs_new) & !is.null(opt$des_new)) {
    xall_new.z <- cbind(xsfs_new.z, xdes_new.z)    
  }
  
  
  ########################################################################################
  # MODELS Computation
  # Compute the models LASSO, Elastic Net, GAM and KRLS for descriptors and if
  # demanded for SFs as well
  ########################################################################################
  
  print(paste("Learning Models for descriptors and SFs..."))
  
  # Learning Process for Descriptors
  
  #  LASSO
  lasso_des <- lasso(x = xall.z, y = yall)
  # Uncomment the bellow  line if you want to use the error in the training 
  # for choosing the best models
  #lasso_error_des_train <- sqrt(mean(lasso_des$lasso_cv$cvm)) 
  lasso_error_pred <- predict(lasso_des$lasso_model,  newx = as.matrix(xall_val.z))
  lasso_error_des <- rmse(as.numeric(lasso_error_pred[,1]),yall_val[,1])
  
  # Elastic Net
  eNet_des <- eNet(x = xall.z, y = yall)
  # Uncomment the bellow  line if you want to use the error in the training 
  # for choosing the best models
  #eNet_error_des_train <- sqrt(mean(eNet_des$eNet_cv$cvm))
  eNet_error_pred <- predict(eNet_des$eNet_model,  newx = as.matrix(xall_val.z))
  eNet_error_des <- rmse(as.numeric(eNet_error_pred[,1]), yall_val[,1])
  
  # GAM
  gam_des <- gam_m(xall.z, yall)
  # Uncomment the bellow  line if you want to use the error in the training 
  # for choosing the best models
  # gam_error_des_train <- sqrt(mean(gam_des$gam_model$gcv.ubre))
  gam_error_pred <- predict(gam_des$gam_model, xall_val.z)
  gam_error_des <- rmse(as.numeric(gam_error_pred), yall_val[,1])
  
  
  # KRLS
  krls_des <- krls_m(xall.z, yall)
  # Uncomment the bellow  line if you want to use the error in the training 
  # for choosing the best models
  #krls_error_des_train <- (krls_des$krls_model$Looe[,1])/nrow(xall.z)
  krls_error_pred <- predict(krls_des$krls_model, xall_val.z)
  krls_error_des <- rmse(krls_error_pred$fit, yall_val)
  
  # Choose the Best Model 
  models <- c("LASSO", "ElasticNet", "GAM", "KRLS")
  performance_vec <- c(lasso_error_des, eNet_error_des, gam_error_des, krls_error_des)
  # Uncomment the bellow  line if you want to use the error in the training 
  # for choosing the best models, default is to use a validation set.
  # performance_vec <- c(lasso_error_des_train, eNet_error_des_train, gam_error_des_train, krls_error_des_train)
  best_model_all <- which.min(performance_vec)
  print("Best Model of the combination of SFs and Descriptors together:")
  print(models[best_model_all])
  
  
  # Best Model: LASSO
  if(best_model_all == 1){
    best_train_all <- predict(lasso_des$lasso_model, newx = as.matrix(xall.z))
    best_val_all <- predict(lasso_des$lasso_model, newx = as.matrix(xall_val.z))
    if(!is.null(opt$sfs_new) & !is.null(opt$des_new)){
      best_new_all <- predict(lasso_des$lasso_model, newx = as.matrix(xall_new.z))
    }
  }
  
  # Best Model: Elastic Net
  if(best_model_all == 2){
    best_train_all <- predict(eNet_des$eNet_model, newx = as.matrix(xall.z))
    best_val_all <- predict(eNet_des$eNet_model, newx = as.matrix(xall_val.z))
    if(!is.null(opt$sfs_new) & !is.null(opt$des_new)){
      best_new_all <- predict(eNet_des$eNet_model, newx = as.matrix(xall_new.z))
    }
  }
  
  # Best Model: GAM
  if(best_model_all == 3){
    best_train_all <- predict(gam_des$gam_model,  xall.z)
    best_val_all <- predict(gam_des$gam_model, xall_val.z)
    if(!is.null(opt$sfs_new) & !is.null(opt$des_new)){
      best_new_all <- predict(gam_des$gam_model, xall_new.z)
    }
  }
  
  # Best Model: KRLS
  if(best_model_all == 4){
    best_train_all <- predict(krls_des$krls_model, xall.z)
    best_train_all <- best_train_all$fit
    best_val_all <- predict(krls_des$krls_model, xall_val.z)
    best_val_all <- best_val_all$fit
    if(!is.null(opt$sfs_new) & !is.null(opt$des_new)){
      best_new_all <- predict(krls_des$krls_model, xall_new.z)
      best_new_all <- best_new_all$fit
    }
  }
  
  
  # LEARNING MODELS FOR SFs
  print("Learning Models for SFs...")
  
  # LASSO
  lasso_sfs <- lasso(x = xsfs.z, y = ysfs)
  # Uncomment the bellow  line if you want to use the error in the training 
  # for choosing the best models
  # lasso_error_sfs_train <- lasso_sfs$lasso_cv_rmse
  lasso_error_sfs_pred <- predict(lasso_sfs$lasso_model, newx = as.matrix(xsfs_val.z))
  lasso_error_sfs <- rmse(as.numeric(lasso_error_sfs_pred[,1]), ysfs_val[,1])
  
  # Elastic Net
  eNet_sfs <- eNet(x = xsfs.z, y = ysfs)
  # Uncomment the bellow  line if you want to use the error in the training 
  # for choosing the best models
  # eNet_error_sfs_train <- eNet_sfs$eNet_cv_rmse
  eNet_error_sfs_pred <- predict(eNet_sfs$eNet_model, newx = as.matrix(xsfs_val.z))
  eNet_error_sfs <- rmse(as.numeric(eNet_error_sfs_pred[,1]), ysfs_val[,1])
  
  # GAM
  gam_sfs <- gam_m(xsfs.z, ysfs)
  # Uncomment the bellow  line if you want to use the error in the training 
  # for choosing the best models
  # gam_error_sfs_train <- gam_sfs$gam_cv_rmse
  gam_error_sfs_pred <- predict(gam_sfs$gam_model,xsfs_val.z)
  gam_error_sfs <- rmse(as.numeric(gam_error_sfs_pred), ysfs_val[,1])
  
  # KRLS
  krls_sfs <- krls_m(xsfs.z, ysfs)
  # Uncomment the bellow  line if you want to use the error in the training 
  # for choosing the best models
  #krls_error_sfs_train <- (krls_sfs$krls_model$Looe[,1])/nrow(xsfs.z)
  krls_error_sfs_pred <- predict(krls_sfs$krls_model, xsfs_val.z)
  krls_error_sfs <- rmse(krls_error_sfs_pred$fit, ysfs_val)
  
  models <- c("LASSO", "ElasticNet", "GAM", "KRLS")
  performance_vec_sfs <- c(lasso_error_sfs, eNet_error_sfs, gam_error_sfs, krls_error_sfs)
  # Uncomment the bellow  line if you want to use the error in the training 
  # for choosing the best models, default is to use a validation set.
  #performance_vec_sfs <- c(lasso_error_sfs_train, eNet_error_sfs_train, gam_error_sfs_train, krls_error_sfs_train)
  best_model_sfs <- which.min(performance_vec_sfs) 
  print("Best Model of the combination of only SFs:")
  print(models[best_model_sfs])
  
  
  # Best Model: LASSO
  if(best_model_sfs == 1){
    best_train_sfs <- predict(lasso_sfs$lasso_model, newx = xsfs.z)
    best_val_sfs <- predict(lasso_sfs$lasso_model, xsfs_val.z)
    if(!is.null(opt$sfs_new) & !is.null(opt$des_new)){
      best_new_sfs <- predict(lasso_sfs$lasso_model, newx = xsfs_new.z)
    }
  }
  
  # Best Model: Elastic Net
  if(best_model_sfs == 2){
    best_train_sfs <- predict(eNet_sfs$eNet_model, newx = xsfs.z)
    best_val_sfs <- predict(eNet_sfs$eNet_model, newx = xsfs_val.z)
    if(!is.null(opt$sfs_new) & !is.null(opt$des_new)){
      best_new_sfs <- predict(eNet_sfs$eNet_model, newx = xsfs_new.z)
    }
  }
  
  # Best Model: GAM
  if(best_model_sfs == 3){
    best_train_sfs <- predict(gam_sfs$gam_model,  xsfs.z)
    best_val_sfs <- predict(gam_sfs$gam_model, xsfs_val.z)
    if(!is.null(opt$sfs_new) & !is.null(opt$des_new)){
      best_new_sfs <- predict(gam_sfs$gam_model, xsfs_new.z)
    }
  }
  
  # Best Model: KRLS
  if(best_model_sfs == 4){
    best_train_sfs <- predict(krls_sfs$krls_model, xsfs.z)
    best_train_sfs <- best_train_sfs$fit
    best_val_sfs <- predict(krls_sfs$krls_model, xsfs_val.z)
    best_val_sfs <- best_val_sfs$fit
    if(!is.null(opt$sfs_new) & !is.null(opt$des_new)){
      best_new_sfs <- predict(krls_sfs$krls_model, xsfs_new.z)
      best_new_sfs <- best_new_sfs$fit
    }
  }
  
  
  ########################################################################################
  # STACKING PROCEDURE
  # Compute the stacking process, using Ridge regression, either for the best model of 
  # the SFs and descriptors or for a set of SFs with the best model of descriptors.
  ########################################################################################
  # Stacking Best Models of SFs and Descriptors
  print("Learning Ridge Regression Stacking...")
  
  # Models For Training
  xmodels_train <- cbind(best_train_sfs, best_train_all)
  colnames(xmodels_train) <- c("Best_sfs", "Best_all")

  stacking_standardization_center <- apply(xmodels_train, 2, mean)
  xmodels_train_variance <- apply(xmodels_train, 2, var)
  xmodels_train_datapoints <- nrow(xmodels_train)
  stacking_standardization_scale <- sqrt((xmodels_train_datapoints-1)/xmodels_train_datapoints)*sqrt(xmodels_train_variance)
  
  xmodels_train <- scale(xmodels_train, center=stacking_standardization_center,
                         scale=stacking_standardization_scale)
  xmodels_train <- as.data.frame(xmodels_train)
  ymodels_train <- ysfs
  
  # Models For Validation
  xmodels_val <- cbind(best_val_sfs, best_val_all)
  colnames(xmodels_val) <- c("Best_sfs", "Best_all")
  xmodels_val <- scale(xmodels_val, center=stacking_standardization_center,
                       scale=stacking_standardization_scale)
  xmodels_val <- as.data.frame(xmodels_val)
  ymodels_val <- ysfs_val
  
  # Models For New data (if there is)
  if(!is.null(opt$sfs_new) & !is.null(opt$des_new)){
    xmodels_new <- cbind(best_new_sfs, best_new_all)
    colnames(xmodels_new) <- c("Best_sfs", "Best_all")
    xmodels_new <- scale(xmodels_new, center=stacking_standardization_center,
                         scale=stacking_standardization_scale)
    xmodels_new <- as.data.frame(xmodels_new)
  }
  
  
  # Ridge Stacking Learning
  ridge_stack_model <- ridge_stack(xmodels_train, ymodels_train)
  # Ridge Stacking Prediction on Validation set
  stack_val_res <- predict(ridge_stack_model$ridge_model, newx = as.matrix(xmodels_val))
  # Ridge Stacking Prediction on New dataset (if there is)
  if(!is.null(opt$sfs_new) & !is.null(opt$des_new)){
    stack_new_res <- predict(ridge_stack_model$ridge_model, newx = as.matrix(xmodels_new))
    newdata_results <- cbind(best_new_sfs, best_new_all, stack_new_res)
    colnames(newdata_results) <- c("Best_SFs", "Best_SFs_Des", "Ridge_Stack")
    write.csv(newdata_results, "newdata_predictions.csv")
  }
  
  
  ########################################################################################
  # PERFORMANCE RESULTS AND EVALUATION
  # The performance results on the validation set are always provided in terms of
  # Pearson Correlation (Rp) and Root Mean Squared Error (RMSE)
  ########################################################################################
  print("Evaluation SFs + descriptors")
  print("Pearson Correlation:")
  print(cor(best_val_all, ymodels_val[,1]))
  print("RMSE:")
  print(rmse(as.numeric(best_val_all), ymodels_val[,1]))

  print("Evaluation SFs")
  print("Pearson Correlation:")
  print(cor(best_val_sfs, ymodels_val[,1]))
  print("RMSE:")
  print(rmse(as.numeric(best_val_sfs), ymodels_val[,1]))
  
  print("Evaluation SFs + descriptors")
  print("Pearson Correlation:")
  print(cor(best_val_all, ymodels_val[,1]))
  print("RMSE:")
  print(rmse(as.numeric(best_val_all), ymodels_val[,1]))
  
  print("Evaluation Ridge")
  print("Pearson Correlation:")
  print(cor(stack_val_res[,1], ymodels_val[,1]))
  print("RMSE:")
  print(rmse(stack_val_res[,1], ymodels_val[,1]))
  
  if (!is.null(opt$new_data)){
    print("Statistics for the predicted values:")
    
    print("Evaluation SFs")
    print("Pearson Correlation:")
    print(cor(newdata_results[,1], ysfs_new[,1]))
    print("RMSE:")
    print(rmse(newdata_results[,1], ysfs_new[,1]))
    
    print("Evaluation SFs + descriptors")
    print("Pearson Correlation:")
    print(cor(newdata_results[,2], ysfs_new[,1]))
    print("RMSE:")
    print(rmse(as.numeric(newdata_results[,2]),ysfs_new[,1]))
    
    print("Evaluation Ridge")
    print("Pearson Correlation:")
    print(cor(newdata_results[,3], ysfs_new[,1]))
    print("RMSE:")
    print(rmse(newdata_results[,3], ysfs_new[,1]))
    
  }
  
  
  ########################################################################################
  # Reports with Plots in PDF 
  ########################################################################################
  pdf("SFs_Results.pdf")
  par(mfrow=c(2,2))
  plot(lasso_sfs$lasso_cv$glmnet.fit, "lambda", label=TRUE, main = "LASSO 10-Fold CV Lambda")
  plot(lasso_sfs$lasso_cv, main = "LASSO shrinkage of coefficients")
  par(mfrow=c(2,2))
  plot(eNet_sfs$eNet_cv$glmnet.fit, "lambda", label=TRUE, main = "Elastic Net 10-Fold CV Lambda")
  plot(eNet_sfs$eNet_cv,main = "Elastic Net shrinkage of coefficients" )
  par(mfrow=c(2,2))
  plot(gam_sfs$gam_model, scheme = 1,residuals=TRUE, all.terms = TRUE)
  par(mfrow=c(2,2))
  plot(krls_sfs$krls_model)
  dev.off()
  
  pdf("Descriptors_SFs_Results.pdf")
  par(mfrow=c(2,2))
  plot(lasso_des$lasso_cv$glmnet.fit, "lambda", label=TRUE, main = "LASSO 10-Fold CV Lambda")
  plot(lasso_des$lasso_cv, main = "LASSO shrinkage of coefficients")
  par(mfrow=c(2,2))
  plot(eNet_des$eNet_cv$glmnet.fit, "lambda", label=TRUE, main = "Elastic Net 10-Fold CV Lambda")
  plot(eNet_des$eNet_cv,main = "Elastic Net shrinkage of coefficients" )
  par(mfrow=c(2,2))
  plot(gam_des$gam_model, scheme = 1,residuals=TRUE, all.terms = TRUE)
  par(mfrow=c(2,2))
  plot(krls_des$krls_model)
  dev.off()
  
  pdf("Stacking_Results.pdf")
  plot(ridge_stack_model$ridge_cv$glmnet.fit, "lambda", label=TRUE, main = "Ridge 1 Fold CV Lambda")
  plot(ridge_stack_model$ridge_cv, main = "Ridge shrinkage of coefficients")
  dev.off()
}

########################################################################################
# ARGUMENTS INPUT  PROCESSING
# This section performs input processing for the arguments that have to be passed and
# should NOT be modified
# ######################################################################################

getParser <- function() {
  option_list = list(
    make_option(c("-a", "--sfs"), type = "character", default = NULL,
                help = "SFs Training dataset REQUIRED", metavar = "csv file"),
    make_option(c("-b", "--sfs_val"), type = "character", default = NULL, 
                help = "SFs Validation dataset OPTIONAL", metavar = "csv file"),
    make_option(c("-c", "--sfs_new"), type = "character", default = NULL, 
                help = "SFs New Data dataset OPTIONAL", metavar = "csv file"),
    make_option(c("-d", "--des"), type = "character", default = NULL, 
                help = "Descriptors Training dataset REQUIRED", metavar = "csv file"),
    make_option(c("-e", "--des_val"), type = "character", default = NULL, 
                help = "Descriptors Validation dataset OPTIONAL", metavar = "csv file"),
    make_option(c("-f", "--des_new"), type = "character", default = NULL, 
                help = "Descriptors New Data dataset OPTIONAL", metavar = "csv file"),
    make_option(c("-g","--new_data"), action = "store_true", default = NULL,
                help = "A flag that allows for experimental energies in the new_data files. Should be only used with new_data values")
  );
  
  OptionParser(option_list = option_list);
}


# To run interactively, set option run.commandline to FALSE
# > options(run.commandline = FALSE)
# then, get a parser, parse the arguments, and call main
# You must first have sourced this file, so that getParser is in your environment
# > opt_parser <- getParser()
# > opt <- parse_args(opt_parser, args=unlist(strsplit("--sfs mysfs.csv --des mydes.csv", " ")))
# > main(opt)
#
# When running from the command line, the option will become, by default, TRUE,
# and options will be obtained from the command line.
if (getOption('run.commandline', default=TRUE)) {
  
  opt_parser <- getParser();
  opt = parse_args(opt_parser);
  
  set.seed(1234)
  main(opt)  
}
