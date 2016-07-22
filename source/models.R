##############################################################################################
#    Models Function: Ridge, LASSO, Elastic Net, GAM, KRLS 
#    @author: Gabriela Hernández
#    EM-DMKM 2014-2016
##############################################################################################
library(glmnet)
library(KRLS)
library(mgcv)
library(caret)
library(ggplot2)
library(caret)
library(hydroGOF)

###################################################################################
#    Model Functions
###################################################################################

# Penalized Linear Regression Models


lasso <- function(x, y){
  # require(glmnet)
  # Performs the LASSO Model, using 10 Fold cross-validation to learn
  # the parameters
  #
  # Args:
  #   x :   predictors (independent variables) matrix/df of the training, 
  #         centered to zero with a standard deviation of 1.
  #   y :   response (dependent or target) variable.
  # Returns:
  #   lasso_cv object:   Results of the 10-Fold CV:
  #                      MSE results
  #                      lambda.min = lambda that minimizes the Cross Validation (CV) error
  #                      lambda.1se = lambda that minimizes the CV error plus one standard error
  #   lasso_model object:    LASSO Model fitted with lambda.1se,
  #                          This object can be used to predict new data.
  lasso_cv <- cv.glmnet(x = as.matrix(x), y = y[,1], alpha = 1)
  lasso_model <- glmnet(x = as.matrix(x), y = y[,1], alpha = 1, 
                        lambda = lasso_cv$lambda.1se)
  return(list(lasso_cv = lasso_cv, lasso_model = lasso_model))
}


eNet <- function(x, y){
  #require(glmnet)
  # Performs the Elastic Net Model, using 10 Fold cross-validation to learn
  # the parameters
  #
  # Args:
  #   x :   predictors (independent variables) matrix/df of the training, 
  #         centered to zero with a standard deviation of 1.
  #   y :   response (dependent or target) variable.
  # Returns:
  #   eNet_cv object:   Results of the 10-Fold CV:
  #                      MSE results
  #                      lambda.min = lambda that minimizes the Cross Validation (CV) error
  #                      lambda.1se = lambda that minimizes the CV error plus one standard error
  #   eNet_model object:    Elastic Net Model fitted with lambda.1se and alpha = 0.5,
  #                          This object can be used to predict new data.
  eNet_cv <- cv.glmnet(as.matrix(x), y[,1], alpha = 0.5)
  eNet_model <- glmnet(as.matrix(x), y[,1], alpha = 0.5, 
                       lambda = eNet_cv$lambda.1se)
  return(list(eNet_cv = eNet_cv, eNet_model = eNet_model))
}


# Semi-parametric Model - Generalized Additive Models (GAM)

gam_m <- function(x,y){
  #require(mgcv)
  # Performs the GAM Model, using GCV for learning parameters, and
  # penalized cubic splines with k=3 knots to avoid overfitting.
  #
  # Args:
  #   x :   predictors (independent variables) matrix/df of the training, 
  #         centered to zero with a standard deviation of 1.
  #   y :   response (dependent or target) variable.
  # Returns:
  #   gam_model object:  gam mgcv object model, with:
  #                      Formula used
  #                      GCV error
  #                      coefficients, residuals, deviance, fitted values
  #                      with this object the scatterplot smoothers can be obtained
  #                      and this object can be used to predict new values.
  
  
  # Create formula for fitting the GAM model with k=3 knots to avoid
  # overfitting.
  vars <- c()
  names <- colnames(x)
  for (i in 1:length(names)){
    beg <- 's('
    mid <- "bs=\'cs\'"
    mid2 <- "k=3"
    comma <- ','
    end <- ')'
    #term <- paste(beg, names[i],comma, mid, end, collapse ="")
    term <- paste(beg, names[i],comma, mid, comma, mid2, end, collapse ="")
    vars <- append(vars, term)
  }
  target <- colnames(y[1])
  form <- paste(target, paste(vars, collapse=" + "), sep=" ~ ")
  formula <- as.formula(form)
  # Fit GAM Model
  gam_model <- mgcv::gam(formula, data= cbind(y, x), select =T,
                         method = "GCV.Cp")
  return(list(gam_model = gam_model))
}


# Non-Parametric Model - Kernel-based Regularized Least Squares (KRLS)

krls_m <- function(x, y){
  # Performs the KRLS Model, using LOOCV for learning the penalized lambda parameter, 
  # Gaussian Kernel and the kernel bandwidth is set to dim(x), i.e. number of dimensions
  #
  # Args:
  #   x :   predictors (independent variables) matrix/df of the training, 
  #         centered to zero with a standard deviation of 1.
  #   y :   response (dependent or target) variable.
  # Returns:
  #   krls_model object:  krls KRLS object model, with:
  #                      Formula used
  #                      LOOEs from the LOOCV errors
  #                      R2 goodness of fit
  #                      coefficients, derivatives, fitted values
  #                      with this object the Estimates of the conditional expectations
  #                      plots can be obtained
  #                      This object can be used to predict new values.
  
  krls_model <- KRLS::krls(X = as.matrix(x), y = y, whichkernel = "gaussian", 
                           print.level = 0)
  
  return(list(krls_model=krls_model))
}


# Ridge Stacking Model

ridge_stack <- function(xmodels, y){
  #require(glmnet)
  # Performs the Ridge Stacking Model, using 10 Fold cross-validation to learn
  # the parameters
  #
  # Args:
  #   xmodels :   predictors (independent variables) matrix/df of the Models to stack, 
  #               centered to zero with a standard deviation of 1.
  #         y :   response (dependent or target) variable.
  # Returns:
  #   ridge_cv object:   Results of the 10-Fold CV:
  #                      MSE results
  #                      lambda.min = lambda that minimizes the Cross Validation (CV) error
  #                      lambda.1se = lambda that minimizes the CV error plus one standard error
  #   ridge_model object:    Ridge Model fitted with lambda.1se,
  #                          This object can be used to predict new data.
  ridge_cv <- cv.glmnet(as.matrix(xmodels), k = 1, y[,1], alpha = 0)
  ridge_model <- glmnet(as.matrix(xmodels),  y[,1], alpha = 0, 
                        lambda = ridge_cv$lambda.1se)
  return(list(ridge_cv = ridge_cv, ridge_model = ridge_model))
  
}


evaluation <- function(ypred, yreal){
  #require(hydroGOF)
  rp <- cor(ypred, yreal)
  rmse <- rmse(as.numeric(ypred), yreal[,1])
  return(list(Rp = rp, RMSE = rmse))
}