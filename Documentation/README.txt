# Protein-Ligand Binding Affinity Prediction

===================================================================================================================================
=                                                        Description
===================================================================================================================================
Predicting protein-ligand binding affinity constitutes a key computational method in the early stages of the drug discovery process. Molecular docking programs attempt to predict them by using mathematical approximations, namely, scoring functions. Constantly, new experiments and techniques are carried out in order to find out crucial descriptors that better characterize the protein-ligand interaction. In this work, we investigate and apply different machine learning and statistical techniques to create a novel
framework capable of combining different descriptors as well as scoring functions in order to both estimate and improve the overall binding affinity. This strategy also filters the strongest descriptors and scoring functions while permitting more complex
interpretations by examining non-linearities and interactions. This approach consists of two steps. First,several descriptors and
scoring functions are separately combined and assessed through models based on penalized linear regression methods with embedded
feature selection, such as LASSO and Elastic Net. In order to avoid strong parametric assumptions, alternative intelligible
non-parametric techniques are exploited such as Generalized Additive Models and Kernel-based Regularized Least Squares. Secondly,
stacking methods  are employed to further boost the binding affinity prediction of different scoring functions, by adding new models
with the descriptors. We apply this methodology to well-studied datasets of high-quality protein-ligand complexes, based on
the 2007 and 2013 PDBbind Benchmarks, achieving a significant improvement in overall prediction of binding affinity.

Further details in the implementation, assumptions, detailed explanations, theory and results are provided inside the folder
Documentation.


===================================================================================================================================
=                                                         Requirements
===================================================================================================================================
The scripts were implemented in R 3.1.4 using the following libraries:

  -optparse
  -caret
  -glmnet
  -KRLS
  -mgcv
  -ggplot2
  -hydroGOF

===================================================================================================================================
=                                   Information about the files, Execution & Reproducibility
===================================================================================================================================
This repository contains the following three folders:

  1. source: Under this folder, we can find three scripts in R:
    - fw1.R:  In this script, either SFs or despcriptors, separately, are combined using the LASSO, Elastic Net, GAM and KRLS models.                   Provide tools to detect the possible subset of most significant variables, analyze the independent interaction of each
              descriptor and SF with  the protein- ligand binding affinity from different perspectives.

              @Preprocessing: Z-score Standardization
              @Input: Either the SFs datasets or descriptors datasets are provided. There can be three different datasets provided:
                      Training Dataset, Validation Dataset, NewData Dataset. At least, the training dataset must be provided. If not
                      validation dataset is provided, then stratified sampling on the training dataset is performed using 70% for
                      training and 30% for validation.  Running from the terminal each dataset should  be provided as follows:
                      -a SFs training dataset
                      -b SFs validation dataset
                      -c SFs newdata dataset
              @Output: If three datasets are provided, it returns the performance in terms of Pearson Correlation and RMSE of all
                       the models in the validation dataset, a pdf file with the plots obtained from the different models for further
                       interpretations such as shrinkage plots, smooth splines plots, extimates of the conditional expectation plots and                          a csv file for the predictions in the NewData dataset. If  either only the training dataset is provided or
                       training and validation datasets are given,  the outputs are the same, except the csv file.
              @Examples: Running the Script from the terminal as:
                       Rscript --vanilla fw1.R -a sfs_training.csv  -b sfs_validation.csv -c sfs_newdata.csv
                       Rscript --vanilla fw1.R -a descriptors_training.csv  -b descriptors_validation.csv -c descriptors_newdata.csv

                       With the data provided in the datasets folder:
                       Rscript --vanilla fw1.R -a datasets/PL_MIN/fw1/set1_5_sfs_715.csv  -b datasets/PL_MIN/fw1/set2_5_sfs_191.csv
                                               -c datasets/PL_MIN/fw1/set3_5_sfs_64.csv
                       Rscript --vanilla fw1.R -a datasets/PL_MIN/fw1/set1_all_descriptors_715.csv
                                               -b datasets/PL_MIN/fw1/set2_all_descriptors_191.csv
                                               -c datasets/PL_MIN/fw1/set3_all_descriptors_64.csv

    - fw2.R: In this script, the stacking procedure is performed. First the SFs and descriptors are assessed with the four models:
             LASSO, Elastic Net, GAM and KRLS; the best model for SFs & descriptors is chosen to later stack them by using a Ridge
             Regression Stacking procedure.

              @Preprocessing: Z-score Standardization to all the variables.
              @Input: Both the SFs and descriptors datasets are provided. There can be six different datasets provided:
                      SFs Training Dataset, SFs Validation Dataset, SFs NewData Dataset, Descriptors Training Dataset, Descriptors
                      Validation Dataset, Descriptors NewData Dataset. At least, the SFs and Descriptors training datasets must be
                      provided. If not validation datasets are provided, then stratified sampling on the SFs & Descriptors training
                      datasets is performed using 70% for training and 30% for validation. Running from the terminal each dataset should
                      be provided as follows:
                      -a SFs training dataset
                      -b SFs validation dataset
                      -c SFs newdata dataset
                      -d Descriptors training dataset
                      -e Descriptors validation dataset
                      -f Descriptors newdata dataset
                      -g T or F flag. I f the option -g T  is used the program will use the last column in the new_data file as the
                      experimental values for this and compute the statistics as well.

              @Output: Pearson Correlation and RMSE of all  the models in the validation dataset, a pdf file with the plots obtained from                        the different models for further interpretations such as shrinkage plots, smooth splines plots, extimates of the
                       conditional expectation plots and a csv file for the predictions in the NewData datasets. If  either only the
                       training datasets are  provided or training and validation datasets are given, the outputs are the same, except
                       the csv file.
              @Examples: Running the Script from the terminal as:
                       Rscript --vanilla fw2.R -a sfs_training.csv  -b sfs_validation.csv -c sfs_newdata.csv - d descriptors_training.csv
                                               -e descriptors_validation.csv -f descriptors_newdata.csv - g F

                       With the data provided in the datasets folder:
                       Rscript --vanilla fw2.R -a datasets/PL_MIN/fw2/set1_5_sfs_715_order.csv
                                               -b datasets/PL_MIN/fw2/set2_5_sfs_191_order.csv
                                               -c datasets/PL_MIN/fw2/set3_5_sfs_64_order.csv
                                               -d datasets/PL_MIN/fw2/set1_all_descriptors_715_order.csv
                                               -e datasets/PL_MIN/fw2/set2_all_descriptors_191_order.csv
                                               -f datasets/PL_MIN/fw2/set3_all_descriptors_64_order.csv

    - models.R: This script contains the computation of the four models: LASSO, Elastic Net, GAM & KRLS, the stakcing prodcedure with
                Ridge Regression and the evaluation measures (Pearson Correlation and RMSE) used by the fw1.R and fw2.R frameworks.

  2. datasets: The datasets are divided by the complexes information, in which PL stands for Protein-Ligand complexes, W if the
               complexes contain Waters, Min if the complexes are Minimzed, NoMin if the complexes are not minimized. This is
               provided for the sake of reproducibility. Many of the results shown in the documentation have been gotten from these
               datasets.
      -PL_MIN: Protein-Ligand Complexes Without Waters and Minimized.
            fw1: Folder containing the datasets that can be used within the fw1.R script.
            fw2: Folder containing the datasets to be used within the fw2.R script.

      -PL_NoMIN: Protein-Ligand Complexes Without Waters and Without Minimized.
            fw1: Folder containing the datasets that can be used within the fw1.R script.
            fw2: Folder containing the datasets to be used within the fw2.R script.

      -PLW_MIN: Protein-Ligand Complexes With Waters and Minimized.
            fw1: Folder containing the datasets that can be used within the fw1.R script.
            fw2: Folder containing the datasets to be used within the fw2.R script.

      -PLW_NoMIN: Protein-Ligand Complexes With Waters and Minimized.
            fw1: Folder containing the datasets that can be used within the fw1.R script.
            fw2: Folder containing the datasets to be used within the fw2.R script.

  3. Documentation: Under this folder, we can find two pdf files:
       -BindingAffinity_Prediction.pdf : A full report describing the whole project. Divided into 7 sections: Introduction, Related
                                         Work, Protein-Ligand Complexes Datasets, Methodology, Results, Discussion and Future Work,
                                         Conclusions. The whole report is fundamental to understand the importance of the work and
                                         how ML and Statistics techniques can contribute to the Drug Discovery area.

       -Models_Plots.pdf: A PDF file including as an example  the plots that are expected as one of the outcomes of both scripts:
                          fw1.R & fw2,

===================================================================================================================================
=                                             Execution
===================================================================================================================================

From the terminal/CMD change directory to the project and execute either the fw1.R script or the fw2.R script as follows:

Rscript --vanilla fw1.R -a sfs_training.csv  -b sfs_validation.csv -c sfs_newdata.csv

Rscript --vanilla source/fw2.R -a sfs_training.csv -b sfs_validation.csv -c sfs_newdata.csv - d descriptors_training.csv
-e descriptors_validation.csv -f descriptors_newdata.csv - g F
