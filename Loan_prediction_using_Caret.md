Loan prediction using Machine Learning
================
Subranjit sahoo

Dream Housing Finance company deals in providing home loans. They have presence across all urban, semi urban and rural areas. Customer first apply for home loan after that company validates the customer eligibility for loan.

Problem Statement-
------------------

Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers. Here they have provided a partial data set.

### &gt; Objective:- To predict the number of customers who are eligible for loan based on other characteristics of customer data.

#### This is the description of the variables in the data set

-   Variable \* Description
-   Loan\_ID \* Unique Loan ID
-   Gender \* Male/ Female
-   Married \* Applicant married (Y/N)
-   Dependents \* Number of dependents
-   Education \* Applicant Education (Graduate/ Under Graduate)
-   Self\_Employed \* Self employed (Y/N)
-   ApplicantIncome \* Applicant income
-   Coapplicant Income \* Coapplicant income
-   LoanAmount \* Loan amount in thousands
-   Loan\_Amount\_Term \* Term of loan in months
-   Credit\_History \* credit history meets guidelines
-   Property\_Area \* Urban/ Semi Urban/ Rural
-   Loan\_Status \* Loan approved (Y/N)

``` r
#loading required packages
library(caret)
library(gbm)
library(ggplot2)
```

Looking at the structure of data `train` contains the data set

``` r
str(train)
```

    ## Classes 'tbl_df', 'tbl' and 'data.frame':    614 obs. of  13 variables:
    ##  $ Loan_ID          : chr  "LP001002" "LP001003" "LP001005" "LP001006" ...
    ##  $ Gender           : chr  "Male" "Male" "Male" "Male" ...
    ##  $ Married          : chr  "No" "Yes" "Yes" "Yes" ...
    ##  $ Dependents       : chr  "0" "1" "0" "0" ...
    ##  $ Education        : chr  "Graduate" "Graduate" "Graduate" "Not Graduate" ...
    ##  $ Self_Employed    : chr  "No" "No" "Yes" "No" ...
    ##  $ ApplicantIncome  : int  5849 4583 3000 2583 6000 5417 2333 3036 4006 12841 ...
    ##  $ CoapplicantIncome: num  0 1508 0 2358 0 ...
    ##  $ LoanAmount       : int  NA 128 66 120 141 267 95 158 168 349 ...
    ##  $ Loan_Amount_Term : int  360 360 360 360 360 360 360 360 360 360 ...
    ##  $ Credit_History   : int  1 1 1 1 1 1 1 0 1 1 ...
    ##  $ Property_Area    : chr  "Urban" "Rural" "Urban" "Urban" ...
    ##  $ Loan_Status      : chr  "Y" "N" "Y" "Y" ...
    ##  - attr(*, "spec")=List of 2
    ##   ..$ cols   :List of 13
    ##   .. ..$ Loan_ID          : list()
    ##   .. .. ..- attr(*, "class")= chr  "collector_character" "collector"
    ##   .. ..$ Gender           : list()
    ##   .. .. ..- attr(*, "class")= chr  "collector_character" "collector"
    ##   .. ..$ Married          : list()
    ##   .. .. ..- attr(*, "class")= chr  "collector_character" "collector"
    ##   .. ..$ Dependents       : list()
    ##   .. .. ..- attr(*, "class")= chr  "collector_character" "collector"
    ##   .. ..$ Education        : list()
    ##   .. .. ..- attr(*, "class")= chr  "collector_character" "collector"
    ##   .. ..$ Self_Employed    : list()
    ##   .. .. ..- attr(*, "class")= chr  "collector_character" "collector"
    ##   .. ..$ ApplicantIncome  : list()
    ##   .. .. ..- attr(*, "class")= chr  "collector_integer" "collector"
    ##   .. ..$ CoapplicantIncome: list()
    ##   .. .. ..- attr(*, "class")= chr  "collector_double" "collector"
    ##   .. ..$ LoanAmount       : list()
    ##   .. .. ..- attr(*, "class")= chr  "collector_integer" "collector"
    ##   .. ..$ Loan_Amount_Term : list()
    ##   .. .. ..- attr(*, "class")= chr  "collector_integer" "collector"
    ##   .. ..$ Credit_History   : list()
    ##   .. .. ..- attr(*, "class")= chr  "collector_integer" "collector"
    ##   .. ..$ Property_Area    : list()
    ##   .. .. ..- attr(*, "class")= chr  "collector_character" "collector"
    ##   .. ..$ Loan_Status      : list()
    ##   .. .. ..- attr(*, "class")= chr  "collector_character" "collector"
    ##   ..$ default: list()
    ##   .. ..- attr(*, "class")= chr  "collector_guess" "collector"
    ##   ..- attr(*, "class")= chr "col_spec"

``` r
summary(train)
```

    ##    Loan_ID             Gender            Married         
    ##  Length:614         Length:614         Length:614        
    ##  Class :character   Class :character   Class :character  
    ##  Mode  :character   Mode  :character   Mode  :character  
    ##                                                          
    ##                                                          
    ##                                                          
    ##                                                          
    ##   Dependents         Education         Self_Employed      ApplicantIncome
    ##  Length:614         Length:614         Length:614         Min.   :  150  
    ##  Class :character   Class :character   Class :character   1st Qu.: 2878  
    ##  Mode  :character   Mode  :character   Mode  :character   Median : 3812  
    ##                                                           Mean   : 5403  
    ##                                                           3rd Qu.: 5795  
    ##                                                           Max.   :81000  
    ##                                                                          
    ##  CoapplicantIncome   LoanAmount    Loan_Amount_Term Credit_History  
    ##  Min.   :    0     Min.   :  9.0   Min.   : 12      Min.   :0.0000  
    ##  1st Qu.:    0     1st Qu.:100.0   1st Qu.:360      1st Qu.:1.0000  
    ##  Median : 1188     Median :128.0   Median :360      Median :1.0000  
    ##  Mean   : 1621     Mean   :146.4   Mean   :342      Mean   :0.8422  
    ##  3rd Qu.: 2297     3rd Qu.:168.0   3rd Qu.:360      3rd Qu.:1.0000  
    ##  Max.   :41667     Max.   :700.0   Max.   :480      Max.   :1.0000  
    ##                    NA's   :22      NA's   :14       NA's   :50      
    ##  Property_Area      Loan_Status       
    ##  Length:614         Length:614        
    ##  Class :character   Class :character  
    ##  Mode  :character   Mode  :character  
    ##                                       
    ##                                       
    ##                                       
    ## 

Checking for 'NA' or missing values

``` r
sum(is.na(train))
```

    ## [1] 86

``` r
#Imputing missing values using KNN.Also centering and scaling numerical columns

preProcValues= preProcess(train, method = c("knnImpute","center","scale"))
preProcValues
```

    ## Created from 529 samples and 13 variables
    ## 
    ## Pre-processing:
    ##   - centered (5)
    ##   - ignored (8)
    ##   - 5 nearest neighbor imputation (5)
    ##   - scaled (5)

Lets predict the missing values after the imputation

``` r
library(RANN)
train_processed=predict(preProcValues,train)
sum(is.na(train_processed))
```

    ## [1] 0

``` r
#Converting outcome variable to numeric
train_processed$Loan_Status<-ifelse(train_processed$Loan_Status=='N',0,1)
```

``` r
id<-train_processed$Loan_ID
train_processed$Loan_ID<-NULL

#Checking the structure of processed train file
str(train_processed)
```

    ## Classes 'tbl_df', 'tbl' and 'data.frame':    614 obs. of  12 variables:
    ##  $ Gender           : chr  "Male" "Male" "Male" "Male" ...
    ##  $ Married          : chr  "No" "Yes" "Yes" "Yes" ...
    ##  $ Dependents       : chr  "0" "1" "0" "0" ...
    ##  $ Education        : chr  "Graduate" "Graduate" "Graduate" "Not Graduate" ...
    ##  $ Self_Employed    : chr  "No" "No" "Yes" "No" ...
    ##  $ ApplicantIncome  : num  0.0729 -0.1343 -0.3934 -0.4617 0.0976 ...
    ##  $ CoapplicantIncome: num  -0.554 -0.0387 -0.554 0.2518 -0.554 ...
    ##  $ LoanAmount       : num  0.0162 -0.2151 -0.9395 -0.3086 -0.0632 ...
    ##  $ Loan_Amount_Term : num  0.276 0.276 0.276 0.276 0.276 ...
    ##  $ Credit_History   : num  0.432 0.432 0.432 0.432 0.432 ...
    ##  $ Property_Area    : chr  "Urban" "Rural" "Urban" "Urban" ...
    ##  $ Loan_Status      : num  1 0 1 1 1 1 1 0 1 0 ...
    ##  - attr(*, "spec")=List of 2
    ##   ..$ cols   :List of 13
    ##   .. ..$ Loan_ID          : list()
    ##   .. .. ..- attr(*, "class")= chr  "collector_character" "collector"
    ##   .. ..$ Gender           : list()
    ##   .. .. ..- attr(*, "class")= chr  "collector_character" "collector"
    ##   .. ..$ Married          : list()
    ##   .. .. ..- attr(*, "class")= chr  "collector_character" "collector"
    ##   .. ..$ Dependents       : list()
    ##   .. .. ..- attr(*, "class")= chr  "collector_character" "collector"
    ##   .. ..$ Education        : list()
    ##   .. .. ..- attr(*, "class")= chr  "collector_character" "collector"
    ##   .. ..$ Self_Employed    : list()
    ##   .. .. ..- attr(*, "class")= chr  "collector_character" "collector"
    ##   .. ..$ ApplicantIncome  : list()
    ##   .. .. ..- attr(*, "class")= chr  "collector_integer" "collector"
    ##   .. ..$ CoapplicantIncome: list()
    ##   .. .. ..- attr(*, "class")= chr  "collector_double" "collector"
    ##   .. ..$ LoanAmount       : list()
    ##   .. .. ..- attr(*, "class")= chr  "collector_integer" "collector"
    ##   .. ..$ Loan_Amount_Term : list()
    ##   .. .. ..- attr(*, "class")= chr  "collector_integer" "collector"
    ##   .. ..$ Credit_History   : list()
    ##   .. .. ..- attr(*, "class")= chr  "collector_integer" "collector"
    ##   .. ..$ Property_Area    : list()
    ##   .. .. ..- attr(*, "class")= chr  "collector_character" "collector"
    ##   .. ..$ Loan_Status      : list()
    ##   .. .. ..- attr(*, "class")= chr  "collector_character" "collector"
    ##   ..$ default: list()
    ##   .. ..- attr(*, "class")= chr  "collector_guess" "collector"
    ##   ..- attr(*, "class")= chr "col_spec"

Cretaing dummy variables using a encoding. The caret package offers some unique features that help in creating dummy variables

``` r
# Converting every categorical variable to numerical using dummy variables
dmy <- dummyVars(" ~ .", data = train_processed,fullRank = T)
# Here, âfullrank=Tâ will create only (n-1) columns for a categorical column with n different levels. This works well particularly for the representing categorical predictors like gender, married, etc.

train_transformed <- data.frame(predict(dmy, newdata = train_processed))
# creating dummy variables on the train_transformed set
str(train_transformed)
```

    ## 'data.frame':    614 obs. of  19 variables:
    ##  $ GenderFemale          : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ GenderMale            : num  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ MarriedNo             : num  1 0 0 0 1 0 0 0 0 0 ...
    ##  $ MarriedYes            : num  0 1 1 1 0 1 1 1 1 1 ...
    ##  $ Dependents0           : num  1 0 1 1 1 0 1 0 0 0 ...
    ##  $ Dependents1           : num  0 1 0 0 0 0 0 0 0 1 ...
    ##  $ Dependents2           : num  0 0 0 0 0 1 0 0 1 0 ...
    ##  $ Dependents3.          : num  0 0 0 0 0 0 0 1 0 0 ...
    ##  $ EducationNot.Graduate : num  0 0 0 1 0 0 1 0 0 0 ...
    ##  $ Self_EmployedNo       : num  1 1 0 1 1 0 1 1 1 1 ...
    ##  $ Self_EmployedYes      : num  0 0 1 0 0 1 0 0 0 0 ...
    ##  $ ApplicantIncome       : num  0.0729 -0.1343 -0.3934 -0.4617 0.0976 ...
    ##  $ CoapplicantIncome     : num  -0.554 -0.0387 -0.554 0.2518 -0.554 ...
    ##  $ LoanAmount            : num  0.0162 -0.2151 -0.9395 -0.3086 -0.0632 ...
    ##  $ Loan_Amount_Term      : num  0.276 0.276 0.276 0.276 0.276 ...
    ##  $ Credit_History        : num  0.432 0.432 0.432 0.432 0.432 ...
    ##  $ Property_AreaSemiurban: num  0 0 0 0 0 0 0 1 0 1 ...
    ##  $ Property_AreaUrban    : num  1 0 1 1 1 1 1 0 1 0 ...
    ##  $ Loan_Status           : num  1 0 1 1 1 1 1 0 1 0 ...

``` r
#Converting the dependent variable back to categorical variable
train_transformed$Loan_Status=as.factor(train_transformed$Loan_Status)
str(train_transformed)
```

    ## 'data.frame':    614 obs. of  19 variables:
    ##  $ GenderFemale          : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ GenderMale            : num  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ MarriedNo             : num  1 0 0 0 1 0 0 0 0 0 ...
    ##  $ MarriedYes            : num  0 1 1 1 0 1 1 1 1 1 ...
    ##  $ Dependents0           : num  1 0 1 1 1 0 1 0 0 0 ...
    ##  $ Dependents1           : num  0 1 0 0 0 0 0 0 0 1 ...
    ##  $ Dependents2           : num  0 0 0 0 0 1 0 0 1 0 ...
    ##  $ Dependents3.          : num  0 0 0 0 0 0 0 1 0 0 ...
    ##  $ EducationNot.Graduate : num  0 0 0 1 0 0 1 0 0 0 ...
    ##  $ Self_EmployedNo       : num  1 1 0 1 1 0 1 1 1 1 ...
    ##  $ Self_EmployedYes      : num  0 0 1 0 0 1 0 0 0 0 ...
    ##  $ ApplicantIncome       : num  0.0729 -0.1343 -0.3934 -0.4617 0.0976 ...
    ##  $ CoapplicantIncome     : num  -0.554 -0.0387 -0.554 0.2518 -0.554 ...
    ##  $ LoanAmount            : num  0.0162 -0.2151 -0.9395 -0.3086 -0.0632 ...
    ##  $ Loan_Amount_Term      : num  0.276 0.276 0.276 0.276 0.276 ...
    ##  $ Credit_History        : num  0.432 0.432 0.432 0.432 0.432 ...
    ##  $ Property_AreaSemiurban: num  0 0 0 0 0 0 0 1 0 1 ...
    ##  $ Property_AreaUrban    : num  1 0 1 1 1 1 1 0 1 0 ...
    ##  $ Loan_Status           : Factor w/ 2 levels "0","1": 2 1 2 2 2 2 2 1 2 1 ...

Now lets create a cross validation set to tune our model.We will use createDataPartition() to do that.We will split our training data into two sets : 75% `trainset` and 25% `testset`

``` r
index=createDataPartition(train_transformed$Loan_Status,p=0.75,list = F)
# createDataPartition() also ensures uniform distribution of outcome variable classs
trainset=train_transformed[index,]
testset=train_transformed[-index,]
```

``` r
# checking the structure of `trainset`
str(trainset)
```

    ## 'data.frame':    461 obs. of  19 variables:
    ##  $ GenderFemale          : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ GenderMale            : num  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ MarriedNo             : num  1 0 0 1 0 0 0 1 0 1 ...
    ##  $ MarriedYes            : num  0 1 1 0 1 1 1 0 1 0 ...
    ##  $ Dependents0           : num  1 0 1 1 0 0 0 1 0 1 ...
    ##  $ Dependents1           : num  0 1 0 0 0 0 0 0 0 0 ...
    ##  $ Dependents2           : num  0 0 0 0 1 1 1 0 1 0 ...
    ##  $ Dependents3.          : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ EducationNot.Graduate : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ Self_EmployedNo       : num  1 1 0 1 1 1 1 1 1 1 ...
    ##  $ Self_EmployedYes      : num  0 0 1 0 0 0 0 0 0 0 ...
    ##  $ ApplicantIncome       : num  0.0729 -0.1343 -0.3934 0.0976 -0.2288 ...
    ##  $ CoapplicantIncome     : num  -0.554 -0.0387 -0.554 -0.554 -0.0325 ...
    ##  $ LoanAmount            : num  0.0162 -0.2151 -0.9395 -0.0632 0.2522 ...
    ##  $ Loan_Amount_Term      : num  0.276 0.276 0.276 0.276 0.276 ...
    ##  $ Credit_History        : num  0.432 0.432 0.432 0.432 0.432 ...
    ##  $ Property_AreaSemiurban: num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ Property_AreaUrban    : num  1 0 1 1 1 1 1 0 1 1 ...
    ##  $ Loan_Status           : Factor w/ 2 levels "0","1": 2 1 2 2 2 2 2 1 2 2 ...

As the training and testing sets are created now we will use select the important variables for our model using the recursive feature elimination or `rfe` method. `rfe` is used to find the best subset of features.

``` r
control= rfeControl(functions = rfFuncs, method = "repeatedcv",
                    repeats=3, verbose=F)
outcomeName='Loan_Status'
predictors=names(trainset)[!names(trainset) %in% outcomeName]
predictors
```

    ##  [1] "GenderFemale"           "GenderMale"            
    ##  [3] "MarriedNo"              "MarriedYes"            
    ##  [5] "Dependents0"            "Dependents1"           
    ##  [7] "Dependents2"            "Dependents3."          
    ##  [9] "EducationNot.Graduate"  "Self_EmployedNo"       
    ## [11] "Self_EmployedYes"       "ApplicantIncome"       
    ## [13] "CoapplicantIncome"      "LoanAmount"            
    ## [15] "Loan_Amount_Term"       "Credit_History"        
    ## [17] "Property_AreaSemiurban" "Property_AreaUrban"

``` r
Loan_Pred_Prof= rfe(trainset[,predictors], trainset[,outcomeName],
                 rfeControl= control)
```

    ## Warning: package 'randomForest' was built under R version 3.3.2

``` r
Loan_Pred_Prof
```

    ## 
    ## Recursive feature selection
    ## 
    ## Outer resampling method: Cross-Validated (10 fold, repeated 3 times) 
    ## 
    ## Resampling performance over subset size:
    ## 
    ##  Variables Accuracy  Kappa AccuracySD KappaSD Selected
    ##          4   0.7738 0.4152    0.04464  0.1126         
    ##          8   0.7889 0.4305    0.03237  0.1015        *
    ##         16   0.7824 0.4265    0.04564  0.1268         
    ##         18   0.7860 0.4326    0.04577  0.1276         
    ## 
    ## The top 5 variables (out of 8):
    ##    Credit_History, LoanAmount, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term

Lets see what all machine learning algorithms are provided by caret

``` r
names(getModelInfo())
```

    ##   [1] "ada"                 "AdaBag"              "AdaBoost.M1"        
    ##   [4] "adaboost"            "amdai"               "ANFIS"              
    ##   [7] "avNNet"              "awnb"                "awtan"              
    ##  [10] "bag"                 "bagEarth"            "bagEarthGCV"        
    ##  [13] "bagFDA"              "bagFDAGCV"           "bam"                
    ##  [16] "bartMachine"         "bayesglm"            "bdk"                
    ##  [19] "binda"               "blackboost"          "blasso"             
    ##  [22] "blassoAveraged"      "Boruta"              "bridge"             
    ##  [25] "brnn"                "BstLm"               "bstSm"              
    ##  [28] "bstTree"             "C5.0"                "C5.0Cost"           
    ##  [31] "C5.0Rules"           "C5.0Tree"            "cforest"            
    ##  [34] "chaid"               "CSimca"              "ctree"              
    ##  [37] "ctree2"              "cubist"              "dda"                
    ##  [40] "deepboost"           "DENFIS"              "dnn"                
    ##  [43] "dwdLinear"           "dwdPoly"             "dwdRadial"          
    ##  [46] "earth"               "elm"                 "enet"               
    ##  [49] "enpls.fs"            "enpls"               "evtree"             
    ##  [52] "extraTrees"          "fda"                 "FH.GBML"            
    ##  [55] "FIR.DM"              "foba"                "FRBCS.CHI"          
    ##  [58] "FRBCS.W"             "FS.HGD"              "gam"                
    ##  [61] "gamboost"            "gamLoess"            "gamSpline"          
    ##  [64] "gaussprLinear"       "gaussprPoly"         "gaussprRadial"      
    ##  [67] "gbm_h2o"             "gbm"                 "gcvEarth"           
    ##  [70] "GFS.FR.MOGUL"        "GFS.GCCL"            "GFS.LT.RS"          
    ##  [73] "GFS.THRIFT"          "glm.nb"              "glm"                
    ##  [76] "glmboost"            "glmnet_h2o"          "glmnet"             
    ##  [79] "glmStepAIC"          "gpls"                "hda"                
    ##  [82] "hdda"                "hdrda"               "HYFIS"              
    ##  [85] "icr"                 "J48"                 "JRip"               
    ##  [88] "kernelpls"           "kknn"                "knn"                
    ##  [91] "krlsPoly"            "krlsRadial"          "lars"               
    ##  [94] "lars2"               "lasso"               "lda"                
    ##  [97] "lda2"                "leapBackward"        "leapForward"        
    ## [100] "leapSeq"             "Linda"               "lm"                 
    ## [103] "lmStepAIC"           "LMT"                 "loclda"             
    ## [106] "logicBag"            "LogitBoost"          "logreg"             
    ## [109] "lssvmLinear"         "lssvmPoly"           "lssvmRadial"        
    ## [112] "lvq"                 "M5"                  "M5Rules"            
    ## [115] "manb"                "mda"                 "Mlda"               
    ## [118] "mlp"                 "mlpML"               "mlpSGD"             
    ## [121] "mlpWeightDecay"      "mlpWeightDecayML"    "multinom"           
    ## [124] "nb"                  "nbDiscrete"          "nbSearch"           
    ## [127] "neuralnet"           "nnet"                "nnls"               
    ## [130] "nodeHarvest"         "oblique.tree"        "OneR"               
    ## [133] "ordinalNet"          "ORFlog"              "ORFpls"             
    ## [136] "ORFridge"            "ORFsvm"              "ownn"               
    ## [139] "pam"                 "parRF"               "PART"               
    ## [142] "partDSA"             "pcaNNet"             "pcr"                
    ## [145] "pda"                 "pda2"                "penalized"          
    ## [148] "PenalizedLDA"        "plr"                 "pls"                
    ## [151] "plsRglm"             "polr"                "ppr"                
    ## [154] "protoclass"          "pythonKnnReg"        "qda"                
    ## [157] "QdaCov"              "qrf"                 "qrnn"               
    ## [160] "randomGLM"           "ranger"              "rbf"                
    ## [163] "rbfDDA"              "Rborist"             "rda"                
    ## [166] "relaxo"              "rf"                  "rFerns"             
    ## [169] "RFlda"               "rfRules"             "ridge"              
    ## [172] "rlda"                "rlm"                 "rmda"               
    ## [175] "rocc"                "rotationForest"      "rotationForestCp"   
    ## [178] "rpart"               "rpart1SE"            "rpart2"             
    ## [181] "rpartCost"           "rpartScore"          "rqlasso"            
    ## [184] "rqnc"                "RRF"                 "RRFglobal"          
    ## [187] "rrlda"               "RSimca"              "rvmLinear"          
    ## [190] "rvmPoly"             "rvmRadial"           "SBC"                
    ## [193] "sda"                 "sddaLDA"             "sddaQDA"            
    ## [196] "sdwd"                "simpls"              "SLAVE"              
    ## [199] "slda"                "smda"                "snn"                
    ## [202] "sparseLDA"           "spikeslab"           "spls"               
    ## [205] "stepLDA"             "stepQDA"             "superpc"            
    ## [208] "svmBoundrangeString" "svmExpoString"       "svmLinear"          
    ## [211] "svmLinear2"          "svmLinear3"          "svmLinearWeights"   
    ## [214] "svmLinearWeights2"   "svmPoly"             "svmRadial"          
    ## [217] "svmRadialCost"       "svmRadialSigma"      "svmRadialWeights"   
    ## [220] "svmSpectrumString"   "tan"                 "tanSearch"          
    ## [223] "treebag"             "vbmpRadial"          "vglmAdjCat"         
    ## [226] "vglmContRatio"       "vglmCumulative"      "widekernelpls"      
    ## [229] "WM"                  "wsrf"                "xgbLinear"          
    ## [232] "xgbTree"             "xyf"

lets go with the gbm and logistic model and check our performance on both the models

``` r
# gradient boosting model
model_gbm=train(trainset[,predictors],trainset[,outcomeName],method='gbm')
```

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1629             nan     0.1000    0.0236
    ##      2        1.1325             nan     0.1000    0.0157
    ##      3        1.1059             nan     0.1000    0.0137
    ##      4        1.0841             nan     0.1000    0.0078
    ##      5        1.0669             nan     0.1000    0.0081
    ##      6        1.0548             nan     0.1000    0.0063
    ##      7        1.0450             nan     0.1000    0.0058
    ##      8        1.0338             nan     0.1000    0.0042
    ##      9        1.0255             nan     0.1000    0.0028
    ##     10        1.0191             nan     0.1000    0.0030
    ##     20        0.9574             nan     0.1000    0.0018
    ##     40        0.8924             nan     0.1000   -0.0008
    ##     60        0.8504             nan     0.1000   -0.0012
    ##     80        0.8227             nan     0.1000   -0.0019
    ##    100        0.7984             nan     0.1000   -0.0019
    ##    120        0.7765             nan     0.1000   -0.0023
    ##    140        0.7590             nan     0.1000   -0.0014
    ##    150        0.7500             nan     0.1000   -0.0009
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1568             nan     0.1000    0.0252
    ##      2        1.1123             nan     0.1000    0.0194
    ##      3        1.0786             nan     0.1000    0.0153
    ##      4        1.0503             nan     0.1000    0.0118
    ##      5        1.0285             nan     0.1000    0.0092
    ##      6        1.0105             nan     0.1000    0.0078
    ##      7        0.9934             nan     0.1000    0.0026
    ##      8        0.9794             nan     0.1000    0.0054
    ##      9        0.9667             nan     0.1000    0.0058
    ##     10        0.9591             nan     0.1000    0.0018
    ##     20        0.8798             nan     0.1000    0.0016
    ##     40        0.7861             nan     0.1000   -0.0005
    ##     60        0.7240             nan     0.1000   -0.0004
    ##     80        0.6705             nan     0.1000   -0.0025
    ##    100        0.6287             nan     0.1000   -0.0008
    ##    120        0.6003             nan     0.1000   -0.0015
    ##    140        0.5614             nan     0.1000   -0.0010
    ##    150        0.5499             nan     0.1000   -0.0008
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1515             nan     0.1000    0.0230
    ##      2        1.1093             nan     0.1000    0.0246
    ##      3        1.0757             nan     0.1000    0.0160
    ##      4        1.0411             nan     0.1000    0.0156
    ##      5        1.0187             nan     0.1000    0.0093
    ##      6        0.9968             nan     0.1000    0.0100
    ##      7        0.9768             nan     0.1000    0.0065
    ##      8        0.9574             nan     0.1000    0.0065
    ##      9        0.9367             nan     0.1000    0.0070
    ##     10        0.9213             nan     0.1000    0.0068
    ##     20        0.8083             nan     0.1000   -0.0001
    ##     40        0.6922             nan     0.1000    0.0007
    ##     60        0.6073             nan     0.1000   -0.0010
    ##     80        0.5600             nan     0.1000   -0.0017
    ##    100        0.5104             nan     0.1000   -0.0021
    ##    120        0.4581             nan     0.1000    0.0005
    ##    140        0.4216             nan     0.1000   -0.0011
    ##    150        0.4044             nan     0.1000   -0.0007
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1556             nan     0.1000    0.0263
    ##      2        1.1171             nan     0.1000    0.0222
    ##      3        1.0886             nan     0.1000    0.0148
    ##      4        1.0664             nan     0.1000    0.0118
    ##      5        1.0459             nan     0.1000    0.0100
    ##      6        1.0298             nan     0.1000    0.0077
    ##      7        1.0179             nan     0.1000    0.0066
    ##      8        1.0066             nan     0.1000    0.0056
    ##      9        1.0031             nan     0.1000   -0.0007
    ##     10        0.9929             nan     0.1000    0.0041
    ##     20        0.9435             nan     0.1000   -0.0037
    ##     40        0.8767             nan     0.1000   -0.0011
    ##     60        0.8412             nan     0.1000   -0.0010
    ##     80        0.8108             nan     0.1000    0.0001
    ##    100        0.7879             nan     0.1000    0.0005
    ##    120        0.7705             nan     0.1000   -0.0002
    ##    140        0.7531             nan     0.1000   -0.0007
    ##    150        0.7458             nan     0.1000   -0.0007
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1546             nan     0.1000    0.0285
    ##      2        1.1118             nan     0.1000    0.0221
    ##      3        1.0762             nan     0.1000    0.0211
    ##      4        1.0442             nan     0.1000    0.0137
    ##      5        1.0207             nan     0.1000    0.0108
    ##      6        1.0004             nan     0.1000    0.0071
    ##      7        0.9852             nan     0.1000    0.0057
    ##      8        0.9684             nan     0.1000    0.0045
    ##      9        0.9561             nan     0.1000    0.0050
    ##     10        0.9460             nan     0.1000    0.0037
    ##     20        0.8697             nan     0.1000    0.0016
    ##     40        0.7720             nan     0.1000   -0.0001
    ##     60        0.7103             nan     0.1000   -0.0001
    ##     80        0.6627             nan     0.1000   -0.0010
    ##    100        0.6271             nan     0.1000    0.0000
    ##    120        0.5941             nan     0.1000   -0.0027
    ##    140        0.5598             nan     0.1000   -0.0003
    ##    150        0.5463             nan     0.1000   -0.0010
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1458             nan     0.1000    0.0332
    ##      2        1.0990             nan     0.1000    0.0219
    ##      3        1.0638             nan     0.1000    0.0175
    ##      4        1.0390             nan     0.1000    0.0083
    ##      5        1.0152             nan     0.1000    0.0072
    ##      6        0.9895             nan     0.1000    0.0104
    ##      7        0.9655             nan     0.1000    0.0077
    ##      8        0.9463             nan     0.1000    0.0080
    ##      9        0.9305             nan     0.1000    0.0060
    ##     10        0.9160             nan     0.1000    0.0047
    ##     20        0.8215             nan     0.1000   -0.0022
    ##     40        0.7086             nan     0.1000   -0.0014
    ##     60        0.6322             nan     0.1000   -0.0035
    ##     80        0.5676             nan     0.1000    0.0003
    ##    100        0.5136             nan     0.1000   -0.0006
    ##    120        0.4752             nan     0.1000   -0.0015
    ##    140        0.4414             nan     0.1000   -0.0015
    ##    150        0.4277             nan     0.1000   -0.0008
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1718             nan     0.1000    0.0256
    ##      2        1.1355             nan     0.1000    0.0187
    ##      3        1.1093             nan     0.1000    0.0146
    ##      4        1.0858             nan     0.1000    0.0120
    ##      5        1.0648             nan     0.1000    0.0081
    ##      6        1.0500             nan     0.1000    0.0079
    ##      7        1.0373             nan     0.1000    0.0060
    ##      8        1.0276             nan     0.1000    0.0054
    ##      9        1.0181             nan     0.1000    0.0046
    ##     10        1.0110             nan     0.1000    0.0042
    ##     20        0.9635             nan     0.1000    0.0009
    ##     40        0.9186             nan     0.1000   -0.0013
    ##     60        0.8836             nan     0.1000   -0.0005
    ##     80        0.8568             nan     0.1000   -0.0012
    ##    100        0.8356             nan     0.1000   -0.0020
    ##    120        0.8164             nan     0.1000   -0.0013
    ##    140        0.8000             nan     0.1000   -0.0004
    ##    150        0.7914             nan     0.1000   -0.0012
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1645             nan     0.1000    0.0278
    ##      2        1.1299             nan     0.1000    0.0172
    ##      3        1.0938             nan     0.1000    0.0181
    ##      4        1.0677             nan     0.1000    0.0132
    ##      5        1.0482             nan     0.1000    0.0075
    ##      6        1.0301             nan     0.1000    0.0094
    ##      7        1.0136             nan     0.1000    0.0060
    ##      8        1.0001             nan     0.1000    0.0048
    ##      9        0.9940             nan     0.1000   -0.0004
    ##     10        0.9846             nan     0.1000    0.0032
    ##     20        0.9113             nan     0.1000   -0.0019
    ##     40        0.8287             nan     0.1000    0.0003
    ##     60        0.7748             nan     0.1000   -0.0007
    ##     80        0.7150             nan     0.1000   -0.0010
    ##    100        0.6668             nan     0.1000   -0.0009
    ##    120        0.6263             nan     0.1000   -0.0010
    ##    140        0.5909             nan     0.1000   -0.0022
    ##    150        0.5705             nan     0.1000   -0.0012
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1663             nan     0.1000    0.0271
    ##      2        1.1166             nan     0.1000    0.0206
    ##      3        1.0764             nan     0.1000    0.0139
    ##      4        1.0449             nan     0.1000    0.0151
    ##      5        1.0210             nan     0.1000    0.0094
    ##      6        1.0001             nan     0.1000    0.0108
    ##      7        0.9837             nan     0.1000    0.0067
    ##      8        0.9673             nan     0.1000    0.0046
    ##      9        0.9531             nan     0.1000    0.0056
    ##     10        0.9403             nan     0.1000    0.0032
    ##     20        0.8483             nan     0.1000   -0.0001
    ##     40        0.7491             nan     0.1000   -0.0014
    ##     60        0.6694             nan     0.1000    0.0010
    ##     80        0.5885             nan     0.1000    0.0002
    ##    100        0.5221             nan     0.1000   -0.0016
    ##    120        0.4795             nan     0.1000   -0.0010
    ##    140        0.4391             nan     0.1000   -0.0026
    ##    150        0.4249             nan     0.1000    0.0003
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1264             nan     0.1000    0.0270
    ##      2        1.0865             nan     0.1000    0.0174
    ##      3        1.0604             nan     0.1000    0.0142
    ##      4        1.0388             nan     0.1000    0.0086
    ##      5        1.0242             nan     0.1000    0.0082
    ##      6        1.0116             nan     0.1000    0.0054
    ##      7        0.9990             nan     0.1000    0.0051
    ##      8        0.9900             nan     0.1000    0.0045
    ##      9        0.9802             nan     0.1000    0.0040
    ##     10        0.9701             nan     0.1000    0.0026
    ##     20        0.9115             nan     0.1000    0.0018
    ##     40        0.8468             nan     0.1000    0.0004
    ##     60        0.8109             nan     0.1000   -0.0000
    ##     80        0.7866             nan     0.1000    0.0003
    ##    100        0.7683             nan     0.1000   -0.0002
    ##    120        0.7497             nan     0.1000   -0.0009
    ##    140        0.7327             nan     0.1000   -0.0017
    ##    150        0.7243             nan     0.1000   -0.0006
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1171             nan     0.1000    0.0272
    ##      2        1.0789             nan     0.1000    0.0245
    ##      3        1.0451             nan     0.1000    0.0176
    ##      4        1.0094             nan     0.1000    0.0155
    ##      5        0.9845             nan     0.1000    0.0112
    ##      6        0.9638             nan     0.1000    0.0089
    ##      7        0.9494             nan     0.1000    0.0008
    ##      8        0.9392             nan     0.1000    0.0037
    ##      9        0.9223             nan     0.1000    0.0061
    ##     10        0.9104             nan     0.1000    0.0017
    ##     20        0.8363             nan     0.1000   -0.0032
    ##     40        0.7545             nan     0.1000   -0.0001
    ##     60        0.6979             nan     0.1000   -0.0004
    ##     80        0.6502             nan     0.1000   -0.0011
    ##    100        0.6169             nan     0.1000   -0.0004
    ##    120        0.5776             nan     0.1000    0.0000
    ##    140        0.5507             nan     0.1000   -0.0010
    ##    150        0.5410             nan     0.1000   -0.0024
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1093             nan     0.1000    0.0311
    ##      2        1.0595             nan     0.1000    0.0215
    ##      3        1.0189             nan     0.1000    0.0169
    ##      4        0.9896             nan     0.1000    0.0100
    ##      5        0.9652             nan     0.1000    0.0104
    ##      6        0.9417             nan     0.1000    0.0093
    ##      7        0.9217             nan     0.1000    0.0077
    ##      8        0.9054             nan     0.1000    0.0080
    ##      9        0.8895             nan     0.1000    0.0044
    ##     10        0.8766             nan     0.1000    0.0032
    ##     20        0.7973             nan     0.1000    0.0025
    ##     40        0.7049             nan     0.1000   -0.0010
    ##     60        0.6165             nan     0.1000   -0.0008
    ##     80        0.5529             nan     0.1000   -0.0008
    ##    100        0.5025             nan     0.1000   -0.0008
    ##    120        0.4663             nan     0.1000   -0.0012
    ##    140        0.4291             nan     0.1000    0.0007
    ##    150        0.4116             nan     0.1000   -0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1305             nan     0.1000    0.0358
    ##      2        1.0899             nan     0.1000    0.0181
    ##      3        1.0534             nan     0.1000    0.0176
    ##      4        1.0249             nan     0.1000    0.0120
    ##      5        1.0029             nan     0.1000    0.0111
    ##      6        0.9835             nan     0.1000    0.0085
    ##      7        0.9698             nan     0.1000    0.0069
    ##      8        0.9577             nan     0.1000    0.0054
    ##      9        0.9471             nan     0.1000    0.0056
    ##     10        0.9378             nan     0.1000    0.0044
    ##     20        0.8881             nan     0.1000   -0.0011
    ##     40        0.8286             nan     0.1000    0.0007
    ##     60        0.7904             nan     0.1000   -0.0004
    ##     80        0.7696             nan     0.1000   -0.0000
    ##    100        0.7491             nan     0.1000   -0.0009
    ##    120        0.7288             nan     0.1000   -0.0006
    ##    140        0.7117             nan     0.1000   -0.0001
    ##    150        0.7068             nan     0.1000   -0.0016
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1285             nan     0.1000    0.0319
    ##      2        1.0824             nan     0.1000    0.0260
    ##      3        1.0410             nan     0.1000    0.0212
    ##      4        1.0086             nan     0.1000    0.0139
    ##      5        0.9841             nan     0.1000    0.0121
    ##      6        0.9609             nan     0.1000    0.0104
    ##      7        0.9436             nan     0.1000    0.0066
    ##      8        0.9293             nan     0.1000    0.0043
    ##      9        0.9156             nan     0.1000    0.0034
    ##     10        0.9045             nan     0.1000    0.0042
    ##     20        0.8356             nan     0.1000    0.0017
    ##     40        0.7441             nan     0.1000   -0.0019
    ##     60        0.6907             nan     0.1000   -0.0004
    ##     80        0.6373             nan     0.1000    0.0003
    ##    100        0.5991             nan     0.1000   -0.0012
    ##    120        0.5607             nan     0.1000   -0.0006
    ##    140        0.5256             nan     0.1000   -0.0007
    ##    150        0.5102             nan     0.1000   -0.0018
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1273             nan     0.1000    0.0332
    ##      2        1.0751             nan     0.1000    0.0251
    ##      3        1.0361             nan     0.1000    0.0138
    ##      4        0.9996             nan     0.1000    0.0184
    ##      5        0.9653             nan     0.1000    0.0157
    ##      6        0.9416             nan     0.1000    0.0079
    ##      7        0.9191             nan     0.1000    0.0089
    ##      8        0.9025             nan     0.1000    0.0059
    ##      9        0.8899             nan     0.1000    0.0064
    ##     10        0.8739             nan     0.1000    0.0056
    ##     20        0.7764             nan     0.1000   -0.0007
    ##     40        0.6731             nan     0.1000   -0.0012
    ##     60        0.5997             nan     0.1000   -0.0022
    ##     80        0.5347             nan     0.1000   -0.0007
    ##    100        0.4820             nan     0.1000   -0.0016
    ##    120        0.4378             nan     0.1000   -0.0011
    ##    140        0.4001             nan     0.1000   -0.0005
    ##    150        0.3838             nan     0.1000   -0.0005
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2192             nan     0.1000    0.0224
    ##      2        1.1834             nan     0.1000    0.0200
    ##      3        1.1560             nan     0.1000    0.0130
    ##      4        1.1382             nan     0.1000    0.0110
    ##      5        1.1190             nan     0.1000    0.0091
    ##      6        1.1019             nan     0.1000    0.0080
    ##      7        1.0889             nan     0.1000    0.0072
    ##      8        1.0757             nan     0.1000    0.0028
    ##      9        1.0671             nan     0.1000    0.0042
    ##     10        1.0627             nan     0.1000    0.0003
    ##     20        1.0105             nan     0.1000   -0.0011
    ##     40        0.9560             nan     0.1000   -0.0001
    ##     60        0.9185             nan     0.1000   -0.0011
    ##     80        0.8926             nan     0.1000   -0.0009
    ##    100        0.8754             nan     0.1000   -0.0024
    ##    120        0.8605             nan     0.1000   -0.0006
    ##    140        0.8502             nan     0.1000   -0.0005
    ##    150        0.8425             nan     0.1000   -0.0022
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2248             nan     0.1000    0.0239
    ##      2        1.1807             nan     0.1000    0.0181
    ##      3        1.1447             nan     0.1000    0.0156
    ##      4        1.1189             nan     0.1000    0.0098
    ##      5        1.0931             nan     0.1000    0.0111
    ##      6        1.0753             nan     0.1000    0.0066
    ##      7        1.0596             nan     0.1000    0.0069
    ##      8        1.0447             nan     0.1000    0.0058
    ##      9        1.0310             nan     0.1000    0.0045
    ##     10        1.0178             nan     0.1000    0.0034
    ##     20        0.9542             nan     0.1000    0.0005
    ##     40        0.8825             nan     0.1000   -0.0012
    ##     60        0.8310             nan     0.1000   -0.0011
    ##     80        0.7788             nan     0.1000   -0.0026
    ##    100        0.7339             nan     0.1000   -0.0006
    ##    120        0.6985             nan     0.1000   -0.0019
    ##    140        0.6648             nan     0.1000   -0.0009
    ##    150        0.6506             nan     0.1000   -0.0010
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2201             nan     0.1000    0.0278
    ##      2        1.1763             nan     0.1000    0.0182
    ##      3        1.1419             nan     0.1000    0.0147
    ##      4        1.1118             nan     0.1000    0.0155
    ##      5        1.0830             nan     0.1000    0.0099
    ##      6        1.0621             nan     0.1000    0.0069
    ##      7        1.0419             nan     0.1000    0.0064
    ##      8        1.0234             nan     0.1000    0.0042
    ##      9        1.0091             nan     0.1000    0.0063
    ##     10        0.9922             nan     0.1000    0.0011
    ##     20        0.9046             nan     0.1000   -0.0021
    ##     40        0.7968             nan     0.1000   -0.0033
    ##     60        0.7308             nan     0.1000    0.0012
    ##     80        0.6752             nan     0.1000   -0.0027
    ##    100        0.6267             nan     0.1000   -0.0021
    ##    120        0.5836             nan     0.1000   -0.0027
    ##    140        0.5400             nan     0.1000   -0.0024
    ##    150        0.5206             nan     0.1000   -0.0020
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2158             nan     0.1000    0.0291
    ##      2        1.1683             nan     0.1000    0.0227
    ##      3        1.1340             nan     0.1000    0.0186
    ##      4        1.1076             nan     0.1000    0.0139
    ##      5        1.0823             nan     0.1000    0.0093
    ##      6        1.0614             nan     0.1000    0.0089
    ##      7        1.0445             nan     0.1000    0.0079
    ##      8        1.0291             nan     0.1000    0.0071
    ##      9        1.0163             nan     0.1000    0.0049
    ##     10        1.0065             nan     0.1000    0.0016
    ##     20        0.9461             nan     0.1000   -0.0012
    ##     40        0.8833             nan     0.1000    0.0001
    ##     60        0.8400             nan     0.1000   -0.0002
    ##     80        0.8157             nan     0.1000   -0.0006
    ##    100        0.7941             nan     0.1000   -0.0011
    ##    120        0.7743             nan     0.1000   -0.0009
    ##    140        0.7611             nan     0.1000   -0.0009
    ##    150        0.7546             nan     0.1000   -0.0014
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2074             nan     0.1000    0.0264
    ##      2        1.1595             nan     0.1000    0.0211
    ##      3        1.1177             nan     0.1000    0.0173
    ##      4        1.0844             nan     0.1000    0.0148
    ##      5        1.0563             nan     0.1000    0.0100
    ##      6        1.0329             nan     0.1000    0.0102
    ##      7        1.0135             nan     0.1000    0.0043
    ##      8        0.9942             nan     0.1000    0.0084
    ##      9        0.9750             nan     0.1000    0.0053
    ##     10        0.9631             nan     0.1000    0.0049
    ##     20        0.8855             nan     0.1000    0.0008
    ##     40        0.7951             nan     0.1000   -0.0007
    ##     60        0.7323             nan     0.1000   -0.0019
    ##     80        0.6834             nan     0.1000   -0.0007
    ##    100        0.6436             nan     0.1000   -0.0014
    ##    120        0.6016             nan     0.1000   -0.0004
    ##    140        0.5642             nan     0.1000   -0.0023
    ##    150        0.5501             nan     0.1000   -0.0019
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2021             nan     0.1000    0.0298
    ##      2        1.1432             nan     0.1000    0.0255
    ##      3        1.1026             nan     0.1000    0.0211
    ##      4        1.0672             nan     0.1000    0.0132
    ##      5        1.0341             nan     0.1000    0.0168
    ##      6        1.0099             nan     0.1000    0.0094
    ##      7        0.9857             nan     0.1000    0.0102
    ##      8        0.9635             nan     0.1000    0.0087
    ##      9        0.9444             nan     0.1000    0.0069
    ##     10        0.9288             nan     0.1000    0.0073
    ##     20        0.8390             nan     0.1000    0.0001
    ##     40        0.7155             nan     0.1000   -0.0007
    ##     60        0.6402             nan     0.1000    0.0015
    ##     80        0.5758             nan     0.1000    0.0004
    ##    100        0.5299             nan     0.1000   -0.0014
    ##    120        0.4828             nan     0.1000   -0.0025
    ##    140        0.4468             nan     0.1000    0.0000
    ##    150        0.4254             nan     0.1000   -0.0014
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2163             nan     0.1000    0.0225
    ##      2        1.1826             nan     0.1000    0.0195
    ##      3        1.1538             nan     0.1000    0.0119
    ##      4        1.1314             nan     0.1000    0.0081
    ##      5        1.1149             nan     0.1000    0.0077
    ##      6        1.1014             nan     0.1000    0.0073
    ##      7        1.0899             nan     0.1000    0.0050
    ##      8        1.0790             nan     0.1000    0.0049
    ##      9        1.0699             nan     0.1000    0.0038
    ##     10        1.0622             nan     0.1000    0.0023
    ##     20        1.0098             nan     0.1000    0.0015
    ##     40        0.9504             nan     0.1000    0.0006
    ##     60        0.9205             nan     0.1000   -0.0010
    ##     80        0.8998             nan     0.1000   -0.0025
    ##    100        0.8828             nan     0.1000   -0.0007
    ##    120        0.8712             nan     0.1000   -0.0019
    ##    140        0.8596             nan     0.1000   -0.0000
    ##    150        0.8538             nan     0.1000   -0.0012
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2041             nan     0.1000    0.0226
    ##      2        1.1674             nan     0.1000    0.0167
    ##      3        1.1352             nan     0.1000    0.0156
    ##      4        1.1066             nan     0.1000    0.0144
    ##      5        1.0847             nan     0.1000    0.0106
    ##      6        1.0655             nan     0.1000    0.0070
    ##      7        1.0527             nan     0.1000    0.0044
    ##      8        1.0441             nan     0.1000    0.0027
    ##      9        1.0297             nan     0.1000    0.0062
    ##     10        1.0168             nan     0.1000    0.0045
    ##     20        0.9479             nan     0.1000   -0.0011
    ##     40        0.8675             nan     0.1000   -0.0016
    ##     60        0.8077             nan     0.1000   -0.0008
    ##     80        0.7633             nan     0.1000   -0.0012
    ##    100        0.7303             nan     0.1000   -0.0016
    ##    120        0.6965             nan     0.1000   -0.0006
    ##    140        0.6685             nan     0.1000   -0.0007
    ##    150        0.6532             nan     0.1000   -0.0010
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1970             nan     0.1000    0.0251
    ##      2        1.1552             nan     0.1000    0.0174
    ##      3        1.1156             nan     0.1000    0.0155
    ##      4        1.0876             nan     0.1000    0.0119
    ##      5        1.0668             nan     0.1000    0.0070
    ##      6        1.0433             nan     0.1000    0.0083
    ##      7        1.0264             nan     0.1000    0.0062
    ##      8        1.0088             nan     0.1000    0.0077
    ##      9        0.9943             nan     0.1000    0.0037
    ##     10        0.9816             nan     0.1000    0.0013
    ##     20        0.8909             nan     0.1000   -0.0021
    ##     40        0.7815             nan     0.1000   -0.0027
    ##     60        0.7141             nan     0.1000   -0.0009
    ##     80        0.6540             nan     0.1000    0.0006
    ##    100        0.6086             nan     0.1000   -0.0007
    ##    120        0.5669             nan     0.1000   -0.0012
    ##    140        0.5299             nan     0.1000   -0.0021
    ##    150        0.5165             nan     0.1000   -0.0012
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2315             nan     0.1000    0.0220
    ##      2        1.1928             nan     0.1000    0.0185
    ##      3        1.1642             nan     0.1000    0.0142
    ##      4        1.1368             nan     0.1000    0.0107
    ##      5        1.1219             nan     0.1000    0.0072
    ##      6        1.1120             nan     0.1000    0.0014
    ##      7        1.0953             nan     0.1000    0.0068
    ##      8        1.0835             nan     0.1000    0.0053
    ##      9        1.0715             nan     0.1000    0.0058
    ##     10        1.0621             nan     0.1000    0.0046
    ##     20        1.0011             nan     0.1000    0.0018
    ##     40        0.9405             nan     0.1000   -0.0025
    ##     60        0.9106             nan     0.1000   -0.0018
    ##     80        0.8900             nan     0.1000   -0.0013
    ##    100        0.8680             nan     0.1000   -0.0016
    ##    120        0.8516             nan     0.1000   -0.0011
    ##    140        0.8362             nan     0.1000   -0.0022
    ##    150        0.8261             nan     0.1000   -0.0014
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2233             nan     0.1000    0.0230
    ##      2        1.1797             nan     0.1000    0.0223
    ##      3        1.1425             nan     0.1000    0.0161
    ##      4        1.1171             nan     0.1000    0.0127
    ##      5        1.0896             nan     0.1000    0.0101
    ##      6        1.0711             nan     0.1000    0.0082
    ##      7        1.0592             nan     0.1000    0.0050
    ##      8        1.0432             nan     0.1000    0.0070
    ##      9        1.0306             nan     0.1000    0.0063
    ##     10        1.0186             nan     0.1000    0.0044
    ##     20        0.9406             nan     0.1000    0.0014
    ##     40        0.8608             nan     0.1000   -0.0029
    ##     60        0.8086             nan     0.1000    0.0000
    ##     80        0.7578             nan     0.1000    0.0016
    ##    100        0.7216             nan     0.1000   -0.0004
    ##    120        0.6828             nan     0.1000   -0.0008
    ##    140        0.6553             nan     0.1000   -0.0018
    ##    150        0.6387             nan     0.1000   -0.0018
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2138             nan     0.1000    0.0275
    ##      2        1.1656             nan     0.1000    0.0191
    ##      3        1.1254             nan     0.1000    0.0190
    ##      4        1.0964             nan     0.1000    0.0085
    ##      5        1.0682             nan     0.1000    0.0112
    ##      6        1.0497             nan     0.1000    0.0072
    ##      7        1.0281             nan     0.1000    0.0081
    ##      8        1.0105             nan     0.1000    0.0086
    ##      9        0.9937             nan     0.1000    0.0061
    ##     10        0.9830             nan     0.1000    0.0044
    ##     20        0.8963             nan     0.1000   -0.0031
    ##     40        0.7948             nan     0.1000   -0.0005
    ##     60        0.7172             nan     0.1000   -0.0028
    ##     80        0.6613             nan     0.1000   -0.0009
    ##    100        0.6043             nan     0.1000   -0.0022
    ##    120        0.5553             nan     0.1000   -0.0008
    ##    140        0.5124             nan     0.1000   -0.0013
    ##    150        0.4941             nan     0.1000   -0.0018
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1777             nan     0.1000    0.0279
    ##      2        1.1360             nan     0.1000    0.0205
    ##      3        1.1036             nan     0.1000    0.0178
    ##      4        1.0765             nan     0.1000    0.0134
    ##      5        1.0549             nan     0.1000    0.0124
    ##      6        1.0335             nan     0.1000    0.0121
    ##      7        1.0154             nan     0.1000    0.0071
    ##      8        1.0007             nan     0.1000    0.0074
    ##      9        0.9896             nan     0.1000    0.0058
    ##     10        0.9810             nan     0.1000    0.0049
    ##     20        0.9203             nan     0.1000    0.0006
    ##     40        0.8522             nan     0.1000    0.0007
    ##     60        0.8104             nan     0.1000    0.0001
    ##     80        0.7799             nan     0.1000   -0.0020
    ##    100        0.7577             nan     0.1000   -0.0003
    ##    120        0.7391             nan     0.1000   -0.0006
    ##    140        0.7202             nan     0.1000   -0.0001
    ##    150        0.7112             nan     0.1000   -0.0018
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1773             nan     0.1000    0.0279
    ##      2        1.1247             nan     0.1000    0.0209
    ##      3        1.0870             nan     0.1000    0.0177
    ##      4        1.0531             nan     0.1000    0.0169
    ##      5        1.0234             nan     0.1000    0.0127
    ##      6        0.9991             nan     0.1000    0.0107
    ##      7        0.9807             nan     0.1000    0.0075
    ##      8        0.9630             nan     0.1000    0.0075
    ##      9        0.9510             nan     0.1000    0.0045
    ##     10        0.9344             nan     0.1000    0.0066
    ##     20        0.8524             nan     0.1000    0.0008
    ##     40        0.7533             nan     0.1000    0.0005
    ##     60        0.6851             nan     0.1000   -0.0021
    ##     80        0.6414             nan     0.1000   -0.0000
    ##    100        0.6057             nan     0.1000   -0.0025
    ##    120        0.5735             nan     0.1000   -0.0007
    ##    140        0.5396             nan     0.1000   -0.0007
    ##    150        0.5298             nan     0.1000   -0.0007
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1779             nan     0.1000    0.0223
    ##      2        1.1240             nan     0.1000    0.0240
    ##      3        1.0798             nan     0.1000    0.0176
    ##      4        1.0417             nan     0.1000    0.0164
    ##      5        1.0099             nan     0.1000    0.0152
    ##      6        0.9779             nan     0.1000    0.0128
    ##      7        0.9504             nan     0.1000    0.0124
    ##      8        0.9288             nan     0.1000    0.0086
    ##      9        0.9107             nan     0.1000    0.0074
    ##     10        0.8960             nan     0.1000    0.0057
    ##     20        0.7935             nan     0.1000    0.0024
    ##     40        0.6839             nan     0.1000    0.0003
    ##     60        0.5976             nan     0.1000    0.0004
    ##     80        0.5340             nan     0.1000   -0.0015
    ##    100        0.4858             nan     0.1000   -0.0005
    ##    120        0.4419             nan     0.1000   -0.0014
    ##    140        0.4080             nan     0.1000   -0.0016
    ##    150        0.3977             nan     0.1000   -0.0012
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1522             nan     0.1000    0.0275
    ##      2        1.1099             nan     0.1000    0.0222
    ##      3        1.0790             nan     0.1000    0.0179
    ##      4        1.0553             nan     0.1000    0.0145
    ##      5        1.0349             nan     0.1000    0.0116
    ##      6        1.0173             nan     0.1000    0.0082
    ##      7        0.9998             nan     0.1000    0.0071
    ##      8        0.9862             nan     0.1000    0.0059
    ##      9        0.9745             nan     0.1000    0.0054
    ##     10        0.9647             nan     0.1000    0.0046
    ##     20        0.9166             nan     0.1000    0.0002
    ##     40        0.8669             nan     0.1000    0.0006
    ##     60        0.8392             nan     0.1000   -0.0006
    ##     80        0.8166             nan     0.1000   -0.0003
    ##    100        0.7954             nan     0.1000   -0.0012
    ##    120        0.7778             nan     0.1000   -0.0006
    ##    140        0.7647             nan     0.1000   -0.0010
    ##    150        0.7571             nan     0.1000   -0.0010
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1414             nan     0.1000    0.0370
    ##      2        1.1005             nan     0.1000    0.0212
    ##      3        1.0630             nan     0.1000    0.0189
    ##      4        1.0297             nan     0.1000    0.0153
    ##      5        1.0075             nan     0.1000    0.0114
    ##      6        0.9882             nan     0.1000    0.0090
    ##      7        0.9688             nan     0.1000    0.0098
    ##      8        0.9520             nan     0.1000    0.0045
    ##      9        0.9396             nan     0.1000    0.0052
    ##     10        0.9260             nan     0.1000    0.0044
    ##     20        0.8620             nan     0.1000   -0.0007
    ##     40        0.7810             nan     0.1000   -0.0004
    ##     60        0.7172             nan     0.1000   -0.0000
    ##     80        0.6762             nan     0.1000   -0.0006
    ##    100        0.6388             nan     0.1000   -0.0034
    ##    120        0.6010             nan     0.1000   -0.0001
    ##    140        0.5666             nan     0.1000   -0.0020
    ##    150        0.5496             nan     0.1000   -0.0020
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1445             nan     0.1000    0.0308
    ##      2        1.0956             nan     0.1000    0.0241
    ##      3        1.0522             nan     0.1000    0.0152
    ##      4        1.0191             nan     0.1000    0.0175
    ##      5        0.9904             nan     0.1000    0.0137
    ##      6        0.9674             nan     0.1000    0.0096
    ##      7        0.9489             nan     0.1000    0.0085
    ##      8        0.9291             nan     0.1000    0.0062
    ##      9        0.9153             nan     0.1000    0.0043
    ##     10        0.8988             nan     0.1000    0.0064
    ##     20        0.8095             nan     0.1000    0.0010
    ##     40        0.7065             nan     0.1000   -0.0025
    ##     60        0.6370             nan     0.1000   -0.0011
    ##     80        0.5759             nan     0.1000   -0.0023
    ##    100        0.5173             nan     0.1000   -0.0008
    ##    120        0.4716             nan     0.1000   -0.0013
    ##    140        0.4283             nan     0.1000   -0.0009
    ##    150        0.4082             nan     0.1000    0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1657             nan     0.1000    0.0369
    ##      2        1.1147             nan     0.1000    0.0217
    ##      3        1.0822             nan     0.1000    0.0178
    ##      4        1.0541             nan     0.1000    0.0141
    ##      5        1.0305             nan     0.1000    0.0118
    ##      6        1.0106             nan     0.1000    0.0088
    ##      7        0.9980             nan     0.1000    0.0075
    ##      8        0.9936             nan     0.1000   -0.0003
    ##      9        0.9787             nan     0.1000    0.0066
    ##     10        0.9681             nan     0.1000    0.0045
    ##     20        0.9122             nan     0.1000   -0.0003
    ##     40        0.8406             nan     0.1000    0.0007
    ##     60        0.7969             nan     0.1000   -0.0005
    ##     80        0.7666             nan     0.1000   -0.0003
    ##    100        0.7461             nan     0.1000   -0.0009
    ##    120        0.7255             nan     0.1000   -0.0008
    ##    140        0.7105             nan     0.1000   -0.0005
    ##    150        0.7016             nan     0.1000   -0.0005
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1610             nan     0.1000    0.0333
    ##      2        1.1119             nan     0.1000    0.0225
    ##      3        1.0723             nan     0.1000    0.0173
    ##      4        1.0388             nan     0.1000    0.0141
    ##      5        1.0180             nan     0.1000    0.0097
    ##      6        0.9958             nan     0.1000    0.0101
    ##      7        0.9785             nan     0.1000    0.0071
    ##      8        0.9617             nan     0.1000    0.0076
    ##      9        0.9464             nan     0.1000    0.0070
    ##     10        0.9317             nan     0.1000    0.0073
    ##     20        0.8499             nan     0.1000    0.0017
    ##     40        0.7522             nan     0.1000    0.0006
    ##     60        0.7000             nan     0.1000   -0.0011
    ##     80        0.6495             nan     0.1000   -0.0001
    ##    100        0.6073             nan     0.1000   -0.0007
    ##    120        0.5682             nan     0.1000   -0.0012
    ##    140        0.5401             nan     0.1000   -0.0009
    ##    150        0.5236             nan     0.1000   -0.0008
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1595             nan     0.1000    0.0300
    ##      2        1.1000             nan     0.1000    0.0214
    ##      3        1.0593             nan     0.1000    0.0142
    ##      4        1.0238             nan     0.1000    0.0134
    ##      5        0.9944             nan     0.1000    0.0120
    ##      6        0.9730             nan     0.1000    0.0063
    ##      7        0.9515             nan     0.1000    0.0104
    ##      8        0.9271             nan     0.1000    0.0103
    ##      9        0.9091             nan     0.1000    0.0046
    ##     10        0.8950             nan     0.1000    0.0071
    ##     20        0.7992             nan     0.1000    0.0000
    ##     40        0.6834             nan     0.1000   -0.0015
    ##     60        0.6091             nan     0.1000   -0.0030
    ##     80        0.5414             nan     0.1000   -0.0001
    ##    100        0.5000             nan     0.1000   -0.0012
    ##    120        0.4603             nan     0.1000   -0.0011
    ##    140        0.4312             nan     0.1000   -0.0023
    ##    150        0.4165             nan     0.1000   -0.0014
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2222             nan     0.1000    0.0201
    ##      2        1.1947             nan     0.1000    0.0148
    ##      3        1.1672             nan     0.1000    0.0113
    ##      4        1.1494             nan     0.1000    0.0095
    ##      5        1.1324             nan     0.1000    0.0074
    ##      6        1.1202             nan     0.1000    0.0061
    ##      7        1.1096             nan     0.1000    0.0054
    ##      8        1.0986             nan     0.1000    0.0030
    ##      9        1.0909             nan     0.1000    0.0038
    ##     10        1.0848             nan     0.1000    0.0027
    ##     20        1.0403             nan     0.1000    0.0003
    ##     40        0.9957             nan     0.1000    0.0003
    ##     60        0.9627             nan     0.1000   -0.0009
    ##     80        0.9396             nan     0.1000   -0.0010
    ##    100        0.9203             nan     0.1000   -0.0002
    ##    120        0.9026             nan     0.1000   -0.0015
    ##    140        0.8850             nan     0.1000   -0.0016
    ##    150        0.8790             nan     0.1000   -0.0010
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2120             nan     0.1000    0.0245
    ##      2        1.1820             nan     0.1000    0.0140
    ##      3        1.1524             nan     0.1000    0.0130
    ##      4        1.1280             nan     0.1000    0.0084
    ##      5        1.1096             nan     0.1000    0.0059
    ##      6        1.0925             nan     0.1000    0.0067
    ##      7        1.0768             nan     0.1000    0.0074
    ##      8        1.0655             nan     0.1000    0.0029
    ##      9        1.0517             nan     0.1000    0.0056
    ##     10        1.0429             nan     0.1000    0.0022
    ##     20        0.9735             nan     0.1000    0.0014
    ##     40        0.8777             nan     0.1000    0.0003
    ##     60        0.8174             nan     0.1000   -0.0029
    ##     80        0.7719             nan     0.1000    0.0006
    ##    100        0.7306             nan     0.1000   -0.0016
    ##    120        0.6929             nan     0.1000   -0.0009
    ##    140        0.6600             nan     0.1000    0.0003
    ##    150        0.6450             nan     0.1000   -0.0004
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2127             nan     0.1000    0.0201
    ##      2        1.1782             nan     0.1000    0.0113
    ##      3        1.1439             nan     0.1000    0.0140
    ##      4        1.1136             nan     0.1000    0.0113
    ##      5        1.0912             nan     0.1000    0.0078
    ##      6        1.0723             nan     0.1000    0.0056
    ##      7        1.0540             nan     0.1000    0.0086
    ##      8        1.0443             nan     0.1000    0.0007
    ##      9        1.0272             nan     0.1000    0.0051
    ##     10        1.0151             nan     0.1000    0.0023
    ##     20        0.9243             nan     0.1000   -0.0001
    ##     40        0.8032             nan     0.1000    0.0003
    ##     60        0.7189             nan     0.1000   -0.0010
    ##     80        0.6492             nan     0.1000   -0.0002
    ##    100        0.5992             nan     0.1000   -0.0011
    ##    120        0.5496             nan     0.1000   -0.0011
    ##    140        0.5093             nan     0.1000   -0.0009
    ##    150        0.4974             nan     0.1000   -0.0024
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2426             nan     0.1000    0.0257
    ##      2        1.2058             nan     0.1000    0.0198
    ##      3        1.1761             nan     0.1000    0.0163
    ##      4        1.1465             nan     0.1000    0.0144
    ##      5        1.1252             nan     0.1000    0.0096
    ##      6        1.1104             nan     0.1000    0.0089
    ##      7        1.0956             nan     0.1000    0.0075
    ##      8        1.0830             nan     0.1000    0.0055
    ##      9        1.0714             nan     0.1000    0.0043
    ##     10        1.0614             nan     0.1000    0.0043
    ##     20        1.0121             nan     0.1000   -0.0010
    ##     40        0.9436             nan     0.1000    0.0001
    ##     60        0.9006             nan     0.1000   -0.0008
    ##     80        0.8734             nan     0.1000    0.0002
    ##    100        0.8510             nan     0.1000   -0.0011
    ##    120        0.8327             nan     0.1000    0.0007
    ##    140        0.8224             nan     0.1000   -0.0008
    ##    150        0.8164             nan     0.1000   -0.0007
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2311             nan     0.1000    0.0232
    ##      2        1.1859             nan     0.1000    0.0219
    ##      3        1.1508             nan     0.1000    0.0147
    ##      4        1.1229             nan     0.1000    0.0101
    ##      5        1.0998             nan     0.1000    0.0109
    ##      6        1.0803             nan     0.1000    0.0087
    ##      7        1.0627             nan     0.1000    0.0089
    ##      8        1.0457             nan     0.1000    0.0065
    ##      9        1.0308             nan     0.1000    0.0061
    ##     10        1.0168             nan     0.1000    0.0047
    ##     20        0.9344             nan     0.1000    0.0020
    ##     40        0.8515             nan     0.1000   -0.0021
    ##     60        0.7936             nan     0.1000    0.0000
    ##     80        0.7480             nan     0.1000   -0.0004
    ##    100        0.7156             nan     0.1000   -0.0018
    ##    120        0.6865             nan     0.1000   -0.0001
    ##    140        0.6529             nan     0.1000   -0.0005
    ##    150        0.6383             nan     0.1000   -0.0022
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2332             nan     0.1000    0.0262
    ##      2        1.1865             nan     0.1000    0.0201
    ##      3        1.1435             nan     0.1000    0.0180
    ##      4        1.1064             nan     0.1000    0.0168
    ##      5        1.0776             nan     0.1000    0.0114
    ##      6        1.0522             nan     0.1000    0.0107
    ##      7        1.0362             nan     0.1000    0.0039
    ##      8        1.0095             nan     0.1000    0.0064
    ##      9        0.9937             nan     0.1000    0.0057
    ##     10        0.9774             nan     0.1000    0.0073
    ##     20        0.8792             nan     0.1000    0.0011
    ##     40        0.7736             nan     0.1000   -0.0017
    ##     60        0.6935             nan     0.1000   -0.0004
    ##     80        0.6304             nan     0.1000   -0.0013
    ##    100        0.5796             nan     0.1000   -0.0016
    ##    120        0.5312             nan     0.1000   -0.0023
    ##    140        0.4931             nan     0.1000   -0.0012
    ##    150        0.4741             nan     0.1000   -0.0011
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1430             nan     0.1000    0.0341
    ##      2        1.0986             nan     0.1000    0.0220
    ##      3        1.0608             nan     0.1000    0.0202
    ##      4        1.0306             nan     0.1000    0.0147
    ##      5        1.0086             nan     0.1000    0.0124
    ##      6        0.9917             nan     0.1000    0.0087
    ##      7        0.9771             nan     0.1000    0.0074
    ##      8        0.9644             nan     0.1000    0.0063
    ##      9        0.9518             nan     0.1000    0.0059
    ##     10        0.9419             nan     0.1000    0.0049
    ##     20        0.8845             nan     0.1000    0.0021
    ##     40        0.8214             nan     0.1000    0.0008
    ##     60        0.7822             nan     0.1000   -0.0012
    ##     80        0.7549             nan     0.1000    0.0000
    ##    100        0.7331             nan     0.1000   -0.0019
    ##    120        0.7139             nan     0.1000   -0.0002
    ##    140        0.6997             nan     0.1000   -0.0004
    ##    150        0.6912             nan     0.1000   -0.0006
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1407             nan     0.1000    0.0350
    ##      2        1.0918             nan     0.1000    0.0269
    ##      3        1.0520             nan     0.1000    0.0221
    ##      4        1.0180             nan     0.1000    0.0147
    ##      5        0.9971             nan     0.1000    0.0087
    ##      6        0.9701             nan     0.1000    0.0123
    ##      7        0.9485             nan     0.1000    0.0097
    ##      8        0.9337             nan     0.1000    0.0072
    ##      9        0.9215             nan     0.1000    0.0060
    ##     10        0.9073             nan     0.1000    0.0052
    ##     20        0.8329             nan     0.1000    0.0009
    ##     40        0.7376             nan     0.1000    0.0003
    ##     60        0.6743             nan     0.1000   -0.0004
    ##     80        0.6192             nan     0.1000   -0.0006
    ##    100        0.5773             nan     0.1000   -0.0002
    ##    120        0.5383             nan     0.1000   -0.0014
    ##    140        0.5052             nan     0.1000    0.0001
    ##    150        0.4912             nan     0.1000   -0.0007
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1327             nan     0.1000    0.0369
    ##      2        1.0802             nan     0.1000    0.0252
    ##      3        1.0393             nan     0.1000    0.0162
    ##      4        1.0073             nan     0.1000    0.0160
    ##      5        0.9780             nan     0.1000    0.0114
    ##      6        0.9549             nan     0.1000    0.0108
    ##      7        0.9248             nan     0.1000    0.0118
    ##      8        0.9061             nan     0.1000    0.0038
    ##      9        0.8922             nan     0.1000    0.0010
    ##     10        0.8787             nan     0.1000    0.0046
    ##     20        0.7816             nan     0.1000    0.0028
    ##     40        0.6563             nan     0.1000   -0.0031
    ##     60        0.5684             nan     0.1000   -0.0010
    ##     80        0.5117             nan     0.1000   -0.0027
    ##    100        0.4554             nan     0.1000   -0.0019
    ##    120        0.4184             nan     0.1000   -0.0016
    ##    140        0.3860             nan     0.1000   -0.0009
    ##    150        0.3711             nan     0.1000   -0.0003
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1823             nan     0.1000    0.0246
    ##      2        1.1439             nan     0.1000    0.0167
    ##      3        1.1157             nan     0.1000    0.0108
    ##      4        1.0912             nan     0.1000    0.0104
    ##      5        1.0743             nan     0.1000    0.0098
    ##      6        1.0557             nan     0.1000    0.0094
    ##      7        1.0511             nan     0.1000   -0.0005
    ##      8        1.0369             nan     0.1000    0.0069
    ##      9        1.0244             nan     0.1000    0.0065
    ##     10        1.0138             nan     0.1000    0.0044
    ##     20        0.9596             nan     0.1000   -0.0002
    ##     40        0.9002             nan     0.1000    0.0000
    ##     60        0.8687             nan     0.1000   -0.0003
    ##     80        0.8436             nan     0.1000   -0.0006
    ##    100        0.8194             nan     0.1000   -0.0006
    ##    120        0.7966             nan     0.1000   -0.0012
    ##    140        0.7801             nan     0.1000   -0.0000
    ##    150        0.7728             nan     0.1000   -0.0015
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1772             nan     0.1000    0.0275
    ##      2        1.1373             nan     0.1000    0.0195
    ##      3        1.1006             nan     0.1000    0.0175
    ##      4        1.0717             nan     0.1000    0.0098
    ##      5        1.0480             nan     0.1000    0.0091
    ##      6        1.0296             nan     0.1000    0.0068
    ##      7        1.0143             nan     0.1000    0.0050
    ##      8        0.9969             nan     0.1000    0.0062
    ##      9        0.9853             nan     0.1000    0.0061
    ##     10        0.9749             nan     0.1000    0.0021
    ##     20        0.8966             nan     0.1000    0.0021
    ##     40        0.7963             nan     0.1000    0.0010
    ##     60        0.7318             nan     0.1000   -0.0014
    ##     80        0.6844             nan     0.1000   -0.0010
    ##    100        0.6472             nan     0.1000   -0.0019
    ##    120        0.6087             nan     0.1000   -0.0008
    ##    140        0.5818             nan     0.1000   -0.0013
    ##    150        0.5695             nan     0.1000   -0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1846             nan     0.1000    0.0225
    ##      2        1.1389             nan     0.1000    0.0207
    ##      3        1.1000             nan     0.1000    0.0205
    ##      4        1.0671             nan     0.1000    0.0134
    ##      5        1.0372             nan     0.1000    0.0130
    ##      6        1.0120             nan     0.1000    0.0078
    ##      7        0.9918             nan     0.1000    0.0089
    ##      8        0.9752             nan     0.1000    0.0065
    ##      9        0.9628             nan     0.1000    0.0038
    ##     10        0.9498             nan     0.1000    0.0045
    ##     20        0.8489             nan     0.1000   -0.0000
    ##     40        0.7320             nan     0.1000   -0.0025
    ##     60        0.6456             nan     0.1000   -0.0015
    ##     80        0.5843             nan     0.1000   -0.0008
    ##    100        0.5317             nan     0.1000   -0.0006
    ##    120        0.4924             nan     0.1000   -0.0007
    ##    140        0.4541             nan     0.1000   -0.0009
    ##    150        0.4365             nan     0.1000   -0.0010
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1422             nan     0.1000    0.0290
    ##      2        1.0960             nan     0.1000    0.0209
    ##      3        1.0600             nan     0.1000    0.0150
    ##      4        1.0328             nan     0.1000    0.0144
    ##      5        1.0128             nan     0.1000    0.0103
    ##      6        0.9954             nan     0.1000    0.0084
    ##      7        0.9835             nan     0.1000    0.0069
    ##      8        0.9722             nan     0.1000    0.0053
    ##      9        0.9613             nan     0.1000    0.0040
    ##     10        0.9526             nan     0.1000    0.0037
    ##     20        0.9049             nan     0.1000    0.0016
    ##     40        0.8542             nan     0.1000   -0.0013
    ##     60        0.8185             nan     0.1000    0.0004
    ##     80        0.7919             nan     0.1000   -0.0007
    ##    100        0.7657             nan     0.1000   -0.0004
    ##    120        0.7477             nan     0.1000   -0.0010
    ##    140        0.7301             nan     0.1000   -0.0006
    ##    150        0.7244             nan     0.1000   -0.0000
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1440             nan     0.1000    0.0285
    ##      2        1.0946             nan     0.1000    0.0237
    ##      3        1.0635             nan     0.1000    0.0166
    ##      4        1.0300             nan     0.1000    0.0095
    ##      5        1.0027             nan     0.1000    0.0124
    ##      6        0.9806             nan     0.1000    0.0089
    ##      7        0.9594             nan     0.1000    0.0073
    ##      8        0.9420             nan     0.1000    0.0063
    ##      9        0.9292             nan     0.1000    0.0065
    ##     10        0.9162             nan     0.1000    0.0047
    ##     20        0.8427             nan     0.1000    0.0006
    ##     40        0.7507             nan     0.1000   -0.0006
    ##     60        0.6857             nan     0.1000   -0.0011
    ##     80        0.6336             nan     0.1000    0.0008
    ##    100        0.5956             nan     0.1000   -0.0001
    ##    120        0.5590             nan     0.1000   -0.0011
    ##    140        0.5281             nan     0.1000   -0.0012
    ##    150        0.5090             nan     0.1000   -0.0006
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1404             nan     0.1000    0.0262
    ##      2        1.0825             nan     0.1000    0.0241
    ##      3        1.0442             nan     0.1000    0.0198
    ##      4        1.0160             nan     0.1000    0.0140
    ##      5        0.9847             nan     0.1000    0.0121
    ##      6        0.9606             nan     0.1000    0.0115
    ##      7        0.9383             nan     0.1000    0.0107
    ##      8        0.9188             nan     0.1000    0.0069
    ##      9        0.9051             nan     0.1000    0.0073
    ##     10        0.8915             nan     0.1000    0.0049
    ##     20        0.7920             nan     0.1000   -0.0003
    ##     40        0.6753             nan     0.1000    0.0007
    ##     60        0.5943             nan     0.1000    0.0003
    ##     80        0.5375             nan     0.1000   -0.0003
    ##    100        0.4933             nan     0.1000   -0.0013
    ##    120        0.4548             nan     0.1000   -0.0022
    ##    140        0.4178             nan     0.1000   -0.0013
    ##    150        0.4035             nan     0.1000   -0.0013
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2162             nan     0.1000    0.0316
    ##      2        1.1625             nan     0.1000    0.0240
    ##      3        1.1231             nan     0.1000    0.0200
    ##      4        1.0893             nan     0.1000    0.0207
    ##      5        1.0618             nan     0.1000    0.0120
    ##      6        1.0371             nan     0.1000    0.0098
    ##      7        1.0163             nan     0.1000    0.0087
    ##      8        0.9996             nan     0.1000    0.0083
    ##      9        0.9845             nan     0.1000    0.0058
    ##     10        0.9730             nan     0.1000    0.0060
    ##     20        0.9117             nan     0.1000    0.0002
    ##     40        0.8557             nan     0.1000    0.0003
    ##     60        0.8165             nan     0.1000   -0.0019
    ##     80        0.7854             nan     0.1000   -0.0007
    ##    100        0.7644             nan     0.1000   -0.0012
    ##    120        0.7490             nan     0.1000   -0.0008
    ##    140        0.7317             nan     0.1000   -0.0017
    ##    150        0.7234             nan     0.1000   -0.0006
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2077             nan     0.1000    0.0363
    ##      2        1.1539             nan     0.1000    0.0218
    ##      3        1.1098             nan     0.1000    0.0196
    ##      4        1.0752             nan     0.1000    0.0161
    ##      5        1.0446             nan     0.1000    0.0122
    ##      6        1.0184             nan     0.1000    0.0123
    ##      7        0.9955             nan     0.1000    0.0110
    ##      8        0.9768             nan     0.1000    0.0062
    ##      9        0.9601             nan     0.1000    0.0057
    ##     10        0.9446             nan     0.1000    0.0058
    ##     20        0.8567             nan     0.1000    0.0005
    ##     40        0.7641             nan     0.1000   -0.0029
    ##     60        0.7041             nan     0.1000   -0.0003
    ##     80        0.6578             nan     0.1000    0.0009
    ##    100        0.6207             nan     0.1000    0.0001
    ##    120        0.5858             nan     0.1000    0.0009
    ##    140        0.5528             nan     0.1000   -0.0009
    ##    150        0.5414             nan     0.1000   -0.0015
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2096             nan     0.1000    0.0333
    ##      2        1.1511             nan     0.1000    0.0266
    ##      3        1.1036             nan     0.1000    0.0198
    ##      4        1.0611             nan     0.1000    0.0194
    ##      5        1.0316             nan     0.1000    0.0142
    ##      6        1.0023             nan     0.1000    0.0110
    ##      7        0.9784             nan     0.1000    0.0121
    ##      8        0.9555             nan     0.1000    0.0058
    ##      9        0.9353             nan     0.1000    0.0069
    ##     10        0.9149             nan     0.1000    0.0081
    ##     20        0.8082             nan     0.1000    0.0015
    ##     40        0.6980             nan     0.1000    0.0001
    ##     60        0.6203             nan     0.1000   -0.0006
    ##     80        0.5512             nan     0.1000   -0.0024
    ##    100        0.5046             nan     0.1000    0.0004
    ##    120        0.4598             nan     0.1000   -0.0001
    ##    140        0.4281             nan     0.1000   -0.0023
    ##    150        0.4144             nan     0.1000   -0.0012
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1453             nan     0.1000    0.0277
    ##      2        1.0988             nan     0.1000    0.0172
    ##      3        1.0631             nan     0.1000    0.0129
    ##      4        1.0345             nan     0.1000    0.0155
    ##      5        1.0129             nan     0.1000    0.0119
    ##      6        0.9953             nan     0.1000    0.0076
    ##      7        0.9821             nan     0.1000    0.0074
    ##      8        0.9680             nan     0.1000    0.0066
    ##      9        0.9569             nan     0.1000    0.0049
    ##     10        0.9456             nan     0.1000    0.0041
    ##     20        0.8900             nan     0.1000   -0.0024
    ##     40        0.8412             nan     0.1000   -0.0006
    ##     60        0.8052             nan     0.1000   -0.0008
    ##     80        0.7764             nan     0.1000   -0.0009
    ##    100        0.7606             nan     0.1000   -0.0008
    ##    120        0.7456             nan     0.1000   -0.0007
    ##    140        0.7295             nan     0.1000   -0.0013
    ##    150        0.7258             nan     0.1000   -0.0016
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1515             nan     0.1000    0.0317
    ##      2        1.0999             nan     0.1000    0.0277
    ##      3        1.0614             nan     0.1000    0.0158
    ##      4        1.0294             nan     0.1000    0.0143
    ##      5        1.0040             nan     0.1000    0.0119
    ##      6        0.9849             nan     0.1000    0.0073
    ##      7        0.9616             nan     0.1000    0.0084
    ##      8        0.9459             nan     0.1000    0.0060
    ##      9        0.9309             nan     0.1000    0.0061
    ##     10        0.9162             nan     0.1000    0.0055
    ##     20        0.8342             nan     0.1000    0.0014
    ##     40        0.7512             nan     0.1000   -0.0002
    ##     60        0.6948             nan     0.1000   -0.0016
    ##     80        0.6535             nan     0.1000   -0.0003
    ##    100        0.6256             nan     0.1000   -0.0014
    ##    120        0.5841             nan     0.1000    0.0002
    ##    140        0.5530             nan     0.1000    0.0017
    ##    150        0.5368             nan     0.1000   -0.0010
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1353             nan     0.1000    0.0342
    ##      2        1.0837             nan     0.1000    0.0187
    ##      3        1.0390             nan     0.1000    0.0211
    ##      4        1.0051             nan     0.1000    0.0129
    ##      5        0.9780             nan     0.1000    0.0108
    ##      6        0.9541             nan     0.1000    0.0114
    ##      7        0.9354             nan     0.1000    0.0079
    ##      8        0.9181             nan     0.1000    0.0091
    ##      9        0.9013             nan     0.1000    0.0057
    ##     10        0.8851             nan     0.1000    0.0051
    ##     20        0.7863             nan     0.1000    0.0009
    ##     40        0.6825             nan     0.1000   -0.0016
    ##     60        0.6125             nan     0.1000   -0.0024
    ##     80        0.5601             nan     0.1000   -0.0004
    ##    100        0.5043             nan     0.1000   -0.0016
    ##    120        0.4663             nan     0.1000   -0.0008
    ##    140        0.4263             nan     0.1000   -0.0024
    ##    150        0.4096             nan     0.1000   -0.0017
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1552             nan     0.1000    0.0311
    ##      2        1.1129             nan     0.1000    0.0195
    ##      3        1.0795             nan     0.1000    0.0130
    ##      4        1.0539             nan     0.1000    0.0127
    ##      5        1.0360             nan     0.1000    0.0111
    ##      6        1.0200             nan     0.1000    0.0078
    ##      7        1.0058             nan     0.1000    0.0066
    ##      8        0.9939             nan     0.1000    0.0057
    ##      9        0.9824             nan     0.1000    0.0050
    ##     10        0.9756             nan     0.1000    0.0036
    ##     20        0.9243             nan     0.1000    0.0014
    ##     40        0.8695             nan     0.1000    0.0005
    ##     60        0.8376             nan     0.1000   -0.0004
    ##     80        0.8147             nan     0.1000    0.0000
    ##    100        0.7930             nan     0.1000   -0.0007
    ##    120        0.7782             nan     0.1000   -0.0013
    ##    140        0.7650             nan     0.1000   -0.0011
    ##    150        0.7584             nan     0.1000   -0.0008
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1504             nan     0.1000    0.0333
    ##      2        1.1032             nan     0.1000    0.0187
    ##      3        1.0670             nan     0.1000    0.0164
    ##      4        1.0400             nan     0.1000    0.0141
    ##      5        1.0192             nan     0.1000    0.0101
    ##      6        0.9985             nan     0.1000    0.0073
    ##      7        0.9830             nan     0.1000    0.0081
    ##      8        0.9687             nan     0.1000    0.0072
    ##      9        0.9577             nan     0.1000    0.0055
    ##     10        0.9440             nan     0.1000    0.0036
    ##     20        0.8695             nan     0.1000    0.0004
    ##     40        0.7921             nan     0.1000   -0.0008
    ##     60        0.7374             nan     0.1000   -0.0007
    ##     80        0.6894             nan     0.1000   -0.0011
    ##    100        0.6475             nan     0.1000   -0.0014
    ##    120        0.6134             nan     0.1000   -0.0006
    ##    140        0.5907             nan     0.1000   -0.0008
    ##    150        0.5781             nan     0.1000   -0.0011
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1449             nan     0.1000    0.0260
    ##      2        1.0997             nan     0.1000    0.0193
    ##      3        1.0661             nan     0.1000    0.0143
    ##      4        1.0356             nan     0.1000    0.0139
    ##      5        1.0036             nan     0.1000    0.0144
    ##      6        0.9806             nan     0.1000    0.0104
    ##      7        0.9595             nan     0.1000    0.0081
    ##      8        0.9423             nan     0.1000    0.0055
    ##      9        0.9259             nan     0.1000    0.0037
    ##     10        0.9153             nan     0.1000    0.0020
    ##     20        0.8129             nan     0.1000    0.0002
    ##     40        0.7076             nan     0.1000    0.0008
    ##     60        0.6418             nan     0.1000   -0.0019
    ##     80        0.5776             nan     0.1000   -0.0035
    ##    100        0.5364             nan     0.1000   -0.0004
    ##    120        0.4998             nan     0.1000   -0.0011
    ##    140        0.4657             nan     0.1000   -0.0014
    ##    150        0.4449             nan     0.1000   -0.0009
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1464             nan     0.1000    0.0303
    ##      2        1.1096             nan     0.1000    0.0170
    ##      3        1.0820             nan     0.1000    0.0171
    ##      4        1.0575             nan     0.1000    0.0102
    ##      5        1.0385             nan     0.1000    0.0101
    ##      6        1.0201             nan     0.1000    0.0079
    ##      7        1.0059             nan     0.1000    0.0083
    ##      8        0.9941             nan     0.1000    0.0041
    ##      9        0.9845             nan     0.1000    0.0050
    ##     10        0.9772             nan     0.1000    0.0036
    ##     20        0.9264             nan     0.1000    0.0019
    ##     40        0.8762             nan     0.1000   -0.0002
    ##     60        0.8392             nan     0.1000    0.0006
    ##     80        0.8104             nan     0.1000   -0.0015
    ##    100        0.7818             nan     0.1000    0.0005
    ##    120        0.7603             nan     0.1000   -0.0000
    ##    140        0.7424             nan     0.1000   -0.0011
    ##    150        0.7337             nan     0.1000   -0.0021
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1480             nan     0.1000    0.0187
    ##      2        1.1096             nan     0.1000    0.0210
    ##      3        1.0735             nan     0.1000    0.0149
    ##      4        1.0459             nan     0.1000    0.0147
    ##      5        1.0225             nan     0.1000    0.0107
    ##      6        1.0021             nan     0.1000    0.0089
    ##      7        0.9848             nan     0.1000    0.0080
    ##      8        0.9710             nan     0.1000    0.0050
    ##      9        0.9580             nan     0.1000    0.0053
    ##     10        0.9451             nan     0.1000    0.0038
    ##     20        0.8780             nan     0.1000    0.0014
    ##     40        0.7940             nan     0.1000   -0.0023
    ##     60        0.7271             nan     0.1000   -0.0009
    ##     80        0.6776             nan     0.1000    0.0004
    ##    100        0.6396             nan     0.1000   -0.0005
    ##    120        0.6034             nan     0.1000   -0.0008
    ##    140        0.5771             nan     0.1000   -0.0019
    ##    150        0.5598             nan     0.1000   -0.0019
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1418             nan     0.1000    0.0229
    ##      2        1.0957             nan     0.1000    0.0164
    ##      3        1.0601             nan     0.1000    0.0170
    ##      4        1.0295             nan     0.1000    0.0128
    ##      5        1.0030             nan     0.1000    0.0123
    ##      6        0.9823             nan     0.1000    0.0072
    ##      7        0.9638             nan     0.1000    0.0087
    ##      8        0.9421             nan     0.1000    0.0093
    ##      9        0.9242             nan     0.1000    0.0057
    ##     10        0.9133             nan     0.1000    0.0028
    ##     20        0.8242             nan     0.1000    0.0003
    ##     40        0.7145             nan     0.1000    0.0006
    ##     60        0.6454             nan     0.1000   -0.0026
    ##     80        0.5747             nan     0.1000   -0.0012
    ##    100        0.5197             nan     0.1000   -0.0006
    ##    120        0.4804             nan     0.1000   -0.0016
    ##    140        0.4417             nan     0.1000   -0.0014
    ##    150        0.4273             nan     0.1000   -0.0009
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2098             nan     0.1000    0.0238
    ##      2        1.1764             nan     0.1000    0.0176
    ##      3        1.1534             nan     0.1000    0.0131
    ##      4        1.1324             nan     0.1000    0.0115
    ##      5        1.1171             nan     0.1000    0.0082
    ##      6        1.0983             nan     0.1000    0.0070
    ##      7        1.0841             nan     0.1000    0.0073
    ##      8        1.0736             nan     0.1000    0.0048
    ##      9        1.0659             nan     0.1000    0.0038
    ##     10        1.0586             nan     0.1000    0.0027
    ##     20        1.0199             nan     0.1000   -0.0010
    ##     40        0.9577             nan     0.1000    0.0014
    ##     60        0.9166             nan     0.1000   -0.0004
    ##     80        0.8829             nan     0.1000   -0.0006
    ##    100        0.8618             nan     0.1000   -0.0003
    ##    120        0.8438             nan     0.1000   -0.0005
    ##    140        0.8233             nan     0.1000   -0.0013
    ##    150        0.8133             nan     0.1000   -0.0005
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2013             nan     0.1000    0.0231
    ##      2        1.1608             nan     0.1000    0.0181
    ##      3        1.1302             nan     0.1000    0.0134
    ##      4        1.1069             nan     0.1000    0.0098
    ##      5        1.0845             nan     0.1000    0.0085
    ##      6        1.0708             nan     0.1000    0.0052
    ##      7        1.0554             nan     0.1000    0.0059
    ##      8        1.0424             nan     0.1000    0.0045
    ##      9        1.0295             nan     0.1000    0.0047
    ##     10        1.0191             nan     0.1000    0.0043
    ##     20        0.9459             nan     0.1000    0.0030
    ##     40        0.8467             nan     0.1000   -0.0011
    ##     60        0.7770             nan     0.1000   -0.0010
    ##     80        0.7290             nan     0.1000    0.0009
    ##    100        0.6819             nan     0.1000   -0.0011
    ##    120        0.6470             nan     0.1000    0.0001
    ##    140        0.6096             nan     0.1000   -0.0010
    ##    150        0.5953             nan     0.1000   -0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2012             nan     0.1000    0.0189
    ##      2        1.1616             nan     0.1000    0.0139
    ##      3        1.1280             nan     0.1000    0.0148
    ##      4        1.1003             nan     0.1000    0.0117
    ##      5        1.0718             nan     0.1000    0.0092
    ##      6        1.0551             nan     0.1000    0.0068
    ##      7        1.0371             nan     0.1000    0.0047
    ##      8        1.0120             nan     0.1000    0.0068
    ##      9        0.9989             nan     0.1000    0.0047
    ##     10        0.9894             nan     0.1000   -0.0003
    ##     20        0.8862             nan     0.1000    0.0020
    ##     40        0.7595             nan     0.1000    0.0008
    ##     60        0.6831             nan     0.1000   -0.0002
    ##     80        0.6237             nan     0.1000   -0.0013
    ##    100        0.5694             nan     0.1000   -0.0018
    ##    120        0.5183             nan     0.1000   -0.0019
    ##    140        0.4816             nan     0.1000   -0.0011
    ##    150        0.4630             nan     0.1000   -0.0005
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2243             nan     0.1000    0.0210
    ##      2        1.1845             nan     0.1000    0.0215
    ##      3        1.1497             nan     0.1000    0.0134
    ##      4        1.1267             nan     0.1000    0.0145
    ##      5        1.1070             nan     0.1000    0.0100
    ##      6        1.0886             nan     0.1000    0.0090
    ##      7        1.0748             nan     0.1000    0.0055
    ##      8        1.0621             nan     0.1000    0.0066
    ##      9        1.0529             nan     0.1000    0.0055
    ##     10        1.0417             nan     0.1000    0.0031
    ##     20        0.9929             nan     0.1000   -0.0004
    ##     40        0.9291             nan     0.1000   -0.0005
    ##     60        0.8934             nan     0.1000    0.0006
    ##     80        0.8636             nan     0.1000    0.0003
    ##    100        0.8453             nan     0.1000   -0.0004
    ##    120        0.8231             nan     0.1000   -0.0012
    ##    140        0.8095             nan     0.1000   -0.0009
    ##    150        0.7986             nan     0.1000   -0.0005
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2219             nan     0.1000    0.0245
    ##      2        1.1749             nan     0.1000    0.0244
    ##      3        1.1367             nan     0.1000    0.0190
    ##      4        1.1079             nan     0.1000    0.0110
    ##      5        1.0832             nan     0.1000    0.0096
    ##      6        1.0641             nan     0.1000    0.0079
    ##      7        1.0455             nan     0.1000    0.0088
    ##      8        1.0290             nan     0.1000    0.0079
    ##      9        1.0151             nan     0.1000    0.0056
    ##     10        1.0024             nan     0.1000    0.0048
    ##     20        0.9230             nan     0.1000    0.0008
    ##     40        0.8376             nan     0.1000    0.0017
    ##     60        0.7636             nan     0.1000   -0.0010
    ##     80        0.7106             nan     0.1000   -0.0005
    ##    100        0.6620             nan     0.1000   -0.0021
    ##    120        0.6264             nan     0.1000    0.0012
    ##    140        0.5914             nan     0.1000   -0.0025
    ##    150        0.5769             nan     0.1000   -0.0008
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.2149             nan     0.1000    0.0299
    ##      2        1.1662             nan     0.1000    0.0218
    ##      3        1.1263             nan     0.1000    0.0176
    ##      4        1.0931             nan     0.1000    0.0159
    ##      5        1.0655             nan     0.1000    0.0098
    ##      6        1.0394             nan     0.1000    0.0109
    ##      7        1.0168             nan     0.1000    0.0097
    ##      8        0.9999             nan     0.1000    0.0047
    ##      9        0.9854             nan     0.1000    0.0053
    ##     10        0.9708             nan     0.1000    0.0053
    ##     20        0.8625             nan     0.1000    0.0018
    ##     40        0.7401             nan     0.1000   -0.0007
    ##     60        0.6599             nan     0.1000   -0.0018
    ##     80        0.6004             nan     0.1000   -0.0013
    ##    100        0.5530             nan     0.1000   -0.0004
    ##    120        0.5051             nan     0.1000   -0.0006
    ##    140        0.4597             nan     0.1000   -0.0007
    ##    150        0.4417             nan     0.1000   -0.0020
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1516             nan     0.1000    0.0328
    ##      2        1.1076             nan     0.1000    0.0219
    ##      3        1.0778             nan     0.1000    0.0148
    ##      4        1.0458             nan     0.1000    0.0136
    ##      5        1.0237             nan     0.1000    0.0120
    ##      6        1.0029             nan     0.1000    0.0100
    ##      7        0.9860             nan     0.1000    0.0061
    ##      8        0.9723             nan     0.1000    0.0077
    ##      9        0.9616             nan     0.1000    0.0050
    ##     10        0.9500             nan     0.1000    0.0042
    ##     20        0.8864             nan     0.1000    0.0015
    ##     40        0.8326             nan     0.1000    0.0003
    ##     60        0.7973             nan     0.1000    0.0003
    ##     80        0.7698             nan     0.1000   -0.0006
    ##    100        0.7508             nan     0.1000   -0.0014
    ##    120        0.7333             nan     0.1000   -0.0019
    ##    140        0.7211             nan     0.1000   -0.0014
    ##    150        0.7126             nan     0.1000   -0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1496             nan     0.1000    0.0296
    ##      2        1.1004             nan     0.1000    0.0217
    ##      3        1.0546             nan     0.1000    0.0270
    ##      4        1.0222             nan     0.1000    0.0128
    ##      5        0.9933             nan     0.1000    0.0140
    ##      6        0.9713             nan     0.1000    0.0080
    ##      7        0.9518             nan     0.1000    0.0111
    ##      8        0.9351             nan     0.1000    0.0070
    ##      9        0.9174             nan     0.1000    0.0058
    ##     10        0.9037             nan     0.1000    0.0073
    ##     20        0.8386             nan     0.1000   -0.0001
    ##     40        0.7510             nan     0.1000   -0.0008
    ##     60        0.6991             nan     0.1000   -0.0011
    ##     80        0.6518             nan     0.1000   -0.0009
    ##    100        0.6110             nan     0.1000   -0.0014
    ##    120        0.5737             nan     0.1000   -0.0009
    ##    140        0.5485             nan     0.1000   -0.0024
    ##    150        0.5308             nan     0.1000   -0.0014
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1475             nan     0.1000    0.0307
    ##      2        1.0893             nan     0.1000    0.0232
    ##      3        1.0445             nan     0.1000    0.0180
    ##      4        1.0032             nan     0.1000    0.0162
    ##      5        0.9716             nan     0.1000    0.0117
    ##      6        0.9415             nan     0.1000    0.0106
    ##      7        0.9194             nan     0.1000    0.0087
    ##      8        0.9003             nan     0.1000    0.0072
    ##      9        0.8810             nan     0.1000    0.0075
    ##     10        0.8628             nan     0.1000    0.0061
    ##     20        0.7747             nan     0.1000   -0.0006
    ##     40        0.6713             nan     0.1000   -0.0005
    ##     60        0.6076             nan     0.1000   -0.0019
    ##     80        0.5582             nan     0.1000    0.0020
    ##    100        0.5024             nan     0.1000   -0.0008
    ##    120        0.4631             nan     0.1000   -0.0009
    ##    140        0.4237             nan     0.1000   -0.0014
    ##    150        0.4085             nan     0.1000   -0.0007
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1358             nan     0.1000    0.0231
    ##      2        1.1019             nan     0.1000    0.0180
    ##      3        1.0749             nan     0.1000    0.0130
    ##      4        1.0510             nan     0.1000    0.0096
    ##      5        1.0357             nan     0.1000    0.0080
    ##      6        1.0221             nan     0.1000    0.0071
    ##      7        1.0151             nan     0.1000    0.0015
    ##      8        1.0045             nan     0.1000    0.0058
    ##      9        0.9932             nan     0.1000    0.0041
    ##     10        0.9832             nan     0.1000    0.0029
    ##     20        0.9266             nan     0.1000    0.0010
    ##     40        0.8584             nan     0.1000   -0.0005
    ##     60        0.8243             nan     0.1000   -0.0012
    ##     80        0.8011             nan     0.1000   -0.0004
    ##    100        0.7839             nan     0.1000   -0.0017
    ##    120        0.7677             nan     0.1000   -0.0009
    ##    140        0.7520             nan     0.1000   -0.0001
    ##    150        0.7449             nan     0.1000   -0.0012
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1259             nan     0.1000    0.0271
    ##      2        1.0864             nan     0.1000    0.0203
    ##      3        1.0470             nan     0.1000    0.0152
    ##      4        1.0199             nan     0.1000    0.0075
    ##      5        0.9982             nan     0.1000    0.0096
    ##      6        0.9816             nan     0.1000    0.0092
    ##      7        0.9680             nan     0.1000    0.0057
    ##      8        0.9526             nan     0.1000    0.0050
    ##      9        0.9418             nan     0.1000    0.0041
    ##     10        0.9319             nan     0.1000    0.0027
    ##     20        0.8562             nan     0.1000    0.0007
    ##     40        0.7746             nan     0.1000   -0.0001
    ##     60        0.7204             nan     0.1000   -0.0006
    ##     80        0.6658             nan     0.1000   -0.0003
    ##    100        0.6263             nan     0.1000    0.0002
    ##    120        0.5876             nan     0.1000   -0.0007
    ##    140        0.5624             nan     0.1000   -0.0006
    ##    150        0.5486             nan     0.1000   -0.0008
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1164             nan     0.1000    0.0318
    ##      2        1.0706             nan     0.1000    0.0224
    ##      3        1.0303             nan     0.1000    0.0179
    ##      4        1.0031             nan     0.1000    0.0128
    ##      5        0.9816             nan     0.1000    0.0067
    ##      6        0.9637             nan     0.1000    0.0093
    ##      7        0.9441             nan     0.1000    0.0080
    ##      8        0.9245             nan     0.1000    0.0065
    ##      9        0.9098             nan     0.1000    0.0023
    ##     10        0.8983             nan     0.1000    0.0026
    ##     20        0.8006             nan     0.1000    0.0014
    ##     40        0.6963             nan     0.1000   -0.0010
    ##     60        0.6154             nan     0.1000   -0.0001
    ##     80        0.5503             nan     0.1000   -0.0009
    ##    100        0.5037             nan     0.1000   -0.0003
    ##    120        0.4623             nan     0.1000   -0.0004
    ##    140        0.4252             nan     0.1000   -0.0015
    ##    150        0.4105             nan     0.1000   -0.0014
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1883             nan     0.1000    0.0238
    ##      2        1.1514             nan     0.1000    0.0168
    ##      3        1.1188             nan     0.1000    0.0135
    ##      4        1.0959             nan     0.1000    0.0117
    ##      5        1.0732             nan     0.1000    0.0092
    ##      6        1.0580             nan     0.1000    0.0077
    ##      7        1.0455             nan     0.1000    0.0066
    ##      8        1.0353             nan     0.1000    0.0044
    ##      9        1.0238             nan     0.1000    0.0043
    ##     10        1.0149             nan     0.1000    0.0039
    ##     20        0.9684             nan     0.1000   -0.0009
    ##     40        0.9219             nan     0.1000   -0.0002
    ##     50        0.9106             nan     0.1000   -0.0019

``` r
# Logistic regression model
model_glm<-train(trainset[,predictors],trainset[,outcomeName],method='glm')
```

    ## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
    ## ifelse(type == : prediction from a rank-deficient fit may be misleading

Lets tune the parameters of our model.We will use 5-Fold cross-validation technique repeated 5 times.

``` r
fitControl= trainControl(
             method = "repeatedcv",
             number = 5, repeats = 5)

#training our model for gbm model
model_gbm=train(trainset[,predictors],trainset[,outcomeName],method='gbm',trControl=fitControl)
```

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1880             nan     0.1000    0.0234
    ##      2        1.1467             nan     0.1000    0.0228
    ##      3        1.1210             nan     0.1000    0.0138
    ##      4        1.0968             nan     0.1000    0.0127
    ##      5        1.0748             nan     0.1000    0.0100
    ##      6        1.0555             nan     0.1000    0.0079
    ##      7        1.0413             nan     0.1000    0.0078
    ##      8        1.0402             nan     0.1000   -0.0029
    ##      9        1.0298             nan     0.1000    0.0051
    ##     10        1.0193             nan     0.1000    0.0040
    ##     20        0.9701             nan     0.1000   -0.0021
    ##     40        0.9325             nan     0.1000    0.0010
    ##     60        0.9069             nan     0.1000   -0.0027
    ##     80        0.8893             nan     0.1000   -0.0018
    ##    100        0.8784             nan     0.1000   -0.0020
    ##    120        0.8695             nan     0.1000   -0.0012
    ##    140        0.8601             nan     0.1000   -0.0012
    ##    150        0.8537             nan     0.1000   -0.0016
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1852             nan     0.1000    0.0282
    ##      2        1.1409             nan     0.1000    0.0233
    ##      3        1.1039             nan     0.1000    0.0136
    ##      4        1.0743             nan     0.1000    0.0119
    ##      5        1.0518             nan     0.1000    0.0118
    ##      6        1.0339             nan     0.1000    0.0070
    ##      7        1.0157             nan     0.1000    0.0075
    ##      8        1.0002             nan     0.1000    0.0077
    ##      9        0.9890             nan     0.1000    0.0048
    ##     10        0.9787             nan     0.1000    0.0030
    ##     20        0.9228             nan     0.1000    0.0001
    ##     40        0.8641             nan     0.1000   -0.0003
    ##     60        0.8151             nan     0.1000   -0.0016
    ##     80        0.7766             nan     0.1000   -0.0020
    ##    100        0.7557             nan     0.1000   -0.0027
    ##    120        0.7352             nan     0.1000   -0.0013
    ##    140        0.7115             nan     0.1000   -0.0026
    ##    150        0.6986             nan     0.1000   -0.0016
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1889             nan     0.1000    0.0238
    ##      2        1.1419             nan     0.1000    0.0163
    ##      3        1.1013             nan     0.1000    0.0130
    ##      4        1.0735             nan     0.1000    0.0128
    ##      5        1.0474             nan     0.1000    0.0119
    ##      6        1.0272             nan     0.1000    0.0042
    ##      7        1.0083             nan     0.1000    0.0068
    ##      8        0.9895             nan     0.1000    0.0035
    ##      9        0.9755             nan     0.1000    0.0041
    ##     10        0.9605             nan     0.1000    0.0044
    ##     20        0.8845             nan     0.1000    0.0006
    ##     40        0.7974             nan     0.1000   -0.0011
    ##     60        0.7525             nan     0.1000   -0.0034
    ##     80        0.7027             nan     0.1000   -0.0026
    ##    100        0.6596             nan     0.1000   -0.0044
    ##    120        0.6286             nan     0.1000   -0.0015
    ##    140        0.5986             nan     0.1000   -0.0041
    ##    150        0.5836             nan     0.1000   -0.0026
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1918             nan     0.1000    0.0245
    ##      2        1.1579             nan     0.1000    0.0159
    ##      3        1.1270             nan     0.1000    0.0126
    ##      4        1.1048             nan     0.1000    0.0093
    ##      5        1.0851             nan     0.1000    0.0073
    ##      6        1.0681             nan     0.1000    0.0067
    ##      7        1.0561             nan     0.1000    0.0064
    ##      8        1.0461             nan     0.1000    0.0042
    ##      9        1.0381             nan     0.1000    0.0029
    ##     10        1.0299             nan     0.1000    0.0033
    ##     20        0.9808             nan     0.1000    0.0006
    ##     40        0.9301             nan     0.1000    0.0002
    ##     60        0.9033             nan     0.1000   -0.0008
    ##     80        0.8815             nan     0.1000   -0.0006
    ##    100        0.8685             nan     0.1000   -0.0012
    ##    120        0.8579             nan     0.1000   -0.0029
    ##    140        0.8512             nan     0.1000   -0.0023
    ##    150        0.8440             nan     0.1000   -0.0009
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1884             nan     0.1000    0.0271
    ##      2        1.1474             nan     0.1000    0.0137
    ##      3        1.1140             nan     0.1000    0.0124
    ##      4        1.0884             nan     0.1000    0.0070
    ##      5        1.0658             nan     0.1000    0.0054
    ##      6        1.0507             nan     0.1000    0.0068
    ##      7        1.0350             nan     0.1000    0.0049
    ##      8        1.0222             nan     0.1000    0.0058
    ##      9        1.0077             nan     0.1000    0.0053
    ##     10        0.9991             nan     0.1000    0.0032
    ##     20        0.9388             nan     0.1000    0.0008
    ##     40        0.8695             nan     0.1000   -0.0003
    ##     60        0.8221             nan     0.1000   -0.0008
    ##     80        0.7840             nan     0.1000   -0.0028
    ##    100        0.7473             nan     0.1000   -0.0030
    ##    120        0.7208             nan     0.1000   -0.0044
    ##    140        0.6987             nan     0.1000   -0.0012
    ##    150        0.6891             nan     0.1000   -0.0016
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1902             nan     0.1000    0.0253
    ##      2        1.1501             nan     0.1000    0.0195
    ##      3        1.1121             nan     0.1000    0.0124
    ##      4        1.0831             nan     0.1000    0.0076
    ##      5        1.0575             nan     0.1000    0.0101
    ##      6        1.0321             nan     0.1000    0.0074
    ##      7        1.0126             nan     0.1000    0.0035
    ##      8        0.9961             nan     0.1000    0.0038
    ##      9        0.9870             nan     0.1000    0.0021
    ##     10        0.9749             nan     0.1000    0.0031
    ##     20        0.9000             nan     0.1000    0.0003
    ##     40        0.8083             nan     0.1000   -0.0035
    ##     60        0.7467             nan     0.1000   -0.0021
    ##     80        0.6996             nan     0.1000   -0.0036
    ##    100        0.6473             nan     0.1000   -0.0039
    ##    120        0.6063             nan     0.1000   -0.0001
    ##    140        0.5730             nan     0.1000   -0.0035
    ##    150        0.5573             nan     0.1000   -0.0026
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1848             nan     0.1000    0.0225
    ##      2        1.1468             nan     0.1000    0.0180
    ##      3        1.1131             nan     0.1000    0.0120
    ##      4        1.0885             nan     0.1000    0.0125
    ##      5        1.0694             nan     0.1000    0.0114
    ##      6        1.0505             nan     0.1000    0.0087
    ##      7        1.0335             nan     0.1000    0.0067
    ##      8        1.0212             nan     0.1000    0.0064
    ##      9        1.0109             nan     0.1000    0.0053
    ##     10        1.0032             nan     0.1000   -0.0005
    ##     20        0.9494             nan     0.1000    0.0007
    ##     40        0.8999             nan     0.1000   -0.0007
    ##     60        0.8766             nan     0.1000   -0.0015
    ##     80        0.8562             nan     0.1000   -0.0022
    ##    100        0.8371             nan     0.1000   -0.0011
    ##    120        0.8265             nan     0.1000   -0.0016
    ##    140        0.8156             nan     0.1000   -0.0016
    ##    150        0.8113             nan     0.1000   -0.0016
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1823             nan     0.1000    0.0226
    ##      2        1.1357             nan     0.1000    0.0206
    ##      3        1.1032             nan     0.1000    0.0182
    ##      4        1.0758             nan     0.1000    0.0109
    ##      5        1.0531             nan     0.1000    0.0116
    ##      6        1.0322             nan     0.1000    0.0084
    ##      7        1.0134             nan     0.1000    0.0066
    ##      8        0.9963             nan     0.1000    0.0051
    ##      9        0.9801             nan     0.1000    0.0058
    ##     10        0.9702             nan     0.1000    0.0049
    ##     20        0.9043             nan     0.1000    0.0004
    ##     40        0.8483             nan     0.1000   -0.0025
    ##     60        0.8004             nan     0.1000   -0.0023
    ##     80        0.7622             nan     0.1000   -0.0039
    ##    100        0.7362             nan     0.1000   -0.0007
    ##    120        0.7123             nan     0.1000   -0.0024
    ##    140        0.6855             nan     0.1000   -0.0011
    ##    150        0.6783             nan     0.1000   -0.0019
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1793             nan     0.1000    0.0321
    ##      2        1.1388             nan     0.1000    0.0182
    ##      3        1.0989             nan     0.1000    0.0134
    ##      4        1.0598             nan     0.1000    0.0124
    ##      5        1.0374             nan     0.1000    0.0047
    ##      6        1.0141             nan     0.1000    0.0067
    ##      7        0.9930             nan     0.1000    0.0085
    ##      8        0.9737             nan     0.1000    0.0046
    ##      9        0.9580             nan     0.1000    0.0060
    ##     10        0.9432             nan     0.1000    0.0039
    ##     20        0.8700             nan     0.1000    0.0005
    ##     40        0.7822             nan     0.1000   -0.0019
    ##     60        0.7174             nan     0.1000   -0.0011
    ##     80        0.6690             nan     0.1000   -0.0015
    ##    100        0.6358             nan     0.1000   -0.0034
    ##    120        0.6040             nan     0.1000   -0.0007
    ##    140        0.5674             nan     0.1000   -0.0012
    ##    150        0.5477             nan     0.1000   -0.0018
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1915             nan     0.1000    0.0262
    ##      2        1.1527             nan     0.1000    0.0167
    ##      3        1.1200             nan     0.1000    0.0144
    ##      4        1.0970             nan     0.1000    0.0133
    ##      5        1.0799             nan     0.1000    0.0088
    ##      6        1.0645             nan     0.1000    0.0075
    ##      7        1.0518             nan     0.1000    0.0063
    ##      8        1.0399             nan     0.1000    0.0063
    ##      9        1.0296             nan     0.1000    0.0052
    ##     10        1.0203             nan     0.1000    0.0040
    ##     20        0.9699             nan     0.1000    0.0012
    ##     40        0.9165             nan     0.1000   -0.0009
    ##     60        0.8886             nan     0.1000   -0.0012
    ##     80        0.8733             nan     0.1000   -0.0008
    ##    100        0.8606             nan     0.1000   -0.0005
    ##    120        0.8485             nan     0.1000   -0.0016
    ##    140        0.8366             nan     0.1000   -0.0003
    ##    150        0.8295             nan     0.1000   -0.0005
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1875             nan     0.1000    0.0268
    ##      2        1.1452             nan     0.1000    0.0214
    ##      3        1.1083             nan     0.1000    0.0159
    ##      4        1.0800             nan     0.1000    0.0137
    ##      5        1.0622             nan     0.1000    0.0094
    ##      6        1.0425             nan     0.1000    0.0087
    ##      7        1.0270             nan     0.1000    0.0035
    ##      8        1.0121             nan     0.1000    0.0056
    ##      9        1.0001             nan     0.1000    0.0041
    ##     10        0.9890             nan     0.1000    0.0033
    ##     20        0.9193             nan     0.1000    0.0004
    ##     40        0.8392             nan     0.1000   -0.0008
    ##     60        0.7922             nan     0.1000   -0.0035
    ##     80        0.7641             nan     0.1000   -0.0024
    ##    100        0.7273             nan     0.1000   -0.0027
    ##    120        0.7036             nan     0.1000   -0.0016
    ##    140        0.6768             nan     0.1000   -0.0019
    ##    150        0.6643             nan     0.1000   -0.0023
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1781             nan     0.1000    0.0253
    ##      2        1.1376             nan     0.1000    0.0169
    ##      3        1.0977             nan     0.1000    0.0163
    ##      4        1.0694             nan     0.1000    0.0092
    ##      5        1.0449             nan     0.1000    0.0107
    ##      6        1.0207             nan     0.1000    0.0090
    ##      7        1.0003             nan     0.1000    0.0061
    ##      8        0.9821             nan     0.1000    0.0057
    ##      9        0.9687             nan     0.1000    0.0033
    ##     10        0.9542             nan     0.1000    0.0030
    ##     20        0.8742             nan     0.1000    0.0005
    ##     40        0.7752             nan     0.1000   -0.0013
    ##     60        0.7099             nan     0.1000   -0.0019
    ##     80        0.6643             nan     0.1000   -0.0052
    ##    100        0.6170             nan     0.1000   -0.0032
    ##    120        0.5826             nan     0.1000   -0.0023
    ##    140        0.5498             nan     0.1000   -0.0017
    ##    150        0.5377             nan     0.1000   -0.0027
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1936             nan     0.1000    0.0292
    ##      2        1.1551             nan     0.1000    0.0206
    ##      3        1.1225             nan     0.1000    0.0143
    ##      4        1.0976             nan     0.1000    0.0132
    ##      5        1.0757             nan     0.1000    0.0108
    ##      6        1.0558             nan     0.1000    0.0080
    ##      7        1.0385             nan     0.1000    0.0056
    ##      8        1.0264             nan     0.1000    0.0080
    ##      9        1.0134             nan     0.1000    0.0055
    ##     10        1.0076             nan     0.1000    0.0008
    ##     20        0.9543             nan     0.1000    0.0015
    ##     40        0.9108             nan     0.1000    0.0004
    ##     60        0.8738             nan     0.1000   -0.0001
    ##     80        0.8528             nan     0.1000   -0.0001
    ##    100        0.8354             nan     0.1000   -0.0005
    ##    120        0.8231             nan     0.1000   -0.0008
    ##    140        0.8130             nan     0.1000   -0.0024
    ##    150        0.8059             nan     0.1000   -0.0012
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1847             nan     0.1000    0.0308
    ##      2        1.1417             nan     0.1000    0.0190
    ##      3        1.1057             nan     0.1000    0.0159
    ##      4        1.0775             nan     0.1000    0.0144
    ##      5        1.0550             nan     0.1000    0.0056
    ##      6        1.0340             nan     0.1000    0.0067
    ##      7        1.0178             nan     0.1000    0.0055
    ##      8        1.0082             nan     0.1000    0.0048
    ##      9        0.9943             nan     0.1000    0.0065
    ##     10        0.9821             nan     0.1000    0.0045
    ##     20        0.9092             nan     0.1000   -0.0027
    ##     40        0.8341             nan     0.1000   -0.0011
    ##     60        0.7813             nan     0.1000   -0.0014
    ##     80        0.7428             nan     0.1000   -0.0006
    ##    100        0.7060             nan     0.1000   -0.0051
    ##    120        0.6797             nan     0.1000   -0.0020
    ##    140        0.6551             nan     0.1000   -0.0015
    ##    150        0.6454             nan     0.1000   -0.0026
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1854             nan     0.1000    0.0251
    ##      2        1.1386             nan     0.1000    0.0207
    ##      3        1.0984             nan     0.1000    0.0206
    ##      4        1.0672             nan     0.1000    0.0126
    ##      5        1.0390             nan     0.1000    0.0099
    ##      6        1.0206             nan     0.1000    0.0070
    ##      7        1.0044             nan     0.1000    0.0043
    ##      8        0.9861             nan     0.1000    0.0063
    ##      9        0.9695             nan     0.1000    0.0038
    ##     10        0.9558             nan     0.1000    0.0010
    ##     20        0.8707             nan     0.1000   -0.0026
    ##     40        0.7830             nan     0.1000   -0.0030
    ##     60        0.7147             nan     0.1000   -0.0041
    ##     80        0.6630             nan     0.1000   -0.0024
    ##    100        0.6207             nan     0.1000   -0.0023
    ##    120        0.5878             nan     0.1000   -0.0036
    ##    140        0.5539             nan     0.1000   -0.0032
    ##    150        0.5377             nan     0.1000   -0.0024
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1820             nan     0.1000    0.0301
    ##      2        1.1425             nan     0.1000    0.0171
    ##      3        1.1080             nan     0.1000    0.0113
    ##      4        1.0808             nan     0.1000    0.0119
    ##      5        1.0601             nan     0.1000    0.0101
    ##      6        1.0422             nan     0.1000    0.0105
    ##      7        1.0310             nan     0.1000    0.0067
    ##      8        1.0233             nan     0.1000    0.0003
    ##      9        1.0105             nan     0.1000    0.0056
    ##     10        0.9994             nan     0.1000    0.0047
    ##     20        0.9500             nan     0.1000    0.0006
    ##     40        0.8973             nan     0.1000   -0.0021
    ##     60        0.8713             nan     0.1000   -0.0001
    ##     80        0.8482             nan     0.1000   -0.0029
    ##    100        0.8352             nan     0.1000   -0.0009
    ##    120        0.8246             nan     0.1000   -0.0034
    ##    140        0.8169             nan     0.1000   -0.0006
    ##    150        0.8099             nan     0.1000   -0.0013
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1854             nan     0.1000    0.0303
    ##      2        1.1387             nan     0.1000    0.0204
    ##      3        1.0999             nan     0.1000    0.0218
    ##      4        1.0700             nan     0.1000    0.0140
    ##      5        1.0444             nan     0.1000    0.0099
    ##      6        1.0241             nan     0.1000    0.0072
    ##      7        1.0055             nan     0.1000    0.0088
    ##      8        0.9898             nan     0.1000    0.0036
    ##      9        0.9780             nan     0.1000    0.0048
    ##     10        0.9651             nan     0.1000    0.0013
    ##     20        0.8964             nan     0.1000   -0.0031
    ##     40        0.8340             nan     0.1000   -0.0012
    ##     60        0.7840             nan     0.1000   -0.0035
    ##     80        0.7528             nan     0.1000   -0.0007
    ##    100        0.7169             nan     0.1000   -0.0041
    ##    120        0.6880             nan     0.1000   -0.0013
    ##    140        0.6628             nan     0.1000   -0.0020
    ##    150        0.6548             nan     0.1000   -0.0011
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1771             nan     0.1000    0.0287
    ##      2        1.1342             nan     0.1000    0.0226
    ##      3        1.0982             nan     0.1000    0.0149
    ##      4        1.0657             nan     0.1000    0.0126
    ##      5        1.0377             nan     0.1000    0.0099
    ##      6        1.0133             nan     0.1000    0.0092
    ##      7        0.9956             nan     0.1000    0.0042
    ##      8        0.9760             nan     0.1000    0.0065
    ##      9        0.9619             nan     0.1000    0.0025
    ##     10        0.9502             nan     0.1000    0.0016
    ##     20        0.8623             nan     0.1000   -0.0026
    ##     40        0.7768             nan     0.1000   -0.0037
    ##     60        0.7249             nan     0.1000   -0.0026
    ##     80        0.6763             nan     0.1000   -0.0054
    ##    100        0.6369             nan     0.1000   -0.0017
    ##    120        0.5958             nan     0.1000   -0.0021
    ##    140        0.5575             nan     0.1000   -0.0035
    ##    150        0.5402             nan     0.1000   -0.0006
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1896             nan     0.1000    0.0246
    ##      2        1.1531             nan     0.1000    0.0197
    ##      3        1.1216             nan     0.1000    0.0162
    ##      4        1.1029             nan     0.1000    0.0100
    ##      5        1.0826             nan     0.1000    0.0062
    ##      6        1.0659             nan     0.1000    0.0065
    ##      7        1.0530             nan     0.1000    0.0064
    ##      8        1.0408             nan     0.1000    0.0032
    ##      9        1.0379             nan     0.1000   -0.0008
    ##     10        1.0282             nan     0.1000    0.0044
    ##     20        0.9859             nan     0.1000    0.0001
    ##     40        0.9349             nan     0.1000   -0.0008
    ##     60        0.9094             nan     0.1000   -0.0033
    ##     80        0.8883             nan     0.1000   -0.0010
    ##    100        0.8714             nan     0.1000   -0.0004
    ##    120        0.8628             nan     0.1000   -0.0035
    ##    140        0.8545             nan     0.1000   -0.0003
    ##    150        0.8473             nan     0.1000   -0.0010
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1881             nan     0.1000    0.0268
    ##      2        1.1478             nan     0.1000    0.0154
    ##      3        1.1161             nan     0.1000    0.0134
    ##      4        1.0924             nan     0.1000    0.0125
    ##      5        1.0660             nan     0.1000    0.0105
    ##      6        1.0458             nan     0.1000    0.0071
    ##      7        1.0276             nan     0.1000    0.0079
    ##      8        1.0131             nan     0.1000    0.0061
    ##      9        1.0014             nan     0.1000    0.0061
    ##     10        0.9913             nan     0.1000    0.0038
    ##     20        0.9312             nan     0.1000   -0.0010
    ##     40        0.8693             nan     0.1000   -0.0041
    ##     60        0.8209             nan     0.1000   -0.0013
    ##     80        0.7873             nan     0.1000   -0.0038
    ##    100        0.7559             nan     0.1000   -0.0030
    ##    120        0.7327             nan     0.1000   -0.0025
    ##    140        0.7091             nan     0.1000   -0.0016
    ##    150        0.6987             nan     0.1000   -0.0027
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1929             nan     0.1000    0.0176
    ##      2        1.1476             nan     0.1000    0.0185
    ##      3        1.1130             nan     0.1000    0.0112
    ##      4        1.0873             nan     0.1000    0.0107
    ##      5        1.0579             nan     0.1000    0.0102
    ##      6        1.0339             nan     0.1000    0.0091
    ##      7        1.0150             nan     0.1000    0.0044
    ##      8        1.0016             nan     0.1000    0.0051
    ##      9        0.9892             nan     0.1000    0.0036
    ##     10        0.9773             nan     0.1000    0.0027
    ##     20        0.8997             nan     0.1000   -0.0008
    ##     40        0.8127             nan     0.1000   -0.0012
    ##     60        0.7543             nan     0.1000   -0.0033
    ##     80        0.7206             nan     0.1000   -0.0045
    ##    100        0.6776             nan     0.1000   -0.0017
    ##    120        0.6385             nan     0.1000   -0.0032
    ##    140        0.6066             nan     0.1000   -0.0014
    ##    150        0.5900             nan     0.1000   -0.0033
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1939             nan     0.1000    0.0187
    ##      2        1.1537             nan     0.1000    0.0180
    ##      3        1.1287             nan     0.1000    0.0124
    ##      4        1.1102             nan     0.1000    0.0109
    ##      5        1.0956             nan     0.1000    0.0072
    ##      6        1.0821             nan     0.1000    0.0078
    ##      7        1.0690             nan     0.1000    0.0074
    ##      8        1.0556             nan     0.1000    0.0063
    ##      9        1.0455             nan     0.1000    0.0043
    ##     10        1.0362             nan     0.1000    0.0035
    ##     20        0.9829             nan     0.1000    0.0007
    ##     40        0.9421             nan     0.1000   -0.0002
    ##     60        0.9197             nan     0.1000   -0.0011
    ##     80        0.9019             nan     0.1000   -0.0007
    ##    100        0.8858             nan     0.1000   -0.0017
    ##    120        0.8733             nan     0.1000   -0.0035
    ##    140        0.8678             nan     0.1000   -0.0028
    ##    150        0.8645             nan     0.1000   -0.0012
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1916             nan     0.1000    0.0203
    ##      2        1.1524             nan     0.1000    0.0142
    ##      3        1.1223             nan     0.1000    0.0109
    ##      4        1.0907             nan     0.1000    0.0128
    ##      5        1.0735             nan     0.1000    0.0064
    ##      6        1.0501             nan     0.1000    0.0093
    ##      7        1.0340             nan     0.1000    0.0068
    ##      8        1.0197             nan     0.1000    0.0064
    ##      9        1.0050             nan     0.1000    0.0030
    ##     10        0.9961             nan     0.1000    0.0011
    ##     20        0.9334             nan     0.1000   -0.0004
    ##     40        0.8871             nan     0.1000   -0.0040
    ##     60        0.8537             nan     0.1000   -0.0013
    ##     80        0.8226             nan     0.1000   -0.0018
    ##    100        0.7941             nan     0.1000   -0.0015
    ##    120        0.7717             nan     0.1000   -0.0010
    ##    140        0.7443             nan     0.1000   -0.0024
    ##    150        0.7317             nan     0.1000   -0.0032
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1887             nan     0.1000    0.0266
    ##      2        1.1451             nan     0.1000    0.0158
    ##      3        1.1124             nan     0.1000    0.0151
    ##      4        1.0842             nan     0.1000    0.0119
    ##      5        1.0618             nan     0.1000    0.0066
    ##      6        1.0444             nan     0.1000    0.0043
    ##      7        1.0288             nan     0.1000    0.0051
    ##      8        1.0119             nan     0.1000    0.0031
    ##      9        0.9976             nan     0.1000    0.0050
    ##     10        0.9836             nan     0.1000    0.0056
    ##     20        0.9142             nan     0.1000   -0.0032
    ##     40        0.8324             nan     0.1000   -0.0023
    ##     60        0.7793             nan     0.1000   -0.0012
    ##     80        0.7355             nan     0.1000   -0.0014
    ##    100        0.7020             nan     0.1000   -0.0047
    ##    120        0.6691             nan     0.1000   -0.0032
    ##    140        0.6357             nan     0.1000   -0.0044
    ##    150        0.6252             nan     0.1000   -0.0028
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1858             nan     0.1000    0.0225
    ##      2        1.1410             nan     0.1000    0.0228
    ##      3        1.1136             nan     0.1000    0.0150
    ##      4        1.0861             nan     0.1000    0.0130
    ##      5        1.0657             nan     0.1000    0.0096
    ##      6        1.0490             nan     0.1000    0.0088
    ##      7        1.0387             nan     0.1000    0.0065
    ##      8        1.0277             nan     0.1000    0.0051
    ##      9        1.0229             nan     0.1000   -0.0014
    ##     10        1.0135             nan     0.1000    0.0046
    ##     20        0.9582             nan     0.1000   -0.0017
    ##     40        0.9162             nan     0.1000   -0.0003
    ##     60        0.8931             nan     0.1000   -0.0014
    ##     80        0.8765             nan     0.1000   -0.0015
    ##    100        0.8629             nan     0.1000   -0.0013
    ##    120        0.8525             nan     0.1000   -0.0007
    ##    140        0.8413             nan     0.1000   -0.0017
    ##    150        0.8360             nan     0.1000   -0.0019
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1786             nan     0.1000    0.0243
    ##      2        1.1332             nan     0.1000    0.0183
    ##      3        1.1014             nan     0.1000    0.0154
    ##      4        1.0745             nan     0.1000    0.0117
    ##      5        1.0497             nan     0.1000    0.0090
    ##      6        1.0307             nan     0.1000    0.0075
    ##      7        1.0150             nan     0.1000    0.0058
    ##      8        1.0008             nan     0.1000    0.0033
    ##      9        0.9900             nan     0.1000    0.0045
    ##     10        0.9773             nan     0.1000    0.0024
    ##     20        0.9165             nan     0.1000    0.0004
    ##     40        0.8449             nan     0.1000   -0.0019
    ##     60        0.8076             nan     0.1000   -0.0038
    ##     80        0.7794             nan     0.1000   -0.0030
    ##    100        0.7504             nan     0.1000   -0.0017
    ##    120        0.7271             nan     0.1000   -0.0019
    ##    140        0.6956             nan     0.1000   -0.0023
    ##    150        0.6862             nan     0.1000   -0.0029
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1822             nan     0.1000    0.0271
    ##      2        1.1376             nan     0.1000    0.0221
    ##      3        1.0987             nan     0.1000    0.0155
    ##      4        1.0693             nan     0.1000    0.0129
    ##      5        1.0468             nan     0.1000    0.0073
    ##      6        1.0257             nan     0.1000    0.0085
    ##      7        1.0053             nan     0.1000    0.0087
    ##      8        0.9870             nan     0.1000    0.0043
    ##      9        0.9753             nan     0.1000    0.0020
    ##     10        0.9630             nan     0.1000    0.0031
    ##     20        0.8907             nan     0.1000   -0.0041
    ##     40        0.8124             nan     0.1000   -0.0021
    ##     60        0.7511             nan     0.1000   -0.0038
    ##     80        0.6997             nan     0.1000   -0.0011
    ##    100        0.6573             nan     0.1000   -0.0030
    ##    120        0.6102             nan     0.1000   -0.0018
    ##    140        0.5733             nan     0.1000   -0.0020
    ##    150        0.5572             nan     0.1000   -0.0028
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1929             nan     0.1000    0.0265
    ##      2        1.1465             nan     0.1000    0.0237
    ##      3        1.1161             nan     0.1000    0.0145
    ##      4        1.0895             nan     0.1000    0.0131
    ##      5        1.0701             nan     0.1000    0.0087
    ##      6        1.0520             nan     0.1000    0.0082
    ##      7        1.0389             nan     0.1000    0.0080
    ##      8        1.0285             nan     0.1000    0.0057
    ##      9        1.0171             nan     0.1000    0.0042
    ##     10        1.0072             nan     0.1000    0.0052
    ##     20        0.9550             nan     0.1000    0.0020
    ##     40        0.9004             nan     0.1000   -0.0008
    ##     60        0.8752             nan     0.1000   -0.0001
    ##     80        0.8540             nan     0.1000   -0.0012
    ##    100        0.8382             nan     0.1000   -0.0005
    ##    120        0.8253             nan     0.1000   -0.0021
    ##    140        0.8116             nan     0.1000   -0.0012
    ##    150        0.8076             nan     0.1000   -0.0019
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1810             nan     0.1000    0.0256
    ##      2        1.1374             nan     0.1000    0.0219
    ##      3        1.1011             nan     0.1000    0.0146
    ##      4        1.0784             nan     0.1000    0.0106
    ##      5        1.0565             nan     0.1000    0.0085
    ##      6        1.0317             nan     0.1000    0.0099
    ##      7        1.0123             nan     0.1000    0.0070
    ##      8        1.0012             nan     0.1000    0.0047
    ##      9        0.9869             nan     0.1000    0.0041
    ##     10        0.9779             nan     0.1000    0.0006
    ##     20        0.8939             nan     0.1000    0.0022
    ##     40        0.8194             nan     0.1000   -0.0018
    ##     60        0.7808             nan     0.1000   -0.0003
    ##     80        0.7496             nan     0.1000   -0.0052
    ##    100        0.7239             nan     0.1000   -0.0041
    ##    120        0.6987             nan     0.1000   -0.0016
    ##    140        0.6724             nan     0.1000   -0.0024
    ##    150        0.6627             nan     0.1000   -0.0012
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1815             nan     0.1000    0.0201
    ##      2        1.1315             nan     0.1000    0.0227
    ##      3        1.0965             nan     0.1000    0.0147
    ##      4        1.0624             nan     0.1000    0.0168
    ##      5        1.0333             nan     0.1000    0.0108
    ##      6        1.0136             nan     0.1000    0.0068
    ##      7        0.9903             nan     0.1000    0.0071
    ##      8        0.9741             nan     0.1000    0.0018
    ##      9        0.9601             nan     0.1000    0.0065
    ##     10        0.9437             nan     0.1000    0.0038
    ##     20        0.8555             nan     0.1000    0.0004
    ##     40        0.7634             nan     0.1000   -0.0025
    ##     60        0.7067             nan     0.1000   -0.0036
    ##     80        0.6671             nan     0.1000   -0.0050
    ##    100        0.6299             nan     0.1000   -0.0038
    ##    120        0.5961             nan     0.1000   -0.0009
    ##    140        0.5671             nan     0.1000   -0.0020
    ##    150        0.5583             nan     0.1000   -0.0020
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1942             nan     0.1000    0.0266
    ##      2        1.1585             nan     0.1000    0.0184
    ##      3        1.1295             nan     0.1000    0.0187
    ##      4        1.1015             nan     0.1000    0.0127
    ##      5        1.0798             nan     0.1000    0.0108
    ##      6        1.0597             nan     0.1000    0.0052
    ##      7        1.0454             nan     0.1000    0.0067
    ##      8        1.0308             nan     0.1000    0.0053
    ##      9        1.0247             nan     0.1000    0.0001
    ##     10        1.0150             nan     0.1000    0.0037
    ##     20        0.9616             nan     0.1000    0.0004
    ##     40        0.9120             nan     0.1000   -0.0036
    ##     60        0.8858             nan     0.1000   -0.0009
    ##     80        0.8713             nan     0.1000   -0.0010
    ##    100        0.8581             nan     0.1000   -0.0016
    ##    120        0.8466             nan     0.1000   -0.0014
    ##    140        0.8371             nan     0.1000   -0.0008
    ##    150        0.8328             nan     0.1000   -0.0039
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1828             nan     0.1000    0.0305
    ##      2        1.1415             nan     0.1000    0.0196
    ##      3        1.1025             nan     0.1000    0.0157
    ##      4        1.0738             nan     0.1000    0.0143
    ##      5        1.0518             nan     0.1000    0.0097
    ##      6        1.0307             nan     0.1000    0.0065
    ##      7        1.0127             nan     0.1000    0.0057
    ##      8        0.9965             nan     0.1000    0.0067
    ##      9        0.9836             nan     0.1000    0.0038
    ##     10        0.9736             nan     0.1000    0.0033
    ##     20        0.9135             nan     0.1000    0.0001
    ##     40        0.8534             nan     0.1000   -0.0015
    ##     60        0.8094             nan     0.1000   -0.0026
    ##     80        0.7813             nan     0.1000   -0.0027
    ##    100        0.7524             nan     0.1000   -0.0020
    ##    120        0.7345             nan     0.1000   -0.0017
    ##    140        0.7053             nan     0.1000   -0.0043
    ##    150        0.6885             nan     0.1000   -0.0021
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1825             nan     0.1000    0.0267
    ##      2        1.1365             nan     0.1000    0.0151
    ##      3        1.0973             nan     0.1000    0.0174
    ##      4        1.0677             nan     0.1000    0.0126
    ##      5        1.0458             nan     0.1000    0.0121
    ##      6        1.0186             nan     0.1000    0.0077
    ##      7        1.0024             nan     0.1000    0.0063
    ##      8        0.9844             nan     0.1000    0.0072
    ##      9        0.9698             nan     0.1000    0.0062
    ##     10        0.9537             nan     0.1000    0.0032
    ##     20        0.8789             nan     0.1000   -0.0040
    ##     40        0.8025             nan     0.1000   -0.0018
    ##     60        0.7452             nan     0.1000   -0.0050
    ##     80        0.6998             nan     0.1000   -0.0010
    ##    100        0.6625             nan     0.1000   -0.0005
    ##    120        0.6304             nan     0.1000   -0.0021
    ##    140        0.5975             nan     0.1000   -0.0043
    ##    150        0.5832             nan     0.1000   -0.0028
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1945             nan     0.1000    0.0213
    ##      2        1.1577             nan     0.1000    0.0180
    ##      3        1.1290             nan     0.1000    0.0133
    ##      4        1.1068             nan     0.1000    0.0125
    ##      5        1.0912             nan     0.1000    0.0090
    ##      6        1.0747             nan     0.1000    0.0069
    ##      7        1.0642             nan     0.1000    0.0051
    ##      8        1.0540             nan     0.1000    0.0048
    ##      9        1.0453             nan     0.1000    0.0045
    ##     10        1.0432             nan     0.1000   -0.0013
    ##     20        0.9944             nan     0.1000   -0.0004
    ##     40        0.9589             nan     0.1000   -0.0033
    ##     60        0.9324             nan     0.1000   -0.0012
    ##     80        0.9107             nan     0.1000   -0.0021
    ##    100        0.8957             nan     0.1000   -0.0011
    ##    120        0.8829             nan     0.1000   -0.0009
    ##    140        0.8721             nan     0.1000   -0.0022
    ##    150        0.8657             nan     0.1000   -0.0021
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1894             nan     0.1000    0.0226
    ##      2        1.1540             nan     0.1000    0.0160
    ##      3        1.1246             nan     0.1000    0.0111
    ##      4        1.0983             nan     0.1000    0.0120
    ##      5        1.0755             nan     0.1000    0.0075
    ##      6        1.0574             nan     0.1000    0.0079
    ##      7        1.0441             nan     0.1000    0.0043
    ##      8        1.0282             nan     0.1000    0.0047
    ##      9        1.0163             nan     0.1000    0.0043
    ##     10        1.0084             nan     0.1000    0.0016
    ##     20        0.9503             nan     0.1000    0.0005
    ##     40        0.8967             nan     0.1000   -0.0017
    ##     60        0.8438             nan     0.1000   -0.0018
    ##     80        0.8116             nan     0.1000   -0.0011
    ##    100        0.7888             nan     0.1000   -0.0022
    ##    120        0.7662             nan     0.1000   -0.0007
    ##    140        0.7357             nan     0.1000   -0.0036
    ##    150        0.7225             nan     0.1000   -0.0024
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1907             nan     0.1000    0.0225
    ##      2        1.1521             nan     0.1000    0.0157
    ##      3        1.1170             nan     0.1000    0.0114
    ##      4        1.0887             nan     0.1000    0.0109
    ##      5        1.0663             nan     0.1000    0.0100
    ##      6        1.0462             nan     0.1000    0.0051
    ##      7        1.0292             nan     0.1000    0.0054
    ##      8        1.0175             nan     0.1000    0.0020
    ##      9        1.0068             nan     0.1000    0.0021
    ##     10        0.9936             nan     0.1000   -0.0013
    ##     20        0.9210             nan     0.1000   -0.0020
    ##     40        0.8347             nan     0.1000   -0.0019
    ##     60        0.7713             nan     0.1000   -0.0032
    ##     80        0.7222             nan     0.1000   -0.0030
    ##    100        0.6841             nan     0.1000   -0.0037
    ##    120        0.6439             nan     0.1000   -0.0028
    ##    140        0.6142             nan     0.1000   -0.0021
    ##    150        0.5996             nan     0.1000   -0.0018
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1847             nan     0.1000    0.0273
    ##      2        1.1397             nan     0.1000    0.0177
    ##      3        1.1089             nan     0.1000    0.0153
    ##      4        1.0827             nan     0.1000    0.0121
    ##      5        1.0576             nan     0.1000    0.0100
    ##      6        1.0376             nan     0.1000    0.0078
    ##      7        1.0234             nan     0.1000    0.0064
    ##      8        1.0118             nan     0.1000    0.0054
    ##      9        1.0016             nan     0.1000    0.0038
    ##     10        0.9967             nan     0.1000   -0.0004
    ##     20        0.9407             nan     0.1000   -0.0013
    ##     40        0.8751             nan     0.1000    0.0003
    ##     60        0.8472             nan     0.1000    0.0002
    ##     80        0.8189             nan     0.1000   -0.0005
    ##    100        0.7968             nan     0.1000   -0.0019
    ##    120        0.7843             nan     0.1000   -0.0009
    ##    140        0.7720             nan     0.1000   -0.0017
    ##    150        0.7645             nan     0.1000   -0.0008
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1892             nan     0.1000    0.0297
    ##      2        1.1365             nan     0.1000    0.0195
    ##      3        1.1008             nan     0.1000    0.0187
    ##      4        1.0690             nan     0.1000    0.0081
    ##      5        1.0450             nan     0.1000    0.0128
    ##      6        1.0214             nan     0.1000    0.0102
    ##      7        1.0003             nan     0.1000    0.0079
    ##      8        0.9841             nan     0.1000    0.0038
    ##      9        0.9712             nan     0.1000    0.0047
    ##     10        0.9601             nan     0.1000    0.0052
    ##     20        0.8824             nan     0.1000   -0.0002
    ##     40        0.8131             nan     0.1000   -0.0003
    ##     60        0.7584             nan     0.1000   -0.0011
    ##     80        0.7189             nan     0.1000   -0.0027
    ##    100        0.6852             nan     0.1000   -0.0016
    ##    120        0.6544             nan     0.1000   -0.0007
    ##    140        0.6324             nan     0.1000   -0.0016
    ##    150        0.6180             nan     0.1000   -0.0013
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1841             nan     0.1000    0.0222
    ##      2        1.1358             nan     0.1000    0.0206
    ##      3        1.0920             nan     0.1000    0.0209
    ##      4        1.0572             nan     0.1000    0.0124
    ##      5        1.0274             nan     0.1000    0.0099
    ##      6        1.0017             nan     0.1000    0.0097
    ##      7        0.9809             nan     0.1000    0.0077
    ##      8        0.9615             nan     0.1000    0.0041
    ##      9        0.9407             nan     0.1000    0.0079
    ##     10        0.9254             nan     0.1000    0.0068
    ##     20        0.8413             nan     0.1000   -0.0022
    ##     40        0.7478             nan     0.1000   -0.0024
    ##     60        0.6786             nan     0.1000   -0.0038
    ##     80        0.6356             nan     0.1000   -0.0007
    ##    100        0.5928             nan     0.1000   -0.0013
    ##    120        0.5561             nan     0.1000   -0.0011
    ##    140        0.5248             nan     0.1000   -0.0010
    ##    150        0.5097             nan     0.1000   -0.0009
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1923             nan     0.1000    0.0257
    ##      2        1.1547             nan     0.1000    0.0183
    ##      3        1.1238             nan     0.1000    0.0148
    ##      4        1.0994             nan     0.1000    0.0114
    ##      5        1.0777             nan     0.1000    0.0088
    ##      6        1.0614             nan     0.1000    0.0074
    ##      7        1.0472             nan     0.1000    0.0060
    ##      8        1.0379             nan     0.1000    0.0051
    ##      9        1.0292             nan     0.1000    0.0040
    ##     10        1.0264             nan     0.1000   -0.0007
    ##     20        0.9822             nan     0.1000   -0.0017
    ##     40        0.9279             nan     0.1000   -0.0019
    ##     60        0.8950             nan     0.1000   -0.0006
    ##     80        0.8810             nan     0.1000   -0.0026
    ##    100        0.8709             nan     0.1000   -0.0018
    ##    120        0.8574             nan     0.1000   -0.0008
    ##    140        0.8449             nan     0.1000   -0.0022
    ##    150        0.8394             nan     0.1000   -0.0005
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1903             nan     0.1000    0.0250
    ##      2        1.1513             nan     0.1000    0.0191
    ##      3        1.1222             nan     0.1000    0.0137
    ##      4        1.0933             nan     0.1000    0.0166
    ##      5        1.0705             nan     0.1000    0.0082
    ##      6        1.0522             nan     0.1000    0.0088
    ##      7        1.0352             nan     0.1000    0.0048
    ##      8        1.0154             nan     0.1000    0.0063
    ##      9        1.0036             nan     0.1000    0.0018
    ##     10        1.0001             nan     0.1000   -0.0023
    ##     20        0.9325             nan     0.1000   -0.0012
    ##     40        0.8556             nan     0.1000   -0.0006
    ##     60        0.8122             nan     0.1000   -0.0020
    ##     80        0.7790             nan     0.1000   -0.0026
    ##    100        0.7557             nan     0.1000   -0.0012
    ##    120        0.7319             nan     0.1000   -0.0017
    ##    140        0.6962             nan     0.1000   -0.0020
    ##    150        0.6793             nan     0.1000   -0.0015
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1853             nan     0.1000    0.0200
    ##      2        1.1382             nan     0.1000    0.0229
    ##      3        1.1025             nan     0.1000    0.0124
    ##      4        1.0737             nan     0.1000    0.0117
    ##      5        1.0487             nan     0.1000    0.0090
    ##      6        1.0267             nan     0.1000    0.0080
    ##      7        1.0066             nan     0.1000    0.0076
    ##      8        0.9917             nan     0.1000    0.0017
    ##      9        0.9829             nan     0.1000    0.0014
    ##     10        0.9683             nan     0.1000    0.0025
    ##     20        0.8952             nan     0.1000   -0.0015
    ##     40        0.8112             nan     0.1000   -0.0015
    ##     60        0.7580             nan     0.1000   -0.0008
    ##     80        0.7037             nan     0.1000   -0.0044
    ##    100        0.6589             nan     0.1000   -0.0032
    ##    120        0.6205             nan     0.1000   -0.0039
    ##    140        0.5913             nan     0.1000   -0.0030
    ##    150        0.5760             nan     0.1000   -0.0031
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1913             nan     0.1000    0.0293
    ##      2        1.1469             nan     0.1000    0.0179
    ##      3        1.1160             nan     0.1000    0.0159
    ##      4        1.0958             nan     0.1000    0.0119
    ##      5        1.0737             nan     0.1000    0.0107
    ##      6        1.0546             nan     0.1000    0.0069
    ##      7        1.0415             nan     0.1000    0.0065
    ##      8        1.0325             nan     0.1000    0.0051
    ##      9        1.0271             nan     0.1000   -0.0002
    ##     10        1.0229             nan     0.1000   -0.0006
    ##     20        0.9690             nan     0.1000   -0.0011
    ##     40        0.9176             nan     0.1000   -0.0003
    ##     60        0.8908             nan     0.1000   -0.0001
    ##     80        0.8728             nan     0.1000   -0.0024
    ##    100        0.8590             nan     0.1000   -0.0018
    ##    120        0.8480             nan     0.1000   -0.0028
    ##    140        0.8405             nan     0.1000   -0.0017
    ##    150        0.8321             nan     0.1000   -0.0027
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1825             nan     0.1000    0.0272
    ##      2        1.1389             nan     0.1000    0.0172
    ##      3        1.1093             nan     0.1000    0.0141
    ##      4        1.0847             nan     0.1000    0.0130
    ##      5        1.0603             nan     0.1000    0.0109
    ##      6        1.0384             nan     0.1000    0.0084
    ##      7        1.0234             nan     0.1000    0.0043
    ##      8        1.0077             nan     0.1000    0.0057
    ##      9        0.9988             nan     0.1000    0.0038
    ##     10        0.9864             nan     0.1000    0.0053
    ##     20        0.9200             nan     0.1000   -0.0001
    ##     40        0.8497             nan     0.1000   -0.0010
    ##     60        0.8011             nan     0.1000   -0.0021
    ##     80        0.7572             nan     0.1000   -0.0030
    ##    100        0.7262             nan     0.1000   -0.0011
    ##    120        0.7054             nan     0.1000   -0.0025
    ##    140        0.6837             nan     0.1000   -0.0015
    ##    150        0.6739             nan     0.1000   -0.0032
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1852             nan     0.1000    0.0305
    ##      2        1.1359             nan     0.1000    0.0166
    ##      3        1.0997             nan     0.1000    0.0170
    ##      4        1.0718             nan     0.1000    0.0112
    ##      5        1.0443             nan     0.1000    0.0104
    ##      6        1.0263             nan     0.1000    0.0044
    ##      7        1.0034             nan     0.1000    0.0046
    ##      8        0.9885             nan     0.1000    0.0018
    ##      9        0.9719             nan     0.1000    0.0033
    ##     10        0.9612             nan     0.1000    0.0030
    ##     20        0.8844             nan     0.1000   -0.0027
    ##     40        0.8072             nan     0.1000   -0.0019
    ##     60        0.7484             nan     0.1000   -0.0015
    ##     80        0.7001             nan     0.1000   -0.0018
    ##    100        0.6565             nan     0.1000   -0.0053
    ##    120        0.6120             nan     0.1000   -0.0036
    ##    140        0.5800             nan     0.1000   -0.0023
    ##    150        0.5661             nan     0.1000   -0.0009
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1893             nan     0.1000    0.0251
    ##      2        1.1462             nan     0.1000    0.0210
    ##      3        1.1142             nan     0.1000    0.0136
    ##      4        1.0902             nan     0.1000    0.0099
    ##      5        1.0670             nan     0.1000    0.0121
    ##      6        1.0519             nan     0.1000    0.0064
    ##      7        1.0378             nan     0.1000    0.0060
    ##      8        1.0279             nan     0.1000    0.0063
    ##      9        1.0188             nan     0.1000    0.0046
    ##     10        1.0081             nan     0.1000    0.0042
    ##     20        0.9562             nan     0.1000    0.0010
    ##     40        0.9137             nan     0.1000   -0.0010
    ##     60        0.8902             nan     0.1000   -0.0014
    ##     80        0.8727             nan     0.1000   -0.0030
    ##    100        0.8521             nan     0.1000    0.0001
    ##    120        0.8411             nan     0.1000   -0.0029
    ##    140        0.8333             nan     0.1000   -0.0026
    ##    150        0.8302             nan     0.1000   -0.0018
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1847             nan     0.1000    0.0323
    ##      2        1.1411             nan     0.1000    0.0194
    ##      3        1.1094             nan     0.1000    0.0141
    ##      4        1.0798             nan     0.1000    0.0125
    ##      5        1.0545             nan     0.1000    0.0080
    ##      6        1.0338             nan     0.1000    0.0058
    ##      7        1.0187             nan     0.1000    0.0051
    ##      8        1.0052             nan     0.1000    0.0053
    ##      9        0.9937             nan     0.1000    0.0015
    ##     10        0.9817             nan     0.1000    0.0043
    ##     20        0.9176             nan     0.1000   -0.0004
    ##     40        0.8615             nan     0.1000   -0.0010
    ##     60        0.8251             nan     0.1000   -0.0036
    ##     80        0.7817             nan     0.1000   -0.0011
    ##    100        0.7544             nan     0.1000   -0.0006
    ##    120        0.7286             nan     0.1000   -0.0028
    ##    140        0.7087             nan     0.1000   -0.0005
    ##    150        0.6970             nan     0.1000   -0.0019
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1828             nan     0.1000    0.0269
    ##      2        1.1418             nan     0.1000    0.0190
    ##      3        1.1081             nan     0.1000    0.0170
    ##      4        1.0780             nan     0.1000    0.0112
    ##      5        1.0491             nan     0.1000    0.0114
    ##      6        1.0279             nan     0.1000    0.0049
    ##      7        1.0068             nan     0.1000    0.0063
    ##      8        0.9906             nan     0.1000    0.0074
    ##      9        0.9750             nan     0.1000    0.0047
    ##     10        0.9640             nan     0.1000    0.0028
    ##     20        0.8838             nan     0.1000    0.0004
    ##     40        0.7956             nan     0.1000   -0.0013
    ##     60        0.7358             nan     0.1000   -0.0005
    ##     80        0.6961             nan     0.1000   -0.0074
    ##    100        0.6519             nan     0.1000   -0.0027
    ##    120        0.6176             nan     0.1000   -0.0034
    ##    140        0.5871             nan     0.1000   -0.0014
    ##    150        0.5732             nan     0.1000   -0.0032
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1863             nan     0.1000    0.0265
    ##      2        1.1582             nan     0.1000    0.0185
    ##      3        1.1238             nan     0.1000    0.0144
    ##      4        1.0994             nan     0.1000    0.0134
    ##      5        1.0784             nan     0.1000    0.0122
    ##      6        1.0642             nan     0.1000    0.0070
    ##      7        1.0463             nan     0.1000    0.0061
    ##      8        1.0313             nan     0.1000    0.0051
    ##      9        1.0228             nan     0.1000    0.0048
    ##     10        1.0138             nan     0.1000    0.0024
    ##     20        0.9654             nan     0.1000   -0.0007
    ##     40        0.9236             nan     0.1000    0.0006
    ##     60        0.8980             nan     0.1000   -0.0032
    ##     80        0.8795             nan     0.1000   -0.0019
    ##    100        0.8691             nan     0.1000   -0.0007
    ##    120        0.8597             nan     0.1000   -0.0016
    ##    140        0.8454             nan     0.1000   -0.0005
    ##    150        0.8384             nan     0.1000   -0.0014
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1846             nan     0.1000    0.0264
    ##      2        1.1415             nan     0.1000    0.0188
    ##      3        1.1053             nan     0.1000    0.0146
    ##      4        1.0800             nan     0.1000    0.0117
    ##      5        1.0529             nan     0.1000    0.0113
    ##      6        1.0294             nan     0.1000    0.0116
    ##      7        1.0173             nan     0.1000    0.0035
    ##      8        1.0039             nan     0.1000    0.0064
    ##      9        0.9910             nan     0.1000    0.0052
    ##     10        0.9776             nan     0.1000    0.0033
    ##     20        0.9217             nan     0.1000   -0.0007
    ##     40        0.8535             nan     0.1000   -0.0005
    ##     60        0.8099             nan     0.1000   -0.0010
    ##     80        0.7810             nan     0.1000    0.0007
    ##    100        0.7570             nan     0.1000   -0.0035
    ##    120        0.7305             nan     0.1000   -0.0015
    ##    140        0.7085             nan     0.1000   -0.0040
    ##    150        0.6996             nan     0.1000   -0.0015
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1818             nan     0.1000    0.0241
    ##      2        1.1376             nan     0.1000    0.0108
    ##      3        1.0970             nan     0.1000    0.0121
    ##      4        1.0668             nan     0.1000    0.0131
    ##      5        1.0423             nan     0.1000    0.0094
    ##      6        1.0222             nan     0.1000    0.0051
    ##      7        1.0010             nan     0.1000    0.0027
    ##      8        0.9878             nan     0.1000    0.0037
    ##      9        0.9715             nan     0.1000    0.0049
    ##     10        0.9528             nan     0.1000    0.0025
    ##     20        0.8807             nan     0.1000    0.0009
    ##     40        0.8142             nan     0.1000   -0.0043
    ##     60        0.7520             nan     0.1000   -0.0032
    ##     80        0.7030             nan     0.1000   -0.0037
    ##    100        0.6593             nan     0.1000   -0.0009
    ##    120        0.6196             nan     0.1000   -0.0031
    ##    140        0.5789             nan     0.1000   -0.0014
    ##    150        0.5658             nan     0.1000   -0.0023
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1976             nan     0.1000    0.0228
    ##      2        1.1577             nan     0.1000    0.0192
    ##      3        1.1246             nan     0.1000    0.0149
    ##      4        1.0973             nan     0.1000    0.0124
    ##      5        1.0716             nan     0.1000    0.0101
    ##      6        1.0558             nan     0.1000    0.0092
    ##      7        1.0361             nan     0.1000    0.0054
    ##      8        1.0234             nan     0.1000    0.0065
    ##      9        1.0200             nan     0.1000    0.0000
    ##     10        1.0069             nan     0.1000    0.0040
    ##     20        0.9562             nan     0.1000    0.0005
    ##     40        0.9016             nan     0.1000   -0.0010
    ##     60        0.8735             nan     0.1000   -0.0025
    ##     80        0.8496             nan     0.1000   -0.0009
    ##    100        0.8359             nan     0.1000   -0.0008
    ##    120        0.8211             nan     0.1000   -0.0019
    ##    140        0.8107             nan     0.1000   -0.0023
    ##    150        0.8048             nan     0.1000   -0.0022
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1869             nan     0.1000    0.0300
    ##      2        1.1418             nan     0.1000    0.0207
    ##      3        1.1088             nan     0.1000    0.0159
    ##      4        1.0777             nan     0.1000    0.0134
    ##      5        1.0511             nan     0.1000    0.0100
    ##      6        1.0335             nan     0.1000    0.0083
    ##      7        1.0127             nan     0.1000    0.0084
    ##      8        0.9970             nan     0.1000    0.0055
    ##      9        0.9824             nan     0.1000    0.0033
    ##     10        0.9687             nan     0.1000    0.0059
    ##     20        0.9060             nan     0.1000   -0.0007
    ##     40        0.8371             nan     0.1000   -0.0014
    ##     60        0.7895             nan     0.1000   -0.0025
    ##     80        0.7540             nan     0.1000   -0.0004
    ##    100        0.7217             nan     0.1000   -0.0025
    ##    120        0.6969             nan     0.1000   -0.0024
    ##    140        0.6688             nan     0.1000   -0.0029
    ##    150        0.6608             nan     0.1000   -0.0030
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1819             nan     0.1000    0.0275
    ##      2        1.1313             nan     0.1000    0.0175
    ##      3        1.0939             nan     0.1000    0.0157
    ##      4        1.0642             nan     0.1000    0.0121
    ##      5        1.0350             nan     0.1000    0.0091
    ##      6        1.0118             nan     0.1000    0.0077
    ##      7        0.9932             nan     0.1000    0.0054
    ##      8        0.9746             nan     0.1000    0.0055
    ##      9        0.9568             nan     0.1000    0.0065
    ##     10        0.9389             nan     0.1000    0.0016
    ##     20        0.8555             nan     0.1000    0.0009
    ##     40        0.7583             nan     0.1000   -0.0007
    ##     60        0.7020             nan     0.1000   -0.0035
    ##     80        0.6461             nan     0.1000   -0.0008
    ##    100        0.6055             nan     0.1000   -0.0027
    ##    120        0.5743             nan     0.1000   -0.0022
    ##    140        0.5491             nan     0.1000   -0.0011
    ##    150        0.5400             nan     0.1000   -0.0013
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1899             nan     0.1000    0.0241
    ##      2        1.1495             nan     0.1000    0.0191
    ##      3        1.1165             nan     0.1000    0.0165
    ##      4        1.0928             nan     0.1000    0.0101
    ##      5        1.0723             nan     0.1000    0.0088
    ##      6        1.0541             nan     0.1000    0.0083
    ##      7        1.0425             nan     0.1000    0.0056
    ##      8        1.0382             nan     0.1000   -0.0002
    ##      9        1.0306             nan     0.1000    0.0014
    ##     10        1.0207             nan     0.1000    0.0053
    ##     20        0.9604             nan     0.1000    0.0014
    ##     40        0.9080             nan     0.1000   -0.0006
    ##     60        0.8862             nan     0.1000   -0.0022
    ##     80        0.8681             nan     0.1000   -0.0014
    ##    100        0.8539             nan     0.1000   -0.0008
    ##    120        0.8426             nan     0.1000   -0.0008
    ##    140        0.8333             nan     0.1000   -0.0016
    ##    150        0.8296             nan     0.1000   -0.0014
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1878             nan     0.1000    0.0292
    ##      2        1.1386             nan     0.1000    0.0220
    ##      3        1.1062             nan     0.1000    0.0148
    ##      4        1.0777             nan     0.1000    0.0084
    ##      5        1.0546             nan     0.1000    0.0084
    ##      6        1.0381             nan     0.1000    0.0058
    ##      7        1.0216             nan     0.1000    0.0092
    ##      8        1.0036             nan     0.1000    0.0069
    ##      9        0.9895             nan     0.1000    0.0049
    ##     10        0.9758             nan     0.1000    0.0051
    ##     20        0.9163             nan     0.1000    0.0002
    ##     40        0.8514             nan     0.1000   -0.0001
    ##     60        0.8141             nan     0.1000   -0.0023
    ##     80        0.7762             nan     0.1000   -0.0017
    ##    100        0.7589             nan     0.1000   -0.0015
    ##    120        0.7278             nan     0.1000   -0.0006
    ##    140        0.7075             nan     0.1000   -0.0032
    ##    150        0.6998             nan     0.1000   -0.0051
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1881             nan     0.1000    0.0251
    ##      2        1.1482             nan     0.1000    0.0173
    ##      3        1.1152             nan     0.1000    0.0154
    ##      4        1.0774             nan     0.1000    0.0150
    ##      5        1.0488             nan     0.1000    0.0110
    ##      6        1.0244             nan     0.1000    0.0086
    ##      7        1.0051             nan     0.1000    0.0071
    ##      8        0.9879             nan     0.1000    0.0046
    ##      9        0.9705             nan     0.1000    0.0056
    ##     10        0.9582             nan     0.1000    0.0021
    ##     20        0.8838             nan     0.1000   -0.0023
    ##     40        0.8088             nan     0.1000   -0.0019
    ##     60        0.7474             nan     0.1000   -0.0018
    ##     80        0.6994             nan     0.1000   -0.0044
    ##    100        0.6607             nan     0.1000   -0.0012
    ##    120        0.6211             nan     0.1000   -0.0029
    ##    140        0.5921             nan     0.1000   -0.0017
    ##    150        0.5781             nan     0.1000   -0.0024
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1944             nan     0.1000    0.0235
    ##      2        1.1552             nan     0.1000    0.0187
    ##      3        1.1269             nan     0.1000    0.0161
    ##      4        1.1071             nan     0.1000    0.0085
    ##      5        1.0897             nan     0.1000    0.0074
    ##      6        1.0774             nan     0.1000    0.0060
    ##      7        1.0623             nan     0.1000    0.0045
    ##      8        1.0525             nan     0.1000    0.0051
    ##      9        1.0423             nan     0.1000    0.0020
    ##     10        1.0347             nan     0.1000    0.0026
    ##     20        0.9931             nan     0.1000   -0.0010
    ##     40        0.9502             nan     0.1000   -0.0020
    ##     60        0.9183             nan     0.1000   -0.0003
    ##     80        0.8979             nan     0.1000   -0.0007
    ##    100        0.8829             nan     0.1000   -0.0007
    ##    120        0.8662             nan     0.1000   -0.0017
    ##    140        0.8578             nan     0.1000   -0.0025
    ##    150        0.8544             nan     0.1000   -0.0025
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1897             nan     0.1000    0.0208
    ##      2        1.1528             nan     0.1000    0.0148
    ##      3        1.1222             nan     0.1000    0.0153
    ##      4        1.0976             nan     0.1000    0.0117
    ##      5        1.0748             nan     0.1000    0.0086
    ##      6        1.0553             nan     0.1000    0.0054
    ##      7        1.0387             nan     0.1000    0.0063
    ##      8        1.0223             nan     0.1000    0.0028
    ##      9        1.0078             nan     0.1000    0.0044
    ##     10        0.9993             nan     0.1000    0.0024
    ##     20        0.9400             nan     0.1000    0.0018
    ##     40        0.8701             nan     0.1000   -0.0001
    ##     60        0.8190             nan     0.1000   -0.0014
    ##     80        0.7833             nan     0.1000   -0.0020
    ##    100        0.7594             nan     0.1000   -0.0015
    ##    120        0.7403             nan     0.1000   -0.0032
    ##    140        0.7118             nan     0.1000    0.0001
    ##    150        0.7021             nan     0.1000   -0.0030
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1943             nan     0.1000    0.0230
    ##      2        1.1482             nan     0.1000    0.0217
    ##      3        1.1137             nan     0.1000    0.0142
    ##      4        1.0858             nan     0.1000    0.0107
    ##      5        1.0612             nan     0.1000    0.0074
    ##      6        1.0423             nan     0.1000    0.0067
    ##      7        1.0232             nan     0.1000    0.0084
    ##      8        1.0095             nan     0.1000    0.0045
    ##      9        0.9964             nan     0.1000    0.0036
    ##     10        0.9807             nan     0.1000    0.0040
    ##     20        0.9079             nan     0.1000   -0.0013
    ##     40        0.8158             nan     0.1000   -0.0009
    ##     60        0.7619             nan     0.1000   -0.0018
    ##     80        0.7128             nan     0.1000   -0.0034
    ##    100        0.6712             nan     0.1000   -0.0035
    ##    120        0.6386             nan     0.1000   -0.0043
    ##    140        0.6027             nan     0.1000   -0.0025
    ##    150        0.5839             nan     0.1000   -0.0022
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1781             nan     0.1000    0.0301
    ##      2        1.1316             nan     0.1000    0.0211
    ##      3        1.0940             nan     0.1000    0.0156
    ##      4        1.0647             nan     0.1000    0.0150
    ##      5        1.0393             nan     0.1000    0.0127
    ##      6        1.0238             nan     0.1000    0.0090
    ##      7        1.0083             nan     0.1000    0.0087
    ##      8        0.9947             nan     0.1000    0.0068
    ##      9        0.9835             nan     0.1000    0.0051
    ##     10        0.9763             nan     0.1000    0.0025
    ##     20        0.9125             nan     0.1000   -0.0023
    ##     40        0.8573             nan     0.1000    0.0000
    ##     60        0.8252             nan     0.1000    0.0003
    ##     80        0.8091             nan     0.1000   -0.0020
    ##    100        0.7897             nan     0.1000   -0.0011
    ##    120        0.7795             nan     0.1000   -0.0005
    ##    140        0.7685             nan     0.1000   -0.0015
    ##    150        0.7653             nan     0.1000   -0.0014
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1745             nan     0.1000    0.0367
    ##      2        1.1217             nan     0.1000    0.0206
    ##      3        1.0853             nan     0.1000    0.0163
    ##      4        1.0521             nan     0.1000    0.0165
    ##      5        1.0268             nan     0.1000    0.0113
    ##      6        1.0041             nan     0.1000    0.0109
    ##      7        0.9828             nan     0.1000    0.0073
    ##      8        0.9627             nan     0.1000    0.0092
    ##      9        0.9454             nan     0.1000    0.0080
    ##     10        0.9362             nan     0.1000    0.0042
    ##     20        0.8518             nan     0.1000    0.0014
    ##     40        0.7801             nan     0.1000   -0.0007
    ##     60        0.7371             nan     0.1000   -0.0009
    ##     80        0.7018             nan     0.1000   -0.0014
    ##    100        0.6701             nan     0.1000   -0.0021
    ##    120        0.6479             nan     0.1000   -0.0018
    ##    140        0.6253             nan     0.1000   -0.0036
    ##    150        0.6138             nan     0.1000   -0.0026
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1705             nan     0.1000    0.0334
    ##      2        1.1134             nan     0.1000    0.0210
    ##      3        1.0726             nan     0.1000    0.0148
    ##      4        1.0329             nan     0.1000    0.0191
    ##      5        1.0007             nan     0.1000    0.0133
    ##      6        0.9770             nan     0.1000    0.0084
    ##      7        0.9521             nan     0.1000    0.0102
    ##      8        0.9314             nan     0.1000    0.0062
    ##      9        0.9150             nan     0.1000    0.0063
    ##     10        0.8979             nan     0.1000    0.0044
    ##     20        0.8177             nan     0.1000    0.0014
    ##     40        0.7362             nan     0.1000   -0.0029
    ##     60        0.6708             nan     0.1000   -0.0017
    ##     80        0.6268             nan     0.1000   -0.0043
    ##    100        0.5835             nan     0.1000   -0.0032
    ##    120        0.5492             nan     0.1000   -0.0021
    ##    140        0.5175             nan     0.1000   -0.0026
    ##    150        0.5052             nan     0.1000   -0.0042
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1916             nan     0.1000    0.0218
    ##      2        1.1593             nan     0.1000    0.0147
    ##      3        1.1303             nan     0.1000    0.0169
    ##      4        1.1053             nan     0.1000    0.0110
    ##      5        1.0896             nan     0.1000    0.0089
    ##      6        1.0747             nan     0.1000    0.0076
    ##      7        1.0677             nan     0.1000   -0.0018
    ##      8        1.0569             nan     0.1000    0.0062
    ##      9        1.0440             nan     0.1000    0.0051
    ##     10        1.0373             nan     0.1000    0.0033
    ##     20        0.9883             nan     0.1000   -0.0000
    ##     40        0.9497             nan     0.1000   -0.0031
    ##     60        0.9147             nan     0.1000   -0.0009
    ##     80        0.9001             nan     0.1000   -0.0037
    ##    100        0.8835             nan     0.1000   -0.0010
    ##    120        0.8758             nan     0.1000   -0.0007
    ##    140        0.8646             nan     0.1000   -0.0009
    ##    150        0.8608             nan     0.1000   -0.0021
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1853             nan     0.1000    0.0170
    ##      2        1.1466             nan     0.1000    0.0146
    ##      3        1.1105             nan     0.1000    0.0114
    ##      4        1.0823             nan     0.1000    0.0117
    ##      5        1.0616             nan     0.1000    0.0095
    ##      6        1.0451             nan     0.1000    0.0074
    ##      7        1.0312             nan     0.1000    0.0041
    ##      8        1.0170             nan     0.1000    0.0051
    ##      9        1.0066             nan     0.1000    0.0026
    ##     10        0.9969             nan     0.1000    0.0030
    ##     20        0.9396             nan     0.1000   -0.0016
    ##     40        0.8681             nan     0.1000   -0.0009
    ##     60        0.8298             nan     0.1000   -0.0024
    ##     80        0.7991             nan     0.1000   -0.0026
    ##    100        0.7700             nan     0.1000   -0.0020
    ##    120        0.7435             nan     0.1000   -0.0016
    ##    140        0.7183             nan     0.1000   -0.0023
    ##    150        0.7054             nan     0.1000   -0.0037
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1870             nan     0.1000    0.0248
    ##      2        1.1435             nan     0.1000    0.0167
    ##      3        1.1074             nan     0.1000    0.0131
    ##      4        1.0834             nan     0.1000    0.0111
    ##      5        1.0640             nan     0.1000    0.0060
    ##      6        1.0447             nan     0.1000    0.0078
    ##      7        1.0265             nan     0.1000    0.0054
    ##      8        1.0123             nan     0.1000    0.0028
    ##      9        0.9952             nan     0.1000    0.0064
    ##     10        0.9812             nan     0.1000    0.0019
    ##     20        0.8999             nan     0.1000    0.0008
    ##     40        0.8361             nan     0.1000   -0.0025
    ##     60        0.7745             nan     0.1000   -0.0021
    ##     80        0.7310             nan     0.1000   -0.0041
    ##    100        0.6851             nan     0.1000   -0.0013
    ##    120        0.6377             nan     0.1000   -0.0018
    ##    140        0.6088             nan     0.1000   -0.0034
    ##    150        0.5951             nan     0.1000   -0.0023
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1969             nan     0.1000    0.0236
    ##      2        1.1584             nan     0.1000    0.0207
    ##      3        1.1280             nan     0.1000    0.0130
    ##      4        1.1012             nan     0.1000    0.0088
    ##      5        1.0809             nan     0.1000    0.0081
    ##      6        1.0656             nan     0.1000    0.0069
    ##      7        1.0529             nan     0.1000    0.0044
    ##      8        1.0427             nan     0.1000    0.0043
    ##      9        1.0337             nan     0.1000    0.0040
    ##     10        1.0246             nan     0.1000    0.0025
    ##     20        0.9861             nan     0.1000    0.0014
    ##     40        0.9405             nan     0.1000   -0.0014
    ##     60        0.9181             nan     0.1000   -0.0007
    ##     80        0.9001             nan     0.1000   -0.0050
    ##    100        0.8891             nan     0.1000   -0.0005
    ##    120        0.8741             nan     0.1000   -0.0023
    ##    140        0.8668             nan     0.1000   -0.0017
    ##    150        0.8609             nan     0.1000   -0.0020
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1928             nan     0.1000    0.0186
    ##      2        1.1515             nan     0.1000    0.0219
    ##      3        1.1165             nan     0.1000    0.0162
    ##      4        1.0907             nan     0.1000    0.0108
    ##      5        1.0678             nan     0.1000    0.0118
    ##      6        1.0475             nan     0.1000    0.0068
    ##      7        1.0351             nan     0.1000    0.0040
    ##      8        1.0201             nan     0.1000    0.0060
    ##      9        1.0097             nan     0.1000    0.0040
    ##     10        1.0001             nan     0.1000    0.0037
    ##     20        0.9393             nan     0.1000   -0.0004
    ##     40        0.8853             nan     0.1000   -0.0014
    ##     60        0.8394             nan     0.1000   -0.0015
    ##     80        0.8154             nan     0.1000   -0.0032
    ##    100        0.7837             nan     0.1000   -0.0038
    ##    120        0.7582             nan     0.1000   -0.0035
    ##    140        0.7409             nan     0.1000   -0.0024
    ##    150        0.7286             nan     0.1000   -0.0028
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1859             nan     0.1000    0.0233
    ##      2        1.1461             nan     0.1000    0.0145
    ##      3        1.1078             nan     0.1000    0.0131
    ##      4        1.0797             nan     0.1000    0.0130
    ##      5        1.0565             nan     0.1000    0.0069
    ##      6        1.0317             nan     0.1000    0.0082
    ##      7        1.0158             nan     0.1000    0.0032
    ##      8        1.0030             nan     0.1000    0.0021
    ##      9        0.9902             nan     0.1000    0.0018
    ##     10        0.9800             nan     0.1000    0.0023
    ##     20        0.9160             nan     0.1000   -0.0039
    ##     40        0.8432             nan     0.1000   -0.0040
    ##     60        0.7815             nan     0.1000   -0.0028
    ##     80        0.7318             nan     0.1000   -0.0032
    ##    100        0.6884             nan     0.1000   -0.0007
    ##    120        0.6502             nan     0.1000   -0.0019
    ##    140        0.6206             nan     0.1000   -0.0020
    ##    150        0.6054             nan     0.1000   -0.0014
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1993             nan     0.1000    0.0222
    ##      2        1.1633             nan     0.1000    0.0190
    ##      3        1.1365             nan     0.1000    0.0140
    ##      4        1.1124             nan     0.1000    0.0116
    ##      5        1.0892             nan     0.1000    0.0097
    ##      6        1.0678             nan     0.1000    0.0068
    ##      7        1.0537             nan     0.1000    0.0075
    ##      8        1.0436             nan     0.1000    0.0047
    ##      9        1.0319             nan     0.1000    0.0027
    ##     10        1.0251             nan     0.1000    0.0028
    ##     20        0.9839             nan     0.1000   -0.0002
    ##     40        0.9376             nan     0.1000   -0.0016
    ##     60        0.9181             nan     0.1000   -0.0009
    ##     80        0.9015             nan     0.1000   -0.0041
    ##    100        0.8838             nan     0.1000   -0.0012
    ##    120        0.8681             nan     0.1000   -0.0011
    ##    140        0.8575             nan     0.1000   -0.0006
    ##    150        0.8541             nan     0.1000   -0.0011
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1896             nan     0.1000    0.0234
    ##      2        1.1537             nan     0.1000    0.0156
    ##      3        1.1220             nan     0.1000    0.0138
    ##      4        1.0984             nan     0.1000    0.0136
    ##      5        1.0739             nan     0.1000    0.0091
    ##      6        1.0565             nan     0.1000    0.0072
    ##      7        1.0391             nan     0.1000    0.0062
    ##      8        1.0223             nan     0.1000    0.0023
    ##      9        1.0116             nan     0.1000    0.0026
    ##     10        0.9981             nan     0.1000    0.0023
    ##     20        0.9315             nan     0.1000    0.0007
    ##     40        0.8690             nan     0.1000   -0.0028
    ##     60        0.8253             nan     0.1000   -0.0012
    ##     80        0.7924             nan     0.1000   -0.0001
    ##    100        0.7595             nan     0.1000   -0.0035
    ##    120        0.7329             nan     0.1000   -0.0020
    ##    140        0.7062             nan     0.1000   -0.0018
    ##    150        0.6906             nan     0.1000   -0.0015
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1872             nan     0.1000    0.0245
    ##      2        1.1450             nan     0.1000    0.0178
    ##      3        1.1080             nan     0.1000    0.0131
    ##      4        1.0768             nan     0.1000    0.0133
    ##      5        1.0526             nan     0.1000    0.0097
    ##      6        1.0329             nan     0.1000    0.0086
    ##      7        1.0127             nan     0.1000    0.0015
    ##      8        0.9992             nan     0.1000    0.0068
    ##      9        0.9855             nan     0.1000    0.0034
    ##     10        0.9747             nan     0.1000    0.0035
    ##     20        0.8932             nan     0.1000   -0.0007
    ##     40        0.8132             nan     0.1000   -0.0020
    ##     60        0.7516             nan     0.1000   -0.0022
    ##     80        0.7070             nan     0.1000   -0.0028
    ##    100        0.6683             nan     0.1000   -0.0017
    ##    120        0.6369             nan     0.1000   -0.0035
    ##    140        0.6082             nan     0.1000   -0.0044
    ##    150        0.5938             nan     0.1000   -0.0026
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1843             nan     0.1000    0.0264
    ##      2        1.1402             nan     0.1000    0.0188
    ##      3        1.1089             nan     0.1000    0.0141
    ##      4        1.0866             nan     0.1000    0.0106
    ##      5        1.0666             nan     0.1000    0.0093
    ##      6        1.0503             nan     0.1000    0.0094
    ##      7        1.0352             nan     0.1000    0.0066
    ##      8        1.0287             nan     0.1000   -0.0006
    ##      9        1.0177             nan     0.1000    0.0065
    ##     10        1.0077             nan     0.1000    0.0048
    ##     20        0.9564             nan     0.1000    0.0001
    ##     40        0.8998             nan     0.1000    0.0007
    ##     60        0.8777             nan     0.1000   -0.0010
    ##     80        0.8603             nan     0.1000   -0.0008
    ##    100        0.8486             nan     0.1000   -0.0027
    ##    120        0.8344             nan     0.1000   -0.0009
    ##    140        0.8267             nan     0.1000   -0.0014
    ##    150        0.8228             nan     0.1000   -0.0014
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1862             nan     0.1000    0.0331
    ##      2        1.1431             nan     0.1000    0.0213
    ##      3        1.1046             nan     0.1000    0.0206
    ##      4        1.0747             nan     0.1000    0.0124
    ##      5        1.0501             nan     0.1000    0.0103
    ##      6        1.0338             nan     0.1000    0.0059
    ##      7        1.0127             nan     0.1000    0.0088
    ##      8        0.9980             nan     0.1000    0.0044
    ##      9        0.9863             nan     0.1000    0.0040
    ##     10        0.9772             nan     0.1000    0.0009
    ##     20        0.9104             nan     0.1000    0.0002
    ##     40        0.8438             nan     0.1000   -0.0009
    ##     60        0.8049             nan     0.1000   -0.0033
    ##     80        0.7712             nan     0.1000   -0.0023
    ##    100        0.7440             nan     0.1000   -0.0023
    ##    120        0.7106             nan     0.1000   -0.0020
    ##    140        0.6770             nan     0.1000   -0.0015
    ##    150        0.6683             nan     0.1000   -0.0027
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1869             nan     0.1000    0.0291
    ##      2        1.1327             nan     0.1000    0.0233
    ##      3        1.0981             nan     0.1000    0.0165
    ##      4        1.0628             nan     0.1000    0.0142
    ##      5        1.0386             nan     0.1000    0.0097
    ##      6        1.0185             nan     0.1000    0.0076
    ##      7        0.9986             nan     0.1000    0.0081
    ##      8        0.9803             nan     0.1000    0.0068
    ##      9        0.9641             nan     0.1000    0.0041
    ##     10        0.9527             nan     0.1000    0.0038
    ##     20        0.8746             nan     0.1000   -0.0002
    ##     40        0.7930             nan     0.1000    0.0001
    ##     60        0.7305             nan     0.1000   -0.0014
    ##     80        0.6760             nan     0.1000   -0.0037
    ##    100        0.6413             nan     0.1000   -0.0040
    ##    120        0.6050             nan     0.1000   -0.0019
    ##    140        0.5667             nan     0.1000   -0.0020
    ##    150        0.5490             nan     0.1000   -0.0011
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1917             nan     0.1000    0.0255
    ##      2        1.1535             nan     0.1000    0.0201
    ##      3        1.1247             nan     0.1000    0.0142
    ##      4        1.1026             nan     0.1000    0.0126
    ##      5        1.0808             nan     0.1000    0.0088
    ##      6        1.0607             nan     0.1000    0.0074
    ##      7        1.0468             nan     0.1000    0.0078
    ##      8        1.0362             nan     0.1000    0.0057
    ##      9        1.0260             nan     0.1000    0.0043
    ##     10        1.0219             nan     0.1000   -0.0009
    ##     20        0.9698             nan     0.1000    0.0000
    ##     40        0.9268             nan     0.1000   -0.0005
    ##     50        0.9172             nan     0.1000   -0.0025

``` r
print(model_gbm)
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 461 samples
    ##  18 predictor
    ##   2 classes: '0', '1' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold, repeated 5 times) 
    ## Summary of sample sizes: 369, 368, 369, 369, 369, 369, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  Accuracy   Kappa    
    ##   1                   50      0.8038960  0.4663209
    ##   1                  100      0.8025632  0.4698574
    ##   1                  150      0.7986831  0.4634829
    ##   2                   50      0.7956534  0.4511798
    ##   2                  100      0.7943115  0.4586859
    ##   2                  150      0.7834699  0.4353282
    ##   3                   50      0.7934133  0.4530930
    ##   3                  100      0.7851807  0.4440514
    ##   3                  150      0.7742820  0.4278140
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## 
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final values used for the model were n.trees = 50, interaction.depth
    ##  = 1, shrinkage = 0.1 and n.minobsinnode = 10.

``` r
plot(model_gbm)
```

![](Loan_prediction_using_Caret_files/figure-markdown_github/unnamed-chunk-15-1.png)

``` r
#training our model for logistic regression model
model_glm=train(trainset[,predictors],trainset[,outcomeName],method='glm',trControl=fitControl)
print(model_glm)
```

    ## Generalized Linear Model 
    ## 
    ## 461 samples
    ##  18 predictor
    ##   2 classes: '0', '1' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold, repeated 5 times) 
    ## Summary of sample sizes: 370, 369, 368, 369, 368, 369, ... 
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa    
    ##   0.7887478  0.4297176
    ## 
    ## 

Now lets look at the important variables in our model

``` r
#Checking variable importance for GLM
varImp(object = model_glm)
```

    ## glm variable importance
    ## 
    ##                         Overall
    ## Credit_History         100.0000
    ## Property_AreaSemiurban  31.6526
    ## EducationNot.Graduate   17.3784
    ## CoapplicantIncome       13.5682
    ## LoanAmount              10.4747
    ## Dependents1              8.7339
    ## Dependents3.             7.6814
    ## GenderMale               3.9674
    ## GenderFemale             2.8886
    ## ApplicantIncome          2.8589
    ## Self_EmployedNo          2.6464
    ## Dependents0              2.5555
    ## Property_AreaUrban       2.2327
    ## Self_EmployedYes         2.0046
    ## Loan_Amount_Term         1.4348
    ## MarriedNo                0.1319
    ## MarriedYes               0.1205
    ## Dependents2              0.0000

``` r
plot(varImp(object = model_glm),main="GLM- Variable importance")
```

![](Loan_prediction_using_Caret_files/figure-markdown_github/unnamed-chunk-16-1.png) Finally its time to do prediction on the test set

``` r
# Prediction for Logistic regression model
predictions=predict.train(object = model_glm,testset[,predictors],type = "raw")
table(predictions)
```

    ## predictions
    ##   0   1 
    ##  27 126

``` r
plot(predictions,main="Prediction report")

# Prediction for GBM  model
predictions2=predict.train(object = model_gbm,testset[,predictors],type = "raw")
table(predictions2)
```

    ## predictions2
    ##   0   1 
    ##  27 126

``` r
plot(predictions2,main = "Prediction report")
```

![](Loan_prediction_using_Caret_files/figure-markdown_github/unnamed-chunk-17-1.png) We find that both the model perform really well.The %of data captured looks good.
