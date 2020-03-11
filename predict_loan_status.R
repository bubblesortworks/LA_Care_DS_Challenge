
# download and unzip file
download.file("https://s3.us-east-2.amazonaws.com/example.data/lacare_ml_challenge.zip", destfile = "lacare_ml_challenge.zip")
unzip("lacare_ml_challenge.zip")

# if the following packages need to be installed, unquote the installation code

# install.packages("readr")
# install.packages("gbm")
# install.packages("dplyr")
# install.packages ("pROC")

library(readr)
library(gbm)
library(dplyr)
library(pROC)


data <- read.csv("loan_data.csv")

# data variables characteristics and stats
summary(data)


# dependent variable responses: "Charged Off" & "Fully Paid"
unique(data$loan_status)


# variables out_prncp and out_prncp_inv have only value 0. should not be included in model
# some of numeric/integer variables have NA (NULL) in them. We use mean imputation to fill them

#mean imputation for numeric/integer NA values
for (var in colnames(data)){
        if (class(data[,var]) %in% c("integer","numeric")) {
                data[is.na(data[,var]),var] <- mean(data[,var], na.rm = TRUE) 
        }
        
}


# create a binary variable (0/1) for loan_status
data$loan_status_01 <- case_when( data$loan_status == "Fully Paid" ~ 1,
                                  TRUE ~ 0)

# 1 million rows of data, only 2% of data has value "Charged Off"
print(NROW(data))
print(NROW(data[data$loan_status_01==0,])/NROW(data))

# rare event -- we do stratified sampling, 
# assigning 80% of 0s and 80% of 1s to train, and 20% rest to test
data_0 <- data[data$loan_status_01 == 0,]
data_1 <- data[data$loan_status_01 == 1,]

smp_0 <- sample (1:NROW(data_0), 0.8 * NROW(data_0), replace = FALSE)
smp_1 <- sample (1:NROW(data_1), 0.8 * NROW(data_1), replace = FALSE)

data_train <- rbind.data.frame(data_0[smp_0,], data_1[smp_1,])
data_test <- rbind.data.frame(data_0[-smp_0,], data_1[-smp_1,])

# NROW(data_train)
# NROW(data_test)

# One possible approach for modeling is logistic/linear regression. For that approach we would
# need to create dummy variables for categorical independent variables. However we are going
# to use GBM which is boosted tree and works with categorical as well as numeric variables.
# no need for dummy variables

# GBM takes 7-10 minutes to train
gbm_model <- gbm(formula = 
                         loan_status_01 ~ grade + total_rec_late_fee + term + dti + home_ownership + addr_state + emp_length + annual_inc + verification_status + inq_last_6mths ,
                   distribution = "bernoulli"
                   ,data = data_train
                   ,n.trees = 500
                   ,shrinkage = 0.01
                   ,n.minobsinnode = 10)


# relative influence of predictors
summary(gbm_model)


# predict probability of fully paid for test dataset
test_predict <- predict(object = gbm_model, 
                        newdata = data_test, 
                        n.tree = 500,
                        type = "response")


# add the probabilities to the test dataset
data_test$probability_fully_paid <- test_predict

# Model predictive power
ROC_curve <- roc(data_test$loan_status_01 ~ data_test$probability_fully_paid)

# AUC ROC about 71%
print(ROC_curve$auc)

# draw the AUC curve
par(mfrow=c(1,1))
plot(ROC_curve,
     cex.main=0.8,
     xlab = "False Positive Rate",
     ylab = "True Positive Rate",
     legacy.axes = TRUE,
     main = paste("Gradient Boosting Machine ROC,
                  AUC = ",
                  round(auc(ROC_curve)[1],2))
)



# to calculate the optimum cut-off value for the probability of "fully paid"  to classify 
# the test data as fully paid or charged off, we need to 
# have weights for misclassifying "fully paid" and "charged off". We assume the weight of
# misclassifying charged off is 10 times that of fully paid -- reasoning is institution makes 
# 5% profit - interest rate - while charge off means losing 50% of the loan amount on average

cost <- 0
for (i in seq(0,1,0.001)){
        # current_cutoff <- data_test[i,]$probability_fully_paid
        current_cutoff <- quantile(data_test$probability_fully_paid, i, type = 3)
        # number of "fully paid" misclassifications
        n_FN <- NROW(data_test[data_test$loan_status_01 == 1 & data_test$probability_fully_paid < current_cutoff,])
        
        # number of "charge off" misclassification
        n_FP <- NROW(data_test[data_test$loan_status_01 == 0 & data_test$probability_fully_paid > current_cutoff,])
        
        # cost of all misclassifications
        current_cost <- n_FN + (10 * n_FP)

        # if this loop's calculated cost is less than all previous ones, then store current cutoff and cost
        # exception for loop 1 when cost and cutoff are assigned anyway
        if (current_cost < cost | i == 0){
                cost <- current_cost
                cutoff <- current_cutoff
                which_i <- i
        }
}

# cutoff value
cat(cutoff)

# predict loan_status
data_test$predict_loan_status <- ifelse(data_test$probability_fully_paid <= cutoff, "Charged Off", "Fully Paid")

# Positive: Fully Paid
# Negative: Charged Off

# TP
NROW(data_test[data_test$loan_status == "Fully Paid" & data_test$predict_loan_status == "Fully Paid",])

# TN
NROW(data_test[data_test$loan_status == "Charged Off" & data_test$predict_loan_status == "Charged Off",])

# FP
NROW(data_test[data_test$loan_status == "Charged Off" & data_test$predict_loan_status == "Fully Paid",])

# FN
NROW(data_test[data_test$loan_status == "Fully Paid" & data_test$predict_loan_status == "Charged Off",])


