library(tidyverse)
library(dlookr)
library(janitor)

#load loan predicition dataset 
Customer_loan <- read.csv('Customer_loan.csv')
# To view  data structure 
glimpse(Customer_loan)
str(Customer_loan)
View(Customer_loan)

# using the dlookr package to scurinize data
Customer_loan %>% group_by() %>% describe() %>% view()

head(Customer_loan, 10)

summary(Customer_loan)

#check for missing values
sum(is.na(Customer_loan))

#Calculating the Debt to income ratio and adding new column as Dti
refined_Data <- Customer_loan %>% mutate(Dti = debts/income)

#Creating a new variable named 'loan_decision_status where the value would be '0 of  loan decison is equal to denied else 1

refined_Data <- refined_Data %>% mutate(loan_decision_status = ifelse(loan_decision_type == 'Denied',0,1))
head(refined_Data)

# Visualizing  loan decision status
ggplot(refined_Data, aes(x = factor(loan_decision_status), fill = factor(loan_decision_status))) +
  geom_bar() +
  labs(title = "Loan Decision Status",
       x = "Loan Decision Status", 
       y = "Count",
       fill = "Loan Decision Status")


#converting the loan_decision_status variable into a factor
refined_Data$loan_decision_status <- as.factor(refined_Data$loan_decision_status)

#checking the data type of the column
str(refined_Data$loan_decision_status)


#Creating a new data-set named ‘customer_loan_refined’, which would have these column numbers from the original dataframe - (3,4,6,7,8,11,13,14)

Customer_loan_refined <- refined_Data %>%  select(3,4,6,7,8,11,13,14)
head(Customer_loan_refined)
  
#Encode ‘gender’, ‘marital_status’, ‘occupation’, and ‘loan_type’ as factors and then convert them into numeric
# encode categorical variables
Customer_loan_refined <- Customer_loan_refined %>%
  mutate(gender = factor(gender),
         marital_status = factor(marital_status),
         occupation = factor(occupation),
         loan_type = factor(loan_type)) %>%
  mutate(across(c(gender, marital_status, occupation, loan_type), as.numeric))

# check the new data frame
str(Customer_loan_refined)

#Building the model and dividing data into train and test sets and split ratio as 70:30
install.packages("caret")
library(caret)
 # create  training and test datasets

X <- Customer_loan_refined %>%  select(-loan_decision_status)
Y <- Customer_loan_refined$loan_decision_status

#Setting random seed for reproducibility

set.seed(123)

# splitting the data set into train and test set respectively using "CreateDataPartition()

refined_train <- createDataPartition(Y, p = 0.7, times = 1, list = FALSE)

X_train <- X[refined_train, ]
X_test <- X[-refined_train, ]
Y_train <- Y[refined_train]
Y_test <- Y[-refined_train]

#Applying feature scaling on all columns except  "loan_decision_status" , In order to standardize the range of values and improve model performance

Scaled_set <- preProcess(X_train, method = c('center', 'scale'))
X_train_scaled <- predict(Scaled_set, X_train)
X_test_scaled <- predict(Scaled_set, X_test)


#Applying  principal component analysis on the first 7 columns of ‘train’ & ‘test’ set and  The number of principal components obtained should be 2

Pca_df <- prcomp(X_train [, 1:7], center = TRUE, scale. = TRUE, retx = TRUE)

X_train_pca <- predict(Pca_df, X_train_scaled [, 1:7])
X_train_pca <- X_train_pca[, 1:2]
X_test_pca <- predict(Pca_df, X_test_scaled [, 1:7])
X_test_pca <- X_test_pca [, 1:2]


# Using the Naive Bayes on the train set

library(e1071)

model_data  <- naiveBayes(X_train_pca, Y_train)



#predicting values of the test set

Pred <- predict(model_data, X_test_pca)

#building the confusion matrix

confusionMatrix(Pred, Y_test)


#Based on the confusion matrix result, it can be deduced that the model has a high accuracy of 85.5%.
