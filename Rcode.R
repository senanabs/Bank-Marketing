# setting working directory
setwd("~/Desktop/Quicken")

# data import
data<-read.csv("dataset.csv")

# checking the class to see if data.frame
class(data)

# checking number of rows and columns
dim(data)

# checking for any missing data in numerical columns
any(is.na(data))

# Removing all unknowns from categorical variables
data2 <- data[rowSums(data == "unknown")==0, , drop = FALSE]

# Create a new column and mark pdays < 999 as yes and pdays = 999 as no
data2$Was_Previously_Contacted[data2$pdays < 999] <- "Yes"
data2$Was_Previously_Contacted[data2$pdays == 999] <- "No"

# Drop pdays
data2 <- data2[,-13]

# Write cleaned dataset to a new csv file
write.csv(data2, "~/Desktop/Quicken/cleaned.csv", row.names=FALSE)

# Import cleaned data
data<-read.csv("cleaned.csv")

# Changing variables from character to factor
data$y <- as.factor(data$y)
data$job <- as.factor(data$job)
data$marital <- as.factor(data$marital)
data$education <- as.factor(data$education)
data$default <- as.factor(data$default)
data$housing <- as.factor(data$housing)
data$loan <- as.factor(data$loan)
data$contact <- as.factor(data$contact)
data$month <- as.factor(data$month)
data$day_of_week <- as.factor(data$day_of_week)
data$poutcome <- as.factor(data$poutcome)
data$Was_Previously_Contacted <- factor(data2$Was_Previously_Contacted)

# Quick summary of the data
summary(data)

# histograms of numerical columns to examine the underline distribution
library(plyr)
hist(data2$age)
hist(data2$duration)
hist(data2$campaign)
hist(data2$previous)
hist(data2$emp.var.rate)
hist(data2$cons.price.idx)
hist(data2$cons.conf.idx)
hist(data2$euribor3m)
hist(data2$nr.employed)

# boxplots of numerical columns to examine outliers
boxplot(data2$age)
boxplot(data2$duration)
boxplot(data2$campaign)
boxplot(data2$previous)
boxplot(data2$emp.var.rate)
boxplot(data2$cons.price.idx)
boxplot(data2$cons.conf.idx)
boxplot(data2$euribor3m)
boxplot(data2$nr.employed)

#####Logistic Regression will be utilized for analysis#####

# Building a logistic regreesion model

library(tidyverse)
library(broom)
theme_set(theme_classic())

# Fit the logistic regression model
model <- glm(y ~. , data=data, family=binomial)
summary(model)

# Predict the probability (p)
probabilities <- predict(model, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, "pos", "neg")
head(predicted.classes)

## Checking for assumptions##

## Linearity Assumption

# Remove qualitative variables from the original data frame and bind the logit values to the data:

library(dplyr)
# Select only numeric predictors
linearity <- data2 %>%
  dplyr::select_if(is.numeric) 
predictors <- colnames(linearity)
# Bind the logit and tidying the data for plot
linearity <- linearity %>%
  mutate(logit = log(probabilities/(1-probabilities))) %>%
  gather(key = "predictors", value = "predictor.value", -logit)

# Create the scatter plots:
ggplot(linearity, aes(logit, predictor.value))+
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = "loess") + 
  theme_bw() + 
  facet_wrap(~predictors, scales = "free_y")

# Influential values with with Cook's distance values
plot(model, which = 4, id.n = 3)

# Multicollinearity
library(car)
car::vif(model)

# Removed some variables to get multicollinearity under control
model <- glm(y ~.  - month - duration -euribor3m -emp.var.rate -poutcome, data = data, family = binomial)
car::vif(model)

# Perform Lasso Regression to force coefficients of some less contributive variables to be zero

library(tidyverse)
library(caret)
library(glmnet)

# Split the data into training and test set
set.seed(123)
training.samples <- data$y %>% 
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- data[training.samples, ]
test.data <- data[-training.samples, ]


xfactors <- model.matrix(y ~ job + marital + education + default + housing + loan + contact + day_of_week + Was_Previously_Contacted, data=train.data)[, -1]
x        <- as.matrix(data.frame(train.data$age, train.data$campaign, train.data$previous, train.data$cons.price.idx, train.data$cons.conf.idx, train.data$nr.employed, xfactors))
y <- as.factor(train.data$y)

#####Fit the lasso regression model#####

# Find the best lambda using cross-validation
set.seed(123) 
cv.lasso <- cv.glmnet(x, y, alpha = 1, family = "binomial")
plot(cv.lasso)

# Lambda value that minimizes the prediction error
cv.lasso$lambda.min

# Lambda value with the smallest number of predictors that also gives a good accuracy
cv.lasso$lambda.1se

# coefficients with lambda.min
coef(cv.lasso, cv.lasso$lambda.min)

# coefficients with lambda.1se
coef(cv.lasso, cv.lasso$lambda.1se)

# Model fit for variables from lambda.min
lasso.model <- glm(y ~. - month - duration -euribor3m -emp.var.rate -poutcome - age - default - housing - loan, data=data, family=binomial)
summary(lasso.model)

library(DAAG)
cvFitLasso <- CVbinary(lasso.model)

# ROC curve for lambda.min
library(pROC)
roc.curve.lasso <- roc(lasso.model$y, lasso.model$fitted.values, data = data)
plot(roc.curve.lasso)    
auc(roc.curve.lasso)

# confusion matrix for lambda.min
library(regclass)
confusion_matrix(lasso.model, DATA = data)

# Model fit for variables from lambda.1se
lasso.model <- glm(y ~ previous + cons.conf.idx + nr.employed + job + education + contact  + day_of_week + Was_Previously_Contacted, data=data, family=binomial)
summary(lasso.model)

library(DAAG)
cvFitLasso <- CVbinary(lasso.model)

# ROC curve for lambda.1se
library(pROC)
roc.curve.lasso <- roc(lasso.model$y, lasso.model$fitted.values, data = data)
plot(roc.curve.lasso)    
auc(roc.curve.lasso)

# Confusion matrix for lambda.1se
library(regclass)
confusion_matrix(lasso.model, DATA = data)

# Odds Ratios for lambda.1se
exp(coef(lasso.model))














