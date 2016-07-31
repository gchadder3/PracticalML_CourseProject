## PA1 Raw Script Code for Practical Machine Learning Course
##
## Last update: 7/30/16 (George Chadderdon)

## Load the dplyr package.
library(dplyr)

##
## Raw data read-in
##

trainset <- read.csv("pml-training.csv")
testset <- read.csv("pml-testing.csv")

##
## Data Cleaning
##

## Remove the extra row index column X.
trainset <- select(trainset, -X)
testset <- select(testset, -X)

## Remove columns where the test set has all NAs; these features are 
## useless to use (at least on the test set).
NAcols <- which(apply(is.na(testset), 2, sum) == 20)
testset <- select(testset, -NAcols)
trainset <- select(trainset, -NAcols)

## Remove the early, non-feature columns.
trainset <- select(trainset, -(user_name:num_window))
testset <- select(testset, -(user_name:num_window))

##
## Predictive Modeling
##

## Load the caret package.
library(caret)

## Partition the training dataset so that we have 70% of the data in the 
## (sub-)training set.
set.seed(1)
trainPart <- createDataPartition(trainset$classe, p=0.7)[[1]]
trainsubset <- trainset[trainPart,]
testsubset <- trainset[-trainPart,]



## Fit a CART decision tree.

## Takes less than a minute to train.
modelFit.cart <- train(classe ~ ., method="rpart", data=trainsubset)
## Accuracy is 51.5% on training set.

## Test on the test subset.
testPreds.cart <- predict(modelFit.cart, testsubset)
confusionMatrix(testPreds.cart, testsubset$classe)
## Accuracy is 49.1% on this set.


## Fit a random forest.

## This took about 2 hours to run!
## Accuracy on the training set is 98.9%.
modelFit.rf <- train(classe ~ ., method="rf", data=trainsubset)

## Test on the test subset.
testPreds.rf <- predict(modelFit.rf, testsubset)
confusionMatrix(testPreds.rf, testsubset$classe)
## Accuracy is 99.5% on this set!

## Get predictions for the final test set.
bigTestPreds.rf <- predict(modelFit.rf, testset)
#  [1] B A B A A E D B A A B C B A E E A B B B
# All of these were correct for the quiz.


## Train a CART model using 10-fold cross-validation.
set.seed(1)
fitControl <- trainControl(method="repeatedcv", number=10, repeats=1)
modelFit.cart <- train(classe ~ ., method="rpart", trControl=fitControl, 
    data=trainset)

## Show the training set performance.
trainPreds.cart <- predict(modelFit.cart, trainset)
CM.cart <- confusionMatrix(trainPreds.cart, trainset$classe)
CM.cart$overall[1]
CM.cart$table  ## confusion matrix

## Show the decision tree architecture.
fancyRpartPlot(modelFit.cart$finalModel)


## Train a random forest model using 10-fold cross-validation.
set.seed(1)
fitControl <- trainControl(method="repeatedcv", number=10, repeats=1)
modelFit.rf <- train(classe ~ ., method="rf", trControl=fitControl, 
    data=trainset)

## Show the training set performance.
trainPreds.rf <- predict(modelFit.rf, trainset)
CM.rf <- confusionMatrix(trainPreds.rf, trainset$classe)
CM.rf$overall[1]
CM.rf$table  ## confusion matrix

## Get predictions for the final test set.
bigTestPreds.rf <- predict(modelFit.rf, testset)
#  [1] B A B A A E D B A A B C B A E E A B B B
# All of these were correct for the quiz.


# modelFit <- train(classe ~ ., method="glm", family="binomial", data=trainsubset)
