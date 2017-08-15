require(mlbench)
data(Vehicle)

library(tree)
library(boot)

set.seed(0923)

## the full data set


model.1 = tree(Class ~., data=Vehicle)         # tree model on the full data set
predict.1 = predict(model.1, type='class')     # perform on the full data set
tab.1 = table(Vehicle$Class, predict.1)        # confusion matrix

print(tab.1)
print(paste('The prediction accuracy rate is', 
            round(sum(diag(tab.1))/sum(tab.1),5)))   # prediction accuracy 


## 10-fold cross validation

## I used the cross-validation code from diary0221 


K = 10    # number of fold
N = 846   # number of observations 

cvindex = rep(1:K, length=N)
cvindex = sample(cvindex, N, replace=F)

confusion = matrix(rep(0,4), ncol=4, nrow=4) # create a empty confusion matrix 

for (j in 1:K)
{  
  testset = cvindex==j              
  testdf <- Vehicle[testset,]       # Define the test set
  traindf <- Vehicle[-testset,]     # Build the train set
  
  tree.fit <- tree(Class ~., data = traindf)   # fit a tree model on the train set
  pred <- predict(tree.fit, newdata = testdf, type = 'class')  # perform on the test set
  confusion <- confusion + table(testdf$Class, pred)   # confusion matrix
}

print(confusion)
print(paste('From 10-fold cross validation, the prediction accuracy rate is',
            round(sum(diag(confusion))/sum(confusion),5)))   # prediction accuracy

set.seed(0923)

## 10-fold cross validation

## I used the PCA code from diary0418 


Vehicle1 = Vehicle[,-19]    # remove the class from the original dataset 
Vehicle.pca = prcomp(as.matrix(Vehicle1), scale = TRUE)   # scaling the predictors
X = Vehicle.pca$x  # pca matrix with 18 predictors


K = 10    # number of fold
N = 846   # number of observations 

cvindex = rep(1:K, length=N)
cvindex = sample(cvindex, N, replace=F)

confusion = matrix(rep(0,4), ncol=4, nrow=4)

pca.fold <- function(K,m)    # create a function that computes the misclassification rate
  # of tree model of k first principal components using confusion
  # matrix
{
  X = data.frame(X[,1:m])
  X$Class = Vehicle$Class    # add Class to X
  for(j in 1:K)
  {
    testset = cvindex==j
    testdf <- X[testset,]
    traindf <- X[-testset,]
    
    tree.fit <- tree(Class ~., data=traindf)   # fit a tree model 
    pred <- predict(tree.fit, newdata=testdf, type='class')   # perform on the test
    confusion <- confusion + table(testdf$Class, pred)   # confusion matrix
  }
  return(1 - round(sum(diag(confusion))/sum(confusion),5)) # misclassification rate
}

# tree model of 1 to 18 first principal components 

pca.list <- c(pca.fold(10,1), pca.fold(10,2), pca.fold(10,3), pca.fold(10,4),
              pca.fold(10,5), pca.fold(10,6), pca.fold(10,7), pca.fold(10,8),
              pca.fold(10,9), pca.fold(10,10), pca.fold(10,11), pca.fold(10,12),
              pca.fold(10,13), pca.fold(10,14), pca.fold(10,15), pca.fold(10,16),
              pca.fold(10,17), pca.fold(10,18))  

print('The misclassification rates are listed from k first principals components = 1 to 18:')
print(pca.list) 
print(paste('From 10-fold cross validation, the best k is', which.min(pca.list)))
plot(pca.list, xlab = 'pca.fold(10,x)', type = 'b')

library(randomForest)

set.seed(0923)

## 10-fold cross validation


K = 10
N = 846

cvindex = rep(1:K, length=N)
cvindex = sample(cvindex, N, replace=F)

confusion = matrix(rep(0,4), ncol=4, nrow=4)


rand.fold <- function(K, m)   # create a function that computes the misclassification rate
  # of a random forest model of m maxnodes using confusion
  # matrix
{
  for (j in 1:K)
  {
    testset = cvindex==j
    testdf <- Vehicle[testset,]
    traindf <- Vehicle[-testset,]
    
    rf.fit <- randomForest(Class ~., data=traindf, mtry=sqrt(18), 
                           maxnodes=m)    # fit a random forest model 
    pred <- predict(rf.fit, newdata=testdf, type='class')   # perform on the test set
    confusion <- confusion + table(testdf$Class, pred)   # confusion matrix 
  }
  return(1 - round(sum(diag(confusion))/sum(confusion),5))  # the misclassification rate
}

# random forest model of 2 to 10 maxnodes 

rand.list <- c(NA, rand.fold(10,2), rand.fold(10,3), rand.fold(10,4),rand.fold(10,5),    
               rand.fold(10,6), rand.fold(10,7), rand.fold(10,8), rand.fold(10,9), 
               rand.fold(10,10))

print('The misclassification rates are listed from maxnodes = 2 to 10:')
print(rand.list)
print(paste('From 10-fold cross validation, the best m is', which.min(rand.list)))
plot(c(rand.list) , xlim = c(2,10), type = 'b', xlab = 'rand.fold(10,x)')

set.seed(0923)

## a subset of vehicle data of Saabs and Opels only


subVehicle <- subset(Vehicle, Vehicle$Class == 'opel' | Vehicle$Class == 'saab')
glm.fit <- glm(Class ~., data = subVehicle, family = 'binomial')
summary(glm.fit)

set.seed(0923)

## 10-fold cross validation
## containing only the five predictors stated above


subVehicle1 = subVehicle[c('Holl.Ra', 'Kurt.Maxis', 'Ra.Gyr', 'Circ', 'Comp')]
subVehicle1$Class = subVehicle$Class


K = 10     # the number of fold
N1 = 429   # the number of observations of the subset

cvindex1 = rep(1:K, length = N1)
cvindex1 = sample(cvindex1, N1, replace = F)

confusion1 = matrix(rep(0,2), ncol = 4, nrow = 2)


for (j in 1:10)
{
  testset1 = cvindex1 == j
  testdf1 <- subVehicle1[testset1,]
  traindf1 <- subVehicle1[-testset1,]
  
  glm.fit1 <- glm(Class ~., data = traindf1, family = 'binomial')  
  prob <- predict(glm.fit1, newdata=testdf1, type = 'response') 
  
  pred <- rep("down", length(prob))
  pred[prob >.5] = "up"
  
  tab <- table(pred,testdf1$Class)
  confusion1 <- confusion1 + tab
}
confusion1 <- confusion1[,2:3]

print(confusion1)
print(round(sum(diag(confusion1))/sum(confusion1),5))