#7143-data mining-group project

#loading libraries
library(caret)
library(nnet)
library(neuralnet)
library(forecast)
library(FNN)
library(pROC)
library(RColorBrewer)
library(xgboost)
library(adabag)
library(ipred)
library(rpart)
library(rpart.plot)

#read the data
usbank<-read.csv('Churn_Modelling.csv')

#delete unrelated columns, like "RowNumber", "CustomerId" and "Surname"
usbank<-usbank[,-(1:3)]

summary(usbank)
sum(usbank$Exited)
#2037 1, 7963 0
#highly unbalanced
#use upsampling to fix
#Upsampling: this method increases the size of the minority class by sampling with replacement so that the classes will have the same size.

#upsampling
set.seed(123)
usbank$Exited<-as.factor(usbank$Exited)
usbank<-upSample(x=usbank[,-which(names(usbank)=="Exited")],y=usbank$Exited)

#now we have same amount of 0 and 1
table(usbank$Class)

#change the variable name back to "Exited"
colnames(usbank)[11] <- "Exited"
usbank$Exited<-as.numeric(as.character(usbank$Exited))

#create dummy variables for the categorical values, "Geography" and "Gender"
#Gender only has 2 categories "Female" and "Male", we create one dummy variable
usbank$Gender_Female<-1*(usbank$Gender=='Female')
#Geography has 3 categories "France"  "Spain"   "Germany", we create 2 dummy variable
usbank$Geography_France<-1*(usbank$Geography=='France')
usbank$Geography_Spain<-1*(usbank$Geography=='Spain')

usbank<-usbank[,-which(names(usbank) %in% c("Gender","Geography"))]

#partitioning, we have 10000 rows data in total, we use 70% of the data as training set, 30% of the data as validation set
set.seed(123)
train.index<-sample(row.names(usbank),0.7*dim(usbank)[1])
valid.index<-setdiff(row.names(usbank),train.index)
train.df<-usbank[train.index,]
valid.df<-usbank[valid.index,]

#normalize the data
norm.values<-preProcess(train.df,method = 'range')
train.norm.df<-predict(norm.values,train.df)
valid.norm.df<-predict(norm.values,valid.df)


##################################################################
#modeling

#bench mark
#The naive model: majority rule
#79.63% 0, 20.37% 1
#prediction made on every new data would be 0
#accuracy would be 79.63%

######################################################################
# logistic regression exploration ----

reg <- glm(Exited ~ ., data = train.norm.df,family = "binomial")
summary(reg)

#find the most suitable cut off value, metrics give the sensitivity and specificity of different cut off values
metrics <- matrix(,nrow = 21, ncol = 2)
i <- 1
for (cut in seq(0,1,0.05)) {
  fitted.values <- (reg$fitted.values > cut) * 1
  metrics[i,] <- confusionMatrix(factor(fitted.values , levels = c(0,1)),
                                 factor(train.norm.df$Exited,
                                        levels = c(0,1)))$byClass[1:2]
  i <- i + 1
}
metrics
#choose from the confusion matrix to find the best cutoff
#we want the TP rate to be as big as possible as we want to predict the ones who are going to leave, but keep the TNR at the reasonable level
#so we focus on the sensitivity
#0.55 seems like the best choice

fitted.values <- (reg$fitted.values > 0.55) * 1
confusionMatrix(factor(fitted.values , levels = c(0,1)),
                factor(train.norm.df$Exited,
                       levels = c(0,1)))
#model perform better
#ROC automates the choice of the cut off
reg_predict <- predict(reg, newdata = valid.norm.df, type = "response")

predicted.values <- (reg_predict > 0.55)*1
reg_cm<-confusionMatrix(factor(predicted.values , levels = c(0,1)),
                factor(valid.norm.df$Exited,
                       levels = c(0,1)))

reg_cm
# Accuracy : 0.7091
#Sensitivity : 0.7782
#Specificity : 0.6392

# Let's look at ROC
roc <- roc(valid.norm.df$Exited,
           predicted.values)
plot(roc)
auc(roc)
#Area under the curve: 0.7087

#overall performance is ok, Accuracy : 0.7091 Sensitivity : 0.7782

#################################################################################################
#Decision Tree
#need to be explainable, not using the normalized data, using the original data
tr<-rpart(Exited~.,data=train.df,minbucket=50, maxdepth=7)
tr
prp(tr)
#pruned tree
pfit<- prune(tr, cp = tr$cptable[which.min(tr$cptable[,"xerror"]),"CP"])
pfit
prp(tr)
#same tree, the tree is already the best

#variable importance
t(t(tr$variable.importance))

#set of rules
tr

#prediction on the validation set
pred <- predict(tr, valid.df)
predicted.values <- (pred > 0.5)*1
#confusion matrix
tr_cm<-confusionMatrix(factor(predicted.values , levels = c(0,1)),factor(valid.df$Exited,levels = c(0,1)))
tr_cm
#Accuracy : 0.7336
#Sensitivity : 0.8356
#better than logistic, not good enough

####################################################################################
#try boosting
train.df.new<-train.df
train.df.new$Exited <- as.factor(train.df.new$Exited)

set.seed(123)
boost <- boosting(Exited ~ ., data = train.df.new)
pred <- predict(boost, valid.df)
boosting_cm<-confusionMatrix(as.factor(pred$class), as.factor(valid.df$Exited))
boosting_cm
#accuracy 0.7846
#sensitivity 0.8061
#Prediction    0    1
#          0 1937  563
#          1  466 1812

#variable importance
t(t(boost$importance))

############################################################
#bagging
bag <- bagging(
  formula = Exited ~ .,
  data = train.df,
  nbagg = 100,
  coob = TRUE,
  control = rpart.control(minsplit = 2, cp = 0)
)

bag

#for validation set
pred <- predict(bag, valid.df)
predicted.values <- (pred > 0.5)*1

bag_cm<-confusionMatrix(factor(predicted.values , levels = c(0,1)),factor(valid.df$Exited,levels = c(0,1)))

#variable importance
#calculate variable importance
VI=varImp(bag)
#Age              221.880514
#Balance          247.697764
#CreditScore      313.228677
#EstimatedSalary  309.651187
#Gender_Female     67.169944
#Geography_France  38.252799
#Geography_Spain   34.458023
#HasCrCard         78.964193
#IsActiveMember    21.911524
#NumOfProducts      5.240094
#Tenure           226.928709


#Financial cost
#keepcost: the cost to keep one customer
#newcost: the cost to attract a new customer
#cm: confusionMatrix
#each time we predict a false positive, we spend the newcost in order to gain one more customer, but actually the customer is not leaving, so we should have only spend the keepcost, thus the cost is newcost-keepcost
#each time we predict a false negative, we spend the keepcost, but also need to spend the newcost to attract one more customer, so the cost is keepcost+newcost
financial_cost<-function(keepcost,newcost,cm){
  cm$table[2]*(newcost-keepcost)+cm$table[3]*(keepcost+newcost)
}
#if we spend 5 euros on attracting one new customer, and 2 euros on keeping one existing customer
#financial cost for naive model
0*(5-1)+sum(valid.df$Exited)*(5+1)
#14250
#financial cost for the logistic regression model
financial_cost(1,5,reg_cm)
#[1] 7274
#financial cost for the decision tree  model
financial_cost(1,5,tr_cm)
#[1] 6848
#financial cost for the boosting model
financial_cost(1,5,boosting_cm)
#[1] 5242
#financial cost for the bagging model
financial_cost(1,5,bag_cm)
#[1] 1292


