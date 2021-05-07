#Name: Harini G
#Roll No: 2048034
#Lab5
#1.Install the package “titanic”
#install.packages("titanic")

#2.Load Titanic library to get the dataset
library("titanic")
titanic_gender_class_model
titanic_gender_model
titanic_test
titanic_train
?titanic_test
#Inference:The train data contains 12 attributes with 891 observations
#The test data contains 11 attributes with 418 observations.(The test data doesn't contains the survived attribute)

#3.Set Survived column for test data to NA
data(titanic_test, package="titanic")
titanic_test
str(titanic_test)
dim(titanic_test)
titanic_test$Survived <- NA
titanic_test
data(titanic_train, package="titanic")
titanic_train
str(titanic_train)
dim(titanic_train)
#titanic_train$Survived <- NA

#4.Combine the  Training and Testing dataset
complete_data <- rbind(titanic_train, titanic_test)
complete_data

#5.Get the data structure
str(complete_data)
dim(complete_data)

#6. Check for any missing values in the data
colSums(is.na(complete_data))
#inference:The Survived data contains 418 missing values, the Age contains 263 missing values and Fare contains 1 missing data

#7.Check for any empty values
colSums(complete_data=="")
#inference:The survived coulmn, the age column, cabin, Embarked and fare contains empty values

#8.Check number of unique values for each column to find out which column we can convert to factors
apply(complete_data,2, function(x) length(unique(x)))
unique(complete_data$PassengerId )
unique(complete_data$Survived)
unique(complete_data$Pclass)
unique(complete_data$Name)
unique(complete_data$Sex)
unique(complete_data$Age)
unique(complete_data$SibSp)
unique(complete_data$Parch)
unique(complete_data$Ticket)
unique(complete_data$Fare)
unique(complete_data$Cabin)
unique(complete_data$Embarked)

#9.Remove Cabin as it has very high missing values, passengerId, Ticket and Name are not required
#drop=c("Cabin")
data_1 = subset(complete_data, select = -c(Cabin,PassengerId,Ticket,Name) )
data_1

#10.Convert "Survived","Pclass","Sex","Embarked" to factors
data_1$Sex <- ifelse(data_1$Sex == "female", 1, 0)
data_1$Sex <- factor(data_1$Sex, levels = c(0, 1))
class(data_1$Sex)

data_1$Embarked <- factor(data_1$Embarked, levels=c("S","C","Q",""), labels=c(0,1,2,3))
data_1$Embarked
class(data_1$Embarked)

data_1$Survived <- factor(data_1$Survived)
class(data_1$Survived)

data_1$Pclass <- factor(data_1$Pclass)
class(data_1$Pclass)

colSums(is.na(data_1))

data_1$Age[is.na(data_1$Age)] <- round(mean(data_1$Age, na.rm = TRUE))

colSums(is.na(data_1))

library("tidyr")
data_1 <- data_1 %>% drop_na() 
sum(is.na(data_1))
colSums(is.na(data_1))

#11.Splitting training and test data
dim(data_1)
trainingRowIndex=sort(sample(nrow(data_1), nrow(data_1)*.7))
trainingData=data_1[trainingRowIndex,]
trainingData
testData=data_1[-trainingRowIndex,]
testData
dim(trainingData)
dim(testData)

#12.Create a model 
lmModel=glm(Survived~.,family=binomial,data=trainingData)
summary(lmModel)

pred<-predict(lmModel,testData,type="response")
s_pred_num <- ifelse(pred > 0.5, 1, 0)
s_pred <- factor(s_pred_num, levels=c(0, 1))
s_pred

#install.packages("pscl")
pscl::pR2(lmModel)["McFadden"]
#As the value is less than 0.4 so, this model is not best fit for the prediction*

#install.packages("caret")
caret::varImp(lmModel)
#Attribute Embarkeds and Fare is not that important for model prediction

#install.packages("car")
car::vif(lmModel) #Checking for multicollinearity
#None of the attributes values is greater than 5. So, the given data doesn't suffer from multi collinearity

#13.Visualize the model summary
library(rpart)
#install.packages("rpart.plot")
library(rpart.plot)
model_dt<- rpart(Survived ~.,data=trainingData)
rpart.plot(model_dt)

#14.Analyse the test of deviance using anova()
anova(lmModel, test="Chisq")

#15.Compute confusion matrix and ROC curve
#confusion matrix
library(caret)
confusionMatrix(s_pred,testData$Survived)

#ROC curve
#install.packages("ROCR")
library(ROCR)
ROCRpred <- prediction(pred, testData$Survived)
ROCRperf <- performance(ROCRpred, measure = "tpr", x.measure = "fpr")
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7), print.cutoffs.at = seq(0,1,0.1))
auc<-performance(ROCRpred,measure="auc")
auc<-auc@y.values[[1]]
auc
#Therefore, the prediction model is 82% correctly categorizing the data