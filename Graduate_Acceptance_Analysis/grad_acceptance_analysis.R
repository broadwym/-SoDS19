library(tidyverse)
library(caret)
library(boot)
library(MASS)
library(glmnet)
library(lme4)
library(mctest)
library(ridge)
library(lmtest)
library(DMwR)
library(naniar)
library(ppcor)
library(Hmisc)
library(ridge)

# Clean data
Admission_Predict <- Admission_Predict[,-1]
Admission_Predict$Research <- as.factor(Admission_Predict$Research)
Admission_Predict$University.Rating <- as.factor(Admission_Predict$University.Rating)

# Goal: Predict chance of admission (target) based on predictor variables 
## Explore data
summary(Admission_Predict)
table(Admission_Predict$University.Rating)
plot(Admission_Predict, main = "Matrix Scatterplot")

## Visualize data 
hist(Admission_Predict$GRE.Score) #data fairly normally distributed: use pearson's to determine strenth of correlation 
hist(Admission_Predict$TOEFL.Score)
hist(Admission_Predict$CGPA)

plot(Admission_Predict$GRE.Score, Admission_Predict$Chance.of.Admit) #check for linearity
cor(Admission_Predict$GRE.Score, Admission_Predict$Chance.of.Admit, method = "pearson")

plot(Admission_Predict$TOEFL.Score, Admission_Predict$Chance.of.Admit)
cor(Admission_Predict$TOEFL.Score, Admission_Predict$Chance.of.Admit, method = "pearson")

plot(Admission_Predict$CGPA, Admission_Predict$Chance.of.Admit)
cor(Admission_Predict$CGPA, Admission_Predict$Chance.of.Admit, method = "pearson") #strongest correlation between undergraduate GPA and chance of admission 

# Linear model on dataset 
lm.all <- lm(Chance.of.Admit ~ ., data = Admission_Predict) #lm() assumes normal distribution while glm you can specify distribution 
summary(lm.all) 
## Summary verifies what we already saw using cor(): Letter of rec and undergrad GPA stand out, followed by Research, GRE, and TOEFL scores 
## A) Do we really need all these predictors? No, probably not. Simpler the better. B) Is the model a good fit for the data? We should try multiple metrics to find best model, then find best model with smallest RMSE, highest R-squared. 
plot(lm.all)
confint(lm.all, conf.level = .95)

# Drop predictors 
lm.2 <- lm(Chance.of.Admit ~ GRE.Score + TOEFL.Score + LOR + CGPA, data = Admission_Predict)
summary(lm.2)
plot(lm.2)

# In-sample Metrics: 
## Stepwise to choose best model (best predictors) based on AIC
stepwise.metric <- step(lm(Chance.of.Admit ~ ., data = Admission_Predict)) #metric choosing predictors lm said weren't valueable: stepwise "iffy" on occasion. 

# https://statmodeling.stat.columbia.edu/2009/07/11/when_to_standar/
## Standardize all continous features (Z-score): X = x-mean/std 
## Z-score function  
cont.funct <- function(data) {
  new_data <- ((data - mean(data)) / sd(data)) 
  return(new_data)
}

## Rescale rating on scale of -1 to 1 
## Encode 1 = -1; 2 = -.5; 3 = 0; 4 = .5; 5 = 1
table(Admission_Predict$University.Rating)

standard.CGPA <- cont.funct(Admission_Predict$CGPA)
standard.SOP <- cont.funct(Admission_Predict$SOP)
standard.LOR <- cont.funct(Admission_Predict$LOR)
standard.GRE <- cont.funct(Admission_Predict$GRE.Score)
standard.TOEFL <- cont.funct(Admission_Predict$TOEFL.Score)
standard.Admit <- cont.funct(Admission_Predict$Chance.of.Admit)

## Creating new dataframe with standardized values 
new.admission <- do.call(rbind, Map(data.frame, standard.CGPA=standard.CGPA, standard.SOP=standard.SOP, standard.LOR=standard.LOR, standard.GRE=standard.GRE,standard.TOEFL=standard.TOEFL, standard.Admit=standard.Admit, standard.Rank=Admission_Predict$University.Rating, standard.Research = Admission_Predict$Research))

## Finally, standardize values in University Rank (standard.Rank)
## This is a roundabout way to encode this, but because there were multiple values present at same time, my endoing got messy. That's why I have "high" and "low" temporarily 
new.admission$standard.Rank = gsub(5, "high", new.admission$standard.Rank)
new.admission$standard.Rank = gsub(1, "low", new.admission$standard.Rank)
new.admission$standard.Rank = gsub(2, -0.50, new.admission$standard.Rank)
new.admission$standard.Rank = gsub(3, 0.0, new.admission$standard.Rank)
new.admission$standard.Rank = gsub(4, 0.50, new.admission$standard.Rank)
new.admission$standard.Rank = gsub("high", 1.0, new.admission$standard.Rank)
new.admission$standard.Rank = gsub("low", -1.0, new.admission$standard.Rank)

# Tidying data
new.admission$standard.Rank <- as.numeric(new.admission$standard.Rank)
new.admission$standard.Research <- as.numeric(levels(new.admission$standard.Research))[new.admission$standard.Research]

## Building new model with standardized data 
new.lm <- glm(standard.Admit ~ ., data = new.admission)
summary(new.lm) # Deviance Resid: should be centered around (mean) zero for standard normal (Guaussian) distribution. It is heavy in the tails. ~.09% 
plot(new.lm) # Normal QQ is showing that I have "over dispersed" data. The tails are very fat due to increased outliers. **Look into this.** 
confint(new.lm, conf.level = .95)

# Dropping predictors with a new model 
new.lm <- lm(standard.Admit ~ standard.CGPA + standard.GRE + standard.LOR + standard.TOEFL, data = new.admission)
summary(new.lm) #R-squared will always go up more predictors you add (see lm.all), but this has nearly an equal R-squared and even lower RSE (.415).  
plot(new.lm) #Residuals still decreasing as fitted Y values increase 
lmtest::bptest(new.lm) #Breusch-Pagan test: p-value smaller than .05 so we do have heterodasticity?  

## We need to reevaluate data (possibly use another method to standardize at some point, and remove outliers). For now, use stepwise to pick model 
step(lm(standard.Admit ~ ., data = new.admission)) ## AIC lowest with all variables, still

## K-Fold Cross Validation 
admission.cv <- train(standard.Admit ~ standard.CGPA + standard.GRE + standard.LOR + standard.TOEFL,
                      data = new.admission, 
                      method = "lm",
                      trControl=trainControl(
                        method = "cv",
                        number=10,
                        savePredictions = TRUE,
                        verboseIter = TRUE)
)


admission.cv$results # MAE is 32% (yikes!) with TOEFL and 33% without it. RMSE (lower = better fit) is 51% for model without TOEFL and 49% with it. 

# Back to visualizing my data: are outliers to blame? Over-dispersed data suggests it
ggplot(Admission_Predict, aes(x=University.Rating, y=Chance.of.Admit)) + 
  geom_boxplot(outlier.colour="red")
ggplot(Admission_Predict, aes(x=Research, y=Chance.of.Admit)) + 
  geom_boxplot(outlier.colour="red")

ggplot(data = Admission_Predict, aes(x = GRE.Score, y = Chance.of.Admit)) + 
  geom_point(color='darkblue') +
  geom_smooth(method = "lm", se = FALSE)

ggplot(data = Admission_Predict, aes(x = TOEFL.Score, y = Chance.of.Admit)) + 
  geom_point(color='darkblue') +
  geom_smooth(method = "lm", se = FALSE)

ggplot(data = Admission_Predict, aes(x = CGPA, y = standard.Admit)) + 
  geom_point(color='darkblue') +
  geom_smooth(method = "lm", se = FALSE)

boxplot(standard.Admit ~ CGPA, data=Admission_Predict)
boxplot(standard.Admit ~ GRE.Score, data=Admission_Predict)
boxplot(standard.Admit ~ TOEFL.Score, data=Admission_Predict)

# Finding and plotting Cook's Distance to explicitly id outliers
cooks.lm.all <- cooks.distance(lm.all)
plot(cooks.lm.all, pch="*", cex=1, main="Influential Observations by Cooks distance")
abline(h = 4*mean(cooks.lm.all, na.rm=T), col="red")  #add cutoff line
text(x=1:length(cooks.lm.all)+1, y=cooks.lm.all, labels=ifelse(cooks.lm.all>4*mean(cooks.lm.all, na.rm=T),names(cooks.lm.all),""), col="red")  #add labels

influential <- as.numeric(names(cooks.lm.all)[(cooks.lm.all > 4*mean(cooks.lm.all, na.rm=T))])  #influential row numbers
Admission_Predict[influential, ]  #influential observations.
car::outlier.test(lm.all) #row 10 is the most extreme observation, which we also see in the cook's influential observation test 


# Let's try 1) testing for multicollinarity and 2) potentially transforming the outliers 
X = Admission_Predict[,1:7]
Y = Admission_Predict[8]
X$University.Rating <- as.numeric(X$University.Rating)
X$Research <- as.numeric(levels(X$Research))[X$Research]
X <- as.matrix(X)
omcdiag(x=X, y=Y) #high chi-squared = multicollinarity 
imcdiag(x = X, y = Y) #high F-value is predictor that is causing multicollinarity. Multicollinary can be caused by outliers. As a rule, VIF of above 5 or 10 needs to go (i.e. GPA)
pcor(X, method = "pearson") #under statistic, TOEFL score and GRE score have high correlation (of course). SOP and LOR do, too, ish.   

## Coding outliers as NAs then replacing them with mean values
new.df <- Admission_Predict %>%
  replace_with_na_all(condition = ~.x %in% influential)
new.df$GRE.Score <- impute(new.df$GRE.Score, mean) 
new.df$GRE.Score <- round(new.df$GRE.Score)
new.df$TOEFL.Score <- impute(new.df$TOEFL.Score, mean)
new.df$TOEFL.Score <- round(new.df$TOEFL.Score)

## Transforming Y-variable in data to Box-Cox
## **Don't run yet!**  
acceptBCMod <- BoxCoxTrans(new.df$Chance.of.Admit)
new.df <- cbind(Admission_Predict, accept_new=predict(acceptBCMod, new.df$Chance.of.Admit))
lmMod_bc <- lm(accept_new ~ GRE.Score + TOEFL.Score + LOR + CGPA, data=new.df)
bptest(lmMod_bc)
plot(lmMod_bc)

# Implementing Ridge Regression, which works with multilinear regression and multicollinarity 
## Test and train data
set.seed(123) # set seed to replicate results
trainingIndex <- sample(1:nrow(Admission_Predict), 0.6*nrow(Admission_Predict)) # indices for 60% training data
trainingData <- Admission_Predict[trainingIndex, ] #training data
testData <- Admission_Predict[-trainingIndex, ] #test data

lmMod <- lm(Chance.of.Admit ~ GRE.Score + TOEFL.Score + LOR + CGPA, trainingData)  
summary (lmMod) 

predicted <- predict (lmMod, testData) #predict on test data
compare <- cbind (actual=testData$Chance.of.Admit, predicted)  #combine actual and predicted
mean(apply(compare, 1, min)/apply(compare, 1, max)) #calculate accuracy: 94% accuracy

# Apply Ridge Regression 
linRidgeMod <- linearRidge(Chance.of.Admit ~ GRE.Score + TOEFL.Score + LOR + CGPA, data = trainingData) 
predicted <- predict(linRidgeMod, testData) 
compare <- cbind(actual=testData$Chance.of.Admit, predicted)  
mean(apply(compare, 1, min)/apply(compare, 1, max)) #calculate accuracy: Ridge only improves model by miniscule amount 
