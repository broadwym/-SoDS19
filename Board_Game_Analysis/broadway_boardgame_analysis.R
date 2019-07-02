library(tidyverse)
library(tidytext)
library(tm)
library(caret)
library(MASS)
library(glmnet)
library(lmtest)
library(wordcloud2)
library(ROCR)
library(randomForest)

# Use the words in the description to predict a few selected categories 
# Read-in data
board_games <- readr::read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-03-12/board_games.csv")

# Transforming description variable 
board_games[c(1,3,11,15,17)] <- NULL
description <- Corpus(VectorSource(board_games$description))
description <- tm_map(description, content_transformer(tolower))
description <- tm_map(description, content_transformer(stripWhitespace))
description <- tm_map(description, removeNumbers)
description <- tm_map(description, removeWords, stopwords("english"))
description <- tm_map(description, removeWords, stopwords("SMART"))
stopwords <- read.csv("C:/Users/broadwmc/Desktop/game.csv", header = FALSE)
stopwords <- as.character(stopwords$V1)
stopwords <- c(stopwords, stopwords())
description <- tm_map(description, removeWords, stopwords)
dtm <- DocumentTermMatrix(description)
smalldtm <- removeSparseTerms(dtm,0.95)
dtmMatrix <- as.matrix(smalldtm)
smalldtmMatrix <- as.matrix(dtm)

# Refine stopwords with custom stoplist 
## I created this stoplist retroactively after finding dwords 


# Description's frequency distribution 
dtm <- TermDocumentMatrix(description)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
dwords <- data.frame(word = names(v),freq=v)
head(dwords, 25)


dwords %>%
  select(word, freq) %>%
  filter(freq > 1100) %>%
  ggplot(aes(x = reorder(x = word, -freq), y=freq)) +
  geom_bar(stat = "identity") + 
  theme(axis.text.x = element_text(angle = 45)) +
  xlab("word") +
  ylab("count")


wordcloud2(dwords[1:300, ], size = .5)

# Category's frequency distribution
category <- gsub(",([A-Za-z])", ", \\1", board_games$category)
category <- strsplit(board_games$category, ",")
unfant <- unlist(category)
un <- table(unfant)
v_fant <- sort(table(unfant),decreasing=TRUE)
head(v_fant, 20)
v_fant <- as.data.frame(v_fant)


v_fant %>%
  select(unfant, Freq) %>%
  filter(Freq > 676) %>%
  ggplot(aes(x = reorder(x = unfant, -Freq), y=Freq)) +
  geom_bar(stat = "identity") + 
  theme(axis.text.x = element_text(angle = 45)) +
  xlab("word") +
  ylab("count")

wordcloud2(v_fant[1:83, ], size = .5)

# Transforming category data 
dataframe <- as.data.frame(dtmMatrix)
fantasy <- sapply(category,function(x){any(x=="Fantasy")})
table(fantasy)

category <- Corpus(VectorSource(board_games$category))
category <- tm_map(category, content_transformer(stripWhitespace)) 
dtm_cat <- DocumentTermMatrix(category)
smalldtm_cat<- removeSparseTerms(dtm_cat,0.95)
dtmMatrixcat <- as.matrix(smalldtm_cat)
smalldtmMatrix_cat <- as.matrix(dtm_cat)
newmatrix <- cbind(dtmMatrix, dtmMatrixcat)


# Dataframe  
df <- data.frame(fantasy, dtmMatrix) 
table(df$fantasy)
df <- na.omit(df)

# Build model
set.seed(123)
model <- glm(fantasy ~ ., data = df, family = "binomial")
summary(model)

model.lm <- glm(fantasy ~ scale + powerful + tactical + battle + fight + role + money + abilities + characters + power + german + space + world + attack + opponents + war + unique, data = df, family = "binomial")
summary(model.lm)
plot(model.lm)

# ANOVA comparing models using Chi-squared 
anova(model, model.lm, test ="Chisq")
lrtest(model, model.lm)
varImp(model)
varImp(model.lm) #powerful is a powerful predictor

# Test and train for LDA 
set.seed(123)
new.df <- df
new.df$fantasy <- as.factor(new.df$fantasy)
new.df <- na.omit(new.df)
sample <- sample.int(n=nrow(new.df), size=floor(.75*nrow(new.df)), replace=F)
train <- new.df[sample, ]
test <- new.df[-sample, ]

# LDA model
lda.1 <- lda(fantasy ~  war + battle + power + money + attack + world + building + german + space + fun + area + scoring + role + powerful + fight + tactical + characters + abilities + power + scale + army + opponents + unique + years, data = train, family = "binomial")
lda.predict <- predict(lda.1, newdata = test) 

# Construction AUC ROC plot
# Get the posteriors as a dataframe.
predict.posteriors <- as.data.frame(lda.predict$posterior)

# Evaluate the model
pred <- prediction(predict.posteriors[,2], test$fantasy)
roc.perf <- performance(pred, "tpr", "fpr")
table(is.na(test))
auc.train <- performance(pred, measure = "auc")
auc.train <- auc.train@y.values

# Plot
plot(roc.perf)
abline(a=0, b= 1)
text(x = .25, y = .65 ,paste("AUC = ", round(auc.train[[1]],3), sep = "")) #.5 = bad (irrelevent) classifiers and 1 = good (relevent) classifiers 
## running PCA before LDA will improve the model: to-do (but from my understanding PCA works with continuous data, not discrete)


# Sentiment analysis 
nrc_pos <- get_sentiments("nrc") %>% 
  filter(sentiment == "positive")

dwords %>%
  inner_join(nrc_pos) %>%
  filter(freq > 500) %>%
  ggplot(aes(x = reorder(x = word, -freq), y=freq)) +
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 45))

bing_word_counts <- dwords %>%
  inner_join(get_sentiments("bing"))

bing_word_counts %>%
  group_by(sentiment) %>%
  filter(freq > 400) %>%
  ggplot(aes(x = reorder(x=word, +freq), y=freq, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(y = "Contribution to sentiment",
       x = NULL) +
  coord_flip()

# Random Forest
set.seed(123)
rf <- randomForest(fantasy ~ war + battle + power + attack + world + building + german + area + scoring + role + powerful + fight + tactical + characters + abilities + power + scale + army + opponents + unique + years, ntree = 370, data=train)
predictRF <- predict(rf, newdata=test)
table(test$fantasy, predictRF) # Correctly classified 2281 FALSEs. Incorrectly classified 287 FALSEs. Correctly and incorrectly classified 21 TRUEs. 
varImpPlot(rf)

error_df <- data.frame(error_rate = rf$err.rate[,'OOB'],
                      num_trees = 1:rf$ntree)
ggplot(error_df, aes(x=num_trees, y=error_rate)) +
  geom_line() # error rate rapidly decreases from .130 and stabilizes around .115 (100-200 trees) # retroactively change tree n in model
