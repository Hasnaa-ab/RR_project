# Each comment is a .ipynb cell
# Install and load necessary packages


#install.packages(c("tm", "SnowballC", "slam", "topicmodels", "quanteda", "caret", "e1071", "randomForest", "kernlab", "cluster", "topicmodels", "LDAvis", "ggplot2", 'rlang', 'ranger))

library(tm)
library(slam)
library(quanteda)
library(caret)
library(e1071)
library(randomForest)
library(kernlab)
library(cluster)
library(topicmodels)
library(LDAvis)
library(ggplot2)
library(stringr)
library(tokenizers)
library(SnowballC)
library(ranger)
library(Matrix)

# Load data
fulldata <- read.csv("fulldata-updated.csv")
fulldata <- fulldata[!duplicated(fulldata$title), ]
fulldata <- fulldata[order(rownames(fulldata)), ]
fulldata <- as.data.frame(lapply(fulldata, type.convert))
rownames(fulldata) <- NULL

#
labelcount <- table(fulldata$label)
repeated <- names(labelcount[labelcount > 1])
fulldata <- fulldata[fulldata$label %in% repeated, ]
fulldata <- droplevels(fulldata)  # Drop unused levels if needed
rownames(fulldata) <- seq_len(nrow(fulldata))

#
fulldata$date <- as.POSIXct(fulldata$date, format = "%Y-%m-%d %H:%M:%S")
fulldata$label <- as.factor(fulldata$label)
month_stats <- summary(as.numeric(format(fulldata$date, "%m")))
fulldata$article <- as.character(fulldata$article)

# Clean text
articles <- str_replace_all(fulldata$article, "\n", " ")
articles <- str_replace_all(articles, "[0-9]+", "")
articles <- str_replace_all(articles, "[,\\!?/:;''()``’“-”—#]", "")
articles <- str_replace_all(articles, "[.]+", "")
articles <- tolower(articles)
articles <- str_replace_all(articles, "\\b\\w\\b", "")
articles <- as.character(articles)

# Tokenizer
articles <- sapply(articles, function(x) tokenizers::tokenize_words(x))
articles <- as.list(articles)

# PorterStemmer

stemmed_articles <- list()
for (i in 1:length(articles)) {
  words <- c()
  
  for (word in articles[[i]]) {
    stemmed_word <- SnowballC::wordStem(word, "english")
    words <- c(words, stemmed_word)
  }
  
  stemmed_articles[[i]] <- words
}

stemmed_articles <- as.list(stemmed_articles)

# 
stopwordscustom <- read.csv('stp.csv', header = FALSE, col.names = c('word'))
stopwordscustom <- as.character(stopwordscustom$word)

# Remove custom stopwords
articles <- lapply(articles, function(x) x[!x %in% stopwordscustom])

# Generate a document-term matrix
dtm <- DocumentTermMatrix(Corpus(VectorSource(articles)))

# Transform to a term frequency-inverse document frequency (TF-IDF) matrix
dtm_tfidf <- weightTfIdf(dtm)

# Label encoding
fulldata$labelnumber <- as.numeric(as.factor(fulldata$label))

# Merge the labels with the dtm
dtm_tfidf <- cbind(dtm_tfidf, fulldata$labelnumber)

# Split the dataset into training and testing sets
set.seed(4545) #seed for the reproducibility
trainIndex <- createDataPartition(fulldata$labelnumber, p = .6, list = FALSE, times = 1)

# Split the labels as well
train_labels <- fulldata$labelnumber[trainIndex]
test_labels  <- fulldata$labelnumber[-trainIndex]

# Split the dtm_tfidf
train <- dtm_tfidf[trainIndex,]
test  <- dtm_tfidf[-trainIndex,]

# Generate a document-term matrix
dtm <- DocumentTermMatrix(Corpus(VectorSource(articles)))

# Transform to a term frequency-inverse document frequency (TF-IDF) matrix
dtm_tfidf <- weightTfIdf(dtm)

# Label encoding
fulldata$labelnumber <- as.numeric(as.factor(fulldata$label))

# Merge the labels with the dtm
dtm_tfidf <- cbind(dtm_tfidf, fulldata$labelnumber)

# Split the dataset into training and testing sets
set.seed(4545) #seed for the reproducibility
trainIndex <- createDataPartition(fulldata$labelnumber, p = .6, list = FALSE, times = 1)

# Split the labels as well
train_labels <- fulldata$labelnumber[trainIndex]
test_labels  <- fulldata$labelnumber[-trainIndex]

# Split the dtm_tfidf
train <- dtm_tfidf[trainIndex,]
test  <- dtm_tfidf[-trainIndex,]

# Prepare matrices suitable for Random Forest
train_df <- as.data.frame(as.matrix(train))
test_df <- as.data.frame(as.matrix(test))

# Rename the columns to use only alphanumeric characters and underscores
names(train_df) <- make.names(names(train_df), unique = TRUE)
names(test_df) <- make.names(names(test_df), unique = TRUE)

### ATTEMPT TO FIX PROBLEMS

# Train the model
model <- ranger(train_labels ~ ., data = train_df, 
                importance = 'impurity', num.trees = 500)

# Predict on the test set
predictions <- predict(model, test_df)

# Evaluate model performance
table(predictions$predictions, test_labels)

# Check accuracy
accuracy <- sum(round(predictions$predictions, 0)  == test_labels) / length(test_labels)