# Each comment is a .ipynb cell
# Install and load necessary packages
#install.packages(c("tm", "SnowballC", "slam", "topicmodels", "quanteda", "caret", "e1071", "randomForest", "kernlab", "cluster", "topicmodels", "LDAvis", "ggplot2", 'rlang'))
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

# Convert the sparse matrix to a dense matrix
train_dense <- as.matrix(train)

# Convert the dense matrix to a data frame
train_df <- as.data.frame(train_dense)

# Now add the labels
train_df$labelnumber <- train_labels

# Make column names unique
names(train_df) <- make.names(names(train_df), unique = TRUE)

# Run a Extra Trees model
model <- randomForest(as.factor(train_df$labelnumber) ~ ., data = train_df, ntree = 100, mtry = 2, importance = TRUE)

# Convert the sparse matrix to a dense matrix
test_dense <- as.matrix(test)

# Convert the dense matrix to a data frame
test_df <- as.data.frame(test_dense)

# Now add the labels
test_df$labelnumber <- test_labels

# Make column names unique
names(test_df) <- make.names(names(test_df), unique = TRUE)


# Predict on the test set
predictions <- predict(model, newdata = test_df)

# Print classification report
print(confusionMatrix(predictions, test$labelnumber))

