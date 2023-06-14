# Each comment is a .ipynb cell
# Install and load necessary packages


#install.packages(c("tm", "SnowballC", "slam", "topicmodels", "quanteda", "caret", "e1071", "randomForest", "kernlab", "cluster", "topicmodels", "LDAvis", "ggplot2", 'rlang', 'ranger', 'text))

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
library(dplyr)
library(text)



################## function to read txt and return data frame
multiTextFile <- function(directoryPath) {
  # Get the list of file names in the directory
  fileNames <- list.files(path = directoryPath, pattern = "\\.txt$", full.names = TRUE)
  
  # Create an empty list to store the data frames
  dataFrames <- list()
  
  # Read each file and store it as a separate data frame
  for (filePath in fileNames) {
    data <- read.table(filePath, header = TRUE)  # Modify read function based on your file format
    
    # Add the data frame to the list
    dataFrames <- c(dataFrames, list(data))
  }
  
  # Merge the data frames into a single data frame
  mergedData <- do.call(rbind, dataFrames)
  names(mergedData) <- 'text'
  mergedData <- mergedData[!duplicated(mergedData$text), ]
  
  
  # Return the merged data frame
  return(mergedData)
}

############################# function to read csv with text

csvText <- function(file, textCol, labelCol) {
  
  data <- read.csv(file)
  data = data[,c(textCol,labelCol)]
  names(data) = c('text','label')
  names(data[,labelCol]) = 'label'
  data <- data[!duplicated(data$text), ]
  data$text = as.character(data$text)
  data$label = as.factor(data$label)
  
  return(data)
  
}
myData = csvText('fulldata-updated.csv', 'title', 'label')


######################### merge label col to data frame

assignLabels <- function(df,labels) {
  
  df <- cbind(df,labels)
  names(df)[2] = 'label'
  return(df)
}

######################drop unique labels

validLabels <- function(df) {
  labelcount <- table(df$label)
  repeated <- names(labelcount[labelcount > 1])
  df <- df[df$label %in% repeated, ]
  df <- droplevels(df)  # Drop unused levels if needed
  rownames(df) <- seq_len(nrow(df))
  return(df)
  
}
myData <- validLabels(myData)

############################ cleaning Text

cleanText <- function(data) {
  
  data$text <- str_replace_all(data$text, "\n", " ")
  data$text <- str_replace_all(data$text, "[0-9]+", "")
  data$text <- str_replace_all(data$text, "[,\\!?/:;''()``’“-”—#]", "")
  data$text <- str_replace_all(data$text, "[.]+", "")
  data$text <- tolower(data$text)
  data$text <- str_replace_all(data$text, "\\b\\w\\b", "")
  data$text <- as.character(data$text)
  
  return(data$text)
}

articles <- cleanText(myData)
################################# create tokens

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
myData$labelnumber <- as.numeric(as.factor(myData$label))

# Merge the labels with the dtm
dtm_tfidf <- cbind(dtm_tfidf, myData$labelnumber)

# Split the dataset into training and testing sets
set.seed(4545) #seed for the reproducibility
trainIndex <- createDataPartition(myData$labelnumber, p = .6, list = FALSE, times = 1)

# Split the labels as well
train_labels <- myData$labelnumber[trainIndex]
test_labels  <- myData$labelnumber[-trainIndex]

# Split the dtm_tfidf
train <- dtm_tfidf[trainIndex,]
test <- dtm_tfidf[-trainIndex,]

# Prepare matrices suitable for Random Forest
train_df <- as.data.frame(as.matrix(train))
test_df <- as.data.frame(as.matrix(test))
colnames(train_df)[ncol(train_df)] <- "labelnumer"
colnames(test_df)[ncol(test_df)] <- "labelnumer"

# Rename the columns to use only alphanumeric characters and underscores
names(train_df) <- make.names(names(train_df), unique = TRUE)
names(test_df) <- make.names(names(test_df), unique = TRUE)

### ATTEMPT TO FIX PROBLEMS

# Train the model
model <- ranger(as.factor(train_labels) ~ ., data = train_df, 
                importance = 'impurity', num.trees = 500)

# Predict on the test set
predictions <- predict(model, test_df)

# Evaluate model performance
table(predictions$predictions, test_labels)

# Check accuracy
accuracy <- sum(predictions$predictions == test_labels) / length(test_labels)
# I need to end confusionMatrix !!!!

# FIX ATTEMPT
# Define the levels that should exist
all_levels <- 1:11 # Adjust this to the levels we expect to have

# Convert predictions and test_labels to factor and explicitly set the levels
predictions_factor <- factor(predictions$predictions, levels=all_levels)
test_labels_factor <- factor(test_labels, levels=all_levels)

# Compute the confusion matrix
cm <- confusionMatrix(predictions_factor, test_labels_factor)

# Print the confusion matrix
print(cm)


# Run a Bagging model
control <- trainControl(method = "cv", number = 2) # Changed method to 'cv' for cross-validation and number to 2 for 2-fold cross-validation, as it is computationally heavy.
model_bag <- caret::train(as.factor(labelnumer) ~ ., data=train_df, trControl=control, method="treebag")
predictions_bag <- predict(model_bag, newdata = test_df, type="raw")
length(myData$labelnumber)

# Compute the confusion matrix for bagging model
cm_bag <- confusionMatrix(predictions_bag, test_labels_factor)

# Print the confusion matrix
print(cm_bag)

# Bagging model accuracy is much worse 

# Print classification report

# Number of levels is not the same, that's why it is not working
# print(confusionMatrix(factor(test_labels, levels=1:11), as.factor(round(predictions_bag)), levels=1:11))
class(train_df)

# Run a LDA model and plot the topics
lda <- LDA(dtm, k = 20, control = list(seed = 3434), )
topics <- tidytext::tidy(lda, matrix = "beta")
top_terms <- topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)
plot_lda <- top_terms %>%
  mutate(term = tidytext::reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  # coord_flip() +
  theme_minimal() +
  labs(title = "Top 10 terms in each LDA topic",
       x = "Beta", y = "")

print(plot_lda)

