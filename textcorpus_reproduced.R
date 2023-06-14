# Each comment is a .ipynb cell
# Install and load necessary packages


#install.packages(c("tm", "SnowballC", "slam", "topicmodels", "quanteda", "caret", "e1071", "randomForest", "kernlab", "cluster", "topicmodels", "LDAvis", "ggplot2", 'rlang', 'ranger', 'text', 'tidytext', 'RWeka'))
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
library(tidytext)
library(RWeka)

set.seed(4545) #seed for the reproducibility

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

################################# remove stopwords

remove_stopwords <- function(texts, stopwords) {
  cleaned_texts <- lapply(texts, function(text) {
    # Tokenize the text
    tokens <- strsplit(tolower(text), "\\s+")
    
    # Remove stopwords
    tokens_clean <- tokens[[1]][!(tokens[[1]] %in% stopwords)]
    
    # Join the cleaned tokens back into a text
    cleaned_text <- paste(tokens_clean, collapse = " ")
    
    # Return the cleaned text
    return(cleaned_text)
  })
  
  return(cleaned_texts)
}

# Create custom stopwords
stopwordscustom <- read.csv('stp.csv', header = FALSE, col.names = c('word'))
stopwordscustom <- c(stopwordscustom$word, 'article', 'with')  # Add 'article' and 'with' to the custom stopwords list

articles_clean <- remove_stopwords(articles, stopwordscustom)

################################# stemming

# Stemming function
stem_articles <- function(articles) {
  stemmed_articles <- character(length(articles))
  
  for (i in 1:length(articles)) {
    words <- c()
    
    for (word in articles[[i]]) {
      stemmed_word <- SnowballC::wordStem(word, "porter")
      words <- c(words, stemmed_word)
    }
    
    stemmed_articles[i] <- paste(words, collapse = " ")
  }
  
  return(stemmed_articles)
}

stemmed_articles <- stem_articles(articles_clean)

# Create a Corpus
corpus <- VCorpus(VectorSource(stemmed_articles))

# Define a custom tokenizer function
myTokenizer <- function(x) {
  NGramTokenizer(x, Weka_control(min = 1, max = 1))
}

# Create a Document-Term Matrix (DTM) with bigrams and trigrams
dtm <- DocumentTermMatrix(corpus, control = list(tokenize = myTokenizer))

# Apply TF-IDF weighing
dtm_tfidf <- weightTfIdf(dtm)

# Label encoding
myData$labelnumber <- as.numeric(as.factor(myData$label))

# Merge the labels with the dtm_tfidf
dtm_tfidf <- cbind(dtm_tfidf, myData$labelnumber)

train_test_split <- function(dtm_tfidf, myData, partition_ratio) {
  
  # Split the dataset into training and testing sets
  trainIndex <- createDataPartition(myData$labelnumber, p = partition_ratio, list = FALSE, times = 1)
  
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
  
  # Return the training and testing data frames and labels
  return(list(train_df = train_df, test_df = test_df, train_labels = train_labels, test_labels = test_labels))
}

split_data <- train_test_split(dtm_tfidf, myData, 0.6)
train_df <- split_data$train_df
test_df <- split_data$test_df
train_labels <- split_data$train_labels
test_labels <- split_data$test_labels

# Train the model
model <- ranger(as.factor(train_labels) ~ ., data = train_df, 
                importance = 'impurity', num.trees = 500)

# Predict on the test set
predictions <- predict(model, test_df)

# Evaluate model performance
table(predictions$predictions, test_labels)

# Check accuracy
accuracy <- sum(predictions$predictions == test_labels) / length(test_labels)

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

# Compute the confusion matrix for bagging model
cm_bag <- confusionMatrix(predictions_bag, test_labels_factor)

# Print the confusion matrix
print(cm_bag)

# Bagging model accuracy is worse in this case

##### LDA

plot_lda_topics <- function(articles, k, num_top_terms = 10, title = "Top Terms in Each LDA Topic", min_ngram=1, max_ngram=2) {
  
  # Create a Corpus
  corpus <- VCorpus(VectorSource(stemmed_articles))
  
  # Define a custom tokenizer function for bigrams and trigrams
  myTokenizer <- function(x) {
    NGramTokenizer(x, Weka_control(min = min_ngram, max = max_ngram))
  }
  
  # Create a Document-Term Matrix (DTM) with bigrams and trigrams
  dtm <- DocumentTermMatrix(corpus, control = list(tokenize = myTokenizer))
  
  # Run LDA model
  lda <- LDA(dtm, k = k, control = list(seed = 3434))
  
  # Get the top terms for each topic
  topics <- tidy(lda, matrix = "beta")
  top_terms <- topics %>%
    group_by(topic) %>%
    top_n(num_top_terms, beta) %>%
    ungroup() %>%
    arrange(topic, -beta)
  
  # Plot the LDA topics
  plot_lda <- top_terms %>%
    mutate(term = reorder_within(term, beta, topic)) %>%
    ggplot(aes(beta, term, fill = factor(topic))) +
    geom_col(show.legend = FALSE) +
    facet_wrap(~ topic, scales = "free") +
    theme_minimal() +
    labs(title = title, x = "Beta", y = "")
  
  print(plot_lda)
}

plot_lda_topics(articles = articles_clean, k = 10, num_top_terms = 10,
                title = "Top 10 Terms in Each LDA Topic", min_ngram = 1, max_ngram = 2)  
