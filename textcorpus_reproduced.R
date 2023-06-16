# Each comment is a .ipynb cell
# Install and load necessary packages

# Load packages and install if necessary
installPackage <- function(packageName) {
  if (!(packageName %in% installed.packages())) {
    install.packages(packageName)
  }
}

# Install and load necessary packages
requiredPackages <- c(
  "tm", "SnowballC", "slam", "topicmodels", "quanteda", 
  "caret", "e1071", "randomForest", "kernlab", "cluster", 
  "topicmodels", "LDAvis", "ggplot2", "rlang", "ranger", "Matrix",
  "RWeka", "caret", "stringr", "tokenizers", "wordcloud", "RColorBrewer",
  "tidytext", "dplyr"
)

# Install required packages if not already installed
invisible(sapply(requiredPackages, installPackage))

# Load the installed packages
lapply(requiredPackages, require, character.only = TRUE)

# Seed for the reproducibility of results
set.seed(4545) 

### Functions

# Function for plotting top n terms in k lda topics
plot_lda_topics <- function(articles, k, num_top_terms = 10, min_ngram=1, max_ngram=2) {
  
  # Create a Corpus
  corpus <- VCorpus(VectorSource(stemmed_articles))
  
  # Define tokenizing function inside a function again for ngram control
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
    labs(title = paste("Top",k, "Terms in Each LDA Topic"), x = "Beta", y = "")
  
  print(plot_lda)
}

# Function to read txt and return data frame. We do not use it in our codes,
# this is just for reproducibility of python codes only.
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

# Function to read csv file
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

# Merge label col to data frame. We do not use it either, this is also for
# reproducibility purposes only.
assignLabels <- function(df,labels) {
  
  df <- cbind(df,labels)
  names(df)[2] = 'label'
  return(df)
}

# Drop labels in case of duplicates
validLabels <- function(df) {
  labelcount <- table(df$label)
  repeated <- names(labelcount[labelcount > 1])
  df <- df[df$label %in% repeated, ]
  df <- droplevels(df)  # Drop unused levels if needed
  rownames(df) <- seq_len(nrow(df))
  return(df)
}

# Text cleaning function
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

# Function to remove stopwords. Lowercase is just in case we would want to use it
# On not cleant texts.
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

# Stemming function to reduce words to their base or root form. In this case we
# use 'Porter Stemmer', to reproduce results from python codes.
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

# Custom tokenizing function, as it is much harder to obtain in R than in Python.
# We use RWeka library here as well, as many packages do not work with Document
# Term Matrix. In the Python codes we create Document Term Matrix with unigrams,
# bigrams and trigrams. But here for the sake of computational speed we use just
# unigrams. Results in accuracy differ due to this fact slightly.
myTokenizer <- function(x) {
  NGramTokenizer(x, Weka_control(min = 1, max = 1))
}

# Function to perform train-test split. It returns a list of objects, which are 
# later used in modelling.
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

# Function to create a word cloud from cleaned articles
createWordCloud <- function(cleaned_articles, min_ngram=1, max_ngram=1) {
  # Define tokenizing function inside a function again for ngram control
  myTokenizer <- function(x) {
    NGramTokenizer(x, Weka_control(min = min_ngram, max = max_ngram))
  }
  # Creat document term matrix from cleaned texts but not stemmed ones.
  corpus <- VCorpus(VectorSource(cleaned_articles))
  
  # Create a Document-Term Matrix (DTM)
  dtm <- DocumentTermMatrix(corpus, control = list(tokenize = myTokenizer))
  
  # Get the word frequencies from the document term matrix
  word_frequencies <- colSums(as.matrix(dtm))
  
  # Sort the word frequencies in decreasing order
  word_frequencies <- sort(word_frequencies, decreasing=TRUE)
  
  # Create a data frame with words and their frequencies
  df <- data.frame(word=names(word_frequencies), freq=word_frequencies)
  
  # Generate the word cloud
  wordcloud(words = df$word, freq = df$freq, min.freq = 1,
            max.words=200, random.order=FALSE, rot.per=0.35, 
            colors=brewer.pal(8, "Dark2"))
}

### Codes
# Read the data
myData <- csvText('fulldata-updated.csv', 'title', 'label')

# Validate the labels
myData <- validLabels(myData)

# Clean texts, remove punctuation and unnecessary symbols
articles <- cleanText(myData)

# Create custom stopwords, so we could merge them with the stp.csv file.
# Words 'article' and 'with' are not useful in this case and all of the files
# contain these words, which just makes the computation longer.
stopwordscustom <- read.csv('stp.csv', header = FALSE, col.names = c('word'))
stopwordscustom <- c(stopwordscustom$word, 'article', 'with')

# Remove stopwords. Now we have truly cleant texts
articles_clean <- remove_stopwords(articles, stopwordscustom)

# Stemming
stemmed_articles <- stem_articles(articles_clean)

# Create a Corpus
corpus <- VCorpus(VectorSource(stemmed_articles))

# Create a Document-Term Matrix (DTM)
dtm <- DocumentTermMatrix(corpus, control = list(tokenize = myTokenizer))

# Apply TF-IDF weighing
dtm_tfidf <- weightTfIdf(dtm)

# Label encoding
myData$labelnumber <- as.numeric(as.factor(myData$label))

# Merge the labels with the dtm_tfidf
dtm_tfidf <- cbind(dtm_tfidf, myData$labelnumber)

# Perform train-test split
split_data <- train_test_split(dtm_tfidf, myData, 0.6)
train_df <- split_data$train_df
test_df <- split_data$test_df
train_labels <- split_data$train_labels
test_labels <- split_data$test_labels

### First model
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

### Second model
# Run a Bagging model
control <- trainControl(method = "cv", number = 2) # Changed method to 'cv' for cross-validation and number to 2 for 2-fold cross-validation, as it is computationally heavy.
model_bag <- caret::train(as.factor(labelnumer) ~ ., data=train_df, trControl=control, method="treebag")
predictions_bag <- predict(model_bag, newdata = test_df, type="raw")

# Compute the confusion matrix for bagging model
cm_bag <- confusionMatrix(predictions_bag, test_labels_factor)

# Print the confusion matrix
print(cm_bag)

# Bagging model accuracy is worse in this case


### LDA
# Plot top terms in topics
plot_lda_topics(articles = articles_clean, k = 10, num_top_terms = 10,
                min_ngram = 1, max_ngram = 2)  
# Word cloud
createWordCloud(articles_clean, 1, 2)