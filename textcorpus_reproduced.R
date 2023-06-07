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
  
  ######################### merge label col to data frame
  
  assignLabels <- function(df,labels) {
    
    df <- cbind(df,labels)
    names(df)[2] = 'label'
    return(df)
  }
  
  ##########################
  
  #
  labelcount <- table(fulldata$label)
  repeated <- names(labelcount[labelcount > 1])
  fulldata <- fulldata[fulldata$label %in% repeated, ]
  fulldata <- droplevels(fulldata)  # Drop unused levels if needed
  rownames(fulldata) <- seq_len(nrow(fulldata))
  
  ######################drop unique labels

  validLabels <- function(df) {
    labelcount <- table(df$label)
    repeated <- names(labelcount[labelcount > 1])
    df <- df[df$label %in% repeated, ]
    df <- droplevels(df)  # Drop unused levels if needed
    rownames(df) <- seq_len(nrow(df))
    
  }
  
  ############################

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

# Run a Bagging model
control <- trainControl(method = "cv", number = 2) # Changed method to 'cv' for cross-validation and number to 2 for 2-fold cross-validation, as it is computationally heavy.
model_bag <- train(as.factor(train_labels) ~ ., data = train_df, trControl = control, method = "treebag")
predictions_bag <- predict(model_bag, newdata = test_df)

# Print classification report
print(confusionMatrix(predictions_bag, test_df$labelnumber))

# Run a LDA model and plot the topics
lda <- LDA(train_df[, !colnames(train_df) %in% "labelnumber"], k = 20, control = list(seed = 3434))
topics <- tidy(lda, matrix = "beta")
top_terms <- topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)
plot_lda <- top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Top 10 terms in each LDA topic",
       x = "Beta", y = "")
print(plot_lda)