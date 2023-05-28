# Each comment is a .ipynb cell

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


