# The final project for the Reproducible Research course

## Team members:
- Hassnaa Abdelghany
- Daniel Śliwiński
- Artur Skowroński

## Project overview:
This code is created for those who did not perform text analysis perfor in R and they have text file they wish to explore. you need to run functions with few parameters to get high quality LDA (topic modeling) visiualization word clowds.
if you have category for your text this code can help you classify new text based on text you have previously classified. The sample file in Quando file is based on News Articles with their categories.

## Installation instructions:
You only need to copy functions and run them or load them into your functions .r file. Using "text" or "label" as names in the data frame is totally forbidden in this context. The function will rely on these names to produce output, so you need to make sure that these names are only used for the text column and label column and not other columns.

## File structure:
To add more stopwords, go to the stp file and add the new word in a new line between two quotations.

## Running Code:
you have two options to create a text file:
- you have csv file with text in each row. This means you should use csvText(). where first argument will be your text file.
- you have seperate  .txt files in a folder. run multiTextFile() and add your file path.

Next, run validLabels() to avoid error when creating train and test sample use validLabels(). it takes one argument which is data and make sure there is at least two labels for each label in the data(one to be assigned in test set and another in train set). this avoid model to face labels for the first time when training or predicting data.

### Cleaning Code
then we shall clean text. The cleanText function is designed to perform text cleaning operations on a data frame containing text data. It takes the input data frame and modifies the text column by applying a series of cleaning steps. the function replaces all newline characters (\n) in the text column with a space, removes any numeric digits present in the text column and, remove characters such as commas, backslashes, exclamation marks, question marks, colons, semicolons, quotation marks, parentheses, hyphens, and hashtags.

remove_stopwords() and stem_articles() are there to remove stopwords and perform steming. first argument is the text column from your dataframe. stopword takes additional argument which is the stp.csv like file.

### Tokenization
the heart of text analysis is to generate the keywords AKA tokens. this is where the words get counted and analyized.
myTokenizer() takes a text input x as its argument. Here's how the function works:

The function uses the NGramTokenizer function from the RWeka package to tokenize the input text. Tokenization is the process of breaking down a text into individual units called tokens.

The NGramTokenizer function is configured with the Weka_control function, which provides control parameters for the tokenizer. In this case, the Weka_control is used to specify the minimum and maximum n-gram lengths for tokenization.

The min = 1 argument in Weka_control specifies that the minimum n-gram length for tokenization is 1, meaning that individual words will be considered as tokens.

The max = 1 argument in Weka_control specifies that the maximum n-gram length for tokenization is also 1, indicating that only single words will be considered as tokens. This means that the function will tokenize the input text into individual words.

### Create Word Clowd

function createWordCloud generates a word cloud from a collection of cleaned articles. explanation:

The function begins by defining an inner function called myTokenizer. This function utilizes the NGramTokenizer function from the RWeka package to tokenize the text. The min_ngram and max_ngram arguments allow control over the minimum and maximum n-gram lengths for tokenization.

The cleaned_articles parameter is expected to be a vector of cleaned article texts.

The VCorpus function from the tm package is used to create a corpus from the cleaned_articles vector.

The DocumentTermMatrix function from the tm package is applied to the corpus. The control argument is set to list(tokenize = myTokenizer) to specify the custom tokenization function.

The word frequencies are calculated by summing the columns of the document-term matrix using colSums.

The word frequencies are sorted in decreasing order using the sort function.

A data frame named df is created with two columns: word (containing the words) and freq (containing the corresponding frequencies).

The word cloud is generated using the wordcloud function from the wordcloud package. The words parameter takes the word column from the df data frame, and the freq parameter takes the freq column.

Additional parameters can be adjusted as desired, such as min.freq to set the minimum word frequency for inclusion, max.words to limit the number of words in the word cloud, random.order to control the randomness of word placement, rot.per to set the proportion of words displayed at a rotated angle, and colors to specify the color palette.

### Classification

1. **Model Training:**
```R
model <- ranger(as.factor(train_labels) ~ ., data = train_df, importance = 'impurity', num.trees = 500)
```
In this step, a random forest model is trained using the `ranger` function from the `ranger` package. The model is trained on the `train_df` dataset, where `train_labels` represent the target variable.

2. **Prediction:**
```R
predictions <- predict(model, test_df)
```
The trained model is used to make predictions on the `test_df` dataset.

3. **Model Performance Evaluation:**
```R
table(predictions$predictions, test_labels)
```
This code snippet generates a table that compares the predicted values (`predictions$predictions`) with the actual labels (`test_labels`). It provides an overview of how the model performs across different classes.

4. **Accuracy Calculation:**
```R
accuracy <- sum(predictions$predictions == test_labels) / length(test_labels)
```
The accuracy of the model is computed by comparing the predicted values with the actual labels (`test_labels`).

5. **Confusion Matrix:**
```R
all_levels <- 1:11 # Adjust this to the levels we expect to have
predictions_factor <- factor(predictions$predictions, levels=all_levels)
test_labels_factor <- factor(test_labels, levels=all_levels)
cm <- confusionMatrix(predictions_factor, test_labels_factor)
```
This code calculates the confusion matrix by converting the predicted values and test labels into factors with explicitly defined levels (`all_levels`). The `confusionMatrix` function from the `caret` package is used to compute the confusion matrix, which provides detailed information about the model's performance, including accuracy, precision, recall, and F1 score.

Please note that you may need to adjust the code snippets based on your specific dataset and requirements.

### LDA Topic Modeling Visualization

This readme file provides an overview of the code snippet used to generate a visualization of topics derived from Latent Dirichlet Allocation (LDA) analysis. The code snippet is as follows:

```R
plot_lda_topics(articles = articles_clean, k = 10, num_top_terms = 10, min_ngram = 1, max_ngram = 2)
```

The `plot_lda_topics` function generates a visualization of topics based on the LDA analysis. Here's how the function works:

- `articles_clean`: This parameter represents the cleaned articles dataset. It is assumed that the dataset has been preprocessed and is in a suitable format for topic modeling.

- `k`: This parameter specifies the number of topics to be generated by the LDA model.

- `num_top_terms`: This parameter determines the number of top terms to display for each topic.

- `min_ngram` and `max_ngram`: These parameters define the minimum and maximum n-gram lengths to consider during topic modeling.

The function uses the LDA algorithm to analyze the provided dataset (`articles_clean`) and extract `k` topics. It then generates a visualization that represents the identified topics and their associated top terms. The visualization provides insights into the main themes or subjects within the dataset.

## Troubleshooting:
Make sure your text or R session is in 'UTF-8'. Stay tuned for full documentation and new package.

## Contact information:
Don't hesitate to contact the creator if you face any issues with running the code.
